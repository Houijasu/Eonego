/// Lc0 (LeelaChessZero) network protobuf loader: hand-rolled wire parser + LINEAR16 dequant +
/// BatchNorm fold into conv weights/bias. Produces an `Lc0Net` of folded float32 weights for the CNN
/// forward pass (Lc0Net.fs). AOT-safe (no protobuf library / reflection). Fail-soft like SfNnue.loadBytes.
///
/// Spec (decoded from nets/20x256SE-jj-9-75000000.pb; see memory lc0-net-spec):
///   Net{magic=1:fixed32 (0x1c0), min_version=3, format=4, weights=10:Weights}.
///   Weights LEGACY-FLAT: input=1:ConvBlock, residual=2:repeated Residual, policyConv=3:ConvBlock,
///     ip_pol_w=4, ip_pol_b=5, valueConv=6:ConvBlock, ip1_val_w=7, ip1_val_b=8, ip2_val_w=9, ip2_val_b=10.
///   ConvBlock{weights=1, biases=2(absent→0), bn_means=3, bn_stddivs=4(=VARIANCE), bn_gammas=5, bn_betas=6}.
///   Residual{conv1=1, conv2=2, se=3}.  SEunit{w1=1,b1=2,w2=3,b2=4}.  Layer{min=1:f32,max=2:f32,params=3:u16[],dims=5}.
///   BN fold (lc0 network_legacy.cc): scale=gamma/sqrt(var+1e-5); W'=W*scale; b'=-scale*(mean-bias)+beta.
module Eonego.Lc0Proto

open System
open System.Buffers.Binary

[<Literal>]
let Lc0Magic = 0x1c0u

[<Literal>]
let BnEps = 1e-5f

[<Literal>]
let InputPlanes = 112

[<Literal>]
let Channels = 256

[<Literal>]
let PolicyOutputs = 1858

// ---------------------------------------------------------------------------
// Parsed (folded) network. Conv weights are [outC][inC*k*k] row-major, BN already folded in.
// ---------------------------------------------------------------------------
type Lc0ConvBlock =
    { W: float32[] // folded weights, length outC*inC*k*k
      B: float32[] // folded bias, length outC
      InC: int
      OutC: int
      K: int } // kernel size (1 or 3)

type Lc0SE =
    { W1: float32[] // C -> SeCh
      B1: float32[]
      W2: float32[] // SeCh -> 2*C  ([scale | bias])
      B2: float32[]
      C: int
      SeCh: int }

type Lc0Residual =
    { Conv1: Lc0ConvBlock
      Conv2: Lc0ConvBlock
      Se: Lc0SE }

type Lc0Net =
    { Input: Lc0ConvBlock // 112 -> 256, 3x3
      Tower: Lc0Residual[] // SE residual blocks (21 for this net)
      PolicyConv: Lc0ConvBlock // 256 -> 256, 1x1
      PolicyW: float32[] // [1858 * (256*64)]
      PolicyB: float32[] // [1858]
      ValueConv: Lc0ConvBlock // 256 -> 64, 1x1
      Value1W: float32[] // [128 * (64*64)]
      Value1B: float32[] // [128]
      Value2W: float32[] // [1 * 128]
      Value2B: float32[] // [1]
      Channels: int }

type Lc0LoadResult =
    | Loaded of Lc0Net
    | Failed of string

// ---------------------------------------------------------------------------
// Protobuf wire reader (varint / tag / fixed32 / length-delimited).
// ---------------------------------------------------------------------------
let private readVarint (b: byte[]) (p: int) : struct (uint64 * int) =
    let mutable shift = 0
    let mutable result = 0UL
    let mutable i = p
    let mutable go = true

    while go do
        let by = b.[i]
        i <- i + 1
        result <- result ||| ((uint64 (by &&& 0x7fuy)) <<< shift)
        if by < 0x80uy then go <- false
        shift <- shift + 7

    struct (result, i)

/// Scan a message [off, off+len) invoking `h field wire payloadOff payloadLen varintVal` per field.
/// wire 2 (len-delim): payloadOff/Len = the bytes. wire 5 (fixed32): off=pos,len=4. wire 0 (varint): val set.
let private scan (b: byte[]) (off: int) (len: int) (h: int -> int -> int -> int -> uint64 -> unit) : unit =
    let endp = off + len
    let mutable p = off

    while p < endp do
        let struct (tag, p1) = readVarint b p
        p <- p1
        let field = int (tag >>> 3)
        let wire = int (tag &&& 7UL)

        match wire with
        | 0 ->
            let struct (v, p2) = readVarint b p
            h field 0 0 0 v
            p <- p2
        | 5 ->
            h field 5 p 4 0UL
            p <- p + 4
        | 1 ->
            h field 1 p 8 0UL
            p <- p + 8
        | 2 ->
            let struct (ln, p2) = readVarint b p
            let l = int ln
            h field 2 p2 l 0UL
            p <- p2 + l
        | _ -> failwith ("bad wire type " + string wire + " at offset " + string p)

// ---------------------------------------------------------------------------
// Layer -> dequantized float32[] (LINEAR16: params uint16 LE, val = min + (max-min)*raw/65535).
// ---------------------------------------------------------------------------
let private parseLayer (b: byte[]) (off: int) (len: int) : float32[] =
    let mutable mn = 0.0f
    let mutable mx = 0.0f
    let mutable pOff = 0
    let mutable pLen = 0

    scan b off len (fun field wire so sl _ ->
        if wire = 5 then
            let f = BinaryPrimitives.ReadSingleLittleEndian(ReadOnlySpan<byte>(b, so, 4))
            if field = 1 then mn <- f
            elif field = 2 then mx <- f
        elif wire = 2 && field = 3 then
            pOff <- so
            pLen <- sl)

    let n = pLen / 2
    let arr = Array.zeroCreate<float32> n
    let scaleq = (mx - mn) / 65535.0f
    let mutable i = 0

    while i < n do
        let raw = BinaryPrimitives.ReadUInt16LittleEndian(ReadOnlySpan<byte>(b, pOff + 2 * i, 2))
        arr.[i] <- mn + scaleq * float32 raw
        i <- i + 1

    arr

// ---------------------------------------------------------------------------
// ConvBlock -> folded conv (BN folded into W',b'). inC is known per call.
// ---------------------------------------------------------------------------
let private parseConvBlock (b: byte[]) (off: int) (len: int) (inC: int) : Lc0ConvBlock =
    let mutable wO = -1
    let mutable wL = 0
    let mutable biO = -1
    let mutable biL = 0
    let mutable meO = -1
    let mutable meL = 0
    let mutable stO = -1
    let mutable stL = 0
    let mutable gaO = -1
    let mutable gaL = 0
    let mutable beO = -1
    let mutable beL = 0

    scan b off len (fun field wire so sl _ ->
        if wire = 2 then
            match field with
            | 1 -> wO <- so; wL <- sl
            | 2 -> biO <- so; biL <- sl
            | 3 -> meO <- so; meL <- sl
            | 4 -> stO <- so; stL <- sl
            | 5 -> gaO <- so; gaL <- sl
            | 6 -> beO <- so; beL <- sl
            | _ -> ())

    let w = parseLayer b wO wL

    if meO < 0 then
        failwith "ConvBlock without batch-norm means (unsupported)"

    let means = parseLayer b meO meL
    let outC = means.Length
    let var = parseLayer b stO stL
    let gam = if gaO >= 0 then parseLayer b gaO gaL else Array.create outC 1.0f
    let beta = if beO >= 0 then parseLayer b beO beL else Array.zeroCreate outC
    let biases = if biO >= 0 then parseLayer b biO biL else Array.zeroCreate outC
    let perOut = w.Length / outC
    let ksq = perOut / inC
    let k = if ksq >= 9 then 3 else 1

    // Fold BN: scale = gamma/sqrt(var+eps); W' = W*scale; b' = -scale*(mean-bias) + beta.
    let bfold = Array.zeroCreate<float32> outC
    let mutable o = 0

    while o < outC do
        let scale = gam.[o] / MathF.Sqrt(var.[o] + BnEps)
        let baseo = o * perOut
        let mutable c = 0

        while c < perOut do
            w.[baseo + c] <- w.[baseo + c] * scale
            c <- c + 1

        bfold.[o] <- -scale * (means.[o] - biases.[o]) + beta.[o]
        o <- o + 1

    { W = w; B = bfold; InC = inC; OutC = outC; K = k }

// ---------------------------------------------------------------------------
// SEunit / Residual / Weights.
// ---------------------------------------------------------------------------
let private parseSE (b: byte[]) (off: int) (len: int) (c: int) : Lc0SE =
    let mutable w1O = -1
    let mutable w1L = 0
    let mutable b1O = -1
    let mutable b1L = 0
    let mutable w2O = -1
    let mutable w2L = 0
    let mutable b2O = -1
    let mutable b2L = 0

    scan b off len (fun field wire so sl _ ->
        if wire = 2 then
            match field with
            | 1 -> w1O <- so; w1L <- sl
            | 2 -> b1O <- so; b1L <- sl
            | 3 -> w2O <- so; w2L <- sl
            | 4 -> b2O <- so; b2L <- sl
            | _ -> ())

    let b1 = parseLayer b b1O b1L

    { W1 = parseLayer b w1O w1L
      B1 = b1
      W2 = parseLayer b w2O w2L
      B2 = parseLayer b b2O b2L
      C = c
      SeCh = b1.Length }

let private parseResidual (b: byte[]) (off: int) (len: int) (c: int) : Lc0Residual =
    let mutable c1O = -1
    let mutable c1L = 0
    let mutable c2O = -1
    let mutable c2L = 0
    let mutable seO = -1
    let mutable seL = 0

    scan b off len (fun field wire so sl _ ->
        if wire = 2 then
            match field with
            | 1 -> c1O <- so; c1L <- sl
            | 2 -> c2O <- so; c2L <- sl
            | 3 -> seO <- so; seL <- sl
            | _ -> ())

    { Conv1 = parseConvBlock b c1O c1L c
      Conv2 = parseConvBlock b c2O c2L c
      Se = parseSE b seO seL c }

let private parseWeights (b: byte[]) (off: int) (len: int) : Lc0Net =
    let residuals = System.Collections.Generic.List<struct (int * int)>()
    let mutable inO = -1
    let mutable inL = 0
    let mutable pcO = -1
    let mutable pcL = 0
    let mutable pwO = -1
    let mutable pwL = 0
    let mutable pbO = -1
    let mutable pbL = 0
    let mutable vcO = -1
    let mutable vcL = 0
    let mutable v1wO = -1
    let mutable v1wL = 0
    let mutable v1bO = -1
    let mutable v1bL = 0
    let mutable v2wO = -1
    let mutable v2wL = 0
    let mutable v2bO = -1
    let mutable v2bL = 0

    scan b off len (fun field wire so sl _ ->
        if wire = 2 then
            match field with
            | 1 -> inO <- so; inL <- sl
            | 2 -> residuals.Add(struct (so, sl))
            | 3 -> pcO <- so; pcL <- sl
            | 4 -> pwO <- so; pwL <- sl
            | 5 -> pbO <- so; pbL <- sl
            | 6 -> vcO <- so; vcL <- sl
            | 7 -> v1wO <- so; v1wL <- sl
            | 8 -> v1bO <- so; v1bL <- sl
            | 9 -> v2wO <- so; v2wL <- sl
            | 10 -> v2bO <- so; v2bL <- sl
            | _ -> ())

    let input = parseConvBlock b inO inL InputPlanes
    let c = input.OutC

    let tower =
        [| for struct (o, l) in residuals -> parseResidual b o l c |]

    { Input = input
      Tower = tower
      PolicyConv = parseConvBlock b pcO pcL c
      PolicyW = parseLayer b pwO pwL
      PolicyB = parseLayer b pbO pbL
      ValueConv = parseConvBlock b vcO vcL c
      Value1W = parseLayer b v1wO v1wL
      Value1B = parseLayer b v1bO v1bL
      Value2W = parseLayer b v2wO v2wL
      Value2B = parseLayer b v2bO v2bL
      Channels = c }

// ---------------------------------------------------------------------------
// Public load (fail-soft).
// ---------------------------------------------------------------------------
let loadBytes (b: byte[]) : Lc0LoadResult =
    try
        let mutable magic = 0u
        let mutable wO = -1
        let mutable wL = 0

        scan b 0 b.Length (fun field wire so sl _ ->
            if field = 1 && wire = 5 then
                magic <- BinaryPrimitives.ReadUInt32LittleEndian(ReadOnlySpan<byte>(b, so, 4))
            elif field = 10 && wire = 2 then
                wO <- so
                wL <- sl)

        // NB: string-concat (not sprintf) throughout — F# Printf crashes under NativeAOT (MakeGenericMethod).
        if magic <> Lc0Magic then
            Failed("bad Lc0 magic " + magic.ToString("x") + " (expected 0x1c0)")
        elif wO < 0 then
            Failed "no weights field (10) in Net message"
        else
            let net = parseWeights b wO wL

            if net.Input.OutC <> Channels then
                Failed("input conv outC " + string net.Input.OutC + " <> " + string Channels)
            elif net.Input.W.Length <> Channels * InputPlanes * 9 then
                Failed("input conv size " + string net.Input.W.Length + " <> " + string (Channels * InputPlanes * 9))
            elif net.PolicyB.Length <> PolicyOutputs then
                Failed("policy outputs " + string net.PolicyB.Length + " <> " + string PolicyOutputs)
            elif net.Value2B.Length <> 1 then
                Failed("value output " + string net.Value2B.Length + " <> 1 (expected scalar)")
            else
                Loaded net
    with ex ->
        Failed("Lc0 parse exception: " + ex.Message)

let load (path: string) : Lc0LoadResult =
    if not (IO.File.Exists path) then
        Failed("file not found: " + path)
    else
        try
            loadBytes (IO.File.ReadAllBytes path)
        with ex ->
            Failed("could not read file: " + ex.Message)
