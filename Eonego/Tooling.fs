/// Eonego — offline tooling subcommands (gen) for self-play data generation.
/// Console.Out only (AOT-safe); parse numbers with InvariantCulture.
module Eonego.Tooling

#nowarn "9"

open System
open System.Collections.Generic
open System.Globalization
open System.IO
open System.Text
open Microsoft.FSharp.NativeInterop
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.Nnue
open Eonego.Search

let private inv = CultureInfo.InvariantCulture

let private writeLine (s: string) = Console.Out.WriteLine(s)

let private errLine (s: string) = Console.Error.WriteLine(s)

let private usage () =
    writeLine "Eonego tooling subcommands:"
    writeLine "  gen --start <fen> --games N --out <file> --net <path> [--depth D | --nodes K] [--temp T] [--seed S] [--random-plies P] [--max-plies M]"
    writeLine "  lstrace --in <fens> --out <trace.tsv> --net <sf.nnue> [--budget N] [--cap C]   (LearnedSearch supervised-trace generator)"
    writeLine "  lsforward --lsnet <eongls> --in <featrows.tsv> [--out <file>]                  (EONGLS priority-net inference, for parity)"

/// Hand-rolled `--key value` parser; returns a map of flag -> optional value.
let private parseFlags (args: string[]) : Map<string, string option> =
    let m = Dictionary<string, string option>(StringComparer.OrdinalIgnoreCase)
    let mutable i = 0

    while i < args.Length do
        let tok = args.[i]

        if tok.StartsWith("--") then
            let key = tok.[2..]

            if i + 1 < args.Length && not (args.[i + 1].StartsWith "--") then
                m.[key] <- Some args.[i + 1]
                i <- i + 2
            else
                m.[key] <- None
                i <- i + 1
        else
            i <- i + 1

    m |> Seq.map (|KeyValue|) |> Map.ofSeq

let private flag (m: Map<string, string option>) (name: string) : string option =
    Map.tryFind name m |> Option.flatten

let private flagInt (m: Map<string, string option>) (name: string) (defaultVal: int) : int =
    match flag m name with
    | Some s ->
        match Int32.TryParse(s, NumberStyles.Integer, inv) with
        | true, v -> v
        | _ -> defaultVal
    | None -> defaultVal

let private flagInt64 (m: Map<string, string option>) (name: string) (defaultVal: int64) : int64 =
    match flag m name with
    | Some s ->
        match Int64.TryParse(s, NumberStyles.Integer, inv) with
        | true, v -> v
        | _ -> defaultVal
    | None -> defaultVal

let private flagFloat (m: Map<string, string option>) (name: string) (defaultVal: float) : float =
    match flag m name with
    | Some s ->
        match Double.TryParse(s, NumberStyles.Float, inv) with
        | true, v -> v
        | _ -> defaultVal
    | None -> defaultVal

let private matchMove (pos: Position) (uci: string) : Move =
    let t = parseUci uci

    if t = MoveNone then
        MoveNone
    else
        let p = NativePtr.stackalloc<Move> MaxMoves
        let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)
        let n = generateLegal pos buf
        let mutable found = MoveNone

        for i in 0 .. n - 1 do
            let m = buf.[i]

            if
                fromSq m = fromSq t
                && toSq m = toSq t
                && (not (isPromotion t) || promoType m = promoType t)
            then
                found <- m

        found

[<Literal>]
let DefaultKgaFen = "rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP2PP/RNBQKBNR w KQkq - 0 3"

let private kgaBookLines =
    [| [| "g1f3"; "g7g5" |]
       [| "g1f3"; "d7d6" |]
       [| "f1c4" |]
       [| "g1f3"; "g7g5"; "h2h4" |]
       [| "g1f3"; "g7g5"; "f1c4"; "g5g4" |]
       [| "d2d4" |]
       [| "b1c3"; "g8f6" |]
       [| "g1f3"; "g7g5"; "e1g1" |] |]

let private collectLegal (pos: Position) : Move[] =
    let p = NativePtr.stackalloc<Move> MaxMoves
    let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)
    let n = generateLegal pos buf
    let moves = Array.zeroCreate n

    for i in 0 .. n - 1 do
        moves.[i] <- buf.[i]

    moves

let private isNoisyBest (pos: Position) (m: Move) : bool =
    isPromotion m || isEnPassant m || not (pos.IsEmpty(toSq m))

let private whiteRelEval (net: SfNetwork) (pos: Position) : int =
    let e = Nnue.evalCp net pos
    if pos.SideToMove = White then e else -e

let private softmaxPick (rng: Random) (temp: float) (moves: Move[]) (scores: int[]) : Move =
    if moves.Length = 1 then
        moves.[0]
    else
        let t = max 1.0 temp
        let maxS = scores |> Array.max
        let weights = scores |> Array.map (fun s -> exp (float (s - maxS) / t))
        let sum = weights |> Array.sum
        let r = rng.NextDouble() * sum
        let mutable acc = 0.0

        let mutable pick = moves.[moves.Length - 1]
        let mutable i = 0
        let mutable found = false

        while not found && i < moves.Length do
            acc <- acc + weights.[i]

            if r <= acc then
                pick <- moves.[i]
                found <- true

            i <- i + 1

        pick

let private gameResult (net: SfNetwork) (pos: Position) (ply: int) (maxPlies: int) : float =
    let legal = collectLegal pos

    if legal.Length = 0 then
        if pos.InCheck then
            if pos.SideToMove = White then 0.0 else 1.0
        else
            0.5
    elif isImmediateDraw pos then
        0.5
    elif ply >= maxPlies then
        let w = whiteRelEval net pos

        if w > 150 then 1.0
        elif w < -150 then 0.0
        else 0.5
    else
        nan // game continues

let private searchMove
    (pos: Position)
    (rootMoves: Move[])
    (depth: int option)
    (nodes: int64 option)
    (cfg: SearchConfig)
    (net: SfNetwork option)
    : struct (int * Move) =
    let fenNow = pos.ToFen()

    match depth, nodes with
    | Some d, _ ->
        let struct (score, _, mv) = searchToDepthNet fenNow [||] d cfg net
        struct (score, mv)
    | None, Some n ->
        let struct (score, _, mv) = searchToNodesNet fenNow [||] n cfg net
        struct (score, mv)
    | _ ->
        let struct (score, _, mv) = searchToDepthNet fenNow [||] 6 cfg net
        struct (score, mv)

type private PendingRec =
    { Fen: string
      Key: uint64
      CpWhite: int }

let private playOneGame
    (writer: StreamWriter)
    (startFen: string)
    (rng: Random)
    (temp: float)
    (randomPlies: int)
    (maxPlies: int)
    (depthOpt: int option)
    (nodesOpt: int64 option)
    (cfg: SearchConfig)
    (net: SfNetwork)
    : unit =
    let pos = Position.OfFen startFen
    let mutable rootMoves = [||]
    let book = kgaBookLines.[rng.Next kgaBookLines.Length]
    let prefixTarget = max randomPlies book.Length
    let mutable prefixDone = 0

    while prefixDone < prefixTarget do
        let legal = collectLegal pos

        if legal.Length = 0 then
            prefixDone <- prefixTarget
        else
            let mvFromBook =
                if prefixDone < book.Length then matchMove pos book.[prefixDone] else MoveNone

            let mv =
                if mvFromBook <> MoveNone then
                    mvFromBook
                else
                    let struct (score, _) = searchMove pos rootMoves depthOpt nodesOpt cfg (Some net)
                    let scores = legal |> Array.map (fun _ -> score)
                    softmaxPick rng temp legal scores

            if mv = MoveNone then
                prefixDone <- prefixTarget
            else
                pos.Make mv
                rootMoves <- Array.append rootMoves [| mv |]
                prefixDone <- prefixDone + 1

    let mutable ply = prefixDone
    let keys = HashSet<uint64>()
    let pending = ResizeArray<PendingRec>()
    let mutable finished = false

    while not finished do
        let result = gameResult net pos ply maxPlies

        if not (Double.IsNaN result) then
            for entry in pending do
                writer.WriteLine(entry.Fen + ";" + string entry.CpWhite + ";" + result.ToString("0.0", inv))

            finished <- true
        else
            let struct (scoreStm, best) = searchMove pos rootMoves depthOpt nodesOpt cfg (Some net)

            if best = MoveNone then
                let r = gameResult net pos ply maxPlies

                for entry in pending do
                    writer.WriteLine(entry.Fen + ";" + string entry.CpWhite + ";" + r.ToString("0.0", inv))

                finished <- true
            else
                if not pos.InCheck && keys.Add pos.Key && not (isNoisyBest pos best) then
                    let cpWhite =
                        if pos.SideToMove = White then scoreStm else -scoreStm

                    pending.Add
                        { Fen = pos.ToFen()
                          Key = pos.Key
                          CpWhite = cpWhite }

                pos.Make best
                rootMoves <- Array.append rootMoves [| best |]
                ply <- ply + 1

let runGen (args: string[]) : int =
    let runGenGames outPath startFen games depthOpt nodesOpt temp seed randomPlies maxPlies (net: SfNetwork) =
        let cfg = { defaultConfig with Threads = 1 }

        use writer = new StreamWriter(outPath, false, Encoding.UTF8)
        writer.WriteLine("# eonego gen v1")
        writer.WriteLine("# start=" + startFen)
        writer.WriteLine("# games=" + string games)
        writer.WriteLine("# seed=" + string seed)
        writer.WriteLine("# depth=" + (match depthOpt with Some d -> string d | None -> "-"))
        writer.WriteLine("# nodes=" + (match nodesOpt with Some n -> string n | None -> "-"))
        writer.Flush()

        for g in 0 .. games - 1 do
            let rng = Random(seed + g)

            playOneGame writer startFen rng temp randomPlies maxPlies depthOpt nodesOpt cfg net

        writer.Flush()
        0

    let m = parseFlags args

    match flag m "out" with
    | None ->
        errLine "gen requires --out <file>"
        1
    | Some outPath ->
        let startFen = flag m "start" |> Option.defaultValue DefaultKgaFen
        let games = flagInt m "games" 1

        let depthOpt =
            match flagInt m "depth" 0 with
            | 0 -> None
            | d -> Some d

        let nodesOpt =
            match flagInt64 m "nodes" 0L with
            | 0L -> None
            | n -> Some n

        let temp = flagFloat m "temp" 50.0
        let seed = flagInt m "seed" 42
        let randomPlies = flagInt m "random-plies" 8
        let maxPlies = flagInt m "max-plies" 200

        match flag m "net" with
        | Some path ->
            match Nnue.load path with
            | Nnue.Failed r ->
                errLine ("failed to load net: " + r)
                1
            | Nnue.Loaded n -> runGenGames outPath startFen games depthOpt nodesOpt temp seed randomPlies maxPlies n
        | None ->
            errLine "gen requires --net <path> (SF NNUE file)"
            1

/// LearnedSearch supervised-trace generator: run W=1 best-first (fixed priority) on each input FEN and
/// dump per-expansion (features, leafEval, root-impact) rows for the priority-net trainer.
let runLsTrace (args: string[]) : int =
    let m = parseFlags args

    match flag m "in", flag m "out", flag m "net" with
    | Some inPath, Some outPath, Some netPath ->
        match Nnue.load netPath with
        | Nnue.Failed r ->
            errLine ("failed to load SF net: " + r)
            1
        | Nnue.Loaded net ->
            let budget = flagInt64 m "budget" 20000L
            let cap = flagInt m "cap" 1_048_576
            use writer = new StreamWriter(outPath, false, Encoding.UTF8)
            writer.WriteLine("# eonego lstrace v1 NF=" + string LearnedSearch.NF + " budget=" + string budget)
            let mutable count = 0

            for line in File.ReadLines inPath do
                let s = line.Trim()

                if s.Length > 0 && not (s.StartsWith "#") then
                    let fen = (s.Split(';').[0]).Trim()
                    let pos = Position.OfFen fen
                    Nnue.bindNnue net pos
                    let tree = LearnedSearch.LsTree(cap)
                    tree.TraceTo(pos, (fun (p: Position) -> Nnue.evalCp net p), budget, writer, fen)
                    count <- count + 1

            writer.Flush()
            writeLine ("lstrace: traced " + string count + " positions -> " + outPath)
            0
    | _ ->
        errLine "lstrace requires --in <fens> --out <trace.tsv> --net <sf.nnue> [--budget N] [--cap C]"
        1

/// EONGLS priority-net inference over feature rows (NF ints per line) — the engine side of the
/// Python↔engine parity gate (compare to trainer/ls_intref.py on identical rows).
let runLsForward (args: string[]) : int =
    let m = parseFlags args

    match flag m "lsnet", flag m "in" with
    | Some lsPath, Some inPath ->
        match LearnedSearch.loadBytes (File.ReadAllBytes lsPath) with
        | None ->
            errLine "failed to load EONGLS (see info string)"
            1
        | Some net ->
            let sb = StringBuilder()
            let x = Array.zeroCreate<int> LearnedSearch.NF

            for line in File.ReadLines inPath do
                let s = line.Trim()

                if s.Length > 0 && not (s.StartsWith "#") then
                    let toks = s.Split([| ' '; '\t' |], StringSplitOptions.RemoveEmptyEntries)

                    if toks.Length >= LearnedSearch.NF then
                        for i in 0 .. LearnedSearch.NF - 1 do
                            x.[i] <-
                                match Int32.TryParse(toks.[i], NumberStyles.Integer, inv) with
                                | true, v -> v
                                | _ -> 0

                        sb.AppendLine(string (LearnedSearch.inferRow net x)) |> ignore

            match flag m "out" with
            | Some outPath ->
                File.WriteAllText(outPath, sb.ToString())
                writeLine ("lsforward: wrote " + outPath)
            | None -> Console.Out.Write(sb.ToString())

            0
    | _ ->
        errLine "lsforward requires --lsnet <eongls> --in <featrows.tsv> [--out <file>]"
        1
