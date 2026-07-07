module Eonego.Syzygy

#nowarn "9"
#nowarn "3261"

open System
open System.IO
open System.Threading
open Eonego.Bitboard

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
[<Literal>]
let private TbPieces = 7

[<Literal>]
let private TbHashBits = 12

[<Literal>]
let private TbMaxPiece = 650

[<Literal>]
let private TbMaxPawn = 861

[<Literal>]
let private TbMaxMoves = 193

[<Literal>]
let private TbMaxCaptures = 64

[<Literal>]
let TbLoss = 0

[<Literal>]
let TbBlessedLoss = 1

[<Literal>]
let TbDraw = 2

[<Literal>]
let TbCursedWin = 3

[<Literal>]
let TbWin = 4

[<Literal>]
let private PromNone = 0

// Fathom piece type (1-based)
[<Literal>]
let private FPawn = 1

[<Literal>]
let private FKnight = 2

[<Literal>]
let private FBishop = 3

[<Literal>]
let private FRook = 4

[<Literal>]
let private FQueen = 5

[<Literal>]
let private FKing = 6

[<Literal>]
let private WPawn = 1

[<Literal>]
let private WKnight = 2

[<Literal>]
let private WBishop = 3

[<Literal>]
let private WRook = 4

[<Literal>]
let private WQueen = 5

[<Literal>]
let private WKing = 6

[<Literal>]
let private BPawn = 9

[<Literal>]
let private BKnight = 10

[<Literal>]
let private BBishop = 11

[<Literal>]
let private BRook = 12

[<Literal>]
let private BQueen = 13

[<Literal>]
let private BKing = 14

let private PrimeWQueen  = 11811845319353239651UL
let private PrimeWRook   = 10979190538029446137UL
let private PrimeWBishop = 12311744257139811149UL
let private PrimeWKnight = 15202887380319082783UL
let private PrimeWPawn   = 17008651141875982339UL
let private PrimeBQueen  = 15484752644942473553UL
let private PrimeBRook   = 18264461213049635989UL
let private PrimeBBishop = 15394650811035483107UL
let private PrimeBKnight = 13469005675588064321UL
let private PrimeBPawn   = 11695583624105689831UL

[<Literal>]
let private WDL = 0

[<Literal>]
let private DTZ = 2

[<Literal>]
let private PieceEnc = 0

[<Literal>]
let private FileEnc = 1

[<Literal>]
let private RankEnc = 2

let private tbSuffix = [| ".rtbw"; ".rtbm"; ".rtbz" |]
let private tbMagic = [| 0x5d23e871u; 0x88ac504bu; 0xa50c66d7u |]

let private pieceToChar = " PNBRQK  pnbrqk"

let private charToPieceType (c: char) =
    match c with
    | 'P' -> FPawn | 'N' -> FKnight | 'B' -> FBishop
    | 'R' -> FRook | 'Q' -> FQueen  | 'K' -> FKing
    | _ -> 0

let inline private colorOfFPiece (pc: int) = pc >>> 3 = 0
let inline private typeOfFPiece (pc: int) = pc &&& 7

let private WdlToDtz = [| -1; -101; 0; 101; 1 |]
let private WdlToMap = [| 1; 3; 0; 2; 0 |]
let private PAFlags = [| 8uy; 0uy; 0uy; 0uy; 4uy |]
let private FileToFile = [| 0uy; 1uy; 2uy; 3uy; 3uy; 2uy; 1uy; 0uy |]

// ---------------------------------------------------------------------------
// Lookup Tables — all byte/int8/int16 arrays; every continuation line
// is at EXACTLY the same indent as the first element.
// ---------------------------------------------------------------------------
let private OffDiag: int8[] = [|
    0y;-1y;-1y;-1y;-1y;-1y;-1y;-1y;
    1y; 0y;-1y;-1y;-1y;-1y;-1y;-1y;
    1y; 1y; 0y;-1y;-1y;-1y;-1y;-1y;
    1y; 1y; 1y; 0y;-1y;-1y;-1y;-1y;
    1y; 1y; 1y; 1y; 0y;-1y;-1y;-1y;
    1y; 1y; 1y; 1y; 1y; 0y;-1y;-1y;
    1y; 1y; 1y; 1y; 1y; 1y; 0y;-1y;
    1y; 1y; 1y; 1y; 1y; 1y; 1y; 0y |]

let private Triangle: int[] = [|
    6; 0; 1; 2; 2; 1; 0; 6;
    0; 7; 3; 4; 4; 3; 7; 0;
    1; 3; 8; 5; 5; 8; 3; 1;
    2; 4; 5; 9; 9; 5; 4; 2;
    2; 4; 5; 9; 9; 5; 4; 2;
    1; 3; 8; 5; 5; 8; 3; 1;
    0; 7; 3; 4; 4; 3; 7; 0;
    6; 0; 1; 2; 2; 1; 0; 6 |]

let private FlipDiag: int[] = [|
     0;  8; 16; 24; 32; 40; 48; 56;
     1;  9; 17; 25; 33; 41; 49; 57;
     2; 10; 18; 26; 34; 42; 50; 58;
     3; 11; 19; 27; 35; 43; 51; 59;
     4; 12; 20; 28; 36; 44; 52; 60;
     5; 13; 21; 29; 37; 45; 53; 61;
     6; 14; 22; 30; 38; 46; 54; 62;
     7; 15; 23; 31; 39; 47; 55; 63 |]

let private Lower: int[] = [|
    28;  0;  1;  2;  3;  4;  5;  6;
     0; 29;  7;  8;  9; 10; 11; 12;
     1;  7; 30; 13; 14; 15; 16; 17;
     2;  8; 13; 31; 18; 19; 20; 21;
     3;  9; 14; 18; 32; 22; 23; 24;
     4; 10; 15; 19; 22; 33; 25; 26;
     5; 11; 16; 20; 23; 25; 34; 27;
     6; 12; 17; 21; 24; 26; 27; 35 |]

let private DiagTbl: int[] = [|
     0;  0;  0;  0;  0;  0;  0;  8;
     0;  1;  0;  0;  0;  0;  9;  0;
     0;  0;  2;  0;  0; 10;  0;  0;
     0;  0;  0;  3; 11;  0;  0;  0;
     0;  0;  0; 12;  4;  0;  0;  0;
     0;  0; 13;  0;  0;  5;  0;  0;
     0; 14;  0;  0;  0;  0;  6;  0;
    15;  0;  0;  0;  0;  0;  0;  7 |]

let private Flap: int[,] =
    array2D [|
        [| 0; 0; 0; 0; 0; 0; 0; 0;  0; 6;12;18;18;12; 6; 0;  1; 7;13;19;19;13; 7; 1;
           2; 8;14;20;20;14; 8; 2;  3; 9;15;21;21;15; 9; 3;  4;10;16;22;22;16;10; 4;
           5;11;17;23;23;17;11; 5;  0; 0; 0; 0; 0; 0; 0; 0 |]
        [| 0; 0; 0; 0; 0; 0; 0; 0;  0; 1; 2; 3; 3; 2; 1; 0;  4; 5; 6; 7; 7; 6; 5; 4;
           8; 9;10;11;11;10; 9; 8; 12;13;14;15;15;14;13;12; 16;17;18;19;19;18;17;16;
          20;21;22;23;23;22;21;20;  0; 0; 0; 0; 0; 0; 0; 0 |]
    |]

let private PawnTwist: int[,] =
    array2D [|
        [|  0; 0; 0; 0; 0; 0; 0; 0; 47;35;23;11;10;22;34;46; 45;33;21; 9; 8;20;32;44;
           43;31;19; 7; 6;18;30;42; 41;29;17; 5; 4;16;28;40; 39;27;15; 3; 2;14;26;38;
           37;25;13; 1; 0;12;24;36;  0; 0; 0; 0; 0; 0; 0; 0 |]
        [|  0; 0; 0; 0; 0; 0; 0; 0; 47;45;43;41;40;42;44;46; 39;37;35;33;32;34;36;38;
           31;29;27;25;24;26;28;30; 23;21;19;17;16;18;20;22; 15;13;11; 9; 8;10;12;14;
            7; 5; 3; 1; 0; 2; 4; 6;  0; 0; 0; 0; 0; 0; 0; 0 |]
    |]

let private KKIdx: int[,] =
    array2D [|
        [| -1; -1; -1;  0;  1;  2;  3;  4; -1; -1; -1;  5;  6;  7;  8;  9;
           10; 11; 12; 13; 14; 15; 16; 17; 18; 19; 20; 21; 22; 23; 24; 25;
           26; 27; 28; 29; 30; 31; 32; 33; 34; 35; 36; 37; 38; 39; 40; 41;
           42; 43; 44; 45; 46; 47; 48; 49; 50; 51; 52; 53; 54; 55; 56; 57 |]
        [| 58; -1; -1; -1; 59; 60; 61; 62; 63; -1; -1; -1; 64; 65; 66; 67;
           68; 69; 70; 71; 72; 73; 74; 75; 76; 77; 78; 79; 80; 81; 82; 83;
           84; 85; 86; 87; 88; 89; 90; 91; 92; 93; 94; 95; 96; 97; 98; 99;
          100;101;102;103;104;105;106;107;108;109;110;111;112;113;114;115 |]
        [|116;117; -1; -1; -1;118;119;120;121;122; -1; -1; -1;123;124;125;
          126;127;128;129;130;131;132;133;134;135;136;137;138;139;140;141;
          142;143;144;145;146;147;148;149;150;151;152;153;154;155;156;157;
          158;159;160;161;162;163;164;165;166;167;168;169;170;171;172;173 |]
        [|174; -1; -1; -1;175;176;177;178;179; -1; -1; -1;180;181;182;183;
          184; -1; -1; -1;185;186;187;188;189;190;191;192;193;194;195;196;
          197;198;199;200;201;202;203;204;205;206;207;208;209;210;211;212;
          213;214;215;216;217;218;219;220;221;222;223;224;225;226;227;228 |]
        [|229;230; -1; -1; -1;231;232;233;234;235; -1; -1; -1;236;237;238;
          239;240; -1; -1; -1;241;242;243;244;245;246;247;248;249;250;251;
          252;253;254;255;256;257;258;259;260;261;262;263;264;265;266;267;
          268;269;270;271;272;273;274;275;276;277;278;279;280;281;282;283 |]
        [|284;285;286;287;288;289;290;291;292;293; -1; -1; -1;294;295;296;
          297;298; -1; -1; -1;299;300;301;302;303; -1; -1; -1;304;305;306;
          307;308;309;310;311;312;313;314;315;316;317;318;319;320;321;322;
          323;324;325;326;327;328;329;330;331;332;333;334;335;336;337;338 |]
        [| -1; -1;339;340;341;342;343;344; -1; -1;345;346;347;348;349;350;
           -1; -1;441;351;352;353;354;355; -1; -1; -1;442;356;357;358;359;
           -1; -1; -1; -1;443;360;361;362; -1; -1; -1; -1; -1;444;363;364;
           -1; -1; -1; -1; -1; -1;445;365; -1; -1; -1; -1; -1; -1; -1;446 |]
        [| -1; -1; -1;366;367;368;369;370; -1; -1; -1;371;372;373;374;375;
           -1; -1; -1;376;377;378;379;380; -1; -1; -1;447;381;382;383;384;
           -1; -1; -1; -1;448;385;386;387; -1; -1; -1; -1; -1;449;388;389;
           -1; -1; -1; -1; -1; -1;450;390; -1; -1; -1; -1; -1; -1; -1;451 |]
        [|452;391;392;393;394;395;396;397; -1; -1; -1; -1;398;399;400;401;
           -1; -1; -1; -1;402;403;404;405; -1; -1; -1; -1;406;407;408;409;
           -1; -1; -1; -1;453;410;411;412; -1; -1; -1; -1; -1;454;413;414;
           -1; -1; -1; -1; -1; -1;455;415; -1; -1; -1; -1; -1; -1; -1;456 |]
        [|457;416;417;418;419;420;421;422; -1;458;423;424;425;426;427;428;
           -1; -1; -1; -1; -1;429;430;431; -1; -1; -1; -1; -1;432;433;434;
           -1; -1; -1; -1; -1;435;436;437; -1; -1; -1; -1; -1;459;438;439;
           -1; -1; -1; -1; -1; -1;460;440; -1; -1; -1; -1; -1; -1; -1;461 |]
    |]

// ---------------------------------------------------------------------------
// Computed index tables
// ---------------------------------------------------------------------------
let private Binomial: uint64[,] = Array2D.zeroCreate 7 64
let private PawnIdxTbl: uint64[,,] = Array3D.zeroCreate 2 6 24
let private PawnFactorFile: uint64[,] = Array2D.zeroCreate 6 4
let private PawnFactorRank: uint64[,] = Array2D.zeroCreate 6 6

let private initIndices () =
    for i in 0..6 do
        for j in 0..63 do
            let mutable f = 1UL
            let mutable l = 1UL
            for k in 0..i-1 do
                f <- f * uint64 (j - k)
                l <- l * uint64 (k + 1)
            Binomial.[i, j] <- f / l

    for i in 0..5 do
        let mutable s = 0UL
        for j in 0..23 do
            PawnIdxTbl.[0, i, j] <- s
            s <- s + Binomial.[i, PawnTwist.[0, (1 + (j % 6)) * 8 + (j / 6)]]
            if (j + 1) % 6 = 0 then
                PawnFactorFile.[i, j / 6] <- s
                s <- 0UL

    for i in 0..5 do
        let mutable s = 0UL
        for j in 0..23 do
            PawnIdxTbl.[1, i, j] <- s
            s <- s + Binomial.[i, PawnTwist.[1, (1 + (j / 4)) * 8 + (j % 4)]]
            if (j + 1) % 4 = 0 then
                PawnFactorRank.[i, j / 4] <- s
                s <- 0UL

// ---------------------------------------------------------------------------
// Internal lightweight position for capture resolution
// ---------------------------------------------------------------------------
[<Struct; NoComparison; NoEquality>]
type private Pos =
    val mutable W: uint64
    val mutable B: uint64
    val mutable K: uint64
    val mutable Q: uint64
    val mutable R: uint64
    val mutable Bi: uint64
    val mutable N: uint64
    val mutable P: uint64
    val mutable Rule50: byte
    val mutable Ep: byte
    val mutable Turn: bool

let private piecesOf (pos: Pos byref) (white: bool) (pt: int) =
    let mask = if white then pos.W else pos.B
    match pt with
    | 1 -> pos.P &&& mask
    | 2 -> pos.N &&& mask
    | 3 -> pos.Bi &&& mask
    | 4 -> pos.R &&& mask
    | 5 -> pos.Q &&& mask
    | 6 -> pos.K &&& mask
    | _ -> 0UL

let private calcKey (pos: Pos byref) (mirror: bool) =
    let w, b = if mirror then pos.B, pos.W else pos.W, pos.B
    uint64 (popCount (w &&& pos.Q))  * PrimeWQueen  +
    uint64 (popCount (w &&& pos.R))  * PrimeWRook   +
    uint64 (popCount (w &&& pos.Bi)) * PrimeWBishop +
    uint64 (popCount (w &&& pos.N))  * PrimeWKnight +
    uint64 (popCount (w &&& pos.P))  * PrimeWPawn   +
    uint64 (popCount (b &&& pos.Q))  * PrimeBQueen  +
    uint64 (popCount (b &&& pos.R))  * PrimeBRook   +
    uint64 (popCount (b &&& pos.Bi)) * PrimeBBishop +
    uint64 (popCount (b &&& pos.N))  * PrimeBKnight +
    uint64 (popCount (b &&& pos.P))  * PrimeBPawn

let private calcKeyFromPcs (pcs: int[]) (mirror: bool) =
    let m = if mirror then 8 else 0
    uint64 pcs.[WQueen ^^^ m]  * PrimeWQueen  +
    uint64 pcs.[WRook ^^^ m]   * PrimeWRook   +
    uint64 pcs.[WBishop ^^^ m] * PrimeWBishop +
    uint64 pcs.[WKnight ^^^ m] * PrimeWKnight +
    uint64 pcs.[WPawn ^^^ m]   * PrimeWPawn   +
    uint64 pcs.[BQueen ^^^ m]  * PrimeBQueen  +
    uint64 pcs.[BRook ^^^ m]   * PrimeBRook   +
    uint64 pcs.[BBishop ^^^ m] * PrimeBBishop +
    uint64 pcs.[BKnight ^^^ m] * PrimeBKnight +
    uint64 pcs.[BPawn ^^^ m]   * PrimeBPawn

// ---------------------------------------------------------------------------
// Mini-movegen (reuses Eonego.Bitboard attack tables)
// Fathom WHITE=true → Eonego White=0; Fathom BLACK=false → Eonego Black=1
// ---------------------------------------------------------------------------
let inline private eColor (fathomWhite: bool) = if fathomWhite then 0 else 1

let inline private doBbMove (b: uint64) (fr: int) (dst: int) =
    (b &&& ~~~(1UL <<< dst) &&& ~~~(1UL <<< fr)) ||| (((b >>> fr) &&& 1UL) <<< dst)

type private TbMove = uint16

let inline private mkTbMove (prom: int) (fr: int) (dst: int) : TbMove =
    uint16 (((prom &&& 7) <<< 12) ||| ((fr &&& 0x3F) <<< 6) ||| (dst &&& 0x3F))

let inline private tbMoveFrom (m: TbMove) = (int m >>> 6) &&& 0x3F
let inline private tbMoveTo (m: TbMove) = int m &&& 0x3F
let inline private tbMovePromotes (m: TbMove) = (int m >>> 12) &&& 7

let private isLegal (pos: Pos byref) =
    let occ = pos.W ||| pos.B
    let us = if pos.Turn then pos.B else pos.W
    let them = if pos.Turn then pos.W else pos.B
    let king = pos.K &&& us
    if king = 0UL then false
    else
        let sq = lsb king
        if kingAttacks sq &&& (pos.K &&& them) <> 0UL then false
        else
            let ratt = rookAttacks sq occ
            let batt = bishopAttacks sq occ
            if ratt &&& (pos.R &&& them) <> 0UL then false
            elif batt &&& (pos.Bi &&& them) <> 0UL then false
            elif (ratt ||| batt) &&& (pos.Q &&& them) <> 0UL then false
            elif knightAttacks sq &&& (pos.N &&& them) <> 0UL then false
            elif pawnAttacks (eColor (not pos.Turn)) sq &&& (pos.P &&& them) <> 0UL then false
            else true

let private isCheck (pos: Pos byref) =
    let occ = pos.W ||| pos.B
    let us = if pos.Turn then pos.W else pos.B
    let them = if pos.Turn then pos.B else pos.W
    let sq = lsb (pos.K &&& us)
    let ratt = rookAttacks sq occ
    let batt = bishopAttacks sq occ
    ratt &&& (pos.R &&& them) <> 0UL
    || batt &&& (pos.Bi &&& them) <> 0UL
    || (ratt ||| batt) &&& (pos.Q &&& them) <> 0UL
    || knightAttacks sq &&& (pos.N &&& them) <> 0UL
    || pawnAttacks (eColor pos.Turn) sq &&& (pos.P &&& them) <> 0UL

let private isEnPassant (pos: Pos byref) (m: TbMove) =
    let fr = tbMoveFrom m
    let dst = tbMoveTo m
    let us = if pos.Turn then pos.W else pos.B
    pos.Ep <> 0uy && dst = int pos.Ep && (1UL <<< fr) &&& us &&& pos.P <> 0UL

let private isCapture (pos: Pos byref) (m: TbMove) =
    let dst = tbMoveTo m
    let them = if pos.Turn then pos.B else pos.W
    them &&& (1UL <<< dst) <> 0UL || isEnPassant &pos m

let private doMove (pos1: Pos byref) (pos0: Pos byref) (m: TbMove) : bool =
    let fr = tbMoveFrom m
    let dst = tbMoveTo m
    let prom = tbMovePromotes m
    pos1.Turn <- not pos0.Turn
    pos1.W <- doBbMove pos0.W fr dst
    pos1.B <- doBbMove pos0.B fr dst
    pos1.K <- doBbMove pos0.K fr dst
    pos1.Q <- doBbMove pos0.Q fr dst
    pos1.R <- doBbMove pos0.R fr dst
    pos1.Bi <- doBbMove pos0.Bi fr dst
    pos1.N <- doBbMove pos0.N fr dst
    pos1.P <- doBbMove pos0.P fr dst
    pos1.Ep <- 0uy
    if prom <> PromNone then
        pos1.P <- pos1.P &&& ~~~(1UL <<< dst)
        match prom with
        | 1 -> pos1.Q <- pos1.Q ||| (1UL <<< dst)
        | 2 -> pos1.R <- pos1.R ||| (1UL <<< dst)
        | 3 -> pos1.Bi <- pos1.Bi ||| (1UL <<< dst)
        | 4 -> pos1.N <- pos1.N ||| (1UL <<< dst)
        | _ -> ()
        pos1.Rule50 <- 0uy
    elif (1UL <<< fr) &&& pos0.P <> 0UL then
        pos1.Rule50 <- 0uy
        if (fr >>> 3) = 1 && (dst >>> 3) = 3 &&
           pawnAttacks (eColor true) (fr + 8) &&& pos0.P &&& pos0.B <> 0UL then
            pos1.Ep <- byte (fr + 8)
        elif (fr >>> 3) = 6 && (dst >>> 3) = 4 &&
             pawnAttacks (eColor false) (fr - 8) &&& pos0.P &&& pos0.W <> 0UL then
            pos1.Ep <- byte (fr - 8)
        elif dst = int pos0.Ep then
            let epTo = if pos0.Turn then dst - 8 else dst + 8
            let epMask = ~~~(1UL <<< epTo)
            pos1.W <- pos1.W &&& epMask
            pos1.B <- pos1.B &&& epMask
            pos1.P <- pos1.P &&& epMask
    elif (1UL <<< dst) &&& (pos0.W ||| pos0.B) <> 0UL then
        pos1.Rule50 <- 0uy
    else
        pos1.Rule50 <- pos0.Rule50 + 1uy
    isLegal &pos1

let private addTbMoves (moves: TbMove[]) (n: byref<int>) (promotes: bool) (fr: int) (dst: int) =
    if not promotes then
        moves.[n] <- mkTbMove PromNone fr dst; n <- n + 1
    else
        moves.[n] <- mkTbMove 1 fr dst; n <- n + 1
        moves.[n] <- mkTbMove 4 fr dst; n <- n + 1
        moves.[n] <- mkTbMove 2 fr dst; n <- n + 1
        moves.[n] <- mkTbMove 3 fr dst; n <- n + 1

let private genCaptures (pos: Pos byref) (moves: TbMove[]) : int =
    let occ = pos.W ||| pos.B
    let us = if pos.Turn then pos.W else pos.B
    let them = if pos.Turn then pos.B else pos.W
    let mutable n = 0
    let ksq = lsb (pos.K &&& us)
    let mutable att = kingAttacks ksq &&& them
    while att <> 0UL do
        let d = popLsb &att in addTbMoves moves &n false ksq d
    let mutable bb = us &&& pos.Q
    while bb <> 0UL do
        let fr = popLsb &bb
        att <- queenAttacks fr occ &&& them
        while att <> 0UL do let d = popLsb &att in addTbMoves moves &n false fr d
    bb <- us &&& pos.R
    while bb <> 0UL do
        let fr = popLsb &bb
        att <- rookAttacks fr occ &&& them
        while att <> 0UL do let d = popLsb &att in addTbMoves moves &n false fr d
    bb <- us &&& pos.Bi
    while bb <> 0UL do
        let fr = popLsb &bb
        att <- bishopAttacks fr occ &&& them
        while att <> 0UL do let d = popLsb &att in addTbMoves moves &n false fr d
    bb <- us &&& pos.N
    while bb <> 0UL do
        let fr = popLsb &bb
        att <- knightAttacks fr &&& them
        while att <> 0UL do let d = popLsb &att in addTbMoves moves &n false fr d
    bb <- us &&& pos.P
    while bb <> 0UL do
        let fr = popLsb &bb
        att <- pawnAttacks (eColor pos.Turn) fr
        if pos.Ep <> 0uy && att &&& (1UL <<< int pos.Ep) <> 0UL then
            addTbMoves moves &n false fr (int pos.Ep)
        att <- att &&& them
        while att <> 0UL do
            let d = popLsb &att
            addTbMoves moves &n ((d >>> 3) = 7 || (d >>> 3) = 0) fr d
    n

let private genMoves (pos: Pos byref) (moves: TbMove[]) : int =
    let occ = pos.W ||| pos.B
    let us = if pos.Turn then pos.W else pos.B
    let them = if pos.Turn then pos.B else pos.W
    let mutable n = 0
    let ksq = lsb (pos.K &&& us)
    let mutable att = kingAttacks ksq &&& ~~~us
    while att <> 0UL do let d = popLsb &att in addTbMoves moves &n false ksq d
    let mutable bb = us &&& pos.Q
    while bb <> 0UL do
        let fr = popLsb &bb
        att <- queenAttacks fr occ &&& ~~~us
        while att <> 0UL do let d = popLsb &att in addTbMoves moves &n false fr d
    bb <- us &&& pos.R
    while bb <> 0UL do
        let fr = popLsb &bb
        att <- rookAttacks fr occ &&& ~~~us
        while att <> 0UL do let d = popLsb &att in addTbMoves moves &n false fr d
    bb <- us &&& pos.Bi
    while bb <> 0UL do
        let fr = popLsb &bb
        att <- bishopAttacks fr occ &&& ~~~us
        while att <> 0UL do let d = popLsb &att in addTbMoves moves &n false fr d
    bb <- us &&& pos.N
    while bb <> 0UL do
        let fr = popLsb &bb
        att <- knightAttacks fr &&& ~~~us
        while att <> 0UL do let d = popLsb &att in addTbMoves moves &n false fr d
    bb <- us &&& pos.P
    while bb <> 0UL do
        let fr = popLsb &bb
        let next = if pos.Turn then fr + 8 else fr - 8
        att <- pawnAttacks (eColor pos.Turn) fr
        if pos.Ep <> 0uy && att &&& (1UL <<< int pos.Ep) <> 0UL then
            addTbMoves moves &n false fr (int pos.Ep)
        att <- att &&& them
        if (1UL <<< next) &&& occ = 0UL then
            att <- att ||| (1UL <<< next)
            let next2 = if pos.Turn then fr + 16 else fr - 16
            if (if pos.Turn then (fr >>> 3) = 1 else (fr >>> 3) = 6)
               && (1UL <<< next2) &&& occ = 0UL then
                att <- att ||| (1UL <<< next2)
        while att <> 0UL do
            let d = popLsb &att
            addTbMoves moves &n ((d >>> 3) = 7 || (d >>> 3) = 0) fr d
    n

let private legalMove (pos: Pos byref) (m: TbMove) =
    let mutable pos1 = Pos()
    doMove &pos1 &pos m

let private genLegal (pos: Pos byref) (moves: TbMove[]) : int =
    let pl = Array.zeroCreate<TbMove> TbMaxMoves
    let nPl = genMoves &pos pl
    let mutable n = 0
    for i in 0..nPl-1 do
        if legalMove &pos pl.[i] then
            moves.[n] <- pl.[i]; n <- n + 1
    n

let private isMate (pos: Pos byref) =
    if not (isCheck &pos) then false
    else
        let moves = Array.zeroCreate<TbMove> TbMaxMoves
        let n = genMoves &pos moves
        let mutable found = false
        for i in 0..n-1 do
            if not found then
                let mutable pos1 = Pos()
                if doMove &pos1 &pos moves.[i] then found <- true
        not found

let private typeOfPieceMoved (pos: Pos byref) (m: TbMove) =
    let fr = tbMoveFrom m
    let bb = 1UL <<< fr
    let us = if pos.Turn then pos.W else pos.B
    if bb &&& us &&& pos.P <> 0UL then FPawn
    elif bb &&& us &&& pos.N <> 0UL then FKnight
    elif bb &&& us &&& pos.Bi <> 0UL then FBishop
    elif bb &&& us &&& pos.R <> 0UL then FRook
    elif bb &&& us &&& pos.Q <> 0UL then FQueen
    else FKing

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------
[<AllowNullLiteral>]
type private PairsData() =
    member val IndexTableOff = 0 with get, set
    member val SizeTableOff = 0 with get, set
    member val DataOff = 0 with get, set
    member val OffsetOff = 0 with get, set
    member val SymLen: byte[] = null with get, set
    member val SymPatOff = 0 with get, set
    member val BlockSize: byte = 0uy with get, set
    member val IdxBits: byte = 0uy with get, set
    member val MinLen: byte = 0uy with get, set
    member val ConstValue: byte[] = Array.zeroCreate 2 with get, set
    member val Base: uint64[] = null with get, set

type private EncInfo() =
    member val Precomp: PairsData = null with get, set
    member val Factor: uint64[] = Array.zeroCreate TbPieces with get, set
    member val Pieces: byte[] = Array.zeroCreate TbPieces with get, set
    member val Norm: byte[] = Array.zeroCreate TbPieces with get, set

// A single entry indexed from the hash table. PieceEntry/PawnEntry hold a ref
// to this plus the encoding/DTZ arrays specific to their layout.
[<AllowNullLiteral>]
type private BaseEntry() =
    member val Key = 0UL with get, set
    member val Data: byte[][] = Array.zeroCreate 3 with get, set
    member val Mapping: IDisposable[] = Array.zeroCreate 3 with get, set
    member val Ready: int[] = Array.zeroCreate 3 with get, set
    member val Num: byte = 0uy with get, set
    member val Symmetric = false with get, set
    member val HasPawns = false with get, set
    member val HasDtm = false with get, set
    member val HasDtz = false with get, set
    member val KkEnc = false with get, set
    member val Pawns: byte[] = Array.zeroCreate 2 with get, set
    member val DtmLossOnly = false with get, set
    member val EntryIndex = -1 with get, set

type private PieceEntry() =
    member val Be = BaseEntry() with get, set
    member val Ei: EncInfo[] = Array.init 5 (fun _ -> EncInfo()) with get, set
    member val DtzFlags: byte = 0uy with get, set
    member val DtzMapOff = 0 with get, set
    member val DtzMapIdx: int[] = Array.zeroCreate 4 with get, set

type private PawnEntry() =
    member val Be = BaseEntry() with get, set
    member val Ei: EncInfo[] = Array.init 24 (fun _ -> EncInfo()) with get, set
    member val DtzFlags: byte[] = Array.zeroCreate 4 with get, set
    member val DtzMapOff = 0 with get, set
    member val DtzMapIdx: int[,] = Array2D.zeroCreate 4 4 with get, set

[<Struct>]
type private TbHashEntry =
    val mutable Key: uint64
    val mutable Ptr: BaseEntry
    val mutable Error: int

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------
let private tbMutex = obj ()
let mutable private initialized = false
let mutable private numPaths = 0
let mutable private paths: string[] = Array.empty
let mutable private tbNumPiece = 0
let mutable private tbNumPawn = 0
let mutable private numWdl = 0
let mutable private numDtz = 0
let mutable private pieceEntries: PieceEntry[] = null
let mutable private pawnEntries: PawnEntry[] = null
let private tbHash: TbHashEntry[] = Array.zeroCreate (1 <<< TbHashBits)

/// Maximum piece count of available tables (0 if none loaded).
let mutable Largest = 0

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------
let private openTb (name: string) (suffix: string) : string option =
    paths |> Array.tryPick (fun p ->
        let file = Path.Combine(p, name + suffix)
        if File.Exists file then Some file else None)

let private testTb (name: string) (suffix: string) : bool =
    match openTb name suffix with
    | None -> false
    | Some file ->
        let fi = FileInfo(file)
        if (fi.Length &&& 63L) <> 16L then
            Console.Error.WriteLine("Incomplete tablebase file " + name + suffix)
            false
        else true

let private mapTb (name: string) (suffix: string) : byte[] =
    match openTb name suffix with
    | None -> null
    | Some file -> File.ReadAllBytes file

let private readLe32 (data: byte[]) (off: int) =
    uint32 data.[off]
    ||| (uint32 data.[off+1] <<< 8)
    ||| (uint32 data.[off+2] <<< 16)
    ||| (uint32 data.[off+3] <<< 24)

let private readLe16 (data: byte[]) (off: int) =
    uint16 data.[off] ||| (uint16 data.[off+1] <<< 8)

// ---------------------------------------------------------------------------
// Hash table
// ---------------------------------------------------------------------------
let private addToHash (ptr: BaseEntry) (key: uint64) =
    let mutable idx = int (key >>> (64 - TbHashBits))
    while tbHash.[idx].Ptr <> null do
        idx <- (idx + 1) &&& ((1 <<< TbHashBits) - 1)
    tbHash.[idx].Key <- key
    tbHash.[idx].Ptr <- ptr
    tbHash.[idx].Error <- 0

// ---------------------------------------------------------------------------
// String generation for material keys
// ---------------------------------------------------------------------------
let private prtStr (pos: Pos byref) (flip: bool) =
    let sb = System.Text.StringBuilder()
    let color = not flip
    for pt in [| FKing; FQueen; FRook; FBishop; FKnight; FPawn |] do
        let n = popCount (piecesOf &pos color pt)
        for _ in 1..n do sb.Append(pieceToChar.[pt]) |> ignore
    sb.Append('v') |> ignore
    let color2 = flip
    for pt in [| FKing; FQueen; FRook; FBishop; FKnight; FPawn |] do
        let n = popCount (piecesOf &pos color2 pt)
        for _ in 1..n do sb.Append(pieceToChar.[pt]) |> ignore
    sb.ToString()

// ---------------------------------------------------------------------------
// Table init
// ---------------------------------------------------------------------------
let private pchr i = pieceToChar.[FQueen - i]

let private initTb (str: string) =
    if not (testTb str tbSuffix.[WDL]) then ()
    else
        let pcs = Array.zeroCreate 16
        let mutable color = 0
        for c in str do
            if c = 'v' then color <- 8
            else
                let pt = charToPieceType c
                if pt <> 0 then
                    pcs.[pt ||| color] <- pcs.[pt ||| color] + 1

        let key = calcKeyFromPcs pcs false
        let key2 = calcKeyFromPcs pcs true
        let hasPawns = pcs.[WPawn] > 0 || pcs.[BPawn] > 0

        let be =
            if hasPawns then
                let pe = pawnEntries.[tbNumPawn]
                pe.Be.EntryIndex <- tbNumPawn
                tbNumPawn <- tbNumPawn + 1
                pe.Be
            else
                let pe = pieceEntries.[tbNumPiece]
                pe.Be.EntryIndex <- tbNumPiece
                tbNumPiece <- tbNumPiece + 1
                pe.Be

        be.HasPawns <- hasPawns
        be.Key <- key
        be.Symmetric <- (key = key2)
        let mutable num = 0uy
        for i in 0..15 do num <- num + byte pcs.[i]
        be.Num <- num

        numWdl <- numWdl + 1
        be.HasDtz <- testTb str tbSuffix.[DTZ]
        if be.HasDtz then numDtz <- numDtz + 1

        if int be.Num > Largest then Largest <- int be.Num

        if not hasPawns then
            let mutable j = 0
            for i in 0..15 do if pcs.[i] = 1 then j <- j + 1
            be.KkEnc <- (j = 2)
        else
            be.Pawns.[0] <- byte pcs.[WPawn]
            be.Pawns.[1] <- byte pcs.[BPawn]
            if pcs.[BPawn] > 0 && (pcs.[WPawn] = 0 || pcs.[WPawn] > pcs.[BPawn]) then
                let tmp = be.Pawns.[0]
                be.Pawns.[0] <- be.Pawns.[1]
                be.Pawns.[1] <- tmp

        addToHash be key
        if key <> key2 then addToHash be key2

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------
let private subfactor (k: uint64) (n: uint64) =
    if k = 0UL then 1UL
    else
        let mutable f = n
        let mutable l = 1UL
        for i in 1UL .. k-1UL do
            f <- f * (n - i)
            l <- l * (i + 1UL)
        f / l

let private initEncInfo (ei: EncInfo) (be: BaseEntry) (data: byte[]) (off: int) (shift: int) (t: int) (enc: int) =
    let morePawns = enc <> PieceEnc && be.Pawns.[1] > 0uy

    for i in 0 .. int be.Num - 1 do
        ei.Pieces.[i] <- (data.[off + i + 1 + (if morePawns then 1 else 0)] >>> shift) &&& 0x0Fuy
        ei.Norm.[i] <- 0uy

    let order = int ((data.[off] >>> shift) &&& 0x0Fuy)
    let order2 = if morePawns then int ((data.[off + 1] >>> shift) &&& 0x0Fuy) else 0x0F

    let mutable k =
        if enc <> PieceEnc then int be.Pawns.[0]
        elif be.KkEnc then 2
        else 3
    ei.Norm.[0] <- byte k

    if morePawns then
        ei.Norm.[k] <- be.Pawns.[1]
        k <- k + int ei.Norm.[k]

    let mutable i2 = k
    while i2 < int be.Num do
        let mutable j2 = i2
        while j2 < int be.Num && ei.Pieces.[j2] = ei.Pieces.[i2] do
            ei.Norm.[i2] <- ei.Norm.[i2] + 1uy
            j2 <- j2 + 1
        i2 <- i2 + int ei.Norm.[i2]

    let mutable n = 64 - k
    let mutable f = 1UL
    let mutable i3 = 0

    while k < int be.Num || i3 = order || i3 = order2 do
        if i3 = order then
            ei.Factor.[0] <- f
            f <- f * (
                if enc = FileEnc then PawnFactorFile.[int ei.Norm.[0] - 1, t]
                elif enc = RankEnc then PawnFactorRank.[int ei.Norm.[0] - 1, t]
                elif be.KkEnc then 462UL
                else 31332UL)
        elif i3 = order2 then
            ei.Factor.[int ei.Norm.[0]] <- f
            f <- f * subfactor (uint64 ei.Norm.[int ei.Norm.[0]]) (uint64 (48 - int ei.Norm.[0]))
        else
            ei.Factor.[k] <- f
            f <- f * subfactor (uint64 ei.Norm.[k]) (uint64 n)
            n <- n - int ei.Norm.[k]
            k <- k + int ei.Norm.[k]
        i3 <- i3 + 1

    f

let private leadingPawn (p: int[]) (be: BaseEntry) (enc: int) =
    for i in 1 .. int be.Pawns.[0] - 1 do
        if Flap.[enc - 1, p.[0]] > Flap.[enc - 1, p.[i]] then
            let tmp = p.[0] in p.[0] <- p.[i]; p.[i] <- tmp
    if enc = FileEnc then int FileToFile.[p.[0] &&& 7] else (p.[0] - 8) >>> 3

let private encode (p: int[]) (ei: EncInfo) (be: BaseEntry) (enc: int) =
    let n = int be.Num
    let mutable idx = 0UL
    let mutable k = 0

    if p.[0] &&& 0x04 <> 0 then
        for i in 0..n-1 do p.[i] <- p.[i] ^^^ 0x07

    if enc = PieceEnc then
        if p.[0] &&& 0x20 <> 0 then
            for i in 0..n-1 do p.[i] <- p.[i] ^^^ 0x38

        let mutable broke = false
        for i in 0..n-1 do
            if not broke && OffDiag.[p.[i]] <> 0y then
                if int OffDiag.[p.[i]] > 0 && i < (if be.KkEnc then 2 else 3) then
                    for j in 0..n-1 do p.[j] <- FlipDiag.[p.[j]]
                broke <- true

        if be.KkEnc then
            idx <- uint64 KKIdx.[Triangle.[p.[0]], p.[1]]
            k <- 2
        else
            let s1 = if p.[1] > p.[0] then 1 else 0
            let s2 = (if p.[2] > p.[0] then 1 else 0) + (if p.[2] > p.[1] then 1 else 0)
            if OffDiag.[p.[0]] <> 0y then
                idx <- uint64 Triangle.[p.[0]] * 63UL * 62UL + uint64 (p.[1] - s1) * 62UL + uint64 (p.[2] - s2)
            elif OffDiag.[p.[1]] <> 0y then
                idx <- 6UL*63UL*62UL + uint64 DiagTbl.[p.[0]] * 28UL*62UL + uint64 Lower.[p.[1]] * 62UL + uint64 (p.[2] - s2)
            elif OffDiag.[p.[2]] <> 0y then
                idx <- 6UL*63UL*62UL + 4UL*28UL*62UL + uint64 DiagTbl.[p.[0]] * 7UL*28UL + uint64 (DiagTbl.[p.[1]] - s1) * 28UL + uint64 Lower.[p.[2]]
            else
                idx <- 6UL*63UL*62UL + 4UL*28UL*62UL + 4UL*7UL*28UL + uint64 DiagTbl.[p.[0]] * 7UL*6UL + uint64 (DiagTbl.[p.[1]] - s1) * 6UL + uint64 (DiagTbl.[p.[2]] - s2)
            k <- 3
        idx <- idx * ei.Factor.[0]
    else
        for i in 1 .. int be.Pawns.[0] - 1 do
            for j in i+1 .. int be.Pawns.[0] - 1 do
                if PawnTwist.[enc-1, p.[i]] < PawnTwist.[enc-1, p.[j]] then
                    let tmp = p.[i] in p.[i] <- p.[j]; p.[j] <- tmp

        k <- int be.Pawns.[0]
        idx <- PawnIdxTbl.[enc-1, k-1, Flap.[enc-1, p.[0]]]
        for i in 1..k-1 do
            idx <- idx + Binomial.[k-i, PawnTwist.[enc-1, p.[i]]]
        idx <- idx * ei.Factor.[0]

        if be.Pawns.[1] > 0uy then
            let t2 = k + int be.Pawns.[1]
            for i in k..t2-1 do
                for j in i+1..t2-1 do
                    if p.[i] > p.[j] then
                        let tmp = p.[i] in p.[i] <- p.[j]; p.[j] <- tmp
            let mutable s = 0UL
            for i in k..t2-1 do
                let sq = p.[i]
                let mutable skips = 0
                for j in 0..k-1 do if sq > p.[j] then skips <- skips + 1
                s <- s + Binomial.[i - k + 1, sq - skips - 8]
            idx <- idx + s * ei.Factor.[k]
            k <- t2

    while k < n do
        let t2 = k + int ei.Norm.[k]
        for i in k..t2-1 do
            for j in i+1..t2-1 do
                if p.[i] > p.[j] then
                    let tmp = p.[i] in p.[i] <- p.[j]; p.[j] <- tmp
        let mutable s = 0UL
        for i in k..t2-1 do
            let sq = p.[i]
            let mutable skips = 0
            for j in 0..k-1 do if sq > p.[j] then skips <- skips + 1
            s <- s + Binomial.[i - k + 1, sq - skips]
        idx <- idx + s * ei.Factor.[k]
        k <- t2

    idx

// ---------------------------------------------------------------------------
// Decompression
// ---------------------------------------------------------------------------
let private calcSymLen (d: PairsData) (data: byte[]) (s: uint32) (tmp: byte[]) =
    let rec calc s =
        let off = d.SymPatOff + 3 * int s
        let s2 = (uint32 data.[off + 2] <<< 4) ||| (uint32 data.[off + 1] >>> 4)
        if s2 = 0x0FFFu then
            d.SymLen.[int s] <- 0uy
        else
            let s1 = ((uint32 (data.[off + 1] &&& 0x0Fuy) <<< 8)) ||| uint32 data.[off]
            if tmp.[int s1] = 0uy then calc s1
            if tmp.[int s2] = 0uy then calc s2
            d.SymLen.[int s] <- d.SymLen.[int s1] + d.SymLen.[int s2] + 1uy
        tmp.[int s] <- 1uy
    calc s

let private setupPairs (data: byte[]) (ptr: byref<int>) (tbSize: uint64) (typ: int) : PairsData * byte * uint64[] =
    let flags = data.[ptr]
    if flags &&& 0x80uy <> 0uy then
        let d = PairsData()
        d.IdxBits <- 0uy
        d.ConstValue.[0] <- if typ = WDL then data.[ptr + 1] else 0uy
        d.ConstValue.[1] <- 0uy
        ptr <- ptr + 2
        d, flags, [| 0UL; 0UL; 0UL |]
    else
        let blockSize = data.[ptr + 1]
        let idxBits = data.[ptr + 2]
        let realNumBlocks = readLe32 data (ptr + 4)
        let numBlocks = realNumBlocks + uint32 data.[ptr + 3]
        let maxLen = int data.[ptr + 8]
        let minLen = int data.[ptr + 9]
        let h = maxLen - minLen + 1
        let numSyms = int (readLe16 data (ptr + 10 + 2 * h))

        let d = PairsData()
        d.BlockSize <- blockSize
        d.IdxBits <- idxBits
        d.OffsetOff <- ptr + 10
        d.SymLen <- Array.zeroCreate numSyms
        d.SymPatOff <- ptr + 12 + 2 * h
        d.MinLen <- byte minLen
        d.Base <- Array.zeroCreate h

        ptr <- ptr + 12 + 2 * h + 3 * numSyms + (numSyms &&& 1)

        let numIndices = (tbSize + (1UL <<< int idxBits) - 1UL) >>> int idxBits
        let size0 = 6UL * numIndices
        let size1 = 2UL * uint64 numBlocks
        let size2 = uint64 realNumBlocks <<< int blockSize

        let tmp = Array.zeroCreate<byte> numSyms
        for s in 0u .. uint32 numSyms - 1u do
            if tmp.[int s] = 0uy then calcSymLen d data s tmp

        if h >= 2 then
            d.Base.[h - 1] <- 0UL
            for i in h-2 .. -1 .. 0 do
                let off1 = int (readLe16 data (d.OffsetOff + 2*i))
                let off2 = int (readLe16 data (d.OffsetOff + 2*(i+1)))
                d.Base.[i] <- (d.Base.[i + 1] + uint64 off1 - uint64 off2) / 2UL
            for i in 0..h-1 do
                d.Base.[i] <- d.Base.[i] <<< (64 - (minLen + i))

        d, flags, [| size0; size1; size2 |]

// ---------------------------------------------------------------------------
// Table init (file mapping + pairs setup)
// ---------------------------------------------------------------------------
let private initTable (be: BaseEntry) (str: string) (typ: int) (eis: EncInfo[]) (eiOff: int) : bool =
    let data = mapTb str tbSuffix.[typ]
    if isNull data then false
    else
        if readLe32 data 0 <> tbMagic.[typ] then
            Console.Error.WriteLine("Corrupted table: " + str + tbSuffix.[typ])
            false
        else
            be.Data.[typ] <- data

            let split = typ <> DTZ && (data.[4] &&& 0x01uy <> 0uy)
            let mutable ptr = 5
            let num = if be.HasPawns then (if typ = 1 then 6 else 4) else 1
            let enc = if not be.HasPawns then PieceEnc elif typ <> 1 then FileEnc else RankEnc

            let tbSize = Array2D.zeroCreate<uint64> 6 2

            for t in 0..num-1 do
                tbSize.[t, 0] <- initEncInfo eis.[eiOff + t] be data ptr 0 t enc
                if split then
                    tbSize.[t, 1] <- initEncInfo eis.[eiOff + num + t] be data ptr 4 t enc
                ptr <- ptr + int be.Num + 1 + (if be.HasPawns && be.Pawns.[1] > 0uy then 1 else 0)
            if ptr &&& 1 <> 0 then ptr <- ptr + 1

            let sizes = Array3D.zeroCreate<uint64> 6 2 3
            for t in 0..num-1 do
                let d, f, sz = setupPairs data &ptr tbSize.[t, 0] typ
                eis.[eiOff + t].Precomp <- d
                sizes.[t, 0, 0] <- sz.[0]; sizes.[t, 0, 1] <- sz.[1]; sizes.[t, 0, 2] <- sz.[2]
                if typ = DTZ && not be.HasPawns then
                    pieceEntries.[be.EntryIndex].DtzFlags <- f
                elif typ = DTZ then
                    pawnEntries.[be.EntryIndex].DtzFlags.[t] <- f
                if split then
                    let d2, _, sz2 = setupPairs data &ptr tbSize.[t, 1] typ
                    eis.[eiOff + num + t].Precomp <- d2
                    sizes.[t, 1, 0] <- sz2.[0]; sizes.[t, 1, 1] <- sz2.[1]; sizes.[t, 1, 2] <- sz2.[2]
                elif typ <> DTZ then
                    eis.[eiOff + num + t].Precomp <- null

            // DTZ map: per-table remap from stored value to actual DTZ (flags bit 1 = map present,
            // bit 4 = 16-bit entries). mapIdx offsets are relative to the map start, in entry units
            // (bytes or uint16s) — mirroring Fathom's pointer arithmetic on the byte-array offsets.
            if typ = DTZ then
                let mapOff = ptr

                if not be.HasPawns then
                    let pe = pieceEntries.[be.EntryIndex]
                    pe.DtzMapOff <- mapOff
                    let flags = pe.DtzFlags

                    if flags &&& 2uy <> 0uy then
                        if flags &&& 16uy = 0uy then
                            for i in 0..3 do
                                pe.DtzMapIdx.[i] <- ptr + 1 - mapOff
                                ptr <- ptr + 1 + int data.[ptr]
                        else
                            ptr <- ptr + (ptr &&& 1)

                            for i in 0..3 do
                                pe.DtzMapIdx.[i] <- (ptr - mapOff) / 2 + 1
                                ptr <- ptr + 2 + 2 * int (readLe16 data ptr)
                else
                    let pe = pawnEntries.[be.EntryIndex]
                    pe.DtzMapOff <- mapOff

                    for t in 0..num-1 do
                        let flags = pe.DtzFlags.[t]

                        if flags &&& 2uy <> 0uy then
                            if flags &&& 16uy = 0uy then
                                for i in 0..3 do
                                    pe.DtzMapIdx.[t, i] <- ptr + 1 - mapOff
                                    ptr <- ptr + 1 + int data.[ptr]
                            else
                                ptr <- ptr + (ptr &&& 1)

                                for i in 0..3 do
                                    pe.DtzMapIdx.[t, i] <- (ptr - mapOff) / 2 + 1
                                    ptr <- ptr + 2 + 2 * int (readLe16 data ptr)

            if ptr &&& 1 <> 0 then ptr <- ptr + 1

            for t in 0..num-1 do
                eis.[eiOff + t].Precomp.IndexTableOff <- ptr
                ptr <- ptr + int sizes.[t, 0, 0]
                if split then
                    eis.[eiOff + num + t].Precomp.IndexTableOff <- ptr
                    ptr <- ptr + int sizes.[t, 1, 0]

            for t in 0..num-1 do
                eis.[eiOff + t].Precomp.SizeTableOff <- ptr
                ptr <- ptr + int sizes.[t, 0, 1]
                if split then
                    eis.[eiOff + num + t].Precomp.SizeTableOff <- ptr
                    ptr <- ptr + int sizes.[t, 1, 1]

            for t in 0..num-1 do
                ptr <- (ptr + 0x3F) &&& ~~~0x3F
                eis.[eiOff + t].Precomp.DataOff <- ptr
                ptr <- ptr + int sizes.[t, 0, 2]
                if split then
                    ptr <- (ptr + 0x3F) &&& ~~~0x3F
                    eis.[eiOff + num + t].Precomp.DataOff <- ptr
                    ptr <- ptr + int sizes.[t, 1, 2]

            Volatile.Write(&be.Ready.[typ], 1)
            true

// ---------------------------------------------------------------------------
// Decompression core
// ---------------------------------------------------------------------------
let private decompressPairs (d: PairsData) (data: byte[]) (idx: uint64) : int =
    if d.IdxBits = 0uy then
        int d.ConstValue.[0]
    else
        let mainIdx = int (idx >>> int d.IdxBits)
        let mutable litIdx = int (idx &&& ((1UL <<< int d.IdxBits) - 1UL)) - (1 <<< (int d.IdxBits - 1))
        let blockOff = d.IndexTableOff + 6 * mainIdx
        let mutable block = int (readLe32 data blockOff)
        let idxOffset = int (readLe16 data (blockOff + 4))
        litIdx <- litIdx + idxOffset

        if litIdx < 0 then
            while litIdx < 0 do
                block <- block - 1
                litIdx <- litIdx + int (readLe16 data (d.SizeTableOff + 2 * block)) + 1
        else
            while litIdx > int (readLe16 data (d.SizeTableOff + 2 * block)) do
                litIdx <- litIdx - int (readLe16 data (d.SizeTableOff + 2 * block)) - 1
                block <- block + 1

        let dataPtr = d.DataOff + (block <<< int d.BlockSize)
        let m = int d.MinLen

        let mutable code =
            (uint64 data.[dataPtr] <<< 56) ||| (uint64 data.[dataPtr+1] <<< 48)
            ||| (uint64 data.[dataPtr+2] <<< 40) ||| (uint64 data.[dataPtr+3] <<< 32)
            ||| (uint64 data.[dataPtr+4] <<< 24) ||| (uint64 data.[dataPtr+5] <<< 16)
            ||| (uint64 data.[dataPtr+6] <<< 8) ||| uint64 data.[dataPtr+7]

        let mutable ptrOff = dataPtr + 8
        let mutable bitCnt = 0
        let mutable sym = 0u
        let mutable found = false

        while not found do
            let mutable l = m
            while code < d.Base.[l - m] do l <- l + 1
            sym <- uint32 (readLe16 data (d.OffsetOff + 2 * (l - m)))
            sym <- sym + uint32 ((code - d.Base.[l - m]) >>> (64 - l))
            if litIdx < int d.SymLen.[int sym] + 1 then
                found <- true
            else
                litIdx <- litIdx - int d.SymLen.[int sym] - 1
                code <- code <<< l
                bitCnt <- bitCnt + l
                if bitCnt >= 32 then
                    bitCnt <- bitCnt - 32
                    let tmp =
                        (uint32 data.[ptrOff] <<< 24)
                        ||| (uint32 data.[ptrOff+1] <<< 16)
                        ||| (uint32 data.[ptrOff+2] <<< 8)
                        ||| uint32 data.[ptrOff+3]
                    ptrOff <- ptrOff + 4
                    code <- code ||| (uint64 tmp <<< bitCnt)

        while d.SymLen.[int sym] <> 0uy do
            let off = d.SymPatOff + 3 * int sym
            let s1 = int ((uint32 (data.[off + 1] &&& 0x0Fuy) <<< 8) ||| uint32 data.[off])
            if litIdx < int d.SymLen.[s1] + 1 then
                sym <- uint32 s1
            else
                litIdx <- litIdx - int d.SymLen.[s1] - 1
                sym <- (uint32 data.[off+2] <<< 4) ||| (uint32 data.[off+1] >>> 4)

        let symOff = d.SymPatOff + 3 * int sym
        int data.[symOff] + ((int (data.[symOff + 1] &&& 0x0Fuy)) <<< 8)

// ---------------------------------------------------------------------------
// Fill squares + probe table
// ---------------------------------------------------------------------------
let private fillSquares (pos: Pos byref) (pieces: byte[]) (flip: bool) (mirror: int) (p: int[]) (i: int) =
    let pc = int pieces.[i]
    let mutable color = colorOfFPiece pc
    if flip then color <- not color
    let mutable bb = piecesOf &pos color (typeOfFPiece pc)
    let mutable idx = i
    while bb <> 0UL do
        let sq = popLsb &bb
        p.[idx] <- sq ^^^ mirror
        idx <- idx + 1
    idx

let private findPieceEntry (be: BaseEntry) =
    pieceEntries.[be.EntryIndex]

let private findPawnEntry (be: BaseEntry) =
    pawnEntries.[be.EntryIndex]

let private getEis (be: BaseEntry) =
    if be.HasPawns then findPawnEntry(be).Ei
    else findPieceEntry(be).Ei

let private getEiOff (be: BaseEntry) (typ: int) =
    if be.HasPawns then (if typ = WDL then 0 elif typ = 1 then 8 else 20)
    else (if typ = WDL then 0 elif typ = 1 then 2 else 4)

let private probeTable (pos: Pos byref) (s: int) (typ: int) : int * bool =
    let key = calcKey &pos false
    if typ = WDL && key = 0UL then 0, true
    else

    let mutable hashIdx = int (key >>> (64 - TbHashBits))
    while tbHash.[hashIdx].Ptr <> null && tbHash.[hashIdx].Key <> key do
        hashIdx <- (hashIdx + 1) &&& ((1 <<< TbHashBits) - 1)

    if tbHash.[hashIdx].Ptr = null || Volatile.Read(&tbHash.[hashIdx].Error) <> 0 then
        0, false
    else
        let be = tbHash.[hashIdx].Ptr
        if typ = DTZ && not be.HasDtz then 0, false
        else
            if Volatile.Read(&be.Ready.[typ]) = 0 then
                let str = prtStr &pos (be.Key <> key)
                lock tbMutex (fun () ->
                    if Volatile.Read(&tbHash.[hashIdx].Error) <> 0 then ()
                    elif Volatile.Read(&be.Ready.[typ]) = 0 then
                        let eis = getEis be
                        let eiOff = getEiOff be typ
                        if not (initTable be str typ eis eiOff) then
                            Volatile.Write(&tbHash.[hashIdx].Error, 1)
                )

            if Volatile.Read(&be.Ready.[typ]) = 0 || Volatile.Read(&tbHash.[hashIdx].Error) <> 0 then
                0, false
            else
                // Fathom: flip = key != be->key; bside = (turn == WHITE) == flip.
                let flip, bside =
                    if not be.Symmetric then
                        let f = key <> be.Key
                        f, (pos.Turn = f)
                    else
                        not pos.Turn, false

                let eis = getEis be
                let eiOff = getEiOff be typ
                let p = Array.zeroCreate TbPieces
                let mutable t = 0

                if not be.HasPawns then
                    let flags =
                        if typ = DTZ then findPieceEntry(be).DtzFlags
                        else 0uy

                    if typ = DTZ then
                        if int (flags &&& 1uy) <> (if bside then 1 else 0) && not be.Symmetric then
                            0, false
                        else
                            let ei = eis.[eiOff]
                            let mutable i2 = 0
                            while i2 < int be.Num do
                                i2 <- fillSquares &pos ei.Pieces flip 0 p i2
                            let idx2 = encode p ei be PieceEnc
                            let v = decompressPairs ei.Precomp be.Data.[typ] idx2
                            let mutable v2 = v

                            if flags &&& 2uy <> 0uy then
                                let pe = findPieceEntry be
                                let m = WdlToMap.[s + 2]

                                if flags &&& 16uy = 0uy then
                                    v2 <- int be.Data.[typ].[pe.DtzMapOff + pe.DtzMapIdx.[m] + v2]
                                else
                                    v2 <- int (readLe16 be.Data.[typ] (pe.DtzMapOff + 2 * (pe.DtzMapIdx.[m] + v2)))

                            if flags &&& PAFlags.[s + 2] = 0uy || (s &&& 1) <> 0 then
                                v2 <- v2 * 2
                            v2, true
                    else
                        let eiIdx = eiOff + (if bside then 1 else 0)
                        let ei = eis.[eiIdx]
                        let mutable i2 = 0
                        while i2 < int be.Num do
                            i2 <- fillSquares &pos ei.Pieces flip 0 p i2
                        let idx2 = encode p ei be PieceEnc
                        let v = decompressPairs ei.Precomp be.Data.[typ] idx2
                        v - 2, true
                else
                    let ei0 = eis.[eiOff]
                    let mutable i2 = fillSquares &pos ei0.Pieces flip (if flip then 0x38 else 0) p 0
                    t <- leadingPawn p be (if typ <> 1 then FileEnc else RankEnc)

                    let flags =
                        if typ = DTZ then findPawnEntry(be).DtzFlags.[t]
                        else 0uy

                    if typ = DTZ then
                        if int (flags &&& 1uy) <> (if bside then 1 else 0) && not be.Symmetric then
                            0, false
                        else
                            let eiIdx = eiOff + t
                            let ei = eis.[eiIdx]
                            while i2 < int be.Num do
                                i2 <- fillSquares &pos ei.Pieces flip (if flip then 0x38 else 0) p i2
                            let idx2 = encode p ei be (if typ <> 1 then FileEnc else RankEnc)
                            let v = decompressPairs ei.Precomp be.Data.[typ] idx2
                            let mutable v2 = v

                            if flags &&& 2uy <> 0uy then
                                let pe = findPawnEntry be
                                let m = WdlToMap.[s + 2]

                                if flags &&& 16uy = 0uy then
                                    v2 <- int be.Data.[typ].[pe.DtzMapOff + pe.DtzMapIdx.[t, m] + v2]
                                else
                                    v2 <- int (readLe16 be.Data.[typ] (pe.DtzMapOff + 2 * (pe.DtzMapIdx.[t, m] + v2)))

                            if flags &&& PAFlags.[s + 2] = 0uy || (s &&& 1) <> 0 then
                                v2 <- v2 * 2
                            v2, true
                    else
                        let off2 =
                            if typ = WDL then eiOff + t + 4 * (if bside then 1 else 0)
                            else eiOff + t
                        let ei = eis.[off2]
                        while i2 < int be.Num do
                            i2 <- fillSquares &pos ei.Pieces flip (if flip then 0x38 else 0) p i2
                        let idx2 = encode p ei be (if typ <> 1 then FileEnc else RankEnc)
                        let v = decompressPairs ei.Precomp be.Data.[typ] idx2
                        v - 2, true

// ---------------------------------------------------------------------------
// WDL + DTZ probing with capture resolution
// ---------------------------------------------------------------------------
let private probeWdlTable (pos: Pos byref) : int * bool =
    probeTable &pos 0 WDL

let private probeDtzTable (pos: Pos byref) (wdl: int) : int * bool =
    probeTable &pos wdl DTZ

let rec private probeAb (pos: Pos byref) (alpha: int) (beta: int) : int * bool =
    let moves = Array.zeroCreate<TbMove> TbMaxCaptures
    let nMoves = genCaptures &pos moves
    let mutable a = alpha
    let mutable success = true
    for i in 0..nMoves-1 do
        if success then
            let m = moves.[i]
            if isCapture &pos m then
                let mutable pos1 = Pos()
                if doMove &pos1 &pos m then
                    let v, ok = probeAb &pos1 (-beta) (-a)
                    let v = -v
                    if not ok then success <- false
                    elif v > a then
                        if v >= beta then a <- v
                        else a <- v
    if not success then 0, false
    else
        let v, ok = probeWdlTable &pos
        if not ok then 0, false
        elif a >= v then a, true
        else v, true

let private probeWdlInternal (pos: Pos byref) : int * bool * int =
    let moves = Array.zeroCreate<TbMove> TbMaxCaptures
    let nMoves = genCaptures &pos moves
    let mutable bestCap = -3
    let mutable bestEp = -3
    let mutable success = true
    let mutable earlyReturn = false
    let mutable successVal = 1

    for i in 0..nMoves-1 do
        if success && not earlyReturn then
            let m = moves.[i]
            if isCapture &pos m then
                let mutable pos1 = Pos()
                if doMove &pos1 &pos m then
                    let v, ok = probeAb &pos1 (-2) (-bestCap)
                    let v = -v
                    if not ok then success <- false
                    elif v > bestCap then
                        if v = 2 then
                            successVal <- 2
                            bestCap <- 2
                            earlyReturn <- true
                        elif not (isEnPassant &pos m) then
                            bestCap <- v
                        elif v > bestEp then
                            bestEp <- v

    if not success then 0, false, 0
    elif earlyReturn then 2, true, 2
    else
        let v, ok = probeWdlTable &pos
        if not ok then 0, false, 0
        else
            let mutable result = v
            let mutable sVal = 1

            if bestEp > bestCap then
                if bestEp > v then
                    sVal <- 2
                    result <- bestEp
                else bestCap <- bestEp

            if bestCap >= v then
                sVal <- 1 + (if bestCap > 0 then 1 else 0)
                result <- bestCap
            elif bestEp > -3 && v = 0 then
                let allMoves = Array.zeroCreate<TbMove> TbMaxMoves
                let nAll = genMoves &pos allMoves
                let mutable foundNonEp = false
                for j in 0..nAll-1 do
                    if not foundNonEp && not (isEnPassant &pos allMoves.[j]) then
                        if legalMove &pos allMoves.[j] then foundNonEp <- true
                if not foundNonEp && not (isCheck &pos) then
                    sVal <- 2
                    result <- bestEp

            result, true, sVal

let rec private probeDtzInternal (pos: Pos byref) : int * bool =
    let wdl, ok, sVal = probeWdlInternal &pos
    if not ok then 0, false
    elif wdl = 0 then 0, true
    elif sVal = 2 then WdlToDtz.[wdl + 2], true
    else
        let moves = Array.zeroCreate<TbMove> TbMaxMoves
        let mutable endIdx = 0
        let mutable pos1 = Pos()

        if wdl > 0 then
            endIdx <- genMoves &pos moves
            let mutable found = false
            for i in 0..endIdx-1 do
                if not found then
                    let m = moves.[i]
                    if typeOfPieceMoved &pos m = FPawn && not (isCapture &pos m) then
                        if doMove &pos1 &pos m then
                            let wdlv, ok2, _ = probeWdlInternal &pos1
                            if ok2 && -wdlv = wdl then found <- true
            if found then WdlToDtz.[wdl + 2], true
            else
                let dtz, ok2 = probeDtzTable &pos wdl
                if ok2 then
                    WdlToDtz.[wdl + 2] + (if wdl > 0 then dtz else -dtz), true
                else
                    let mutable best = Int32.MaxValue
                    let mutable allOk = true
                    for i in 0..endIdx-1 do
                        if allOk then
                            let m = moves.[i]
                            if not (isCapture &pos m) && typeOfPieceMoved &pos m <> FPawn then
                                if doMove &pos1 &pos m then
                                    let v, ok3 = probeDtzInternal &pos1
                                    let v = -v
                                    if not ok3 then allOk <- false
                                    elif v = 1 && isMate &pos1 then best <- 1
                                    elif v > 0 && v + 1 < best then best <- v + 1
                    if not allOk then 0, false
                    else best, true
        else
            let dtz, ok2 = probeDtzTable &pos wdl
            if ok2 then
                WdlToDtz.[wdl + 2] + (if wdl > 0 then dtz else -dtz), true
            else
                endIdx <- genMoves &pos moves
                let mutable best = WdlToDtz.[wdl + 2]
                let mutable allOk = true
                for i in 0..endIdx-1 do
                    if allOk then
                        let m = moves.[i]
                        if not (isCapture &pos m) && typeOfPieceMoved &pos m <> FPawn then
                            if doMove &pos1 &pos m then
                                let v, ok3 = probeDtzInternal &pos1
                                let v = -v
                                if not ok3 then allOk <- false
                                elif v - 1 < best then best <- v - 1
                if not allOk then 0, false
                else best, true

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
let init (path: string) : bool =
    if not initialized then
        initIndices ()
        initialized <- true

    if paths.Length > 0 then
        for i in 0..tbNumPiece-1 do
            for t in 0..2 do
                if pieceEntries.[i].Be.Ready.[t] <> 0 then
                    pieceEntries.[i].Be.Ready.[t] <- 0
        for i in 0..tbNumPawn-1 do
            for t in 0..2 do
                if pawnEntries.[i].Be.Ready.[t] <> 0 then
                    pawnEntries.[i].Be.Ready.[t] <- 0
        numWdl <- 0; numDtz <- 0

    Largest <- 0

    if String.IsNullOrEmpty path || path = "<empty>" then true
    else
        paths <- path.Split(';', StringSplitOptions.RemoveEmptyEntries)
        numPaths <- paths.Length
        tbNumPiece <- 0; tbNumPawn <- 0

        if isNull pieceEntries then
            pieceEntries <- Array.init TbMaxPiece (fun _ -> PieceEntry())
            pawnEntries <- Array.init TbMaxPawn (fun _ -> PawnEntry())

        for i in 0 .. (1 <<< TbHashBits) - 1 do
            tbHash.[i].Key <- 0UL
            tbHash.[i].Ptr <- null

        let pc i = pchr i
        for i in 0..4 do
            initTb ("K" + string (pc i) + "vK")
        for i in 0..4 do
            for j in i..4 do
                initTb ("K" + string (pc i) + "vK" + string (pc j))
        for i in 0..4 do
            for j in i..4 do
                initTb ("K" + string (pc i) + string (pc j) + "vK")
        for i in 0..4 do
            for j in i..4 do
                for k in 0..4 do
                    initTb ("K" + string (pc i) + string (pc j) + "vK" + string (pc k))
        for i in 0..4 do
            for j in i..4 do
                for k in j..4 do
                    initTb ("K" + string (pc i) + string (pc j) + string (pc k) + "vK")

        if IntPtr.Size >= 8 && TbPieces >= 6 then
            for i in 0..4 do
                for j in i..4 do
                    for k in i..4 do
                        for l in (if i = k then j else k) .. 4 do
                            initTb ("K" + string (pc i) + string (pc j) + "vK" + string (pc k) + string (pc l))
            for i in 0..4 do
                for j in i..4 do
                    for k in j..4 do
                        for l in 0..4 do
                            initTb ("K" + string (pc i) + string (pc j) + string (pc k) + "vK" + string (pc l))
            for i in 0..4 do
                for j in i..4 do
                    for k in j..4 do
                        for l in k..4 do
                            initTb ("K" + string (pc i) + string (pc j) + string (pc k) + string (pc l) + "vK")

            if TbPieces >= 7 then
                for i in 0..4 do
                    for j in i..4 do
                        for k in j..4 do
                            for l in k..4 do
                                for m in l..4 do
                                    initTb ("K" + string (pc i) + string (pc j) + string (pc k) + string (pc l) + string (pc m) + "vK")
                for i in 0..4 do
                    for j in i..4 do
                        for k in j..4 do
                            for l in k..4 do
                                for m in 0..4 do
                                    initTb ("K" + string (pc i) + string (pc j) + string (pc k) + string (pc l) + "vK" + string (pc m))
                for i in 0..4 do
                    for j in i..4 do
                        for k in j..4 do
                            for l in 0..4 do
                                for m in l..4 do
                                    initTb ("K" + string (pc i) + string (pc j) + string (pc k) + "vK" + string (pc l) + string (pc m))
        true

let free () = init "" |> ignore

/// Probe WDL during search. Returns -2..+2 or Int32.MinValue on failure.
let probeWDL (pos: Position.Position) : int =
    let mutable p = Pos()
    p.W <- pos.ColorBB(White)
    p.B <- pos.ColorBB(Black)
    p.K <- pos.Pieces(King)
    p.Q <- pos.Pieces(Queen)
    p.R <- pos.Pieces(Rook)
    p.Bi <- pos.Pieces(Bishop)
    p.N <- pos.Pieces(Knight)
    p.P <- pos.Pieces(Pawn)
    p.Rule50 <- 0uy
    p.Ep <- byte (let ep = pos.EpSquare in if ep = 64 then 0 else ep)
    p.Turn <- pos.SideToMove = White
    let v, ok, _ = probeWdlInternal &p
    if ok then v else Int32.MinValue

/// Probe DTZ at root. Returns signed DTZ value or Int32.MinValue on failure.
let probeDTZ (pos: Position.Position) : int =
    let mutable p = Pos()
    p.W <- pos.ColorBB(White)
    p.B <- pos.ColorBB(Black)
    p.K <- pos.Pieces(King)
    p.Q <- pos.Pieces(Queen)
    p.R <- pos.Pieces(Rook)
    p.Bi <- pos.Pieces(Bishop)
    p.N <- pos.Pieces(Knight)
    p.P <- pos.Pieces(Pawn)
    p.Rule50 <- byte pos.Rule50
    p.Ep <- byte (let ep = pos.EpSquare in if ep = 64 then 0 else ep)
    p.Turn <- pos.SideToMove = White
    let v, ok = probeDtzInternal &p
    if ok then v else Int32.MinValue

// ---------------------------------------------------------------------------
// Root probe: DTZ-aware root move filtering (Fathom tb_probe_root's shape on engine types).
// ---------------------------------------------------------------------------

/// Score the position AFTER a candidate root move, mover-relative, on Fathom's DTZ basis
/// (positive = the mover wins; magnitude = plies-to-zeroing + 1; the ±101 band = cursed/blessed).
/// `pos` has the opponent to move, so the raw probe is negated. Int32.MinValue = probe failure.
let private rootMoveDtz (pos: Position.Position) : int =
    if popCount pos.Occupied = 2 then
        0 // bare kings: no table exists, trivially drawn
    elif pos.Rule50 = 0 then
        // The move zeroed the counter: the child WDL is rule-50-exact; map to the DTZ basis.
        let w = probeWDL pos
        if w = Int32.MinValue then Int32.MinValue else WdlToDtz.[(-w) + 2]
    else
        let d = probeDTZ pos

        if d = Int32.MinValue then
            Int32.MinValue
        else
            // The move consumed a ply toward the rule-50 horizon: push wins/losses out by one.
            let v = -d
            if v > 0 then v + 1
            elif v < 0 then v - 1
            else 0

/// DTZ-aware root move filter (the standard TB root technique). Returns the subset of legal root
/// moves preserving the best rule-50-aware outcome:
///   - a rule-50-SAFE win exists  => only the minimal-DTZ safe wins (DTZ strictly falls or the
///     move zeroes, so conversion provably progresses — no TB-win shuffling);
///   - only cursed/unsafe wins    => every winning move (best effort; rule 50 may still save the
///     defender, but a win-class move never becomes worse than the alternatives);
///   - best is a draw             => every drawing move (blessed losses count as draws);
///   - everything loses           => EMPTY: no restriction — max-resistance DTZ filtering throws
///     away practical swindles, so the search's judgement stands.
/// Empty on any gate miss (no tables, too many pieces, live castling rights) or probe failure —
/// callers treat empty as "no restriction". `pos` is mutated via balanced Make/Unmake pairs.
let probeRoot (pos: Position.Position) : Move.Move[] =
    if Largest = 0 || popCount pos.Occupied > Largest || pos.CastlingRights <> 0 then
        [||]
    else
        let buf = Array.zeroCreate<Move.Move> MoveGeneration.MaxMoves
        let n = MoveGeneration.generateLegal pos (Span<Move.Move>(buf))
        let childBuf = Array.zeroCreate<Move.Move> MoveGeneration.MaxMoves
        let scores = Array.zeroCreate<int> (max 1 n)
        let rule50 = pos.Rule50
        let mutable ok = n > 0
        let mutable i = 0

        while ok && i < n do
            pos.Make buf.[i]

            let v =
                if MoveGeneration.generateLegal pos (Span<Move.Move>(childBuf)) = 0 then
                    // Terminal child: mate = a win at the minimal basis; stalemate = draw.
                    (if pos.InCheck then 1 else 0)
                elif pos.Rule50 >= 100 then
                    0 // the move ran into the 50-move rule with legal replies left: draw
                else
                    rootMoveDtz pos

            pos.Unmake buf.[i]

            if v = Int32.MinValue then ok <- false else scores.[i] <- v
            i <- i + 1

        if not ok then
            [||]
        else
            // Outcome class under the ROOT's rule-50 counter: 4 = safe win, 3 = cursed/unsafe win,
            // 2 = draw (including a loss pushed past the rule-50 horizon), 1 = loss.
            let classOf (v: int) =
                if v > 0 && v + rule50 <= 100 then 4
                elif v > 0 then 3
                elif v = 0 || v < rule50 - 100 then 2
                else 1

            let mutable bestClass = 0

            for k in 0 .. n - 1 do
                bestClass <- max bestClass (classOf scores.[k])

            if bestClass <= 1 then
                [||]
            else
                let keep = ResizeArray<Move.Move>(n)

                if bestClass = 4 then
                    let mutable minV = Int32.MaxValue

                    for k in 0 .. n - 1 do
                        if classOf scores.[k] = 4 then
                            minV <- min minV scores.[k]

                    for k in 0 .. n - 1 do
                        if classOf scores.[k] = 4 && scores.[k] = minV then
                            keep.Add buf.[k]
                else
                    for k in 0 .. n - 1 do
                        if classOf scores.[k] = bestClass then
                            keep.Add buf.[k]

                keep.ToArray()
