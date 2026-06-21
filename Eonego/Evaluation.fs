/// PeSTO-style tapered static evaluation: material + piece-square tables only, MG/EG interpolated by
/// game phase. The canonical minimal eval (published exact reference tables -> every value is pinnable).
///
/// THREAD-SAFETY: no post-init reachable writable state. After the one-time `do initTables ()` the Mg/Eg
/// arrays are only ever read, so any number of search threads may call `eval` concurrently on DISTINCT
/// Position instances with zero shared writable state (LazySMP / lockless by construction). Calling `eval`
/// on a Position another thread is mutating is NOT safe — but each LazySMP thread owns its Position.
///
/// `eval` is 0 B/op and purely static: checkmate / stalemate / repetition / 50-move / insufficient material
/// are the SEARCH's concern, never eval's.
///
/// v1 recomputes from scratch. The incremental accumulator is deferred; the EVAL-HOOK comment lines in
/// Position.fs (PutPiece/RemovePiece/MovePiece) mark where a future per-node accumulator would subscribe —
/// it must also snapshot into StateInfo and restore on unmake (the *NK helpers skip the hooks), exactly
/// like the incremental Zobrist key.
module Eonego.Evaluation

open System.Diagnostics // Debug.Assert (compiled out in Release/AOT)
open System.Runtime.CompilerServices
open Eonego.Bitboard
open Eonego.Position

// ---------------------------------------------------------------------------
// PeSTO constants (centipawns).
//
// Material is INTENTIONALLY distinct from Position.pieceValueOf {100,320,330,500,900,0}: that table is
// SEE / MVV-LVA *ordering granularity*; these are *positional-accuracy* values. Stockfish keeps the two
// separate for the same reason. Do NOT unify them. King material = 0 (king still gets full MG/EG PST).
// ---------------------------------------------------------------------------
let private mgValue = [| 82; 337; 365; 477; 1025; 0 |] // P N B R Q K
let private egValue = [| 94; 281; 297; 512; 936; 0 |] // P N B R Q K
let private phaseInc = [| 0; 1; 1; 2; 4; 0 |] // PeSTO game-phase weights, by PieceType

[<Literal>]
let PhaseMax = 24

// ---------------------------------------------------------------------------
// Raw published PeSTO piece-square tables (RofChade / Chess Programming Wiki "PeSTO's Evaluation
// Function"). Written RANK-8-FIRST: visual index 0 = a8, 8 = a7, ..., 56 = a1, 63 = h1.
// LERF square s maps to this visual index via (s ^^^ 56). (Trailing `;` per row: a row ending in a value
// followed by a line starting with a negative number would otherwise parse as function application.)
// ---------------------------------------------------------------------------
let private mgPawn =
    [| 0
       0
       0
       0
       0
       0
       0
       0
       98
       134
       61
       95
       68
       126
       34
       -11
       -6
       7
       26
       31
       65
       56
       25
       -20
       -14
       13
       6
       21
       23
       12
       17
       -23
       -27
       -2
       -5
       12
       17
       6
       10
       -25
       -26
       -4
       -4
       -10
       3
       3
       33
       -12
       -35
       -1
       -20
       -23
       -15
       24
       38
       -22
       0
       0
       0
       0
       0
       0
       0
       0 |]

let private egPawn =
    [| 0
       0
       0
       0
       0
       0
       0
       0
       178
       173
       158
       134
       147
       132
       165
       187
       94
       100
       85
       67
       56
       53
       82
       84
       32
       24
       13
       5
       -2
       4
       17
       17
       13
       9
       -3
       -7
       -7
       -8
       3
       -1
       4
       7
       -6
       1
       0
       -5
       -1
       -8
       13
       8
       8
       10
       13
       0
       2
       -7
       0
       0
       0
       0
       0
       0
       0
       0 |]

let private mgKnight =
    [| -167
       -89
       -34
       -49
       61
       -97
       -15
       -107
       -73
       -41
       72
       36
       23
       62
       7
       -17
       -47
       60
       37
       65
       84
       129
       73
       44
       -9
       17
       19
       53
       37
       69
       18
       22
       -13
       4
       16
       13
       28
       19
       21
       -8
       -23
       -9
       12
       10
       19
       17
       25
       -16
       -29
       -53
       -12
       -3
       -1
       18
       -14
       -19
       -105
       -21
       -58
       -33
       -17
       -28
       -19
       -23 |]

let private egKnight =
    [| -58
       -38
       -13
       -28
       -31
       -27
       -63
       -99
       -25
       -8
       -25
       -2
       -9
       -25
       -24
       -52
       -24
       -20
       10
       9
       -1
       -9
       -19
       -41
       -17
       3
       22
       22
       22
       11
       8
       -18
       -18
       -6
       16
       25
       16
       17
       4
       -18
       -23
       -3
       -1
       15
       10
       -3
       -20
       -22
       -42
       -20
       -10
       -5
       -2
       -20
       -23
       -44
       -29
       -51
       -23
       -15
       -22
       -18
       -50
       -64 |]

let private mgBishop =
    [| -29
       4
       -82
       -37
       -25
       -42
       7
       -8
       -26
       16
       -18
       -13
       30
       59
       18
       -47
       -16
       37
       43
       40
       35
       50
       37
       -2
       -4
       5
       19
       50
       37
       37
       7
       -2
       -6
       13
       13
       26
       34
       12
       10
       4
       0
       15
       15
       15
       14
       27
       18
       10
       4
       15
       16
       0
       7
       21
       33
       1
       -33
       -3
       -14
       -21
       -13
       -12
       -39
       -21 |]

let private egBishop =
    [| -14
       -21
       -11
       -8
       -7
       -9
       -17
       -24
       -8
       -4
       7
       -12
       -3
       -13
       -4
       -14
       2
       -8
       0
       -1
       -2
       6
       0
       4
       -3
       9
       12
       9
       14
       10
       3
       2
       -6
       3
       13
       19
       7
       10
       -3
       -9
       -12
       -3
       8
       10
       13
       3
       -7
       -15
       -14
       -18
       -7
       -1
       4
       -9
       -15
       -27
       -23
       -9
       -23
       -5
       -9
       -16
       -5
       -17 |]

let private mgRook =
    [| 32
       42
       32
       51
       63
       9
       31
       43
       27
       32
       58
       62
       80
       67
       26
       44
       -5
       19
       26
       36
       17
       45
       61
       16
       -24
       -11
       7
       26
       24
       35
       -8
       -20
       -36
       -26
       -12
       -1
       9
       -7
       6
       -23
       -45
       -25
       -16
       -17
       3
       0
       -5
       -33
       -44
       -16
       -20
       -9
       -1
       11
       -6
       -71
       -19
       -13
       1
       17
       16
       7
       -37
       -26 |]

let private egRook =
    [| 13
       10
       18
       15
       12
       12
       8
       5
       11
       13
       13
       11
       -3
       3
       8
       3
       7
       7
       7
       5
       4
       -3
       -5
       -3
       4
       3
       13
       1
       2
       1
       -1
       2
       3
       5
       8
       4
       -5
       -6
       -8
       -11
       -4
       0
       -5
       -1
       -7
       -12
       -8
       -16
       -6
       -6
       0
       2
       -9
       -9
       -11
       -3
       -9
       2
       3
       -1
       -5
       -13
       4
       -20 |]

let private mgQueen =
    [| -28
       0
       29
       12
       59
       44
       43
       45
       -24
       -39
       -5
       1
       -16
       57
       28
       54
       -13
       -17
       7
       8
       29
       56
       47
       57
       -27
       -27
       -16
       -16
       -1
       17
       -2
       1
       -9
       -26
       -9
       -10
       -2
       -4
       3
       -3
       -14
       2
       -11
       -2
       -5
       2
       14
       5
       -35
       -8
       11
       2
       8
       15
       -3
       1
       -1
       -18
       -9
       10
       -15
       -25
       -31
       -50 |]

let private egQueen =
    [| -9
       22
       22
       27
       27
       19
       10
       20
       -17
       20
       32
       41
       58
       25
       30
       0
       -20
       6
       9
       49
       47
       35
       19
       9
       3
       22
       24
       45
       57
       40
       57
       36
       -18
       28
       19
       47
       31
       34
       39
       23
       -16
       -27
       15
       6
       9
       17
       10
       5
       -22
       -23
       -30
       -16
       -16
       -23
       -36
       -32
       -33
       -28
       -22
       -43
       -5
       -32
       -20
       -41 |]

let private mgKing =
    [| -65
       23
       16
       -15
       -56
       -34
       2
       13
       29
       -1
       -20
       -7
       -8
       -4
       -38
       -29
       -9
       24
       2
       -16
       -20
       6
       22
       -22
       -17
       -20
       -12
       -27
       -30
       -25
       -14
       -36
       -49
       -1
       -27
       -39
       -46
       -44
       -33
       -51
       -14
       -14
       -22
       -46
       -44
       -30
       -15
       -27
       1
       7
       -8
       -64
       -43
       -16
       9
       8
       -15
       36
       12
       -54
       8
       -28
       24
       14 |]

let private egKing =
    [| -74
       -35
       -18
       -18
       -11
       15
       4
       -17
       -12
       17
       14
       17
       17
       38
       23
       11
       10
       17
       23
       15
       20
       45
       44
       13
       -8
       22
       24
       27
       26
       33
       26
       3
       -18
       -4
       21
       24
       27
       23
       9
       -11
       -19
       -3
       11
       21
       23
       16
       7
       -9
       -27
       -11
       4
       13
       14
       4
       -5
       -17
       -53
       -34
       -21
       -11
       -28
       -14
       -24
       -43 |]

// Indexed by PieceType (Pawn..King).
let private rawMg: int[][] =
    [| mgPawn; mgKnight; mgBishop; mgRook; mgQueen; mgKing |]

let private rawEg: int[][] =
    [| egPawn; egKnight; egBishop; egRook; egQueen; egKing |]

// ---------------------------------------------------------------------------
// Combined material + PST, flat, indexed (pc <<< 6) + sq with pc = color*6 + pieceType (0..11), exactly
// like Zobrist's Psq. White at LERF square s reads raw[s ^^^ 56]; Black mirrors and reads raw[s]. This
// reproduces PeSTO exactly and yields the structural mirror identity Mg_W(pt,s) == Mg_B(pt, s ^^^ 56).
// ---------------------------------------------------------------------------
let private Mg: int[] = Array.zeroCreate (12 * 64)
let private Eg: int[] = Array.zeroCreate (12 * 64)

let private initTables () =
    for pt in 0..5 do
        for s in 0..63 do
            Mg.[(pt <<< 6) + s] <- mgValue.[pt] + rawMg.[pt].[s ^^^ 56]
            Eg.[(pt <<< 6) + s] <- egValue.[pt] + rawEg.[pt].[s ^^^ 56]
            Mg.[((6 + pt) <<< 6) + s] <- mgValue.[pt] + rawMg.[pt].[s]
            Eg.[((6 + pt) <<< 6) + s] <- egValue.[pt] + rawEg.[pt].[s]

do initTables ()

// Public future-facing accessors over the private flat tables. Assert like Zobrist.zPiece so a bad
// search/pruning caller is caught early (never pass NoPiece=12 -> would index past 768). Asserts vanish
// in Release/AOT. ATTRIBUTE inlining (not F# `inline`): a source-inlined read of a private binding fails
// at cross-assembly callers; the attribute compiles a normal method the JIT still inlines in-assembly.
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let mgScoreOf (pc: Piece) (sq: Square) : int =
    Debug.Assert((pc >= 0 && pc < NoPiece && sq >= 0 && sq < 64), "mgScoreOf: bad pc/sq")
    Mg.[(pc <<< 6) + sq]

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let egScoreOf (pc: Piece) (sq: Square) : int =
    Debug.Assert((pc >= 0 && pc < NoPiece && sq >= 0 && sq < 64), "egScoreOf: bad pc/sq")
    Eg.[(pc <<< 6) + sq]

// ---------------------------------------------------------------------------
// Phase (taper).
//   - cap min(24, phase) BEFORE the multiply (over-promotion can push phase > 24)
//   - divide ONCE on the single signed white-relative numerator; .NET int `/` truncates toward zero, so
//     (-X)/24 == -(X/24) exactly -> score(mirror) == -score holds to the centipawn.
// ---------------------------------------------------------------------------
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let private taper (mg: int) (eg: int) (phase: int) : int =
    let p = if phase > PhaseMax then PhaseMax else phase
    (mg * p + eg * (PhaseMax - p)) / PhaseMax

// ---------------------------------------------------------------------------
// Accumulation — white-relative (mg, eg, phase). struct tuple => stack only, 0 heap alloc.
// AggressiveInlining for consistency with the Zobrist/Bitboard accessor idiom.
// ---------------------------------------------------------------------------
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let private accumulate (pos: Position) : struct (int * int * int) =
    let mutable mg = 0
    let mutable eg = 0
    let mutable phase = 0

    for pt in 0..5 do
        let inc = phaseInc.[pt]
        let wpc = makePiece White pt // = pt
        let bpc = makePiece Black pt // = 6 + pt
        let mutable w = pos.PiecesCT White pt

        while w <> 0UL do
            let s = popLsb &w
            mg <- mg + mgScoreOf wpc s
            eg <- eg + egScoreOf wpc s
            phase <- phase + inc

        let mutable b = pos.PiecesCT Black pt

        while b <> 0UL do
            let s = popLsb &b
            mg <- mg - mgScoreOf bpc s
            eg <- eg - egScoreOf bpc s
            phase <- phase + inc

    struct (mg, eg, phase)

/// White-relative (mg, eg, phase). Diagnostic / test hook; 0 B/op, not on the hot path.
let evalTrace (pos: Position) : struct (int * int * int) = accumulate pos

/// Static evaluation in centipawns, from the side-to-move's perspective (negamax). 0 B/op.
/// No tempo term: the symmetry contract is eval(p) == -eval(mirror p) with the mirror keeping the side to
/// move, which a constant tempo would break.
let eval (pos: Position) : int =
    let struct (mg, eg, phase) = accumulate pos
    let score = taper mg eg phase // white-relative
    if pos.SideToMove = White then score else -score // negamax sign — load-bearing
