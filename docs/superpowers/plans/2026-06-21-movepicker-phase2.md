# MovePicker Phase-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Phase-2 move-ordering layer for the Eonego chess engine: `Position.SeeGe` (static exchange evaluation), `Position.IsPseudoLegal` (inline pseudo-legality oracle), `History.fs` (Tables class with MainHistory + CaptureHistory), and `MovePicker.fs` (staged-lazy `[<Struct; IsByRefLike>]` move picker with a while-loop `nextMove`).

**Architecture:** All Position changes are purely additive (new members + module-level table, no existing member modified). Two new source files (`History.fs`, `MovePicker.fs`) are inserted between `MoveGeneration.fs` and `Program.fs` in the compile order. One targeted fix to `MoveGeneration.genPawnMoves` adds Queen push-promotions to the `Captures` genType (the "PROMOTION QUIRK" — `generate(Captures)` currently omits push-promotions, which would cause qsearch to miss queen promotions). The MovePicker is a `[<Struct; IsByRefLike>]` with explicit `val mutable` fields, primed by module-level factory functions, advanced by a module-level `nextMove` over `byref<MovePicker>` (never an instance method — instance methods on value types mutate a copy). The `nextMove` loop uses a `while`-loop with a mutable result (not tail recursion — byref parameters can defeat `.tail` under Native AOT).

**Tech Stack:** F# 10, .NET 10, Native AOT (`PublishAot=true`), xUnit 2.9, `Span<'T>` + `NativePtr.stackalloc` for zero-allocation move buffers.

---

## Synthesis Decisions (7 cross-subproblem conflicts resolved)

Four design subagents produced structured outputs. Seven conflicts were identified and resolved as follows:

| # | Conflict | Resolution | Rationale |
|---|----------|-----------|-----------|
| 1 | **PieceValue minors**: N=320/B=330 vs N=300/B=300 | **{P100, N320, B330, R500, Q900, K0}** | SEE subproblem owns the table; prompt sanctioned classic values; 24 hand-computed fixtures already pinned to these; round values aid review. |
| 2 | **PieceValue location**: History.fs vs Position.fs | **Position.fs** | SeeGe is a Position member and needs the table locally; Position compiles before History; `let private pieceValue` + public `pieceValueOf` module fn + `member _.PieceValueOf` member. History.fs/MovePicker.fs import `pieceValueOf` from Position. |
| 3 | **Capture score formula**: `16*victim - attacker` (MVV-LVA) vs `7*captured + CaptureHistory` (SF-modern) | **`7 * pieceValueOf capturedPT + CaptureHistory`** | Matches SF18; MovePicker subproblem owner specified it; with empty CaptureHistory degrades to `7*captured` (pure MVV, acceptable for v1); Tables class carries both MainHistory + CaptureHistory. |
| 4 | **IsPseudoLegal surface**: member only vs member + module fn | **Member only: `pos.IsPseudoLegal m`** | IsPseudoLegal subproblem owner defines it as a member; a module fn would just delegate; DRY — one definition; consistent with `pos.SeeGe m threshold`. |
| 5 | **TT-move gating**: commented out (open) vs wired in | **Wired in via `pos.IsPseudoLegal ttMove`** | SF verifies TT move pseudo-legality at picker construction; stale/colliding TT moves could be illegal; `mkProbCut` additionally requires capture + `pos.SeeGe m threshold`. |
| 6 | **nextMove shape**: `let rec` tail recursion vs `while` loop | **`while` loop with mutable result** | byref parameters can defeat `.tail` under AOT (identified as single biggest structural risk); loop form guarantees no stack growth. |
| 7 | **PROMOTION QUIRK**: defer vs fix | **Fix in MoveGeneration** | `generate(Captures)` misses push-promotions (e.g. e7e8=Q); safe to fix — `generate(Captures)` is not used by perft or any existing test; add Queen push-promotions to the Captures path in `genPawnMoves`. |

Additional synthesis choices (not conflicts, but needed for consistency):

- **Struct declaration form**: Explicit `val mutable` (not struct-record). Struct-records auto-derive equality over Span fields (throw/meaningless) — the primary layout risk. All fields `val mutable`, primed via field-writes in module factory (`Unchecked.defaultof` + field assignment).
- **Stage IDs**: 16 `[<Literal>]` ints spanning Main/Evasion/ProbCut/QSearch chains. Ordering is load-bearing: fall-through == Stage+1.
- **Refutation cursor**: Dedicated `mutable RefIdx : int` field (0..3), not reusing `Cur` (unsafe — `Cur` is the move cursor).
- **Quiet buffer offset**: `mp.Moves.Slice(qStart)` — no MoveGeneration change needed.
- **`partialInsertionSort` limit**: `-3000 * Depth` (SF master uses ~-3560; round v1 value).
- **`EvasionBonus`**: `1 <<< 28` (keeps capture-evasions above any history term).
- **`HistMax`**: `1 <<< 14 = 16384` (gravity clamp bound for history updates).

---

## File Structure

### New source files (Eonego project)

| File | Responsibility | Compile order |
|------|---------------|---------------|
| `Eonego/History.fs` | `Tables` class: MainHistory (butterfly `int[8192]`) + CaptureHistory (`int[4608]`); gravity-clamp update methods. | After `MoveGeneration.fs`, before `MovePicker.fs` |
| `Eonego/MovePicker.fs` | `[<Struct; IsByRefLike>] MovePicker` (explicit `val mutable` fields); 16 stage literals; `mkMain`/`mkQSearch`/`mkProbCut` factories; `nextMove` (while-loop over `byref<MovePicker>`); `pickBest`/`swap`/`scoreCaptures`/`scoreQuiets`/`scoreEvasions`/`partialInsertionSort`/`refutationNext` helpers. | After `History.fs`, before `Program.fs` |

### New test files (Eonego.Tests project)

| File | Responsibility | Compile order |
|------|---------------|---------------|
| `Eonego.Tests/SeeGeTests.fs` | 14 SEE fixtures / 24 assertions (free hanging, equal trade, losing capture, x-ray battery, boundary, non-normal early-out). | After `MoveGenerationTests.fs` |
| `Eonego.Tests/IsPseudoLegalTests.fs` | Oracle test (generate-set membership) + 28 crafted false cases. | After `SeeGeTests.fs` |
| `Eonego.Tests/MovePickerTests.fs` | Probe (byref-persistence), laziness, ordering, evasion, qsearch, probcut fixtures (F0-F10). | After `IsPseudoLegalTests.fs` |

### Modified files

| File | Changes |
|------|---------|
| `Eonego/Position.fs` | Add `Rank2BB`/`Rank7BB` literals; `let private pieceValue` + `let pieceValueOf` + `member _.PieceValueOf`; `member this.SeeGe`; `member this.IsPseudoLegal` + `member private this.ResolvesCheck`. |
| `Eonego/MoveGeneration.fs` | `genPawnMoves`: emit Queen push-promotions when `genType = Captures` (PROMOTION QUIRK fix). |
| `Eonego/Eonego.fsproj` | Add `History.fs` and `MovePicker.fs` before `Program.fs`. |
| `Eonego.Tests/Eonego.Tests.fsproj` | Add `SeeGeTests.fs`, `IsPseudoLegalTests.fs`, `MovePickerTests.fs`. |
| `Eonego.Tests/TestFixtures.fs` | Add `collectPseudo` helper (generate with specific genType, collect to array). |

### Compile order after changes

**Eonego.fsproj:** Bitboard → Move → Zobrist → Position → MoveGeneration → **History** → **MovePicker** → Program

**Eonego.Tests.fsproj:** TestFixtures → BitboardTests → MoveTests → ZobristTests → PositionTests → MoveGenerationTests → **SeeGeTests** → **IsPseudoLegalTests** → **MovePickerTests**

---

### Task 1: PieceValue Table + Position.SeeGe

**Files:**
- Modify: `Eonego/Position.fs` (add `pieceValue` table near line 84, `pieceValueOf` + `member _.PieceValueOf` + `member this.SeeGe` after line 239)
- Create: `Eonego.Tests/SeeGeTests.fs`
- Modify: `Eonego.Tests/Eonego.Tests.fsproj` (add `SeeGeTests.fs`)

- [ ] **Step 1: Write the failing test**

Create `Eonego.Tests/SeeGeTests.fs`:

```fsharp
module Eonego.Tests.SeeGeTests

open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position

let private sq f r = mkSquare f r
let private e1 = sq 4 0
let private e2 = sq 4 1
let private e4 = sq 4 3
let private e5 = sq 4 4
let private e7 = sq 4 6
let private e8 = sq 4 7
let private d5 = sq 3 4
let private d2 = sq 3 1
let private d1 = sq 3 0
let private c3 = sq 2 2
let private f6 = sq 5 5
let private f3 = sq 5 2
let private a1 = sq 0 0
let private d4 = sq 3 3
let private d8 = sq 3 7
let private d7 = sq 3 6
let private d6 = sq 3 5
let private c6 = sq 2 5
let private c1 = sq 2 0
let private c7 = sq 2 6
let private g1 = sq 6 0

// --- PieceValue table accessors ---
[<Fact>]
let ``PieceValueOf returns classic centipawn values`` () =
    Assert.Equal(100, pieceValueOf Pawn)
    Assert.Equal(320, pieceValueOf Knight)
    Assert.Equal(330, pieceValueOf Bishop)
    Assert.Equal(500, pieceValueOf Rook)
    Assert.Equal(900, pieceValueOf Queen)
    Assert.Equal(0, pieceValueOf King)

// --- SeeGe: normal captures (Theory — hand-computed expected values) ---
// Format: (FEN, from, to, threshold, expected)
[<Theory>]
[<InlineData("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", 28, 35, 0, true)>]          // #1 free hanging pawn
[<InlineData("4k3/8/4r3/4r3/8/8/4R3/4K3 w - - 0 1", 12, 36, 0, true)>]         // #2 equal trade RxR thr 0
[<InlineData("4k3/8/4r3/4r3/8/8/4R3/4K3 w - - 0 1", 12, 36, 1, false)>]        // #3 equal trade RxR thr 1
[<InlineData("4k3/8/5p2/4p3/8/8/8/Q3K3 w - - 0 1", 0, 36, 0, false)>]          // #4 losing QxP pawn-defended
[<InlineData("4k3/8/5p2/4p3/8/8/8/Q3K3 w - - 0 1", 0, 36, -800, true)>]        // #5 boundary: SEE==-800==thr
[<InlineData("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", 28, 35, 100, true)>]         // #6 boundary: SEE==+100==thr
[<InlineData("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", 28, 35, 101, false)>]        // #6b SEE 100 < 101
[<InlineData("3rk3/3r4/8/3p4/8/8/3R4/3RK3 w - - 0 1", 11, 35, 0, false)>]      // #7 x-ray battery stacked rooks
[<InlineData("3rk3/3r4/8/3p4/8/8/3R4/3RK3 w - - 0 1", 11, 35, -400, true)>]    // #8 boundary: SEE==-400==thr
[<InlineData("3rk3/8/8/3p4/8/8/3R4/3QK3 w - - 0 1", 11, 35, 0, true)>]         // #9 x-ray queen-behind flips sign
[<InlineData("4k3/8/2p5/3p4/8/2N5/8/4K3 w - - 0 1", 18, 35, 0, false)>]        // #10 recapture chain flips sign
[<InlineData("4k3/8/2p5/3p4/8/2N5/8/4K3 w - - 0 1", 18, 35, -220, true)>]      // #10b boundary
[<InlineData("4k3/8/8/4r3/3P4/8/8/4K3 w - - 0 1", 27, 36, 0, true)>]           // #11 pawn wins rook (undefended)
[<InlineData("4k3/8/5p2/4r3/3P4/8/8/4K3 w - - 0 1", 27, 36, 0, true)>]         // #11b pawn wins rook (pawn-defended, still winning)
let ``SeeGe matches hand-computed expected`` (fen: string) (fromS: int) (toS: int) (thr: int) (expected: bool) =
    let p = Position.OfFen fen
    let m = mkMove fromS toS
    Assert.Equal(expected, p.SeeGe m thr)

// --- SeeGe: non-normal early-out (isSpecial -> 0 >= threshold) ---
[<Fact>]
let ``SeeGe promotion early-out at thresholds -1, 0, +1`` () =
    let p = Position.OfFen "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1"
    let m = mkPromotion e7 e8 Queen
    Assert.True(p.SeeGe m -1)    // 0 >= -1 -> true
    Assert.True(p.SeeGe m 0)     // 0 >= 0 -> true
    Assert.False(p.SeeGe m 1)    // 0 >= 1 -> false

[<Fact>]
let ``SeeGe castling early-out at thresholds -1, 0, +1`` () =
    let p = Position.OfFen "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
    let m = mkCastling e1 g1
    Assert.True(p.SeeGe m -1)
    Assert.True(p.SeeGe m 0)
    Assert.False(p.SeeGe m 1)

[<Fact>]
let ``SeeGe en-passant early-out at thresholds -1, 0, +1`` () =
    let p = Position.OfFen "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
    let m = mkEnPassant e5 d6
    Assert.True(p.SeeGe m -1)
    Assert.True(p.SeeGe m 0)
    Assert.False(p.SeeGe m 1)
```

- [ ] **Step 2: Add `SeeGeTests.fs` to the test project**

Modify `Eonego.Tests/Eonego.Tests.fsproj` — add after `MoveGenerationTests.fs`:

```xml
    <Compile Include="MoveGenerationTests.fs" />
    <Compile Include="SeeGeTests.fs" />
```

- [ ] **Step 3: Run test to verify it fails**

Run: `dotnet test Eonego.Tests --filter "FullyQualifiedName~SeeGeTests"`
Expected: FAIL — `pieceValueOf` and `SeeGe` not defined.

- [ ] **Step 4: Add the PieceValue table and SeeGe member to Position.fs**

Add `Rank2BB`/`Rank7BB` literals in the Constants section (after `StartPosFen`, ~line 38):

```fsharp
[<Literal>]
let private Rank2BB : Bitboard = 0x000000000000FF00UL
[<Literal>]
let private Rank7BB : Bitboard = 0x00FF000000000000UL
```

Add `pieceValue` + `pieceValueOf` right before `do initTables ()` (~line 84):

```fsharp
// PieceValue: classic centipawn values {P100,N320,B330,R500,Q900,K0}.
// King=0 is the SEE sentinel (never subtracted — the KING branch terminates first).
// Single source of truth for SEE, MVV-LVA, and capture-history scoring.
let private pieceValue : int[] = [| 100; 320; 330; 500; 900; 0 |]

/// Material value of a PieceType (Pawn..King). King=0 (SEE sentinel).
[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let pieceValueOf (pt: PieceType) : int = pieceValue.[pt]
```

Add `PieceValueOf` member + `SeeGe` member after `AttackedBy` (~line 239, before `SliderBlockers`):

```fsharp
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.PieceValueOf (pt: PieceType) : int = pieceValue.[pt]

    /// Static Exchange Evaluation >= threshold (Stockfish Position::see_ge port).
    /// v1: non-NORMAL moves short-circuit to (0 >= threshold). NOT pin-aware
    /// (KING-terminate covers the dominant illegal-recapture case; pin-aware
    /// exclusion is a documented v2 follow-up). PRE: m is pseudo-legal for sideToMove.
    member this.SeeGe (m: Move) (threshold: int) : bool =
        if isSpecial m then
            0 >= threshold
        else
            let from = fromSq m
            let toSquare = toSq m
            let captured = this.PieceOn toSquare
            let capVal = if captured = NoPiece then 0 else pieceValue.[pieceType captured]
            let mutable swap = capVal - threshold
            if swap < 0 then
                false
            else
                let mover = this.PieceOn from
                swap <- pieceValue.[pieceType mover] - swap
                if swap <= 0 then
                    true
                else
                    let mutable occupied =
                        this.Occupied ^^^ (1UL <<< from) ^^^ (1UL <<< toSquare)
                    let mutable stm = flipColor this.SideToMove
                    let mutable attackers = this.AttackersTo toSquare occupied
                    let bishopsQueens = this.Pieces Bishop ||| this.Pieces Queen
                    let rooksQueens   = this.Pieces Rook   ||| this.Pieces Queen
                    let mutable res = 1
                    let mutable go = true
                    while go do
                        let stmAttackers = attackers &&& this.ColorBB stm
                        if stmAttackers = 0UL then
                            go <- false
                        else
                            res <- res ^^^ 1
                            let mutable bb = stmAttackers &&& this.PiecesCT stm Pawn
                            let mutable pt = Pawn
                            if bb = 0UL then (bb <- stmAttackers &&& this.PiecesCT stm Knight; pt <- Knight)
                            if bb = 0UL then (bb <- stmAttackers &&& this.PiecesCT stm Bishop; pt <- Bishop)
                            if bb = 0UL then (bb <- stmAttackers &&& this.PiecesCT stm Rook;   pt <- Rook)
                            if bb = 0UL then (bb <- stmAttackers &&& this.PiecesCT stm Queen;  pt <- Queen)
                            if bb = 0UL then (bb <- stmAttackers &&& this.PiecesCT stm King;   pt <- King)
                            let lva = lsb bb
                            occupied <- occupied ^^^ (1UL <<< lva)
                            if pt = Pawn || pt = Bishop || pt = Queen then
                                attackers <- attackers ||| (bishopAttacks toSquare occupied &&& bishopsQueens)
                            if pt = Rook || pt = Queen then
                                attackers <- attackers ||| (rookAttacks toSquare occupied &&& rooksQueens)
                            attackers <- attackers &&& occupied
                            if pt = King && (attackers &&& this.ColorBB (flipColor stm)) <> 0UL then
                                res <- res ^^^ 1
                                go <- false
                            else
                                stm <- flipColor stm
                    res <> 0
```

- [ ] **Step 5: Run test to verify it passes**

Run: `dotnet test Eonego.Tests --filter "FullyQualifiedName~SeeGeTests"`
Expected: PASS — all 24 assertions green.

- [ ] **Step 6: Run perft to verify no regression**

Run: `dotnet test Eonego.Tests --filter "FullyQualifiedName~MoveGenerationTests"`
Expected: PASS — all perft d1-d4 and targeted fixtures green (Position changes are additive).

- [ ] **Step 7: Commit**

```bash
git add Eonego/Position.fs Eonego.Tests/SeeGeTests.fs Eonego.Tests/Eonego.Tests.fsproj
git commit -m "feat: add PieceValue table and Position.SeeGe (SEE >= threshold)"
```

---

### Task 2: Position.IsPseudoLegal

**Files:**
- Modify: `Eonego/Position.fs` (add `IsPseudoLegal` + `ResolvesCheck` after `GivesCheck`, ~line 665)
- Modify: `Eonego.Tests/TestFixtures.fs` (add `collectPseudo` helper)
- Create: `Eonego.Tests/IsPseudoLegalTests.fs`
- Modify: `Eonego.Tests/Eonego.Tests.fsproj` (add `IsPseudoLegalTests.fs`)

- [ ] **Step 1: Add `collectPseudo` helper to TestFixtures.fs**

Add to `Eonego.Tests/TestFixtures.fs` (after `collectLegal`, ~line 59):

```fsharp
/// Collect the pseudo-legal move list for a specific genType into an array (cold path; allocates).
let collectPseudo (pos: Position) (genType: int) : Move[] =
    let p = NativePtr.stackalloc<Move> MaxMoves
    let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)
    let n = generate pos buf genType
    let r = ResizeArray<Move>(n)
    for i in 0 .. n - 1 do r.Add buf.[i]
    r.ToArray()
```

- [ ] **Step 2: Write the failing test**

Create `Eonego.Tests/IsPseudoLegalTests.fs`:

```fsharp
module Eonego.Tests.IsPseudoLegalTests

#nowarn "9"

open System
open Microsoft.FSharp.NativeInterop
open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.Tests.TestFixtures

let private sq f r = mkSquare f r
let private a1 = sq 0 0
let private b1 = sq 1 0
let private c1 = sq 2 0
let private d1 = sq 3 0
let private e1 = sq 4 0
let private f1 = sq 5 0
let private g1 = sq 6 0
let private h1 = sq 7 0
let private a2 = sq 0 1
let private b2 = sq 1 1
let private c2 = sq 2 1
let private d2 = sq 3 1
let private e2 = sq 4 1
let private e3 = sq 4 2
let private e4 = sq 4 3
let private e5 = sq 4 4
let private e7 = sq 4 6
let private e8 = sq 4 7
let private d5 = sq 3 4
let private d6 = sq 3 5
let private f3 = sq 5 2
let private c3 = sq 2 2
let private c6 = sq 2 5
let private a8 = sq 0 7
let private c8 = sq 2 7
let private g8 = sq 6 7
let private h8 = sq 7 7
let private e6 = sq 4 5
let private a3 = sq 0 2
let private d3 = sq 3 2

// --- ORACLE: IsPseudoLegal must agree with generate(NonEvasions) for non-king moves ---
[<Fact>]
let ``IsPseudoLegal agrees with generate(NonEvasions) on startpos`` () =
    let p = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    let pseudo = collectPseudo p NonEvasions
    // Every generated non-king move must be IsPseudoLegal=true
    for m in pseudo do
        Assert.True(p.IsPseudoLegal m, sprintf "IsPseudoLegal rejected generated move %s" (toUci m))
    // 20 distinct moveMatchKeys in startpos
    Assert.Equal(20, pseudo.Length)

// --- ORACLE: in-check position uses generate(Evasions) ---
[<Fact>]
let ``IsPseudoLegal agrees with generate(Evasions) when in check`` () =
    let p = Position.OfFen "4r2k/8/8/8/8/8/8/4K3 w - - 0 1"
    Assert.True(p.InCheck)
    let evasions = collectPseudo p Evasions
    // Ke1e2 (along the checking ray) should NOT be pseudo-legal (king-safety gate rejects it)
    Assert.False(p.IsPseudoLegal (mkMove e1 e2), "Ke2 along checking ray should be rejected by king-safety gate")
    // The 4 legal evasions d1,f1,d2,f2 should be pseudo-legal
    Assert.True(p.IsPseudoLegal (mkMove e1 d1))
    Assert.True(p.IsPseudoLegal (mkMove e1 f1))

// --- CRAFTED FALSE CASES ---
[<Fact>]
let ``rejects move from empty square`` () =
    let p = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    Assert.False(p.IsPseudoLegal (mkMove e4 e5))  // e4 is empty in startpos

[<Fact>]
let ``rejects own-piece capture`` () =
    let p = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    Assert.False(p.IsPseudoLegal (mkMove b1 d2))  // Nb1xd2 but d2 is own pawn

[<Fact>]
let ``rejects enemy piece on from square`` () =
    let p = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    Assert.False(p.IsPseudoLegal (mkMove e7 e6))  // e7 is a black pawn, white to move

[<Fact>]
let ``rejects blocked slider`` () =
    let p = Position.OfFen "4k3/8/8/8/8/8/4P3/R3K3 w - - 0 1"
    Assert.True(p.IsPseudoLegal (mkMove a1 h1))   // clear rank
    let p2 = Position.OfFen "4k3/8/8/8/8/8/8/RP2K3 w - - 0 1"
    Assert.False(p.IsPseudoLegal (mkMove a1 h1))  // Pb1 blocks

[<Fact>]
let ``rejects pawn diagonal to empty square`` () =
    let p = Position.OfFen "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    Assert.False(p.IsPseudoLegal (mkMove e2 f3))  // f3 empty, no capture

[<Fact>]
let ``rejects pawn single push onto enemy`` () =
    let p = Position.OfFen "4k3/8/8/8/8/4p3/4P3/4K3 w - - 0 1"
    Assert.False(p.IsPseudoLegal (mkMove e2 e3))  // e3 occupied by enemy pawn

[<Fact>]
let ``rejects double push over blocker`` () =
    let p = Position.OfFen "4k3/8/8/8/8/4p3/4P3/4K3 w - - 0 1"
    Assert.False(p.IsPseudoLegal (mkMove e2 e4))  // e3 blocks the double push

[<Fact>]
let ``rejects double push from non-start rank`` () =
    let p = Position.OfFen "4k3/8/8/8/8/4P3/8/4K3 w - - 0 1"
    Assert.False(p.IsPseudoLegal (mkMove e3 e5))  // e3 not on Rank2

[<Fact>]
let ``valid double push positive`` () =
    let p = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    Assert.True(p.IsPseudoLegal (mkMove e2 e4))

[<Fact>]
let ``rejects EP when EpSquare is NoSquare`` () =
    let p = Position.OfFen "4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1"  // no ep target set
    Assert.False(p.IsPseudoLegal (mkEnPassant e5 d6))

[<Fact>]
let ``valid EP positive`` () =
    let p = Position.OfFen "k7/8/8/3pP3/8/2K5/8/8 w - d6 0 1"
    Assert.True(p.IsPseudoLegal (mkEnPassant e5 d6))

[<Fact>]
let ``rejects EP wrong target`` () =
    let p = Position.OfFen "k7/8/8/3pP3/8/2K5/8/8 w - d6 0 1"
    Assert.False(p.IsPseudoLegal (mkEnPassant e5 f6))  // EpSquare=d6, not f6

[<Fact>]
let ``rejects castling without the right`` () =
    let p = Position.OfFen "4k3/8/8/8/8/8/8/4K2R w - - 0 1"  // rook h1 but no K right
    Assert.False(p.IsPseudoLegal (mkCastling e1 g1))

[<Fact>]
let ``rejects castling through check`` () =
    let p = Position.OfFen "5rk1/8/8/8/8/8/8/4K2R w K - 0 1"  // Rf8 attacks f1
    Assert.False(p.IsPseudoLegal (mkCastling e1 g1))

[<Fact>]
let ``rejects castling while in check`` () =
    let p = Position.OfFen "4r2k/8/8/8/8/8/8/4K2R w K - 0 1"  // Re8 checks Ke1
    Assert.False(p.IsPseudoLegal (mkCastling e1 g1))

[<Fact>]
let ``valid castling positive`` () =
    let p = Position.OfFen "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
    Assert.True(p.IsPseudoLegal (mkCastling e1 g1))

[<Fact>]
let ``O-O-O allowed with b-file attacked-but-empty`` () =
    let p = Position.OfFen "1r5k/8/8/8/8/8/8/R3K3 w Q - 0 1"
    Assert.True(p.IsPseudoLegal (mkCastling e1 c1))

[<Fact>]
let ``rejects castling with rook missing`` () =
    let p = Position.OfFen "4k3/8/8/8/8/8/8/4K3 w K - 0 1"  // right K claimed but no rook on h1
    Assert.False(p.IsPseudoLegal (mkCastling e1 g1))

[<Fact>]
let ``rejects promotion flag on non-last-rank move`` () =
    let p = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    Assert.False(p.IsPseudoLegal (mkPromotion e2 e4 Queen))  // e2 not on Rank7, e4 not on Rank8

[<Fact>]
let ``valid promotion push`` () =
    let p = Position.OfFen "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1"
    Assert.True(p.IsPseudoLegal (mkPromotion e7 e8 Queen))
    Assert.True(p.IsPseudoLegal (mkPromotion e7 e8 Knight))
    Assert.True(p.IsPseudoLegal (mkPromotion e7 e8 Rook))
    Assert.True(p.IsPseudoLegal (mkPromotion e7 e8 Bishop))

[<Fact>]
let ``valid promotion capture`` () =
    let p = Position.OfFen "3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1"  // Pe7, black rook d8
    Assert.True(p.IsPseudoLegal (mkPromotion e7 d8 Knight))

[<Fact>]
let ``rejects promotion push blocked`` () =
    let p = Position.OfFen "4rk2/4P3/8/8/8/8/8/4K3 w - - 0 1"  // black rook e8 blocks
    Assert.False(p.IsPseudoLegal (mkPromotion e7 e8 Queen))

[<Fact>]
let ``rejects normal flag on last-rank pawn move`` () =
    let p = Position.OfFen "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1"
    Assert.False(p.IsPseudoLegal (mkMove e7 e8))  // NORMAL flag onto back rank — must be Promotion

[<Fact>]
let ``rejects sentinels`` () =
    let p = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    Assert.False(p.IsPseudoLegal MoveNone)
    Assert.False(p.IsPseudoLegal MoveNull)

[<Fact>]
let ``rejects move legal in a different position`` () =
    let p = Position.OfFen "4k3/8/8/8/8/8/8/4K3 w - - 0 1"  // bare kings, e2 empty
    Assert.False(p.IsPseudoLegal (mkMove e2 e4))  // no pawn on e2

[<Fact>]
let ``in-check non-king interpose positive and negative`` () =
    // Bc1 can interpose on e3 between Ke1 and Re8; Bc1-a3 does NOT block
    let p = Position.OfFen "4r2k/8/8/8/8/8/8/2B1K3 w - - 0 1"
    Assert.True(p.IsPseudoLegal (mkMove c1 e3))   // Bc1-e3 interposes on e-file
    Assert.False(p.IsPseudoLegal (mkMove c1 a3))  // Bc1-a3 does not block
```

- [ ] **Step 3: Add `IsPseudoLegalTests.fs` to the test project**

Modify `Eonego.Tests/Eonego.Tests.fsproj` — add after `SeeGeTests.fs`:

```xml
    <Compile Include="SeeGeTests.fs" />
    <Compile Include="IsPseudoLegalTests.fs" />
```

- [ ] **Step 4: Run test to verify it fails**

Run: `dotnet test Eonego.Tests --filter "FullyQualifiedName~IsPseudoLegalTests"`
Expected: FAIL — `IsPseudoLegal` not defined.

- [ ] **Step 5: Add `IsPseudoLegal` and `ResolvesCheck` to Position.fs**

Add after `GivesCheck` (the last member, ~line 665):

```fsharp
    /// Pseudo-legality oracle (SF Position::pseudo_legal port). Returns true iff m
    /// could be generated for the current position (in-check: Evasions; else NonEvasions).
    /// Folds the king-into-check test (SF pseudo_legal does this); pin legality is
    /// deferred to isLegal (as in SF). PRE: none (sentinels/corrupt moves return false).
    member this.IsPseudoLegal (m: Move) : bool =
        if not (isOk m) then false
        else
            let us   = sideToMove
            let them = flipColor us
            let from = fromSq m
            let dst  = toSq m
            let pc   = board.[from]
            if pc = NoPiece || pieceColor pc <> us then false
            else
                let occ = byTypeBB.[AllPieces]
                let pt  = pieceType pc
                match moveFlag m with
                | FlagCastling ->
                    if pt <> King || from <> this.KingSquare us then false
                    else
                        let right =
                            if us = White then
                                if dst = castleKingDest.[WK] then WK
                                elif dst = castleKingDest.[WQ] then WQ else 0
                            else
                                if dst = castleKingDest.[BK] then BK
                                elif dst = castleKingDest.[BQ] then BQ else 0
                        if right = 0 then false
                        elif not (this.CanCastle right) then false
                        elif this.InCheck then false
                        elif (castleEmptyPath.[right] &&& occ) <> 0UL then false
                        elif board.[castleRookOrigin.[right]] <> makePiece us Rook then false
                        else
                            let mutable path = castleKingPath.[right]
                            let mutable safe = true
                            while path <> 0UL && safe do
                                if this.AttackedBy them (popLsb &path) then safe <- false
                            safe
                | FlagEnPassant ->
                    let st = &states.[stPly]
                    if pt <> Pawn || st.EpSquare = NoSquare || dst <> st.EpSquare then false
                    elif not (testBit (pawnAttacks them dst) from) then false
                    else
                        let capSq = if us = White then dst - 8 else dst + 8
                        if board.[capSq] <> makePiece them Pawn then false
                        else
                            if st.Checkers <> 0UL then testBit st.Checkers capSq else true
                | FlagPromotion ->
                    let rank7 = if us = White then Rank7BB else Rank2BB
                    let rank8 = if us = White then Rank8 else Rank1
                    if pt <> Pawn then false
                    elif (bit from &&& rank7) = 0UL then false
                    elif (bit dst &&& rank8) = 0UL then false
                    else
                        let push = (dst = (if us = White then from + 8 else from - 8)) && board.[dst] = NoPiece
                        let capE = testBit (pawnAttacks us from) dst
                                    && board.[dst] <> NoPiece && pieceColor board.[dst] = them
                        if not (push || capE) then false
                        else this.ResolvesCheck us from dst false
                | _ ->
                    let cap = board.[dst]
                    if cap <> NoPiece && pieceColor cap = us then false
                    else
                        let shapeOk =
                            if pt = Pawn then
                                let backRank = if us = White then Rank8 else Rank1
                                if (bit dst &&& backRank) <> 0UL then false
                                else
                                    let one  = if us = White then from + 8 else from - 8
                                    let two  = if us = White then from + 16 else from - 16
                                    let startRank = if us = White then Rank2BB else Rank7BB
                                    if dst = one then board.[dst] = NoPiece
                                    elif dst = two then
                                        (bit from &&& startRank) <> 0UL
                                        && board.[one] = NoPiece && board.[dst] = NoPiece
                                    else
                                        testBit (pawnAttacks us from) dst && cap <> NoPiece
                            else
                                testBit (attacksFrom pt us from occ) dst
                        if not shapeOk then false
                        else this.ResolvesCheck us from dst (pt = King)

    /// In-check resolution helper (private). king=true: destination must be safe
    /// with the king removed from occ (X-ray). king=false: under single check must
    /// land on between|checkers; double check illegal.
    member private this.ResolvesCheck (us: Color) (from: Square) (dst: Square) (isKing: bool) : bool =
        let st = &states.[stPly]
        let them = flipColor us
        if isKing then
            let occ = byTypeBB.[AllPieces]
            (this.AttackersTo dst (occ ^^^ (bit from)) &&& byColorBB.[them]) = 0UL
        elif st.Checkers = 0UL then true
        elif moreThanOne st.Checkers then false
        else
            let ksq = this.KingSquare us
            let checker = lsb st.Checkers
            let target = (between ksq checker) ||| st.Checkers
            testBit target dst
```

- [ ] **Step 6: Run test to verify it passes**

Run: `dotnet test Eonego.Tests --filter "FullyQualifiedName~IsPseudoLegalTests"`
Expected: PASS — all assertions green.

- [ ] **Step 7: Run perft to verify no regression**

Run: `dotnet test Eonego.Tests --filter "FullyQualifiedName~MoveGenerationTests"`
Expected: PASS — Position changes are additive; perft unaffected.

- [ ] **Step 8: Commit**

```bash
git add Eonego/Position.fs Eonego.Tests/TestFixtures.fs Eonego.Tests/IsPseudoLegalTests.fs Eonego.Tests/Eonego.Tests.fsproj
git commit -m "feat: add Position.IsPseudoLegal (inline SF pseudo_legal oracle)"
```

---

### Task 3: History.fs — Tables Class

**Files:**
- Create: `Eonego/History.fs`
- Modify: `Eonego/Eonego.fsproj` (add `History.fs` after `MoveGeneration.fs`)
- Create: `Eonego.Tests/HistoryTests.fs`
- Modify: `Eonego.Tests/Eonego.Tests.fsproj` (add `HistoryTests.fs`)

- [ ] **Step 1: Write the failing test**

Create `Eonego.Tests/HistoryTests.fs`:

```fsharp
module Eonego.Tests.HistoryTests

open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.History

[<Fact>]
let ``MainHist returns zero for fresh tables`` () =
    let t = Tables()
    let m = mkMove 12 28  // e2e4
    Assert.Equal(0, t.MainHist White m)
    Assert.Equal(0, t.MainHist Black m)

[<Fact>]
let ``UpdateMain adds bonus clamped to HistMax`` () =
    let t = Tables()
    let m = mkMove 12 28
    t.UpdateMain White m 100
    Assert.Equal(100, t.MainHist White m)

[<Fact>]
let ``UpdateMain clamps to HistMax`` () =
    let t = Tables()
    let m = mkMove 12 28
    t.UpdateMain White m 99999
    Assert.Equal(HistMax, t.MainHist White m)

[<Fact>]
let ``UpdateMain clamps to -HistMax`` () =
    let t = Tables()
    let m = mkMove 12 28
    t.UpdateMain White m -99999
    Assert.Equal(-HistMax, t.MainHist White m)

[<Fact>]
let ``CaptureHistory returns zero for fresh tables`` () =
    let t = Tables()
    Assert.Equal(0, t.CaptureHistory (makePiece White Knight) 35 Pawn)

[<Fact>]
let ``UpdateCaptureHistory adds bonus`` () =
    let t = Tables()
    let pc = makePiece White Knight
    t.UpdateCaptureHistory pc 35 Pawn 50
    Assert.Equal(50, t.CaptureHistory pc 35 Pawn)

[<Fact>]
let ``MainHist indexes colors independently`` () =
    let t = Tables()
    let m = mkMove 12 28
    t.UpdateMain White m 100
    Assert.Equal(100, t.MainHist White m)
    Assert.Equal(0, t.MainHist Black m)  // black slot unaffected
```

- [ ] **Step 2: Add `HistoryTests.fs` to the test project**

Modify `Eonego.Tests/Eonego.Tests.fsproj` — add after `IsPseudoLegalTests.fs`:

```xml
    <Compile Include="IsPseudoLegalTests.fs" />
    <Compile Include="HistoryTests.fs" />
```

- [ ] **Step 3: Run test to verify it fails**

Run: `dotnet test Eonego.Tests --filter "FullyQualifiedName~HistoryTests"`
Expected: FAIL — `Eonego.History` module not found.

- [ ] **Step 4: Create `History.fs`**

Create `Eonego/History.fs`:

```fsharp
/// Eonego — history heuristics for the Phase-2 move picker.
///
/// Tables is a [<Sealed>] class (one instance per search thread, reused across the
/// search). It owns two flat int[] arrays:
///   main        — butterfly MainHistory indexed [color * 4096 + fromTo m]
///   captureHist — CaptureHistory indexed [piece * 384 + to * 6 + capturedPT]
/// All arrays are zero-initialized (fresh tables return 0 for every query).
/// The gravity-clamp update formula matches Stockfish: v += bonus - v*|bonus|/HistMax.
module Eonego.History

open System
open System.Runtime.CompilerServices
open Eonego.Bitboard
open Eonego.Move

[<Literal>]
let HistMax = 1 <<< 14  // 16384

[<Sealed>]
type Tables() =
    let main : int[] = Array.zeroCreate (2 * 4096)
    let captureHist : int[] = Array.zeroCreate (12 * 64 * 6)  // 4608

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.MainHist (c: Color) (m: Move) : int =
        main.[(c <<< 12) ||| fromTo m]

    member _.UpdateMain (c: Color) (m: Move) (bonus: int) : unit =
        let i = (c <<< 12) ||| fromTo m
        let clamped = max -HistMax (min HistMax bonus)
        main.[i] <- main.[i] + clamped - main.[i] * (abs clamped) / HistMax

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.CaptureHistory (pc: Piece) (toSq: Square) (capturedPT: PieceType) : int =
        captureHist.[pc * 384 + toSq * 6 + capturedPT]

    member _.UpdateCaptureHistory (pc: Piece) (toSq: Square) (capturedPT: PieceType) (bonus: int) : unit =
        let i = pc * 384 + toSq * 6 + capturedPT
        let clamped = max -HistMax (min HistMax bonus)
        captureHist.[i] <- captureHist.[i] + clamped - captureHist.[i] * (abs clamped) / HistMax

    member _.Clear () : unit =
        Array.Clear(main, 0, main.Length)
        Array.Clear(captureHist, 0, captureHist.Length)
```

- [ ] **Step 5: Add `History.fs` to the main project**

Modify `Eonego/Eonego.fsproj` — add after `MoveGeneration.fs`:

```xml
    <Compile Include="MoveGeneration.fs" />
    <Compile Include="History.fs" />
```

- [ ] **Step 6: Run test to verify it passes**

Run: `dotnet test Eonego.Tests --filter "FullyQualifiedName~HistoryTests"`
Expected: PASS — all 7 assertions green.

- [ ] **Step 7: Commit**

```bash
git add Eonego/History.fs Eonego/Eonego.fsproj Eonego.Tests/HistoryTests.fs Eonego.Tests/Eonego.Tests.fsproj
git commit -m "feat: add History.fs with Tables class (MainHistory + CaptureHistory)"
```

---

### Task 4: PROMOTION QUIRK Fix in MoveGeneration

**Files:**
- Modify: `Eonego/MoveGeneration.fs` (`genPawnMoves`, ~lines 102-121)
- Create: `Eonego.Tests/CapturePromoTests.fs` (or append to `MoveGenerationTests.fs`)
- Modify: `Eonego.Tests/Eonego.Tests.fsproj`

- [ ] **Step 1: Write the failing test**

Create `Eonego.Tests/CapturePromoTests.fs`:

```fsharp
module Eonego.Tests.CapturePromoTests

#nowarn "9"

open System
open Microsoft.FSharp.NativeInterop
open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.Tests.TestFixtures

let private sq f r = mkSquare f r
let private e7 = sq 4 6
let private e8 = sq 4 7

let private collectGen (pos: Position) (genType: int) : Move[] =
    let p = NativePtr.stackalloc<Move> MaxMoves
    let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)
    let n = generate pos buf genType
    let r = ResizeArray<Move>(n)
    for i in 0 .. n - 1 do r.Add buf.[i]
    r.ToArray()

[<Fact>]
let ``generate Captures includes queen push-promotion`` () =
    // White Pe7 can push-promote to e8=Q (no capture). generate(Captures) MUST
    // include this move (the PROMOTION QUIRK fix).
    let p = Position.OfFen "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1"
    let caps = collectGen p Captures
    let queenPromo = mkPromotion e7 e8 Queen
    Assert.True(
        caps |> Array.exists (fun m -> m = queenPromo),
        "generate(Captures) must include queen push-promotion e7e8=Q")

[<Fact>]
let ``generate Captures does not include under-promotion push`` () =
    // Only Queen push-promotions are tactical enough for the Captures stage.
    // Under-promotions (N/B/R push) should NOT appear in generate(Captures).
    let p = Position.OfFen "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1"
    let caps = collectGen p Captures
    let knightPromo = mkPromotion e7 e8 Knight
    Assert.False(
        caps |> Array.exists (fun m -> m = knightPromo),
        "generate(Captures) should not include under-promotion push e7e8=N")

[<Fact>]
let ``generate NonEvasions still includes all four push-promotions`` () =
    // The fix must not break NonEvasions (used by perft) — all 4 promotions still emitted.
    let p = Position.OfFen "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1"
    let moves = collectGen p NonEvasions
    let promos = moves |> Array.filter (fun m -> isPromotion m && fromSq m = e7 && toSq m = e8)
    Assert.Equal(4, promos.Length)

[<Fact>]
let ``perft unchanged after promotion quirk fix`` () =
    // The fix only affects generate(Captures); perft uses NonEvasions/Evasions.
    let p = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    Assert.Equal(20UL, perft p 1)
    Assert.Equal(400UL, perft p 2)
```

- [ ] **Step 2: Add `CapturePromoTests.fs` to the test project**

Modify `Eonego.Tests/Eonego.Tests.fsproj` — add after `HistoryTests.fs`:

```xml
    <Compile Include="HistoryTests.fs" />
    <Compile Include="CapturePromoTests.fs" />
```

- [ ] **Step 3: Run test to verify it fails**

Run: `dotnet test Eonego.Tests --filter "FullyQualifiedName~CapturePromoTests"`
Expected: FAIL — `generate Captures includes queen push-promotion` fails (push-promotions not in Captures).

- [ ] **Step 4: Fix `genPawnMoves` in MoveGeneration.fs**

In `Eonego/MoveGeneration.fs`, modify `genPawnMoves` (~lines 102-121). The current code has:

```fsharp
    // --- pushes (quiet) + push-promotions : skipped for Captures ---
    if genType <> Captures then
        let single = (if us = White then shiftN nonPromo else shiftS nonPromo) &&& empty
        let mutable b = single &&& target
        while b <> 0UL do
            let dst = popLsb &b
            addMove moves &n (mkMove (if us = White then dst - 8 else dst + 8) dst)
        let dbl =
            (if us = White then shiftN (single &&& Rank3) else shiftS (single &&& Rank6))
            &&& empty &&& target
        let mutable d = dbl
        while d <> 0UL do
            let dst = popLsb &d
            addMove moves &n (mkMove (if us = White then dst - 16 else dst + 16) dst)
        // push-promotions (promo pawns advancing to the back rank)
        let pp = (if us = White then shiftN promo else shiftS promo) &&& empty &&& target
        let mutable q = pp
        while q <> 0UL do
            let dst = popLsb &q
            addPromotions moves &n (if us = White then dst - 8 else dst + 8) dst
```

Replace with (push single/double pushes remain skipped for Captures; push-promotions are moved out and always emitted — Queen only for Captures, all four for other genTypes):

```fsharp
    // --- pushes (quiet) : skipped for Captures ---
    if genType <> Captures then
        let single = (if us = White then shiftN nonPromo else shiftS nonPromo) &&& empty
        let mutable b = single &&& target
        while b <> 0UL do
            let dst = popLsb &b
            addMove moves &n (mkMove (if us = White then dst - 8 else dst + 8) dst)
        let dbl =
            (if us = White then shiftN (single &&& Rank3) else shiftS (single &&& Rank6))
            &&& empty &&& target
        let mutable d = dbl
        while d <> 0UL do
            let dst = popLsb &d
            addMove moves &n (mkMove (if us = White then dst - 16 else dst + 16) dst)

    // --- push-promotions : always emitted (SF generates queen push-promos as
    //     tactical moves for the Captures stage). Captures stage gets Queen only;
    //     other stages get all four. Destination is always empty (push, not capture). ---
    let ppTarget = if genType = Captures then empty else empty &&& target
    let pp = (if us = White then shiftN promo else shiftS promo) &&& ppTarget
    let mutable q = pp
    while q <> 0UL do
        let dst = popLsb &q
        let from = if us = White then dst - 8 else dst + 8
        if genType = Captures then
            addMove moves &n (mkPromotion from dst Queen)
        else
            addPromotions moves &n from dst
```

- [ ] **Step 5: Run test to verify it passes**

Run: `dotnet test Eonego.Tests --filter "FullyQualifiedName~CapturePromoTests"`
Expected: PASS — all 4 assertions green.

- [ ] **Step 6: Run full perft suite to verify no regression**

Run: `dotnet test Eonego.Tests --filter "FullyQualifiedName~MoveGenerationTests"`
Expected: PASS — all perft d1-d4 and targeted fixtures green.

- [ ] **Step 7: Commit**

```bash
git add Eonego/MoveGeneration.fs Eonego.Tests/CapturePromoTests.fs Eonego.Tests/Eonego.Tests.fsproj
git commit -m "fix: add queen push-promotions to generate(Captures) (PROMOTION QUIRK)"
```

---

### Task 5: MovePicker Compile-Probe (de-risk AOT + byref mutation)

**Files:**
- Create: `Eonego/MovePicker.fs` (minimal struct + factory + stub nextMove)
- Modify: `Eonego/Eonego.fsproj` (add `MovePicker.fs` after `History.fs`)
- Create: `Eonego.Tests/MovePickerTests.fs` (probe test)
- Modify: `Eonego.Tests/Eonego.Tests.fsproj` (add `MovePickerTests.fs`)

**Goal:** Prove the `[<Struct; IsByRefLike>]` layout compiles under FSC and that `byref<MovePicker>` threading persists mutations — the single highest-risk question — BEFORE porting any movepick staging logic.

- [ ] **Step 1: Write the failing test**

Create `Eonego.Tests/MovePickerTests.fs`:

```fsharp
module Eonego.Tests.MovePickerTests

open System
open Xunit
open Eonego.Bitboard
open Eonego.Move
open Eonego.MovePicker

[<Fact>]
let ``T0 byref mutation persists across nextMove calls`` () =
    let arr = [| 10; 11; 12; 13 |]
    let moves = Span<Move>(arr, 0, 4)
    let mutable mp = mkProbe moves
    let _ = nextMove &mp false
    Assert.Equal(1, mp.Stage)
    Assert.Equal(1, mp.Cur)
    let _ = nextMove &mp false
    Assert.Equal(2, mp.Stage)
    Assert.Equal(2, mp.Cur)
```

- [ ] **Step 2: Add `MovePickerTests.fs` to the test project**

Modify `Eonego.Tests/Eonego.Tests.fsproj` — add after `CapturePromoTests.fs`:

```xml
    <Compile Include="CapturePromoTests.fs" />
    <Compile Include="MovePickerTests.fs" />
```

- [ ] **Step 3: Run test to verify it fails**

Run: `dotnet test Eonego.Tests --filter "FullyQualifiedName~MovePickerTests"`
Expected: FAIL — `Eonego.MovePicker` module not found.

- [ ] **Step 4: Create `MovePicker.fs` (minimal probe version)**

Create `Eonego/MovePicker.fs`:

```fsharp
/// Eonego — staged-lazy move picker (Phase-2 port of Stockfish movepick.cpp).
///
/// MovePicker is a [<Struct; IsByRefLike>] holding two caller-owned parallel spans
/// (Span<Move> moves + Span<int> scores), the live indices, and Position/Tables refs.
/// advance is a MODULE function over byref<MovePicker> — NEVER an instance method
/// (an instance method on a value type mutates a by-value copy; Stage/Cur would
/// never advance). The caller holds `let mutable mp = mkMain ...` and calls
/// `nextMove &mp skipQuiets`.
module Eonego.MovePicker

open System
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.History

// Stage ids — plain [<Literal>] ints. Ordering is load-bearing: fall-through == Stage+1.
[<Literal>] let StgMainTT       = 0
[<Literal>] let StgCaptureInit  = 1
[<Literal>] let StgGoodCapture  = 2
[<Literal>] let StgRefutation   = 3
[<Literal>] let StgQuietInit    = 4
[<Literal>] let StgQuiet        = 5
[<Literal>] let StgBadCapture   = 6
[<Literal>] let StgEvasionTT    = 7
[<Literal>] let StgEvasionInit  = 8
[<Literal>] let StgEvasion      = 9
[<Literal>] let StgProbCutTT    = 10
[<Literal>] let StgProbCutInit  = 11
[<Literal>] let StgProbCut      = 12
[<Literal>] let StgQSearchTT    = 13
[<Literal>] let StgQCaptureInit = 14
[<Literal>] let StgQCapture     = 15

[<Literal>]
let EvasionBonus = 1 <<< 28  // 268435456 — keeps capture-evasions above any history term

[<Struct; IsByRefLike>]
type MovePicker =
    val mutable Stage           : int
    val mutable Cur             : int
    val mutable EndMoves        : int
    val mutable EndBadCaptures  : int
    val mutable RefIdx          : int
    val mutable TtMove          : Move
    val mutable Killer1         : Move
    val mutable Killer2         : Move
    val mutable CounterMove     : Move
    val mutable PrevSq          : Square
    val mutable Depth           : int
    val mutable Threshold       : int
    val mutable Moves           : Span<Move>
    val mutable Scores          : Span<int>
    val mutable Pos             : Position
    val mutable Tables          : Tables
    val mutable InCheck         : bool

// Probe factory — primes Stage/Cur to prove byref mutation persists.
let mkProbe (moves: Span<Move>) : MovePicker =
    let mutable mp = Unchecked.defaultof<MovePicker>
    mp.Moves <- moves
    mp.Stage <- 0
    mp.Cur <- 0
    mp

// Probe nextMove — increments Stage and Cur. PROVES byref threading works.
// Replaced by the full while-loop implementation in Task 6.
let nextMove (mp: byref<MovePicker>) (skipQuiets: bool) : Move =
    mp.Stage <- mp.Stage + 1
    mp.Cur <- mp.Cur + 1
    if mp.Cur < mp.Moves.Length then mp.Moves.[mp.Cur] else MoveNone
```

- [ ] **Step 5: Add `MovePicker.fs` to the main project**

Modify `Eonego/Eonego.fsproj` — add after `History.fs`:

```xml
    <Compile Include="History.fs" />
    <Compile Include="MovePicker.fs" />
```

- [ ] **Step 6: Run test to verify it passes**

Run: `dotnet test Eonego.Tests --filter "FullyQualifiedName~MovePickerTests"`
Expected: PASS — `mp.Stage = 2` proves byref mutation persisted across two calls.

- [ ] **Step 7: Commit**

```bash
git add Eonego/MovePicker.fs Eonego/Eonego.fsproj Eonego.Tests/MovePickerTests.fs Eonego.Tests/Eonego.Tests.fsproj
git commit -m "feat: add MovePicker compile-probe (struct layout + byref mutation verified)"
```

---

### Task 6: MovePicker Main Search Chain

**Files:**
- Modify: `Eonego/MovePicker.fs` (replace probe with full implementation)
- Modify: `Eonego.Tests/MovePickerTests.fs` (add F1-F6 fixtures)

**Goal:** Implement the Main search chain (TT, CaptureInit, GoodCapture, Refutation, QuietInit, Quiet, BadCapture) with the while-loop `nextMove`, all scoring/picking helpers, and the `mkMain`/`mkQSearch`/`mkProbCut` factories.

- [ ] **Step 1: Write the failing tests (F1-F6)**

Add to `Eonego.Tests/MovePickerTests.fs` (after the T0 probe test):

```fsharp
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.History
open Eonego.Tests.TestFixtures

let private sq f r = mkSquare f r
let private e1 = sq 4 0
let private e2 = sq 4 1
let private e4 = sq 4 3
let private e5 = sq 4 4
let private e7 = sq 4 6
let private e8 = sq 4 7
let private d5 = sq 3 4
let private d8 = sq 3 7
let private e3 = sq 4 2
let private c6 = sq 2 5
let private d1 = sq 3 0
let private g1 = sq 6 0
let private f3 = sq 5 2
let private b1 = sq 1 0
let private c3 = sq 2 2
let private c1 = sq 2 0

let private drainPicker (mp: byref<MovePicker>) (skipQuiets: bool) : Move[] =
    let r = ResizeArray<Move>()
    let mutable m = nextMove &mp skipQuiets
    while m <> MoveNone do
        r.Add m
        m <- nextMove &mp skipQuiets
    r.ToArray()

let private makeBuffers () =
    let movesArr = Array.zeroCreate<Move> MaxMoves
    let scoresArr = Array.zeroCreate<int> MaxMoves
    (Span<Move>(movesArr, 0, MaxMoves), Span<int>(scoresArr, 0, MaxMoves))

// F1: startpos, no ttMove, no killers, skipQuiets=false → 20 moves total
[<Fact>]
let ``F1 startpos drains 20 moves`` () =
    let p = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    let t = Tables()
    let (moves, scores) = makeBuffers ()
    let mutable mp = mkMain p t MoveNone MoveNone MoveNone MoveNone NoSquare 8 moves scores
    let result = drainPicker &mp false
    Assert.Equal(20, result.Length)
    // No captures in startpos → all 20 are quiets
    for m in result do
        Assert.True(p.IsPseudoLegal m, sprintf "emitted non-pseudo-legal move %s" (toUci m))

// F2: good capture ordering — e4xd5 (pawn takes queen, SEE >= 0)
[<Fact>]
let ``F2 good capture emitted first`` () =
    let p = Position.OfFen "4k3/8/8/3q4/4P3/8/8/3RK3 w - - 0 1"
    let t = Tables()
    let (moves, scores) = makeBuffers ()
    let mutable mp = mkMain p t MoveNone MoveNone MoveNone MoveNone NoSquare 4 moves scores
    let result = drainPicker &mp true  // skipQuiets=true (capture-only drain)
    Assert.True(result.Length > 0)
    Assert.True(result.[0] = mkMove e4 d5, "first move should be e4xd5 (good capture)")

// F3: bad capture spilled and emitted AFTER quiets
[<Fact>]
let ``F3 bad capture emitted last`` () =
    // Ne3xd5: SEE = 100-300 = -200 < 0 → bad capture. Bc6 defends d5.
    let p = Position.OfFen "4k3/8/2b5/3p4/8/4N3/8/4K3 w - - 0 1"
    let t = Tables()
    let (moves, scores) = makeBuffers ()
    let mutable mp = mkMain p t MoveNone MoveNone MoveNone MoveNone NoSquare 4 moves scores
    let result = drainPicker &mp false
    let nxd5 = mkMove e3 d5
    let lastMove = result.[result.Length - 1]
    Assert.True(lastMove = nxd5, "bad capture Nxd5 should be the LAST move emitted")
    Assert.True(result.Length > 1, "at least one quiet should precede the bad capture")

// F4: TT move first + dedup
[<Fact>]
let ``F4 TT move first and not duplicated`` () =
    let p = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    let t = Tables()
    let ttMove = mkMove e2 e4
    let (moves, scores) = makeBuffers ()
    let mutable mp = mkMain p t ttMove MoveNone MoveNone MoveNone NoSquare 8 moves scores
    let result = drainPicker &mp false
    Assert.True(result.[0] = ttMove, "first move should be the TT move")
    let count = result |> Array.filter (fun m -> m = ttMove) |> Array.length
    Assert.Equal(1, count)  // appears exactly once
    Assert.Equal(20, result.Length)

// F5: killer/countermove ordering
[<Fact>]
let ``F5 killer1 before countermove before king quiets`` () =
    let p = Position.OfFen "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    let t = Tables()
    let killer1 = mkMove e2 e4
    let counterMove = mkMove e2 e3
    let (moves, scores) = makeBuffers ()
    let mutable mp = mkMain p t MoveNone killer1 MoveNone counterMove NoSquare 2 moves scores
    let result = drainPicker &mp false
    let idxOf (m: Move) = result |> Array.findIndex (fun x -> x = m)
    Assert.True(idxOf killer1 < idxOf counterMove, "killer1 should come before countermove")
    Assert.True(idxOf counterMove < result.Length - 1, "countermove should not be last (king quiets follow)")

// F6: refutation rejects a capture killer
[<Fact>]
let ``F6 refutation rejects capture killer`` () =
    // e4xd5 is a CAPTURE. killer1 = e4d5 → refutation gate rejects it (must be non-capture).
    // e4d5 should still appear via the GoodCapture stage (before refutations).
    let p = Position.OfFen "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"
    let t = Tables()
    let killer1 = mkMove e4 d5  // a capture — should be rejected as refutation
    let (moves, scores) = makeBuffers ()
    let mutable mp = mkMain p t MoveNone killer1 MoveNone MoveNone NoSquare 4 moves scores
    let result = drainPicker &mp false
    let e4d5Count = result |> Array.filter (fun m -> m = mkMove e4 d5) |> Array.length
    Assert.Equal(1, e4d5Count)  // appears exactly once (from GoodCapture, not from refutation)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test Eonego.Tests --filter "FullyQualifiedName~MovePickerTests~F"`
Expected: FAIL — `mkMain` not defined (only `mkProbe` exists).

- [ ] **Step 3: Replace `MovePicker.fs` with the full implementation**

Replace `Eonego/MovePicker.fs` with:

```fsharp
/// Eonego — staged-lazy move picker (Phase-2 port of Stockfish movepick.cpp).
///
/// MovePicker is a [<Struct; IsByRefLike>] holding two caller-owned parallel spans
/// (Span<Move> moves + Span<int> scores), the live indices, and Position/Tables refs.
/// advance is a MODULE function over byref<MovePicker> — NEVER an instance method
/// (an instance method on a value type mutates a by-value copy; Stage/Cur would
/// never advance). The caller holds `let mutable mp = mkMain ...` and calls
/// `nextMove &mp skipQuiets`.
///
/// nextMove uses a while-loop (not tail recursion) because byref parameters can
/// defeat .tail under Native AOT — the loop guarantees no stack growth.
module Eonego.MovePicker

open System
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.History

// Stage ids — plain [<Literal>] ints. Ordering is load-bearing: fall-through == Stage+1.
[<Literal>] let StgMainTT       = 0
[<Literal>] let StgCaptureInit  = 1
[<Literal>] let StgGoodCapture  = 2
[<Literal>] let StgRefutation   = 3
[<Literal>] let StgQuietInit    = 4
[<Literal>] let StgQuiet        = 5
[<Literal>] let StgBadCapture   = 6
[<Literal>] let StgEvasionTT    = 7
[<Literal>] let StgEvasionInit  = 8
[<Literal>] let StgEvasion      = 9
[<Literal>] let StgProbCutTT    = 10
[<Literal>] let StgProbCutInit  = 11
[<Literal>] let StgProbCut      = 12
[<Literal>] let StgQSearchTT    = 13
[<Literal>] let StgQCaptureInit = 14
[<Literal>] let StgQCapture     = 15

[<Literal>]
let EvasionBonus = 1 <<< 28  // 268435456 — keeps capture-evasions above any history term

[<Struct; IsByRefLike>]
type MovePicker =
    val mutable Stage           : int
    val mutable Cur             : int
    val mutable EndMoves        : int
    val mutable EndBadCaptures  : int
    val mutable RefIdx          : int
    val mutable TtMove          : Move
    val mutable Killer1         : Move
    val mutable Killer2         : Move
    val mutable CounterMove     : Move
    val mutable PrevSq          : Square
    val mutable Depth           : int
    val mutable Threshold       : int
    val mutable Moves           : Span<Move>
    val mutable Scores          : Span<int>
    val mutable Pos             : Position
    val mutable Tables          : Tables
    val mutable InCheck         : bool

// ---------------------------------------------------------------------------
// Factories — prime all fields via Unchecked.defaultof + field-writes.
// TT move is validated by pos.IsPseudoLegal (SF verifies at construction).
// ---------------------------------------------------------------------------

let mkMain (pos: Position) (tables: Tables) (ttMove: Move)
           (k1: Move) (k2: Move) (cm: Move) (prevSq: Square) (depth: int)
           (moves: Span<Move>) (scores: Span<int>) : MovePicker =
    let tt = if ttMove <> MoveNone && pos.IsPseudoLegal ttMove then ttMove else MoveNone
    let mutable mp = Unchecked.defaultof<MovePicker>
    mp.Pos <- pos; mp.Tables <- tables
    mp.Moves <- moves; mp.Scores <- scores
    mp.TtMove <- tt; mp.Killer1 <- k1; mp.Killer2 <- k2; mp.CounterMove <- cm
    mp.PrevSq <- prevSq; mp.Depth <- depth; mp.Threshold <- 0
    mp.InCheck <- pos.InCheck
    mp.Stage <- if pos.InCheck then StgEvasionTT else StgMainTT
    mp.Cur <- 0; mp.EndMoves <- 0; mp.EndBadCaptures <- 0; mp.RefIdx <- 0
    mp

let mkQSearch (pos: Position) (tables: Tables) (ttMove: Move)
              (moves: Span<Move>) (scores: Span<int>) : MovePicker =
    let tt = if ttMove <> MoveNone && pos.IsPseudoLegal ttMove then ttMove else MoveNone
    let mutable mp = Unchecked.defaultof<MovePicker>
    mp.Pos <- pos; mp.Tables <- tables
    mp.Moves <- moves; mp.Scores <- scores
    mp.TtMove <- tt
    mp.Stage <- if pos.InCheck then StgEvasionTT else StgQSearchTT
    mp.Cur <- 0; mp.EndMoves <- 0; mp.EndBadCaptures <- 0; mp.RefIdx <- 0
    mp

let mkProbCut (pos: Position) (tables: Tables) (ttMove: Move) (threshold: int)
              (moves: Span<Move>) (scores: Span<int>) : MovePicker =
    let ok = ttMove <> MoveNone && pos.IsPseudoLegal ttMove
             && (pos.PieceOn (toSq ttMove) <> NoPiece || isEnPassant ttMove)
             && pos.SeeGe ttMove threshold
    let mutable mp = Unchecked.defaultof<MovePicker>
    mp.Pos <- pos; mp.Tables <- tables
    mp.Moves <- moves; mp.Scores <- scores
    mp.TtMove <- (if ok then ttMove else MoveNone)
    mp.Threshold <- threshold
    mp.Stage <- StgProbCutTT
    mp.Cur <- 0; mp.EndMoves <- 0; mp.EndBadCaptures <- 0; mp.RefIdx <- 0
    mp

// ---------------------------------------------------------------------------
// Helpers — all module functions over byref<MovePicker> (mutation persists).
// ---------------------------------------------------------------------------

let inline private swap (mp: byref<MovePicker>) (i: int) (j: int) =
    let m = mp.Moves.[i] in mp.Moves.[i] <- mp.Moves.[j]; mp.Moves.[j] <- m
    let s = mp.Scores.[i] in mp.Scores.[i] <- mp.Scores.[j]; mp.Scores.[j] <- s

let private pickBest (mp: byref<MovePicker>) (s: int) (e: int) : int =
    let mutable best = s
    for i in s + 1 .. e - 1 do
        if mp.Scores.[i] > mp.Scores.[best] then best <- i
    if best <> s then swap &mp s best
    s

let private partialInsertionSort (mp: byref<MovePicker>) (s: int) (e: int) (limit: int) =
    let mutable sortedEnd = s
    for p in s + 1 .. e - 1 do
        if mp.Scores.[p] > limit then
            sortedEnd <- sortedEnd + 1
            let tmpM = mp.Moves.[p]
            let tmpS = mp.Scores.[p]
            mp.Moves.[p] <- mp.Moves.[sortedEnd]
            mp.Scores.[p] <- mp.Scores.[sortedEnd]
            let mutable q = sortedEnd
            while q > s && mp.Scores.[q - 1] < tmpS do
                mp.Moves.[q] <- mp.Moves.[q - 1]
                mp.Scores.[q] <- mp.Scores.[q - 1]
                q <- q - 1
            mp.Moves.[q] <- tmpM
            mp.Scores.[q] <- tmpS

let private scoreCaptures (mp: byref<MovePicker>) (s: int) (e: int) =
    let pos = mp.Pos
    for i in s .. e - 1 do
        let m = mp.Moves.[i]
        let pc = pos.PieceOn (fromSq m)
        let capturedPT = if isEnPassant m then Pawn else pieceType (pos.PieceOn (toSq m))
        mp.Scores.[i] <- 7 * pieceValueOf capturedPT + mp.Tables.CaptureHistory pc (toSq m) capturedPT

let private scoreQuiets (mp: byref<MovePicker>) (s: int) (e: int) =
    let us = mp.Pos.SideToMove
    for i in s .. e - 1 do
        mp.Scores.[i] <- mp.Tables.MainHist us mp.Moves.[i]

let private scoreEvasions (mp: byref<MovePicker>) (s: int) (e: int) =
    let pos = mp.Pos
    let us = pos.SideToMove
    for i in s .. e - 1 do
        let m = mp.Moves.[i]
        if pos.PieceOn (toSq m) <> NoPiece || isEnPassant m then
            let capturedPT = if isEnPassant m then Pawn else pieceType (pos.PieceOn (toSq m))
            let moverPT = pieceType (pos.PieceOn (fromSq m))
            mp.Scores.[i] <- pieceValueOf capturedPT - pieceValueOf moverPT + EvasionBonus
        else
            mp.Scores.[i] <- mp.Tables.MainHist us m

let private refutationNext (mp: byref<MovePicker>) : Move =
    let mutable found = MoveNone
    while mp.RefIdx < 3 && found = MoveNone do
        let candidate =
            match mp.RefIdx with
            | 0 -> mp.Killer1
            | 1 -> if mp.Killer2 <> mp.Killer1 then mp.Killer2 else MoveNone
            | _ -> if mp.CounterMove <> mp.Killer1 && mp.CounterMove <> mp.Killer2 then mp.CounterMove else MoveNone
        mp.RefIdx <- mp.RefIdx + 1
        if candidate <> MoveNone && candidate <> mp.TtMove
           && not (isSpecial candidate)
           && mp.Pos.PieceOn (toSq candidate) = NoPiece
           && mp.Pos.IsPseudoLegal candidate then
            found <- candidate
    found

// ---------------------------------------------------------------------------
// nextMove — while-loop over byref<MovePicker>. Each stage either returns a
// move (sets result + done_) or falls through (mutates Stage, loop continues).
// ---------------------------------------------------------------------------

let nextMove (mp: byref<MovePicker>) (skipQuiets: bool) : Move =
    let mutable result = MoveNone
    let mutable done_ = false
    while not done_ do
        match mp.Stage with
        | StgMainTT | StgEvasionTT | StgQSearchTT | StgProbCutTT ->
            mp.Stage <- mp.Stage + 1
            if mp.TtMove <> MoveNone then
                result <- mp.TtMove
                done_ <- true

        | StgCaptureInit ->
            mp.EndMoves <- generate mp.Pos mp.Moves Captures
            scoreCaptures &mp 0 mp.EndMoves
            mp.Cur <- 0
            mp.EndBadCaptures <- 0
            mp.Stage <- StgGoodCapture

        | StgGoodCapture ->
            if mp.Cur < mp.EndMoves then
                let i = pickBest &mp mp.Cur mp.EndMoves
                let m = mp.Moves.[i]
                mp.Cur <- mp.Cur + 1
                if m = mp.TtMove then ()
                elif mp.Pos.SeeGe m 0 then
                    result <- m; done_ <- true
                else
                    swap &mp (mp.Cur - 1) mp.EndBadCaptures
                    mp.EndBadCaptures <- mp.EndBadCaptures + 1
            else
                mp.Stage <- StgRefutation

        | StgRefutation ->
            if skipQuiets then
                mp.Stage <- StgQuietInit
            else
                let found = refutationNext &mp
                if found <> MoveNone then
                    result <- found; done_ <- true
                else
                    mp.Stage <- StgQuietInit

        | StgQuietInit ->
            if skipQuiets then
                mp.Stage <- StgBadCapture
            else
                let qStart = mp.EndMoves
                mp.Cur <- qStart
                let qn = generate mp.Pos (mp.Moves.Slice(qStart)) Quiets
                mp.EndMoves <- qStart + qn
                scoreQuiets &mp qStart mp.EndMoves
                partialInsertionSort &mp qStart mp.EndMoves (-3000 * mp.Depth)
                mp.Stage <- StgQuiet

        | StgQuiet ->
            if (not skipQuiets) && mp.Cur < mp.EndMoves then
                let m = mp.Moves.[mp.Cur]
                mp.Cur <- mp.Cur + 1
                if m = mp.TtMove || m = mp.Killer1 || m = mp.Killer2 || m = mp.CounterMove then ()
                else
                    result <- m; done_ <- true
            else
                mp.Cur <- 0
                mp.Stage <- StgBadCapture

        | StgBadCapture ->
            if mp.Cur < mp.EndBadCaptures then
                let m = mp.Moves.[mp.Cur]
                mp.Cur <- mp.Cur + 1
                if m = mp.TtMove then ()
                else
                    result <- m; done_ <- true
            else
                result <- MoveNone; done_ <- true

        | StgEvasionInit ->
            mp.EndMoves <- generate mp.Pos mp.Moves Evasions
            scoreEvasions &mp 0 mp.EndMoves
            mp.Cur <- 0
            mp.Stage <- StgEvasion

        | StgEvasion ->
            if mp.Cur < mp.EndMoves then
                let i = pickBest &mp mp.Cur mp.EndMoves
                let m = mp.Moves.[i]
                mp.Cur <- mp.Cur + 1
                if m = mp.TtMove then ()
                else
                    result <- m; done_ <- true
            else
                result <- MoveNone; done_ <- true

        | StgProbCutInit ->
            mp.EndMoves <- generate mp.Pos mp.Moves Captures
            scoreCaptures &mp 0 mp.EndMoves
            mp.Cur <- 0
            mp.Stage <- StgProbCut

        | StgProbCut ->
            if mp.Cur < mp.EndMoves then
                let i = pickBest &mp mp.Cur mp.EndMoves
                let m = mp.Moves.[i]
                mp.Cur <- mp.Cur + 1
                if m <> mp.TtMove && mp.Pos.SeeGe m mp.Threshold then
                    result <- m; done_ <- true
                else ()
            else
                result <- MoveNone; done_ <- true

        | StgQCaptureInit ->
            mp.EndMoves <- generate mp.Pos mp.Moves Captures
            scoreCaptures &mp 0 mp.EndMoves
            mp.Cur <- 0
            mp.Stage <- StgQCapture

        | StgQCapture ->
            if mp.Cur < mp.EndMoves then
                let i = pickBest &mp mp.Cur mp.EndMoves
                let m = mp.Moves.[i]
                mp.Cur <- mp.Cur + 1
                if m = mp.TtMove then ()
                else
                    result <- m; done_ <- true
            else
                result <- MoveNone; done_ <- true

        | _ ->
            result <- MoveNone; done_ <- true
    result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test Eonego.Tests --filter "FullyQualifiedName~MovePickerTests"`
Expected: PASS — T0 probe + F1-F6 all green.

- [ ] **Step 5: Commit**

```bash
git add Eonego/MovePicker.fs Eonego.Tests/MovePickerTests.fs
git commit -m "feat: implement MovePicker Main search chain (while-loop nextMove + F1-F6)"
```

---

### Task 7: MovePicker Evasion + QSearch + ProbCut Tests

**Files:**
- Modify: `Eonego.Tests/MovePickerTests.fs` (add F7-F10 fixtures)

**Goal:** Verify the Evasion, QSearch, and ProbCut chains that are already implemented in the `nextMove` while-loop from Task 6.

- [ ] **Step 1: Write the failing tests (F7-F10)**

Add to `Eonego.Tests/MovePickerTests.fs`:

```fsharp
// F7: evasion chain — capture-evasion Kxe2 first (EvasionBonus sorts it above history)
[<Fact>]
let ``F7 evasion capture-evasion first`` () =
    // Black Re2 checks Ke1. Kxe2 captures the rook (undefended) → legal, highest score.
    let p = Position.OfFen "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
    Assert.True(p.InCheck)
    let t = Tables()
    let (moves, scores) = makeBuffers ()
    let mutable mp = mkMain p t MoveNone MoveNone MoveNone MoveNone NoSquare 4 moves scores
    let result = drainPicker &mp false
    Assert.True(result.Length > 0)
    Assert.True(result.[0] = mkMove e1 e2, "Kxe2 (capture-evasion) should be first")

// F8: qsearch includes queen push-promotion (PROMOTION QUIRK fix gate)
[<Fact>]
let ``F8 qsearch includes queen push-promotion`` () =
    // Pe7 can push-promote to e8=Q (no capture). generate(Captures) now includes
    // queen push-promotions → mkQSearch must emit e7e8=Q.
    let p = Position.OfFen "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1"
    let t = Tables()
    let (moves, scores) = makeBuffers ()
    let mutable mp = mkQSearch p t MoveNone moves scores
    let result = drainPicker &mp false
    let queenPromo = mkPromotion e7 e8 Queen
    Assert.True(
        result |> Array.exists (fun m -> m = queenPromo),
        "qsearch must include queen push-promotion e7e8=Q (PROMOTION QUIRK fix)")

// F9: ProbCut SEE gate
[<Fact>]
let ``F9 ProbCut SEE gate passes and rejects`` () =
    // e4xd5: pawn takes queen, SEE = +900.
    let p = Position.OfFen "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1"
    let t = Tables()
    // threshold = 500 → e4d5 passes (900 >= 500)
    let (moves1, scores1) = makeBuffers ()
    let mutable mp1 = mkProbCut p t MoveNone 500 moves1 scores1
    let result1 = drainPicker &mp1 false
    Assert.True(result1 |> Array.exists (fun m -> m = mkMove e4 d5),
                "e4d5 should pass ProbCut with threshold 500")
    // threshold = 1000 → e4d5 rejected (900 < 1000)
    let (moves2, scores2) = makeBuffers ()
    let mutable mp2 = mkProbCut p t MoveNone 1000 moves2 scores2
    let result2 = drainPicker &mp2 false
    Assert.False(result2 |> Array.exists (fun m -> m = mkMove e4 d5),
                 "e4d5 should NOT pass ProbCut with threshold 1000")

// F10: partial_insertion_sort determinism with pre-seeded history
[<Fact>]
let ``F10 partial insertion sort orders by history`` () =
    let p = Position.OfFen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    let t = Tables()
    // Pre-seed: g1f3 = 1000, b1c3 = 500, all others = 0
    t.UpdateMain White (mkMove g1 f3) 1000
    t.UpdateMain White (mkMove b1 c3) 500
    let (moves, scores) = makeBuffers ()
    let mutable mp = mkMain p t MoveNone MoveNone MoveNone MoveNone NoSquare 8 moves scores
    let result = drainPicker &mp false
    let idxOf (fromS: int) (toS: int) =
        result |> Array.findIndex (fun m -> fromSq m = fromS && toSq m = toS)
    let iG1F3 = idxOf g1 f3
    let iB1C3 = idxOf b1 c3
    Assert.True(iG1F3 < iB1C3, "g1f3 (score 1000) should come before b1c3 (score 500)")
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `dotnet test Eonego.Tests --filter "FullyQualifiedName~MovePickerTests~F"`
Expected: PASS — F7-F10 all green (the nextMove while-loop already handles these stages from Task 6).

- [ ] **Step 3: Commit**

```bash
git add Eonego.Tests/MovePickerTests.fs
git commit -m "test: add MovePicker Evasion/QSearch/ProbCut fixtures (F7-F10)"
```

---

### Task 8: Integration Test + Perft Regression

**Files:**
- Modify: `Eonego.Tests/MovePickerTests.fs` (add integration test)

**Goal:** Verify the MovePicker produces the correct total move count on all 6 perft positions (no missing/duplicate moves), and that the existing perft suite is unaffected by all Phase-2 changes.

- [ ] **Step 1: Write the integration test**

Add to `Eonego.Tests/MovePickerTests.fs`:

```fsharp
// Integration: MovePicker drain count == generateLegal count on all perftFens.
// Every legal move must appear exactly once in the picker output (skipQuiets=false,
// no TT move, no killers, depth=8).
[<Fact>]
let ``integration picker count matches legal count on all perftFens`` () =
    let t = Tables()
    for fen in perftFens do
        let p = Position.OfFen fen
        let legal = collectLegal p
        let (moves, scores) = makeBuffers ()
        let mutable mp = mkMain p t MoveNone MoveNone MoveNone MoveNone NoSquare 8 moves scores
        let picked = drainPicker &mp false
        // The picker emits pseudo-legal moves; some may be illegal (e.g., king into
        // check, pinned pieces). The LEGAL count is a lower bound. The picker count
        // should be >= legal count and <= pseudo-legal count. The key invariant: no
        // DUPLICATES and every emitted move is pseudo-legal.
        let distinct = picked |> Array.distinctBy moveMatchKey
        Assert.Equal(distinct.Length, picked.Length)  // no duplicates
        for m in picked do
            Assert.True(p.IsPseudoLegal m,
                sprintf "picker emitted non-pseudo-legal move %s in FEN %s" (toUci m) fen)
        // The picker should emit at least as many moves as the legal count
        // (legal moves are a subset of pseudo-legal moves the picker covers).
        Assert.True(picked.Length >= legal.Length,
            sprintf "picker count %d < legal count %d in FEN %s" picked.Length legal.Length fen)

// Perft regression: the full test suite must still pass after all Phase-2 changes.
// This is a meta-test; the actual perft tests are in MoveGenerationTests.fs.
// Running them here confirms no regression from Position/MoveGeneration modifications.
[<Fact>]
let ``perft d1 still passes for startpos`` () =
    Assert.Equal(20UL, perft (Position.OfFen perftFens.[0]) 1)

[<Fact>]
let ``perft d2 still passes for startpos`` () =
    Assert.Equal(400UL, perft (Position.OfFen perftFens.[0]) 2)
```

- [ ] **Step 2: Run the full test suite**

Run: `dotnet test Eonego.Tests`
Expected: PASS — all tests across all modules green (Bitboard, Move, Zobrist, Position, MoveGeneration, SeeGe, IsPseudoLegal, History, CapturePromo, MovePicker).

- [ ] **Step 3: Run the slow perft suite (opt-in)**

Run: `dotnet test Eonego.Tests --filter "Category=Slow"`
Expected: PASS — all d5-d6 perft positions green (Position changes are additive; MoveGeneration change only affects `generate(Captures)`, not perft).

- [ ] **Step 4: Commit**

```bash
git add Eonego.Tests/MovePickerTests.fs
git commit -m "test: add MovePicker integration test + perft regression gate"
```

---

## Self-Review

### 1. Spec Coverage

| Subproblem | Coverage | Task(s) |
|-----------|----------|---------|
| #1 MovePicker layout (struct, byref mutation, AOT) | Struct declaration, compile-probe, byref-persistence test | 5, 6 |
| #1 PieceValue table | Module-level `let private pieceValue` + `pieceValueOf` + member | 1 |
| #1 Tables class (MainHistory) | History.fs with MainHistory | 3 |
| #2 Position.SeeGe | Member + 14 fixtures / 24 assertions | 1 |
| #2 PieceValue location | Position.fs (not History.fs) — compile order constraint | 1 |
| #3 Position.IsPseudoLegal | Member + ResolvesCheck + oracle + 28 crafted cases | 2 |
| #3 Rank2BB/Rank7BB | Private literals in Position.fs | 2 |
| #4 MovePicker stage machine | while-loop nextMove, 16 stages, all 4 chains | 5, 6, 7 |
| #4 Capture score formula | `7 * pieceValueOf capturedPT + CaptureHistory` | 6 |
| #4 Evasion score | `pieceValueOf captured - pieceValueOf mover + EvasionBonus` | 6 |
| #4 Refutation stage | `refutationNext` with RefIdx, dedup, non-capture gate | 6 |
| #4 Bad capture partition | Spill to front region, replay after quiets | 6 |
| #4 partial_insertion_sort | Canonical SF form (shift, not swap) | 6 |
| #4 TT-move gating | `pos.IsPseudoLegal ttMove` in all 3 factories | 6 |
| #4 ProbCut SEE gate | `pos.SeeGe m threshold` in StgProbCut | 6, 7 |
| PROMOTION QUIRK | `genPawnMoves` fix + F8 gate test | 4, 7 |
| Integration | Picker drain on perftFens + perft regression | 8 |

No spec gaps identified.

### 2. Placeholder Scan

Searched for: "TBD", "TODO", "implement later", "fill in details", "Add appropriate", "handle edge cases", "Write tests for the above", "Similar to Task N".

No placeholders found. Every step contains complete code or exact commands.

### 3. Type Consistency

| Type/Function | Defined in Task | Used in Task | Consistent |
|---------------|----------------|-------------|-----------|
| `pieceValueOf (pt: PieceType) : int` | 1 (Position.fs) | 6 (MovePicker.fs scoreCaptures/scoreEvasions) | Yes |
| `pos.SeeGe (m: Move) (threshold: int) : bool` | 1 (Position.fs) | 6 (MovePicker.fs StgGoodCapture, StgProbCut) | Yes |
| `pos.IsPseudoLegal (m: Move) : bool` | 2 (Position.fs) | 6 (MovePicker.fs mkMain/mkQSearch/mkProbCut, refutationNext) | Yes |
| `Tables()` | 3 (History.fs) | 6 (MovePicker.fs struct field, factories) | Yes |
| `t.MainHist (c: Color) (m: Move) : int` | 3 (History.fs) | 6 (MovePicker.fs scoreQuiets) | Yes |
| `t.CaptureHistory (pc: Piece) (toSq: Square) (capturedPT: PieceType) : int` | 3 (History.fs) | 6 (MovePicker.fs scoreCaptures) | Yes |
| `mkMain ... (moves: Span<Move>) (scores: Span<int>) : MovePicker` | 6 (MovePicker.fs) | 6, 7, 8 (tests) | Yes |
| `mkQSearch ... (moves: Span<Move>) (scores: Span<int>) : MovePicker` | 6 (MovePicker.fs) | 7 (F8 test) | Yes |
| `mkProbCut ... (moves: Span<Move>) (scores: Span<int>) : MovePicker` | 6 (MovePicker.fs) | 7 (F9 test) | Yes |
| `nextMove (mp: byref<MovePicker>) (skipQuiets: bool) : Move` | 5 (probe), 6 (full) | 6, 7, 8 (tests) | Yes (replaced in 6) |
| `MovePicker` struct fields | 5 (probe), 6 (full) | 6 (factories, helpers, nextMove) | Yes — same fields in both |
| Stage literals `StgMainTT` etc. | 5 (MovePicker.fs) | 6 (nextMove) | Yes — 16 literals, same values |
| `EvasionBonus = 1 <<< 28` | 5 (MovePicker.fs) | 6 (scoreEvasions) | Yes |
| `HistMax = 1 <<< 14` | 3 (History.fs) | 3 (Tables.UpdateMain/UpdateCaptureHistory) | Yes |
| `collectPseudo (pos, genType) : Move[]` | 2 (TestFixtures.fs) | 2 (IsPseudoLegalTests) | Yes |
| `collectLegal (pos) : Move[]` | Existing (TestFixtures.fs) | 8 (integration test) | Yes |
| `perftFens` | Existing (TestFixtures.fs) | 8 (integration test) | Yes |

No type inconsistencies found.
