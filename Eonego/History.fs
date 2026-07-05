/// Eonego — per-thread move-ordering tables (butterfly main history, capture history, counter-moves,
/// killers). A single [<Sealed>] mutable CLASS: ONE heap allocation per worker thread, NEVER shared
/// between threads (LazySMP / lockless by construction — there is no module-level mutable state and no
/// lock; each search thread owns its own Tables instance). The MovePick reads/scores through it.
///
/// CONTRACTS:
///  - main          : butterfly main history, flat int16[2*4096] indexed [color<<<12 | fromTo m].
///  - capture       : capture history, flat int16[12*64*8] indexed [((pc*64)+to)<<<3 + capturedPT].
///  - counter       : counter-moves, flat Move[12*64] indexed [prevPc*64 + prevTo].
///  - killers        : per-ply refutation pair, Move[2*MaxPly] indexed [ply*2 + slot] (killers are kept, as in the reference:
///                     in the search Stack; stored here so the per-thread object owns them).
///  - All stats use the "gravity" update: entry += clamp(bonus,-D,D) - entry*abs(clamp)/D. The fixpoint
///    keeps |entry| < D < int16 max, so the int16 store never overflows. The update functions are NOT
///    driven by a search yet (Phase 2 has no search) — they exist + are unit-tested for saturation.
///  - Continuation history and QuietChecks are deliberately out of scope (a later phase).
module Eonego.History

open System
open System.Runtime.CompilerServices
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position

// Gravity divisors (also the bonus clamp bound). All < int16 max (32767) so the store can't overflow —
// the contract is enforced by the Tunables clamps. Names kept so consumers/tests compile unchanged;
// values now come from the tuning-campaign env statics (defaults identical to the old literals).
let MainHistD = Tunables.MainHistD // butterfly (main) history divisor
let CaptureHistD = Tunables.CaptureHistD // capture history divisor
let ContHistD = Tunables.ContHistD // continuation history divisor
let CorrHistD = Tunables.CorrHistD // correction-history divisor

/// Stat bonus as a function of depth (shape tunable via EONEGO_T_STATB_*).
let inline statBonus (depth: int) : int =
    min (Tunables.StatBonusMul * depth - 100) Tunables.StatBonusCap

[<Sealed>]
type Tables() =
    // [color<<<12 | fromTo(12-bit)] -> int16 saturating main (butterfly) history.
    let main: int16[] = Array.zeroCreate (2 * 4096)
    // [((pc*64)+to)<<<3 + capturedPT] -> int16 capture history. pc 0..11, to 0..63, capturedPT 0..4 (8-wide slot).
    let capture: int16[] = Array.zeroCreate (12 * 64 * 8)
    // [prevPc*64 + prevTo] -> the move that refuted the previous move.
    let counter: Move[] = Array.create (12 * 64) MoveNone
    // [ply*2 + slot] -> killer pair for that ply.
    let killers: Move[] = Array.create (2 * MaxPly) MoveNone
    // Continuation history: int16[768*768] indexed [(prevPc*64+prevTo)*768 + (pc*64+to)]. cont1 keyed by the
    // 1-ply-back move (ss-1), cont2 by the 2-ply-back move (ss-2). 768 = 12 pieces * 64 squares.
    let cont1: int16[] = Array.zeroCreate (768 * 768)
    let cont2: int16[] = Array.zeroCreate (768 * 768)
    // cont4: keyed by the 4-ply-back move (ss-4, same side to move) — the "deepened" continuation
    // table. Read WEIGHTED (Tunables.Cont4Div) in the LMR history term only; taught the full bonus.
    // Lazily allocated (EnsureAux): UseCont4 defaults OFF and this is 1.125 MiB per worker that the
    // default config never touches. Readers/writers stay unguarded — they are cfg-gated in Search,
    // and Worker.SetupRoot ensures allocation before any search that can reach them.
    let mutable cont4: int16[] = Array.empty
    // Correction history: [stm<<<14 | pawnKey&16383] -> persistent (bestValue - staticEval) error for this
    // side's pawn structure, gravity-updated. Read as a static-eval correction wherever eval feeds pruning.
    let corr: int16[] = Array.zeroCreate (2 * 16384)
    // Minor-piece correction history: same contract, keyed by Position.MinorKey (knights+bishops+kings).
    // Lazily allocated like cont4 (UseCorrMinor defaults OFF; 64 KiB per worker).
    let mutable corrMinor: int16[] = Array.empty
    // Major-piece correction history: keyed by Position.MajorKey (rooks+queens).
    let mutable corrMajor: int16[] = Array.empty
    // Non-pawn correction history: keyed by Position.NonPawnKey (all non-pawn pieces).
    let mutable corrNonPawn: int16[] = Array.empty
    // Continuation correction history: keyed by previous move (prevPc*64+prevTo), 2 sides.
    let corrCont: int16[] = Array.zeroCreate (2 * 768)

    /// Allocate the config-gated tables this search will actually use (idempotent; called by
    /// Worker.SetupRoot with the active config flags before the search starts).
    member _.EnsureAux (useCont4: bool) (useCorrMinor: bool) (useCorrMajor: bool) (useCorrNonPawn: bool) : unit =
        if useCont4 && cont4.Length = 0 then
            cont4 <- Array.zeroCreate (768 * 768)

        if useCorrMinor && corrMinor.Length = 0 then
            corrMinor <- Array.zeroCreate (2 * 16384)

        if useCorrMajor && corrMajor.Length = 0 then
            corrMajor <- Array.zeroCreate (2 * 16384)

        if useCorrNonPawn && corrNonPawn.Length = 0 then
            corrNonPawn <- Array.zeroCreate (2 * 16384)

    /// Zero every table (new game / clear between searches).
    member _.Clear() : unit =
        Array.Clear(main, 0, main.Length)
        Array.Clear(capture, 0, capture.Length)
        Array.Fill(counter, MoveNone)
        Array.Fill(killers, MoveNone)
        Array.Clear(cont1, 0, cont1.Length)
        Array.Clear(cont2, 0, cont2.Length)
        Array.Clear(cont4, 0, cont4.Length)
        Array.Clear(corr, 0, corr.Length)
        Array.Clear(corrMinor, 0, corrMinor.Length)
        Array.Clear(corrMajor, 0, corrMajor.Length)
        Array.Clear(corrNonPawn, 0, corrNonPawn.Length)
        Array.Clear(corrCont, 0, corrCont.Length)

    /// Worker pool, between MOVES of one game: drop the per-search hint moves (killers are ply-indexed
    /// and ply meanings shift; counters conservatively too) but keep every gravity table — warm
    /// butterfly/capture/cont/corr history is the point of pooling.
    member _.NewSearch() : unit =
        Array.Fill(counter, MoveNone)
        Array.Fill(killers, MoveNone)

    // --- reads (AggressiveInlining ATTRIBUTE: they touch the private arrays yet inline in-assembly) ---
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.MainHistory (c: Color) (fromToKey: int) : int = int main.[(c <<< 12) ||| fromToKey]

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.CaptureHistory (pc: Piece) (dst: Square) (capturedPT: PieceType) : int =
        int capture.[((pc * 64 + dst) <<< 3) + capturedPT]

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.CounterMove (prevPc: Piece) (prevTo: Square) : Move = counter.[prevPc * 64 + prevTo]

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.Killer (ply: int) (slot: int) : Move = killers.[ply * 2 + slot]

    // Continuation-history reads. Callers pass a `-1` prevPc sentinel (root / after null / NoPiece) -> 0.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.ContHistory1 (prevPc: int) (prevTo: int) (pc: Piece) (dst: Square) : int =
        if prevPc < 0 then
            0
        else
            int cont1.[(prevPc * 64 + prevTo) * 768 + (pc * 64 + dst)]

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.ContHistory2 (prevPc: int) (prevTo: int) (pc: Piece) (dst: Square) : int =
        if prevPc < 0 then
            0
        else
            int cont2.[(prevPc * 64 + prevTo) * 768 + (pc * 64 + dst)]

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.ContHistory4 (prevPc: int) (prevTo: int) (pc: Piece) (dst: Square) : int =
        if prevPc < 0 then
            0
        else
            int cont4.[(prevPc * 64 + prevTo) * 768 + (pc * 64 + dst)]

    /// Raw correction-history entry for (side to move, pawn structure); callers scale (/16) into eval units.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.CorrHist (c: Color) (pawnKey: uint64) : int =
        int corr.[(c <<< 14) ||| int (pawnKey &&& 16383UL)]

    /// Raw minor-piece correction-history entry for (side to move, minor structure); same scaling contract.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.CorrHistMinor (c: Color) (minorKey: uint64) : int =
        int corrMinor.[(c <<< 14) ||| int (minorKey &&& 16383UL)]

    /// Raw major-piece correction-history entry for (side to move, major structure); same scaling contract.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.CorrHistMajor (c: Color) (majorKey: uint64) : int =
        int corrMajor.[(c <<< 14) ||| int (majorKey &&& 16383UL)]

    /// Raw non-pawn correction-history entry for (side to move, non-pawn structure); same scaling contract.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.CorrHistNonPawn (c: Color) (nonPawnKey: uint64) : int =
        int corrNonPawn.[(c <<< 14) ||| int (nonPawnKey &&& 16383UL)]

    /// Raw continuation correction-history entry for (side to move, previous move); same scaling contract.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member _.CorrHistCont (c: Color) (prevPc: int) (prevTo: int) : int =
        if prevPc < 0 then 0
        else int corrCont.[(c <<< 10) ||| (prevPc * 64 + prevTo)]

    // --- writes -----------------------------------------------------------------------------------
    member _.SetCounter (prevPc: Piece) (prevTo: Square) (m: Move) : unit = counter.[prevPc * 64 + prevTo] <- m

    /// Record a killer for `ply` (slide slot0 -> slot1; ignore a duplicate so the pair stays distinct).
    member _.SetKiller (ply: int) (m: Move) : unit =
        let i = ply * 2

        if killers.[i] <> m then
            killers.[i + 1] <- killers.[i]
            killers.[i] <- m

    /// Gravity update of main history (saturates within int16 toward +/-MainHistD).
    member _.UpdateMain (c: Color) (m: Move) (bonus: int) : unit =
        let i = (c <<< 12) ||| (fromTo m)
        let b = max -MainHistD (min MainHistD bonus)
        let v = int main.[i]
        main.[i] <- int16 (v + b - v * (abs b) / MainHistD)

    /// Gravity update of capture history.
    member _.UpdateCapture (pc: Piece) (dst: Square) (capturedPT: PieceType) (bonus: int) : unit =
        let i = ((pc * 64 + dst) <<< 3) + capturedPT
        let b = max -CaptureHistD (min CaptureHistD bonus)
        let v = int capture.[i]
        capture.[i] <- int16 (v + b - v * (abs b) / CaptureHistD)

    /// Gravity update of 1-ply continuation history (no-op when prevPc < 0).
    member _.UpdateCont1 (prevPc: int) (prevTo: int) (pc: Piece) (dst: Square) (bonus: int) : unit =
        if prevPc >= 0 then
            let i = (prevPc * 64 + prevTo) * 768 + (pc * 64 + dst)
            let b = max -ContHistD (min ContHistD bonus)
            let v = int cont1.[i]
            cont1.[i] <- int16 (v + b - v * (abs b) / ContHistD)

    /// Gravity update of correction history (bonus pre-clamped by the caller; re-clamped for safety).
    member _.UpdateCorr (c: Color) (pawnKey: uint64) (bonus: int) : unit =
        let i = (c <<< 14) ||| int (pawnKey &&& 16383UL)
        let b = max -CorrHistD (min CorrHistD bonus)
        let v = int corr.[i]
        corr.[i] <- int16 (v + b - v * (abs b) / CorrHistD)

    /// Gravity update of minor-piece correction history.
    member _.UpdateCorrMinor (c: Color) (minorKey: uint64) (bonus: int) : unit =
        let i = (c <<< 14) ||| int (minorKey &&& 16383UL)
        let b = max -CorrHistD (min CorrHistD bonus)
        let v = int corrMinor.[i]
        corrMinor.[i] <- int16 (v + b - v * (abs b) / CorrHistD)

    /// Gravity update of major-piece correction history.
    member _.UpdateCorrMajor (c: Color) (majorKey: uint64) (bonus: int) : unit =
        let i = (c <<< 14) ||| int (majorKey &&& 16383UL)
        let b = max -CorrHistD (min CorrHistD bonus)
        let v = int corrMajor.[i]
        corrMajor.[i] <- int16 (v + b - v * (abs b) / CorrHistD)

    /// Gravity update of non-pawn correction history.
    member _.UpdateCorrNonPawn (c: Color) (nonPawnKey: uint64) (bonus: int) : unit =
        let i = (c <<< 14) ||| int (nonPawnKey &&& 16383UL)
        let b = max -CorrHistD (min CorrHistD bonus)
        let v = int corrNonPawn.[i]
        corrNonPawn.[i] <- int16 (v + b - v * (abs b) / CorrHistD)

    /// Gravity update of continuation correction history (no-op when prevPc < 0).
    member _.UpdateCorrCont (c: Color) (prevPc: int) (prevTo: int) (bonus: int) : unit =
        if prevPc >= 0 then
            let i = (c <<< 10) ||| (prevPc * 64 + prevTo)
            let b = max -CorrHistD (min CorrHistD bonus)
            let v = int corrCont.[i]
            corrCont.[i] <- int16 (v + b - v * (abs b) / CorrHistD)

    /// Gravity update of 2-ply continuation history (no-op when prevPc < 0).
    member _.UpdateCont2 (prevPc: int) (prevTo: int) (pc: Piece) (dst: Square) (bonus: int) : unit =
        if prevPc >= 0 then
            let i = (prevPc * 64 + prevTo) * 768 + (pc * 64 + dst)
            let b = max -ContHistD (min ContHistD bonus)
            let v = int cont2.[i]
            cont2.[i] <- int16 (v + b - v * (abs b) / ContHistD)

    /// Gravity update of 4-ply continuation history (no-op when prevPc < 0).
    member _.UpdateCont4 (prevPc: int) (prevTo: int) (pc: Piece) (dst: Square) (bonus: int) : unit =
        if prevPc >= 0 then
            let i = (prevPc * 64 + prevTo) * 768 + (pc * 64 + dst)
            let b = max -ContHistD (min ContHistD bonus)
            let v = int cont4.[i]
            cont4.[i] <- int16 (v + b - v * (abs b) / ContHistD)
