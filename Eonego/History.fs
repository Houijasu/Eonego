/// Eonego — per-thread move-ordering tables (butterfly main history, capture history, counter-moves,
/// killers). A single [<Sealed>] mutable CLASS: ONE heap allocation per worker thread, NEVER shared
/// between threads (LazySMP / lockless by construction — there is no module-level mutable state and no
/// lock; each search thread owns its own Tables instance). The MovePick reads/scores through it.
///
/// CONTRACTS:
///  - main          : butterfly main history, flat int16[2*4096] indexed [color<<<12 | fromTo m].
///  - capture       : capture history, flat int16[12*64*8] indexed [((pc*64)+to)<<<3 + capturedPT].
///  - counter       : counter-moves, flat Move[12*64] indexed [prevPc*64 + prevTo].
///  - killers        : per-ply refutation pair, Move[2*MaxPly] indexed [ply*2 + slot] (SF keeps killers
///                     in the search Stack; stored here so the per-thread object owns them).
///  - All stats use SF's "gravity" update: entry += clamp(bonus,-D,D) - entry*abs(clamp)/D. The fixpoint
///    keeps |entry| < D < int16 max, so the int16 store never overflows. The update functions are NOT
///    driven by a search yet (Phase 2 has no search) — they exist + are unit-tested for saturation.
///  - Continuation history and QuietChecks are deliberately out of scope (a later phase).
module Eonego.History

open System
open System.Runtime.CompilerServices
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position

// SF gravity divisors (also the bonus clamp bound). Both < int16 max (32767) so the store can't overflow.
[<Literal>]
let MainHistD = 7183 // SF ButterflyHistory

[<Literal>]
let CaptureHistD = 10692 // SF CapturePieceToHistory

[<Literal>]
let ContHistD = 29952 // SF PieceToHistory (continuation); < int16 max so the gravity store can't overflow

/// Stat bonus as a function of depth (representative SF shape; tunable when the search lands).
let inline statBonus (depth: int) : int = min (160 * depth - 100) 1700

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

    /// Zero every table (new game / clear between searches).
    member _.Clear() : unit =
        Array.Clear(main, 0, main.Length)
        Array.Clear(capture, 0, capture.Length)
        Array.Fill(counter, MoveNone)
        Array.Fill(killers, MoveNone)
        Array.Clear(cont1, 0, cont1.Length)
        Array.Clear(cont2, 0, cont2.Length)

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

    // --- writes -----------------------------------------------------------------------------------
    member _.SetCounter (prevPc: Piece) (prevTo: Square) (m: Move) : unit = counter.[prevPc * 64 + prevTo] <- m

    /// Record a killer for `ply` (slide slot0 -> slot1; ignore a duplicate so the pair stays distinct).
    member _.SetKiller (ply: int) (m: Move) : unit =
        let i = ply * 2

        if killers.[i] <> m then
            killers.[i + 1] <- killers.[i]
            killers.[i] <- m

    /// SF gravity update of main history (saturates within int16 toward +/-MainHistD).
    member _.UpdateMain (c: Color) (m: Move) (bonus: int) : unit =
        let i = (c <<< 12) ||| (fromTo m)
        let b = max -MainHistD (min MainHistD bonus)
        let v = int main.[i]
        main.[i] <- int16 (v + b - v * (abs b) / MainHistD)

    /// SF gravity update of capture history.
    member _.UpdateCapture (pc: Piece) (dst: Square) (capturedPT: PieceType) (bonus: int) : unit =
        let i = ((pc * 64 + dst) <<< 3) + capturedPT
        let b = max -CaptureHistD (min CaptureHistD bonus)
        let v = int capture.[i]
        capture.[i] <- int16 (v + b - v * (abs b) / CaptureHistD)

    /// SF gravity update of 1-ply continuation history (no-op when prevPc < 0).
    member _.UpdateCont1 (prevPc: int) (prevTo: int) (pc: Piece) (dst: Square) (bonus: int) : unit =
        if prevPc >= 0 then
            let i = (prevPc * 64 + prevTo) * 768 + (pc * 64 + dst)
            let b = max -ContHistD (min ContHistD bonus)
            let v = int cont1.[i]
            cont1.[i] <- int16 (v + b - v * (abs b) / ContHistD)

    /// SF gravity update of 2-ply continuation history (no-op when prevPc < 0).
    member _.UpdateCont2 (prevPc: int) (prevTo: int) (pc: Piece) (dst: Square) (bonus: int) : unit =
        if prevPc >= 0 then
            let i = (prevPc * 64 + prevTo) * 768 + (pc * 64 + dst)
            let b = max -ContHistD (min ContHistD bonus)
            let v = int cont2.[i]
            cont2.[i] <- int16 (v + b - v * (abs b) / ContHistD)
