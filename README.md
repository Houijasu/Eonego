# Eonego

A from-scratch chess engine written in **F#** (.NET 10), built for speed and verified for correctness. Eonego runs a **Monte-Carlo Tree Search at the root with a fixed-depth alpha-beta / NNUE leaf** — a true MCTS + αβ hybrid — on top of a hand-tuned bitboard representation: PEXT/magic sliders, a Stockfish-style staged move picker, alpha-beta / PVS with LazySMP and aggressive forward pruning (ProbCut, log-based LMR, null-move, futility/SEE), and a lock-free transposition table — all allocation-free on the hot path and validated against published perft node counts and an in-test minimax oracle. The leaf eval is a from-scratch port of **Stockfish-16 NNUE** (embedded, incrementally-updated, AVX2); MCTS priors come from a from-scratch **Lc0 `20x256SE` CNN** (or a history-softmax fallback when no Lc0 net is present).

- **Author:** Houijasu
- **Language:** F# on `net10.0`
- **Interface:** UCI (Universal Chess Interface) — drop it into any UCI GUI or drive it from the console
- **Target:** Native AOT, speed-tuned for x86-64 with BMI2/PEXT

---

## Table of contents

- [Feature highlights](#feature-highlights)
- [Architecture at a glance](#architecture-at-a-glance)
- [Bitboard foundation](#bitboard-foundation)
- [Position and move representation](#position-and-move-representation)
- [Move generation (the perft gate)](#move-generation-the-perft-gate)
- [Evaluation](#evaluation)
- [Search](#search)
- [Transposition table](#transposition-table)
- [Move ordering](#move-ordering)
- [UCI interface](#uci-interface)
- [Build and run](#build-and-run)
- [Testing and verification](#testing-and-verification)
- [Benchmarks](#benchmarks)
- [Project structure](#project-structure)
- [Design notes and non-goals](#design-notes-and-non-goals)
- [License](#license)

---

## Feature highlights

**Board representation**
- LERF (Little-Endian Rank-File) bitboards with wrap-safe directional shifts
- Hardware-capability snapshot: POPCNT, BMI1, Bmi2/PEXT, LZCNT, AVX2, x64
- Leaper attack tables (pawn / knight / king) precomputed at init
- Sliders via **fancy magic bitboards**, with a **BMI2 PEXT** fast path selected at JIT time (zero-cost branch fold)
- Magics are *generated deterministically* at startup (fixed-seed xorshift64) and validated against a classical ray-cast oracle — no embedded hex constants, AOT-clean
- Geometry tables: `Between`, `Line`, `ForwardRanks`, diagonal/anti-diagonal masks

**Position**
- Single mutable `Position` class reused via `Make`/`Unmake` (make/unmake, **not** copy-make)
- Stockfish-style piece boards: `byTypeBB[0..6]` + `byColorBB[2]` + a `Piece[64]` mailbox
- Irreversible/undo state in a preallocated `StateInfo[MaxPly]` stack (no per-node allocation)
- Incremental Zobrist key maintained across make/unmake/null-move
- Full FEN parse + export, including the SF en-passant capturer gate
- **SEE** (`see_ge`) static-exchange evaluation, faithful to Stockfish's swap loop
- `GivesCheck`, `IsPseudoLegal`, slider blockers/pinners, cached check-info

**Move generation**
- Staged, allocation-free pseudo-legal generator writing into caller-owned `stackalloc Span<Move>`
- All move kinds: normal, captures, double pushes, en passant, promotions (Q/R/B/N split per gen type), castling
- Legality filter using cached pin/check state (en-passant discovered-check trap handled explicitly)
- `perft` + `perftDivide` driver; node counts pinned to the canonical CPW positions 1–6

**Evaluation**
- Two evaluators: a fast **material-only** static eval (the NNUE-off default / fallback) and an **NNUE** network eval
- **NNUE**: a custom *region-piececount* net (`2577 → 64 → 32 → 16 → 16 → 1`, int8/int16 quantized) with an **incremental first-layer accumulator** maintained across make/unmake/null — bit-exact int8 inference on both AVX2 and scalar paths, ~5–10× faster than recomputing, 0 B/op
- Both are 0 B/op and thread-safe (read-only weights), white-relative with a negamax sign flip

**Search**
- Alpha-beta / **PVS**, fail-soft, with aspiration iterative deepening
- **LazySMP**: N independent iterative-deepening workers, each with its own `Position` + history tables + stack/PV/buffers; the only shared writable state is the lock-free TT plus an atomic stop flag
- Forward pruning: mate-distance, reverse-futility, null-move (+ optional zugzwang verification), **ProbCut**, **log-based late-move reductions (LMR)**, move-count (late-move) pruning, futility pruning, SEE pruning of losing quiets/captures, **history pruning** of late quiet losers, and **delta pruning** in qsearch — all scaled by an `improving` signal
- **Continuation history** (1-/2-ply), **singular / double extensions**, a former-PV `ttPv` flag, and **Cut/All/PV node-type** threading feeding the LMR / extension / pruning decisions
- **Internal iterative reductions (IIR)** when no TT move is available to improve ordering
- **Razoring**: qsearch verification of shallow positions where static eval is far below alpha
- Check + singular extensions, quiescence search with SEE-based capture pruning
- Triangular PV, `info` lines with `depth/seldepth/score/nodes/nps/time/pv` (aggregate node/nps across all workers)
- Time management: `movetime`, `wtime`/`winc`/`btime`/`binc`/`movestogo`, `depth`, `nodes`, `mate`, `infinite` — soft + hard time stops
- `UsePruning = false` collapses to plain full-window alpha-beta — the built-in correctness oracle

**Transposition table**
- Lock-free **XOR-lockless** (Hyatt) 16-byte entries, 4-entry cache-line clusters
- `Data` packing: `move:16 | score:int16 | eval:int16 | depth:uint8 | genBound:uint8` (`genBound = generation5 | ttPv | bound`)
- Mate-ply-corrected scores (caller-side `valueToTt`/`valueFromTt`), 5-bit generation aging, a sticky `ttPv` flag, depth-and-age replacement

**Move ordering**
- Stockfish-style staged **MovePick**: TT move → good captures (SEE-split) → refutations (killers + countermove) → quiets → bad captures; separate evasion and ProbCut/qsearch chains
- Per-thread history tables: butterfly main history, capture history, counter-moves, killer pairs
- SF "gravity" saturating stat updates; `partialInsertionSort` for quiets

**Tooling**
- xUnit test suite: perft gate, SEE, pseudo-legality, promotion split, history, move picker, evaluation, transposition (incl. torn-read injection), draw detection, search oracle, LazySMP
- BenchmarkDotNet harness covering bitloops, attack lookups, move encode/decode, movegen, MovePick, eval, and fixed-depth search — all targeting 0 B/op steady state

---

## Architecture at a glance

The solution (`Eonego.slnx) contains three projects:

| Project | Type | Role |
|---|---|---|
| `Eonego` | Console exe (`net10.0`) | The engine |
| `Eonego.Tests` | xUnit | Correctness suite, perft gate, search oracle |
| `Eonego.Benchmarks` | BenchmarkDotNet | Microbenchmarks + allocation profiling |

The engine is compiled as a single ordered module chain (compile order is load-bearing — each module builds on the previous):

```
Bitboard → Move → Zobrist → NnueRegions → Position → Evaluation → NnueNetwork → Nnue
        → MoveGeneration → History → MovePick → Transposition → Search → Tooling → Uci → Program
```

| Module | Responsibility |
|---|---|
| [`Bitboard.fs`](Eonego/Bitboard.fs) | LERF bitboards, hardware caps, leaf bit ops, geometry tables, classical + magic + PEXT sliders, leaper tables |
| [`Move.fs`](Eonego/Move.fs) | 16-bit packed move encoding, accessors, predicates, TT compaction, UCI parse/format |
| [`Zobrist.fs`](Eonego/Zobrist.fs) | Fixed-seed Zobrist key tables (Psq, castling rights, en-passant file, side) |
| [`NnueRegions.fs`](Eonego/NnueRegions.fs) | Region-feature map (204 k×k regions × 12 channels) + incremental accumulator primitives (compiles before `Position`) |
| [`Position.fs`](Eonego/Position.fs) | Mutable board state, make/unmake/null, FEN, SEE, check-info, pseudo-legality, NNUE region + L1 accumulators |
| [`Evaluation.fs`](Eonego/Evaluation.fs) | Material-only static eval (NNUE-off default / fallback) |
| [`NnueNetwork.fs`](Eonego/NnueNetwork.fs) | NNUE loader + quantized int8/int16 forward kernels (AVX2 + scalar) + incremental L1 forward |
| [`Nnue.fs`](Eonego/Nnue.fs) | NNUE eval seam — feature assembly / maintained-accumulator read, clamp + negamax sign |
| [`MoveGeneration.fs`](Eonego/MoveGeneration.fs) | Staged allocation-free generator, legality filter, perft |
| [`History.fs`](Eonego/History.fs) | Per-thread move-ordering tables (main/capture history, killers, counters) |
| [`MovePick.fs`](Eonego/MovePick.fs) | Stockfish-style staged lazy move picker |
| [`Transposition.fs`](Eonego/Transposition.fs) | Lock-free XOR-lockless transposition table |
| [`Search.fs`](Eonego/Search.fs) | PVS / alpha-beta, LazySMP, iterative deepening, pruning, time budget |
| [`Tooling.fs`](Eonego/Tooling.fs) | Offline CLI subcommands: `gen` (self-play data), `featuredump`, `nnforward` (training / parity) |
| [`Uci.fs`](Eonego/Uci.fs) | UCI driver (master search thread, console I/O) |
| [`Program.fs`](Eonego/Program.fs) | Entry point: bitboard warmup + `Uci.run()` |

---

## Bitboard foundation

`Bitboard.fs` is the performance bedrock. Squares use **LERF** indexing (`a1 = 0`, `h8 = 63`), so `north = bb <<< 8` and `rank = sq >>> 3`, `file = sq &&& 7`.

- **Leaf bit ops** (`popCount`, `lsb`, `msb`, `popLsb`) lower to `POPCNT` / `TZCNT` / `LZCNT` / `BLSR` on x64, with correct `BitOperations` software fallbacks.
- **Leaper tables** (pawn / knight / king) are occupancy-independent, indexed by square (pawn attacks by color).
- **Sliders** have three implementations:
  1. **Classical** ray-cast (obstruction difference) — the oracle and the table builder.
  2. **Fancy magic bitboards** — magics found by a deterministic fixed-seed sparse search and validated against the classical oracle; flat tables with per-square offsets (rook 102400, bishop 5248 entries).
  3. **BMI2 PEXT** tables — built when `Bmi2.X64.IsSupported`; `ParallelBitExtract` compresses the masked occupancy into a dense index.
- **Unified dispatch**: `rookAttacks`/`bishopAttacks` select PEXT vs magic via `Bmi2.X64.IsSupported`, which the JIT/AOT folds to a constant — the dead branch is eliminated, so selection is zero-cost. `usesPext` reports the active path.

Hardware capability is read once (cold path) into a `Caps` struct; the hot path never re-queries it.

---

## Position and move representation

### Move (`Move.fs`)

A move is a 16-bit payload carried in a 32-bit `int` working register:

```
bit: 15 14 | 13 12 | 11 10 09 08 07 06 | 05 04 03 02 01 00
      flag  | promo |        from        |        to

flag:  0=Normal 1=Promotion 2=EnPassant 3=Castling
promo: 0=N 1=B 2=R 3=Q   (only meaningful when flag=Promotion)
```

Sentinels: `MoveNone = 0`, `MoveNull = 0x41` (both have `from == to`, rejected by `isOk`). Two derived keys serve two jobs: `fromTo` (12-bit, butterfly history index) and `moveMatchKey` (14-bit, UCI legal-list re-stamp key that disambiguates under-promotions). `packed16`/`ofPacked` decouple the 16-bit TT storage width from the 32-bit working type.

### Position (`Position.fs`)

`Position` is a single `[<Sealed>]` mutable class — one heap allocation, reused across the entire search via `Make`/`Unmake`. It owns:

- `byTypeBB[0..5]` (Pawn..King) plus `byTypeBB[6]` = full occupancy
- `byColorBB[2]`
- a `Piece[64]` mailbox for O(1) `pieceOn`
- a preallocated `StateInfo[MaxPly]` (`MaxPly = 1024`) undo stack holding the irreversible/derived undo data and cached check-info (checkers, blockers, pinners, check squares)

All board mutation flows through three choke points (`PutPiece`/`RemovePiece`/`MovePiece`) that also maintain the incremental Zobrist key. `Unmake` uses key-less siblings (`*NK`) for pure board restore — zero key-drift risk; the parent frame holds the key. `MakeNull`/`UnmakeNull` implement null moves (precondition: not in check).

The material table `pieceValue {100, 320, 330, 500, 900, 0}` (Pawn..King) is the single source of truth for **SEE and move ordering**; King = 0 is the SEE sentinel. The material-only `Evaluation.fs` sums piece values; all positional understanding lives in the NNUE (the former PeSTO tables were removed).

`Position` also implements `SeeGe` (faithful swap-loop), `GivesCheck` (direct, discovered, and special-move post-occupancy cases), `IsPseudoLegal` (hand-rolled from Position accessors so the MovePick can emit a TT/killer/counter move without generating), and `AttackersTo(occ)`.

---

## Move generation (the perft gate)

`MoveGeneration.fs` is a Stockfish-style pseudo-legal generator plus a cheap legality filter. It is **allocation-free**: every move buffer is a caller-owned `stackalloc Span<Move>` living on the consuming thread's stack. There is no module-level mutable state — LazySMP-safe by construction.

- `GenType`: `Captures | Quiets | QuietChecks | Evasions | NonEvasions | Legal` (literals, so call sites pass a compile-time constant)
- `MaxMoves = 256`
- Per-piece generators for pawn (pushes, double pushes, captures, push/capture-promotions, en passant), knight, bishop, rook, queen, king, plus castling with a full safe-path + rook-presence guard
- The promotion split (`addPromotionsGen`) routes queen promotions as capture-class (so qsearch sees them) and splits under-promotions by capture vs quiet — while `Evasions`/`NonEvasions` still emit all four (perft-preserving)
- `isLegal` handles the three cases that need more than shape: en passant (rebuilds occupancy with both pawns gone to catch the discovered-check trap), king moves (X-ray through the vacated square), and pinned pieces (must move along the pin ray)
- `generateLegal` does an in-place 0-alloc compaction with the SF fast-path (only EP, king moves, and pinned pieces pay the `isLegal` test)
- `perft` recurses fully to depth 0 so every move type's `Make`/`Unmake` is exercised at the leaf; `perftDivide` localizes mismatches per root move

---

## Evaluation

The evaluation is **Stockfish-16 NNUE** (HalfKAv2_hm), ported from scratch and the **sole, mandatory** evaluator — there is no PeSTO/material fallback and no custom net.

### NNUE (`SfAccumulator.fs` → `SfNnue.fs`)

`SfNnue.evalCp` runs the SF feature transformer + layer stack behind the one negamax-signed seam. The perspective accumulator is maintained **incrementally** across `Make`/`Unmake` (per-ply snapshot, dirty-piece deltas, king-move full refresh), so eval reads the maintained accumulator instead of recomputing. Kernels are AVX2 (`vpmaddubsw`/`vpmaddwd`) with a bit-identical scalar fallback (`EONEGO_FORCE_SCALAR=1`). The `sf16.nnue` net is **embedded in the binary** (self-contained exe); the engine refuses to search without it. Checkmate / stalemate / repetition / 50-move / insufficient material remain the **search's** concern, never eval's.

---

## Search — MCTS root + alpha-beta leaf

The production search is a **true hybrid**: a Monte-Carlo Tree Search at the root (`Mcts.fs`, Baier–Winands MCTS-MB) whose every leaf is evaluated by the fixed-depth **alpha-beta / SF-NNUE** search described below (reused unchanged).

- **Priors** come from a from-scratch **Lc0 `20x256SE` CNN** (`Lc0Net.fs` / `Lc0Proto.fs` / `Lc0Encoder.fs`) when a `.pb` net sits beside the exe or `EONEGO_LC0` points at one; otherwise a history-softmax fallback.
- **MCTS-Solver** proves win/loss/draw **with mate distance** — it prefers the shortest forced mate and the longest defence, and reports the true `mate <n>`.
- **Selection** is PUCT with **adaptive cpuct** (grows with parent visits) and a **parent-Q FPU** for unvisited children.
- **Root-parallel** across `Threads` (N independent trees sharing only the lock-free TT + atomic stop flag, merged into one decision); `go depth N` is a **global** `N × 1000`-iteration budget split across the workers. A root too deep for a fixed-depth leaf falls back to plain alpha-beta.

### Alpha-beta leaf (`Search.fs`)

`Search.fs` is a fail-soft **alpha-beta / PVS** search with **LazySMP** orchestration — it is both the MCTS leaf evaluator and the standalone fallback engine.

### Workers and shared state

Each worker is a `Worker` object owning its own `Position`, `History.Tables`, search `Stack`, triangular PV, and preallocated move/score/quiet buffers — rebuilt from the immutable root at the start of each `go()`. The only cross-thread writable state is the lock-free TT and an atomic `Volatile` stop flag on `SearchControl`. There is no locking on the hot path.

### The search loop

- **Iterative deepening** with **aspiration windows** (full window for depth <= 4 or near-mate; re-searches widen by doubling `delta` on fail-low/fail-high)
- **Negamax / PVS**: first move is searched full window; subsequent moves get a zero-window scout search at reduced depth (LMR), re-searched at full depth and then full window only if they beat alpha
- **Internal iterative reductions (IIR)**: when no TT move is available at an interior node, reduce the searched depth by one ply to let the next iteration seed better move ordering
- **Quiescence** (`qsearch`): fail-soft leaves with stand-pat, SEE-based capture pruning, **delta pruning** of captures that cannot lift alpha even after winning the piece, and mate scoring when in check with no moves
- **Pruning** (gated by `UsePruning`, non-PV, not in check):
  - Mate-distance pruning
  - Reverse futility (`staticEval - 120*depth >= beta`, `depth <= 6`)
  - **Razoring** (`depth <= 3`): qsearch verification when static eval is far below alpha; fail low if captures cannot lift alpha
  - Null-move pruning (`depth >= 3`, `staticEval >= beta`, non-pawn material; `R = 3 + depth/4 + (eval≫beta)`), with optional **zugzwang verification** (`UseNmpVerify`): at `depth >= 12` a fail-high is re-searched on the original position with NMP suppressed and only trusted if it also fails high
  - **ProbCut** (`depth >= 5`): a strong-enough capture (SEE) that *holds* a reduced `negamax(depth-4)` above `beta + margin` is a cutoff — verified by qsearch then the reduced search, stored as a `BoundLower` at `depth-3`
  - **LMR**: a log-based reduction table `r[depth][moveCount]`, deeper for non-improving / quiet moves, lighter for captures; with `UseLmrTweaks` also reduced less at PV / TT-move / former-PV (`ttPv`) nodes, more at expected cut-nodes, and scaled by the combined main + continuation history; zero-window re-search on fail-high
  - Move-count (late-move) pruning, futility pruning of late quiets, SEE pruning of losing quiets / shallow losing captures, and **history pruning** of late quiet moves with strongly negative butterfly history
- **Extensions**: check giving + SEE >= 0 → +1 ply; plus optional **singular / double extensions** (`UseSingular`, `depth >= 8`): when the TT move is a depth-sufficient lower bound, an exclusion search (the move banned via `StackEntry.ExcludedMove`) that fails low confirms it is singular → +1 ply (+2 if it fails low by a clear margin), capped so `newDepth <= depthIn + 1`
- **Node-type threading**: a `cutNode` flag is propagated (Cut/All/PV) and feeds the LMR / IIR decisions
- An **`improving`** signal (static eval vs two plies ago) tightens or loosens the pruning margins
- ProbCut, IIR, razoring, history pruning, delta pruning, singular extensions, NMP verification, the richer-LMR tweaks, and the aspiration tweaks are *heuristic* — they trade a little accuracy for depth and are **not** score-preserving (unlike the TT). Exact correctness is the pruning-off oracle below; each can be toggled at runtime via `setoption` for A/B testing.
- **Move ordering feedback**: on a beta cutoff, the mover gets a `statBonus(depth)` history bump (main **and** 1-/2-ply continuation history), quiet refuters get a negative bump, killers are set, and the countermove is recorded

### Time budget

`computeTimes` derives an **optimum (soft)** and **maximum (hard)** millisecond budget from `wtime`/`winc`/`movestogo` (SF-style `optScale`/`maxScale` over `timeLeft = time + inc*(mtg-1) - overhead*(2+mtg)`, capped at 80 % of the remaining clock); `movetime` is a pure hard limit; `depth`/`nodes`/`mate`/`infinite` disable the time stop. A `MoveOverhead` UCI option (default 10 ms) reserves time for I/O. The main worker checks the hard limit / node budget every 2048 nodes; between iterations it rescales the soft limit by **best-move stability** (a stable root move spends less) and stops early if the next iteration almost certainly cannot finish (predicted ≈ 1.8× the last iteration).

### Reporting

`info` lines report `depth`, `seldepth`, `score` (`cp <n>` or `mate <n>`), aggregate `nodes`/`nps`/`time` across all workers, and the `pv`. Exactly one `bestmove` is emitted per `go`.

### Correctness oracle

With `UsePruning = false` the search becomes plain full-window alpha-beta (PVS / LMR / extensions / singular extensions / null-move / NMP verification / ProbCut / IIR / razoring / history pruning / delta pruning / mate-distance / qsearch-SEE / history-writes all disabled; continuation-history reads only reorder moves, which cannot change a full-window score). The test suite compares this against an exhaustive in-test minimax that shares the engine's own `qsearch` leaf and draw/terminal/mate semantics — they must return identical scores. This is the exact-correctness anchor; the heuristic pruning layered on top (IIR, razoring, history pruning, delta pruning, ProbCut, LMR, null-move, futility) only ever *speeds* the search relative to this oracle, never redefines correctness.

Constants: `MaxSearchPly = 246`, `MATE = 32000`, `INF = 32001`, `MATE_IN_MAX_PLY = 31754`.

---

## Transposition table

`Transposition.fs` is the only shared writable structure in the engine — a **lock-free XOR-lockless** table (Hyatt scheme).

- Each 16-byte entry is two naturally-aligned `uint64` fields, `Key = realKey ^^^ Data`. A probe accepts an entry iff `Key ^^^ Data = realKey`; a torn read (one field written between the reader's two reads) breaks that equality and is rejected as a miss. Aligned 64-bit reads/writes are atomic on the 64-bit runtime, so each field is individually torn-free and the XOR ties them together.
- `Data` packing (LSB → MSB): `move:16 | score:int16 | eval:int16 | depth:uint8 | genBound:uint8`, where `genBound = (generation5 << 3) | (ttPv << 2) | bound` (one bit was given to the SF-style `ttPv` flag, so the generation wraps mod 32). `ttPv` is sticky across overwrites of the same key.
- **Clusters of 4 entries** (64 B, cache-line sized). Replacement: empty slot or key-match first, else the entry minimizing `depth - relativeAge*2`.
- `Volatile.Read`/`Write` on both fields is mandatory — it stops the JIT from hoisting a stale `Key` read and pins the store order (Data before Key, so a fresh Key implies a fresh Data).
- Scores are mate-ply-corrected by the caller (`valueToTt`/`valueFromTt`) so stored mate scores are root-relative.
- `Resize`/`Clear` must not run concurrently with probes — the UCI driver guarantees this by stopping + joining any active search before `setoption Hash` / `ucinewgame`.

---

## Move ordering

`MovePick.fs` is a faithful port of Stockfish's staged move picker. It is a `[<Struct; IsByRefLike>]` value type carrying two caller-owned `stackalloc` buffers (`Span<Move>` + `Span<int>` scores) plus live stage/cursor state. Advance is the **module** function `nextMove (mp: byref<MovePick>) skipQuiets` — never an instance method, because a method on a by-ref-like struct would mutate a copy and the stage would never advance.

Stages (main search): TT move → capture init → good captures (SEE-split, losers spilled to a front "bad captures" region) → refutations (killer1, killer2, countermove) → quiet init → quiets (`partialInsertionSort`-ed) → bad captures (replayed last). Evasion, ProbCut (SEE >= threshold), and qsearch (captures + queen push-promotions) have their own chains. Laziness is structural: `generate(Quiets)` is called only inside the quiet-init stage, so an early beta cutoff (or `skipQuiets`) never materializes quiets.

`History.fs` provides per-thread ordering tables:

- **Butterfly main history** `int16[2*4096]` indexed `color<<<12 | fromTo`
- **Capture history** `int16[12*64*8]` indexed `((pc*64)+to)<<<3 | capturedPT`
- **Continuation history** `int16[768*768]` (×2, for the 1-ply and 2-ply predecessors) indexed `(prevPc*64+prevTo)*768 + (pc*64+to)`; summed into the quiet move score and the LMR / history signals (divisor `ContHistD = 29952`)
- **Counter-moves** `Move[12*64]` indexed `prevPc*64 + prevTo`
- **Killers** `Move[2*MaxPly]` per-ply refutation pair

Stat updates use SF's "gravity" formula `entry += clamp(bonus,-D,D) - entry*|clamp|/D`, whose fixpoint keeps `|entry| < D < int16 max`, so the `int16` store never overflows. `statBonus(depth) = min(160*depth - 100, 1700)`.

---

## UCI interface

`Uci.fs` is a minimal UCI driver. The search runs on its own master thread so the read loop stays responsive to `stop`/`quit`. All output goes through `Console.Out` — never `printfn` (the documented AOT crash). `go`, `quit`, `ucinewgame`, `setoption`, and `position` all stop + join any active search first, so the TT is never cleared under a live probe and exactly one `bestmove` is emitted per `go`.

### Supported commands

| Command | Behavior |
|---|---|
| `uci` | Emit `id name Eonego` / `id author Houijasu`, the `Threads` and `Move Overhead` options, and `uciok` |
| `isready` | Reply `readyok` |
| `ucinewgame` | Stop any search, clear the TT, reset to startpos |
| `position [startpos \| fen <fields>] moves m1 m2 ...` | Set the root; UCI moves are re-stamped against legal generation (recovers EP/castling flags, disambiguates under-promotions) |
| `go [depth \| nodes \| movetime \| wtime winc btime binc movestogo \| infinite \| mate]` | Start the search on the master thread; emit `info` lines and one `bestmove` |
| `stop` | Signal the atomic stop flag; the search exits at its next check point |
| `setoption name Threads value <N>` | Worker / root-parallel count (1..256) |
| `setoption name Move Overhead value <ms>` | Reserve I/O time per move, 0..5000 ms (default 10) |
| `quit` | Stop + join, then exit |

Only `Threads` and `Move Overhead` are tunable; **everything else is hardwired ON for release** — MCTS, the full pruning stack, the fixed leaf depth, and a 256 MiB `Hash`. The SF leaf net is embedded in the binary; the optional Lc0 policy net loads from disk (`EONEGO_LC0`, or a `.pb` placed beside the exe).

### Defaults

- `Threads = 1`, `Hash = 16` MiB
- Forward-pruning / heuristic toggles default to `true`: `UseProbCut`, `UseIir`, `UseRazoring`, `UseHistoryPruning`, `UseDeltaPruning`, `UseContHist`, `UseSingular`, `UseNmpVerify`, `UseLmrTweaks`, `UseAspTweaks`; `MoveOverhead = 10` ms
- **`UseNnue` defaults to `true` when `eonego.nnue` is found beside the exe** (auto-loaded at startup); otherwise it falls back to the material eval. `setoption name NnueFile value <path>` swaps the net at runtime; `UseNnue value false` forces material
- Standard chess only (Chess960 castling is out of v1 scope; castling uses standard king-to-square encoding `e1g1`/`e1c1`/`e8g8`/`e8c8`)

### Draw detection (search-owned)

Repetition (twofold-in-tree, bounded by `Rule50` and `PliesFromNull` so the scan never crosses a null move), insufficient material (KvK, KNvK, KBvK, KB-vs-KB same colour), and the 50-move rule (`Rule50 >= 100`, with the checkmate-on-the-100th-ply exception preserved).

---

## Build and run

Requirements: the **.NET 10 SDK**.

```pwsh
# Build everything
dotnet build Eonego.slnx

# Run the engine (then type UCI commands, e.g. "uci", "isready", "position startpos", "go depth 20")
dotnet run --project Eonego -c Release

# Run the tests (default tier: perft d1-d4 + all suites)
dotnet test Eonego.slnx

# Run the slow perft tier (d5-d6) too
dotnet test Eonego.slnx --filter "Category=Slow"

# Run the benchmarks
dotnet run --project Eonego.Benchmarks -c Release
```

### Native AOT

`Eonego.fsproj` is configured for **Native AOT** publishing (`PublishAot=true`, `IlcOptimizationPreference=Speed`, `IlcInstructionSet=native`). The AOT binary targets the **build machine's full ISA** (the reference build CPU is a 13980HX), so it uses the PEXT slider backend and POPCNT/LZCNT/BMI1/BMI2 intrinsics directly — the resulting binary requires a CPU with at least those features to run.

```pwsh
dotnet publish Eonego -c Release      # produces a native AOT executable
```

Other build-time tuning: `ServerGarbageCollection` + `ConcurrentGarbageCollection` for throughput, `TieredCompilation=false`, `ReadyToRun=true`, `StackTraceSupport=false`, `Tailcalls=true`, `AllowUnsafeBlocks=true` (for `stackalloc` move buffers).

---

## Testing and verification

`Eonego.Tests` is an xUnit suite spanning every module. The verification strategy layers a **perft gate** (movegen + make/unmake correctness) under a **search oracle** (search correctness).

### Perft gate

Node counts are validated against the canonical CPW positions 1–6. Default tier (d1–d4) runs in the normal suite; the slow tier (d5–d6) is opt-in via the `Slow` trait.

| Position | d1 | d2 | d3 | d4 | d5 | d6 |
|---|---:|---:|---:|---:|---:|---:|
| Startpos | 20 | 400 | 8,902 | 197,281 | 4,865,609 | 119,060,324 |
| Kiwipete | 48 | 2,039 | 97,862 | 4,085,603 | 193,690,690 | — |
| Pos 3 | 14 | 191 | 2,812 | 43,238 | 674,624 | 11,030,083 |
| Pos 4 | 6 | 264 | 9,467 | 422,333 | 15,833,292 | — |
| Pos 5 | 44 | 1,486 | 62,379 | 2,103,487 | 89,941,194 | — |
| Pos 6 | 46 | 2,079 | 89,890 | 3,894,594 | 164,075,551 | — |

### Search oracle

`SearchTests.fs` compares `UsePruning=false; UseTt=false` negamax against an exhaustive in-test minimax that shares the engine's own `qsearch` leaf and draw/terminal/mate semantics — they must return identical scores. It also checks single-thread node-count determinism, TT-on vs TT-off score invariance (pruning off), mate-in-1 / mate-in-2 suites with ply-adjusted scores, and the mate-ply TT round-trip.

### Suite inventory

`BitboardTests`, `MoveTests`, `ZobristTests` (pinned key values), `PositionTests` (FEN idempotency, make/unmake round-trip, key == from-scratch), `MoveGenerationTests` (perft gate + targeted edge cases), `SeeTests`, `IsPseudoLegalTests`, `PromotionSplitTests`, `HistoryTests` (gravity saturation), `MovePickTests` + `MovePickProbeTests` (staging, laziness, ordering, byref-mutation persistence), `EvaluationTests` (mirror symmetry, phase taper), `TranspositionTests` (XOR validation, torn-read injection, replacement), `PliesFromNullTests`, `SearchTests` (oracle, mate suites, TT invariance), `DrawDetectionTests`, `LazySmpTests`, `ProbCutTests` (ProbCut isolated from newer heuristics: tactic-not-hidden, node reduction, best-move agreement), `PruningTests` (full default pruning stack: IIR, razoring, history, delta, ProbCut).

---

## Benchmarks

`Eonego.Benchmarks` is a BenchmarkDotNet harness profiling every hot path with memory diagnostics. The steady-state targets are **0 B/op** for move generation, MovePick drains, eval, and fixed-depth search (preallocated per-worker buffers, byref-like picker on the stack, 0-B/op eval, no boxing).

| Benchmark | What it measures |
|---|---|
| `BitloopBench` | Bit-serialization loop (`b &= b-1` vs `b ^= 1<<sq` vs BMI1 `BLSR`) across 4096 bitboards of varied density |
| `AttackBench` | Rook/bishop **magic vs PEXT**, unified queen, knight lookup over 4096 (square, occupancy) pairs |
| `MoveBench` | Move encode/decode, `fromTo` key, `packed16` round-trip, `ScoredMove` byref fold, plus cold `toUci`/`parseUci` |
| `MoveGenBench` | `generate NonEvasions`/`generateLegal`, `perft` startpos d4, Kiwipete d3 (0 B/op expected) |
| `MovePickBench` | Full picker drain, simulated early cutoff (quiets never generated), `see_ge` drain |
| `EvalBench` | `eval` over a phase-spread FEN set (0 B/op expected) |
| `SearchBench` | Fixed-depth `negamax` depth 7 on Kiwipete over a preallocated worker (0 B/op steady state) |

Run with `dotnet run --project Eonego.Benchmarks -c Release` (no args runs every benchmark).

---

## Training (NNUE)

The `trainer/` directory is a self-contained Python pipeline for training the NNUE, anchored at the **King's Gambit Accepted** opening (a tactical-specialist focus). It is **bit-exact to the engine**: a numpy feature encoder is gated byte-for-byte against the engine's `featuredump`, and an integer-forward reference is gated against `nnforward`, so a net trained in PyTorch deploys with identical evals.

- **Data:** the engine's `gen` subcommand self-plays KGA games (single-threaded, deterministic) into `(fen, cp, result)` records; `sflabel.py` re-labels positions with deep **Stockfish 18** eval for distillation.
- **Train:** `train.py` fits the quantized net in win-probability space (`sigmoid(cp/K)`, eval/WDL blend), on GPU when available, and exports the `EONGNNUE` file with a round-trip parity check.
- **Evaluate:** `match.py` (UCI match driver + SPRT) and `tactics.py` (Stockfish-verified KGA tactical suite) gate each net — all anchored at the KGA position.
- **Loops:** `loop.py` (self-play RL) and `sfloop.py` (Stockfish-distillation, distribution-matched + warm-start) iterate generations with promotion gating.

---

## Project structure

```
Eonego/
├── Eonego.slnx
├── Eonego/
│   ├── Eonego.fsproj
│   ├── Bitboard.fs  Move.fs  Zobrist.fs
│   ├── NnueRegions.fs              # region-feature map + accumulator primitives
│   ├── Position.fs  Evaluation.fs
│   ├── NnueNetwork.fs  Nnue.fs     # NNUE loader/kernels + eval seam
│   ├── MoveGeneration.fs  History.fs  MovePick.fs  Transposition.fs  Search.fs
│   ├── Tooling.fs                  # gen / featuredump / nnforward subcommands
│   ├── Uci.fs  Program.fs
│   └── eonego.nnue                 # shipped default net (auto-loaded beside the exe)
├── Eonego.Tests/                   # xUnit: perft gate, search oracle, NNUE parity (incl. NnueL1AccumulatorTests)
├── Eonego.Benchmarks/              # BenchmarkDotNet microbenchmarks
└── trainer/                        # Python NNUE training pipeline (data/ nets/ are gitignored)
    ├── encoder.py  intref.py       # byte-exact feature encoder + integer-forward oracle
    ├── model.py  train.py  export.py  dataset.py
    ├── sflabel.py                  # parallel Stockfish-18 position labeler
    ├── match.py  tactics.py  build_suite.py
    ├── loop.py  sfloop.py          # self-play / Stockfish-distillation reinforcement loops
    └── suites/kga_tactics.tsv      # Stockfish-verified KGA tactical suite
```

---

## Design notes and non-goals

- **No `printfn`** anywhere on the output path — AOT-safe `Console.Out.WriteLine` only.
- **No module-level mutable state** in move generation / move picking / history — LazySMP / lockless by construction; each thread owns its buffers and tables.
- **`Position` is not thread-safe** — each search thread gets its own instance.
- **Make assumes the move is legal** (as in every engine core); legality is movegen's job.
- **Two evaluators, no HCE**: a minimal material-only static eval (SEE/ordering keep the dedicated `{100,320,330,500,900,0}` values) plus the NNUE network — the former PeSTO/PST eval was removed.
- **Zobrist draw order is load-bearing** — reordering the PRNG draws shifts every subsequent key; `ZobristTests` pins specific entries to literals.

### Deferred / out of v1 scope

- NNUE: the region-piececount architecture is small (~165k params) — a tactical *specialist* ceiling, not a general 3000-Elo net; a larger king-conditioned (HalfKP / HalfKA) feature set is the path beyond it
- SIMD for the incremental L1 update loops; lazy (skip-interior-node) accumulator updates
- `QuietChecks` generation
- Chess960 / Fischer random castling
- Pin-aware SEE refinement (the king-terminate rule already covers the dominant illegal-recapture case)

---

## License

Eonego is licensed under the **GNU Affero General Public License, version 3 or later
(AGPL-3.0-or-later)** — see [`LICENSE`](LICENSE) for the full text.

This means: you are free to use, study, modify, and redistribute Eonego, but any distributed
work that includes or is derived from it must also be offered under the AGPL-3.0, with
complete corresponding source — **including** when the program is made available to users
over a network (AGPL §13).

The AGPL-3.0 was chosen so that Eonego can incorporate networks from copyleft engines —
notably **[Monty](https://github.com/official-monty/Monty)** (AGPL-3.0, policy network) and
**[Stockfish](https://github.com/official-stockfish/Stockfish)** (GPL-3.0, NNUE value
network). Monty's AGPL-3.0 is the binding term; the GPL-3.0/AGPL-3.0 §13 combination clauses
make the two compatible in one combined work. Any incorporated network or upstream-derived
code remains under its own license — see [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).

> Note: this notice is not legal advice. The controlling terms are in [`LICENSE`](LICENSE)
> and the upstream projects' license files.
