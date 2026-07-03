# Eonego

A UCI chess engine written from scratch in **F# on .NET 10**, compiled to a single
self-contained native binary with **NativeAOT** (Windows x64). It plays strong, positionally
minded chess driven by a bit-exact port of Stockfish's NNUE evaluation and a heavily
measured alpha-beta search — every search feature in the engine earned its place through
node-count sweeps and SPRT self-play matches.

## Architecture

### Board
- Bitboard position with **PEXT (BMI2) slider attacks** (magic-bitboard fallback, lazily built)
- Full legal move generation, **perft-exact to depth 6**
- Incremental Zobrist keys: full key + pawn-structure key + minor-piece key
  (the latter two feed correction history)

### Evaluation
- Bit-exact F# port of the Stockfish **"FullThreats" NNUE** (`nn-f8a759c05f9f.nnue`,
  dual-input HalfKA + threat features, L1 = 1024)
- Incremental **int16 accumulator** with lazy catch-up walks, finny (bucket-refresh) tables,
  a fused delta-apply kernel and sparse fc0 propagation
- **AVX2 / AVX-VNNI** kernels with a bit-exact scalar fallback (`EONEGO_FORCE_SCALAR=1`)
- The net is embedded into the executable at build time when present (see *Building*)

### Search
- Negamax / PVS with quiescence, **LazySMP** parallelism, and SF-style **thread voting**
  over the workers' root results
- XOR-lockless (Hyatt) transposition table, 4-entry 64-byte clusters, 5-bit aging
- Aspiration windows, iterative deepening; optional root-move work-queue mode
- Selectivity: null-move pruning (+ verification), reverse futility, razoring, ProbCut, IIR,
  LMR with history-driven tweaks, late-move (move-count) pruning, history / futility / SEE
  pruning, singular extensions with **multicut and negative extensions**, qsearch TT
  cutoffs + stand-pat stores, bound-consistent TT-score-as-eval
- **Correction history** (pawn-structure keyed; minor-piece rider) corrects the static eval
  wherever it feeds pruning
- Move ordering: staged lazy picker (TT move, captures by MVV + capture history, killers,
  counter-move, quiets by butterfly + continuation history)
- All ~40 search margins live in `Tunables.fs` as `EONEGO_T_*` environment overrides;
  the defaults are **SPSA-tuned** (self-play, ~10k games per wave)

## Building

Requirements:
- .NET SDK with `net10.0` support
- Visual Studio **"Desktop development with C++"** workload (NativeAOT links with MSVC)
- The NNUE net file at `nets/nn-f8a759c05f9f.nnue` (gitignored — it is a Stockfish
  network and subject to Stockfish's licensing; the engine refuses to search without a net)

```powershell
pwsh ./publish.ps1
# -> Eonego/bin/Release/net10.0/win-x64/publish/Eonego.exe  (~90 MB, net embedded)
```

Note: the project sets `<IlcInstructionSet>native</IlcInstructionSet>` — the AOT binary
targets the **build machine's** exact CPU features (PEXT, AVX2, AVX-VNNI, …) and will
refuse to start on lesser CPUs. Build on the machine you play on, or pick an explicit
baseline (e.g. `x86-x64-v3`) for a portable binary.

For development, a JIT build works anywhere the SDK does:

```powershell
dotnet build Eonego/Eonego.fsproj -c Release
dotnet test  Eonego.Tests/Eonego.Tests.fsproj -c Release   # 320 tests
```

## UCI options

| Option | Default | Notes |
|---|---|---|
| `Threads` | 1 | LazySMP workers (1–256) |
| `Hash` | 256 | Transposition table MB (1–65536) |
| `MultiPV` | 1 | Extra reported lines (main worker only) |
| `Move Overhead` | 10 | ms safety margin per move |
| `Use Work Queue` | false | ABDADA subtree deferral on LazySMP (threads skip subtrees a sibling already searches; experimental, Threads > 1 only) |

Everything else is deliberately hardwired; experimental features ship behind
environment variables (default off unless noted) so A/B matches never need a rebuild:

| Env var | Effect |
|---|---|
| `EONEGO_T_*` | Override any tuned search margin (see `Tunables.fs`) |
| `EONEGO_POOL=1` | Persistent worker pool: skips per-move worker allocation, keeps history tables warm across moves |
| `EONEGO_CONT4=1` | Weighted ss-4 continuation history in the LMR history term |
| `EONEGO_CAPFUT=1` | Capture futility pruning |
| `EONEGO_CORRMINOR=1` | Minor-piece correction-history rider |
| `EONEGO_TT_REFRESH=1` | Probe hits re-stamp TT entry age |
| `EONEGO_PARTIAL=1` | Adopt a hard-stopped iteration's best root move |
| `EONEGO_ROOTPAR=1` | Legacy root-move work-queue parallelism (what `Use Work Queue` used to select) |
| `EONEGO_CORRHIST=0` | Disable correction history (default on) |
| `EONEGO_QSTT=0` / `EONEGO_TTEVADJ=0` | Disable qsearch-TT / TT-eval-adjust (default on) |
| `EONEGO_CHECKEXT=1` / `EONEGO_QSEVCAP=1` | Legacy check extension / qsearch evasion cap |
| `EONEGO_FORCE_SCALAR=1` | Scalar NNUE kernels (bit-exactness debugging) |
| `EONEGO_PROF=1` | Per-search phase profile line (1-thread semantics) |

## Development harness

- `scripts/nodesweep.ps1` — deterministic 1T node-count sweep over a 6-FEN suite; the
  byte-identity gate for "this change is inert by default" claims. Invoke array params via
  `pwsh -Command`, not `-File` (PowerShell `-File` mangles `-Depths 13,14,15`).
- `scripts/bench.ps1` — fixed-position benchmark.
- `trainer/match.py` — self-play match driver: SPRT early stopping, per-player env/UCI-option
  channels, old-vs-new binary matches (`--exe-b`), concurrency with color-swapped opening
  pairs per worker, fixed `movetime`/`nodes` budgets or real game clocks (`--tc "10+0.1"`,
  flag fall = loss).
- `trainer/spsa.py` — SPSA tuner over the `EONEGO_T_*` margins (fishtest-style gain
  schedules, resumable per-wave state, `--fake-objective` self-test).

The working discipline: **nothing ships without a measurement.** Fixed-depth node counts
adjudicate tree mechanics; SPRT matches adjudicate everything else.

## Testing

320 xunit tests cover perft, make/unmake round-trips (full state + all three incremental
keys), SEE, the staged move picker, TT torn-read behavior, NNUE bit-exactness against
golden evals (AVX2 == scalar), draw detection, LazySMP determinism at 1 thread, thread
voting, and oracle equivalence of the pruned search against plain full-window alpha-beta.

## Status

Actively developed. Strength is tracked against the engine's own history via SPRT
(hundreds of Elo gained through 2026 search/eval campaigns); on equal hardware it still
searches a larger tree than Stockfish to reach the same depth — closing that gap, plus
training an own network, is the roadmap.
