# Eonego

A UCI chess engine written from scratch in **F# on .NET 10**, compiled to a single
self-contained native binary with **NativeAOT** (Windows x64). It plays strong, positionally
minded chess driven by an in-house **FullThreats-architecture NNUE** and a heavily
measured alpha-beta search — every search feature in the engine earned its place through
node-count sweeps and SPRT self-play matches.

## Architecture

### Board
- Bitboard position with **PEXT (BMI2) slider attacks** (magic-bitboard fallback, lazily built)
- Full legal move generation, **perft-exact to depth 6**
- Incremental Zobrist keys: full key + pawn-structure key + minor-piece key
  (the latter two feed correction history)

### Evaluation
- **FullThreats-architecture NNUE** (dual-input HalfKA + threat features, L1 = 1024), trained in-house
- Incremental **int16 accumulator** with lazy catch-up walks, finny (bucket-refresh) tables,
  a fused delta-apply kernel and sparse fc0 propagation
- **AVX2 / AVX-VNNI** kernels with a bit-exact scalar fallback (`EONEGO_FORCE_SCALAR=1`)
- **Rule-50 draw-proximity damping**: the static eval decays linearly with the halfmove
  counter, so fortresses and shuffling positions drift toward 0.00 instead of holding a
  full advantage until the 50-move draw hits the search horizon
- The net is embedded into the executable at build time when present (see *Building*)

### Policy sidecar (optional)
- **EONPOL02 sidecar** sharing the NNUE trunk's 1024-wide FT pairwise-product buffer — the
  expensive accumulator work is already paid for at eval nodes
- Piece-aware from/to heads (6 piece types × 64 STM-relative squares → 384 logits each);
  per-move score = `from[pt*64 + relSq(from)] + to[pt*64 + relSq(to)]`
- Optional **WDL head** (3 outputs per bucket off the value stack's a1 activation), evaluated
  on demand at root/PV only
- **Default OFF** (`EONEGO_POLICY` gate in UCI): when loaded, lazy per-node logits are filled at
  `StgQuietInit` and consumed by search as an **LMR term** (quiets below `PolLmrThresh` get one
  extra reduction step); an inert ordering blend (`PolOrdMul` default 0) is also wired
- `ftHash`-bound to the loaded `.nnue` trunk — refuses a head trained on a foreign net

### Search
- Negamax / PVS with quiescence, **LazySMP** parallelism, and **thread voting**
  over the workers' root results
- XOR-lockless (Hyatt) transposition table, 4-entry 64-byte clusters, 5-bit aging
- Aspiration windows, iterative deepening
- Selectivity: null-move pruning (+ verification), reverse futility, razoring, ProbCut, IIR,
  LMR with history-driven tweaks, late-move (move-count) pruning, history / futility / SEE
  pruning, singular extensions with **multicut and negative extensions**, qsearch TT
  cutoffs + stand-pat stores, bound-consistent TT-score-as-eval, optional quiet checking
  moves at the first quiescence ply
- **Correction history** (pawn-structure keyed; minor-piece rider) corrects the static eval
  wherever it feeds pruning
- Move ordering: staged lazy picker (TT move, captures by MVV + capture history, killers,
  counter-move, quiets by butterfly + continuation history)
- Mate scores report the **complete principal variation to the mate** (the reported line is
  extended from the transposition table, legality-checked, capped at the mate distance)
- On-demand 3-man retrograde tables are solved in the background from low-material roots and
  probed for exact DTM scores once published (`EONEGO_RETRO=0` disables this)
- DFPN mate oracle (`EONEGO_DFPN=1`): a df-pn proof-number solver runs beside the search,
  proving checks-only forced mates; every proof is re-verified by a table-free replay before
  it can override the bestmove, and `go mate N` returns as soon as a certified mate ≤ N exists
- Search margins, time-management constants, and resource budgets live in `Tunables.fs`
  as `EONEGO_T_*` environment overrides; the tuned defaults are SPSA/SPRT-measured
  where strength-sensitive.

## Building

Requirements:
- .NET SDK with `net10.0` support
- Visual Studio **"Desktop development with C++"** workload (NativeAOT links with MSVC)
- **Git LFS** for `nets/main.nnue` (~106 MB): `git lfs install` then clone (or `git lfs pull`)

```powershell
pwsh ./publish.ps1             # embeds nets/main.nnue + nets/main.policy when present
# -> Eonego/bin/Release/net10.0/win-x64/publish/Eonego.exe  (~110 MB with net embedded)
```

If `main.nnue` is missing (LFS not pulled), `publish.ps1` falls back to `scripts/fetch-net.ps1`.
A build without `nets/main.nnue` still succeeds, but the exe cannot search until you have the net
or set `EONEGO_NET=<path>` at runtime. Pre-built zips on [Releases](https://github.com/Houijasu/Eonego/releases)
also include the embedded net.

Equivalent: `dotnet publish Eonego/Eonego.fsproj -c Release -r win-x64` (the script also puts the VS
Installer on `PATH` so the MSVC link step does not fail with exit code 123).

Note: the project sets `<IlcInstructionSet>native</IlcInstructionSet>` — the AOT binary
targets the **build machine's** exact CPU features (PEXT, AVX2, AVX-VNNI, …) and will
refuse to start on lesser CPUs. Build on the machine you play on, or pick an explicit
baseline (e.g. `x86-x64-v3`) for a portable binary.

For development, a JIT build works anywhere the SDK does:

```powershell
dotnet build Eonego/Eonego.fsproj -c Release
dotnet test  Eonego.Tests/Eonego.Tests.fsproj -c Release   # 361 tests
```

## UCI options

| Option | Default | Notes |
|---|---|---|
| `Threads` | 1 | LazySMP workers (1–256) |
| `Hash` | 256 | Transposition table MB (1–65536) |
| `MultiPV` | 1 | Extra reported lines (main worker only) |
| `Move Overhead` | 10 | ms safety margin per move |

`go searchmoves <m1> <m2> ...` is supported — restrict the root to specific candidate moves
(the way to force a deep verdict on one move a normal search keeps reducing away).

Everything else is deliberately hardwired; A/B, tuning, and debug switches live behind
environment variables so matches never need a rebuild:

| Env var | Effect |
|---|---|
| `EONEGO_NET=<path>` | Load a compatible net from disk instead of the embedded resource |
| `EONEGO_POLICY=<path>` | Load a `.policy` sidecar (`=1` requests embedded `policy.dat` when present) |
| `EONEGO_T_*` | Override tuned margins and resource budgets (see `Tunables.fs`) |
| `EONEGO_T_POL_MINDEPTH` | No policy inference below this picker depth (default 3) |
| `EONEGO_T_POL_LMR` | Quiets with from+to logit below this get one extra LMR step (default 0 = off) |
| `EONEGO_T_POL_ORDMUL` | scoreQuiets ordering blend multiplier (default 0 = inert) |
| `EONEGO_T_POL_ORDSHIFT` / `EONEGO_T_POL_CLAMP` | Ordering-blend shift and clamp |
| `EONEGO_CUTDUMP=<path>` | Append matched-state cut records (policy rank-evaluation baseline) |
| `EONEGO_POOL=0` | Disable the default persistent worker pool and return to fresh workers per move |
| `EONEGO_RETRO=0` | Disable background retrograde solving and exact 3-man probes |
| `EONEGO_R50DAMP=0` | Disable rule-50 draw-proximity eval damping |
| `EONEGO_CONT4=1` | Weighted ss-4 continuation history in the LMR history term |
| `EONEGO_CAPFUT=1` | Capture futility pruning |
| `EONEGO_CORRMINOR=1` | Minor-piece correction-history rider |
| `EONEGO_TT_REFRESH=1` | Probe hits re-stamp TT entry age |
| `EONEGO_PARTIAL=1` | Adopt a hard-stopped iteration's best root move |
| `EONEGO_QSCHECKS=1` | Quiet checking moves at the first qsearch ply (SEE-losing checks skipped) |
| `EONEGO_ROOTEFFORT=1` | Re-sort root moves between iterations by subtree node effort |
| `EONEGO_ROOTVERIFY=1` | Re-search rotating non-best root moves when the root score stagnates |
| `EONEGO_T_ASP_FHRED=0` | Disable the aspiration fail-high re-search depth reduction (slow-win suppression fix) |
| `EONEGO_CORRHIST=0` | Disable correction history (default on) |
| `EONEGO_QSTT=0` / `EONEGO_TTEVADJ=0` | Disable qsearch-TT / TT-eval-adjust (default on) |
| `EONEGO_CHECKEXT=1` / `EONEGO_QSEVCAP=1` | Legacy check extension / qsearch evasion cap |
| `EONEGO_FINNY=0` | Disable finny bucket-refresh tables and use from-scratch accumulator refresh |
| `EONEGO_FORCE_SCALAR=1` / `EONEGO_FORCE_NOVNNI=1` / `EONEGO_FORCE_DENSE=1` | NNUE kernel fallbacks for bit-exactness and performance debugging |
| `EONEGO_PROF=1` | Per-search phase profile line (1-thread semantics); tree-shape counters `prof4`–`prof6` |

## Development harness

- `Eonego.exe gen --start <fen> --games N --out <file> --net <path>` — self-play
  label generation (`--depth D` or `--nodes K`, temperature/random-prefix controls).
  Records are **v2**: `fen;cp_white;result_white;best_uci` (4th field = search best move).
- `Eonego.exe dumpft --net <path> --in <fens> --out <bin>` — trainer feature dump:
  1034-byte records containing bucket, side-to-move, PSQT/eval internals and the 1024-byte FT vector.
- `Eonego.exe dumppolicy --net <nnue> --policy <sidecar> --in <fens> --out <txt>` — policy
  parity dump: `fen TAB 64 from-logits TAB 64 to-logits [TAB w d l]`.
- `Eonego.exe tbgen --tb <dir[;dir2]> --out <file> --signatures <list>` — native Syzygy-labeled
  endgame records with WDL-preserving move sets (for policy training).
- `Eonego.exe retro <fen> [--verify]` — solve the position's reachable low-material
  retrograde signatures, print stats, and optionally run the self-consistency proof.
- `scripts/nodesweep.ps1` — deterministic 1T node-count sweep over a 6-FEN suite; the
  byte-identity gate for "this change is inert by default" claims. Invoke array params via
  `pwsh -Command`, not `-File` (PowerShell `-File` mangles `-Depths 13,14,15`).
- `scripts/bench.ps1` — fixed-position benchmark.
- `scripts/b4fixture.ps1` — targeted root-move fixture runner for slow-win/root-search
  pathology checks.
- `trainer/match.py` — self-play match driver: SPRT early stopping, per-player env/UCI-option
  channels, old-vs-new binary matches (`--exe-b`), concurrency with color-swapped opening
  pairs per worker, fixed `movetime`/`nodes` budgets or real game clocks (`--tc "10+0.1"`,
  flag fall = loss).
- `trainer/spsa.py` — SPSA tuner over the `EONEGO_T_*` margins (fishtest-style gain
  schedules, resumable per-wave state, `--fake-objective` self-test).
- `trainer/train.py` / `trainer/export.py` — train and quantize one EONGNNUE generation
  from generated labels; parity/blending helpers live beside them.
- `trainer/policy_train.py` / `trainer/policy_export.py` — train and export the EONPOL02
  policy sidecar (masked quiet-CE + optional WDL-CE on gen/tbgen + dumpft records).
- `trainer/policy_parity.py` / `trainer/policy_intref.py` — bit-exact engine vs integer
  oracle parity for the policy head.
- `trainer/policy_rankeval.py` — matched-state rank gate: policy vs live history on
  `EONEGO_CUTDUMP` states (go/no-go before SPRT promotion).
- `trainer/policy_loop.py` — resumable end-to-end policy pipeline orchestrator.

The working discipline: **nothing ships without a measurement.** Fixed-depth node counts
adjudicate tree mechanics; SPRT matches adjudicate everything else. See
[`docs/policy-net-phase0.md`](docs/policy-net-phase0.md) for the policy-net campaign
measurement log.

## Testing

361 xunit tests cover perft, make/unmake round-trips (full state + all three incremental
keys), SEE, the staged move picker, TT torn-read behavior, NNUE bit-exactness against
golden evals (AVX2 == scalar), draw detection, LazySMP determinism at 1 thread, thread
voting, retrograde solving, worker-pool reuse, policy kernel parity and LMR consumption,
and oracle equivalence of the pruned search against plain full-window alpha-beta.

## License

MIT — see [LICENSE](LICENSE). Engine source and trained network weights distributed with
this repository are original to the project.

## Status

Actively developed. Strength is tracked against the engine's own history via SPRT
(hundreds of Elo gained through 2026 search/eval campaigns). The **policy-net campaign**
(optional EONPOL02 sidecar, LMR consumption, offline training pipeline) is scaffolded
default-OFF; promotion awaits rankeval + SPRT. Further search efficiency gains and
continued net training are the roadmap.
