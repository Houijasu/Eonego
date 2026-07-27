# Eonego

A UCI chess engine written from scratch in **F# on .NET 10**, compiled to a single
self-contained native binary with **NativeAOT** (Windows and Linux x64). It plays strong, positionally
minded chess driven by an in-house **FullThreats-architecture NNUE** and a measured
alpha-beta search.

## Architecture

### Board

- **Bitboard**
  - PEXT (BMI2) slider attacks with magic-bitboard fallback
  - Full legal move generation, perft-exact to depth 6
- **Zobrist keys**
  - Full position key plus pawn-structure and minor-piece keys for correction history

### Evaluation

- **NNUE**
  - FullThreats architecture: dual-input HalfKA + threat features
  - Incremental accumulator with finny refresh, fused delta updates, and SIMD kernels
  - Rule-50 draw-proximity damping so shuffling endgames drift toward a draw score
  - Network embedded in the published binary (see *Building*)

### Search

- **Negamax / PVS**
  - Principal-variation search with quiescence, aspiration windows, and iterative deepening
  - LazySMP multi-threading with thread voting at the root
  - Mate lines reported through to the full mate sequence
- **DFPN**
  - Proof-number mate solver running beside the main search (on by default; `EONEGO_DFPN=0` disables)
  - Finds forced mates in checks-only lines; proofs are verified before they override the best move
- **Policy head**
  - Neural move prior loaded by default: a sidecar on the NNUE trunk, plus a separate endgame net
    for low-material positions (`EONEGO_POLICY=0` disables)
  - Feeds late-move reduction and emits win/draw/loss estimates at the root (see `UCI_ShowWDL`)
- **LMR**
  - Late-move reductions guided by search history and move-count pruning
  - Can use the policy head to reduce unlikely quiet moves more aggressively
- **Selectivity & pruning**
  - Null-move pruning, reverse futility, razoring, ProbCut, internal iterative reduction
  - Singular extensions, history and SEE pruning, quiescence transposition cutoffs
- **Correction history**
  - Learns static-eval bias from pawn structure, minor/major/non-pawn material, and the previous move
- **Move ordering**
  - Staged picker: transposition move, captures, killers, counter-moves, quiet history
  - Quiet history blends butterfly, continuation (1/2/4-ply), and pawn-structure tables
- **Transposition table**
  - Lockless clustered hash table with aging
- **Tablebases & endgame**
  - Syzygy WDL/DTZ probing and optional root move filtering
  - Background retrograde solving for exact mates in very low-material positions

## Building

Requirements:

- .NET SDK with `net10.0` support, and PowerShell 7+ (`pwsh`) to run the build script
- A native toolchain for NativeAOT: Visual Studio **"Desktop development with C++"** on Windows,
  `clang` + `zlib` headers + `binutils` on Linux
- **Git LFS** for `nets/main.nnue` (~106 MB): `git lfs install` then clone (or `git lfs pull`)

```powershell
pwsh ./publish.ps1
# -> Eonego/bin/Release/net10.0/<rid>/publish/Eonego[.exe]
```

If the network file is missing, `publish.ps1` can fetch it automatically. Pre-built releases
on [GitHub](https://github.com/Houijasu/Eonego/releases) ship with the net already embedded.

`publish.ps1` runs on Windows, Linux, and macOS and targets the host by default. NativeAOT fixes
the instruction set at compile time, so the default baseline is **x86-64-v3 (AVX2 + BMI2)** — the
practical floor for the SIMD NNUE kernels, and portable to any Intel Haswell (2013) or AMD Zen 1
(2017) or newer CPU.

| Command | Output |
|---|---|
| `pwsh ./publish.ps1` | Portable AOT for the host OS — AVX2 baseline |
| `pwsh ./publish.ps1 -InstructionSet x86-64-v4` | AOT with AVX-512, for Skylake-X / Zen 4+ testers |
| `pwsh ./publish.ps1 -InstructionSet native` | Max-speed AOT pinned to the build machine — **faults on any other CPU** |
| `pwsh ./publish.ps1 -Mode jit` | Self-contained single-file build that detects the CPU at runtime |
| `pwsh ./publish.ps1 -Mode jit -Rid linux-x64` | Linux tester binary, cross-published from any host |

NativeAOT cannot cross-compile between operating systems, so a Linux AOT binary must be built on
Linux; `-Mode jit` cross-publishes from anywhere (the tester then needs `chmod +x`). The instruction
set and the AOT/JIT choice do not change the search tree — node counts are identical across all of
these builds, so the nodesweep byte-identity contract holds for every variant.

For development:

```powershell
dotnet build Eonego/Eonego.fsproj -c Release
dotnet test  Eonego.Tests/Eonego.Tests.fsproj -c Release
```

## UCI options

| Option | Default | Notes |
|---|---|---|
| `Threads` | 1 | Search threads (1–256) |
| `Hash` | 256 | Transposition table size in MB |
| `Clear Hash` | — | Button: wipe the transposition table |
| `MultiPV` | 1 | Number of principal variations to report |
| `Move Overhead` | 10 | Milliseconds reserved per move for GUI/communication |
| `Ponder` | false | Declares support for `go ponder` / `ponderhit` |
| `UCI_ShowWDL` | false | Append `wdl W D L` (per-mille, side-to-move) to info lines; advertised only when the embedded policy sidecar carries a WDL head |
| `SyzygyPath` | *(empty)* | Path to Syzygy tablebase files |

`go searchmoves` restricts the root to a given move list — useful for analyzing a single
candidate that a full search might prune away early.

## Development harness

Command-line tools for training data, benchmarks, and regression checks:

- **Self-play** — `gen` produces labeled positions from engine games
- **Feature dump** — `dumpft` exports NNUE trainer records from FEN lists
- **Policy parity** — `dumppolicy` / `dumppolicyown` compare engine inference against trainer oracles
- **Tablebase labels** — `tbgen` builds Syzygy-labeled endgame training sets
- **Retrograde** — `retro` solves and verifies low-material retrograde tables
- **Scripts** — `nodesweep.ps1`, `bench.ps1`, and `b4fixture.ps1` for node-count and timing regression
- **Trainer** — Python pipelines for match testing (SPRT), SPSA tuning, NNUE training, and policy-net campaigns

Changes are validated with fixed-depth node-count sweeps (tree mechanics) and SPRT self-play
matches (playing strength).

## Testing

408 automated tests cover move generation, make/unmake, SEE, move ordering, history and
correction tables, transposition-table behavior, NNUE correctness, draw detection,
multi-threading, tablebase integration, policy/WDL inference, time management, and search
equivalence against an unpruned baseline.

## License

MIT — see [LICENSE](LICENSE).

## Status

Actively developed; strength is tracked through SPRT self-play against earlier versions. Current
release is **0.1.2**, which is throughput-only over 0.1.1 (NNUE accumulator kernel tuning; node
counts are byte-identical). Since 0.1.0, the previously experimental search features — DFPN, the policy
head, the dynamic time-management components, and the extended history/correction terms — are
enabled by default, each still individually disableable through its `EONEGO_*` environment
variable. These defaults are pre-SPRT and under validation; the time manager also reserves a small
proportional safety margin on the clock to avoid flag-fall under load.
