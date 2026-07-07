# Eonego

A UCI chess engine written from scratch in **F# on .NET 10**, compiled to a single
self-contained native binary with **NativeAOT** (Windows x64). It plays strong, positionally
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
  - Optional proof-number mate solver running beside the main search
  - Finds forced mates in checks-only lines; proofs are verified before they override the best move
- **Policy head**
  - Optional neural move prior: a sidecar on the NNUE trunk, or a separate endgame net for
    low-material positions
  - Feeds late-move reduction and can emit win/draw/loss estimates at the root
- **LMR**
  - Late-move reductions guided by search history and move-count pruning
  - Can use the policy head to reduce unlikely quiet moves more aggressively
- **Selectivity & pruning**
  - Null-move pruning, reverse futility, razoring, ProbCut, internal iterative reduction
  - Singular extensions, history and SEE pruning, quiescence transposition cutoffs
- **Correction history**
  - Learns static-eval bias from pawn structure and minor-piece placement
- **Move ordering**
  - Staged picker: transposition move, captures, killers, counter-moves, quiet history
- **Transposition table**
  - Lockless clustered hash table with aging
- **Tablebases & endgame**
  - Syzygy WDL/DTZ probing and optional root move filtering
  - Background retrograde solving for exact mates in very low-material positions

## Building

Requirements:

- .NET SDK with `net10.0` support
- Visual Studio **"Desktop development with C++"** workload (NativeAOT links with MSVC)
- **Git LFS** for `nets/main.nnue` (~106 MB): `git lfs install` then clone (or `git lfs pull`)

```powershell
pwsh ./publish.ps1
# -> Eonego/bin/Release/net10.0/win-x64/publish/Eonego.exe
```

If the network file is missing, `publish.ps1` can fetch it automatically. Pre-built releases
on [GitHub](https://github.com/Houijasu/Eonego/releases) ship with the net already embedded.

The AOT binary targets the **build machine's** CPU (PEXT, AVX2, AVX-VNNI, …). Build on the
machine you play on, or choose an explicit instruction-set baseline for a portable binary.

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
| `MultiPV` | 1 | Number of principal variations to report |
| `Move Overhead` | 10 | Milliseconds reserved per move for GUI/communication |
| `Ponder` | false | Declares support for `go ponder` / `ponderhit` |
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

400 automated tests cover move generation, make/unmake, SEE, move ordering, transposition
table behavior, NNUE correctness, draw detection, multi-threading, tablebase integration,
policy inference, and search equivalence against an unpruned baseline.

## License

MIT — see [LICENSE](LICENSE).

## Status

Actively developed. Strength is tracked through SPRT self-play against earlier versions.
Syzygy probing, pondering, the endgame policy path, and several experimental search features
are implemented but not all promoted to default-on yet.
