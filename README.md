# Eonego

A from-scratch chess engine in **F# / .NET 10**, compiled to a self-contained **NativeAOT** binary.

Eonego's production search is a **hybrid**: a Monte-Carlo tree search at the root (Baier–Winands MCTS-MB),
where every leaf is evaluated by a fixed-depth **alpha-beta** search using a from-scratch port of the
**Stockfish-16 NNUE** evaluator, and the MCTS **policy priors** come from a from-scratch port of a
**Leela (Lc0) 20×256-SE CNN** quantized to int8. It also contains a complete, standalone alpha-beta engine
(reused as the MCTS leaf and selectable on its own).

> Status: working and self-playing; **strength vs. plain alpha-beta is not yet SPRT-validated** — see
> `trainer/match.py`. Treat the hybrid as experimental until measured.

## Architecture

| Layer | Files | Notes |
|---|---|---|
| Board / movegen | `Bitboard.fs`, `Move.fs`, `Zobrist.fs`, `Position.fs`, `MoveGeneration.fs` | perft-verified |
| Eval (leaf) | `SfAccumulator.fs`, `SfNnue.fs` | SF-16 NNUE, incremental accumulator + AVX2 kernels |
| Policy (priors) | `Lc0Proto.fs`, `Lc0Quant.fs`, `Lc0PolicyMap.fs`, `Lc0Encoder.fs`, `Lc0Net.fs` | Lc0 20×256-SE CNN, int8 forward |
| Move ordering | `History.fs`, `MovePick.fs` | staged lazy picker |
| Search | `Transposition.fs`, `Search.fs` | lockless TT, LazySMP alpha-beta + PVS + qsearch |
| MCTS | `Mcts.fs` | root-parallel MCTS-MB; alpha-beta leaf; Lc0 (or history) priors |
| UCI | `Tooling.fs`, `Uci.fs`, `Program.fs` | |

- **Leaf eval** = Stockfish-16 NNUE (incremental accumulator maintained across make/unmake; AVX2 + scalar
  paths, parity-gated). Embedded into the binary when `nets/sf16.nnue` is present.
- **Priors** = Lc0 CNN policy, quantized to int8 (`Lc0Quant.fs`, ~4× faster forward than fp32). The Lc0 net
  is loaded from disk (it is large); without one, MCTS falls back to weak history-softmax priors.
- **Parallelism** = root-parallel: N independent trees share only the lockless TT and a stop flag, merged
  into one decision.

## Build

Requires the .NET 10 SDK.

```sh
dotnet build Eonego.slnx -c Release
dotnet test  Eonego.Tests/Eonego.Tests.fsproj -c Release
```

Tests that need a net (`tryLoadSfNet` / the Lc0 loader) **soft-skip** when `nets/` is empty, so the suite
runs without the (gitignored) weights — coverage is just narrower.

### NativeAOT publish (the GUI deliverable)

```pwsh
pwsh ./publish.ps1
```

Produces `Eonego/bin/Release/net10.0/win-x64/publish/Eonego.exe` (self-contained, ~42 MB). NativeAOT links
with MSVC, so the VS Installer dir must be on `PATH` — `publish.ps1` handles that. The binary is built with
`IlcInstructionSet=native`, so it **requires a CPU with at least this machine's ISA (AVX2 / BMI2)**.

## Nets

`nets/` is gitignored (the weights are large / separately licensed). Place them yourself:

- `nets/sf16.nnue` — Stockfish-16 NNUE (CC0). Embedded into the binary at build time when present.
- `nets/<lc0>.pb` — an Lc0 `20×256-SE` network. Loaded from disk; auto-discovered next to the exe, or set
  `EONEGO_LC0=<path>`.

## UCI options & env knobs

The **UCI option surface is intentionally minimal**: only `Threads` and `Move Overhead`. Every other knob is
an environment variable (so the release surface stays clean and SPRT sweeps need no rebuild):

| Env var | Effect |
|---|---|
| `EONEGO_MCTS` | `0` → plain alpha-beta (`Search.go`); default `1` → MCTS+Lc0 hybrid |
| `EONEGO_LC0` | path to an Lc0 `.pb`; `none`/`off`/`0` → disable (history priors); else auto-discover |
| `EONEGO_LC0_FP32` | force the fp32 Lc0 forward instead of the default int8 |
| `EONEGO_CPUCT` | PUCT exploration constant ×100 (default 150) |
| `EONEGO_LEAFDEPTH` | fixed negamax depth at each MCTS leaf (default 8) |
| `EONEGO_K` | logistic cp→win-prob scale (default 200) |
| `EONEGO_EVAL_PRIORS` | no-Lc0 only: SF-eval priors instead of history-softmax |
| `EONEGO_BATCH` | batch B Lc0 leaves per worker (experimental; default 1) |

## Measurement (trainer/)

`trainer/match.py` runs UCI self-play + GSPRT between two configurations differentiated by **env vars**
(not UCI options), equalized on `go movetime` (the only fair budget once the searches differ):

```sh
# Does the MCTS+Lc0 hybrid beat plain alpha-beta at equal time?
python trainer/match.py --a "EONEGO_MCTS=1" --b "EONEGO_MCTS=0" \
    --shared "EONEGO_LC0=nets/<lc0>.pb" --movetime 200 --openings 200 --sprt
```

`trainer/tactics.py` scores a config on a Stockfish-verified tactical suite (`trainer/suites/`).

## Benchmarks

```sh
dotnet run --project Eonego.Benchmarks -c Release -- --filter "*Lc0ForwardBench*"
```
