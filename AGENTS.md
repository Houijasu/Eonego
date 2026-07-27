# AGENTS.md

A guide for agents working in the Eonego codebase — a UCI chess engine in F# on .NET 10, published as a NativeAOT native binary. This file captures the non-obvious commands, conventions, and hazards that an agent would otherwise discover only through trial and error.

## Project layout

Three projects in `Eonego.slnx` (the new XML solution format, not `.sln`):

- **`Eonego/`** — the engine. UCI driver + offline tooling subcommands. Single project, no dependencies.
- **`Eonego.Tests/`** — xUnit test suite (~408 tests). References `Eonego` via `InternalsVisibleTo` (declared in `Transposition.fs`).
- **`Eonego.Benchmarks/`** — BenchmarkDotNet harness. NOT published AOT (it can't self-host in-process).

Side components:

- **`trainer/`** — Python pipelines for self-play, NNUE/policy training, SPSA tuning, SPRT match testing, parity gates.
- **`scripts/`** — PowerShell harnesses: `nodesweep.ps1` (byte-identity gate), `bench.ps1`, `b4fixture.ps1`, `fetch-net.ps1`.
- **`native/eonego_cuda/`** — C ABI DLL for the graph evaluator (CUDA with CPU fallback), P/Invoked from `GraphGpu.fs`.
- **`nets/`** — trained weights tracked in git (`main.nnue` via Git LFS, `main.policy`, `main.ownpolicy`).
- **`data/`** — training data and fixtures (e.g. `graph_syzygy3/`).

## Essential commands

```powershell
# Build the engine (managed, no AOT) — the default dev loop
dotnet build Eonego/Eonego.fsproj -c Release

# Run the test suite (CI runs exactly this — Slow category excluded)
dotnet test Eonego.Tests/Eonego.Tests.fsproj -c Release --no-build --filter "Category!=Slow"

# Run the benchmarks (NOT AOT — BenchmarkDotNet needs the JIT)
dotnet run --project Eonego.Benchmarks -c Release

# Publish the NativeAOT shipping binary (~128 MB with embedded net); portable AVX2 baseline
pwsh ./publish.ps1
# -> Eonego/bin/Release/net10.0/<rid>/publish/Eonego[.exe]

# Variants: AVX-512 / this-CPU-only / runtime-dispatch / Linux binary from a Windows host
pwsh ./publish.ps1 -InstructionSet x86-64-v4
pwsh ./publish.ps1 -InstructionSet native
pwsh ./publish.ps1 -Mode jit
pwsh ./publish.ps1 -Mode jit -Rid linux-x64

# Fetch the LFS net if missing (publish.ps1 auto-runs this)
pwsh ./scripts/fetch-net.ps1

# Byte-identity node-count sweep (the campaign gate; see below)
pwsh ./scripts/nodesweep.ps1 -Exe <path-to-Eonego.exe> [-Depths 13,14,15] [-EnvSpec "EONEGO_T_RFP_MARGIN=121,..."]

# Fixed-position bench
pwsh ./scripts/bench.ps1 [-Exe <path>] [-Depth 15] [-Prof]

# Run the engine interactively (UCI loop) — no args
./Eonego.exe
# Or invoke an offline tooling subcommand (see Program.fs)
./Eonego.exe gen --start <fen> --games N --out <file> [--depth D] [--net <path>]
./Eonego.exe dumpft --net <path> --in <fens> --out <bin>
./Eonego.exe dumppolicy --net <nnue> --policy <sidecar> --in <fens> --out <txt>
./Eonego.exe tbgen --tb <dir[;dir2]> --out <file> --signatures <list>
./Eonego.exe retro <fen> [--verify]
```

### Build prerequisites

- **.NET 10 SDK** (`net10.0` target) and **PowerShell 7+** (`pwsh`) — `publish.ps1` uses `$IsWindows`/`$IsLinux`.
- **A native toolchain for AOT**: on Windows the Visual Studio "Desktop development with C++" workload — NativeAOT links with MSVC (`link.exe`), and `publish.ps1` imports the full vcvars64 environment because AOT link fails with an opaque "exit code 123" otherwise. On Linux: `clang`, zlib headers, `binutils` (`publish.ps1` checks and names the apt/dnf packages).
- **Git LFS** for `nets/main.nnue` (~106 MB): `git lfs install` then clone (or `git lfs pull`).

**ISA selection is a compile-time decision.** NativeAOT constant-folds `Avx2.IsSupported` / `Bmi2.X64.IsSupported` — an AOT image does no runtime CPU dispatch. `publish.ps1` therefore defaults to `-p:IlcInstructionSet=x86-64-v3` (AVX2 + BMI1/BMI2 + FMA/LZCNT/MOVBE; .NET 10 folds those into the single `avx2` level, so "AVX2 without PEXT" is not expressible). Below AVX2 the NNUE has no SSE path and falls back to scalar — measured ~17x slower — so v3 is the floor, not a preference. `-InstructionSet native` pins to the build host and **faults on any lesser CPU**; use it only for local play, never for a binary handed to someone else. `-Mode jit` publishes a self-contained single-file managed build that does dispatch at runtime (on par with AOT/v3 within bench noise, ~70 MB larger) — that is also the only mode that can cross-publish to another OS, since AOT links with the host toolchain.

Neither the ISA nor the AOT/JIT choice changes the search tree: node counts are byte-identical across all of these builds, so the nodesweep contract holds for every variant.

## NativeAOT hazards — read before editing

This is the single most important gotcha. The engine is `PublishAot=true`, and several patterns are forbidden because they crash or mis-execute under AOT:

- **Never use `printfn` / `Console.WriteLine` with formatting** — use `Console.Out.WriteLine(s)` with a pre-built string. Every module header repeats this rule. The history is a documented AOT crash.
- **Never read assembly attributes via reflection** — `Engine.fs` keeps `Name`/`Version`/`Author` as `[<Literal>]` strings for exactly this reason. When bumping the version, also update `<Version>` in `Eonego.fsproj` (kept in sync by hand).
- **`TieredCompilation` is off in both the engine and test projects.** The tiered JIT's OSR transition mis-executes the retrograde solver's 524k-iteration scans (intermittent NREs in movegen, test-host crashes). Do not re-enable it.
- **Never hold a `Span` over a `stackalloc` across a long loop** — the stack memory dies on return. Preallocated heap arrays are used in `Retrograde.fs` and `DFPN.fs` for this reason.
- **`#nowarn "9"`** is required at the top of any file using `NativePtr.stackalloc` (already present in the modules that need it).

## Source file compile order matters

F# initializes modules in compile order, and the `.fsproj` `<Compile Include>` order is load-bearing:

```
Engine → Tunables → Bitboard → Move → Zobrist → Accumulator → AccumulatorCache →
Position → Threats → NNUE → Policy → MoveGeneration → GraphFeatures → GraphGpu →
Retrograde → History → MovePick → Transposition → DFPN → Syzygy → Search →
Tooling → UCI → Program
```

`Tunables.fs` is compiled **first** (after `Engine`) so its `envInt` statics exist before any consumer module's init — notably the LMR `Reductions` table built at `Search.fs` module init. Adding a new module means inserting it in the right place in `Eonego.fsproj` AND `Eonego.Tests.fsproj` (tests compile `TestFixtures.fs` first).

## The byte-identity contract

With **no `EONEGO_*` env vars set**, node counts must equal the pre-Tunables binary exactly. This is verified by `scripts/nodesweep.ps1` (1T, fixed depth, 6-FEN audit suite — load- and thermal-independent). `match.py`'s `--a`/`--b` env channels flow per-player, so tuning matches never need a rebuild.

Consequences:

- New `Tunables.fs` defaults must be numerically identical to the old hardcoded literals (old values are kept in trailing comments for the SPSA-tuned ones).
- New search feature flags default to whatever the legacy tree was. The "kitchen-sink" convention (2026-07-08): most `EONEGO_*` feature flags use `<> "0"` (ON unless set to `"0"`); a few pre-SPRT experiments use `= "1"` (opt-in). Check the existing pattern in `UCI.fs:buildConfig` before adding one.
- `EONEGO_FORCE_SCALAR=1` and `EONEGO_FORCE_NOAVX512=1` are the ISA overrides; `EONEGO_FINNY=0` falls back to the from-scratch accumulator refresh path.

## LazySMP / lockless code pattern

Several hot modules (`MoveGeneration.fs`, `MovePick.fs`, `History.fs`) follow a strict pattern:

- **No `let mutable` at module scope.** Module-level mutable state would force a lock under LazySMP.
- **Per-thread heap objects** — `History.Tables` and `Position` are allocated once per worker and never shared.
- **`stackalloc Span<Move>` buffers** owned by the consuming frame. A function may NEVER return a `Span` over its own stackalloc — the memory dies on return.
- **Byref-like structs (`[<Struct; IsByRefLike>] MovePick`)** — never add an instance method that mutates state; it would mutate a copy and the stage would never advance. Use a module function with `byref<MovePick>` (see `MovePick.nextMove`).
- **`byref<int>` write index** threaded through generators (not returned).
- **`#nowarn "9"`** at the top of files using `NativePtr.stackalloc`.

`Transposition.fs` uses Hyatt's XOR-lockless scheme: `Key = realKey ^^^ Data`, two aligned `uint64` Volatile reads, reject as miss on torn read. `InternalsVisibleTo("Eonego.Tests")` is declared here.

## `Eonego.Position` contracts

`Position` is a single `[<Sealed>]` mutable **class** (one heap allocation, reused via make/unmake — NOT copy-make). Undo data lives in a preallocated `StateInfo[MaxPly]` stack indexed by `stPly` (no per-node allocation). Hard rules from the module header:

- A fresh `Position()` is **empty** — must call `LoadFen` / `SetStartPos` / `OfFen` before any `Make`/query.
- `Make` assumes the move is **legal** (no self-check validation — that's MoveGen's job). Passing an illegal move is UB.
- **NOT thread-safe** — each search thread gets its own instance.
- **Never pass `&states.[stPly].Field` as a byref out-param** (binds a copy / trips FS3228). Compute into a `let mutable` local, then assign back.
- Zobrist key convention: `key = XOR(zPiece) ^^^ zCastle ^^^ (zEp file iff ep) ^^^ (Side iff Black)`. `LoadFen` / `Make` / `RecomputeKey` must agree bit-for-bit. `TestFixtures.assertRoundTrips` checks this.

`Position.AccStackLimit = 255` (must stay `<= AccMaxPly - 1` where the type-private `AccMaxPly` is 256). The search's relative `ply` cap (`MaxSearchPly = 246`) alone is insufficient when the root carries many played moves — the accumulator frames index by the **absolute** `stPly`.

## Search feature flags

`Search.fs:108-259` defines `SearchConfig` — a large record of feature flags. `UCI.fs:buildConfig` (around line 219) populates it from `EONEGO_*` env vars. `defaultConfig` at `Search.fs:277` is the managed default. When adding a search feature:

1. Add a `Use*` field to `SearchConfig`.
2. Set it in `defaultConfig`.
3. Read the env var in `UCI.fs:buildConfig` — pick the `<> "0"` (default-on) or `= "1"` (opt-in) convention deliberately, following the comment block at `UCI.fs:244-247`.
4. If it's SPSA-tunable, add it to `Tunables.fs` (with clamps) and to `trainer/spsa.py`'s `PARAMS_WAVE*` table.
5. Gate it behind `UsePruning` if it's a pruning feature (the `UsePruning=false` oracle config in `SearchTests.fs` must stay a pure full-window alpha-beta).

`SearchLimits` (`Search.fs:261`) carries `MoveTime`, `WTime`/`BTime`/`WInc`/`BInc`, `MovesToGo`, `Depth`, `Nodes`, `Infinite`, `Mate`, `Ponder`, `SearchMoves`. Tests build limits with `{ defaultLimits with ... }`.

## Tunables and SPSA

`Tunables.fs` hosts every search margin the SPSA campaign varies, as module-level statics initialized ONCE from `EONEGO_T_*` env vars (unset/unparseable → default, clamped to a safety range). Add a new tunable with `envInt "EONEGO_T_<NAME>" <default> <lo> <hi>` and an old-value trailing comment.

`trainer/spsa.py` is the SPSA driver. Per-wave parameter tables live in `PARAMS_WAVE1`, `PARAMS_WAVE2`, etc. — use `--wave N` and a per-wave checkpoint file (`spsa_state_wN.json`) so a new wave never clobbers an older wave's state. `--fake-objective` runs a synthetic convergence self-test without engines.

## Trainer (Python) — running and parity gates

The trainer scripts are stdlib + `torch`/`numpy`/`chess`. Run with `uv` (the canonical invocation in the script headers):

```
uv run --with torch,numpy,chess python trainer/<script>.py ...
```

Key scripts:

- **`match.py`** — UCI match driver + SPRT. `--a "EONEGO_*=VAL,..."` / `--b "..."` per-player env channels; `--exe` / `--exe-b` for old-vs-new binaries; `--concurrency N` runs N worker threads each owning one engine pair (color-swapped openings on the same worker so throttling penalizes both players symmetrically). `run_match(args)` is importable (used by `spsa.py`).
- **`loop.py` / `teacherloop.py` / `policy_loop.py`** — end-to-end training orchestrators (gen → label → train → parity → match). Each stage is skipped when its output exists (delete a stage's output to redo it), so interrupted runs resume.
- **`parity_check.py`** — the Phase-1 gate: Python `encoder.py` vs engine `dumpft`, byte-exact. Pin this as a CI check.
- **`policy_parity.py`** — same idea for the policy head: engine `dumppolicy` vs `policy_intref.py` integer reference forward. The FEN set MUST include Black-to-move and promotion-heavy positions (STM mirroring is the classic silent skew).
- **`tactics.py`** — scores a config on the teacher-verified KGA tactical suite (fraction of positions where the engine plays the verified best move at a fixed budget).

Default exe paths in the trainer assume `Eonego/bin/Release/net10.0/[win-x64/publish/]Eonego.exe` — override with `--exe` if needed. `run_pool_sprt.cmd` is the nightly queue (Task Scheduler, sequential SPRT runs).

## Nets and embedding

`Eonego.fsproj` conditionally embeds three nets as manifest resources (conditional on the file existing so a sparse checkout still builds):

| File | Resource name | Loaded when |
|---|---|---|
| `nets/main.nnue` | `eval.nnue` | Always (if present) — the NNUE trunk |
| `nets/main.policy` | `policy.dat` | `EONEGO_POLICY=1` (reads embedded) or `=<path>` (reads file) |
| `nets/main.ownpolicy` | `ownpolicy.dat` | `EONEGO_POLICY=own1` — only fires at ≤6-piece positions |

Runtime overrides (no rebuild): `EONEGO_NET=<path>`, `EONEGO_POLICY=<path>`. Tests that need eval **soft-skip** when `main.nnue` is absent (see `TestFixtures.tryLoadNet`, which walks up to find `Eonego.slnx` then `nets/main.nnue`).

The NNUE is a **FullThreats** architecture (version `0x6A448AFA`): dual-input HalfKAv2_hm + threat features, L1=1024 accumulator with 8 PSQT buckets, finny refresh + fused delta updates + SIMD kernels. `Threats.fs` uses the NNUE reference piece encoding (PAWN=1..KING=6, `make_piece=(c<<3)+pt`) — NOT the Eonego `color*6+type` — because the index LUTs are generated in it. Conversion happens at the enumeration boundary.

Policy head (EONPOL02, `Policy.fs`) is piece-aware hidden-decomposed from→to sharing the NNUE trunk's 1024-wide FT buffer. Squares are **STM-relative** (Black flipped, `s ^^^ 56`) — MUST match `trainer/move_encoder.py`.

## Graph evaluator (CUDA bridge)

`GraphGpu.fs` P/Invokes `native/eonego_cuda/eonego_cuda.dll`. The model format is EONGR01 (36-byte engine header + explicit EONGRW1 tensor table). The public contract is **fallback-first**: search keeps using NNUE whenever native inference is unavailable or `TryEvaluate` returns `ValueNone`.

Runtime gates:

- `EONEGO_GRAPH=<path>` — load an EONGR01 graph model.
- `EONEGO_GRAPH_MODE=leaf|policy|all` (default `leaf`) — `policy`/`all` feed graph logits into negamax move ordering and LMR policy slots (no MCTS).
- `EONEGO_GRAPH_CUDA=0` — force the scalar CPU backend.
- `EONEGO_GRAPH_LEARNED=0` — disable the learned GraphNet forward path, use the bring-up graph-feature evaluator.
- `eogpu_backend` / `eogpu_model_status` report the selected backend and whether the loaded table is a complete GraphNet contract.

The DLL is built with CMake (no `nvcc`/MSVC `cl.exe` required — it dynamically loads CUDA/NVRTC):

```powershell
cmake -S native/eonego_cuda -B artifacts/eonego_cuda_build
cmake --build artifacts/eonego_cuda_build --config Release
```

Put the build dir on `PATH` or copy `eonego_cuda.dll` next to the engine exe.

## Testing approach

- **xUnit**, `[<Fact>]` and `[<Theory>]` with `[<InlineData>]`. ~408 tests across 38 files.
- **`TestFixtures.fs`** is compiled first and exposes `perftFens` (the 6 CPW canonical positions), `Snap`/`snap`/`assertRoundTrips` (Make/Unmake round-trip + incremental key == from-scratch), and `tryLoadNet` (soft-skip when the net is absent).
- **Perft gate** (`MoveGenerationTests.fs`): d1–d4 in the default tier; d5/d6 behind `[<Trait("Category", "Slow")>]`. CI excludes `Slow` — the slow tier is a known parallel-load flake that passes in isolation.
- **Search oracle** (`SearchTests.fs`): `refMinimax` is exhaustive negamax (no alpha-beta, no TT, no ordering) topped with the engine's own `qsearch`. `oracleCfg` pins `UseTt=false`, `UsePruning=false`, `UseRetro=false` — any new search feature that breaks the oracle must be gated to run only when `UsePruning=true` (or explicitly pinned off in `oracleCfg`).
- **Worker construction** in tests: `makeWorker fen cfg` builds a `SearchControl` + `Worker`, calls `SetupRoot()`, `Reset()`, `StartClock 0L 0L`. Match this pattern.
- Use `{ defaultLimits with ... }` and `{ defaultConfig with ... }` for test configs.

## Code conventions

From `.editorconfig` and observed style:

- **Indent**: 4 spaces for `.fs`; 2 spaces for `.fsproj`/`.slnx`/`.json`/`.yml`/`.md`.
- **Line length**: 120 max. LF endings. UTF-8. Trim trailing whitespace. Final newline at EOF.
- **Open ordering** (Fantomas): `System` → `Microsoft` → external libraries → project modules. Keep blank lines between groups manually (Fantomas preserves but won't insert them).
- **Module header XML doc** (`///`) on every `.fs` file — explains purpose, contracts, AOT notes, LazySMP notes. Read these first when entering a module; they encode the invariants.
- **Type aliases for primitives**: `Bitboard = uint64`, `Square = int`, `Color = int`, `PieceType = int`, `Piece = int` (see `Bitboard.fs:23-27`). `[<Literal>]` for constants (White=0, Black=1, Pawn=0, ... Knight=1, Bishop=2, Rook=3, Queen=4, King=5; `Piece = color*6 + pieceType`, `12 = NoPiece`).
- **Squares are LERF** (Little-Endian Rank-File): a1=0, h1=7, a2=8, h8=63. `rank = sq >>> 3`, `file = sq &&& 7`. `north = bb <<< 8`.
- **`module Eonego.X`** naming — every file is a module under the `Eonego` namespace.
- **Fail-soft** throughout the search.
- **No comments explaining *what*** — comments explain *why* (the why of a tuning value, the AOT hazard, the LazySMP invariant). Existing comments are dense with audit dates and SPRT outcomes; preserve them.

## Validation flow when changing search

1. **Tree mechanics**: `scripts/nodesweep.ps1` over the 6-FEN audit suite at d13/d14/d15, 1T. Compare node totals before/after. With no env vars set, totals must be byte-identical (the contract).
2. **Playing strength**: `trainer/match.py --sprt` self-play, typically `--movetime 100 --openings 200 --concurrency 8` for fast SPRT, or `--tc 10+0.1` for real-clock. Use `--a "EONEGO_*=VAL"` for the experimental arm, `--b ""` for baseline.
3. **Tactical suite**: `trainer/tactics.py --movetime 500` — fraction of KGA suite positions where the engine plays the teacher-verified best move.
4. **Nightly queue**: `trainer/run_pool_sprt.cmd` runs the queued SPRT matches sequentially via Task Scheduler.

## Common gotchas

- **Don't re-enable `TieredCompilation`** in either fsproj. The retrograde solver and DFPN mis-execute under OSR.
- **Don't add `let mutable` at module scope** in `MoveGeneration`/`MovePick`/`History` — it breaks LazySMP locklessness.
- **Don't return a `Span` over a `stackalloc`** from a function — the stack memory dies on return.
- **Don't add instance methods to `MovePick`** (or other `[<Struct; IsByRefLike>]` types) that mutate state — they mutate a copy. Use a module function with `byref<MovePick>`.
- **Don't use `printfn`** — `Console.Out.WriteLine` only. Every module header repeats this.
- **Don't pass `&states.[stPly].Field` as a byref out-param** in `Position` (FS3228 / copy binding).
- **Don't change `<Compile Include>` order** in `Eonego.fsproj` without considering module init order (`Tunables` before everything that reads it; `Accumulator` before `Position`).
- **Don't bump `Engine.fs` `Version` without also bumping `<Version>` in `Eonego.fsproj`** — the fsproj block only stamps Windows file metadata, but it's kept in sync by hand.
- **Don't change NNUE piece encoding in `Threats.fs`** — it deliberately uses the reference encoding (PAWN=1..KING=6, `make_piece=(c<<3)+pt`) because the index LUTs are generated in it. Convert at the boundary.
- **Don't change policy square mirroring** (`s ^^^ 56`, STM-relative) without also changing `trainer/move_encoder.py` — they must match.
- **Don't rely on `main.nnue` being present** in tests — soft-skip via `tryLoadNet`. The CI runner checks out with LFS, but local sparse checkouts may not.
- **`SyzygyPath` is inert until set** via `setoption` — `Syzygy.Largest = 0` gates every probe, so `UseSyzygy` default-on preserves byte-identical classic search.
- **DFPN never stops a normal search** — only under `go mate N`. It only overrides the final bestmove on a VERIFIED proof (the `Verify` pass replays the claim with a strict full-window proof search, no table, path repetition = fail, rule-50 = fail).

## Where to look first

- **`README.md`** — architecture overview, UCI options, build instructions.
- **`Eonego/Engine.fs`** — identity literals (Name, Version, Author).
- **`Eonego/Tunables.fs`** — every SPSA-tunable margin, with old-value comments and audit dates.
- **`Eonego/Search.fs:108-259`** — `SearchConfig` fields (annotated with the rationale and SPRT status of each feature).
- **`Eonego/UCI.fs:219-300`** — `buildConfig`, where `EONEGO_*` env vars map to config fields.
- **`Eonego/Position.fs`** header — the mutable Position contracts (read before touching make/unmake).
- **`Eonego/MoveGeneration.fs`** + **`MovePick.fs`** headers — the LazySMP / stackalloc / byref-like contracts.
- **`native/eonego_cuda/README.md`** — the CUDA bridge ABI and runtime gates.
- **`nets/README.md`** — net embed/runtime-override reference.
- **`trainer/match.py`** header — the match driver CLI and concurrency model.
- **`trainer/spsa.py`** header — the SPSA scheduler and per-wave state files.
