# Policy-net campaign — Phase 0 measurement (2026-07-06)

Instrumentation: `EONEGO_PROF=1` tree-shape counters (PosProf in `Position.fs`, written by
`Search.fs` and `MovePick.fs`, dumped as `info string prof4/prof5/prof6`). Gate verified:
nodesweep d13-14 byte-identical PROF-off vs PROF-on; 349/349 tests green.

Run: 1T, `go depth 16`, JIT scratch build (`scratch-build/prof0`), net = embedded `nets/main.nnue`.
All percentages are ratios, robust to binary speed (AOT publish is faster in absolute terms).

## Tree shape

| pos      | nodes     | nps     | nodesMain | quietInit | qi % of main | qi % of ALL | cutoffs | cutTT% | cutCap% | cutRef% | **cutQuietTail%** |
|----------|-----------|---------|-----------|-----------|--------------|-------------|---------|--------|---------|---------|-------------------|
| startpos | 273,986   | 396,506 | 51,119    | 15,351    | 30.0         | 5.6         | 35,532  | 23.2   | 35.9    | 35.0    | **5.9**           |
| midgame  | 1,288,367 | 449,848 | 233,289   | 62,106    | 26.6         | 4.8         | 169,422 | 17.4   | 44.1    | 33.2    | **5.3**           |
| kiwipete | 884,958   | 438,098 | 123,974   | 35,911    | 29.0         | 4.1         | 73,725  | 27.9   | 58.8    | 8.9     | **4.4**           |
| cpw6-mid | 634,464   | 454,487 | 88,416    | 26,817    | 30.3         | 4.2         | 56,991  | 25.2   | 50.3    | 19.7    | **4.7**           |

`quietInit` = main-picker nodes that actually generated + scored quiets (`StgQuietInit`, not
skipQuiets) — the nodes that would pay a policy inference. `cutQuietTail` = beta cutoffs caused by
a **non-refutation quiet** — the only cutoff class a policy ordering prior can address (TT move,
good captures, killers, countermove are emitted by earlier picker stages regardless of ordering).

Within cutQuietTail, share already cut at tried-quiet index ≤ 3 (i.e., history already ranks the
cutter near the front): startpos 81.2%, midgame 81.0%, kiwipete 78.5%, cpw6 80.0%. Cuts at index
≥ 4 — the genuinely reorderable mass — are **~1% of all cutoffs**.

## Cost side

- ns/eval incl. lazy accumulator materialization: 2.55–2.78 µs (`evalMs/nEval`).
- Pure NNUE forward ≈ (evalMs − ensureMs)/nEval ≈ **0.65–0.70 µs** — the yardstick for a policy
  inference (ftProduct recompute + pfc0 1024→128 + 2×128→64 ≈ one forward, ~0.7–1.2 µs), plus
  forced materialization walks at TT-eval-hit nodes (unbudgeted by the baseline, 0.3–1.5 µs when hit).
- Node cost = 1/nps ≈ 2.2–2.5 µs. Policied nodes = 4.1–5.6% of all nodes ⇒ estimated nps tax of a
  lazy StgQuietInit-only policy ≈ **1.6–2.7%** (excl. forced walks; ~2–4% with them).
- Derived break-even node reduction: **~2–4%**.

## Verdict — the pre-registered re-scope gate FIRES

The plan's gate: if the addressable cutoff slice is under ~15%, ordering-only cannot pay for
itself. Measured: **4.4–5.9%**, of which ~80% already cuts at index ≤ 3. The theoretical ceiling of
perfect quiet-tail reordering is on the order of the ~2–4% break-even — expected SPRT [0,5] gain ≈ 0.

Caveat recorded for honesty: this metric measures only direct cutoff moves. Better quiet ordering
also lowers the `moveCount` of good moves (less LMR reduction via `Reductions[depth, moveCount]`)
and raises alpha earlier at PV/ALL nodes — effects this counter cannot see. Ordering is demoted,
not disproven.

## Decision (per plan Phase 0 abort/re-scope clause)

- **v1 policy consumption = LMR term** (`PolLmrThresh`: low-policy quiets get one extra reduction
  step; high-leverage, compounds exponentially, acts on every searched quiet — not just cutters),
  followed by the policy override of history pruning as its own increment.
- The `scoreQuiets` ordering blend stays implemented behind its own inert tunable (`PolOrdMul`
  default 0) — cheap to carry, its SPRT runs only after the LMR term is judged.
- Phase 1 scaffold (ftInto, Policy.fs, per-ply logit buffers, dumppolicy, parity) is unchanged —
  the logit arrays serve LMR/pruning reads exactly as they serve ordering.

# Phase 1 gate results (2026-07-06, same session)

Scaffold shipped default-OFF: `NNUE.ftInto`/`a1FromFt` taps (evalFromAcc untouched — value forward
byte-identical by construction), `Policy.fs` (EONPOL01 loader ftHash-bound to the trunk, 3-branch
kernels, `fillLogits`, `evalWDL`), lazy fill in MovePick `StgQuietInit` (Zobrist-guarded per-ply
Worker buffers), LMR term (`EONEGO_T_POL_LMR`) + inert ordering blend (`EONEGO_T_POL_ORDMUL`=0),
`EONEGO_POLICY` env gate in UCI, `dumppolicy` subcommand, `PolicyTests.fs` (11 tests).

- **OFF byte-identity:** nodesweep d13-14 byte-identical vs the pre-policy binary (verified twice,
  including after the bug fix below).
- **Kernel parity:** scalar == AVX2 == VNNI bit-exact on random weights/inputs (8 trials).
- **Bug found & fixed by the random-weights probe:** the LMR read site runs after `pos.Make`, so
  guarding on `pos.Key` there compared the CHILD key against the parent-filled slot — the term was
  silently dead (tree byte-identical to OFF while the fill provably ran; the exact failure class the
  bare-bool critique predicted, in key-guard clothing). Fixed by capturing `policyNodeKey` before
  the move loop; pinned by the `parent-key guard regression` test. Verified live: random logits at
  threshold 0 perturb the tree; threshold +inf (reduce every quiet) shrinks it 262k -> 242k @ d13.
- **Random-weights cost bench** (pure inference tax: threshold -1M so the term never fires — tree
  identical to OFF, nps delta = the fill cost): midgame **3.2%**, kiwipete **0.8%** @ d15 1T JIT.
  Matches the Phase-0 estimate (1.6-2.7% excl. forced walks). Break-even for the LMR/pruning
  consumption stands at ~2-4% tree reduction.
- Pending: AOT publish smoke (deferred — the scaffold is default-OFF and ships nothing until the
  Phase-3 SPRT; run `publish.ps1` + `EONEGO_POLICY` smoke before any match arm uses the AOT exe).

# Phase 2 pipeline results (2026-07-06, same session)

Engine side: `gen` v2 emits the previously-discarded search best move as a backward-compatible 4th
field (`fen;cp_white;result_white;best_uci`) and the prefix softmax degeneracy is fixed (one root
score was broadcast to all legal moves ⇒ uniform-random regardless of `--temp`; now each prefix
move is scored by post-move static eval). New `EONEGO_CUTDUMP=<path>` tap: at each beta cutoff by
a non-refutation quiet (depth ≥ 3) it appends `fen \t depth \t cutter \t quiet:histScore ...` —
the matched-state baseline the quality gate ranks against. Gates re-verified: 360/360 tests,
nodesweep byte-identical vs the pre-campaign baseline (cutdump/policy env unset).

Trainer side (all new): `move_encoder.py` (STM-relative UCI codec), `policy_dataset.py`
(gen+dumpft alignment, quiet-conditional masks via python-chess, frozen-stack a1 for WDL; note the
BOM: gen files need `utf-8-sig`), `policy_model.py` (QAT torch heads, STE round/floor on the
engine's exact int grid), `policy_train.py` (masked quiet-CE + λ·WDL-CE), `policy_export.py`
(EONPOL01 writer, ftHash from the .nnue header), `policy_intref.py` (stdlib integer oracle),
`policy_parity.py` (engine dumppolicy == intref, bit-exact), `policy_rankeval.py` (policy rank vs
history rank on cutdump states — the go/no-go), `policy_loop.py` (resumable orchestrator).

End-to-end smoke (3 games @ d6, 175 positions): train 8 epochs → export → **PARITY OK (175 rows,
0 mismatch)** → cutdump 716 states from two d14 searches → rankeval runs. The smoke head loses to
history as expected (top-1 0.017 vs 0.295); the measured bar for a real head on this state class:
**history top-1 ≈ 0.30, top-3 ≈ 0.58, MRR ≈ 0.48**.

Next (user-scheduled compute): at-scale corpus — ≥2M diverse positions beyond KGA (`gen` at d9-12,
wide `--start` book), internal-node harvest, optional d16 Eonego-teacher subset via label.py, wide
cutdump capture — then `policy_loop.py`, rankeval gate, and only if policy beats history: Phase 3
wall-clock SPRT with the LMR term.
