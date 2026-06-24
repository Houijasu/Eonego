# Third-party notices

Eonego is licensed under the **GNU Affero General Public License, version 3 or later
(AGPL-3.0-or-later)** — see [`LICENSE`](LICENSE).

## Why AGPL-3.0

Eonego is intended to incorporate and/or combine with neural networks (and, where
applicable, derived inference code) from the following projects:

| Project | Upstream | License | Used for |
|---|---|---|---|
| **Jackal** | <https://github.com/TomaszJaworski777/Jackal> | **GPL-3.0** (engine + networks) | policy network (the shipped `.network` weights) |
| **Monty** | <https://github.com/official-monty/Monty> | AGPL-3.0 | origin of the policy move-index/feature **encoding** (Jackal copied it; we reimplement it) |
| **Stockfish** | <https://github.com/official-stockfish/Stockfish> | GPL-3.0 (engine source) / **CC0-1.0 (net files)** | NNUE value network |

> **Policy source pivoted to Jackal (2026-06-23):** Monty's net distribution server was unreachable.
> Jackal's policy net (`p8008192009q.network`, on HuggingFace `Snekkers/networks`) uses the *same*
> 3072-input / 7840-move-index encoding (its source notes the output mapping is copied from Monty),
> so the port is identical, but the **weights we actually ship are Jackal's (GPL-3.0)** — no Monty
> weights are distributed. The Stockfish value net stays (CC0).

**Licensing note.** Because no Monty *weights* are shipped — only Jackal's GPL-3.0 policy weights, the
CC0 Stockfish value net, and a clean-room reimplementation of the (Jackal/Monty) policy encoding — the
combined work can be distributed under **GPL-3.0** (Jackal's license) rather than the stricter AGPL-3.0.
The repo is currently licensed **AGPL-3.0** (the safe superset that also satisfies GPL-3.0 and respects
the Monty lineage of the encoding); relaxing to GPL-3.0 (dropping AGPL's network-use source-disclosure
clause) is a defensible option now that Monty is out of the dependency set. **The Stockfish facts below
are unchanged:**

- **Stockfish network files (`.nnue`) are CC0-1.0 (public domain)** — see the
  `official-stockfish/networks` repo. CC0 imposes **no copyleft and no attribution**. Eonego
  implements the NNUE inference **clean-room from the published format/architecture spec** (the
  `nnue-pytorch` docs and `serialize.py`), *without copying Stockfish's GPL C++ source*, so the
  Stockfish value-net half carries **no GPL obligation**. (If we ever copy/translate SF's C++
  inference directly, that derived code would be GPL-3.0 and must be treated as such.)
- **Monty is AGPL-3.0** for both its engine source *and* its trained networks. Porting Monty's
  policy architecture / `map_move_to_index` / `map_features` logic is translating AGPL code, and
  shipping Monty's `.network` weights distributes an AGPL deliverable. Either one makes the
  combined work **AGPL-3.0**, including the §13 obligation to provide complete corresponding
  source to users who interact with it **over a network**.

So: were Eonego to use *only* the Stockfish (CC0) net, it would need no copyleft license at all;
it is the **Monty policy net that requires AGPL-3.0**, and AGPL therefore governs the whole
combined work. Each incorporated network and any upstream-derived code remains under its own
license; this notice and the upstream license files must be preserved in any redistribution.

## Status

This is a **licensing change made in anticipation of** integrating those networks. The
technical integration is separate, not-yet-completed work:

- **Stockfish NNUE** uses the `HalfKAv2_hm` feature transformer — a different architecture
  from Eonego's native region-piececount net (`2577→64→32→16→16→1`). Using it requires a
  HalfKA inference implementation; an SF `.nnue` cannot be loaded by the current loader.
- **Monty policy** is a Rust project with its own network format and move encoding; using
  it requires a matching policy-inference implementation.

When a third-party network is actually shipped or downloaded by Eonego, record the exact
upstream commit/revision and the network file's own license alongside this file.

## Obligations summary (informational, not legal advice)

- Distribute Eonego (and any combined work) under **AGPL-3.0-or-later**, with complete
  corresponding source.
- If Eonego is **made available to users over a network**, offer those users the complete
  corresponding source (AGPL §13).
- Preserve all copyright notices and license texts of incorporated upstream components.

This summary is provided for convenience only and is **not legal advice**; the controlling
terms are those in [`LICENSE`](LICENSE) and the upstream projects' own license files.
