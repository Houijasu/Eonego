"""Generate 3-man Syzygy graph data, train GraphNet, and export EONGR01.

Example:
    uv run --with torch,numpy,chess python trainer/graph_syzygy_train.py \
        --tb C:/Syzygy --out-dir data/graph_syzygy3 --epochs 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_SIGNATURES = "KQvK,KRvK,KBvK,KNvK,KPvK"


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def syzygy_dirs(root: Path) -> str:
    """Return comma-separated directories for python-chess Syzygy open/add_directory."""
    if not root.exists():
        raise SystemExit(f"Syzygy path does not exist: {root}")

    def has_tables(p: Path) -> bool:
        return any(p.glob("*.rtbw")) or any(p.glob("*.rtbz"))

    if has_tables(root):
        return str(root)

    dirs = [p for p in root.iterdir() if p.is_dir() and has_tables(p)]
    wdl = [p for p in dirs if "wdl" in p.name.lower()]
    dtz = [p for p in dirs if "dtz" in p.name.lower()]
    ordered = wdl + dtz + [p for p in dirs if p not in wdl and p not in dtz]
    if not ordered:
        raise SystemExit(f"No .rtbw/.rtbz files found under {root}")
    return ",".join(str(p) for p in ordered)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    trainer = repo / "trainer"

    ap = argparse.ArgumentParser()
    ap.add_argument("--tb", default=r"C:\Syzygy", help="Syzygy root or comma-separated table dirs")
    ap.add_argument("--out-dir", default=str(repo / "data" / "graph_syzygy3"))
    ap.add_argument("--signatures", default=DEFAULT_SIGNATURES)
    ap.add_argument("--per-signature", type=int, default=2000)
    ap.add_argument("--total", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tries-mult", type=int, default=100)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--configuration", default="Debug", help="dotnet configuration for dumpgraph")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-dump", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = out_dir / "syzygy3.txt"
    dump = out_dir / "syzygy3_graph.bin"
    ckpt = out_dir / "graph_syzygy3.pt"
    model = out_dir / "graph_syzygy3.eongr01"

    tb_arg = args.tb
    if "," not in tb_arg:
        tb_arg = syzygy_dirs(Path(tb_arg))

    if not args.skip_generate:
        run(
            [
                sys.executable,
                str(trainer / "policy_endgen.py"),
                "--tb",
                tb_arg,
                "--out",
                str(records),
                "--signatures",
                args.signatures,
                "--per-signature",
                str(args.per_signature),
                "--total",
                str(args.total),
                "--seed",
                str(args.seed),
                "--max-tries-mult",
                str(args.max_tries_mult),
            ],
            repo,
        )

    if not args.skip_dump:
        run(
            [
                "dotnet",
                "run",
                "--project",
                str(repo / "Eonego" / "Eonego.fsproj"),
                "-c",
                args.configuration,
                "--",
                "dumpgraph",
                "--in",
                str(records),
                "--out",
                str(dump),
            ],
            repo,
        )

    if not args.skip_train:
        run(
            [
                sys.executable,
                str(trainer / "graph_train.py"),
                "--data",
                str(records),
                "--dump",
                str(dump),
                "--out",
                str(ckpt),
                "--d-model",
                str(args.d_model),
                "--layers",
                str(args.layers),
                "--epochs",
                str(args.epochs),
                "--batch",
                str(args.batch),
                "--lr",
                str(args.lr),
                "--policy-weight",
                str(args.policy_weight),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
            ],
            repo,
        )

    if not args.skip_export:
        run(
            [sys.executable, str(trainer / "graph_export.py"), "--ckpt", str(ckpt), "--out", str(model)],
            repo,
        )

    print(f"records: {records}")
    print(f"dump:    {dump}")
    print(f"ckpt:    {ckpt}")
    print(f"model:   {model}")


if __name__ == "__main__":
    main()
