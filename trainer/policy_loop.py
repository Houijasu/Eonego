"""End-to-end policy pipeline orchestrator (teacherloop.py shape): gen -> dumpft -> train ->
export -> parity -> rankeval. Each stage is skipped when its output already exists (delete a
stage's output to redo it), so an interrupted run resumes.

    uv run --with torch,numpy,chess python policy_loop.py \
        --exe ../Eonego/bin/Release/net10.0/win-x64/publish/Eonego.exe \
        --net ../nets/main.nnue --work data/pol1 \
        --games 2000 --depth 10 [--cutdump data/cuts.tsv] [--epochs 30]

The SPRT stage is deliberately NOT automated here — promotion matches are run via match.py with
EONEGO_POLICY on one arm, wall-clock TC (plan: fixed-nodes hides the time tax).
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, env=None):
    print("+ " + " ".join(cmd), flush=True)
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e)
    if r.returncode != 0:
        sys.exit(f"stage failed ({r.returncode})")


def stage(path, name):
    if os.path.exists(path):
        print(f"[skip] {name}: {path} exists")
        return False
    print(f"[run ] {name} -> {path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--work", required=True, help="working directory for artifacts")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--start", default=None, help="gen --start FEN (default: gen's default)")
    ap.add_argument("--cutdump", default=None, help="pre-recorded EONEGO_CUTDUMP tsv for rankeval")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    gen_txt = os.path.join(args.work, "gen.txt")
    ft_bin = os.path.join(args.work, "ft.bin")
    npz = os.path.join(args.work, "head.npz")
    sidecar = os.path.join(args.work, "main.policy")
    py = [sys.executable]

    if stage(gen_txt, "gen"):
        cmd = [args.exe, "gen", "--out", gen_txt, "--net", args.net, "--games", str(args.games),
               "--depth", str(args.depth), "--seed", str(args.seed)]
        if args.start:
            cmd += ["--start", args.start]
        run(cmd)

    if stage(ft_bin, "dumpft"):
        run([args.exe, "dumpft", "--net", args.net, "--in", gen_txt, "--out", ft_bin])

    if stage(npz, "train"):
        run(py + [os.path.join(HERE, "policy_train.py"), "--data", gen_txt, "--dump", ft_bin,
                  "--out", npz, "--net", args.net, "--epochs", str(args.epochs), "--seed", str(args.seed)])

    if stage(sidecar, "export"):
        run(py + [os.path.join(HERE, "policy_export.py"), "--npz", npz, "--net", args.net, "--out", sidecar])

    # Parity gate (always; cheap): reuse the gen fens as the parity set — gen games include
    # Black-to-move throughout; add promotion-heavy fens to the file for full coverage.
    run(py + [os.path.join(HERE, "policy_parity.py"), "--exe", args.exe, "--net", args.net,
              "--policy", sidecar, "--fens", gen_txt, "--tmp", args.work])

    if args.cutdump:
        run(py + [os.path.join(HERE, "policy_rankeval.py"), "--exe", args.exe, "--net", args.net,
                  "--policy", sidecar, "--cutdump", args.cutdump, "--tmp", args.work])

    print("pipeline complete: " + sidecar)
    print("next: SPRT via match.py with --a EONEGO_POLICY=" + sidecar + " (wall-clock TC)")


if __name__ == "__main__":
    main()
