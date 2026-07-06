"""Merge gen-v2 shards into one training corpus, deduplicated. Stdlib only.

Dedupe key = the first four FEN fields (board, stm, castling, ep) — move counters excluded, so the
same position reached in different games/shards keeps only its first record. Records without the
4th best_uci field (v1 files) are dropped: they carry no policy target.

    python policy_merge.py --out data/pol1/merged.txt data/pol1/gen_*.txt
"""

import argparse
import glob
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()

    paths = []
    for pat in args.inputs:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        sys.exit("no input files")

    seen = set()
    kept = dropped_dup = dropped_nobest = 0
    with open(args.out, "w", encoding="utf-8") as out:
        out.write("# eonego gen v2 merged fen;cp_white;result_white;best_uci\n")
        for path in paths:
            with open(path, encoding="utf-8-sig") as f:
                for line in f:
                    t = line.strip()
                    if not t or t.startswith("#"):
                        continue
                    parts = t.split(";")
                    if len(parts) < 4 or not parts[3].strip():
                        dropped_nobest += 1
                        continue
                    key = " ".join(parts[0].split()[:4])
                    if key in seen:
                        dropped_dup += 1
                        continue
                    seen.add(key)
                    out.write(t + "\n")
                    kept += 1
    print(f"{len(paths)} shards -> {kept} unique records ({dropped_dup} dups, {dropped_nobest} no-best dropped)")


if __name__ == "__main__":
    main()
