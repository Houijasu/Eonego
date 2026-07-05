"""Extract the dump/label subset covered by a (possibly partial) teacher checkpoint.

The stack tuner needs --dump and --labels row-aligned. While a checkpointed labeling run is
still in flight, only some of the base rows have teacher labels; training on the full base
would silently fall back to the weak self-play cp for the rest. This tool slices BOTH files
to exactly the teacher-covered rows, preserving base (= dump) order:

    python subset.py --base data/kga26_labeled2.txt --teacher data/kga26_eteacher2.txt \
                     --dump data/kga26_dump2.bin --out-prefix data/kga26_part

writes <prefix>.bin (1034-byte records) and <prefix>.txt (fen;cp_teacher;result_white).
result_white comes from the base self-play record (the teacher checkpoint stores 0.5).
"""

import argparse

import numpy as np

REC = np.dtype([("bucket", "u1"), ("stm", "u1"), ("psqt", "<i4"),
                ("eval", "<i4"), ("ft", "u1", (1024,))])

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="fen;cp_self;result (order matches --dump)")
ap.add_argument("--teacher", required=True, help="checkpointed fen;cp_teacher;0.5 (any order)")
ap.add_argument("--dump", required=True)
ap.add_argument("--out-prefix", required=True)
args = ap.parse_args()

teacher = {}
with open(args.teacher, encoding="utf-8") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split(";")
        if len(parts) >= 2:
            teacher[parts[0]] = parts[1]

idx, rows = [], []
with open(args.base, encoding="utf-8") as f:
    i = 0
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        fen, _cp_self, res = line.strip().split(";")
        cp = teacher.get(fen)
        if cp is not None:
            idx.append(i)
            rows.append(f"{fen};{cp};{res}\n")
        i += 1

recs = np.fromfile(args.dump, dtype=REC)
assert i == recs.size, f"base/dump mismatch: {i} vs {recs.size}"
sel = recs[np.array(idx, dtype=np.int64)]
sel.tofile(args.out_prefix + ".bin")
with open(args.out_prefix + ".txt", "w", encoding="utf-8") as f:
    f.writelines(rows)

buckets = np.bincount(sel["bucket"], minlength=8)
print(f"subset: {len(idx)}/{recs.size} rows -> {args.out_prefix}.bin/.txt")
print("bucket counts:", " ".join(str(int(b)) for b in buckets))
