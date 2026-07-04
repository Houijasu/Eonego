"""Join teacher cp labels onto the base self-play records, preserving base line order.

Base:    fen;cp_self;result_white   (order matches the dumpft dump — MUST be preserved)
Teacher: fen;cp_teacher;0.5         (order/coverage arbitrary)
Out:     fen;cp_teacher;result_white  with cp_self fallback where the teacher skipped.

    python join_labels.py --base data/kga26_labeled1.txt --out data/kga26_teacherlab.txt data/kga26_teacher_*.txt
"""

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("teacher", nargs="+")
args = ap.parse_args()

teacher = {}
for path in args.teacher:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(";")
            if len(parts) >= 2:
                teacher[parts[0]] = parts[1]

n, missing = 0, 0
with open(args.base, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        fen, cp_self, res = line.strip().split(";")
        cp = teacher.get(fen)
        if cp is None:
            missing += 1
            cp = cp_self
        out.write(f"{fen};{cp};{res}\n")
        n += 1

print(f"joined {n} records ({missing} teacher-missing, self-cp fallback) -> {args.out}")
