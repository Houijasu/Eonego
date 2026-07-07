"""OWN-TRUNK policy network — trained end-to-end from BOARD features on tbgen tablebase labels.

The frozen-NNUE-trunk sidecar saturated at ~0.69 discriminative preserve regardless of head width
(H=256 == H=512): the value net's features simply don't carry everything move choice needs. This
model owns its trunk: a 768-feature one-hot board encoding (12 STM-relative piece planes x 64
STM-relative squares — an NNUE-style feature transformer seed) -> summed embedding -> MLP ->
piece-aware from/to heads (384 each). Float training (no quantization yet — EONPOL03 engine
integration quantizes later); the deliverable here is the LEARNING ceiling.

    uv run --python 3.12 --with torch,numpy python policy_own.py \
        --data data/pol1/tbgen345_big.txt --out data/pol1/own_trunk.pt \
        [--trunk 1024] [--layers 2] [--epochs 24]

Corpus positions have no castling rights and no en-passant square by construction (tbgen random
placements), so 768 features identify each position exactly; labels are deterministic (Bayes
error 0) — every accuracy shortfall is model or data coverage, never noise.
"""

import argparse

import numpy as np

from move_encoder import PIECE_TYPE, board_of_fen, encode_uci_piece, fen_black_to_move

QMAX = 72
MAXPC = 7  # <= 6 piece-square features (5-man corpora) + 1 joint king-pair feature
# Feature space: 12 piece planes x 64 squares, then a JOINT king-pair table (ownK*64+oppK,
# STM-relative). Opposition/key-square geometry is a lookup in the joint table but nearly
# incomputable from independent king features — the round-3 lever after width AND depth saturated.
NFEAT = 768 + 4096
PAD = NFEAT  # EmbeddingBag padding index


def stream_build(path, qmax: int = QMAX, chunk: int = 250_000, skip: int = 0, take: int = 0):
    """Stream the corpus straight into numpy chunks (never a Python list-of-lists — the 8.7M-row
    corpus peaked ~20 GB that way and crashed this box). -> dict of arrays for discriminative rows:
    feat (N,MAXPC) i16  one-hot feature indices (plane*64 + relSq; -1 padded)
    qf/qt (N,qmax) i16  piece-aware head indices; qn i16; good (N,qmax) bool"""
    chunks = []

    def new_buf():
        return {
            "feat": np.full((chunk, MAXPC), -1, dtype=np.int16),
            "qf": np.zeros((chunk, qmax), dtype=np.int16),
            "qt": np.zeros((chunk, qmax), dtype=np.int16),
            "qn": np.zeros(chunk, dtype=np.int16),
            "good": np.zeros((chunk, qmax), dtype=bool),
        }

    buf = new_buf()
    fill = 0
    total = 0
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            parts = t.split(";")
            if len(parts) < 6:
                continue
            total += 1
            if total <= skip:
                continue
            if take and total > skip + take:
                break
            fen, good_str, quiet_str = parts[0], parts[4], parts[5]
            quiets = quiet_str.split()
            goods = set(good_str.split()) & set(quiets)
            if len(quiets) < 2 or len(quiets) > qmax or not goods or len(goods) >= len(quiets):
                continue
            black = fen_black_to_move(fen)
            board = board_of_fen(fen)
            nf = 0
            ok = True
            own_k = opp_k = -1
            for sq in range(64):
                c = board[sq]
                if c != " ":
                    if nf >= MAXPC - 1:
                        ok = False
                        break
                    own = c.isupper() != black
                    plane = (0 if own else 6) + PIECE_TYPE[c.lower()]
                    rel = sq ^ 56 if black else sq
                    if c.lower() == "k":
                        if own:
                            own_k = rel
                        else:
                            opp_k = rel
                    buf["feat"][fill, nf] = plane * 64 + rel
                    nf += 1
            if not ok or own_k < 0 or opp_k < 0:
                buf["feat"][fill, :] = -1
                continue
            buf["feat"][fill, nf] = 768 + own_k * 64 + opp_k
            nf += 1
            for j, u in enumerate(quiets):
                fidx, tidx = encode_uci_piece(u, black, board)
                buf["qf"][fill, j] = fidx
                buf["qt"][fill, j] = tidx
                buf["good"][fill, j] = u in goods
            buf["qn"][fill] = len(quiets)
            fill += 1
            if fill == chunk:
                chunks.append(buf)
                buf = new_buf()
                fill = 0
    if fill or not chunks:
        chunks.append({k: v[:fill] for k, v in buf.items()})  # fill=0 -> empty arrays (short corpus)
    out = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}
    return total, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trunk", type=int, default=1024, help="feature-transformer width")
    ap.add_argument("--layers", type=int, default=2, help="1 = FT->heads, 2/3 add hidden layers")
    ap.add_argument("--hidden2", type=int, default=512, help="second-layer width when --layers >= 2")
    ap.add_argument("--hidden3", type=int, default=512, help="third-layer width when --layers >= 3")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.01)
    # Sharded prep (this box randomly kills LARGE long-lived Python processes — native crashes with
    # impossible tracebacks; short resumable shard processes survive): --prep parses a record slice
    # into an .npz; --from-npz trains from prepped shards, skipping the fragile parse entirely.
    ap.add_argument("--prep", action="store_true", help="parse only: write arrays to --out (.npz)")
    ap.add_argument("--skip", type=int, default=0, help="prep: skip this many records first")
    ap.add_argument("--take", type=int, default=0, help="prep: process at most this many records")
    ap.add_argument("--from-npz", default=None, help="train from comma-separated prepped .npz shards")
    ap.add_argument(
        "--no-kp",
        action="store_true",
        help="mask the joint king-pair feature at batch time (round-3 finding: at sparse density the "
        "4096-entry table memorizes instead of generalizing — this ablates it without re-prepping)",
    )
    args = ap.parse_args()

    if args.prep:
        total, arrs = stream_build(args.data, skip=args.skip, take=args.take)
        np.savez(args.out, **arrs)
        print(f"prep: records {args.skip}..{args.skip + (args.take or total)} -> "
              f"{arrs['feat'].shape[0]} rows -> {args.out}", flush=True)
        return

    import torch

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(4, torch.get_num_threads()))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.from_npz:
        shards = [np.load(p) for p in args.from_npz.split(",")]
        arrs = {k: np.concatenate([s[k] for s in shards], axis=0) for k in shards[0].files}
        n = arrs["feat"].shape[0]
        print(f"{len(shards)} npz shards -> {n} discriminative rows", flush=True)
    else:
        print("parsing records (streaming) ...", flush=True)
        total, arrs = stream_build(args.data)
        n = arrs["feat"].shape[0]
        print(f"{total} records -> {n} discriminative rows", flush=True)

    feat_np, qf_np, qt_np = arrs["feat"], arrs["qf"], arrs["qt"]
    qn_np, good_np = arrs["qn"], arrs["good"]

    class OwnTrunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ft = torch.nn.EmbeddingBag(NFEAT + 1, args.trunk, mode="sum", padding_idx=PAD)
            self.ft_b = torch.nn.Parameter(torch.zeros(args.trunk))
            if args.layers >= 2:
                self.mid = torch.nn.Linear(args.trunk, args.hidden2)
                head_in = args.hidden2
            else:
                self.mid = None
                head_in = args.trunk
            if args.layers >= 3:
                self.mid2 = torch.nn.Linear(head_in, args.hidden3)
                head_in = args.hidden3
            else:
                self.mid2 = None
            self.hf = torch.nn.Linear(head_in, 384)
            self.ht = torch.nn.Linear(head_in, 384)

        def forward(self, feats):
            x = torch.relu(self.ft(feats) + self.ft_b)
            if self.mid is not None:
                x = torch.relu(self.mid(x))
            if self.mid2 is not None:
                x = torch.relu(self.mid2(x))
            return self.hf(x), self.ht(x)

    model = OwnTrunk().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Crash-tolerant campaign mode: this box randomly kills big long-lived Python processes, so
    # every epoch checkpoints model+opt+sched, and a rerun with the same --out resumes from the
    # last completed epoch (the auto-resume wrapper loops until the final save lands).
    ckpt_path = args.out + ".ckpt"
    start_epoch = 0
    import os

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"]
        print(f"resumed from {ckpt_path} at epoch {start_epoch}", flush=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1024, int(n * args.val_frac))
    val_i, tr_i = perm[:n_val], perm[n_val:]

    def batch_tensors(idx):
        f = feat_np[idx].astype(np.int64)
        f[f < 0] = PAD  # padding_idx
        if args.no_kp:
            f[f >= 768] = PAD  # ablate the joint king-pair feature
        feats = torch.from_numpy(f).to(dev)
        qf = torch.from_numpy(qf_np[idx].astype(np.int64)).to(dev)
        qt = torch.from_numpy(qt_np[idx].astype(np.int64)).to(dev)
        qn = torch.from_numpy(qn_np[idx].astype(np.int64)).to(dev)
        good = torch.from_numpy(good_np[idx]).to(dev)
        return feats, qf, qt, qn, good

    def goodset_ce(fl, tl, qf, qt, qn, good):
        ql = fl.gather(1, qf) + tl.gather(1, qt)
        mask = torch.arange(qf.shape[1], device=qf.device).unsqueeze(0) < qn.unsqueeze(1)
        ql = ql.masked_fill(~mask, float("-inf"))
        good_ql = ql.masked_fill(~good, float("-inf"))
        return torch.logsumexp(ql, dim=1) - torch.logsumexp(good_ql, dim=1), ql

    def preserve(idx_set):
        model.eval()
        hits = tot = 0
        with torch.no_grad():
            for s in range(0, idx_set.size, args.batch):
                idx = idx_set[s : s + args.batch]
                feats, qf, qt, qn, good = batch_tensors(idx)
                fl, tl = model(feats)
                _, ql = goodset_ce(fl, tl, qf, qt, qn, good)
                arg = ql.argmax(dim=1)
                hits += int(good.gather(1, arg.unsqueeze(1)).sum())
                tot += idx.size
        model.train()
        return hits / tot

    model.train()
    train_probe = tr_i[:50_000]
    for ep in range(start_epoch, args.epochs):
        ep_perm = rng.permutation(tr_i)
        tot_loss, cnt = 0.0, 0
        for s in range(0, ep_perm.size, args.batch):
            idx = ep_perm[s : s + args.batch]
            feats, qf, qt, qn, good = batch_tensors(idx)
            fl, tl = model(feats)
            ce, _ = goodset_ce(fl, tl, qf, qt, qn, good)
            loss = ce.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss.detach()) * idx.size
            cnt += idx.size
        sched.step()
        vp = preserve(val_i)
        tp = preserve(train_probe)
        print(
            f"epoch {ep + 1}/{args.epochs} loss={tot_loss / cnt:.4f} "
            f"val_preserve={vp:.4f} train_preserve={tp:.4f}",
            flush=True,
        )
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1},
            ckpt_path,
        )

    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
