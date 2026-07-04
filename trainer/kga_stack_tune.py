"""KGA-specialist fine-tuning of the FullThreats NNUE output stacks (FT frozen).

The engine's `dumpft` tooling emits, per position, the fc_0 input (the u8 FT pairwise-product
buffer, STM perspective first), the psqt term, the bucket and the engine's evalInternal.
With the FT frozen, the 8 per-bucket stacks (fc0 1024->32, fc1 62->32, fc2 32->1 + skip lane)
are small MLPs trainable in minutes; trained int8/int32 weights are patched back into a
candidate .nnue (container codec reused from blend_nnue.parse / raw writer).

Subcommands:
  prep   : merge+dedupe engine `gen` records -> labeled fens file (fen;cp_white;result_white)
  parity : exact python int-forward vs engine evalInternal from a dumpft dump  (must be 100%)
  train  : QAT fine-tune stacks on dump+labels, export patched candidate net

Dump record (little-endian, 1034 B): [bucket u8][stm u8][psqt i32][eval i32][ft u8*1024]
"""

import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blend_nnue as bn

L1 = bn.L1
NPV = 356  # NormalizeToPawnValue
DUMP_DTYPE = np.dtype([("bucket", "u1"), ("stm", "u1"), ("psqt", "<i4"),
                       ("eval", "<i4"), ("ft", "u1", (L1,))])


def read_dump(path):
    recs = np.fromfile(path, dtype=DUMP_DTYPE)
    assert recs.size > 0, f"empty dump {path}"
    return recs


def tdiv(a, b):
    """C-style truncation-toward-zero integer division (matches F# `/` and .NET int64 `/`)."""
    return np.trunc(np.asarray(a, dtype=np.float64) / b)


def stacks_of(net):
    """blend_nnue.parse() result -> list of 8 dicts of int64 arrays."""
    out = []
    for s in range(bn.STACKS):
        out.append({k: net["regions"][f"s{s}.{k}"] for k, _, _ in bn.STACK_REGIONS})
    return out


def int_forward(stacks, recs):
    """Exact integer forward of the stack layers for every dump record. float64 matmuls are
    exact here (all intermediates < 2^53). Returns evalInternal per record (np.int64)."""
    n = recs.size
    out = np.zeros(n, dtype=np.int64)
    ft = recs["ft"].astype(np.float64)
    psqt = recs["psqt"].astype(np.int64)
    buckets = recs["bucket"]
    for b in range(bn.STACKS):
        idx = np.nonzero(buckets == b)[0]
        if idx.size == 0:
            continue
        st = stacks[b]
        w0 = st["fc0w"].reshape(32, L1).astype(np.float64)
        fc0 = (ft[idx] @ w0.T + st["fc0b"].astype(np.float64)).astype(np.int64)
        x = fc0[:, :31]
        skip = fc0[:, 31]
        conc = np.zeros((idx.size, 64), dtype=np.int64)
        conc[:, :31] = np.minimum(127, (x * x) >> 21)
        conc[:, 31:62] = np.clip(x >> 7, 0, 127)
        w1 = st["fc1w"].reshape(32, 64).astype(np.float64)
        fc1 = (conc.astype(np.float64) @ w1.T + st["fc1b"].astype(np.float64)).astype(np.int64)
        a1 = np.clip(fc1 >> 6, 0, 127)
        fc2 = (a1.astype(np.float64) @ st["fc2w"].astype(np.float64)).astype(np.int64) + st["fc2b"][0]
        fwd = fc2 + skip
        output_value = tdiv(fwd * 9600, 16384)
        psqt_v = tdiv(psqt[idx], 16)
        pos_v = tdiv(output_value, 16)
        out[idx] = tdiv(125 * psqt_v + 131 * pos_v, 128).astype(np.int64)
    return out


def write_net(path, base, regions):
    """Write a full net (RAW encoding) with `regions` (dict name->int array) into `path`."""
    desc = base["desc"]
    with open(path, "wb") as f:
        f.write(struct.pack("<I", bn.VERSION))
        f.write(struct.pack("<I", base["arch_hash"]))
        f.write(struct.pack("<I", len(desc)))
        f.write(desc)
        f.write(struct.pack("<I", base["ft_hash"]))
        for name, count, dtype in bn.REGIONS:
            f.write(regions[name].astype(np.dtype(dtype)).tobytes())
        for s in range(bn.STACKS):
            f.write(struct.pack("<I", base["stack_hashes"][s]))
            for name, count, dtype in bn.STACK_REGIONS:
                f.write(regions[f"s{s}.{name}"].astype(np.dtype(dtype)).tobytes())


# ---------------------------------------------------------------- prep

def cmd_prep(args):
    seen = {}
    order = []
    for src in args.inputs:
        with open(src, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split(";")
                if len(parts) < 3:
                    continue
                fen = parts[0].strip()
                try:
                    cp = float(parts[1])
                    res = float(parts[2])
                except ValueError:
                    continue
                if fen in seen:
                    seen[fen][0] += cp
                    seen[fen][1] += res
                    seen[fen][2] += 1
                else:
                    seen[fen] = [cp, res, 1]
                    order.append(fen)
    with open(args.out, "w", encoding="utf-8") as f:
        for fen in order:
            cp, res, n = seen[fen]
            f.write(f"{fen};{cp / n:.1f};{res / n:.3f}\n")
    print(f"prep: {len(order)} unique positions -> {args.out}")


# ---------------------------------------------------------------- parity

def cmd_parity(args):
    net = bn.parse(args.net)
    recs = read_dump(args.dump)
    got = int_forward(stacks_of(net), recs)
    want = recs["eval"].astype(np.int64)
    bad = np.nonzero(got != want)[0]
    print(f"parity: {recs.size - bad.size}/{recs.size} exact")
    if bad.size:
        for i in bad[:10]:
            print(f"  rec {i}: python {got[i]} engine {want[i]} bucket {recs['bucket'][i]}")
        raise SystemExit(1)
    print("PARITY OK")


# ---------------------------------------------------------------- train

def read_labels(path):
    cps, results = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(";")
            cps.append(float(parts[1]))
            results.append(float(parts[2]))
    return np.array(cps, dtype=np.float32), np.array(results, dtype=np.float32)


def cmd_train(args):
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = bn.parse(args.net)
    stacks = stacks_of(net)
    recs = read_dump(args.dump)
    cp, res = read_labels(args.labels)
    assert cp.size == recs.size, f"label/dump mismatch: {cp.size} vs {recs.size}"

    cp_c = np.clip(cp, -args.cp_clamp, args.cp_clamp)
    sign = np.where(recs["stm"] == 0, 1.0, -1.0).astype(np.float32)  # white-rel -> stm-rel later
    wp = 1.0 / (1.0 + np.exp(-cp_c / args.K))
    target_white = (args.lam * wp + (1.0 - args.lam) * res).astype(np.float32)

    def ste_round(x):
        return (torch.round(x) - x).detach() + x

    class Stack(torch.nn.Module):
        def __init__(self, st):
            super().__init__()
            self.w0 = torch.nn.Parameter(torch.tensor(st["fc0w"].reshape(32, L1), dtype=torch.float32))
            self.b0 = torch.nn.Parameter(torch.tensor(st["fc0b"], dtype=torch.float32))
            self.w1 = torch.nn.Parameter(torch.tensor(st["fc1w"].reshape(32, 64), dtype=torch.float32))
            self.b1 = torch.nn.Parameter(torch.tensor(st["fc1b"], dtype=torch.float32))
            self.w2 = torch.nn.Parameter(torch.tensor(st["fc2w"], dtype=torch.float32))
            self.b2 = torch.nn.Parameter(torch.tensor(st["fc2b"], dtype=torch.float32))

        def forward(self, ft, psqt):
            w0 = torch.clamp(ste_round(self.w0), -127, 127)
            fc0 = ft @ w0.T + ste_round(self.b0)
            x = fc0[:, :31]
            skip = fc0[:, 31]
            conc_sqr = torch.clamp(x * x / 2097152.0, max=127.0)
            conc_lin = torch.clamp(x / 128.0, 0.0, 127.0)
            conc = torch.cat([conc_sqr, conc_lin], dim=1)  # engine cols 62,63 are zero
            w1 = torch.clamp(ste_round(self.w1), -127, 127)
            fc1 = conc @ w1[:, :62].T + ste_round(self.b1)
            a1 = torch.clamp(fc1 / 64.0, 0.0, 127.0)
            w2 = torch.clamp(ste_round(self.w2), -127, 127)
            fc2 = a1 @ w2 + ste_round(self.b2)
            fwd = fc2 + skip
            output_value = fwd * (9600.0 / 16384.0)
            return (125.0 * (psqt / 16.0) + 131.0 * (output_value / 16.0)) / 128.0

    buckets = recs["bucket"]
    trained = {}
    for b in range(bn.STACKS):
        idx = np.nonzero(buckets == b)[0]
        if idx.size < args.min_bucket:
            print(f"bucket {b}: {idx.size} samples < {args.min_bucket}, keeping original stack")
            continue
        rng = np.random.default_rng(args.seed + b)
        perm = rng.permutation(idx)
        n_val = max(64, int(perm.size * 0.05))
        val_i, tr_i = perm[:n_val], perm[n_val:]

        model = Stack(stacks[b]).to(dev)
        w0_anchor = model.w0.detach().clone()
        w1_anchor = model.w1.detach().clone()
        w2_anchor = model.w2.detach().clone()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        def batch_eval(ii):
            ft_t = torch.tensor(recs["ft"][ii], dtype=torch.float32, device=dev)
            ps_t = torch.tensor(recs["psqt"][ii].astype(np.float32), device=dev)
            internal = model(ft_t, ps_t)
            cp_stm = internal * (100.0 / NPV)
            cp_white = cp_stm * torch.tensor(sign[ii], device=dev)
            pred = torch.sigmoid(cp_white / args.K)
            tgt = torch.tensor(target_white[ii], device=dev)
            return torch.mean((pred - tgt) ** 2)

        best_val, best_state, bad_epochs = float("inf"), None, 0
        bs = args.batch
        for ep in range(args.epochs):
            model.train()
            rng.shuffle(tr_i)
            tot, nb = 0.0, 0
            for k in range(0, tr_i.size, bs):
                ii = tr_i[k:k + bs]
                loss = batch_eval(ii)
                anchor = (torch.mean((model.w0 - w0_anchor) ** 2)
                          + torch.mean((model.w1 - w1_anchor) ** 2)
                          + torch.mean((model.w2 - w2_anchor) ** 2))
                total = loss + args.anchor * anchor
                opt.zero_grad()
                total.backward()
                opt.step()
                tot += float(loss)
                nb += 1
            model.eval()
            with torch.no_grad():
                vl = float(batch_eval(val_i))
            if vl < best_val - 1e-7:
                best_val, bad_epochs = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
            if ep % 10 == 0 or bad_epochs >= args.patience:
                print(f"bucket {b} ep {ep:3d}  train {tot / max(1, nb):.6f}  val {vl:.6f}  best {best_val:.6f}")
            if bad_epochs >= args.patience:
                break
        model.load_state_dict(best_state)
        trained[b] = {
            "fc0w": np.clip(np.rint(best_state["w0"].cpu().numpy()), -127, 127).astype(np.int64).reshape(-1),
            "fc0b": np.rint(best_state["b0"].cpu().numpy()).astype(np.int64),
            "fc1w": np.clip(np.rint(best_state["w1"].cpu().numpy()), -127, 127).astype(np.int64).reshape(-1),
            "fc1b": np.rint(best_state["b1"].cpu().numpy()).astype(np.int64),
            "fc2w": np.clip(np.rint(best_state["w2"].cpu().numpy()), -127, 127).astype(np.int64),
            "fc2b": np.rint(best_state["b2"].cpu().numpy()).astype(np.int64),
        }
        print(f"bucket {b}: trained on {tr_i.size} (val {val_i.size}), best val {best_val:.6f}")

    regions = dict(net["regions"])
    for b, st in trained.items():
        for k in st:
            regions[f"s{b}.{k}"] = st[k]
    write_net(args.out, net, regions)
    print(f"wrote candidate {args.out} (trained buckets: {sorted(trained)})")

    # report int-eval drift vs the incumbent on the dump sample
    cand = bn.parse(args.out)
    got = int_forward(stacks_of(cand), recs)
    drift = (got - recs["eval"]).astype(np.float64) * 100.0 / NPV
    print(f"eval drift vs incumbent (cp): mean {np.mean(np.abs(drift)):.1f}  p95 "
          f"{np.percentile(np.abs(drift), 95):.1f}  max {np.max(np.abs(drift)):.1f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep")
    p.add_argument("--out", required=True)
    p.add_argument("inputs", nargs="+")

    p = sub.add_parser("parity")
    p.add_argument("--net", required=True)
    p.add_argument("--dump", required=True)

    p = sub.add_parser("train")
    p.add_argument("--net", required=True, help="incumbent .nnue (warm start + frozen FT)")
    p.add_argument("--dump", required=True, help="dumpft output for the labeled fens (same net!)")
    p.add_argument("--labels", required=True, help="fen;cp_white;result_white aligned with dump")
    p.add_argument("--out", required=True)
    p.add_argument("--K", type=float, default=180.0)
    p.add_argument("--lam", type=float, default=0.75)
    p.add_argument("--cp-clamp", type=float, default=2000.0)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--anchor", type=float, default=1e-4)
    p.add_argument("--min-bucket", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    {"prep": cmd_prep, "parity": cmd_parity, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
