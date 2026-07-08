"""Executable checks for the bilinear policy head (no pytest in this repo — run directly):

    uv run --python 3.12 --with torch,numpy python checks_bilin.py

Covers: signature_of_fen canonicalization; BilinearPolicyHead shapes + int8-grid ranges;
masked_goodset_ce_bilin == masked_goodset_ce when the interaction is off (dim=0 path)."""

import sys

import policy_dataset as pd


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    return ok


def main():
    all_ok = True

    # --- signature_of_fen ---
    s = pd.signature_of_fen("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
    all_ok &= check("KPvK signature", s == "KPvK")
    s = pd.signature_of_fen("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1")
    all_ok &= check("black-strong side still first", s == "KPvK")
    s = pd.signature_of_fen("r3k3/8/8/8/8/8/8/QR2K3 w - - 0 1")
    all_ok &= check("value ordering (QR beats R)", s == "KQRvKR")
    s1 = pd.signature_of_fen("4k3/8/8/8/8/8/8/RN2K3 w - - 0 1")
    s2 = pd.signature_of_fen("rn2k3/8/8/8/8/8/8/4K3 w - - 0 1")
    all_ok &= check("color independence", s1 == s2)

    # --- model ---
    import torch

    from policy_model import BilinearPolicyHead, masked_goodset_ce, masked_goodset_ce_bilin

    torch.manual_seed(0)
    B, Q, D = 5, 7, 8
    ft = torch.randint(0, 128, (B, 1024)).float()
    m = BilinearPolicyHead(hidden=64, dim=D, seed=1)
    fl, tl, ef, et = m(ft)
    all_ok &= check("from/to logit shapes", fl.shape == (B, 384) and tl.shape == (B, 384))
    all_ok &= check("embedding shapes", ef.shape == (B, 64, D) and et.shape == (B, 64, D))
    all_ok &= check(
        "embeddings on the int8 grid", bool((ef.abs() <= 127).all() and (et.abs() <= 127).all())
    )

    m0 = BilinearPolicyHead(hidden=64, dim=0, seed=1)
    fl0, tl0, ef0, et0 = m0(ft)
    all_ok &= check("dim=0 returns no embeddings", ef0 is None and et0 is None)

    qf = torch.randint(0, 384, (B, Q))
    qt = torch.randint(0, 384, (B, Q))
    qn = torch.full((B,), Q)
    good = torch.zeros(B, Q, dtype=torch.bool)
    good[:, 0] = True
    ce_ref, ql_ref = masked_goodset_ce(fl0, tl0, qf, qt, qn, good)
    ce_new, ql_new = masked_goodset_ce_bilin(fl0, tl0, None, None, qf, qt, qn, good)
    all_ok &= check(
        "bilin CE == goodset CE when interaction off",
        bool(torch.allclose(ce_ref, ce_new) and torch.allclose(ql_ref, ql_new)),
    )

    ce_i, ql_i = masked_goodset_ce_bilin(fl, tl, ef, et, qf, qt, qn, good)
    all_ok &= check("interaction changes the logits", not bool(torch.allclose(ql_i, ql_new)))
    all_ok &= check("interaction loss finite", bool(torch.isfinite(ce_i).all()))

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
