"""Torch modules for the EONPOL02 heads — quantization-aware (STE int8 grid, engine-exact hidden
activation): pfc0 1024->hidden CReLU(>>shift0) -> piece-aware pfrom/pto hidden->384 raw logits
(6 piece types x 64 STM-relative squares); WDL 32->3 per bucket.

The hidden activation uses an STE FLOOR on acc/2^shift0 so the float forward sits exactly on the
engine's integer grid (F#'s arithmetic >>> floors); weights/biases are STE-rounded and clamped to
the int8/int32 ranges the exporter writes. Same recipe as kga_stack_tune.cmd_train.
"""

import torch

HEAD_OUT = 384  # 6 piece types x 64 squares
SHIFT0 = 6
WDL_SHIFT = 6
SHIFT_EMB = 6  # bilinear interaction embeddings: signed int8 grid, floor(acc/2^SHIFT_EMB)
L1 = 1024


def ste_round(x):
    return (torch.round(x) - x).detach() + x


def ste_floor(x):
    return (torch.floor(x) - x).detach() + x


class PolicyHead(torch.nn.Module):
    def __init__(self, hidden: int = 128, seed: int = 0):
        super().__init__()
        self.hidden = hidden
        g = torch.Generator().manual_seed(seed)
        # Small int-grid init: weights a few quanta wide, biases zero.
        self.w0 = torch.nn.Parameter(torch.randn(hidden, L1, generator=g) * 3.0)
        self.b0 = torch.nn.Parameter(torch.zeros(hidden))
        self.wf = torch.nn.Parameter(torch.randn(HEAD_OUT, hidden, generator=g) * 3.0)
        self.bf = torch.nn.Parameter(torch.zeros(HEAD_OUT))
        self.wt = torch.nn.Parameter(torch.randn(HEAD_OUT, hidden, generator=g) * 3.0)
        self.bt = torch.nn.Parameter(torch.zeros(HEAD_OUT))

    def forward(self, ft):
        """ft: (B, 1024) float in [0,127] -> (from_logits, to_logits), each (B, 384)."""
        w0 = torch.clamp(ste_round(self.w0), -127, 127)
        acc = ft @ w0.T + ste_round(self.b0)
        hid = torch.clamp(ste_floor(acc / (1 << SHIFT0)), 0, 127)
        wf = torch.clamp(ste_round(self.wf), -127, 127)
        wt = torch.clamp(ste_round(self.wt), -127, 127)
        return hid @ wf.T + ste_round(self.bf), hid @ wt.T + ste_round(self.bt)


class WDLHead(torch.nn.Module):
    """Per-bucket 32->3 head off the frozen value stack's a1 activation."""

    def __init__(self, buckets: int = 8, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed + 1)
        self.w = torch.nn.Parameter(torch.randn(buckets, 3, 32, generator=g) * 2.0)
        self.b = torch.nn.Parameter(torch.zeros(buckets, 3))

    def forward(self, a1, bucket):
        """a1: (B, 32) float in [0,127]; bucket: (B,) long -> (B, 3) raw logits."""
        w = torch.clamp(ste_round(self.w), -127, 127)[bucket]  # (B, 3, 32)
        b = ste_round(self.b)[bucket]  # (B, 3)
        return torch.bmm(w, a1.unsqueeze(2)).squeeze(2) + b


def masked_goodset_ce(from_l, to_l, qf, qt, qn, good):
    """Multi-label quiet-conditional CE: maximize the probability MASS on the WDL-preserving set —
    loss = -(logsumexp over good quiets - logsumexp over all quiets). Reduces exactly to standard
    CE when `good` is one-hot. Returns (loss_per_row, masked_logits)."""
    ql = from_l.gather(1, qf) + to_l.gather(1, qt)  # (B, Q)
    mask = torch.arange(qf.shape[1], device=qf.device).unsqueeze(0) < qn.unsqueeze(1)
    ql = ql.masked_fill(~mask, float("-inf"))
    good_ql = ql.masked_fill(~good, float("-inf"))
    return torch.logsumexp(ql, dim=1) - torch.logsumexp(good_ql, dim=1), ql


class BilinearPolicyHead(torch.nn.Module):
    """PolicyHead + a low-rank bilinear from-square x to-square interaction: the additive
    from+to factorization cannot couple source and destination; score(m) adds ef[from].et[to].
    dim=0 degrades to exactly the additive head (the ablation baseline arm)."""

    def __init__(self, hidden: int = 256, dim: int = 32, seed: int = 0):
        super().__init__()
        self.hidden = hidden
        self.dim = dim
        g = torch.Generator().manual_seed(seed)
        self.w0 = torch.nn.Parameter(torch.randn(hidden, L1, generator=g) * 3.0)
        self.b0 = torch.nn.Parameter(torch.zeros(hidden))
        self.wf = torch.nn.Parameter(torch.randn(HEAD_OUT, hidden, generator=g) * 3.0)
        self.bf = torch.nn.Parameter(torch.zeros(HEAD_OUT))
        self.wt = torch.nn.Parameter(torch.randn(HEAD_OUT, hidden, generator=g) * 3.0)
        self.bt = torch.nn.Parameter(torch.zeros(HEAD_OUT))
        if dim > 0:
            self.wef = torch.nn.Parameter(torch.randn(64 * dim, hidden, generator=g) * 3.0)
            self.bef = torch.nn.Parameter(torch.zeros(64 * dim))
            self.wet = torch.nn.Parameter(torch.randn(64 * dim, hidden, generator=g) * 3.0)
            self.bet = torch.nn.Parameter(torch.zeros(64 * dim))

    def forward(self, ft):
        """ft (B,1024) in [0,127] -> (from_l (B,384), to_l (B,384), ef (B,64,D)|None, et|None).
        Embeddings sit on the SIGNED int8 grid: clamp(floor(acc/2^SHIFT_EMB), -127, 127)."""
        w0 = torch.clamp(ste_round(self.w0), -127, 127)
        acc = ft @ w0.T + ste_round(self.b0)
        hid = torch.clamp(ste_floor(acc / (1 << SHIFT0)), 0, 127)
        wf = torch.clamp(ste_round(self.wf), -127, 127)
        wt = torch.clamp(ste_round(self.wt), -127, 127)
        from_l = hid @ wf.T + ste_round(self.bf)
        to_l = hid @ wt.T + ste_round(self.bt)
        if self.dim == 0:
            return from_l, to_l, None, None
        wef = torch.clamp(ste_round(self.wef), -127, 127)
        wet = torch.clamp(ste_round(self.wet), -127, 127)
        ef = torch.clamp(ste_floor((hid @ wef.T + ste_round(self.bef)) / (1 << SHIFT_EMB)), -127, 127)
        et = torch.clamp(ste_floor((hid @ wet.T + ste_round(self.bet)) / (1 << SHIFT_EMB)), -127, 127)
        b = ft.shape[0]
        return from_l, to_l, ef.view(b, 64, self.dim), et.view(b, 64, self.dim)


def masked_goodset_ce_bilin(from_l, to_l, ef, et, qf, qt, qn, good):
    """masked_goodset_ce with the bilinear interaction added to each quiet's logit
    (ef/et None = pure additive path, numerically identical to masked_goodset_ce)."""
    ql = from_l.gather(1, qf) + to_l.gather(1, qt)  # (B, Q)
    if ef is not None:
        d = ef.shape[2]
        fsq = (qf % 64).unsqueeze(-1).expand(-1, -1, d)
        tsq = (qt % 64).unsqueeze(-1).expand(-1, -1, d)
        ql = ql + (ef.gather(1, fsq) * et.gather(1, tsq)).sum(-1)
    mask = torch.arange(qf.shape[1], device=qf.device).unsqueeze(0) < qn.unsqueeze(1)
    ql = ql.masked_fill(~mask, float("-inf"))
    good_ql = ql.masked_fill(~good, float("-inf"))
    return torch.logsumexp(ql, dim=1) - torch.logsumexp(good_ql, dim=1), ql


def masked_quiet_ce(from_l, to_l, qf, qt, qn, tgt):
    """Quiet-conditional cross-entropy: softmax over each position's legal quiets only.
    from_l/to_l: (B, 64); qf/qt: (B, Q) rel-square indices (padded); qn: (B,) counts; tgt: (B,)."""
    ql = from_l.gather(1, qf) + to_l.gather(1, qt)  # (B, Q)
    mask = torch.arange(qf.shape[1], device=qf.device).unsqueeze(0) < qn.unsqueeze(1)
    ql = ql.masked_fill(~mask, float("-inf"))
    return -(ql.gather(1, tgt.unsqueeze(1)).squeeze(1) - torch.logsumexp(ql, dim=1)), ql
