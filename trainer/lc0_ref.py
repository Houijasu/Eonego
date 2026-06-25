"""Generate reference policy/value from the REAL lc0 for a handful of FENs, to validate Eonego's
Lc0Net forward pass (Phase 5 oracle). Requires the `lczero.backends` python bindings (built from lc0
with `-Dpython=true`) and the .pb net. Output: JSON {fen: {value, top:[(uci,prior),...]}}.

This is OPTIONAL: Eonego's Lc0NetTests already validate via (a) scalar==AVX2 parity, (b) the 1858
round-trip, (c) structural dim gates, and (d) the "startpos best policy is a principled opening"
end-to-end check. The oracle additionally pins the policy DISTRIBUTION (top-k) and value tolerance.

Usage:
    python trainer/lc0_ref.py nets/20x256SE-jj-9-75000000.pb > trainer/lc0_ref.json
Then compare in F#: load lc0_ref.json, run Lc0Net.forward on each fen, assert top-5 move agreement
and |value - ref| < 2e-2.
"""
import sys, json

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",      # startpos
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",      # startpos, black to move (flip)
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # kiwipete (castling)
    "8/PPP2k2/8/8/8/8/2K2ppp/8 w - - 0 1",                          # promotions
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # quiet open game
]

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python lc0_ref.py <net.pb>")
    net_path = sys.argv[1]
    try:
        import chess
        from lczero.backends import Weights, Backend, GameState
    except ImportError as e:
        sys.exit(
            f"missing dependency ({e}). Install python-chess and build lc0 python bindings:\n"
            "  pip install chess\n"
            "  (build lc0 with meson -Dpython=true, then add build dir to PYTHONPATH)\n"
            "Skipping oracle generation; Eonego's in-suite tests still validate the forward pass."
        )

    w = Weights(net_path)
    backend = Backend(weights=w)  # default (blas/eigen) CPU backend
    out = {}
    for fen in FENS:
        gs = GameState(fen=fen)
        inp = gs.as_input(backend)
        result = backend.evaluate(inp)[0]
        # policy: lc0 returns logits/probs over the legal moves in gs.moves() order.
        moves = gs.moves()
        probs = list(result.p_softmax(*range(len(moves))))
        pairs = sorted(zip(moves, probs), key=lambda kv: -kv[1])[:5]
        out[fen] = {"value": float(result.q()), "top": [(m, float(p)) for m, p in pairs]}
    json.dump(out, sys.stdout, indent=2)

if __name__ == "__main__":
    main()
