"""UCI move <-> STM-relative (from, to) square indices for the EONPOL policy head. Stdlib only.

STM-relative convention (MUST match engine Policy.relSq, Eonego/Policy.fs): White reads the board
as-is; Black is vertically flipped (sq ^ 56 — rank mirror, the accumulator perspective convention).
Squares are LERF (a1=0 .. h8=63). Promotion moves alias the plain from-to slot (queen-promo included;
underpromotions are ignored by the v1 head).
"""


def sq_of_str(s: str) -> int:
    """'e4' -> LERF square index."""
    f = ord(s[0]) - ord("a")
    r = ord(s[1]) - ord("1")
    if not (0 <= f < 8 and 0 <= r < 8):
        raise ValueError(f"bad square {s!r}")
    return r * 8 + f


def rel_sq(sq: int, black_to_move: bool) -> int:
    """STM-relative square (engine Policy.relSq)."""
    return sq ^ 56 if black_to_move else sq


def encode_uci(uci: str, black_to_move: bool) -> tuple[int, int]:
    """UCI move string -> (from_rel, to_rel). Promotion suffix is ignored (slot aliasing)."""
    if len(uci) < 4:
        raise ValueError(f"bad uci {uci!r}")
    return rel_sq(sq_of_str(uci[0:2]), black_to_move), rel_sq(sq_of_str(uci[2:4]), black_to_move)


def fen_black_to_move(fen: str) -> bool:
    return fen.split()[1] == "b"
