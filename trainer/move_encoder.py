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


PIECE_TYPE = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}  # engine PieceType Pawn=0..King=5


def board_of_fen(fen: str) -> list[str]:
    """FEN board field -> 64-char list, LERF order (index 0 = a1). ' ' = empty."""
    out = [" "] * 64
    ranks = fen.split()[0].split("/")
    for r, row in enumerate(ranks):  # ranks[0] = rank 8
        f = 0
        for c in row:
            if c.isdigit():
                f += int(c)
            else:
                out[(7 - r) * 8 + f] = c
                f += 1
    return out


def piece_type_at(board: list[str], sq: int) -> int:
    """Engine piece-type index (0..5) of the piece on LERF square `sq`."""
    return PIECE_TYPE[board[sq].lower()]


def encode_uci_piece(uci: str, black_to_move: bool, board: list[str]) -> tuple[int, int]:
    """UCI move -> EONPOL02 piece-aware logit indices (pt*64 + rel_from, pt*64 + rel_to)."""
    f = sq_of_str(uci[0:2])
    t = sq_of_str(uci[2:4])
    pt = piece_type_at(board, f)
    return pt * 64 + rel_sq(f, black_to_move), pt * 64 + rel_sq(t, black_to_move)
