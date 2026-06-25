"""Generate Eonego/Lc0PolicyMap.fs from lc0's kMoveStrs (the canonical 1858 NN-index move order)."""
import re, sys

src = open("trainer/lc0_encoder.cc").read()
# Extract the kMoveStrs array body.
m = re.search(r'const char\* kMoveStrs\[\]\s*=\s*\{(.*?)\};', src, re.S)
body = m.group(1)
moves = re.findall(r'"([a-h][1-8][a-h][1-8][qrbn]?)"', body)
assert len(moves) == 1858, f"expected 1858 moves, got {len(moves)}"
# Sanity: promotions seen are only q/r/b (knight promo reuses the plain slot).
promos = sorted(set(mv[4] for mv in moves if len(mv) == 5))
assert promos == ['b', 'q', 'r'], f"unexpected promo chars {promos}"

joined = " ".join(moves)
fs = f'''/// Lc0 1858-policy move-index map. GENERATED from lc0 src/neural/encoder.cc kMoveStrs by
/// trainer/gen_policymap.py — do not edit by hand. Replicates lc0 MoveToNNIndex(transform=0):
///   packed = promo*4096 + from*64 + to ;  nnIndex = table[packed]  (knight-promo=0 reuses plain slot).
/// lc0 promo idx (n=0,b=1,r=2,q=3) == Eonego Move promo field; squares are LERF a1=0 (== Eonego).
module Eonego.Lc0PolicyMap

open Eonego.Move

[<Literal>]
let NumPolicy = 1858

/// The 1858 canonical lc0 move strings (white/mover perspective), space-separated, in NN-index order.
let private moveStrsRaw =
    "{joined}"

/// NN-index -> UCI move string (mover perspective); public for the round-trip test.
let nnIndexToUci: string[] = moveStrsRaw.Split(' ')

/// table.[promo*4096 + from*64 + to] = NN policy index (0..1857), or -1 if not a policy move.
let private packedToNN: int[] =
    let t = Array.create (64 * 64 * 4) -1
    let strs = nnIndexToUci
    let sq (s: string) (o: int) = (int s.[o] - int 'a') + (int s.[o + 1] - int '1') * 8
    let promoIdx (c: char) =
        match c with
        | 'n' -> 0
        | 'b' -> 1
        | 'r' -> 2
        | 'q' -> 3
        | _ -> 0
    for i in 0 .. strs.Length - 1 do
        let s = strs.[i]
        let frm = sq s 0
        let dst = sq s 2
        let promo = if s.Length = 5 then promoIdx s.[4] else 0
        t.[promo * 4096 + frm * 64 + dst] <- i
    t

/// Map an Eonego move to its lc0 1858 policy index (-1 if unmapped). `stmIsBlack` => vertical flip
/// (sq ^ 56) of from/to so the move is expressed in the mover's-perspective frame, matching the encoder.
let moveToNNIndex (stmIsBlack: bool) (m: Move) : int =
    let flip s = if stmIsBlack then s ^^^ 56 else s
    let frm = flip (fromSq m)
    let dst = flip (toSq m)
    let promo = if isPromotion m then (m >>> 12) &&& 0x3 else 0
    let packed = promo * 4096 + frm * 64 + dst
    if packed >= 0 && packed < packedToNN.Length then packedToNN.[packed] else -1
'''
open("Eonego/Lc0PolicyMap.fs", "w", encoding="utf-8").write(fs)
print(f"wrote Eonego/Lc0PolicyMap.fs ({len(moves)} moves, {len(joined)} chars)")
