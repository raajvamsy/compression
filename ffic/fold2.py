"""Fold shapes that keep the stack's neighbours close (follow-up to fold.py, same guesser and coder, ideal bits).
    python fold2.py DATADIR [image ...]
  raster   every pixel from left/above neighbours
  hfold    fold columns once: stack = even columns (above/below still 1 px away); fold wave = odd columns, sees left AND right
  checker  checkerboard fold: stack = (x+y) even (diagonal neighbours 1.4 px away); fold wave sees left, right, above, below
"""
import json, os, sys
import numpy as np
from fold import ls_code, raster_bits

def shift(a, dy, dx, k=4):
    p = np.pad(a, k, mode="edge"); H, W = a.shape; return p[k + dy:k + dy + H, k + dx:k + dx + W]

def taps_at(P, offs, mask):
    return np.stack([shift(P, dy, dx)[mask] for dy, dx in offs], 1)

def act_ctx(P, offs, mask):
    v = np.stack([shift(P, dy, dx)[mask] for dy, dx in offs], 1); a = v.max(1) - v.min(1)
    return np.digitize(a, [2, 4, 8, 16, 32, 64, 128])

def hfold(P):
    H, W = P.shape; X = np.arange(W)[None, :].repeat(H, 0); s = X % 2 == 0; f = ~s
    so = [(0, -2), (-1, 0), (-1, -2), (-1, 2), (0, -4), (-2, 0), (-2, -2), (-2, 2), (-1, -4), (-1, 4)]   # stack: causal, stack lattice
    fo = [(0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (0, -2), (-2, 0), (0, 3), (0, -3)]  # fold wave: stack both sides + own known
    return (ls_code(taps_at(P, so, s), P[s], act_ctx(P, so[:4], s)) + ls_code(taps_at(P, fo, f), P[f], act_ctx(P, fo[:6], f)), s.sum())

def checker(P):
    H, W = P.shape; Y, X = np.mgrid[0:H, 0:W]; s = (X + Y) % 2 == 0; f = ~s
    so = [(0, -2), (-1, -1), (-1, 1), (-2, 0), (-2, -2), (-2, 2), (-1, -3), (-1, 3), (0, -4), (-3, -1), (-3, 1)]
    fo = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -2), (-1, 2), (1, -2), (1, 2), (-2, -1), (-2, 1), (2, -1), (2, 1), (0, -2), (-1, -1), (-1, 1)]
    return (ls_code(taps_at(P, so, s), P[s], act_ctx(P, so[:4], s)) + ls_code(taps_at(P, fo, f), P[f], act_ctx(P, fo[:4], f)), s.sum())

if __name__ == "__main__":
    D = sys.argv[1]; ref = json.load(open(os.path.join(D, "ref.json"))); names = sys.argv[2:] or list(ref); tot = {}
    for n in names:
        r = ref[n]; img = np.fromfile(f"{D}/{n}.rgb", np.uint8).reshape(r["h"], r["w"], 3); row = {"raster": 0, "hfold": 0, "checker": 0}
        for c in range(3):
            P = img[..., c].astype(float); row["raster"] += raster_bits(P) / 8
            row["hfold"] += hfold(P)[0] / 8; row["checker"] += checker(P)[0] / 8
        for k, v in row.items(): tot[k] = tot.get(k, 0) + v
        print(n, "  ".join(f"{k} {v:,.0f}" + (f" ({100 * (v / row['raster'] - 1):+.2f}%)" if k != "raster" else "") for k, v in row.items()), flush=True)
    if len(names) > 1: print("TOTAL", "  ".join(f"{k} {v:,.0f} ({100 * (v / tot['raster'] - 1):+.2f}%)" for k, v in tot.items()))
