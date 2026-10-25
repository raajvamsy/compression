"""SLICES: cut the brightness landscape into 8 flat layers (bit planes, top layer first) and code each layer as a yes/no map.

    python slices.py DATADIR [image ...]       -> experiments/slices.json

Each yes/no is coded with an adaptive counter chosen by its context (ideal code length; the decoder can follow the same counts):
  - the same layer's already-coded neighbours (W, N, NW, NE, WW, NN)
  - the layers ABOVE, which are fully known in EVERY direction (this pixel and its W, E, N, S, NE, SW neighbours)
  - how far down this pixel already is (its value so far, top 2 bits)
Two ways to cut: plain binary layers and Gray-code layers (neighbouring heights differ in fewer layers).
Compared with the same-family pixel coder of fold.py (least-squares guesser + adaptive Laplace) and with FFIC's real bits.
"""
import json, os, sys
import numpy as np
from fold import raster_bits

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(a, dy, dx):
    p = np.pad(a, 2); H, W = a.shape; return p[2 + dy:2 + dy + H, 2 + dx:2 + dx + W]


def adaptive_bits(bits, ctx):                         # cost of each bit with a per-context adaptive counter, raster order
    b = bits.ravel().astype(np.int64); c = ctx.ravel().astype(np.int64)
    o = np.argsort(c, kind="stable"); cs, bs = c[o], b[o]
    start = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]; grp = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(cs)]))
    rank = np.arange(len(cs)) - start[grp]
    ones = np.cumsum(bs) - bs; ones_before = ones - (np.cumsum(bs) - bs)[start][grp]
    p1 = (ones_before + 0.4) / (rank + 0.8)
    return float(-np.log2(np.where(bs == 1, p1, 1 - p1)).sum())


def slice_bits(P, gray):
    v = P.astype(np.int64); v = v ^ (v >> 1) if gray else v
    planes = [((v >> k) & 1) for k in range(7, -1, -1)]          # top layer first
    total, per = 0.0, []
    for i, B in enumerate(planes):
        ctx = np.zeros_like(B)
        for k, (dy, dx) in enumerate(((0, -1), (-1, 0), (-1, -1), (-1, 1), (0, -2), (-2, 0))):
            ctx |= sh(B, dy, dx) << k                               # this layer, already coded
        if i > 0:
            A = planes[i - 1]
            for k, (dy, dx) in enumerate(((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, 1), (1, -1))):
                ctx |= sh(A, dy, dx) << (6 + k)                     # layer above: every direction
        if i > 1: ctx |= planes[i - 2] << 13
        ctx |= i << 14
        b = adaptive_bits(B, ctx); per.append(b / 8); total += b
    return total, per


if __name__ == "__main__":
    D = sys.argv[1]; ref = json.load(open(os.path.join(D, "ref.json"))); names = sys.argv[2:] or list(ref)
    log = {}
    for l in open(os.path.join(HERE, "experiments/log.jsonl")): r = json.loads(l); log[r["id"]] = r
    res, T = {}, {"pixel": 0, "slices": 0, "gray": 0, "ffic": 0}
    for n in names:
        r = ref[n]; img = np.fromfile(f"{D}/{n}.rgb", np.uint8).reshape(r["h"], r["w"], 3)
        row = {"pixel": 0.0, "slices": 0.0, "gray": 0.0, "layers": []}
        for c in range(3):
            P = img[..., c]; row["pixel"] += raster_bits(P) / 8
            b, per = slice_bits(P, False); row["slices"] += b / 8
            g, perg = slice_bits(P, True); row["gray"] += g / 8
            if c == 1: row["layers"] = [round(x) for x in per]; row["layers_gray"] = [round(x) for x in perg]
        row["ffic"] = log["best_now"]["per_image"][n]["bytes"]
        res[n] = {k: (round(v) if isinstance(v, float) else v) for k, v in row.items()}
        for k in T: T[k] += row[k]
        print(f"{n:14} pixel coder {row['pixel']:9,.0f}  slices {row['slices']:9,.0f} ({100 * (row['slices'] / row['pixel'] - 1):+.1f}%)  "
              f"gray slices {row['gray']:9,.0f} ({100 * (row['gray'] / row['pixel'] - 1):+.1f}%)  FFIC {row['ffic']:9,}", flush=True)
        print(f"{'':14} green layers top->bottom (bytes): {res[n]['layers']}  gray: {res[n]['layers_gray']}", flush=True)
    if len(names) > 1:
        print(f"TOTAL pixel {T['pixel']:,.0f}  slices {T['slices']:,.0f} ({100 * (T['slices'] / T['pixel'] - 1):+.1f}%)  gray {T['gray']:,.0f} ({100 * (T['gray'] / T['pixel'] - 1):+.1f}%)  FFIC {T['ffic']:,}")
    json.dump(res, open(os.path.join(HERE, "experiments/slices.json"), "w"), indent=1)
