"""SLABS: slice the brightness landscape into n thick height bands; each band is a new image holding the ORIGINAL pixels whose
value falls in it (holes elsewhere). Store the band map, then compress each band image, then combine.

    python slabs.py DATADIR [image ...]       -> experiments/slabs.json

Decoding order: (1) the band map (which slab each pixel is in), coded with an adaptive counter per context of the neighbours'
bands; (2) the slabs one after another, each pixel in raster order. When a slab is decoded, every pixel of the earlier slabs is
known in EVERY direction, and every still-unknown neighbour's band (hence its range) is known from the map.
Guesser for a pixel = least squares per (slab, busy-level) funnel over 8 neighbours (W, N, NW, NE, E, S, SW, SE): a neighbour's
real value if already decoded, else the middle of its band; plus this pixel's own band middle. Leftover coded with an adaptive
Laplace, renormalised to the band's range. n = 1 is the plain whole-image coder with the same guesser (reference).
Ideal code lengths of the same family for every n, so the comparison between n values is the result.
"""
import json, os, sys
import numpy as np


def geo_sum(a, b, q):                                  # sum_{k=a..b} q^|k|, vectorised, a <= b integers
    a = a.astype(float); b = b.astype(float); one = 1 - q
    pos = lambda lo, hi: (q ** lo - q ** (hi + 1)) / one                       # lo..hi >= 0
    return np.where(a >= 0, pos(np.maximum(a, 0), b), np.where(b <= 0, pos(np.maximum(-b, 0), -a), pos(0, -a) + pos(0, b) - 1))


def ls_code_range(T, y, ctx, lo, hi):                  # least squares per funnel; leftover = adaptive Laplace RENORMALISED to [lo, hi]
    bits = 0.0
    for g in np.unique(ctx):
        m = ctx == g; X = np.c_[T[m], np.ones(m.sum())]; t = y[m].astype(float)
        w = np.linalg.lstsq(X, t, rcond=None)[0]; rp = np.clip(np.rint(X @ w), lo[m], hi[m]); e = t - rp
        k = len(e); sc = np.maximum((np.cumsum(np.abs(e)) - np.abs(e) + 2.0) / (np.arange(k) + 2.0), .3); q = np.exp(-1 / sc)
        mass = geo_sum(lo[m] - rp, hi[m] - rp, q)
        bits += float(-np.log2(np.maximum(q ** np.abs(e) / mass, 1e-12)).sum()) + 16 * X.shape[1]
    return bits

HERE = os.path.dirname(os.path.abspath(__file__))
OFFS = ((0, -1), (-1, 0), (-1, -1), (-1, 1), (0, 1), (1, 0), (1, -1), (1, 1))


def sh(a, dy, dx):
    p = np.pad(a, 1, mode="edge"); H, W = a.shape; return p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]


def adaptive_sym_bits(sym, ctx, nsym):                 # adaptive frequency counts per context (+0.5 prior), raster order
    s = sym.ravel().astype(np.int64); c = ctx.ravel().astype(np.int64); key = c * nsym + s
    o = np.argsort(c, kind="stable"); cs = c[o]; ks = key[o]
    start = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]; grp = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(cs)]))
    rank = np.arange(len(cs)) - start[grp]
    o2 = np.lexsort((np.arange(len(ks)), ks)); kk = ks[o2]
    st2 = np.r_[0, np.flatnonzero(np.diff(kk)) + 1]; g2 = np.repeat(np.arange(len(st2)), np.diff(np.r_[st2, len(kk)]))
    same = np.empty(len(ks), np.int64); same[o2] = np.arange(len(kk)) - st2[g2]
    p = (same + 0.5) / (rank + 0.5 * nsym)
    return float(-np.log2(p).sum())


def slab_bits(P, n):
    P = P.astype(np.int64); H, W = P.shape; width = 256 // n
    band = P // width; mid = band * width + (width - 1) / 2.0
    # (1) band map: context = bands of W, N, NW, NE (already decoded in raster order)
    bits_map = 0.0
    if n > 1:
        ctx = np.zeros_like(band)
        for k, (dy, dx) in enumerate(OFFS[:4]): ctx = ctx * n + sh(band, dy, dx)
        bits_map = adaptive_sym_bits(band, ctx, n)
    # (2) slabs in order; known[y,x] = value if its slab is earlier, or same slab and earlier in raster order
    idx = np.arange(H * W).reshape(H, W); bits_px = 0.0
    for b in range(n):
        m = band == b
        if not m.any(): continue
        taps = []
        for dy, dx in OFFS:
            nb, nv, ni = sh(band, dy, dx), sh(P, dy, dx), sh(idx, dy, dx)
            known = (nb < b) | ((nb == b) & (ni < idx))
            taps.append(np.where(known, nv, sh(mid, dy, dx))[m])
        taps.append(mid[m]); T = np.stack(taps, 1).astype(float)
        act = np.abs(T[:, 0] - T[:, 1]) + np.abs(T[:, 1] - T[:, 2]) + np.abs(T[:, 1] - T[:, 3])
        ctx = np.digitize(act, [3, 6, 12, 24, 48, 96]) + 8 * b
        lo = (band * width)[m]; hi = np.minimum(lo + width - 1, 255)
        bits_px += ls_code_range(T, P[m], ctx, lo, hi)
    return bits_map, bits_px


if __name__ == "__main__":
    D = sys.argv[1]; ref = json.load(open(os.path.join(D, "ref.json"))); names = sys.argv[2:] or list(ref); res = {}
    NS = (1, 2, 4, 8)
    for name in names:
        r = ref[name]; img = np.fromfile(f"{D}/{name}.rgb", np.uint8).reshape(r["h"], r["w"], 3); row = {}
        for n in NS:
            bm = bp = 0.0
            for c in range(3): a, b = slab_bits(img[..., c], n); bm += a; bp += b
            row[n] = {"map": round(bm / 8), "pixels": round(bp / 8), "total": round((bm + bp) / 8)}
        res[name] = row; base = row[1]["total"]
        print(name, "  ".join(f"{n} slab{'s' if n > 1 else ''}: {row[n]['total']:,} (map {row[n]['map']:,} + pixels {row[n]['pixels']:,})"
                              + (f" {100 * (row[n]['total'] / base - 1):+.1f}%" if n > 1 else "") for n in NS), flush=True)
    json.dump(res, open(os.path.join(HERE, "experiments/slabs.json"), "w"), indent=1)
