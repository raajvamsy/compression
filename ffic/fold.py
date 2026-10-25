"""FOLDING WAVES: fold the image in half again and again; what stays is a small stack of coarse waves, and every fold leaves a
FOLD WAVE that says exactly how to unfold it (integer, reversible: no loss).

    python fold.py DATADIR [image ...]            -> experiments/fold.json

One fold (both directions) turns a W x H layer into a W/2 x H/2 stack (LL: the waves at half the resolution) plus three fold
waves (HL: across columns, LH: across rows, HH: diagonal). The decoder unfolds coarse -> fine; when it reaches a fold wave,
the whole stack of that level is already known IN EVERY DIRECTION (left, right, above, below), plus the coarser fold waves.

Fair comparison, the same guesser and coder for both:
  raster  each pixel guessed from its known neighbours (left/above only)
  fold    each fold-wave value guessed from the stack around it (all directions) + its known neighbours in the fold wave
          + the same fold wave one level coarser (its "parent funnel")
Guesser = least squares per funnel (context = level x band x how busy the stack is around it); cost = ideal code length of
the integer leftovers with a per-funnel adaptive Laplace model. Bits are estimates of the same kind for both, so the RATIO is
the result; the absolute numbers are not FFIC's.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


SHARP = os.environ.get("FOLD_SHARP") == "1"            # sharp fold: keep every other value exactly; the fold wave = the other values


def sfold(a, axis):                                   # S-transform: pairs -> (rounded mean, difference), exactly invertible
    a = np.moveaxis(a, axis, 0); n = a.shape[0] // 2 * 2
    x, y = a[0:n:2], a[1:n:2]
    if SHARP: return np.moveaxis(x, 0, axis), np.moveaxis(y, 0, axis)
    d = y - x; s = x + (d >> 1)
    return np.moveaxis(s, 0, axis), np.moveaxis(d, 0, axis)


def sunfold(s, d, axis):
    s = np.moveaxis(s, axis, 0); d = np.moveaxis(d, axis, 0)
    x, y = (s, d) if SHARP else (s - (d >> 1), s - (d >> 1) + d)
    out = np.empty((2 * s.shape[0],) + s.shape[1:], s.dtype); out[0::2] = x; out[1::2] = y
    return np.moveaxis(out, 0, axis)


def fold2(a):
    L, H = sfold(a, 1); LL, LH = sfold(L, 0); HL, HH = sfold(H, 0)
    return LL, (HL, LH, HH)


def unfold2(LL, b):
    HL, LH, HH = b; return sunfold(sunfold(LL, LH, 0), sunfold(HL, HH, 0), 1)


def laplace_bits(r, groups):                          # ideal adaptive cost: per group, Laplace with running mean |r| (decoder can do it)
    bits = 0.0
    for g in np.unique(groups):
        e = r[groups == g].astype(float); n = len(e)
        sc = np.maximum((np.cumsum(np.abs(e)) - np.abs(e) + 2.0) / (np.arange(n) + 2.0), .3)   # mean |r| BEFORE this value
        b = sc / np.sqrt(2) * 1.0
        p = np.exp(-np.abs(e) / sc) * (1 - np.exp(-1 / sc)) / (1 + np.exp(-1 / sc))          # discrete Laplace
        bits += -np.log2(np.maximum(p, 1e-12)).sum()
    return bits


def ls_code(taps, target, ctx):                       # least squares per funnel, + its weights' cost (16 bits each)
    r = np.empty(len(target)); wbits = 0
    for g in np.unique(ctx):
        m = ctx == g; X = np.c_[taps[m], np.ones(m.sum())]; y = target[m].astype(float)
        w = np.linalg.lstsq(X, y, rcond=None)[0]; r[m] = np.rint(y - X @ w); wbits += 16 * X.shape[1]
    return laplace_bits(r, ctx) + wbits


def shift(a, dy, dx):                                 # neighbour at offset, edge-padded
    p = np.pad(a, 2, mode="edge"); H, W = a.shape; return p[2 + dy:2 + dy + H, 2 + dx:2 + dx + W]


def busy(a):
    act = sum(np.abs(a - shift(a, dy, dx)) for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)))
    return np.digitize(act, [2, 4, 8, 16, 32, 64, 128])


def raster_bits(P):
    P = P.astype(float); offs = [(0, -1), (-1, 0), (-1, -1), (-1, 1), (0, -2), (-2, 0), (-2, -1), (-2, 1), (-1, -2), (-1, 2)]
    taps = np.stack([shift(P, *o).ravel() for o in offs], 1)
    W_, N_, NW, NE, WW, NN = (shift(P, *o) for o in offs[:6])
    act = np.abs(W_ - WW) + np.abs(N_ - NW) + np.abs(N_ - NE) + np.abs(W_ - NW) + np.abs(N_ - NN)
    return ls_code(taps, P.ravel(), np.digitize(act, [3, 6, 12, 24, 48, 96, 192]).ravel())


def fold_bits(P, levels):
    P = P.astype(np.int64); stacks, bands = [P], []
    for _ in range(levels):
        LL, b = fold2(stacks[-1]); stacks.append(LL); bands.append(b)
    assert np.array_equal(P, _unfold_all(stacks[-1], bands)), "fold is not reversible"
    top = stacks[-1]; bits = raster_bits(top)                               # the final small stack: coded like a tiny image
    for lev in range(levels - 1, -1, -1):
        LL = stacks[lev + 1].astype(float); ctxb = busy(LL)
        known_same = []
        for bi, D in enumerate(bands[lev]):
            D = D.astype(float); taps = []
            for dy in (-1, 0, 1):                                              # the stack around it: ALL directions
                for dx in (-1, 0, 1): taps.append(shift(LL, dy, dx).ravel())
            for o in ((0, -1), (-1, 0), (-1, -1), (-1, 1)): taps.append(shift(D, *o).ravel())   # known neighbours in this fold wave
            for K in known_same:                                               # fold waves of this level already unfolded
                for dy, dx in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)): taps.append(shift(K, dy, dx).ravel())
            if lev + 1 < levels:                                               # parent: same fold wave one level coarser
                par = bands[lev + 1][bi].astype(float); up = np.kron(par, np.ones((2, 2)))[:D.shape[0], :D.shape[1]]
                up = np.pad(up, ((0, D.shape[0] - up.shape[0]), (0, D.shape[1] - up.shape[1])), mode="edge"); taps.append(up.ravel())
            ctx = (bi * 8 + ctxb).ravel() + 100 * lev
            bits += ls_code(np.stack(taps, 1), D.ravel(), ctx)
            known_same.append(D)
    return bits


def _unfold_all(top, bands):
    s = top
    for b in reversed(bands): s = unfold2(s, b)
    return s


if __name__ == "__main__":
    D = sys.argv[1]; ref = json.load(open(os.path.join(D, "ref.json"))); names = sys.argv[2:] or list(ref); res = {}
    for n in names:
        r = ref[n]; img = np.fromfile(f"{D}/{n}.rgb", np.uint8).reshape(r["h"], r["w"], 3)
        row = {"raster": 0.0}; LV = (1, 2, 3, 5)
        for L in LV: row[f"fold{L}"] = 0.0
        for c in range(3):
            P = img[..., c]; row["raster"] += raster_bits(P) / 8
            for L in LV: row[f"fold{L}"] += fold_bits(P, L) / 8
        res[n] = {k: round(v) for k, v in row.items()}
        print(n, "  ".join(f"{k} {v:,.0f}" + (f" ({100 * (v / row['raster'] - 1):+.1f}%)" if k != "raster" else "") for k, v in row.items()), flush=True)
    if len(names) > 1:
        tot = {k: sum(r[k] for r in res.values()) for k in next(iter(res.values()))}
        print("TOTAL", "  ".join(f"{k} {v:,} ({100 * (v / tot['raster'] - 1):+.2f}%)" for k, v in tot.items()))
    json.dump(res, open(os.path.join(HERE, "experiments/fold" + ("_sharp" if SHARP else "") + ".json"), "w"), indent=1)
