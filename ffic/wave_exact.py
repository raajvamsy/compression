"""TEST D, WAVES ALL THE WAY DOWN: the file is ONLY a wave tree, no per-pixel corrections.

    python wave_exact.py DATADIR [image ...]

Each region of a colour layer is drawn as  mean + horizontal wave(x) + vertical wave(y)  (each wave: frequency, amplitude,
phase of the strongest sinusoid in the region's average profile). A region is split until its drawing, rounded, equals every
pixel exactly; a 1x1 region is just its value. The stream = split flags + wave numbers, compressed with lzma (xz -9e).
A separate decoder rebuilds the image FROM THE STREAM ONLY and the result is compared with the original byte for byte.
Writes experiments/wave_exact.json.
"""
import json, lzma, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def fit_wave(prof):
    n = len(prof)
    if n < 2: return 0, 0, 0
    F = np.fft.rfft(prof - prof.mean()); k = 1 + int(np.argmax(np.abs(F[1:])))
    return k, int(round(2 * np.abs(F[k]) / n)), int(round(np.angle(F[k]) / (2 * np.pi) * 64)) % 64


def wave_of(n, k, amp, ph):
    if not amp: return np.zeros(n)
    t = np.arange(n); return amp * np.cos(2 * np.pi * k * t / n + ph / 64 * 2 * np.pi)


def draw(h, w, p):                                       # p = (mean, kh, ah, phh, kv, av, phv)
    return np.clip(np.rint(p[0] + wave_of(w, *p[1:4])[None, :] + wave_of(h, *p[4:7])[:, None]), 0, 255)


def halves(y0, x0, h, w):                                # split in 4 (or in 2 when one side is 1 pixel)
    if h > 1 and w > 1:
        h2, w2 = h // 2, w // 2
        return [(y0, x0, h2, w2), (y0, x0 + w2, h2, w - w2), (y0 + h2, x0, h - h2, w2), (y0 + h2, x0 + w2, h - h2, w - w2)]
    if h > 1: return [(y0, x0, h // 2, w), (y0 + h // 2, x0, h - h // 2, w)]
    return [(y0, x0, h, w // 2), (y0, x0 + w // 2, h, w - w // 2)]


def encode(P):
    H, W = P.shape; flags, nums = [], []; leaves = [0, 0]
    stack = [(0, 0, H, W)]
    while stack:
        y0, x0, h, w = stack.pop(); b = P[y0:y0 + h, x0:x0 + w]
        if h * w == 1: nums.append(int(b[0, 0])); leaves[1] += 1; continue          # a single pixel: no flag, just its value
        p = (int(round(b.mean())),) + fit_wave(b.mean(0)) + fit_wave(b.mean(1))
        if np.array_equal(draw(h, w, p), b): flags.append(0); nums.extend(p); leaves[0] += 1
        else: flags.append(1); stack.extend(reversed(halves(y0, x0, h, w)))
    return flags, nums, leaves


def decode(flags, nums, H, W):
    out = np.zeros((H, W)); fi = ni = 0; stack = [(0, 0, H, W)]
    while stack:
        y0, x0, h, w = stack.pop()
        if h * w == 1: out[y0, x0] = nums[ni]; ni += 1; continue
        f = flags[fi]; fi += 1
        if f == 0: out[y0:y0 + h, x0:x0 + w] = draw(h, w, nums[ni:ni + 7]); ni += 7
        else: stack.extend(reversed(halves(y0, x0, h, w)))
    return out


def run(args):
    D, name, h, w = args; t0 = time.time()
    img = np.fromfile(f"{D}/{name}.rgb", np.uint8).reshape(h, w, 3).astype(float)
    planes, streams, waves, singles = [], [], 0, 0
    for c in range(3):
        fl, nu, lv = encode(img[..., c]); waves += lv[0]; singles += lv[1]
        blob = np.packbits(np.array(fl, np.uint8)).tobytes() + np.array(nu, np.int16).tobytes()
        streams.append(lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME))
        # decoder: from the compressed stream only
        raw = lzma.decompress(streams[-1]); nf = len(fl)
        fl2 = np.unpackbits(np.frombuffer(raw[:(nf + 7) // 8], np.uint8))[:nf].tolist()
        nu2 = np.frombuffer(raw[(nf + 7) // 8:], np.int16).tolist()
        planes.append(decode(fl2, nu2, h, w))
    exact = bool(np.array_equal(np.stack(planes, -1), img))
    return name, {"bytes": sum(len(s) for s in streams), "wave_regions": waves, "single_pixels": singles, "exact": exact, "s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    D = sys.argv[1]; ref = json.load(open(os.path.join(D, "ref.json"))); names = sys.argv[2:] or list(ref)
    with ProcessPoolExecutor(4) as ex:
        res = dict(ex.map(run, [(D, n, ref[n]["h"], ref[n]["w"]) for n in names]))
    log = {}
    for l in open(os.path.join(HERE, "experiments/log.jsonl")): r = json.loads(l); log[r["id"]] = r
    best = log["best_now"]["per_image"]
    for n, r in res.items():
        print(f"{n:14} waves-only {r['bytes']:>9,} B  vs JXL {100 * (r['bytes'] / ref[n]['jxl9'] - 1):+6.1f}%  vs best coder {100 * (r['bytes'] / best[n]['bytes'] - 1):+6.1f}%  "
              f"wave regions {r['wave_regions']:>7,}  single pixels {r['single_pixels']:>9,}  exact {r['exact']}", flush=True)
    if len(res) == len(ref):
        tot = sum(r["bytes"] for r in res.values()); jx = sum(v["jxl9"] for v in ref.values())
        print(f"TOTAL {tot:,} B = {100 * (tot / jx - 1):+.1f}% vs JXL e9, {100 * (tot / log['best_now']['total_bytes'] - 1):+.1f}% vs best coder; all exact: {all(r['exact'] for r in res.values())}")
    json.dump(res, open(os.path.join(HERE, "experiments/wave_exact.json"), "w"), indent=1)
