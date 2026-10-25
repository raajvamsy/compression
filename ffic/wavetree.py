"""WAVE TREE: describe each colour layer with one horizontal and one vertical wave per region, splitting a region into four
where the two waves do not fit (like a funnel splitting where it needs more detail).

    python wavetree.py DATADIR OUTDIR T MINSIZE

For every image and colour layer: the region's pixel ~ hwave(x) + vwave(y) - mean, each wave = amplitude, frequency, phase of
the strongest sinusoid in the region's column / row average profile. Split when the RMS misfit > T and the region is at least
2 x MINSIZE. Parameters are quantised (mean, amplitude to integers; phase to 64 steps) and the model is rebuilt FROM THE
QUANTISED parameters, so a decoder holding the parameters gets exactly the same model.
Writes OUTDIR/<image>.wave (float32, planes R,G,B) and OUTDIR/<image>.wave.side (byte count of the lzma-compressed tree:
split flags + parameters), which ffic adds to the file size.
"""
import json, lzma, os, sys
import numpy as np


def fit_wave(prof):
    n = len(prof); m = prof.mean()
    if n < 4: return 0, 0, 0
    F = np.fft.rfft(prof - m); k = 1 + int(np.argmax(np.abs(F[1:])))
    amp = int(round(2 * np.abs(F[k]) / n)); ph = int(round(np.angle(F[k]) / (2 * np.pi) * 64)) % 64
    return k, min(amp, 255), ph


def wave_of(n, k, amp, ph):
    t = np.arange(n); return amp * np.cos(2 * np.pi * k * t / n + ph / 64 * 2 * np.pi) if amp else np.zeros(n)


def build(P, T, minsz):
    H, W = P.shape; model = np.zeros((H, W), np.float32); flags, params = [], []
    def go(y0, x0, h, w):
        b = P[y0:y0 + h, x0:x0 + w]; mean = int(round(b.mean()))
        kh, ah, ph_h = fit_wave(b.mean(0)); kv, av, ph_v = fit_wave(b.mean(1))
        m = mean + wave_of(w, kh, ah, ph_h)[None, :] + wave_of(h, kv, av, ph_v)[:, None]
        err = np.sqrt(((b - m) ** 2).mean())
        if err > T and h >= 2 * minsz and w >= 2 * minsz:
            flags.append(1); h2, w2 = h // 2, w // 2
            for a, c, hh, ww in ((0, 0, h2, w2), (0, w2, h2, w - w2), (h2, 0, h - h2, w2), (h2, w2, h - h2, w - w2)): go(y0 + a, x0 + c, hh, ww)
        else:
            if h >= 2 * minsz and w >= 2 * minsz: flags.append(0)
            params.extend([mean, kh, ah, ph_h, kv, av, ph_v]); model[y0:y0 + h, x0:x0 + w] = np.clip(m, 0, 255)
    go(0, 0, H, W)
    return model, flags, params


if __name__ == "__main__":
    D, OUT, T, MS = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    os.makedirs(OUT, exist_ok=True); ref = json.load(open(os.path.join(D, "ref.json"))); tot_side = tot_nodes = 0
    for name, r in ref.items():
        img = np.fromfile(f"{D}/{name}.rgb", np.uint8).reshape(r["h"], r["w"], 3).astype(float)
        models, blob, nodes = [], bytearray(), 0
        for c in range(3):
            m, fl, pa = build(img[..., c], T, MS); models.append(m); nodes += len(pa) // 7
            blob += np.packbits(np.array(fl, np.uint8)).tobytes() + np.array(pa, np.int16).tobytes()   # k may exceed 255
        np.stack(models).astype(np.float32).tofile(f"{OUT}/{name}.wave")
        side = len(lzma.compress(bytes(blob), preset=9 | lzma.PRESET_EXTREME)); open(f"{OUT}/{name}.wave.side", "w").write(str(side))
        tot_side += side; tot_nodes += nodes
        print(f"{name:14} regions {nodes:6}  tree {side:7,} B", flush=True)
    print(f"total regions {tot_nodes:,}, tree bytes {tot_side:,}")
