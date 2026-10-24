"""Real photos: EquaFile / FunnelZip vs JPEG XL (and JPEG, WebP).

  python bench/jxl_vs.py PHOTO_DIR [--out results/jxl_vs.json]

Needs: pip install pillow-jxl-plugin scipy numpy pillow
Part 1, lossless: FunnelZip on raw pixels vs JXL lossless / WebP lossless / PNG (all pixel-exact, verified).
Part 2, lossy:    V7 cosine-equation codec vs JXL / WebP / JPEG, compared at equal file size (PSNR, SSIM).
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pillow_jxl  # noqa: F401  (registers the JXL plugin)
from PIL import Image
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from equafile import funnelzip, waves  # noqa: E402


def save(rgb, fmt, **kw) -> bytes:
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, fmt, **kw)
    return b.getvalue()


def load(data) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def psnr(a, b) -> float:
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def ssim(a, b) -> float:
    """Mean SSIM on luma, 7x7 box window."""
    y = lambda x: x.astype(float) @ [0.299, 0.587, 0.114]
    a, b = y(a), y(b)
    c1, c2, f = (0.01 * 255) ** 2, (0.03 * 255) ** 2, lambda x: uniform_filter(x, 7)
    ma, mb = f(a), f(b)
    va, vb, cov = f(a * a) - ma * ma, f(b * b) - mb * mb, f(a * b) - ma * mb
    return float(np.mean(((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))))


def lossless(rgb) -> dict:
    out, raw = {}, rgb.tobytes()
    t = time.time(); d = save(rgb, "PNG", optimize=True); out["PNG"] = (len(d), time.time() - t)
    t = time.time(); d = save(rgb, "WEBP", lossless=True, quality=100, method=6); out["WebP lossless"] = (len(d), time.time() - t)
    assert np.array_equal(load(d), rgb)
    for e in (7, 9):
        t = time.time(); d = save(rgb, "JXL", lossless=True, effort=e); dt = time.time() - t
        assert np.array_equal(load(d), rgb)
        out[f"JXL lossless e{e}"] = (len(d), dt)
    t = time.time(); enc = funnelzip.encode(raw, threads=4); dt = time.time() - t
    assert funnelzip.decode(enc.payload) == raw   # bit-exact
    out["FunnelZip (raw pixels)"] = (len(enc.payload), dt)
    return out


def curve(rgb, make):
    """make(param) -> (bytes, decoded rgb). Returns [(bytes, psnr, ssim)] sorted by size."""
    pts = []
    for p in make.params:
        n, dec = make(p)
        pts.append((n, psnr(rgb, dec), ssim(rgb, dec)))
    return sorted(pts)


def v7(rgb):
    def make(step):
        payload, _ = waves.encode_image(rgb, step)
        return len(payload), waves.decode_image(payload)
    make.params = [3, 5, 8, 12, 18, 26, 36, 50, 70]
    return make


def codec(rgb, fmt, key, params, **extra):
    def make(p):
        d = save(rgb, fmt, **{key: p}, **extra)
        return len(d), load(d)
    make.params = params
    return make


def at_size(pts, size, col):
    """Interpolate metric col (1=psnr, 2=ssim) at a byte size, on a log-size axis. None if out of range."""
    xs = np.log([p[0] for p in pts]); ys = [p[col] for p in pts]
    s = np.log(size)
    return float(np.interp(s, xs, ys)) if xs[0] <= s <= xs[-1] else None


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path("results/jxl_vs.json")
    res = {}
    for f in sorted(x for x in src.iterdir() if x.suffix.lower() in (".png", ".jpg", ".jpeg")):
        rgb = np.asarray(Image.open(f).convert("RGB"))
        h, w, _ = rgb.shape
        if h > 768 or w > 1024:   # centre crop at native scale (no resampling): keeps real sensor detail, keeps runtime sane
            ch, cw = min(h, 768), min(w, 1024)
            y, x = (h - ch) // 2, (w - cw) // 2
            rgb = np.ascontiguousarray(rgb[y:y + ch, x:x + cw])
        print(f"== {f.name} {rgb.shape[1]}x{rgb.shape[0]}", flush=True)
        r = {"raw": rgb.nbytes, "lossless": lossless(rgb)}
        for k, (n, t) in r["lossless"].items():
            print(f"  {k:26s} {n:9,d} B  {t:6.1f}s", flush=True)
        r["curves"] = {
            "V7 cosine equation": curve(rgb, v7(rgb)),
            "JXL e7": curve(rgb, codec(rgb, "JXL", "quality", [30, 45, 55, 65, 75, 82, 88, 93], effort=7)),
            "WebP": curve(rgb, codec(rgb, "WEBP", "quality", [10, 20, 30, 45, 60, 75, 88], method=6)),
            "JPEG": curve(rgb, codec(rgb, "JPEG", "quality", [10, 20, 30, 45, 60, 75, 88], optimize=True)),
        }
        res[f.name] = r
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))

    print("\n=== LOSSLESS, total over all photos (pixel-exact) ===")
    tot = {}
    for r in res.values():
        for k, (n, _) in r["lossless"].items():
            tot[k] = tot.get(k, 0) + n
    raw = sum(r["raw"] for r in res.values())
    for k, n in sorted(tot.items(), key=lambda kv: kv[1]):
        print(f"  {k:26s} {n:10,d} B  {n / raw * 100:5.1f}% of raw  {raw / n:4.2f}x")

    print("\n=== LOSSY at equal size (mean over photos) ===")
    for label, ref in (("JXL-size", "JXL e7"),):
        for frac in (0.04, 0.08, 0.16):   # fraction of raw size
            rows = {}
            for r in res.values():
                size = r["raw"] * frac
                for k, pts in r["curves"].items():
                    p, s = at_size(pts, size, 1), at_size(pts, size, 2)
                    if p is not None:
                        rows.setdefault(k, []).append((p, s))
            print(f"  target {frac * 100:.0f}% of raw ({24 * frac:.2f} bpp)")
            for k, v in rows.items():
                v = np.array(v)
                print(f"    {k:20s} PSNR {v[:, 0].mean():5.2f} dB  SSIM {v[:, 1].mean():.4f}  ({len(v)} imgs)")


if __name__ == "__main__":
    main()
