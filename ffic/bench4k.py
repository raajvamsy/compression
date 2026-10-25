"""4K benchmark: centre 3840x2160 crops of full-size Pexels originals, FFIC presets vs JPEG XL lossless (real files, all decoded
and compared). python bench4k.py SRCDIR OUTDIR  -> experiments/BENCH4K.md + bench4k.json"""
import glob, io, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import pillow_jxl  # noqa: F401  (registers JXL)
import ffic

HERE = os.path.dirname(os.path.abspath(__file__))


def crop(path):
    im = Image.open(path).convert("RGB"); w, h = im.size
    if w < 3840 or h < 2160: return None
    x, y = (w - 3840) // 2, (h - 2160) // 2; return im.crop((x, y, x + 3840, y + 2160))


def jxl(im, effort):
    t = time.time(); b = io.BytesIO(); im.save(b, "JXL", lossless=True, effort=effort); te = time.time() - t
    data = b.getvalue(); t = time.time(); back = Image.open(io.BytesIO(data)).convert("RGB"); td = time.time() - t
    return len(data), te, td, back.tobytes() == im.tobytes()


def ours(png, out, preset, ref):
    f = os.path.join(out, os.path.basename(png).replace(".png", f".{preset}.ffic")); o = f.replace(".ffic", ".back.png")
    se, _ = ffic.encode(png, f, preset); sd = ffic.decode(f, o)
    ok = Image.open(o).convert("RGB").tobytes() == ref.tobytes(); n = os.path.getsize(f); os.remove(o)
    return n, se, sd, ok


def main():
    src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); ffic.build(); res = {}
    for p in sorted(glob.glob(f"{src}/*.jpg")):
        im = crop(p)
        if im is None: continue
        name = os.path.basename(p)[:-4]; png = f"{out}/{name}_4k.png"; im.save(png)
        row = {"jxl9": jxl(im, 9)}
        with ThreadPoolExecutor(2) as ex:
            fm, ff = ex.submit(ours, png, out, "max", im), ex.submit(ours, png, out, "fast", im)
            row["ffic_max"], row["ffic_fast"] = fm.result(), ff.result()
        res[name] = row; j = row["jxl9"][0]
        print(name, "  ".join(f"{k} {v[0]:,} B ({100 * (v[0] / j - 1):+.2f}%) enc {v[1]:.1f}s dec {v[2]:.1f}s {'exact' if v[3] else 'MISMATCH'}" for k, v in row.items()), flush=True)
    json.dump(res, open(os.path.join(HERE, "experiments/bench4k.json"), "w"), indent=1)
    tot = {k: sum(r[k][0] for r in res.values()) for k in ("jxl9", "ffic_max", "ffic_fast")}
    L = ["# 4K benchmark (3840 x 2160, lossless, real files)", "",
         "Centre crops of full-size Pexels originals (camera JPEGs, so the crops still carry their 8x8 JPEG structure). Every file was decoded and compared with the original.", "",
         "| photo | JPEG XL e9 | FFIC max | FFIC fast |", "|---|---|---|---|"]
    for n, r in res.items():
        j = r["jxl9"][0]
        L.append(f"| {n} | {j:,} B ({r['jxl9'][1]:.0f} s / {r['jxl9'][2]:.1f} s) | " + " | ".join(f"{r[k][0]:,} B, {100 * (r[k][0] / j - 1):+.2f}% ({r[k][1]:.0f} s / {r[k][2]:.0f} s){'' if r[k][3] else ' MISMATCH'}" for k in ("ffic_max", "ffic_fast")) + " |")
    L.append(f"| **total** | **{tot['jxl9']:,} B** | **{tot['ffic_max']:,} B, {100 * (tot['ffic_max'] / tot['jxl9'] - 1):+.2f}%** | **{tot['ffic_fast']:,} B, {100 * (tot['ffic_fast'] / tot['jxl9'] - 1):+.2f}%** |")
    L += ["", "Times are encode / decode seconds on this machine (FFIC max and fast ran side by side, sharing 4 cores)."]
    open(os.path.join(HERE, "experiments/BENCH4K.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
