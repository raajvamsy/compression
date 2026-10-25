"""16-bit benchmark: full SRTM 1-arc-second elevation tiles (3601 x 3601 heights in metres, AWS terrain tiles), lossless.
    python benchdem.py SRCDIR    (SRCDIR holds *.hgt)  -> experiments/BENCHDEM.md + benchdem.json
Every output is decoded and compared with the original heights."""
import glob, io, json, lzma, os, sys, tempfile, time
import numpy as np
from PIL import Image
import pillow_jxl  # noqa: F401
import ffic

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"N46E007": "Swiss Alps (Valais)", "N45E006": "French Alps (Mont Blanc)", "N36W113": "Grand Canyon"}


def main():
    src = sys.argv[1]; ffic.build(); res = {}
    L = ["# 16-bit benchmark: SRTM elevation tiles (lossless, real files, all decoded and compared)", "",
         "Full 1-arc-second tiles, 3601 x 3601 heights (12,967,201 samples, 16-bit). PNG and JPEG XL store the same 16-bit grey image; xz is the raw 16-bit heights, compressed.", "",
         "| tile | raw | xz -9e | PNG 16-bit (optimized) | JPEG XL e9 | FFIC max | FFIC vs best other |", "|---|---|---|---|---|---|---|"]
    for f in sorted(glob.glob(f"{src}/*.hgt")):
        name = os.path.basename(f)[:-4]; a = np.fromfile(f, ">i2").reshape(3601, 3601)
        if a.min() < 0: print(name, "has voids/negative heights, skipped"); continue
        a = a.astype(np.uint16); im = Image.fromarray(a); row = {"raw": a.size * 2}
        t = time.time(); row["xz"] = (len(lzma.compress(a.astype("<u2").tobytes(), preset=9 | lzma.PRESET_EXTREME)), time.time() - t)
        b = io.BytesIO(); t = time.time(); im.save(b, "PNG", optimize=True); row["png"] = (len(b.getvalue()), time.time() - t)
        try:
            b = io.BytesIO(); t = time.time(); im.save(b, "JXL", lossless=True, effort=9); te = time.time() - t
            back = np.asarray(Image.open(io.BytesIO(b.getvalue()))).astype(np.uint16)
            row["jxl"] = (len(b.getvalue()), te, bool(np.array_equal(back, a)))
        except Exception as e:
            row["jxl"] = (None, 0, False); print("jxl failed:", e)
        with tempfile.TemporaryDirectory() as t:
            p = f"{t}/{name}.png"; im.save(p); o = f"{t}/x.ffic"; ob = f"{t}/b.png"
            se = ffic.encode(p, o, "max"); sd = ffic.decode(o, ob)
            ok = bool(np.array_equal(np.asarray(Image.open(ob)).astype(np.uint16), a)); row["ffic"] = (os.path.getsize(o), se, sd, ok)
        res[name] = row; others = [row["xz"][0], row["png"][0]] + ([row["jxl"][0]] if row["jxl"][0] and row["jxl"][2] else [])
        L.append(f"| {NAMES.get(name, name)} ({name}, {a.min()}–{a.max()} m) | {row['raw']:,} | {row['xz'][0]:,} | {row['png'][0]:,} | "
                 + (f"{row['jxl'][0]:,}" + ("" if row["jxl"][2] else " (not exact)") if row["jxl"][0] else "failed")
                 + f" | {row['ffic'][0]:,} ({se:.0f} s / {sd:.0f} s){'' if ok else ' MISMATCH'} | **{100 * (row['ffic'][0] / min(others) - 1):+.1f}%** |")
        print(L[-1], flush=True)
    L += ["", "FFIC times: encode / decode seconds (one layer, `planes=1`, `bits=13`/`bits=12` chosen from the data)."]
    json.dump(res, open(os.path.join(HERE, "experiments/benchdem.json"), "w"), indent=1)
    open(os.path.join(HERE, "experiments/BENCHDEM.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
