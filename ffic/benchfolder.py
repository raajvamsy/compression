"""Folder benchmark: a folder of ordinary PNG photos compressed as one archive, like zip.
    python benchfolder.py DATADIR WORKDIR   (DATADIR: the 20 test photos as .rgb + ref.json) -> experiments/BENCHFOLDER.md
Builds WORKDIR/photos/ with the 20 photos as normal optimized PNGs, then compares: zip -9, tar.xz -9e, 7z LZMA2 ultra,
converting every photo to JPEG XL lossless e9, and `ffic.py pack` (unpacked again and every photo compared pixel for pixel)."""
import json, lzma, os, shutil, subprocess, sys, tarfile, time, io
import numpy as np
from PIL import Image
import pillow_jxl  # noqa: F401
import py7zr
import ffic

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    D, W = sys.argv[1], sys.argv[2]; P = f"{W}/photos"; shutil.rmtree(W, ignore_errors=True); os.makedirs(P)
    ref = json.load(open(f"{D}/ref.json"))
    for n, r in ref.items():
        Image.fromarray(np.fromfile(f"{D}/{n}.rgb", np.uint8).reshape(r["h"], r["w"], 3)).save(f"{P}/{n}.png", optimize=True)
    files = sorted(os.listdir(P)); size = sum(os.path.getsize(f"{P}/{f}") for f in files); res = {"folder of PNGs": (size, 0, 0)}
    t = time.time(); subprocess.run(["zip", "-q", "-9", "-r", f"{W}/a.zip", "photos"], cwd=W, check=True); res["zip -9"] = (os.path.getsize(f"{W}/a.zip"), time.time() - t, 0)
    t = time.time()
    with tarfile.open(f"{W}/a.tar", "w") as tf: tf.add(P, "photos")
    open(f"{W}/a.tar.xz", "wb").write(lzma.compress(open(f"{W}/a.tar", "rb").read(), preset=9 | lzma.PRESET_EXTREME)); res["tar.xz -9e"] = (os.path.getsize(f"{W}/a.tar.xz"), time.time() - t, 0)
    t = time.time()
    with py7zr.SevenZipFile(f"{W}/a.7z", "w", filters=[{"id": py7zr.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]) as z: z.writeall(P, "photos")
    res["7z LZMA2 ultra"] = (os.path.getsize(f"{W}/a.7z"), time.time() - t, 0)
    t = time.time(); jx = 0
    for f in files:
        b = io.BytesIO(); Image.open(f"{P}/{f}").save(b, "JXL", lossless=True, effort=9); jx += len(b.getvalue())
    res["every photo -> JPEG XL e9"] = (jx, time.time() - t, 0)
    man, se = ffic.pack(P, f"{W}/a.ffa", "max"); U = f"{W}/unpacked"; _, sd = ffic.unpack(f"{W}/a.ffa", U)
    ok = all(np.array_equal(np.asarray(Image.open(f"{P}/{f}")), np.asarray(Image.open(f"{U}/{f}"))) for f in files)
    res["ffic.py pack (max)"] = (os.path.getsize(f"{W}/a.ffa"), se, sd, ok)
    L = ["# Folder benchmark: 20 photos as ordinary PNG files, compressed as one archive", "",
         "The 20 test photos (12 Pexels crops 1024x768, 8 Kodak 768x512) saved as normal optimized PNGs in one folder, then archived. "
         "zip / tar.xz / 7z keep the PNG files byte for byte; JPEG XL and FFIC keep every pixel exactly (the PNG files they give back hold "
         "identical pixels, but the PNG bytes themselves are re-encoded).", "",
         "| method | archive size | vs the PNG folder | time to pack |", "|---|---|---|---|"]
    for k, v in res.items():
        L.append(f"| {k} | {v[0]:,} B | {100 * (v[0] / size - 1):+.1f}% | {v[1]:.0f} s |" if k != "folder of PNGs" else f"| {k} (20 files) | {v[0]:,} B | — | — |")
    L += ["", f"FFIC unpack: {sd:.0f} s, every photo pixel-exact: {'yes' if ok else 'NO'}. FFIC packs 4 photos at a time on 4 cores."]
    open(os.path.join(HERE, "experiments/BENCHFOLDER.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
