"""Raw RGB + JXL e9 reference sizes for the FFIC test set. usage: prep_data.py OUTDIR PEXELS_DIR KODAK_DIR"""
import glob, io, json, os, sys
import numpy as np
from PIL import Image
import pillow_jxl  # noqa
out, pex, kod = sys.argv[1:4]
ref = {}
def add(name, im):
    a = np.asarray(im.convert("RGB")); h, w, _ = a.shape
    if h > 768 or w > 1024:
        ch, cw = min(h, 768), min(w, 1024); y, x = (h - ch) // 2, (w - cw) // 2; a = np.ascontiguousarray(a[y:y + ch, x:x + cw]); h, w, _ = a.shape
    open(f"{out}/{name}.rgb", "wb").write(a.tobytes())
    b = io.BytesIO(); Image.fromarray(a).save(b, "JXL", lossless=True, effort=9)
    b7 = io.BytesIO(); Image.fromarray(a).save(b7, "JXL", lossless=True, effort=7)
    ref[name] = {"w": w, "h": h, "jxl9": len(b.getvalue()), "jxl7": len(b7.getvalue())}
for f in sorted(glob.glob(pex + "/*.jpg")): add("pex_" + os.path.basename(f)[:-4], Image.open(f))
for f in sorted(glob.glob(kod + "/kodim*.png")): add("kod_" + os.path.basename(f)[:-4], Image.open(f))
json.dump(ref, open(out + "/ref.json", "w"), indent=1); print(len(ref), "images")
