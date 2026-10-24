"""Raaj's question: compress the image FILE (its binary) or its PIXELS (positions + values)?

For each image: size of the file itself, FunnelZip on the file bytes, FunnelZip on the decoded pixels
(pixel-exact), and whether re-encoding those pixels gives back the identical file (file-exact).
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from equafile import funnelzip as F  # noqa: E402

S = ROOT / "samples" / "image"
X = ROOT / "samples" / "extra"


def fz(data: bytes) -> int:
    e = F.encode(data)
    assert F.decode(e.payload) == data
    return len(e.payload) + 1


def main() -> None:
    X.mkdir(exist_ok=True)
    land = Image.open(S / "landscape.png").convert("RGB")
    for q in (90, 75):
        land.save(X / f"landscape_q{q}.jpg", quality=q)
    files = [S / "landscape.png", S / "beach_photo.png", S / "wallpaper.png", X / "landscape_q90.jpg", X / "landscape_q75.jpg"]
    rows = []
    for p in files:
        blob = p.read_bytes()
        im = Image.open(io.BytesIO(blob)).convert("RGB")
        px = np.asarray(im).tobytes()
        # can we rebuild the exact same file from the pixels? (try the obvious re-encodings)
        same = False
        for kw in ({"format": "PNG"}, {"format": "PNG", "optimize": True}, {"format": "JPEG", "quality": 90}, {"format": "JPEG", "quality": 75}):
            buf = io.BytesIO()
            try:
                im.save(buf, **kw)
            except OSError:
                continue
            if hashlib.sha256(buf.getvalue()).digest() == hashlib.sha256(blob).digest():
                same = True
        row = {"file": p.name, "kind": p.suffix[1:].upper(), "file_bytes": len(blob), "pixels_raw": len(px),
               "fz_file": fz(blob), "fz_pixels": fz(px), "file_exact_from_pixels": same}
        rows.append(row)
        print(row)
    (ROOT / "results" / "pixels_vs_file.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
