"""Raaj's box-in-box idea: split an image into n x n boxes, each box into n x n again, ... and compress boxes
instead of the whole image. Four ways to apply it, all lossless, compared with the whole image:

  fixed      every box at depth d coded by its own fresh model (independent, so all boxes run in parallel)
  warm       same boxes, but each box's model first reads the box above and the box to the left (no bytes
             stored for them: the decoder has them already), so no box starts from nothing
  adaptive   quadtree: split a box into 4 only if the 4 children together code smaller than the box
  nested     one model, whole image, but pixels visited box-in-box (Morton / Z-order) instead of row by row

Writes results/box_test.json."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from equafile import funnelzip as F  # noqa: E402
from funnelzip_bench import files  # noqa: E402

IMAGES = {"landscape (raw RGB)": (512, 512), "wallpaper (raw RGB)": (512, 288), "beach_photo (raw RGB)": (214, 130),
          "synthetic (raw RGB)": (512, 128)}
HDR = 3   # bytes of side info per coded box (its compressed length), counted honestly


def code(px: np.ndarray, mask: int, prime: bytes | None = None) -> int:
    h, w, _ = px.shape
    if prime:
        return len(F.raw_compress(px.tobytes(), 1, 3, w * 3, mask, 0, True, prime)) + HDR
    return len(F.raw_compress(px.tobytes(), 1, 3, w * 3, mask, 0)) + HDR


def boxes(h: int, w: int, n: int, depth: int) -> list[tuple[int, int, int, int]]:
    k = n ** depth
    ys, xs = np.linspace(0, h, k + 1).astype(int), np.linspace(0, w, k + 1).astype(int)
    return [(ys[i], ys[i + 1], xs[j], xs[j + 1]) for i in range(k) for j in range(k)]


def fixed(img, n, depth, mask):
    bs = boxes(*img.shape[:2], n, depth)
    with ThreadPoolExecutor(12) as ex:
        return sum(ex.map(lambda b: code(img[b[0]:b[1], b[2]:b[3]], mask), bs)), len(bs)


def warm(img, n, depth, mask):
    """Each box's model first reads its upper and left neighbour boxes (resized to the box's width), stored nowhere."""
    bs = boxes(*img.shape[:2], n, depth)

    def one(b):
        y0, y1, x0, x1 = b
        ctx = []
        if y0 > 0:
            ctx.append(img[max(0, y0 - (y1 - y0)):y0, x0:x1])
        if x0 > 0:
            left = img[y0:y1, max(0, x0 - (x1 - x0)):x0]
            if left.shape[1] == x1 - x0:
                ctx.append(left)
        prime = b"".join(c.tobytes() for c in ctx)
        return code(img[y0:y1, x0:x1], mask, prime or None)
    with ThreadPoolExecutor(12) as ex:
        return sum(ex.map(one, bs)), len(bs)


def adaptive(img, mask, min_side=32):
    """Bottom-up quadtree: a box keeps its children only if they code smaller. One flag bit per decision."""
    def best(y0, y1, x0, x1):
        whole = code(img[y0:y1, x0:x1], mask)
        if (y1 - y0) < 2 * min_side or (x1 - x0) < 2 * min_side:
            return whole, 1
        ym, xm = (y0 + y1) // 2, (x0 + x1) // 2
        kids = [best(*b) for b in ((y0, ym, x0, xm), (y0, ym, xm, x1), (ym, y1, x0, xm), (ym, y1, xm, x1))]
        split = sum(k[0] for k in kids)
        return (split, sum(k[1] for k in kids)) if split < whole else (whole, 1)
    h, w = img.shape[:2]
    return best(0, h, 0, w)


def morton(img):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    key = np.zeros((h, w), dtype=np.int64)
    for b in range(10):
        key |= ((xx >> b) & 1) << (2 * b) | ((yy >> b) & 1) << (2 * b + 1)
    order = np.argsort(key.ravel(), kind="stable")
    return img.reshape(-1, 3)[order]


def main() -> None:
    data = {name: d for name, kind, d, _ in files() if name in IMAGES}
    published = {r["file"]: r for r in json.loads((ROOT / "results" / "funnelzip.json").read_text())["rows"]}
    out = []
    for name, (w, h) in IMAGES.items():
        img = np.frombuffer(data[name], dtype=np.uint8).reshape(h, w, 3)
        mask = published[name]["params"][6]
        row = {"file": name, "w": w, "h": h, "full_funnelzip": published[name]["funnelzip"],
               "best_existing": published[name]["best_existing"], "runs": []}
        t = time.perf_counter()
        whole = code(img, mask)
        row["runs"].append({"how": "whole image, one model", "bytes": whole, "boxes": 1, "s": round(time.perf_counter() - t, 2)})
        for n in (2, 3):
            for depth in (1, 2, 3):
                if n ** depth > min(w, h) // 8:
                    continue
                for kind, fn in (("fixed", fixed), ("warm", warm)):
                    t = time.perf_counter()
                    b, k = fn(img, n, depth, mask)
                    row["runs"].append({"how": f"{kind}: {n}x{n} boxes, depth {depth}", "kind": kind, "n": n, "depth": depth,
                                        "bytes": b, "boxes": k, "s": round(time.perf_counter() - t, 2)})
        t = time.perf_counter()
        b, k = adaptive(img, mask)
        row["runs"].append({"how": "adaptive quadtree (split only if smaller)", "kind": "adaptive", "bytes": b + (k * 4 + 7) // 8,
                            "boxes": k, "s": round(time.perf_counter() - t, 2)})
        t = time.perf_counter()
        z = morton(img)
        b = len(F.raw_compress(z.tobytes(), 1, 3, 0, mask, 0)) + HDR
        row["runs"].append({"how": "one model, pixels in box-in-box (Z) order", "kind": "nested", "bytes": b, "boxes": 1,
                            "s": round(time.perf_counter() - t, 2)})
        for r in row["runs"]:
            print(f"{name:24} {r['how']:44} {r['bytes']:>8}  {100 * (r['bytes'] / whole - 1):+6.1f}%  boxes {r['boxes']:>4}  {r['s']:5.2f}s", flush=True)
        out.append(row)
    (ROOT / "results" / "box_test.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
