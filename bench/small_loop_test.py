"""Raaj's follow-up: single-digit loop counts + simple equations + best choice per piece."""
from __future__ import annotations

import hashlib
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from equafile import baselines, smallloop  # noqa: E402
from equafile.core import registry, run  # noqa: E402

S = ROOT / "samples"


def files() -> dict[str, bytes]:
    f = {p.name: p.read_bytes() for p in sorted((S / "text").iterdir())}
    f["beach photo (raw pixels)"] = np.asarray(Image.open(S / "image" / "beach_photo.png").convert("RGB")).tobytes()
    with wave.open(str(S / "audio" / "chord.wav")) as w:
        f["chord (raw PCM)"] = w.readframes(w.getnframes())
    return f


def main() -> None:
    T = smallloop.table()
    reach = []
    for L in range(3, 10):
        uniq = len(np.unique(T[:, :L], axis=0))
        reach.append({"L": L, "distinct": uniq, "log2_possible": 8 * L, "log2_distinct": float(np.log2(uniq))})
        print(f"L={L}: {uniq:,} distinct sequences out of 2^{8 * L}")
    rows = []
    others = {k: v for k, v in registry().items() if k not in ("v2-loop", "v3-seed", "v8-smallloop")}
    for name, data in files().items():
        t = time.perf_counter()
        enc = smallloop.encode(data, levels=3)
        assert smallloop.decode(enc.payload) == data, name
        lv = enc.info["level_sizes"]
        one = smallloop.encode(data, levels=1)
        assert hashlib.sha256(smallloop.decode(one.payload)).digest() == hashlib.sha256(data).digest()
        best_other = min(((r["stored"], k) for k, v in others.items()
                          if (r := run(v, data))["status"] == "ok"), default=(None, None))
        base = baselines.lossless(data)
        bname, bsize = min(base.items(), key=lambda kv: kv[1])
        row = {"file": name, "size": len(data), "v8": len(one.payload) + 1, "coverage": one.info["covered"] / len(data),
               "levels": lv, "best_other_eq": best_other, "best_existing": [bname, bsize],
               "auto": min(len(one.payload) + 1, best_other[0] or 1 << 60), "example": one.equation,
               "sec": round(time.perf_counter() - t, 1)}
        rows.append(row)
        print(f"{name:26} {len(data):>7} v8={row['v8']:>7} cover={row['coverage']:.1%} levels={lv} "
              f"other_eq={best_other} existing={bname}:{bsize}")
    out = ROOT / "results" / "smallloop.json"
    out.write_text(json.dumps({"reach": reach, "rows": rows}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
