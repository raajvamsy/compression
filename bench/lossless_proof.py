"""Is FunnelZip lossless? For every file: compress, decompress, and compare SHA-256 and every byte."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from equafile import funnelzip as F  # noqa: E402
from funnelzip_bench import files  # noqa: E402


def main() -> None:
    rows = []
    for name, kind, data, _ in files():
        t = time.perf_counter()
        payload = F.encode(data).payload
        te = time.perf_counter() - t
        t = time.perf_counter()
        back = F.decode(payload)
        td = time.perf_counter() - t
        a = np.frombuffer(data, dtype=np.uint8)
        b = np.frombuffer(back, dtype=np.uint8)
        diff = int(np.count_nonzero(a != b)) if len(a) == len(b) else -1
        row = {"file": name, "kind": kind, "size": len(data), "stored": len(payload) + 1,
               "sha_in": hashlib.sha256(data).hexdigest(), "sha_out": hashlib.sha256(back).hexdigest(),
               "bytes_different": diff, "same_length": len(a) == len(b),
               "enc_mbps": round(len(data) / 1e6 / te, 3), "dec_mbps": round(len(data) / 1e6 / td, 3),
               "enc_s": round(te, 2), "dec_s": round(td, 2)}
        row["lossless"] = row["sha_in"] == row["sha_out"] and diff == 0
        rows.append(row)
        print(f"{name:44} {len(data):>9} -> {row['stored']:>9}  diff bytes={diff}  lossless={row['lossless']}  "
              f"enc {row['enc_mbps']:.2f} MB/s  dec {row['dec_mbps']:.2f} MB/s")
    out = ROOT / "results" / "lossless_proof.json"
    out.write_text(json.dumps(rows, indent=1))
    print("all lossless:", all(r["lossless"] for r in rows), "| wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
