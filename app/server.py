"""EquaFile sample app: a local web UI over the engine (stdlib server, no framework).

    .venv/bin/python app/server.py      ->  http://localhost:8765
"""
from __future__ import annotations

import base64
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equafile import baselines, loop, polynomial, waves  # noqa: E402
from equafile.core import registry, run  # noqa: E402

PORT = 8765
APP_LIMITS = {"v2-loop": 2}  # keep the UI responsive: the pure loop on 3 bytes takes ~10 s


def b64png(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def api_encode(data: bytes) -> dict:
    rows = []
    for v in registry().values():
        if len(data) > APP_LIMITS.get(v.ID, 1 << 62):
            rows.append({"variant": v.ID, "name": v.NAME, "status": "skipped",
                         "reason": f"~2^{len(data) * 8} loop steps (CLI handles up to 3 bytes)"})
            continue
        r = run(v, data)
        r.pop("info", None)
        rows.append(r)
    return {"size": len(data), "variants": rows, "baselines": baselines.lossless(data),
            "bits": "".join(f"{b:08b}" for b in data[:8])}


def api_loop(text: str) -> dict:
    data = text.encode()[:2] or b"hi"
    n = len(data) * 8
    target = int.from_bytes(data, "big")
    L, head = loop.loop_search(target, n, trace=40)
    seed = loop.seed_search(target, n)
    tree = loop.TreeVariant.encode(data)
    return {
        "text": data.decode(errors="replace"), "bits": n, "target": f"{target:0{n}b}",
        "trace": [[s, f"{v:0{n}b}"] for s, v in head], "L": L, "L_bits": L.bit_length(),
        "seed": seed, "seed_bits": seed.bit_length(),
        "poly": polynomial.encode(data).equation, "tree": tree.equation,
        "tree_bits": tree.info["level_bits"],
    }


def _jpeg_like(rgb: np.ndarray, fmt: str, budget: int):
    """Best quality whose file fits in the equation's byte budget."""
    lo, hi, best = 1, 95, None
    while lo <= hi:
        q = (lo + hi) // 2
        n, rec = baselines.image_lossy(rgb, fmt, q)
        if n <= budget:
            best, lo = (n, rec, q), q + 1
        else:
            hi = q - 1
    if best is None:
        n, rec = baselines.image_lossy(rgb, fmt, 1)
        best = (n, rec, 1)
    n, rec, q = best
    return {"bytes": n, "psnr": round(waves.psnr(rgb, rec), 2), "q": q, "img": b64png(rec)}


def api_image(data: bytes | None, step: float, tile: int) -> dict:
    src = io.BytesIO(data) if data else ROOT / "samples" / "image" / "landscape.png"
    im = Image.open(src).convert("RGB")
    if im.width > 512:
        im = im.resize((512, round(512 * im.height / im.width)), Image.LANCZOS)
    rgb = np.asarray(im)
    payload, terms = waves.encode_image(rgb, step, tile or None)
    rec = waves.decode_image(payload)
    size = len(payload) + 1
    return {
        "raw": rgb.nbytes, "png": baselines.image_lossless(rgb)["PNG"],
        "original": b64png(rgb),
        "equation": {"bytes": size, "terms": terms, "psnr": round(waves.psnr(rgb, rec), 2), "img": b64png(rec)},
        "jpeg": _jpeg_like(rgb, "JPEG", size), "webp": _jpeg_like(rgb, "WEBP", size),
    }


class H(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(200, json.dumps(obj, default=float).encode(), "application/json")

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                self._send(200, (ROOT / "app" / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/loop":
                self._json(api_loop(q.get("text", "hi")))
            elif u.path == "/api/image":
                self._json(api_image(None, float(q.get("step", 16)), int(q.get("tile", 16))))
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:  # surface engine errors to the page
            self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")

    def do_POST(self) -> None:
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            if u.path == "/api/encode":
                self._json(api_encode(body))
            elif u.path == "/api/image":
                self._json(api_image(body, float(q.get("step", 16)), int(q.get("tile", 16))))
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")

    def log_message(self, fmt, *args) -> None:
        sys.stderr.write("[equafile] " + fmt % args + "\n")


if __name__ == "__main__":
    print(f"EquaFile app on http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
