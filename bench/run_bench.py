"""Run every experiment and write results/<section>.json.

usage: python bench/run_bench.py [lossless loop tree image audio video]
"""
from __future__ import annotations

import json
import random
import sys
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equafile import baselines, loop, waves  # noqa: E402
from equafile.core import registry, run  # noqa: E402

S, R = ROOT / "samples", ROOT / "results"
R.mkdir(exist_ok=True)


def save(name: str, obj) -> None:
    (R / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    print(f"wrote results/{name}.json")


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def read_video(name: str) -> tuple[np.ndarray, int]:
    meta = json.loads((S / "video" / f"{name}.json").read_text())
    raw = np.frombuffer((S / "video" / f"{name}.rgb").read_bytes(), dtype=np.uint8)
    return raw.reshape(meta["frames"], meta["h"], meta["w"], 3), meta["fps"]


# ------------------------------------------------------------------ 1. lossless table
def lossless() -> None:
    files = {p.name: p.read_bytes() for p in sorted((S / "text").iterdir())}
    for p in sorted((S / "image").glob("*.png")):
        if p.stem in ("beach_photo", "synthetic"):
            files[f"{p.stem}.rgb (raw pixels)"] = np.asarray(Image.open(p).convert("RGB")).tobytes()
    files["chord.wav (raw PCM)"] = read_wav(S / "audio" / "chord.wav").tobytes()
    rows = []
    for name, data in files.items():
        entry = {"file": name, "size": len(data), "baselines": baselines.lossless(data), "variants": []}
        if name.endswith("(raw pixels)"):
            rgb = np.asarray(Image.open(S / "image" / name.replace(".rgb (raw pixels)", ".png")).convert("RGB"))
            entry["baselines"].update(baselines.image_lossless(rgb))
        for v in registry().values():
            r = run(v, data)
            r.pop("info", None)
            if r.get("equation") and len(r["equation"]) > 1500:
                r["equation"] = r["equation"][:1500] + " …"
            entry["variants"].append(r)
            print(f"{name:28} {v.ID:12} {r['status']:10} {r.get('stored', '-')}")
        rows.append(entry)
    save("lossless", rows)


# ------------------------------------------------------------------ 2. loop / seed search scaling
def loop_scaling() -> None:
    rnd = random.Random(42)
    out = {"v2": [], "v3": []}
    for n in range(4, 23, 2):
        for _ in range(5):
            target = rnd.getrandbits(n)
            t = time.perf_counter()
            L, _ = loop.loop_search(target, n)
            out["v2"].append({"bits": n, "stored_bits": L.bit_length(), "steps": L, "sec": time.perf_counter() - t})
        print("v2", n, [r["stored_bits"] for r in out["v2"][-5:]])
    for n in range(4, 29, 4):
        for _ in range(5 if n < 28 else 2):
            target = rnd.getrandbits(n)
            t = time.perf_counter()
            s = loop.seed_search(target, n)
            out["v3"].append({"bits": n, "stored_bits": s.bit_length(), "seed": s, "sec": time.perf_counter() - t})
        print("v3", n, [r["stored_bits"] for r in out["v3"][-5:]])
    save("loop", out)


# ------------------------------------------------------------------ 3. FunnelBrain tree depth
def tree() -> None:
    out = {}
    for name in ("hi.txt", "sentence.txt", "license.txt", "random.bin"):
        data = (S / "text" / name).read_bytes()
        _, sizes, _ = loop.TreeVariant.build(data, 5)
        out[name] = sizes
        print(name, sizes)
    save("tree", out)


# ------------------------------------------------------------------ 4. images
STEPS = [3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
QUALS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]


def image() -> None:
    out = {}
    for p in sorted((S / "image").glob("*.png")):
        rgb = np.asarray(Image.open(p).convert("RGB"))
        e = {"shape": rgb.shape, "raw": rgb.nbytes, "lossless": baselines.image_lossless(rgb), "curves": {}}
        for label, tile in (("Equation: 16x16 sub-equations", 16), ("Equation: one global equation", None)):
            pts = []
            for st in STEPS:
                payload, terms = waves.encode_image(rgb, st, tile)
                pts.append({"bytes": len(payload) + 1, "psnr": waves.psnr(rgb, waves.decode_image(payload)),
                            "terms": terms, "step": st})
            e["curves"][label] = pts
        for fmt, label in (("JPEG", "JPEG"), ("WEBP", "WebP")):
            pts = []
            for q in QUALS:
                n, rec = baselines.image_lossy(rgb, fmt, q)
                pts.append({"bytes": n, "psnr": waves.psnr(rgb, rec), "q": q})
            e["curves"][label] = pts
        out[p.stem] = e
        print("image", p.stem)
    save("image", out)


# ------------------------------------------------------------------ 5. audio
def audio() -> None:
    out = {}
    for p in sorted((S / "audio").glob("*.wav")):
        x = read_wav(p)
        e = {"seconds": len(x) / 44100, "raw": x.nbytes, "curves": {}}
        e["flac"] = baselines.audio_codec(p, "flac", None)[0]
        pts = []
        for st in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
            payload, terms = waves.encode_audio(x, st)
            pts.append({"bytes": len(payload) + 1, "snr": waves.snr(x, waves.decode_audio(payload)),
                        "terms": terms, "step": st})
        e["curves"]["Equation: cosine terms / frame"] = pts
        for codec, label, rates in (("libmp3lame", "MP3", [32, 48, 64, 96, 128, 192]),
                                    ("libopus", "Opus", [12, 16, 24, 32, 48, 64, 96])):
            pts = []
            for kb in rates:
                n, dec = baselines.audio_codec(p, codec, kb)
                pts.append({"bytes": n, "snr": waves.snr(x, baselines.align(x, dec)), "kbps": kb})
            e["curves"][label] = pts
        out[p.stem] = e
        print("audio", p.stem)
    save("audio", out)


# ------------------------------------------------------------------ 6. video
def video() -> None:
    out = {}
    for name in ("pan_landscape", "testsrc"):
        frames, fps = read_video(name)
        e = {"shape": frames.shape, "raw": frames.nbytes, "curves": {}}
        pts = []
        for st in [6, 12, 24, 48, 96]:
            t = time.perf_counter()
            payload, terms = waves.encode_video(frames, st)
            rec = waves.decode_video(payload)
            pts.append({"bytes": len(payload) + 1, "psnr": waves.psnr(frames, rec), "terms": terms,
                        "step": st, "sec": time.perf_counter() - t})
            print("  eq", st, pts[-1]["bytes"], round(pts[-1]["psnr"], 2))
        e["curves"]["Equation: 3-D cosine per 8x16x16 block"] = pts
        pts = []
        for crf in [12, 18, 23, 28, 33, 38, 45, 51]:
            n, rec = baselines.video_x264(frames, fps, crf)
            pts.append({"bytes": n, "psnr": waves.psnr(frames, rec), "crf": crf})
        e["curves"]["H.264 (x264 veryslow)"] = pts
        out[name] = e
        print("video", name)
    save("video", out)


SECTIONS = {"lossless": lossless, "loop": loop_scaling, "tree": tree,
            "image": image, "audio": audio, "video": video}

if __name__ == "__main__":
    for k in sys.argv[1:] or SECTIONS:
        t = time.perf_counter()
        SECTIONS[k]()
        print(f"== {k} done in {time.perf_counter() - t:.1f}s")
