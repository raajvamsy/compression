"""FunnelZip vs everything: sizes, speed, round trips, and an ablation (each lens off)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import zstandard
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from equafile import baselines, codefilter, funnelzip as F  # noqa: E402

S = ROOT / "samples"
X = S / "extra"


def extra_samples() -> None:
    X.mkdir(exist_ok=True)
    if not (X / "stdlib_2mb.txt").exists():
        lib = Path(os.__file__).parent
        buf = bytearray()
        for p in sorted(lib.glob("*.py")):
            buf += p.read_bytes()
            if len(buf) >= 2 << 20:
                break
        (X / "stdlib_2mb.txt").write_bytes(bytes(buf[: 2 << 20]))
    if not (X / "zsh_binary").exists():
        (X / "zsh_binary").write_bytes(Path("/bin/zsh").read_bytes())


def raw_rgb(name: str) -> bytes:
    return np.asarray(Image.open(S / "image" / f"{name}.png").convert("RGB")).tobytes()


def video_lossless(raw: bytes, w: int, h: int) -> dict:
    out = {}
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "v.rgb"
        src.write_bytes(raw)
        for name, args, ext in (("FFV1", ["-c:v", "ffv1", "-level", "3"], "mkv"),
                                ("x264 lossless", ["-c:v", "libx264rgb", "-qp", "0", "-preset", "veryslow"], "mkv")):
            dst = Path(td) / f"o.{ext}"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                            "-r", "24", "-i", str(src), *args, str(dst)], check=True)
            out[name] = dst.stat().st_size
    return out


def files() -> list[tuple[str, str, bytes, dict]]:
    extra_samples()
    rows = []
    for n in ("hi.txt", "sentence.txt", "license.txt", "json_decoder.py", "typing.py", "sensor.csv", "counter.bin", "random.bin"):
        rows.append((n, "text/data", (S / "text" / n).read_bytes(), {}))
    rows.append(("stdlib_2mb.txt (Python source)", "text/data", (X / "stdlib_2mb.txt").read_bytes(), {}))
    zsh = (X / "zsh_binary").read_bytes()
    xz_bcj = len(subprocess.run(["xz", "-c", "--x86", "--arm64", "--lzma2=preset=9e"], input=zsh, capture_output=True, check=True).stdout)
    rows.append(("zsh (program binary)", "binary", zsh, {"xz -9e + BCJ x86/arm64": xz_bcj}))
    for n in ("chord", "system_sounds"):
        p = S / "audio" / f"{n}.wav"
        rows.append((f"{n}.wav", "audio", p.read_bytes(), {"FLAC -12": baselines.audio_codec(p, "flac", None)[0]}))
    for n in ("beach_photo", "landscape", "wallpaper", "synthetic"):
        rgb = np.asarray(Image.open(S / "image" / f"{n}.png").convert("RGB"))
        rows.append((f"{n} (raw RGB)", "image", rgb.tobytes(), baselines.image_lossless(rgb)))
    R_ = S / "repeat"
    for n in ("doc_versions.txt", "server.log"):
        rows.append((n, "repeats", (R_ / n).read_bytes(), {}))
    for n in ("music_loop.wav", "music_loop_clean.wav"):
        rows.append((n, "repeats", (R_ / n).read_bytes(), {"FLAC -12": baselines.audio_codec(R_ / n, "flac", None)[0]}))
    rows.append(("stereo_mix.wav", "audio", (R_ / "stereo_mix.wav").read_bytes(),
                 {"FLAC -12": baselines.audio_codec(R_ / "stereo_mix.wav", "flac", None)[0]}))
    for n in ("landscape.png", "beach_photo.png", "wallpaper.png", "synthetic.png"):
        rows.append((f"{n} (PNG file)", "image file", (S / "image" / n).read_bytes(), {"the PNG itself": (S / "image" / n).stat().st_size}))
    for n, path in (("landscape_q90.jpg", X / "landscape_q90.jpg"), ("landscape_q75.jpg", X / "landscape_q75.jpg"),
                    ("sonoma_apple.jpg", X / "sonoma_apple.jpg"), ("beach_apple.jpg", X / "beach_apple.jpg"),
                    ("macOS DefaultBackground.jpg (4K)", Path("/System/Library/CoreServices/DefaultBackground.jpg"))):
        blob = path.read_bytes()
        rows.append((f"{n} (JPEG file)", "image file", blob, {"the JPEG itself": len(blob)}))
    cam = (R_ / "static_camera.rgb").read_bytes()
    rows.append(("static camera video (raw RGB, 24 frames)", "repeats", cam, video_lossless(cam, 256, 144)))
    meta = json.loads((S / "video" / "pan_landscape.json").read_text())
    raw = (S / "video" / "pan_landscape.rgb").read_bytes()
    rows.append(("camera pan video (raw RGB, 48 frames)", "video", raw, video_lossless(raw, meta["w"], meta["h"])))
    return rows


def main() -> None:
    results = []
    for name, kind, data, extra in files():
        t = time.perf_counter()
        enc = F.encode(data)
        te = time.perf_counter() - t
        t = time.perf_counter()
        ok = hashlib.sha256(F.decode(enc.payload)).digest() == hashlib.sha256(data).digest()
        td = time.perf_counter() - t
        base = baselines.lossless(data)
        base["zstd -22"] = len(zstandard.ZstdCompressor(level=22).compress(data))
        base.update(extra)
        bname, bsize = min(base.items(), key=lambda kv: kv[1])
        fz = len(enc.payload) + 1
        results.append({"file": name, "kind": kind, "size": len(data), "funnelzip": fz, "ok": ok,
                        "enc_s": round(te, 2), "dec_s": round(td, 2),
                        "params": [enc.info.get(k) for k in ("W", "C", "S", "R", "F", "P", "mask")], "path": enc.info.get("path"),
                        "baselines": base, "best_existing": [bname, bsize]})
        print(f"{name:40} {len(data):>9} fz={fz:>9} best={bname}:{bsize:>9} {'OK' if ok else 'MISMATCH'} "
              f"{(fz / bsize - 1) * 100:+6.1f}%  enc {te:.1f}s dec {td:.1f}s  {enc.equation.splitlines()[0][:70]}")

    # ablation: same listen choice, one lens off at a time
    pick = {"license.txt", "typing.py", "sensor.csv", "counter.bin", "system_sounds.wav", "beach_photo (raw RGB)",
            "zsh (program binary)", "music_loop_clean.wav", "server.log", "static camera video (raw RGB, 24 frames)",
            "stereo_mix.wav", "camera pan video (raw RGB, 48 frames)", "synthetic (raw RGB)"}
    abl = {}
    for name, kind, data, _ in files():
        if name not in pick:
            continue
        r = next(r for r in results if r["file"] == name)
        W, C, S_, R, Fl, P, mask = r["params"]
        src = codefilter.apply(data, Fl, True) if Fl else data
        full = len(F.raw_compress(src, W, C, S_, mask, R, bool(P)))
        row = {"all": full}
        for lens, bit in F.LENSES.items():
            row[f"-{lens}"] = len(F.raw_compress(src, W, C, S_, mask & ~bit, R, bool(P))) if mask & bit else full
        if P:
            row["-primer"] = len(F.raw_compress(src, W, C, S_, mask, R, False))
        if Fl:
            row["-codefilter"] = len(F.raw_compress(data, W, C, S_, mask, R, bool(P)))
        row["memory only"] = len(F.raw_compress(data, W, C, S_, F.LENSES["memory"], 0))
        abl[name] = row
        print("ablation", name, row)
    (ROOT / "results" / "funnelzip.json").write_text(json.dumps({"rows": results, "ablation": abl}, indent=1))
    print("wrote results/funnelzip.json")


if __name__ == "__main__":
    main()
