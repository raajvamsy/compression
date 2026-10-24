"""Existing codecs we compare against (sizes are full standard file formats)."""
from __future__ import annotations

import bz2
import gzip
import io
import lzma
import subprocess
import tempfile
from pathlib import Path

import brotli
import numpy as np
import zstandard
from PIL import Image


def lossless(data: bytes) -> dict[str, int]:
    return {
        "gzip -9": len(gzip.compress(data, 9, mtime=0)),
        "bzip2 -9": len(bz2.compress(data, 9)),
        "xz -9e": len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)),
        "zstd -19": len(zstandard.ZstdCompressor(level=19).compress(data)),
        "brotli -11": len(brotli.compress(data, quality=11)),
    }


# ------------------------------------------------------------------ image
def _img_bytes(rgb: np.ndarray, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, fmt, **kw)
    return buf.getvalue()


def image_lossless(rgb: np.ndarray) -> dict[str, int]:
    return {
        "PNG": len(_img_bytes(rgb, "PNG", optimize=True)),
        "WebP lossless": len(_img_bytes(rgb, "WEBP", lossless=True, quality=100, method=6)),
    }


def image_lossy(rgb: np.ndarray, fmt: str, quality: int) -> tuple[int, np.ndarray]:
    kw = {"quality": quality}
    if fmt == "WEBP":
        kw["method"] = 6
    if fmt == "JPEG":
        kw["optimize"] = True
    data = _img_bytes(rgb, fmt, **kw)
    return len(data), np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


# ------------------------------------------------------------------ audio / video via ffmpeg
def _run(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def audio_codec(wav: Path, codec: str, kbps: int | None) -> tuple[int, np.ndarray]:
    """Encode with ffmpeg, return (size, decoded int16 mono)."""
    ext = {"libmp3lame": "mp3", "libopus": "opus", "flac": "flac"}[codec]
    with tempfile.TemporaryDirectory() as td:
        enc, dec = Path(td) / f"a.{ext}", Path(td) / "d.raw"
        args = ["-i", str(wav), "-c:a", codec] + (["-compression_level", "12"] if codec == "flac" else [])
        if kbps:
            args += ["-b:a", f"{kbps}k"]
        _run(*args, str(enc))
        _run("-i", str(enc), "-f", "s16le", "-ac", "1", "-ar", "44100", str(dec))
        return enc.stat().st_size, np.frombuffer(dec.read_bytes(), dtype="<i2")


def align(ref: np.ndarray, test: np.ndarray, max_lag: int = 4096) -> np.ndarray:
    """Undo codec delay (MP3/Opus add padding) by best cross-correlation lag."""
    n = min(len(ref), 44100)
    best, best_lag = -np.inf, 0
    r = ref[:n].astype(np.float64)
    for lag in range(0, max_lag, 1):
        t = test[lag:lag + n].astype(np.float64)
        if len(t) < n:
            break
        c = float(np.dot(r, t))
        if c > best:
            best, best_lag = c, lag
    out = test[best_lag:best_lag + len(ref)]
    return np.pad(out, (0, len(ref) - len(out)))


def video_x264(frames: np.ndarray, fps: int, crf: int) -> tuple[int, np.ndarray]:
    t, h, w, _ = frames.shape
    with tempfile.TemporaryDirectory() as td:
        raw, mp4, dec = Path(td) / "in.rgb", Path(td) / "o.mp4", Path(td) / "d.rgb"
        raw.write_bytes(frames.tobytes())
        pix = "yuv444p" if crf == 0 else "yuv420p"
        _run("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", str(raw),
             "-c:v", "libx264", "-preset", "veryslow", "-crf", str(crf), "-pix_fmt", pix, str(mp4))
        _run("-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", str(dec))
        out = np.frombuffer(dec.read_bytes(), dtype=np.uint8).reshape(-1, h, w, 3)[:t]
        return mp4.stat().st_size, out
