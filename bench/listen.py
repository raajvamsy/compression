"""Write matched-size audio and image samples to outputs/listen/ so quality can be judged by ear and eye."""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from equafile import baselines, waves  # noqa: E402

OUT = ROOT / "outputs" / "listen"
OUT.mkdir(parents=True, exist_ok=True)


def write_wav(p: Path, x: np.ndarray) -> None:
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(44100), w.writeframes(x.astype("<i2").tobytes())


def closest_step(x, budget):
    """Bisect the quantiser step (log scale) for the largest equation that fits the budget."""
    lo, hi, best = np.log(1.0), np.log(4096.0), None
    for _ in range(12):
        st = float(np.exp((lo + hi) / 2))
        payload, _ = waves.encode_audio(x, st, entropy=True)
        if len(payload) <= budget:
            best, hi = (st, len(payload), payload), np.log(st)
        else:
            lo = np.log(st)
    return best


for name in ("system_sounds", "chord"):
    src = ROOT / "samples" / "audio" / f"{name}.wav"
    with wave.open(str(src)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    for kb in (16, 32):
        size, dec = baselines.audio_codec(src, "libopus", kb)
        write_wav(OUT / f"{name}_opus{kb}k_{size}B.wav", baselines.align(x, dec))
        st, n, payload = closest_step(x, size)
        write_wav(OUT / f"{name}_equation_{n}B.wav", waves.decode_audio(payload))
        print(f"{name}: opus {kb}k {size} B vs equation {n} B (step {st:.1f})")
