"""Raaj's follow-up #2: waves, superposition, resonance. Writes results/resonance.json."""
from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from equafile import baselines, predictive, resonance as R  # noqa: E402

S = ROOT / "samples"


def wav_bytes(name: str) -> bytes:
    with wave.open(str(S / "audio" / f"{name}.wav")) as w:
        return w.readframes(w.getnframes())


def bank(x: np.ndarray, freqs_hz: np.ndarray, fs: float, r: float = 0.999) -> np.ndarray:
    """A literal bank of driven, damped oscillators: y[n] = 2r·cos(ω)·y[n-1] - r²·y[n-2] + x[n].
    Returns how much energy each one holds after being driven by x."""
    w = 2 * np.pi * freqs_hz / fs
    a1, a2 = 2 * r * np.cos(w), -r * r
    y1 = np.zeros_like(w)
    y2 = np.zeros_like(w)
    energy = np.zeros_like(w)
    for v in x.astype(np.float64):
        y = a1 * y1 + a2 * y2 + v
        y2, y1 = y1, y
        energy += y * y
    return energy / energy.max()


def main() -> None:
    files = {
        "chord.wav (3 notes)": (wav_bytes("chord"), 2, "audio"),
        "system_sounds.wav": (wav_bytes("system_sounds"), 2, "audio"),
        "sensor.csv": ((S / "text" / "sensor.csv").read_bytes(), 1, "data"),
        "counter.bin": ((S / "text" / "counter.bin").read_bytes(), 1, "data"),
        "license.txt": ((S / "text" / "license.txt").read_bytes(), 1, "text"),
        "typing.py": ((S / "text" / "typing.py").read_bytes(), 1, "text"),
        "random.bin": ((S / "text" / "random.bin").read_bytes(), 1, "noise"),
        "beach photo (raw pixels)": (np.asarray(Image.open(S / "image" / "beach_photo.png").convert("RGB")).tobytes(), 1, "image"),
    }
    rows = []
    for name, (data, width, kind) in files.items():
        row = {"file": name, "kind": kind, "size": len(data), "width": width}
        for V in (R.WaveVariant, R.ResonatorVariant):
            t = time.perf_counter()
            e = V.encode(data, width)
            ok = V.decode(e.payload) == data
            row[V.ID] = {"stored": len(e.payload) + 1, "ok": ok, "params": e.info["param_bytes"],
                         "residual": e.info["residual_bytes"], "avg_models": float(np.mean(e.info["models"])),
                         "pieces_using": int(sum(1 for m in e.info["models"] if m)), "pieces": len(e.info["models"]),
                         "sec": round(time.perf_counter() - t, 1)}
            print(f"{name:26} {V.ID:14} {len(data):>8} -> {row[V.ID]['stored']:>8} ok={ok} {row[V.ID]['sec']}s")
        row["v6"] = len(predictive.compress(data)) + 2
        base = baselines.lossless(data)
        if kind == "audio":
            base["FLAC"] = baselines.audio_codec(S / "audio" / (name.split(".")[0] + ".wav"), "flac", None)[0]
        if kind == "image":
            base.update(baselines.image_lossless(np.asarray(Image.open(S / "image" / "beach_photo.png").convert("RGB"))))
        row["best_existing"] = list(min(base.items(), key=lambda kv: kv[1]))
        row["baselines"] = base
        rows.append(row)

    # visuals: resonance bank responses, superposition of the chord, "hi" from waves
    fs = 44100.0
    chord = np.frombuffer(wav_bytes("chord"), dtype="<i2")[:4096]
    hz = np.linspace(200, 900, 351)
    waves, _ = R.fit_waves(chord.astype(np.int64))
    t = np.arange(400)
    sup = {"hz": hz.tolist(), "ring": bank(chord, hz, fs).tolist(),
           "found_hz": [float(w) * fs / (2 * np.pi) for w, _, _ in waves],
           "parts": [(float(a) * np.cos(float(w) * t) + float(b) * np.sin(float(w) * t)).round(1).tolist() for w, a, b in waves],
           "original": chord[:400].astype(int).tolist()}
    lic = np.frombuffer((S / "text" / "license.txt").read_bytes()[:4096], dtype=np.uint8).astype(np.float64)
    lic_ring = R.resonance_spectrum(lic - lic.mean())[: R.PAD * 4096 // 2]
    chord_ring = R.resonance_spectrum(chord.astype(np.float64))[: R.PAD * 4096 // 2]
    rnd = np.frombuffer((S / "text" / "random.bin").read_bytes(), dtype=np.uint8).astype(np.float64)
    rnd_ring = R.resonance_spectrum(rnd - rnd.mean())[: R.PAD * 4096 // 2]
    share = lambda sp, k: float(np.sort(sp)[::-1][:k].sum() / sp.sum())
    conc = {nm: [share(sp, k) for k in (1, 3, 10, 30, 100, 300)] for nm, sp in
            (("chord", chord_ring), ("license.txt", lic_ring), ("random.bin", rnd_ring))}
    hi = np.array([int(b) for b in "0110100001101001"], dtype=np.float64)
    F = np.fft.rfft(hi)
    order = np.argsort(-np.abs(F))
    hi_rows = []
    for k in range(1, len(F) + 1):
        G = np.zeros_like(F)
        G[order[:k]] = F[order[:k]]
        rec = np.fft.irfft(G, len(hi))
        hi_rows.append({"waves": k, "bits_wrong": int(np.sum((rec > 0.5) != (hi > 0.5))), "curve": rec.round(3).tolist()})
    out = {"rows": rows, "superposition": sup, "concentration": conc, "hi": {"bits": hi.astype(int).tolist(), "rows": hi_rows}}
    (ROOT / "results" / "resonance.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote results/resonance.json")


if __name__ == "__main__":
    main()
