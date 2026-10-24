"""Files whose structure is 'repeats of repeats' or 'repeats with variation' (built locally, deterministic)."""
from __future__ import annotations

import random
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples" / "repeat"


def read_aiff(name: str) -> np.ndarray:
    import subprocess
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"/System/Library/Sounds/{name}.aiff", "-ac", "1", "-ar", "44100",
                          "-f", "s16le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float64)


def write_wav(p: Path, x: np.ndarray) -> None:
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(44100)
        w.writeframes(np.clip(np.round(x), -32768, 32767).astype("<i2").tobytes())


def music_loop() -> None:
    """One bar built from system sounds, played 8 times: louder/softer each time, plus a little noise."""
    bar = np.zeros(int(44100 * 0.75))
    for name, at, g in (("Funk", 0.0, 0.8), ("Tink", 0.1875, 0.5), ("Pop", 0.375, 0.7), ("Tink", 0.5625, 0.5)):
        s = read_aiff(name)[: len(bar)] * g
        i = int(at * 44100)
        bar[i:i + len(s)] += s[: len(bar) - i]
    rnd = np.random.default_rng(3)
    gains = [1.0, 0.8, 0.95, 0.7, 1.0, 0.85, 0.9, 0.75]
    song = np.concatenate([bar * g + rnd.normal(0, 3, len(bar)) for g in gains])
    write_wav(OUT / "music_loop.wav", song)
    # the same loop made digitally (copy-pasted bars, volume automation, no recording noise)
    write_wav(OUT / "music_loop_clean.wav", np.concatenate([bar * g for g in gains]))


def stereo() -> None:
    """Stereo mix: the same sounds in both channels, panned and slightly delayed, plus one sound only on the right."""
    seq = np.concatenate([read_aiff(n) for n in ("Glass", "Hero", "Submarine", "Funk", "Purr")])
    extra = np.zeros_like(seq)
    t = read_aiff("Tink")
    for at in range(0, len(seq) - len(t), 22050):
        extra[at:at + len(t)] += 0.4 * t
    left = 0.9 * seq
    right = 0.6 * np.concatenate([np.zeros(3), seq[:-3]]) + extra
    inter = np.empty(2 * len(seq))
    inter[0::2], inter[1::2] = left, right
    with wave.open(str(OUT / "stereo_mix.wav"), "wb") as w:
        w.setnchannels(2), w.setsampwidth(2), w.setframerate(44100)
        w.writeframes(np.clip(np.round(inter), -32768, 32767).astype("<i2").tobytes())


def versions() -> None:
    """Three versions of a document with small edits, like a revision history."""
    base = (ROOT / "samples" / "text" / "license.txt").read_text()
    v2 = base.replace("Python", "PyLang").replace("2001", "2026")
    v3 = v2.replace("Software", "Code").replace("\n\n", "\n\n# revised\n", 5)
    (OUT / "doc_versions.txt").write_text(base + "\n=====\n" + v2 + "\n=====\n" + v3)


def server_log() -> None:
    """A web server log: every line has the same shape, fields change."""
    rnd = random.Random(11)
    ips = [f"10.0.{rnd.randint(0, 9)}.{rnd.randint(1, 254)}" for _ in range(40)]
    paths = ["/api/users", "/api/orders", "/static/app.js", "/static/app.css", "/login", "/api/items?page=2", "/health"]
    t = 1_760_000_000
    lines = []
    for _ in range(25000):
        t += rnd.choice([0, 0, 1, 1, 2])
        code = rnd.choices([200, 200, 200, 304, 404, 500], k=1)[0]
        lines.append(f'{ips[rnd.randrange(40)]} - - [{t}] "GET {rnd.choice(paths)} HTTP/1.1" {code} '
                     f'{rnd.randint(200, 9000)} {rnd.randint(1, 250)}ms')
    (OUT / "server.log").write_text("\n".join(lines) + "\n")


def static_video() -> None:
    """Static camera: 24 frames of the same scene with a small moving square and sensor noise."""
    from PIL import Image
    w, h = 256, 144
    scene = np.asarray(Image.open(ROOT / "samples" / "image" / "landscape.png").convert("RGB").resize((w, h)), dtype=np.int16)
    rnd = np.random.default_rng(5)
    frames = []
    for f in range(24):
        fr = scene.copy()
        x = 20 + 8 * f
        fr[60:80, x:x + 20] = (240, 40, 40)
        fr = fr + rnd.integers(-1, 2, fr.shape)
        frames.append(np.clip(fr, 0, 255).astype(np.uint8))
    (OUT / "static_camera.rgb").write_bytes(np.stack(frames).tobytes())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (music_loop, stereo, versions, server_log, static_video):
        f()
    for p in sorted(OUT.iterdir()):
        print(p.name, p.stat().st_size)
