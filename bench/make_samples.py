"""Build the sample set in samples/ from files already on this Mac (no downloads)."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "samples"
PYLIB = Path(os.__file__).parent
PICS = Path("/System/Library/Desktop Pictures")


def sh(*args: str) -> None:
    subprocess.run(args, check=True, capture_output=True)


def text_samples() -> None:
    (S / "text").mkdir(parents=True, exist_ok=True)
    w = lambda name, data: (S / "text" / name).write_bytes(data)
    w("hi.txt", b"hi")
    w("hello.txt", b"hello world")
    w("sentence.txt", b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps again.")
    w("license.txt", (PYLIB / "LICENSE.txt").read_bytes())
    w("json_decoder.py", (PYLIB / "json" / "decoder.py").read_bytes())
    w("typing.py", (PYLIB / "typing.py").read_bytes())
    rnd = random.Random(7)
    rows = ["time,sensor,temp_c,humidity"]
    for t in range(2000):
        rows.append(f"{t},S{t % 4},{20 + 5 * np.sin(t / 50):.2f},{rnd.randint(40, 60)}")
    w("sensor.csv", ("\n".join(rows) + "\n").encode())
    w("counter.bin", bytes(range(256)) * 16)
    w("random.bin", random.Random(1).randbytes(4096))


def image_samples() -> None:
    d = S / "image"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "_tmp.png"
    sources = {
        "landscape": PICS / "Sonoma.heic",
        "wallpaper": Path("/System/Library/CoreServices/DefaultBackground.jpg"),
        "beach_photo": PICS / ".thumbnails" / "The Beach.heic",
    }
    for name, src in sources.items():
        sh("sips", "-s", "format", "png", str(src), "--out", str(tmp))
        im = Image.open(tmp).convert("RGB")
        if im.width > 512:
            im = im.resize((512, round(512 * im.height / im.width)), Image.LANCZOS)
        im.save(d / f"{name}.png")
    tmp.unlink()
    # synthetic: smooth gradient + hard shapes + text-like noise
    y, x = np.mgrid[0:256, 0:256]
    img = np.stack([x, y, 255 - (x + y) // 2], -1).astype(np.uint8)
    img[60:120, 40:200] = (230, 40, 40)
    img[150:230, 100:140] = (20, 20, 20)
    Image.fromarray(img).save(d / "synthetic.png")


def audio_samples() -> None:
    d = S / "audio"
    d.mkdir(parents=True, exist_ok=True)
    names = ["Glass", "Hero", "Submarine", "Funk", "Purr", "Sosumi", "Ping", "Blow"]
    lst = d / "_list.txt"
    lst.write_text("".join(f"file '/System/Library/Sounds/{n}.aiff'\n" for n in names))
    sh("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-ac", "1", "-ar", "44100",
       "-c:a", "pcm_s16le", str(d / "system_sounds.wav"))
    lst.unlink()
    # a pure chord: a file that really *is* a short equation
    t = np.arange(44100 * 3) / 44100
    tone = 0.3 * (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 554.37 * t) + np.sin(2 * np.pi * 659.25 * t))
    pcm = (tone / 3 * 32767).astype("<i2").tobytes()
    import wave
    with wave.open(str(d / "chord.wav"), "wb") as wv:
        wv.setnchannels(1), wv.setsampwidth(2), wv.setframerate(44100), wv.writeframes(pcm)


def video_samples() -> None:
    d = S / "video"
    d.mkdir(parents=True, exist_ok=True)
    w, h, n = 256, 144, 48
    clips = {
        "pan_landscape": ["-loop", "1", "-i", str(S / "image" / "landscape.png"), "-vf",
                          f"zoompan=z='1.0+0.004*on':x='iw/2-(iw/zoom/2)+on*2':y='ih/2-(ih/zoom/2)':d={n}:s={w}x{h}:fps=24"],
        "testsrc": ["-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=24"],
    }
    for name, args in clips.items():
        raw = d / f"{name}.rgb"
        sh("ffmpeg", "-y", *args, "-frames:v", str(n), "-pix_fmt", "rgb24", "-f", "rawvideo", str(raw))
        (d / f"{name}.json").write_text(json.dumps({"w": w, "h": h, "frames": n, "fps": 24}))


if __name__ == "__main__":
    which = sys.argv[1:] or ["text", "image", "audio", "video"]
    for k in which:
        globals()[f"{k}_samples"]()
        print("built", k)
