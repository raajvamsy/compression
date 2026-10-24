"""Speed round: the v4 C core vs today's, same files and layouts, one core each, no Listen.
Writes results/speed_round.json (read by make_funnelzip_visual.py)."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
X = ROOT / "samples" / "extra"
# file, label, W C S R, lens set in v4 (ALL / TEXT), lens set Listen picks now
JOBS = [
    (X / "text_1mb.txt", "1 MB text", "1 1 0 0", 5575, 1479),
    (ROOT / "samples/audio/system_sounds.wav", "1 MB sounds", "2 1 0 682", 8191, 3581),
    (X / "landscape.rgb", "photo pixels", "1 3 1536 3072", 8191, 8189),
    (ROOT / "samples/text/sensor.csv", "sensor CSV", "1 1 0 136", 5575, 5575),
    (X / "zsh_binary", "program binary", "1 1 16 0", 8191, 8189),
]
# What each step did, as measured during the round (sizes in bytes; speeds in MB/s of the original file).
# rows: [name, before, after, best existing or None]; speed: [name, before, after]
STEPS = [
    {"key": "huffman", "title": "JPEG Huffman layer in C", "verdict": "worked",
     "idea": "Undoing and redoing a JPEG's Huffman codes was pure Python, about 6 s each way on the 4K photo. Now it's C. "
             "It writes the identical scan bits, including restart markers and 0xFF byte stuffing, and was checked "
             "against the Python version on all 5 JPEGs (the 4K one has 480 restart markers).",
     "speed": [["4K photo: Huffman decode", 5.62 / 6.0, 5.62 / 0.093], ["4K photo: Huffman encode", 5.62 / 6.0, 5.62 / 0.136]]},
    {"key": "engine", "title": "Engine: fix what the profile shows", "verdict": "worked",
     "idea": "A fast exp() (polynomial + exponent bits), a counter learning rate from a table instead of a division, "
             "cached 'surprise' intervals, one shared input vector for a 4-lane mixer, no third mixer when no signal "
             "lens runs, and a cheaper LPC refit for pixels. The first counter table rounded small negative steps the "
             "wrong way. On JPEG planes (94% zeros) that cost +1.7%, and the JPEG test caught it. With exact rounding "
             "the planes are smaller than before.",
     "rows": [["Sonoma brightness planes: first fast counters", 35389, 35992, None],
              ["… rounding fixed (v4 was 35,389)", 35992, 35357, None]]},
    {"key": "lenses", "title": "Listen drops lenses a file can't use", "verdict": "worked",
     "idea": "Every lens costs time on every bit, even one that finds nothing (word lens on sound, code lens on prose), "
             "and its noise makes the mix slightly worse. Listen now tries a few lens sets in the same parallel batch "
             "as its rhythm trials and keeps the smallest, so the choice adds no extra wait. Files got smaller "
             "and faster.",
     "rows": [["1 MB sounds (−word −code −motion)", 78681, 78153, ["FLAC -12", 139535]],
              ["zsh binary (−word)", 485958, 484208, ["xz -9e + BCJ", 486188]],
              ["1 MB text (−code)", 177820, 177171, None],
              ["photo pixels (−word)", 94270, 94034, ["WebP lossless", 119562]]]},
    {"key": "jpegplanes", "title": "JPEG: each frequency group gets its own model, all in parallel", "verdict": "worked",
     "idea": "The 4K photo's 50 MB of coefficients ran on 3 threads. Now the DC values stay 16-bit and the AC values "
             "are stored one byte each (value + 128; only 0.08% need an escape). The 63 AC planes are split into "
             "groups of about 1 MB, each coded by its own model on its own core. Low and high frequencies behave "
             "differently, so separate models also compress better: smaller and 10× faster.",
     "rows": [["4K photo (5.6 MB)", 4846803, 4508904, ["bzip2 -9", 5441742]],
              ["Sonoma wallpaper", 45820, 43604, ["xz -9e", 67880]],
              ["landscape q90", 17574, 16931, ["brotli -11", 23509]],
              ["landscape q75", 10037, 9559, ["zstd -19", 14133]]],
     "speed": [["4K photo: compress", 5.62 / 56.0, 5.62 / 5.5], ["4K photo: decompress", 5.62 / 49.5, 5.62 / 5.3]]},
    {"key": "dropped", "title": "Tried and dropped (ratio comes first)", "verdict": "didnt",
     "idea": "Each of these was faster but cost ratio, or gave nothing:",
     "notes": ["Smaller context tables: text +1.9% at half size for ~5% speed. Kept full size for files over 16 KB",
               "64 KB Listen trials instead of 128 KB: 16% faster overall, but zsh +3.05% and landscape +0.78% (wrong picks)",
               "Compiler tuning (-mcpu=native, LTO): no gain",
               "Fewer byte-history contexts for 16-bit audio: ±0.4% size, +3% speed"]},
]
LINE = re.compile(r"-> (\d+)\s+compress ([\d.]+) MB/s\s+decompress ([\d.]+) MB/s\s+(\w+)")


def build(src: Path, out: Path) -> None:
    subprocess.run(["clang", "-O3", "-o", str(out), str(ROOT / "bench" / "speed_harness.c"), "-x", "c", str(src), "-lm"],
                   check=True)


def run(exe: Path, f: Path, geo: str, mask: int) -> dict:
    best = None
    for _ in range(2):   # best of 2: other processes can steal a core
        out = subprocess.run([str(exe), str(f), *geo.split(), str(mask), "1"], capture_output=True, text=True).stdout
        m = LINE.search(out)
        r = {"size": int(m[1]), "enc": float(m[2]), "dec": float(m[3]), "ok": m[4] == "ok"}
        best = r if best is None or r["enc"] > best["enc"] else best
    return best


def main() -> None:
    import sys
    if "--steps-only" in sys.argv:   # refresh the step cards, keep the measured core speeds
        f = ROOT / "results" / "speed_round.json"
        d = json.loads(f.read_text())
        d["steps"] = STEPS
        f.write_text(json.dumps(d, indent=1))
        return
    tmp = Path(tempfile.mkdtemp())
    old, new = tmp / "v4", tmp / "now"
    build(ROOT / "funnelzip" / "funnelzip_v4.c.bak", old)
    build(ROOT / "funnelzip" / "funnelzip.c", new)
    rows = []
    for f, label, geo, m4, mnow in JOBS:
        a, b = run(old, f, geo, m4), run(new, f, geo, mnow)
        rows.append({"file": label, "size": f.stat().st_size, "v4": a, "now": b})
        print(f"{label:16} v4 {a['size']:>8} {a['enc']:.2f}/{a['dec']:.2f} MB/s   now {b['size']:>8} {b['enc']:.2f}/{b['dec']:.2f} MB/s")
    (ROOT / "results" / "speed_round.json").write_text(json.dumps({"core": rows, "steps": STEPS}, indent=1))


if __name__ == "__main__":
    main()
