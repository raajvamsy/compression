"""results/funnelzip.json + recorded lens traces -> outputs/visuals/funnelzip.html"""
from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from equafile import codefilter, funnelzip as F  # noqa: E402

MAXIN = 32
GROUPS = ["Memory", "Word", "Echo", "Equation", "Resonator", "Grid", "Rhythm", "Motion", "Code"]


def layout(mask: int, S: int) -> tuple[list[list[int]], int]:
    """Funnel input order in funnelzip.c for a given lens mask (contexts first, then direct/surprise inputs, bias)."""
    L = F.LENSES
    g = {name: [] for name in GROUPS}
    k = 0
    def take(name, n):
        nonlocal k
        g[name] += list(range(k, k + n))
        k += n
    for name, bit, n in (("Memory", "memory", 6), ("Word", "word", 2), ("Echo", "echo", 1), ("Equation", "equation", 2),
                         ("Resonator", "resonator", 1)):
        if mask & L[bit]:
            take(name, n)
    if mask & L["grid"] and S > 0:
        take("Grid", 3)
    for name, bit in (("Rhythm", "rhythm"), ("Code", "code"), ("Motion", "motion")):
        if mask & L[bit]:
            take(name, 2)
    for name, bit in (("Echo", "echo"), ("Resonator", "resonator"), ("Rhythm", "rhythm"), ("Motion", "motion"),
                      ("Equation", "equation"), ("Resonator", "resonator"), ("Grid", "grid"), ("Rhythm", "rhythm")):
        if mask & L[bit]:
            take(name, 1)
    if mask & L["motion"]:
        take("Motion", 2)
    return [g[n] for n in GROUPS], k + 1


def trace(data: bytes, params: list, start: int, nbytes: int, label: str, title: str) -> dict:
    W, C, S, R, Fl, P, mask = params
    lib = F.lib()
    lib.fz_trace.restype = ctypes.c_int
    lib.fz_trace.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    src = codefilter.apply(data, Fl, True) if Fl else data
    pre = F.PRIME if P else b""
    buf_in = pre + src[: start + nbytes]
    off = len(pre) + start
    nbits = (off + nbytes) * 8
    buf = (ctypes.c_float * (nbits * (MAXIN + 2)))()
    nin = lib.fz_trace(buf_in, len(buf_in), W, C, S, mask, R, len(pre), buf, nbits)
    rows = np.frombuffer(buf, dtype=np.float32).reshape(nbits, MAXIN + 2)[off * 8:]
    gidx, expect = layout(mask, S)
    assert nin == expect, (nin, expect)
    groups = []
    for members in gidx:
        out = []
        for r in rows[:, members] if members else np.zeros((len(rows), 0)):
            nz = r[np.abs(r) > 1e-6]
            out.append(None if len(nz) == 0 else round(float(nz.mean()), 3))
        groups.append(out)
    chunk = data[start:start + nbytes]
    chars = [chr(b) if 32 <= b < 127 else ("↵" if b == 10 else f"{b:02x}") for b in chunk]
    return {"label": label, "title": title, "chars": chars, "groups": groups,
            "p": [round(float(min(max(p, 1e-5), 1 - 1e-5)), 5) for p in rows[:, MAXIN]],
            "bits": [int(b) for b in rows[:, MAXIN + 1]]}


def main() -> None:
    d = json.loads((ROOT / "results" / "funnelzip.json").read_text())
    res = json.loads((ROOT / "results" / "resonance.json").read_text())
    S = ROOT / "samples"
    p = {r["file"]: r["params"] for r in d["rows"]}
    lic = (S / "text" / "license.txt").read_bytes()
    k = lic.find(b"Software", 6000)
    assert k > 0
    sensor = (S / "text" / "sensor.csv").read_bytes()
    k2 = sensor.find(b"\n", 20000) + 1
    assert k2 > 0
    sounds = (S / "audio" / "system_sounds.wav").read_bytes()
    pcm = np.abs(np.frombuffer(sounds[44:], dtype="<i2").astype(np.int64))
    loud = int(np.argmax(np.convolve(pcm[30000:200000], np.ones(64), "valid"))) + 30000
    s_off = 44 + 2 * loud
    cam = (S / "video" / "pan_landscape.rgb").read_bytes()
    c_off = 3 * 110592 + 60 * 768 + 200
    traces = {
        "license": trace(lic, p["license.txt"], k, 14, "Text (license)", f"license.txt from byte {k:,}"),
        "sensor": trace(sensor, p["sensor.csv"], k2, 14, "Sensor CSV", f"sensor.csv from byte {k2:,}"),
        "sound": trace(sounds, p["system_sounds.wav"], s_off, 14, "Sound (16-bit)", f"system_sounds.wav, loudest moment, from byte {s_off:,} (7 samples, low byte then high byte)"),
        "camera": trace(cam, p["camera pan video (raw RGB, 48 frames)"], c_off, 14, "Video (camera pan)", f"camera pan, frame 4, from byte {c_off:,}: Rhythm reads the same pixel one frame earlier, Motion reads where it moved from"),
    }
    ss = next(r for r in d["rows"] if r["file"] == "system_sounds.wav")
    rs = next(r for r in res["rows"] if r["file"] == "system_sounds.wav")
    journey = [
        ["raw file", ss["size"], "var(--ref)"],
        ["gzip −9", ss["baselines"]["gzip -9"], "var(--ref)"],
        ["xz −9e", ss["baselines"]["xz -9e"], "var(--ref)"],
        ["V6 predictive (memory only)", rs["v6"], "color-mix(in srgb, var(--accent) 45%, var(--panel2))"],
        ["FLAC −12 (best existing)", ss["baselines"]["FLAC -12"], "var(--ref)"],
        ["V9a waves found by resonance", rs["v9a-waves"]["stored"], "color-mix(in srgb, var(--accent) 60%, var(--panel2))"],
        ["V9b resonant filter", rs["v9b-resonator"]["stored"], "color-mix(in srgb, var(--accent) 75%, var(--panel2))"],
        ["FunnelZip (all lenses mixed)", ss["funnelzip"], "var(--accent)"],
    ]
    rows = d["rows"]
    proof = json.loads((ROOT / "results" / "lossless_proof.json").read_text())
    sr = json.loads((ROOT / "results" / "speed_round.json").read_text())
    v4 = {r["file"]: r for r in json.loads((ROOT / "results" / "funnelzip_v4.json").read_text())["rows"]}
    tx = next(r for r in sr["core"] if r["file"] == "1 MB text")
    sig = [r["now"]["enc"] for r in sr["core"] if r["file"] in ("1 MB sounds", "photo pixels", "program binary")]
    lo, hi = min(sig), max(sig)
    j4 = next(r for r in rows if "4K" in r["file"])
    tot_now = sum(r["funnelzip"] for r in rows if r["file"] in v4)
    tot_v4 = sum(v4[r["file"]]["funnelzip"] for r in rows if r["file"] in v4)
    sr["stats"] = [[f"{j4['enc_s']:.1f} s", f"4K JPEG compress (was {v4[j4['file']]['enc_s']:.0f} s)"],
                   [f"{tx['now']['enc']:.2f} MB/s", f"text on one core (was {tx['v4']['enc']:.2f})"],
                   [f"{100 * (tot_now / tot_v4 - 1):+.1f}%", "total size vs last round, all 32 files"],
                   [f"{sum(r['ok'] for r in rows)} / {len(rows)}", "still byte-identical"]]
    mbps = sum(r["size"] for r in rows) / 1e6 / sum(r["enc_s"] for r in rows)
    dps = sum(r["size"] for r in rows) / 1e6 / sum(r["dec_s"] for r in rows)
    synth = next(r for r in rows if r["file"].startswith("synthetic"))
    wall = next(r for r in rows if r["file"].startswith("wallpaper"))
    jp = [r for r in rows if r["kind"] == "image file" and "JPEG" in r["file"]]
    pn = [r for r in rows if r["kind"] == "image file" and "PNG" in r["file"]]
    jr = [100 * (1 - r["funnelzip"] / r["size"]) for r in jp]
    pr_ = [100 * (1 - r["funnelzip"] / r["size"]) for r in pn]
    verdict = [
        ["✓ Combining beats choosing", "win", "Every lens is always on and the funnel learns whom to trust. On sound it beats our best single method (V9b) by another 20%, and on the sensor log it beats V6 by half."],
        ["✓ Each idea earns its place", "win", "Switch one lens off and the file it was built for grows: resonator on sound, word/column on CSV, memory on code, rhythm on a music loop, motion on a camera pan, grid on photos. The Equation lens is the exception: the resonator covers it."],
        ["✓ Repeats of repeats", "win", "A repeated equation gets its own resonance: the rhythm lens and long-term predictor turned a looped bar into ~46 KB vs FLAC's 112 KB, and found a video's frame period by listening."],
        ["✓ Image files, where the structure lives", "win", f"JPEG files shrink {min(jr):.0f}–{max(jr):.0f}% by recompressing their cosine coefficients, and PNG files up to {max(pr_):.0f}% via their pixels. Every one is rebuilt byte-identical."],
        ["◐ Flat synthetic pictures", "tie", f"The prediction-residual reshape took the synthetic image from 904 to {synth['funnelzip']:,} bytes; WebP-lossless is at {synth['best_existing'][1]:,}. Nearly closed."],
        ["✓ Pure lossless, proven", "win", f"Every file is decompressed and compared: SHA-256 and every byte identical, {sum(max(0, r['bytes_different']) for r in proof)} bytes different over {sum(r['size'] for r in proof) / 1e6:.1f} MB of test data."],
        ["◐ Speed: 1 MB/s for text and big JPEGs, not yet for signals", "tie", f"One core now codes text at {tx['now']['enc']:.2f} MB/s and the 4K JPEG runs at {5.62 / j4['enc_s']:.2f} MB/s (was 0.10). Sound, pixels and program code are at {lo:.2f}–{hi:.2f} MB/s on one core: a bit-by-bit mixing model is sequential, so the last 2× needs fewer lenses per bit or files split into parts that really differ (as JPEG planes do). Whole benchmark: {mbps:.2f} MB/s compress, {dps:.2f} MB/s decompress (Listen's trials included). xz/brotli are still far faster."],
        ["= Random data", "tie", "No gain, as the counting argument says: random.bin +0.3%. Nothing is printed from nothing. English prose (license.txt) is still 3% behind brotli's built-in English dictionary."],
        ["→ Next steps", "tie", "See 'Next steps — tried' above for what was built this round, and the list under it for what comes after."],
    ]
    rep = []
    for r in rows:
        if r["kind"] == "repeats" and r["file"] in d["ablation"]:
            a = d["ablation"][r["file"]]
            rep.append({"file": r["file"], "without": a["-rhythm"], "with": a["all"], "best": r["best_existing"], "R": r["params"][3]})
    pvf = json.loads((ROOT / "results" / "pixels_vs_file.json").read_text())
    nxt = json.loads((ROOT / "results" / "next_steps.json").read_text())
    before = {r["file"]: r["funnelzip"] for r in json.loads((ROOT / "results" / "funnelzip_v3.json").read_text())["rows"]}
    after = {r["file"]: r for r in rows}
    how = {"synthetic (raw RGB)": "prediction residual", "beach_photo (raw RGB)": "colour transform",
           "wallpaper (raw RGB)": "colour transform", "beach_photo.png (PNG file)": "pixels + colour transform",
           "wallpaper.png (PNG file)": "pixels + colour transform", "synthetic.png (PNG file)": "pixels + prediction residual (or bytes if smaller)",
           "typing.py": "caps + words", "json_decoder.py": "caps + words", "license.txt": "tried caps + words: no gain, kept as is",
           "landscape_q90.jpg (JPEG file)": "coefficient layout told, AC one byte each, own model per group",
           "sonoma_apple.jpg (JPEG file)": "coefficient layout told, AC one byte each, own model per group",
           "macOS DefaultBackground.jpg (4K) (JPEG file)": "coefficient layout told, AC one byte each, own model per group"}
    reshape = [{"file": f, "how": h, "before": before[f], "after": after[f]["funnelzip"], "best": after[f]["best_existing"]}
               for f, h in how.items() if f in before and f in after]
    out = {"speedround": sr, "proof": proof, "reshape": reshape, "next": nxt, "pvf": pvf, "repeat": rep, "rows": rows, "ablation": d["ablation"], "trace": {"groups": GROUPS, "files": traces}, "journey": journey, "verdict": verdict}
    tpl = (ROOT / "bench" / "funnelzip_template.html").read_text()
    dst = ROOT / "outputs" / "visuals" / "funnelzip.html"
    dst.write_text(tpl.replace("/*DATA*/null", json.dumps(out)))
    print("wrote", dst.relative_to(ROOT), f"enc {mbps:.2f} MB/s dec {dps:.2f} MB/s")


if __name__ == "__main__":
    main()
