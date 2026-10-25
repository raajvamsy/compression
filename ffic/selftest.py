"""Self-test: every awkward image shape round-trips exactly in every mode.

    python selftest.py [DATADIR]     (DATADIR with kod_kodim20.rgb for real-photo crops; otherwise synthetic images only)

Shapes: 1x1, single rows and columns, odd sizes, pure noise, all black, all white, plus crops of a real photo.
Modes: max preset, fast preset (three threads), pixel-by-pixel with whole-colour match. Also checks that a damaged file is
refused (checksum). Exit code 0 only if everything passes.
"""
import os, subprocess, sys, tempfile
import numpy as np
from ffic import EXE, PRESETS, build

MODES = {"max": PRESETS["max"], "fast": PRESETS["fast"], "whole-colour": PRESETS["max"].replace(" terrain=1 tcoup=0.1 tmag=0.15 twall=10", "") + " inter=1 cmatch=1 mexcl=1 minf=1 snapk=32"}


def images(datadir):
    r = np.random.default_rng(1); out = {}
    if datadir and os.path.exists(f"{datadir}/kod_kodim20.rgb"):
        ph = np.fromfile(f"{datadir}/kod_kodim20.rgb", np.uint8).reshape(512, 768, 3)
    else:
        y, x = np.mgrid[0:300, 0:300]; ph = np.stack([(x + y) % 256, (x * 3) % 256, (y * 5) % 256], -1).astype(np.uint8)
    for w, h in [(1, 1), (2, 1), (1, 7), (3, 2), (5, 5), (17, 3), (64, 1), (1, 64), (127, 33), (300, 70)]:
        out[f"crop {w}x{h}"] = ph[:h, :w].copy()
    out["noise 29x31"] = r.integers(0, 256, (31, 29, 3), dtype=np.uint8)
    out["black 50x40"] = np.zeros((40, 50, 3), np.uint8); out["white 50x40"] = np.full((40, 50, 3), 255, np.uint8)
    return out


def main():
    build(); bad = 0
    with tempfile.TemporaryDirectory() as t:
        for name, img in images(sys.argv[1] if len(sys.argv) > 1 else None).items():
            h, w, _ = img.shape; src = f"{t}/in.rgb"; img.tofile(src)
            for m, cfg in MODES.items():
                e = subprocess.run([EXE, src, str(w), str(h)] + cfg.split() + [f"out={t}/x.ffic"], capture_output=True)
                d = subprocess.run([EXE, "-d", f"{t}/x.ffic", f"{t}/out.rgb"], capture_output=True)
                ok = e.returncode == 0 and d.returncode == 0 and open(src, "rb").read() == open(f"{t}/out.rgb", "rb").read()
                bad += not ok
                if not ok: print(f"FAIL {name} [{m}]")
        blob = bytearray(open(f"{t}/x.ffic", "rb").read()); blob[len(blob) // 2] ^= 0x10; open(f"{t}/bad.ffic", "wb").write(blob)
        refused = subprocess.run([EXE, "-d", f"{t}/bad.ffic", f"{t}/bad.rgb"], capture_output=True).returncode != 0
        bad += not refused
    n = len(images(None)) * len(MODES)
    print(f"{n - (bad - (not refused))}/{n} round trips exact; damaged file {'refused' if refused else 'NOT refused'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
