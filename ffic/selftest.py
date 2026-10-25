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
    fb = formats()
    n = len(images(None)) * len(MODES)
    print(f"{n - (bad - (not refused))}/{n} round trips exact; damaged file {'refused' if refused else 'NOT refused'}")
    sys.exit(1 if bad or fb else 0)


def formats():
    """every file format the tool accepts, checked against the ORIGINAL file (a converted copy once hid a 16-bit bug)"""
    from PIL import Image
    import ffic
    r = np.random.default_rng(2); y, x = np.mgrid[0:37, 0:53]
    rgb = np.stack([(x * 5) % 256, (y * 7) % 256, (x + y) % 256], -1).astype(np.uint8); al = ((x * y) % 256).astype(np.uint8)
    cases = {"rgba.png": Image.fromarray(np.dstack([rgb, al]), "RGBA"), "la.png": Image.fromarray(np.dstack([rgb[..., 0], al]), "LA"),
             "grey.png": Image.fromarray(rgb[..., 1]), "palette.png": Image.fromarray(rgb).quantize(20),
             "grey16.png": Image.fromarray((x * 977 + y * 53 + r.integers(0, 40, x.shape)).astype(np.uint16)),
             "rgb.tiff": Image.fromarray(rgb)}
    bad = 0
    with tempfile.TemporaryDirectory() as t:
        for name, im in cases.items():
            p = f"{t}/{name}"; im.save(p); ext = ".tiff" if name.endswith(".tiff") else ".png"
            ffic.encode(p, f"{t}/x.ffic", "fast"); ffic.decode(f"{t}/x.ffic", f"{t}/back{ext}")
            ok = ffic.same(p, f"{t}/back{ext}") and np.array_equal(np.asarray(Image.open(p)), np.asarray(Image.open(f"{t}/back{ext}")))
            bad += not ok; print(f"format {name}: {'exact' if ok else 'MISMATCH'}")
        yuv = f"{t}/v.y4m"                                                    # 3 frames of 4:2:0 video, odd size
        with open(yuv, "wb") as f:
            f.write(b"YUV4MPEG2 W33 H17 F25:1 Ip A1:1 C420jpeg\n")
            for k in range(3): f.write(b"FRAME\n" + r.integers(0, 256, 33 * 17 + 2 * 17 * 9, dtype=np.uint8).tobytes())
        ffic.encode(yuv, f"{t}/v.ffic", "fast"); ffic.decode(f"{t}/v.ffic", f"{t}/vb.y4m")
        ok = open(yuv, "rb").read() == open(f"{t}/vb.y4m", "rb").read(); bad += not ok; print(f"format video .y4m 4:2:0 33x17: {'exact' if ok else 'MISMATCH'}")
    return bad


if __name__ == "__main__":
    main()
