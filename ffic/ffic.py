#!/usr/bin/env python3
"""FFIC: FunnelField Image Coder, lossless. Command-line tool around the `ffic` coder.

    python ffic.py encode photo.png photo.ffic [--preset max|fast]
    python ffic.py decode photo.ffic photo.png
    python ffic.py info   photo.ffic
    python ffic.py test   photo.png [...]          encode + decode + compare, every preset

Input: any image Pillow can read (PNG, PPM, BMP, TIFF, ...), 8 bits per channel, RGB or grey. Grey images come back grey.
Images with transparency or 16-bit samples are refused (not supported yet), so nothing is silently changed.
Presets:  max   smallest files (best coder + terrain balls), layers coded one after another
          fast  same model, the three colour layers on three threads (about 1.8x faster, ~0.05% bigger)
Every file carries a CRC-32 of the pixels; decoding fails loudly if the rebuilt image differs.
"""
import argparse, os, struct, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "ffic")
BEST = "ltpcrewb sal=-1 salp=3 salmax=8 salbias=2 register=1 grav=0.5 field=1 merge=0.08 kres=1 sleep=16 fmax=128 ms=1 bounce=1 salall=2 apm=1 mix2=1 mline=2 matf=1 mctx=1"
PRESETS = {"max": BEST + " terrain=1 tcoup=0.1 tmag=0.15 twall=10", "fast": BEST + " pipe=1 snapk=32"}


def build():
    src = os.path.join(HERE, "ffic.c")
    if not os.path.exists(EXE) or os.path.getmtime(EXE) < os.path.getmtime(src):
        subprocess.check_call(["cc", "-O2", "-march=native", "-pthread", "-o", EXE, src, "-lm"])


def load(path):
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"): sys.exit(f"{path}: 16-bit / float images are not supported yet")
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info): sys.exit(f"{path}: images with transparency are not supported yet")
    grey = im.mode in ("L", "1")
    return im.convert("RGB"), grey


def read_header(path):
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic not in (b"FFIC1\0", b"FFIC2\0"): sys.exit(f"{path}: not an FFIC file")
        w, h, ol = struct.unpack("<III", f.read(12)); opts = f.read(ol).decode(); sl = struct.unpack("<III", f.read(12))
    return {"version": magic[:5].decode(), "width": w, "height": h, "options": opts, "streams": sl, "bytes": os.path.getsize(path)}


def encode(src, dst, preset):
    build(); im, grey = load(src)
    with tempfile.TemporaryDirectory() as t:
        raw = os.path.join(t, "in.rgb"); open(raw, "wb").write(im.tobytes())
        cfg = PRESETS[preset] + (" src=1" if grey else "")
        t0 = time.time(); out = subprocess.run([EXE, raw, str(im.width), str(im.height)] + cfg.split() + [f"out={dst}"], capture_output=True, text=True)
        if out.returncode: sys.exit(out.stderr.strip() or "encoding failed")
    return time.time() - t0, im.width * im.height * (1 if grey else 3)


def decode(src, dst):
    build(); hd = read_header(src)
    from PIL import Image
    with tempfile.TemporaryDirectory() as t:
        raw = os.path.join(t, "out.rgb"); t0 = time.time()
        out = subprocess.run([EXE, "-d", src, raw], capture_output=True, text=True)
        if out.returncode: sys.exit(out.stderr.strip() or "decoding failed")
        im = Image.frombytes("RGB", (hd["width"], hd["height"]), open(raw, "rb").read())
    if "src=1" in hd["options"].split(): im = im.convert("L")
    im.save(dst); return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode"); e.add_argument("src"); e.add_argument("dst"); e.add_argument("--preset", choices=PRESETS, default="max")
    d = sub.add_parser("decode"); d.add_argument("src"); d.add_argument("dst")
    i = sub.add_parser("info"); i.add_argument("src")
    tt = sub.add_parser("test"); tt.add_argument("src", nargs="+")
    a = ap.parse_args()
    if a.cmd == "encode":
        s, n = encode(a.src, a.dst, a.preset); b = os.path.getsize(a.dst)
        print(f"{a.dst}: {b:,} bytes ({8 * b / n * (1 if n else 0):.3f} bits per sample), {s:.1f} s, preset {a.preset}")
    elif a.cmd == "decode":
        s = decode(a.src, a.dst); print(f"{a.dst}: decoded in {s:.1f} s, checksum ok")
    elif a.cmd == "info":
        h = read_header(a.src); print(f"{a.src}: {h['version']} {h['width']}x{h['height']}, {h['bytes']:,} bytes, streams G/R/B {h['streams']}\n  options: {h['options']}")
    else:
        from PIL import Image
        bad = 0
        with tempfile.TemporaryDirectory() as t:
            for src in a.src:
                ref, grey = load(src)
                for p in PRESETS:
                    f, o = os.path.join(t, "x.ffic"), os.path.join(t, "x.png")
                    se, _ = encode(src, f, p); sd = decode(f, o)
                    back = Image.open(o).convert("RGB")
                    ok = back.tobytes() == ref.tobytes() and (Image.open(o).mode == "L") == grey
                    bad += not ok
                    print(f"{os.path.basename(src)} [{p}]: {os.path.getsize(f):,} bytes, encode {se:.1f} s, decode {sd:.1f} s, {'exact' if ok else 'MISMATCH'}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
