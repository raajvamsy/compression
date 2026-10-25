#!/usr/bin/env python3
"""FFIC: FunnelField Image Coder, lossless. Command-line tool around the `ffic` coder.

    python ffic.py encode in.png   out.ffic [--preset max|fast]     images
    python ffic.py encode in.y4m   out.ffic [--preset max|fast]     video (YUV4MPEG2, 8-bit, 4:2:0 / 4:2:2 / 4:4:4 / mono)
    python ffic.py decode in.ffic  out.png | out.y4m
    python ffic.py info   in.ffic
    python ffic.py test   in.png|in.y4m [...]                      encode + decode + compare, every preset
    python ffic.py pack   folder/ archive.ffa [--preset max|fast]   a whole folder, like zip (images by FFIC, other files by xz)
    python ffic.py unpack archive.ffa folder/

Images: anything Pillow reads (PNG, PPM, BMP, TIFF, WebP, ...): 8-bit RGB, grey, 16-bit grey (depth, elevation, scientific), RGB + transparency, grey +
transparency, palette (+ transparency). Pixels and transparency come back exactly; a palette image comes back as the same
palette image (same colour numbers, palette and transparency). RGB images with 256 colours or fewer are also tried as a palette.
16-bit colour and float images are refused (not supported yet) so nothing is silently changed.
Video: the brightness (Y) and colour (U, V) layers are stored exactly as they are in the file (no colour conversion); each
layer's frames are stacked so the funnels keep learning from frame to frame and see the previous frame in every direction.
The output .y4m is byte-identical to the input.

Files: a plain RGB or grey image is one FFIC2 stream. Anything with several layers (transparency, video) is an FFX1 container:
"FFX1" + JSON description (original format, how to rebuild it) + one FFIC2 stream per layer. Every stream carries a CRC-32
of its pixels and decoding fails loudly if anything differs.
Presets:  max   smallest files (best coder + terrain balls), layers coded one after another
          fast  same model, colour layers on three threads (about 2x faster, ~0.05% bigger)
"""
import argparse, json, os, struct, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "ffic")
BEST = "ltpcrewb sal=-1 salp=3 salmax=8 salbias=2 register=1 grav=0.5 field=1 merge=0.08 kres=1 sleep=16 fmax=128 ms=1 bounce=1 salall=2 apm=1 mix2=1 mline=2 matf=1 mctx=1"
PRESETS = {"max": BEST + " terrain=1 tcoup=0.1 tmag=0.15 twall=10", "fast": BEST + " pipe=1 snapk=32"}
GREY = " src=1 planes=1"


def build():
    src = os.path.join(HERE, "ffic.c")
    if not os.path.exists(EXE) or os.path.getmtime(EXE) < os.path.getmtime(src):
        subprocess.check_call(["cc", "-O2", "-march=native", "-pthread", "-o", EXE, src, "-lm"])


# ---------------------------------------------------------------- one layer <-> one FFIC2 stream
def code_layer(arr, cfg, frameh=0):
    """arr: HxWx3 (colour) or HxW (grey), uint8 or uint16 -> FFIC2 bytes. 16-bit layers are coded with just the bits they use."""
    grey = arr.ndim == 2; h, w = arr.shape[:2]
    rgb = np.repeat(arr[..., None], 3, 2) if grey else arr
    bits = 8 if arr.dtype == np.uint8 else max(9, int(arr.max()).bit_length())
    if bits > 8: rgb = rgb.astype("<u2")
    with tempfile.TemporaryDirectory() as t:
        np.ascontiguousarray(rgb).tofile(f"{t}/in.rgb")
        c = cfg + (GREY if grey else "") + (f" frameh={frameh}" if frameh else "") + (f" bits={bits}" if bits > 8 else "")
        r = subprocess.run([EXE, f"{t}/in.rgb", str(w), str(h)] + c.split() + [f"out={t}/o.ffic"], capture_output=True, text=True)
        if r.returncode: sys.exit(r.stderr.strip() or "encoding failed")
        return open(f"{t}/o.ffic", "rb").read()


def decode_layer(blob):
    hd = header(blob)
    with tempfile.TemporaryDirectory() as t:
        open(f"{t}/i.ffic", "wb").write(blob)
        r = subprocess.run([EXE, "-d", f"{t}/i.ffic", f"{t}/o.rgb"], capture_output=True, text=True)
        if r.returncode: sys.exit(r.stderr.strip() or "decoding failed")
        deep = any(o.startswith("bits=") for o in hd["options"].split())
        a = np.fromfile(f"{t}/o.rgb", "<u2" if deep else np.uint8).reshape(hd["height"], hd["width"], 3)
        if deep: a = a.astype(np.uint16)
    return a[..., 1].copy() if "planes=1" in hd["options"].split() or "src=1" in hd["options"].split() else a


def header(blob):
    if blob[:6] not in (b"FFIC1\0", b"FFIC2\0"): sys.exit("not an FFIC stream")
    w, h, ol = struct.unpack("<III", blob[6:18]); opts = blob[18:18 + ol].decode(); sl = struct.unpack("<III", blob[18 + ol:30 + ol])
    return {"version": blob[:5].decode(), "width": w, "height": h, "options": opts, "streams": sl}


# ---------------------------------------------------------------- container
def write_container(dst, meta, parts):
    meta = dict(meta, parts=[[n, len(b)] for n, b in parts]); js = json.dumps(meta, separators=(",", ":")).encode()
    with open(dst, "wb") as f:
        f.write(b"FFX1" + struct.pack("<I", len(js)) + js)
        for _, b in parts: f.write(b)


def read_any(path):
    data = open(path, "rb").read()
    if data[:4] != b"FFX1": return {"kind": "plain"}, {"image": data}
    n = struct.unpack("<I", data[4:8])[0]; meta = json.loads(data[8:8 + n]); pos = 8 + n; parts = {}
    for name, size in meta["parts"]: parts[name] = data[pos:pos + size]; pos += size
    return meta, parts


# ---------------------------------------------------------------- images
def load_image(path):
    from PIL import Image
    im = Image.open(path); im.load()
    if getattr(im, "n_frames", 1) > 1: sys.exit(f"{path}: animated / multi-page images are not supported yet")
    if im.mode in ("F", "RGB;16", "RGBA;16"): sys.exit(f"{path}: float and 16-bit colour images are not supported yet")
    if im.mode == "P": im = im.convert("RGBA" if "transparency" in im.info else "RGB")
    if im.mode == "PA": im = im.convert("RGBA")
    if im.mode == "1": im = im.convert("L")
    if im.mode in ("I;16B", "I;16L"): im = im.convert("I;16")
    if im.mode not in ("RGB", "L", "RGBA", "LA", "I;16", "I"): sys.exit(f"{path}: {im.mode} images are not supported yet")
    return im


def palette_parts(idx, pal, cfg):
    """few-colour image: palette sorted by brightness + the map of colour numbers (a smooth grey image the funnels can guess)"""
    pal = np.asarray(list(pal) + [0] * (768 - len(pal)), np.int64).reshape(-1, 3); order = np.argsort(pal @ [299, 587, 114], kind="stable")
    rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order))
    return code_layer(rank[idx].astype(np.uint8), cfg), order.tolist()


def image_extras(im):
    """what an image file carries besides pixels: colour profile, camera EXIF, DPI, PNG text notes (kept and re-attached)"""
    import base64
    x = {k: base64.b64encode(im.info[k]).decode() for k in ("icc_profile", "exif") if isinstance(im.info.get(k), bytes) and im.info[k]}
    if "dpi" in im.info: x["dpi"] = [float(v) for v in im.info["dpi"]]
    if getattr(im, "text", None): x["text"] = dict(im.text)
    return x


def save_kwargs(meta, ext):
    import base64
    from PIL import PngImagePlugin
    x = meta.get("extras", {}); kw = {k: base64.b64decode(x[k]) for k in ("icc_profile", "exif") if k in x}
    if "dpi" in x: kw["dpi"] = tuple(x["dpi"])
    if "text" in x and ext == ".png":
        p = PngImagePlugin.PngInfo()
        for k, v in x["text"].items(): p.add_text(k, v)
        kw["pnginfo"] = p
    return kw


def encode_image(src, dst, preset):
    from PIL import Image
    cfg = PRESETS[preset]; raw = Image.open(src); raw.load()
    if getattr(raw, "n_frames", 1) > 1: sys.exit(f"{src}: animated / multi-page images are not supported yet")
    extras = image_extras(raw)
    if raw.mode == "P" and raw.getpalette() is not None:                 # palette image: keep it a palette image, exactly
        pal = raw.getpalette()[:3 * 256]; plen = len(pal); pal += [0] * (768 - len(pal)); idx = np.asarray(raw)
        blob, order = palette_parts(idx, pal, cfg); tr = raw.info.get("transparency")
        meta = {"kind": "image", "mode": "P", "width": raw.width, "height": raw.height, "palette": pal[:plen], "order": order, "extras": extras,
                "transparency": (list(tr) if isinstance(tr, bytes) else tr)}
        write_container(dst, meta, [("index", blob)]); return
    im = load_image(src); a = np.asarray(im)
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):                       # 16-bit grey (depth, elevation, scientific)
        a = np.asarray(im).astype(np.int64)
        if a.min() < 0 or a.max() > 65535: sys.exit(f"{src}: values outside 0..65535 are not supported")
        write_container(dst, {"kind": "image", "mode": im.mode, "width": im.width, "height": im.height, "extras": extras}, [("grey16", code_layer(a.astype(np.uint16), cfg))]); return
    if im.mode in ("RGB", "L"):
        best = code_layer(a, cfg)
        if im.mode == "RGB":                                                # few colours (graphics, screenshots): try a palette too
            flat = a.reshape(-1, 3); key = (flat[:, 0].astype(np.int64) << 16) | (flat[:, 1].astype(np.int64) << 8) | flat[:, 2]
            cols, inv = np.unique(key, return_inverse=True)
            if len(cols) <= 256:
                pal = np.stack([cols >> 16, (cols >> 8) & 255, cols & 255], 1).ravel().tolist(); pal += [0] * (768 - len(pal))
                blob, order = palette_parts(inv.reshape(a.shape[:2]), pal, cfg)
                if len(blob) + 2 * len(cols) * 3 < len(best):
                    write_container(dst, {"kind": "image", "mode": "RGB", "as": "palette", "width": im.width, "height": im.height,
                                          "palette": pal, "order": order, "extras": extras}, [("index", blob)]); return
        if extras: write_container(dst, {"kind": "image", "mode": im.mode, "width": im.width, "height": im.height, "extras": extras}, [("image", best)])
        else: open(dst, "wb").write(best)
        return
    colour, alpha = a[..., :-1], a[..., -1]
    if colour.shape[2] == 1: colour = colour[..., 0]
    parts, meta = [("colour", code_layer(colour, cfg))], {"kind": "image", "mode": im.mode, "width": im.width, "height": im.height, "extras": extras}
    if (alpha == alpha.flat[0]).all(): meta["alpha_const"] = int(alpha.flat[0])     # constant transparency: one number
    else: parts.append(("alpha", code_layer(alpha, cfg)))
    write_container(dst, meta, parts)


def decode_image(meta, parts):
    from PIL import Image
    if meta["kind"] == "plain" or "image" in parts:
        a = decode_layer(parts["image"]); return Image.fromarray(a, "L" if a.ndim == 2 else "RGB")
    if "grey16" in parts:
        a = decode_layer(parts["grey16"]); from PIL import Image as I
        return I.fromarray(a.astype(np.uint16)) if meta["mode"] != "I" else I.fromarray(a.astype(np.int32))
    if "index" in parts:                                                   # palette: map the colour numbers back
        from PIL import Image as I
        order = np.asarray(meta["order"]); idx = order[decode_layer(parts["index"])].astype(np.uint8)
        if meta["mode"] == "P":
            im = I.fromarray(idx, "P"); im.putpalette(meta["palette"])
            tr = meta.get("transparency")
            if tr is not None: im.info["transparency"] = bytes(tr) if isinstance(tr, list) else tr
            return im
        pal = list(meta["palette"]) + [0] * (768 - len(meta["palette"]))
        return I.fromarray(np.asarray(pal, np.uint8).reshape(-1, 3)[idx], "RGB")
    c = decode_layer(parts["colour"])
    al = decode_layer(parts["alpha"]) if "alpha" in parts else np.full(c.shape[:2], meta["alpha_const"], np.uint8)
    a = np.dstack([c, al]) if c.ndim == 3 else np.dstack([c, al])
    return Image.fromarray(a, meta["mode"])


# ---------------------------------------------------------------- video (YUV4MPEG2)
def read_y4m(path):
    data = open(path, "rb").read(); nl = data.index(b"\n"); head = data[:nl].decode()
    tok = head.split(); assert tok[0] == "YUV4MPEG2", "not a YUV4MPEG2 file"
    w = int(next(t[1:] for t in tok if t[0] == "W")); h = int(next(t[1:] for t in tok if t[0] == "H"))
    cs = next((t[1:] for t in tok if t[0] == "C"), "420jpeg")
    if cs.startswith("420"): cw, ch = (w + 1) // 2, (h + 1) // 2
    elif cs.startswith("422"): cw, ch = (w + 1) // 2, h
    elif cs.startswith("444") and cs in ("444",): cw, ch = w, h
    elif cs.startswith("mono"): cw, ch = 0, 0
    else: sys.exit(f"{path}: colour space C{cs} not supported (8-bit 420/422/444/mono only)")
    pos = nl + 1; fl, Y, U, V = [], [], [], []; fs = w * h + 2 * cw * ch
    while pos < len(data):
        e = data.index(b"\n", pos); fl.append(data[pos:e].decode()); pos = e + 1
        fr = np.frombuffer(data, np.uint8, fs, pos); pos += fs
        Y.append(fr[:w * h].reshape(h, w))
        if cw: U.append(fr[w * h:w * h + cw * ch].reshape(ch, cw)); V.append(fr[w * h + cw * ch:].reshape(ch, cw))
    return head, fl, w, h, cw, ch, Y, U, V


def encode_video(src, dst, preset):
    head, fl, w, h, cw, ch, Y, U, V = read_y4m(src); cfg = PRESETS[preset]
    jobs = [("Y", np.concatenate(Y), h)] + ([("U", np.concatenate(U), ch), ("V", np.concatenate(V), ch)] if cw else [])
    with ThreadPoolExecutor(3) as ex:                                  # the three layers are independent files: code them at once
        parts = list(zip([j[0] for j in jobs], ex.map(lambda j: code_layer(j[1], cfg, j[2]), jobs)))
    meta = {"kind": "y4m", "header": head, "frames": len(fl), "w": w, "h": h, "cw": cw, "ch": ch}
    if any(f != "FRAME" for f in fl): meta["frame_lines"] = fl
    write_container(dst, meta, parts)


def decode_video(meta, parts, dst):
    with ThreadPoolExecutor(3) as ex: layers = dict(zip(parts, ex.map(decode_layer, parts.values())))
    w, h, cw, ch, n = meta["w"], meta["h"], meta["cw"], meta["ch"], meta["frames"]; fl = meta.get("frame_lines", ["FRAME"] * n)
    with open(dst, "wb") as f:
        f.write((meta["header"] + "\n").encode())
        for i in range(n):
            f.write((fl[i] + "\n").encode()); f.write(layers["Y"][i * h:(i + 1) * h].tobytes())
            if cw: f.write(layers["U"][i * ch:(i + 1) * ch].tobytes()); f.write(layers["V"][i * ch:(i + 1) * ch].tobytes())


# ---------------------------------------------------------------- commands
def encode(src, dst, preset):
    build(); t0 = time.time()
    (encode_video if src.lower().endswith(".y4m") else encode_image)(src, dst, preset)
    return time.time() - t0


def decode(src, dst):
    build(); t0 = time.time(); meta, parts = read_any(src)
    if meta["kind"] == "y4m":
        if not dst.lower().endswith(".y4m"): sys.exit("this file holds video: decode to a .y4m file")
        decode_video(meta, parts, dst)
    else:
        im = decode_image(meta, parts); ext = os.path.splitext(dst)[1].lower()
        if ext in (".jpg", ".jpeg"): sys.exit("JPEG output would lose pixels: decode to .png (or another lossless format)")
        im.save(dst, **save_kwargs(meta, ext), **({"lossless": True, "quality": 100, "method": 4} if ext == ".webp" else {}))
    return time.time() - t0


def same(src, out):
    if src.lower().endswith(".y4m"): return open(src, "rb").read() == open(out, "rb").read()
    from PIL import Image
    A, B = Image.open(src), Image.open(out); A.load(); B.load()
    if A.mode == "P":                                                       # palette in, palette out: same numbers, palette, transparency
        return B.mode == "P" and A.tobytes() == B.tobytes() and A.getpalette() == B.getpalette() and A.info.get("transparency") == B.info.get("transparency")
    a = A.convert("L") if A.mode == "1" else A                              # compare with the ORIGINAL file, not a converted copy
    if A.mode in ("I", "I;16", "I;16B", "I;16L"): return np.array_equal(np.asarray(A).astype(np.int64), np.asarray(B).astype(np.int64)) and B.mode.startswith("I")
    return a.mode == B.mode and np.array_equal(np.asarray(a), np.asarray(B))


# ---------------------------------------------------------------- folders (like zip)
IMG_EXT = {".png", ".tif", ".tiff", ".bmp", ".ppm", ".pgm", ".pnm", ".webp", ".gif", ".tga", ".y4m"}


def _pack_one(path, preset):
    """-> (kind, blob, note). Images that FFIC stores exactly become FFIC streams; everything else is kept byte for byte (xz)."""
    import lzma
    ext = os.path.splitext(path)[1].lower()
    if ext in IMG_EXT:
        with tempfile.TemporaryDirectory() as t:
            f = f"{t}/x.ffic"
            try:
                r = subprocess.run([sys.executable, os.path.abspath(__file__), "encode", path, f, "--preset", preset], capture_output=True, text=True)
                if r.returncode == 0: return "image", open(f, "rb").read(), ""
                note = (r.stderr.strip().splitlines() or ["not supported"])[-1]
            except Exception as e:
                note = str(e)
    else: note = ""
    return "file", lzma.compress(open(path, "rb").read(), preset=9 | lzma.PRESET_EXTREME), note


def pack(folder, dst, preset, jobs=4):
    """FFA1 archive: "FFA1" + JSON manifest (path, kind, size, original format) + blobs. Images keep their pixels exactly;
    other files keep their bytes exactly."""
    files = sorted(os.path.relpath(os.path.join(dp, f), folder) for dp, _, fs in os.walk(folder) for f in fs)
    build(); t0 = time.time()
    with ThreadPoolExecutor(jobs) as ex: out = list(ex.map(lambda p: _pack_one(os.path.join(folder, p), preset), files))
    man = [{"path": p, "kind": k, "size": os.path.getsize(os.path.join(folder, p)), "bytes": len(b), **({"note": n} if n else {})} for p, (k, b, n) in zip(files, out)]
    js = json.dumps({"files": man}, separators=(",", ":")).encode()
    with open(dst, "wb") as f:
        f.write(b"FFA1" + struct.pack("<I", len(js)) + js)
        for _, b, _ in out: f.write(b)
    return man, time.time() - t0


def unpack(src, folder, jobs=4):
    import lzma
    data = open(src, "rb").read()
    if data[:4] != b"FFA1": sys.exit(f"{src}: not an FFIC folder archive")
    n = struct.unpack("<I", data[4:8])[0]; man = json.loads(data[8:8 + n])["files"]; pos = 8 + n; t0 = time.time(); jobs_ = []
    for m in man:
        blob = data[pos:pos + m["bytes"]]; pos += m["bytes"]; out = os.path.join(folder, m["path"])
        if os.path.isabs(m["path"]) or ".." in m["path"].split(os.sep): sys.exit(f"unsafe path in archive: {m['path']}")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True); jobs_.append((m, blob, out))
    def one(j):
        m, blob, out = j
        if m["kind"] == "file": open(out, "wb").write(lzma.decompress(blob)); return
        with tempfile.TemporaryDirectory() as t:
            open(f"{t}/x.ffic", "wb").write(blob)
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "decode", f"{t}/x.ffic", out], capture_output=True, text=True)
            if r.returncode: sys.exit(f"{m['path']}: {r.stderr.strip()}")
    with ThreadPoolExecutor(jobs) as ex: list(ex.map(one, jobs_))
    return man, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode"); e.add_argument("src"); e.add_argument("dst"); e.add_argument("--preset", choices=PRESETS, default="max")
    d = sub.add_parser("decode"); d.add_argument("src"); d.add_argument("dst")
    i = sub.add_parser("info"); i.add_argument("src")
    pk = sub.add_parser("pack"); pk.add_argument("folder"); pk.add_argument("dst"); pk.add_argument("--preset", choices=PRESETS, default="max"); pk.add_argument("--jobs", type=int, default=4)
    up = sub.add_parser("unpack"); up.add_argument("src"); up.add_argument("folder"); up.add_argument("--jobs", type=int, default=4)
    tt = sub.add_parser("test"); tt.add_argument("src", nargs="+"); tt.add_argument("--preset", choices=list(PRESETS) + ["all"], default="all")
    a = ap.parse_args()
    if a.cmd == "pack":
        man, s = pack(a.folder, a.dst, a.preset, a.jobs); tot = sum(m["size"] for m in man)
        print(f"{a.dst}: {len(man)} files ({sum(m['kind'] == 'image' for m in man)} images), {tot:,} -> {os.path.getsize(a.dst):,} bytes, {s:.0f} s")
        for m in man:
            if m.get("note"): print(f"  {m['path']}: stored as a plain file ({m['note']})")
        return
    if a.cmd == "unpack":
        man, s = unpack(a.src, a.folder, a.jobs); print(f"{a.folder}: {len(man)} files restored in {s:.0f} s, checksums ok"); return
    if a.cmd == "encode":
        s = encode(a.src, a.dst, a.preset); print(f"{a.dst}: {os.path.getsize(a.dst):,} bytes, {s:.1f} s, preset {a.preset}")
    elif a.cmd == "decode":
        s = decode(a.src, a.dst); print(f"{a.dst}: decoded in {s:.1f} s, checksums ok")
    elif a.cmd == "info":
        meta, parts = read_any(a.src)
        print(f"{a.src}: {os.path.getsize(a.src):,} bytes, {meta.get('kind')}" + (f" {meta.get('mode', '')}" if meta.get("mode") else ""))
        for n, b in parts.items(): h = header(b); print(f"  {n}: {h['version']} {h['width']}x{h['height']}, {len(b):,} bytes, options: {h['options']}")
    else:
        bad = 0
        with tempfile.TemporaryDirectory() as t:
            for src in a.src:
                ext = ".y4m" if src.lower().endswith(".y4m") else ".png"
                for p in (PRESETS if a.preset == "all" else [a.preset]):
                    f, o = f"{t}/x.ffic", f"{t}/x{ext}"
                    se = encode(src, f, p); sd = decode(f, o); ok = same(src, o); bad += not ok
                    print(f"{os.path.basename(src)} [{p}]: {os.path.getsize(src):,} -> {os.path.getsize(f):,} bytes, encode {se:.1f} s, decode {sd:.1f} s, {'exact' if ok else 'MISMATCH'}", flush=True)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
