"""Any file as an image: lay a file's bytes out as a grey image (one byte = one pixel) and compress it with FFIC.
    python benchbytes.py FILEDIR    -> experiments/BENCHBYTES.md + benchbytes.json
For each file the encoder tries several widths: 256, 1024, and the file's own repeat length (the lag at which bytes most
often repeat, e.g. a record size or a row length), keeping the smallest. Also tried: reading the file as 16-bit numbers
(both byte orders) laid out the same way. Compared with zip (deflate -9), bzip2 -9, xz -9e, and FLAC for the WAV file.
The best FFIC layout of every file is decoded again and compared byte for byte."""
import bz2, glob, json, lzma, os, subprocess, sys, tempfile, zlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import ffic

HERE = os.path.dirname(os.path.abspath(__file__))


def repeat_len(a, maxlag=4096):
    n = min(len(a), 400000); x = a[:n]; best, bl = 0, 1
    for lag in range(2, min(maxlag, n // 4)):
        m = np.count_nonzero(x[lag:] == x[:-lag]) / (n - lag)
        if m > best * 1.02: best, bl = m, lag
    return bl


def layouts(data):
    b = np.frombuffer(data, np.uint8); out = []
    for w in sorted({256, 1024, repeat_len(b)}):
        h = -(-len(b) // w); img = np.zeros(h * w, np.uint8); img[:len(b)] = b; out.append((f"bytes, width {w}", img.reshape(h, w)))
    if len(b) % 2 == 0:
        for order in ("<", ">"):
            s = np.frombuffer(data, order + "u2").astype(np.uint16)
            for w in sorted({256, 1024, repeat_len(s.astype(np.int64) & 0xFFFF)}):
                h = -(-len(s) // w); img = np.zeros(h * w, np.uint16); img[:len(s)] = s
                out.append((f"16-bit {'little' if order == '<' else 'big'}-endian, width {w}", img.reshape(h, w)))
    return out


def main():
    src = sys.argv[1]; ffic.build(); res = {}
    L = ["# Any file as an image (lossless, real streams, best layout decoded and compared)", "",
         "Each file's bytes are laid out as a grey image (one byte = one pixel), or read as 16-bit numbers, at several widths; FFIC (preset max, one layer) keeps the smallest.", "",
         "| file | size | zip -9 | bzip2 -9 | xz -9e | FLAC | FFIC as an image | best layout | FFIC vs best other |", "|---|---|---|---|---|---|---|---|---|"]
    for f in sorted(glob.glob(f"{src}/*")):
        data = open(f, "rb").read(); name = os.path.basename(f)
        o = {"zip -9": len(zlib.compress(data, 9)), "bzip2 -9": len(bz2.compress(data, 9)), "xz -9e": len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME))}
        if name.endswith(".wav"):
            with tempfile.TemporaryDirectory() as t:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f, "-c:a", "flac", "-compression_level", "12", f"{t}/a.flac"], check=True); o["FLAC"] = os.path.getsize(f"{t}/a.flac")
        cand = layouts(data)
        with ThreadPoolExecutor(4) as ex: sizes = list(ex.map(lambda c: len(ffic.code_layer(c[1], ffic.PRESETS["max"])), cand))
        i = int(np.argmin(sizes)); blob = ffic.code_layer(cand[i][1], ffic.PRESETS["max"]); back = ffic.decode_layer(blob)
        ok = np.array_equal(back.astype(np.int64), cand[i][1].astype(np.int64))
        res[name] = {"size": len(data), "others": o, "ffic": sizes[i] + 8, "layout": cand[i][0], "all": dict(zip([c[0] for c in cand], sizes)), "exact": ok}
        best = min(o.values())
        L.append(f"| {name} | {len(data):,} | {o['zip -9']:,} | {o['bzip2 -9']:,} | {o['xz -9e']:,} | {o.get('FLAC', '—') if isinstance(o.get('FLAC'), str) else format(o['FLAC'], ',') if 'FLAC' in o else '—'} | "
                 f"{sizes[i] + 8:,}{'' if ok else ' MISMATCH'} | {cand[i][0]} | **{100 * ((sizes[i] + 8) / best - 1):+.1f}%** |")
        print(L[-1], flush=True)
    L += ["", "FFIC size includes 8 bytes for the original length and layout. 'best other' is the smallest of zip, bzip2, xz (and FLAC for audio)."]
    json.dump(res, open(os.path.join(HERE, "experiments/benchbytes.json"), "w"), indent=1)
    open(os.path.join(HERE, "experiments/BENCHBYTES.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
