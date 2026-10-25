"""Lossless video benchmark: 30 RGB frames per clip (352x288), FFIC frame mode vs FFV1 / x264 RGB lossless / x265 lossless.
   python benchvideo.py "<ffic options>" CLIPDIR   (CLIPDIR holds akiyo30.rgb, foreman30.rgb, mobile30.rgb: first 30 frames of the Xiph
   CIF clips as rgb24, e.g. ffmpeg -i akiyo_cif.y4m -frames:v 30 -pix_fmt rgb24 -f rawvideo akiyo30.rgb)
Every output is decoded and compared with the source frames byte for byte."""
import os, subprocess, sys, time, hashlib, json
V = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__)); W, H, T = 352, 288, 30
EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffic")
CFG = sys.argv[1]


def run(cmd): return subprocess.run(cmd, capture_output=True, text=True)


def ff_roundtrip(src, out, enc_args, pix):
    t = time.time(); r = run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", src] + enc_args + ["-pix_fmt", pix, out])
    te = time.time() - t
    if r.returncode: return None
    t = time.time(); back = out + ".rgb"; run(["ffmpeg", "-v", "error", "-y", "-i", out, "-f", "rawvideo", "-pix_fmt", "rgb24", back]); td = time.time() - t
    ok = open(src, "rb").read() == open(back, "rb").read(); os.remove(back)
    return os.path.getsize(out), te, td, ok


def ffic(src, out, extra):
    t = time.time(); run([EXE, src, str(W), str(H * T)] + CFG.split() + extra.split() + [f"out={out}"]); te = time.time() - t
    t = time.time(); run([EXE, "-d", out, out + ".rgb"]); td = time.time() - t
    ok = open(src, "rb").read() == open(out + ".rgb", "rb").read(); os.remove(out + ".rgb")
    return os.path.getsize(out), te, td, ok


res = {}
for clip in ("akiyo", "foreman", "mobile"):
    src = f"{V}/{clip}30.rgb"; row = {}
    row["FFV1 (archive standard)"] = ff_roundtrip(src, f"{V}/{clip}.ffv1.mkv", ["-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1", "-g", "1", "-slices", "4"], "gbrp")
    row["x264 RGB lossless, veryslow"] = ff_roundtrip(src, f"{V}/{clip}.x264.mkv", ["-c:v", "libx264rgb", "-qp", "0", "-preset", "veryslow"], "rgb24")
    row["x265 lossless, veryslow"] = ff_roundtrip(src, f"{V}/{clip}.x265.mkv", ["-c:v", "libx265", "-preset", "veryslow", "-x265-params", "lossless=1:log-level=error"], "gbrp")
    row["FFIC, frames independent"] = ffic(src, f"{V}/{clip}.still.ffic", "")
    row["FFIC video (frameh=288)"] = ffic(src, f"{V}/{clip}.video.ffic", f"frameh={H}")
    res[clip] = row; raw = W * H * 3 * T
    print(f"\n{clip}: raw {raw:,} B")
    for k, v in row.items():
        print(f"  {k:30} {'failed' if v is None else f'{v[0]:>10,} B  {8 * v[0] / (W * H * T):5.2f} bits/px  enc {v[1]:6.1f}s dec {v[2]:6.1f}s  ' + ('exact' if v[3] else 'NOT EXACT')}", flush=True)
json.dump(res, open(f"{V}/vidbench.json", "w"), indent=1)
