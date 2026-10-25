"""Formats benchmark (real files, every output decoded and compared exactly).
    python benchformats.py IMGDIR CLIPDIR   -> experiments/BENCHFORMATS.md + benchformats.json
IMGDIR: test images (transparency, palette); CLIPDIR: Xiph CIF .y4m clips (first 30 frames are used, native YUV 4:2:0)."""
import glob, io, json, os, subprocess, sys, tempfile, time
import numpy as np
from PIL import Image
import pillow_jxl  # noqa: F401
import ffic

HERE = os.path.dirname(os.path.abspath(__file__))


def ours(src, ext, preset="max"):
    with tempfile.TemporaryDirectory() as t:
        f, o = f"{t}/x.ffic", f"{t}/x{ext}"; se = ffic.encode(src, f, preset); sd = ffic.decode(f, o)
        return os.path.getsize(f), se, sd, ffic.same(src, o)


def img_others(src):
    im = Image.open(src); im.load(); out = {}
    b = io.BytesIO(); t = time.time(); im.save(b, "PNG", optimize=True); out["PNG (optimized)"] = (len(b.getvalue()), time.time() - t)
    j = im.convert("RGBA") if im.mode in ("P", "PA") and "transparency" in im.info else (im.convert("RGB") if im.mode == "P" else im)
    b = io.BytesIO(); t = time.time(); j.save(b, "JXL", lossless=True, effort=9); te = time.time() - t
    back = Image.open(io.BytesIO(b.getvalue())); back.load(); ok = back.convert(j.mode).tobytes() == j.tobytes()
    out["JPEG XL e9"] = (len(b.getvalue()), te, ok)
    return out


def vid_others(src):
    out = {}
    for name, args in [("FFV1", ["-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1", "-g", "1", "-slices", "4"]),
                       ("x264 lossless, veryslow", ["-c:v", "libx264", "-qp", "0", "-preset", "veryslow"]),
                       ("x265 lossless, veryslow", ["-c:v", "libx265", "-preset", "veryslow", "-x265-params", "lossless=1:log-level=error"])]:
        with tempfile.TemporaryDirectory() as t:
            f = f"{t}/o.mkv"; t0 = time.time()
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src] + args + ["-pix_fmt", "yuv420p", f], check=True); te = time.time() - t0
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f, "-f", "rawvideo", "-pix_fmt", "yuv420p", f"{t}/b.yuv"], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-f", "rawvideo", "-pix_fmt", "yuv420p", f"{t}/a.yuv"], check=True)
            out[name] = (os.path.getsize(f), te, open(f"{t}/a.yuv", "rb").read() == open(f"{t}/b.yuv", "rb").read())
    return out


def main():
    imgdir, clipdir = sys.argv[1], sys.argv[2]; ffic.build(); res = {"images": {}, "video": {}}
    L = ["# Formats benchmark (lossless, real files, all decoded and compared exactly)", ""]
    L += ["## Images with transparency and palette images", "", "| image | what | PNG (optimized) | JPEG XL e9 | FFIC max | FFIC vs best other |", "|---|---|---|---|---|---|"]
    for src in sorted(glob.glob(f"{imgdir}/*.png")):
        n = os.path.basename(src); im = Image.open(src); kind = im.mode + (" + transparency" if "transparency" in im.info else "")
        o = img_others(src); f = ours(src, ".png"); res["images"][n] = {"others": o, "ffic": f}
        best = min(v[0] for v in o.values())
        L.append(f"| {n} | {kind}, {im.width}x{im.height} | {o['PNG (optimized)'][0]:,} | {o['JPEG XL e9'][0]:,} | {f[0]:,}{'' if f[3] else ' MISMATCH'} | **{100 * (f[0] / best - 1):+.1f}%** |")
        print(L[-1], flush=True)
    L += ["", "## Video in its native format (YUV 4:2:0, first 30 frames of the Xiph CIF clips, no colour conversion)", "",
          "| clip | FFV1 | x264 lossless | x265 lossless | FFIC max | FFIC vs best other |", "|---|---|---|---|---|---|"]
    for clip in ("akiyo", "foreman", "mobile"):
        src = f"{clipdir}/{clip}_cif.y4m"
        with tempfile.TemporaryDirectory() as t:
            s30 = f"{t}/{clip}30.y4m"; subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-frames:v", "30", "-f", "yuv4mpegpipe", "-strict", "-1", s30], check=True)
            o = vid_others(s30); f = ours(s30, ".y4m")
        res["video"][clip] = {"others": o, "ffic": f}; best = min(v[0] for v in o.values())
        L.append(f"| {clip} | " + " | ".join(f"{o[k][0]:,} ({o[k][1]:.1f} s)" for k in o) + f" | {f[0]:,} ({f[1]:.0f} s / {f[2]:.0f} s){'' if f[3] else ' MISMATCH'} | **{100 * (f[0] / best - 1):+.1f}%** |")
        print(L[-1], flush=True)
    L += ["", "FFIC times are encode / decode seconds; other tools' times are encode seconds. FFIC video codes the Y, U and V layers in parallel (3 threads)."]
    json.dump(res, open(os.path.join(HERE, "experiments/benchformats.json"), "w"), indent=1)
    open(os.path.join(HERE, "experiments/BENCHFORMATS.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
