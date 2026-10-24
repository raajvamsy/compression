"""Spatial layers in parallel: each image cut into K horizontal strips, each strip coded independently (one core each).
Records a standard experiment entry: bytes = sum of strips (+4 bytes per strip for its length), ms = slowest strip (wall time
with K cores). usage: strips.py DATADIR K ID"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import exp
from experiments import EXPERIMENTS
D, K, eid = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SD = os.path.join(os.path.dirname(D.rstrip("/")), f"ffic_strips{K}")
cfg = {e["id"]: e["cfg"] for e in EXPERIMENTS}["best_now"]
ref = json.load(open(os.path.join(D, "ref.json"))); sref = json.load(open(os.path.join(SD, "ref.json")))
exp.build(); fp, head = exp.fingerprint()
def run(sn):
    r = sref[sn]; out = subprocess.check_output([exp.EXE, f"{SD}/{sn}.rgb", str(r["w"]), str(r["h"])] + cfg.split(), text=True)
    return {k: float(v) for k, v in (t.split("=") for t in out.split() if "=" in t and t.split("=")[1].replace(".", "").replace("-", "").isdigit())}
with ThreadPoolExecutor(4) as pool: res = dict(zip(sref, pool.map(run, sref)))
per = {}
for n in ref:
    parts = [res[f"{n}__{k}"] for k in range(K)]
    per[n] = {"bytes": sum(p["bytes"] for p in parts) + 4 * K, "R": int(sum(p["R"] for p in parts)), "G": int(sum(p["G"] for p in parts)), "B": int(sum(p["B"] for p in parts)),
              "ms": max(p["ms"] for p in parts), "cpu_ms": sum(p["ms"] for p in parts)}
fz = exp.funnelzip_sizes(D); tot = sum(v["bytes"] for v in per.values()); jxl = sum(r["jxl9"] for r in ref.values()); pex = [n for n in ref if n in fz]
part = lambda p: sum(v["bytes"] for n, v in per.items() if n.startswith(p)) / sum(ref[n]["jxl9"] for n in ref if n.startswith(p))
rec = {"id": eid, "group": "layers", "mechanism": f"Spatial layers in parallel: each photo cut into {K} horizontal strips, each coded independently with the best single coder on its own core.",
       "funnelbrain": "Parallel worlds read as parallel layers: independent regions processed at once, combined afterwards.",
       "hypothesis": f"About {K}x less wall time; size grows because each strip relearns from zero.", "cfg": cfg + f"   strips={K}", "data": f"ffic_strips{K}",
       "code_sha1": fp, "git_head": head, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": 0,
       "cpu_s": round(sum(v["ms"] for v in per.values()) / 1000, 1), "total_bytes": int(tot), "vs_jxl9_pct": round(100 * (tot / jxl - 1), 3),
       "pexels_bytes": int(sum(per[n]["bytes"] for n in pex)), "vs_funnelzip_pct": round(100 * (sum(per[n]["bytes"] for n in pex) / sum(fz[n] for n in pex) - 1), 3),
       "kodak_vs_jxl9_pct": round(100 * (part("kod") - 1), 3), "pexels_vs_jxl9_pct": round(100 * (part("pex") - 1), 3),
       "per_image": {n: {"bytes": int(v["bytes"]), "R": v["R"], "G": v["G"], "B": v["B"], "ms": int(v["ms"]), "vs_jxl9_pct": round(100 * (v["bytes"] / ref[n]["jxl9"] - 1), 2),
                         "stats": {"cpu_ms_all_strips": int(v["cpu_ms"])}} for n, v in per.items()}, "planes": {p: sum(v[p] for v in per.values()) for p in "RGB"}}
with open(exp.LOG, "a") as f: f.write(json.dumps(rec) + "\n")
print(f"{eid:22} {int(tot):>11,} B  {rec['vs_jxl9_pct']:+6.2f}% vs JXL   wall (slowest strip) {rec['cpu_s']:.1f} s over 20 images")
exp.report()
