"""Parallel worlds experiments: run complete coders ("worlds") with per-pixel cost dumps, mix them exactly
(worlds_mix.c), append each mixture to experiments/log.jsonl like any other experiment.
usage: python worlds.py DATADIR"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import exp
from experiments import EXPERIMENTS
HERE = os.path.dirname(os.path.abspath(__file__)); D = sys.argv[1]
CACHE = os.path.join(os.path.dirname(D.rstrip("/")), "worlds_cache"); os.makedirs(CACHE, exist_ok=True)
cfg = {e["id"]: e["cfg"] for e in EXPERIMENTS}
best = cfg["matf_ctx"]
WORLDS = {
    "best": best,
    "old": cfg["ref_best"],
    "fastforget": best + " lam=0.99",
    "nofield": best.replace(" field=1 merge=0.08 kres=1 sleep=16 fmax=128", ""),
    "salonly": cfg["sal_p3"],
}
ref = json.load(open(os.path.join(D, "ref.json")))
fp, head = exp.fingerprint()
cpu = {}
def run(w, n):
    out = os.path.join(CACHE, f"{fp}_{w}_{n}.cost")
    r = ref[n]
    if not os.path.exists(out):
        env = dict(os.environ, FFIC_PIXCOST=out)
        res = subprocess.check_output([exp.EXE, f"{D}/{n}.rgb", str(r["w"]), str(r["h"])] + WORLDS[w].split(), text=True, env=env)
        ms = float(res.split("ms=")[1].split()[0]); open(out + ".ms", "w").write(str(ms))
    return float(open(out + ".ms").read())
exp.build()
with ThreadPoolExecutor(4) as pool:
    futs = {(w, n): pool.submit(run, w, n) for w in WORLDS for n in ref}
    for k, f in futs.items(): cpu[k] = f.result()
MIXES = [
    ("worlds2_bayes",  ["best", "old"], "bayes", 0.002, 1.0),
    ("worlds2_fs",     ["best", "old"], "fixedshare", 0.002, 1.0),
    ("worlds2_local",  ["best", "old"], "local_fs", 0.002, 0.5),
    ("worlds3_local",  ["best", "old", "fastforget"], "local_fs", 0.002, 0.5),
    ("worlds5_bayes",  list(WORLDS), "bayes", 0.002, 0.5),
    ("worlds5_fs",     list(WORLDS), "fixedshare", 0.002, 0.5),
    ("worlds5_local",  list(WORLDS), "local", 0.002, 0.5),
    ("worlds5_localfs", list(WORLDS), "local_fs", 0.002, 0.5),
    ("worlds5_localfs_b1", list(WORLDS), "local_fs", 0.002, 1.0),
]
for mid, ws, kind, alpha, beta in MIXES:
    per = {}
    for n, r in ref.items():
        out = subprocess.check_output([os.path.join(HERE, "worlds_mix"), str(r["w"]), str(r["h"]), str(alpha), str(beta)] +
                                      [os.path.join(CACHE, f"{fp}_{w}_{n}.cost") for w in ws], text=True).split()
        vals = dict(zip(out[1 + len(ws)::2], map(float, out[2 + len(ws)::2])))
        per[n] = {"bytes": vals[kind], "R": 0, "G": 0, "B": 0, "ms": sum(cpu[(w, n)] for w in ws),
                  "single": dict(zip(ws, map(float, out[1:1 + len(ws)])))}
    e = dict(id=mid, group="parallel worlds", cfg=" | ".join(WORLDS[w] for w in ws) + f"   mix={kind} alpha={alpha} beta={beta}",
             mechanism=f"Parallel worlds: {len(ws)} complete coders ({', '.join(ws)}) run side by side; each pixel is coded with their "
                       f"weighted mixture ({kind}: " + {"bayes": "weights = cumulative evidence", "fixedshare": f"evidence with fixed-share switching alpha={alpha}",
                       "local": f"weights from each world's cost on the W,N,NW,NE pixels, beta={beta}", "local_fs": f"local (beta={beta}) x fixed-share (alpha={alpha})"}[kind] + ").",
             funnelbrain="Many-worlds / Born rule: every branch exists with a weight; evidence updates the weights; decoherence = branches stop interfering locally.",
             hypothesis="Worlds that are right in different places add up; the mixture beats the best single world by a margin that grows with diversity.")
    # record in the standard format
    fz = exp.funnelzip_sizes(D); tot = sum(v["bytes"] for v in per.values()); jxl = sum(r["jxl9"] for r in ref.values())
    pex = [n for n in ref if n in fz]
    rec = {"id": mid, "group": e["group"], "mechanism": e["mechanism"], "funnelbrain": e["funnelbrain"], "hypothesis": e["hypothesis"], "cfg": e["cfg"],
           "data": "ffic_data", "code_sha1": fp, "git_head": head, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": 0,
           "cpu_s": round(sum(v["ms"] for v in per.values()) / 1000, 1), "total_bytes": int(tot), "vs_jxl9_pct": round(100 * (tot / jxl - 1), 3),
           "pexels_bytes": int(sum(per[n]["bytes"] for n in pex)), "vs_funnelzip_pct": round(100 * (sum(per[n]["bytes"] for n in pex) / sum(fz[n] for n in pex) - 1), 3),
           "kodak_vs_jxl9_pct": round(100 * (sum(per[n]["bytes"] for n in ref if n.startswith("kod")) / sum(ref[n]["jxl9"] for n in ref if n.startswith("kod")) - 1), 3),
           "pexels_vs_jxl9_pct": round(100 * (sum(per[n]["bytes"] for n in ref if n.startswith("pex")) / sum(ref[n]["jxl9"] for n in ref if n.startswith("pex")) - 1), 3),
           "per_image": {n: {"bytes": int(v["bytes"]), "R": 0, "G": 0, "B": 0, "ms": int(v["ms"]), "vs_jxl9_pct": round(100 * (v["bytes"] / ref[n]["jxl9"] - 1), 2),
                             "stats": {"best_single_world_bytes": int(min(v["single"].values()))}} for n, v in per.items()},
           "planes": {"R": 0, "G": 0, "B": 0}}
    with open(exp.LOG, "a") as f: f.write(json.dumps(rec) + "\n")
    single = {w: sum(v["single"][w] for v in per.values()) for w in ws}
    print(f"{mid:22} {int(tot):>11,} B  {rec['vs_jxl9_pct']:+6.2f}% vs JXL   worlds: " + "  ".join(f"{w} {int(b):,}" for w, b in single.items()), flush=True)
exp.report()
