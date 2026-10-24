"""Run ffic with flag sets over the test set; compare with JXL e9 (+FunnelZip where cached). usage: bench.py DATADIR cfg1 cfg2 ...  (cfg = 'none' | letters | 'letters:mu=0.02,K=32')"""
import json, re, subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor
D = sys.argv[1]; cfgs = sys.argv[2:] or ["none", "ltvpc"]
ref = json.load(open(D + "/ref.json"))
exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffic")
# FunnelZip numbers from the stopped benchmark log (pexels only)
fz = {}
try:
    cur = None
    for line in open(D + "/../pexels.log"):
        m = re.match(r"== (p\d+)\.jpg", line)
        if m: cur = "pex_" + m.group(1)
        m = re.search(r"FunnelZip \(raw pixels\)\s+([\d,]+) B", line)
        if m and cur: fz[cur] = int(m.group(1).replace(",", ""))
except FileNotFoundError: pass
def run(name, cfg):
    parts = cfg.split(":"); args = [parts[0]] + (parts[1].split(",") if len(parts) > 1 else [])
    r = ref[name]; out = subprocess.check_output([exe, f"{D}/{name}.rgb", str(r["w"]), str(r["h"])] + args, text=True)
    return {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", out)}
tot = {c: [0, 0.0] for c in cfgs}; sub = {c: 0 for c in cfgs}; tj = 0; tf = 0; nf = 0; tjf = 0
QUIET = os.environ.get("QUIET")
pool = ThreadPoolExecutor(4); futs = {(n, c): pool.submit(run, n, c) for n in ref for c in cfgs}
if not QUIET: print(f"{'image':14}{'JXL9':>9}{'FunnelZip':>10}" + "".join(f"{c[:14]:>16}" for c in cfgs))
for name in ref:
    row = f"{name:14}{ref[name]['jxl9']:>9,}{fz.get(name, 0) or '':>10}"
    tj += ref[name]["jxl9"]
    if name in fz: tf += fz[name]; nf += 1; tjf += ref[name]["jxl9"]
    for c in cfgs:
        r = futs[(name, c)].result(); tot[c][0] += r["bytes"]; sub[c] += r["bytes"] if name in fz else 0; tot[c][1] += r["ms"]; row += f"{int(r['bytes']):>12,} {100 * (r['bytes'] / ref[name]['jxl9'] - 1):+4.0f}%"
    if not QUIET: print(row)
print(f"\nTOTAL JXL e9 = {tj:,} B" + (f"; FunnelZip on the {nf} pexels images = {tf:,} vs JXL {tjf:,} ({100 * (tf / tjf - 1):+.1f}%)" if nf else ""))
for c in cfgs: print(f"  {c:24} {int(tot[c][0]):>10,} B  {100 * (tot[c][0] / tj - 1):+6.1f}% vs JXL e9   {tot[c][1] / 1000:6.1f} s total   pexels-11: {int(sub[c]):,} ({100 * (sub[c] / tf - 1):+.1f}% vs FunnelZip)")
