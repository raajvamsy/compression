"""EquaFile CLI.

  python -m equafile show   <file>                 every variant + existing compressors
  python -m equafile encode <file> [-v ID] [-o X]  store the best (or chosen) equation as .eqf
  python -m equafile decode <file.eqf> [-o X]      rebuild the file from its equation
  python -m equafile loop   <text>                 Raaj's loop on a short text, step by step
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import baselines, loop
from .core import pack, registry, run, stored_size, unpack


def cmd_show(a) -> None:
    data = Path(a.file).read_bytes()
    print(f"{a.file}: {len(data):,} bytes\n")
    print(f"{'method':44} {'stored':>10} {'ratio':>7}  check")
    for v in registry().values():
        r = run(v, data)
        if r["status"] in ("ok", "MISMATCH"):
            print(f"{v.NAME:44} {r['stored']:>10,} {r['ratio']:>7.3f}  {r['status']}")
        else:
            print(f"{v.NAME:44} {'-':>10} {'':>7}  {r['status']}: {r['reason']}")
    for k, n in baselines.lossless(data).items():
        print(f"{'[existing] ' + k:44} {n:>10,} {n / max(1, len(data)):>7.3f}")
    if a.equations:
        for v in registry().values():
            r = run(v, data)
            if r.get("equation"):
                print(f"\n--- {v.NAME}\n{r['equation']}")


def cmd_encode(a) -> None:
    data = Path(a.file).read_bytes()
    variants = registry()
    pick = [variants[a.variant]] if a.variant else list(variants.values())
    best = None
    for v in pick:
        if v.MAX_INPUT is not None and len(data) > v.MAX_INPUT:
            continue
        try:
            enc = v.encode(data)
        except Exception as e:  # TooBig and friends
            print(f"{v.ID}: {e}", file=sys.stderr)
            continue
        if best is None or len(enc.payload) < len(best[1].payload):
            best = (v, enc)
    if best is None:
        sys.exit("no variant could encode this file")
    v, enc = best
    out = Path(a.output or a.file + ".eqf")
    out.write_bytes(pack(v.ID, enc.payload))
    print(f"{v.NAME}: {len(data):,} -> {stored_size(enc.payload):,} bytes ({out})\n\n{enc.equation}")


def cmd_decode(a) -> None:
    vid, payload = unpack(Path(a.file).read_bytes())
    data = registry()[vid].decode(payload)
    out = Path(a.output or a.file.removesuffix(".eqf") + ".out")
    out.write_bytes(data)
    print(f"{vid}: rebuilt {len(data):,} bytes -> {out}")


def cmd_loop(a) -> None:
    data = a.text.encode()
    n = len(data) * 8
    if n > 24:
        sys.exit(f"{n} bits would need ~2^{n} loop steps; try 1-3 characters")
    target = int.from_bytes(data, "big")
    print(f"file {a.text!r} = {target:0{n}b} ({n} bits)\nx0 = 1, x <- splitmix64(x) ...")
    L, head = loop.loop_search(target, n, trace=12)
    for step, val in head:
        print(f"  L={step:<8} top bits {val:0{n}b}")
    print(f"  ...\n  L={L:<8} top bits {loop.loop_eval(L, n):0{n}b}  <- match")
    print(f"\nstored loop count L = {L:,} needs {L.bit_length()} bits; the file has {n} bits")


def main() -> None:
    p = argparse.ArgumentParser(prog="equafile", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show"); s.add_argument("file"); s.add_argument("--equations", action="store_true")
    s = sub.add_parser("encode"); s.add_argument("file"); s.add_argument("-v", "--variant"); s.add_argument("-o", "--output")
    s = sub.add_parser("decode"); s.add_argument("file"); s.add_argument("-o", "--output")
    s = sub.add_parser("loop"); s.add_argument("text")
    a = p.parse_args()
    {"show": cmd_show, "encode": cmd_encode, "decode": cmd_decode, "loop": cmd_loop}[a.cmd](a)


if __name__ == "__main__":
    main()
