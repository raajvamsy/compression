"""V8 — Raaj's follow-up: tiny loops, simple equations, best choice per piece.

A piece of the file is described by
    one of 8 simple equations  f(x) = x+a | x^a | a-x | a·x | 5x+a | x²+a | a·x+1 | rotl(x)^a   (3 bits)
    a parameter a                                                                              (8 bits)
    a start value x0                                                                           (8 bits)
    a single-digit loop count L = 2..9                                                         (3 bits)
and prints x0, f(x0), f(f(x0)), ... (L bytes). That is 22 bits (+1 flag) for up to 72 bits of file.
Where no equation fits, the byte is stored as a literal (flag + 8 bits).
The encoder tries all 524,288 equations at every position and keeps the longest match.
Sub-equations: the token stream of one level can be fed to the same encoder again (levels > 1).
"""
from __future__ import annotations

import numpy as np

from .bits import BitReader, BitWriter, get_varint, put_varint
from .core import Encoded

ID = "v8-smallloop"
NAME = "V8 Small loops (L≤9) + simple equations"
SUMMARY = "8 simple equations, 1-digit loop count, best one per piece, literal fallback"
MAX_INPUT = 1 << 19
EQ_NAMES = ["x+a", "x^a", "a-x", "a·x", "5x+a", "x²+a", "a·x+1", "rotl(x)^a"]
LMIN, LMAX = 2, 9
_T: np.ndarray | None = None
_KEYS: np.ndarray | None = None
_ORDER: np.ndarray | None = None


def step(eq: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = x.astype(np.int64)
    a = a.astype(np.int64)
    out = np.select(
        [eq == 0, eq == 1, eq == 2, eq == 3, eq == 4, eq == 5, eq == 6],
        [x + a, x ^ a, a - x, x * a, 5 * x + a, x * x + a, a * x + 1],
        default=((x << 1) | (x >> 7)) ^ a,
    )
    return (out & 0xFF).astype(np.uint8)


def table() -> np.ndarray:
    """T[d] = first 9 bytes printed by description d = eq·65536 + a·256 + x0."""
    global _T, _KEYS, _ORDER
    if _T is None:
        d = np.arange(8 << 16)
        eq, a, x = d >> 16, (d >> 8) & 0xFF, (d & 0xFF).astype(np.uint8)
        cols = [x]
        for _ in range(LMAX - 1):
            x = step(eq, a, x)
            cols.append(x)
        _T = np.stack(cols, 1)
        keys = (_T[:, 0].astype(np.int64) << 16) | (_T[:, 1].astype(np.int64) << 8) | _T[:, 2]
        _ORDER = np.argsort(keys, kind="stable")
        _KEYS = keys[_ORDER]
    return _T


def _best_at(data: np.ndarray, i: int) -> tuple[int, int]:
    """(description, length) of the longest equation matching data[i:], length 0 if none >= 3."""
    T = table()
    if i + 3 > len(data):
        return 0, 0
    key = (int(data[i]) << 16) | (int(data[i + 1]) << 8) | int(data[i + 2])
    lo, hi = np.searchsorted(_KEYS, key), np.searchsorted(_KEYS, key, side="right")
    if lo == hi:
        return 0, 0
    cand = _ORDER[lo:hi]
    m = min(LMAX, len(data) - i)
    eqm = T[cand, :m] == data[i:i + m]
    lens = np.where(eqm.all(1), m, eqm.argmin(1))
    k = int(lens.argmax())
    return int(cand[k]), int(lens[k])


def encode_level(data: bytes) -> tuple[bytes, dict]:
    arr = np.frombuffer(data, dtype=np.uint8)
    bw, i, n = BitWriter(), 0, len(arr)
    used, covered, pieces = [0] * 8, 0, []
    while i < n:
        d, L = _best_at(arr, i)
        if L >= 3:
            bw.write(1, 1)
            bw.write(d >> 16, 3)
            bw.write((d >> 8) & 0xFF, 8)
            bw.write(d & 0xFF, 8)
            bw.write(L - LMIN, 3)
            used[d >> 16] += 1
            covered += L
            if len(pieces) < 10:
                pieces.append((i, d, L))
            i += L
        else:
            bw.write(0, 1)
            bw.write(int(arr[i]), 8)
            i += 1
    out = bytearray()
    put_varint(out, n)
    out += bw.to_bytes()
    return bytes(out), {"covered": covered, "used": used, "pieces": pieces}


def decode_level(payload: bytes) -> bytes:
    n, pos = get_varint(payload, 0)
    br, out = BitReader(payload[pos:]), bytearray()
    while len(out) < n:
        if br.read(1):
            eq, a, x, L = br.read(3), br.read(8), br.read(8), br.read(3) + LMIN
            e, aa = np.array([eq]), np.array([a])
            xv = np.array([x], dtype=np.uint8)
            for _ in range(L):
                out.append(int(xv[0]))
                xv = step(e, aa, xv)
        else:
            out.append(br.read(8))
    return bytes(out)


def _describe(d: int, L: int) -> str:
    eq, a, x0 = d >> 16, (d >> 8) & 0xFF, d & 0xFF
    return f"f(x) = {EQ_NAMES[eq].replace('a', str(a)) if 'a' in EQ_NAMES[eq] else EQ_NAMES[eq]}, x0 = {x0}, L = {L}"


def encode(data: bytes, levels: int = 1) -> Encoded:
    cur, sizes, info0 = data, [len(data)], None
    for _ in range(levels):
        cur, info = encode_level(cur)
        info0 = info0 or info
        sizes.append(len(cur))
    out = bytearray()
    put_varint(out, levels)
    out += cur
    lines = [f"{info0['covered']:,} of {len(data):,} bytes printed by equations "
             f"({100 * info0['covered'] / max(1, len(data)):.1f}%), the rest stored as literals"]
    for i, d, L in info0["pieces"][:8]:
        lines.append(f"  @{i}: {_describe(d, L)}  -> {data[i:i + L]!r}")
    lines.append("equations used: " + ", ".join(f"{EQ_NAMES[k]}×{c}" for k, c in enumerate(info0["used"]) if c))
    if levels > 1:
        lines.append("sizes per level (bytes): " + " -> ".join(f"{s:,}" for s in sizes))
    return Encoded(bytes(out), "\n".join(lines), {"covered": info0["covered"], "level_sizes": sizes})


def decode(payload: bytes) -> bytes:
    levels, pos = get_varint(payload, 0)
    cur = payload[pos:]
    for _ in range(levels):
        cur = decode_level(cur)
    return cur
