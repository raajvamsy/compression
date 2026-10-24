"""V5b — segment equations.

The file is covered by pieces, each one an equation over its position i:
    AP    f(i) = (a + b·i) mod 256           (runs, counters, gradients)
    COPY  f(i) = file[i - dist]              (repeats of earlier pieces)
    LIT   f(i) = literal byte
Greedy parse; tag = varint((len << 2) | op).
"""
from __future__ import annotations

from .bits import get_varint, put_varint
from .core import Encoded

ID = "v5-symbolic"
NAME = "V5b Segment equations a+b·i / repeat"
SUMMARY = "Cover the file with arithmetic-progression, repeat and literal pieces"
MAX_INPUT = 1 << 20
LIT, AP, COPY = 0, 1, 2
MIN_AP, MIN_COPY, WINDOW, CHAIN = 5, 4, 1 << 16, 24


def _ap_len(d: bytes, i: int) -> tuple[int, int]:
    if i + 1 >= len(d):
        return 1, 0
    b = (d[i + 1] - d[i]) & 0xFF
    j, v = i + 1, d[i + 1]
    while j + 1 < len(d) and d[j + 1] == (v + b) & 0xFF:
        j += 1
        v = d[j]
    return j - i + 1, b


def _parse(d: bytes):
    ops, lits, heads = [], bytearray(), {}
    i, n = 0, len(d)

    def flush():
        if lits:
            ops.append((LIT, bytes(lits)))
            lits.clear()

    while i < n:
        ap_len, b = _ap_len(d, i)
        best_cp, dist = 0, 0
        if i + 3 < n:
            key = d[i:i + 3]
            for p in reversed(heads.get(key, [])[-CHAIN:]):
                if i - p > WINDOW:
                    break
                k = 0
                while i + k < n and d[p + k] == d[i + k]:
                    k += 1
                if k > best_cp:
                    best_cp, dist = k, i - p
        if ap_len >= MIN_AP and ap_len >= best_cp:
            flush()
            ops.append((AP, d[i], b, ap_len))
            step = ap_len
        elif best_cp >= MIN_COPY:
            flush()
            ops.append((COPY, dist, best_cp))
            step = best_cp
        else:
            lits.append(d[i])
            step = 1
        for k in range(i, min(i + step, n - 2)):
            heads.setdefault(d[k:k + 3], []).append(k)
        i += step
    flush()
    return ops


def encode(data: bytes) -> Encoded:
    ops = _parse(data)
    out = bytearray()
    lines, pos = [], 0
    for op in ops:
        if op[0] == LIT:
            put_varint(out, (len(op[1]) << 2) | LIT)
            out += op[1]
            ln, desc = len(op[1]), f"literal {op[1][:16]!r}" + ("…" if len(op[1]) > 16 else "")
        elif op[0] == AP:
            _, a, b, ln = op
            put_varint(out, (ln << 2) | AP)
            out += bytes([a, b])
            desc = f"f(i) = ({a} + {b}·i) mod 256" if b else f"f(i) = {a}  (run)"
        else:
            _, dist, ln = op
            put_varint(out, (ln << 2) | COPY)
            put_varint(out, dist)
            desc = f"f(i) = file[i - {dist}]  (repeat)"
        if len(lines) < 14:
            lines.append(f"i in [{pos}, {pos + ln}):  {desc}")
        pos += ln
    if len(ops) > 14:
        lines.append(f"… {len(ops)} pieces in total")
    kinds = {k: sum(1 for o in ops if o[0] == v) for k, v in (("lit", LIT), ("ap", AP), ("copy", COPY))}
    return Encoded(bytes(out), "\n".join(lines), {"pieces": len(ops), **kinds})


def decode(payload: bytes) -> bytes:
    out, pos = bytearray(), 0
    while pos < len(payload):
        tag, pos = get_varint(payload, pos)
        op, ln = tag & 3, tag >> 2
        if op == LIT:
            out += payload[pos:pos + ln]
            pos += ln
        elif op == AP:
            a, b = payload[pos], payload[pos + 1]
            pos += 2
            out += bytes((a + b * i) & 0xFF for i in range(ln))
        else:
            dist, pos = get_varint(payload, pos)
            start = len(out) - dist
            for k in range(ln):
                out.append(out[start + k])
    return bytes(out)
