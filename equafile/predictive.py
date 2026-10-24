"""V6 — predictive equation: P(next bit = 1 | past) + binary arithmetic coding.

The "equation" is a model the decoder also runs: it mixes what followed the
last 0..4 bytes (logistic mixing, like PAQ/lpaq) and the arithmetic coder
spends ~ -log2(P) bits per bit. Good prediction == short code.
Pure Python, ~1-2 s per 100 KB.
"""
from __future__ import annotations

import math
from array import array

from .bits import get_varint, put_varint
from .core import Encoded

ID = "v6-predict"
NAME = "V6 Predictive equation (context mixing)"
SUMMARY = "Model P(next bit) from the last 0-4 bytes, arithmetic-code the surprise"
MAX_INPUT = 1 << 19
ORDERS = 5               # order 0..4
TBITS = 22
TMASK = (1 << TBITS) - 1
LR = 0.02
STRETCH = [math.log((i + 0.5) / (4095.5 - i)) for i in range(4096)]


class Model:
    def __init__(self) -> None:
        self.t = [array("H", [32768]) * (1 << TBITS) for _ in range(ORDERS)]
        self.w = [[0.3] * ORDERS for _ in range(256)]
        self.hist = 0      # last 4 bytes
        self.c0 = 1        # partial byte with leading 1
        self.base = [0] * ORDERS
        self.idx = [0] * ORDERS
        self.st = [0.0] * ORDERS
        self.pr = 0.5
        self._new_byte()

    def _new_byte(self) -> None:
        h = self.hist
        for k in range(ORDERS):
            ctx = h & ((1 << (8 * k)) - 1)
            x = (ctx * 0x9E3779B1 + (k + 1) * 0x7F4A7C15) & 0xFFFFFFFF
            x = ((x ^ (x >> 15)) * 0x2C1B3C6D) & 0xFFFFFFFF
            self.base[k] = (x ^ (x >> 12)) & TMASK & ~0xFF

    def p(self) -> int:
        c0, t, st, idx = self.c0, self.t, self.st, self.idx
        dot, w = 0.0, self.w[c0]
        for k in range(ORDERS):
            i = self.base[k] | c0
            idx[k] = i
            s = STRETCH[t[k][i] >> 4]
            st[k] = s
            dot += w[k] * s
        if dot > 20:
            dot = 20.0
        elif dot < -20:
            dot = -20.0
        self.pr = 1.0 / (1.0 + math.exp(-dot))
        p12 = int(self.pr * 4096)
        return 1 if p12 < 1 else 4095 if p12 > 4095 else p12

    def update(self, y: int) -> None:
        err = (y - self.pr) * LR
        w, st, t, idx = self.w[self.c0], self.st, self.t, self.idx
        target = 65535 if y else 0
        for k in range(ORDERS):
            w[k] += err * st[k]
            v = t[k][idx[k]]
            t[k][idx[k]] = v + ((target - v) >> 4)
        self.c0 = (self.c0 << 1) | y
        if self.c0 >= 256:
            self.hist = ((self.hist << 8) | (self.c0 & 0xFF)) & 0xFFFFFFFF
            self.c0 = 1
            self._new_byte()


def compress(data: bytes) -> bytes:
    m, out = Model(), bytearray()
    x1, x2 = 0, 0xFFFFFFFF
    for byte in data:
        for j in range(7, -1, -1):
            y = (byte >> j) & 1
            xmid = x1 + ((x2 - x1) >> 12) * m.p()
            if y:
                x2 = xmid
            else:
                x1 = xmid + 1
            m.update(y)
            while not ((x1 ^ x2) & 0xFF000000):
                out.append(x2 >> 24)
                x1 = (x1 << 8) & 0xFFFFFFFF
                x2 = ((x2 << 8) & 0xFFFFFFFF) | 0xFF
    out.append(x1 >> 24)  # decoder pads with 0xFF, which lands inside [x1, x2]
    return bytes(out)


def decompress(code: bytes, n: int) -> bytes:
    m, out = Model(), bytearray()
    x1, x2, pos = 0, 0xFFFFFFFF, 4
    x = int.from_bytes(code[:4].ljust(4, b"\xff"), "big")
    for _ in range(n):
        c = 0
        for _ in range(8):
            xmid = x1 + ((x2 - x1) >> 12) * m.p()
            y = 1 if x <= xmid else 0
            if y:
                x2 = xmid
            else:
                x1 = xmid + 1
            m.update(y)
            c = (c << 1) | y
            while not ((x1 ^ x2) & 0xFF000000):
                x1 = (x1 << 8) & 0xFFFFFFFF
                x2 = ((x2 << 8) & 0xFFFFFFFF) | 0xFF
                x = ((x << 8) & 0xFFFFFFFF) | (code[pos] if pos < len(code) else 0xFF)
                pos += 1
        out.append(c)
    return bytes(out)


def encode(data: bytes) -> Encoded:
    out = bytearray()
    put_varint(out, len(data))
    code = compress(data)
    out += code
    bpc = 8 * len(code) / max(1, len(data))
    eq = (f"P(bit=1 | last 0-4 bytes) = σ(Σₖ wₖ · logit(pₖ)),  {ORDERS} context tables, learned while reading\n"
          f"arithmetic code spends -log2 P per bit -> {bpc:.3f} bits per byte")
    return Encoded(bytes(out), eq, {"bits_per_byte": round(bpc, 4)})


def decode(payload: bytes) -> bytes:
    n, pos = get_varint(payload, 0)
    return decompress(payload[pos:], n)
