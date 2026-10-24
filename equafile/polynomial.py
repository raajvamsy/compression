"""V1 — one direct equation: byte_i = f(i), f a polynomial over GF(p).

Exactly one polynomial of degree < n passes through n points, so the equation
always exists. Over GF(p) there are only p distinct x values, so the field must
grow with the file: p = 257 (9-bit coefficients) up to 257 bytes, then
p = 65537 (17-bit coefficients). Either way: n coefficients for n bytes.
"""
from __future__ import annotations

import numpy as np

from .bits import BitReader, BitWriter, get_varint, put_varint
from .core import Encoded

ID = "v1-poly"
NAME = "V1 One polynomial f(i)"
SUMMARY = "byte_i = f(i) with one polynomial mod p (Newton form)"
MAX_INPUT = 16384  # O(n^2) fit and evaluation


def _field(n: int) -> tuple[int, int]:
    return (257, 9) if n <= 257 else (65537, 17)


def _newton_coeffs(data: bytes, P: int) -> np.ndarray:
    c = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    for j in range(1, len(c)):
        inv_j = pow(j, P - 2, P)
        c[j:] = ((c[j:] - c[j - 1:-1]) % P) * inv_j % P
    return c


def _eval(c: np.ndarray, P: int) -> np.ndarray:
    n = len(c)
    x = np.arange(n, dtype=np.int64)
    v = np.full(n, c[-1], dtype=np.int64)
    for k in range(n - 2, -1, -1):
        v = (v * ((x - k) % P) + c[k]) % P
    return v


def _monomial(c: list[int], P: int) -> list[int]:
    """Newton form -> plain coefficients a0 + a1 x + ... (only used for display)."""
    poly = [0]
    for k in range(len(c) - 1, -1, -1):
        # poly = poly * (x - k) + c[k]
        new = [0] * (len(poly) + 1)
        for i, a in enumerate(poly):
            new[i + 1] = (new[i + 1] + a) % P
            new[i] = (new[i] - a * k) % P
        new[0] = (new[0] + c[k]) % P
        poly = new
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly


def _show(c: np.ndarray, P: int, bits: int) -> str:
    if len(c) <= 24:
        terms = []
        for i, a in enumerate(_monomial([int(v) for v in c], P)):
            if a == 0 and i:
                continue
            terms.append(str(a) if i == 0 else f"{a}·x" if i == 1 else f"{a}·x^{i}")
        return f"f(x) = {' + '.join(terms)}   (mod {P}),  x = 0..{len(c) - 1}"
    basis = ["", "·x", "·x(x-1)", "·x(x-1)(x-2)"]
    head = " + ".join(f"{int(c[k])}{basis[k]}" for k in range(4))
    return (f"f(x) = {head} + … ({len(c)} Newton terms, mod {P}),  x = 0..{len(c) - 1}\n"
            f"one coefficient per byte, {bits} bits each -> {bits / 8:.3f}x the file")


def encode(data: bytes) -> Encoded:
    P, bits = _field(len(data))
    c = _newton_coeffs(data, P) if data else np.zeros(0, dtype=np.int64)
    out = bytearray()
    put_varint(out, len(data))
    bw = BitWriter()
    for v in c:
        bw.write(int(v), bits)
    out += bw.to_bytes()
    return Encoded(bytes(out), _show(c, P, bits), {"terms": len(c), "bits_per_term": bits, "prime": P})


def decode(payload: bytes) -> bytes:
    n, pos = get_varint(payload, 0)
    P, bits = _field(n)
    br = BitReader(payload[pos:])
    c = np.array([br.read(bits) for _ in range(n)], dtype=np.int64)
    return bytes(_eval(c, P).astype(np.uint8)) if n else b""
