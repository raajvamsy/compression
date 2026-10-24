"""Reversible reshapes: put a file into the form where FunnelZip's lenses see the most structure,
and undo it exactly on decode (Raaj's "convert to a format where compression works best").

Pixels (interleaved, C channels):
  planar   RGBRGB... -> RRR...GGG...BBB...   (each colour becomes its own picture)
  rct      R-G, G, B-G (mod 256)              (lossless colour transform, as in WebP/PNG)
Text:
  caps     'The' -> <cap>'the'                (upper/lower case words look the same)
  words    common English words -> 2 bytes    (marker + index into a fixed word list)
Markers are byte values that do not occur in the file, so decoding is unambiguous.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

PRIME_PATH = Path(__file__).resolve().parents[1] / "funnelzip" / "prime.bin"


# ------------------------------------------------------------------ pixels
def planar(data: bytes, C: int, inverse: bool = False) -> bytes:
    n = len(data) // C * C
    a = np.frombuffer(data[:n], dtype=np.uint8)
    if not inverse:
        return a.reshape(-1, C).T.tobytes() + data[n:]
    return a.reshape(C, -1).T.tobytes() + data[n:]


def rct(data: bytes, C: int, inverse: bool = False) -> bytes:
    if C < 3:
        return data
    n = len(data) // C * C
    a = np.frombuffer(data[:n], dtype=np.uint8).reshape(-1, C).copy()
    g = a[:, 1].astype(np.int16)
    sign = 1 if inverse else -1
    a[:, 0] = (a[:, 0].astype(np.int16) + sign * g) & 0xFF
    a[:, 2] = (a[:, 2].astype(np.int16) + sign * g) & 0xFF
    return a.tobytes() + data[n:]


def med(data: bytes, S: int, C: int, inverse: bool = False) -> bytes:
    """Prediction residual: each byte minus the JPEG-LS (MED) prediction from left, up and corner."""
    import ctypes
    from . import funnelzip as FZ
    lib = FZ.lib()
    fn = lib.fz_med_inverse if inverse else lib.fz_med_forward
    fn.restype = None
    fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int]
    buf = ctypes.create_string_buffer(data, len(data))
    fn(buf, len(data), S, C)
    return buf.raw[:len(data)]


# ------------------------------------------------------------------ text
def unused(data: bytes, k: int) -> list[int] | None:
    present = set(np.unique(np.frombuffer(data, dtype=np.uint8)).tolist()) if data else set()
    free = [b for b in list(range(1, 9)) + list(range(14, 32)) + list(range(128, 256)) if b not in present]
    return free[:k] if len(free) >= k else None


@lru_cache(maxsize=1)
def word_list() -> list[bytes]:
    """The 255 most frequent words (length >= 3) of the shipped primer: fixed, so encoder and decoder agree."""
    text = PRIME_PATH.read_bytes().lower() if PRIME_PATH.exists() else b""
    counts = Counter(w for w in re.findall(rb"[a-z]{3,}", text))
    scored = sorted(counts.items(), key=lambda kv: (-(len(kv[0]) - 2) * kv[1], kv[0]))   # bytes saved
    return [w for w, _ in scored[:255]]


def caps(data: bytes, marker: int, inverse: bool = False) -> bytes:
    m = bytes([marker])
    if not inverse:
        return re.sub(rb"[A-Z]", lambda mo: m + mo.group(0).lower(), data)
    return re.sub(re.escape(m) + rb"([a-z])", lambda mo: mo.group(1).upper(), data)


def words(data: bytes, marker: int, inverse: bool = False) -> bytes:
    wl = word_list()
    m = bytes([marker])
    if not inverse:
        index = {w: i for i, w in enumerate(wl)}
        return re.sub(rb"(?<![a-z])[a-z]{3,}(?![a-z])",
                      lambda mo: m + bytes([index[mo.group(0)]]) if mo.group(0) in index else mo.group(0), data)
    return re.sub(re.escape(m) + rb"(.)", lambda mo: wl[mo.group(1)[0]], data, flags=re.DOTALL)


# ------------------------------------------------------------------ registry
TEXT_CAPS, TEXT_WORDS, TEXT_BOTH = 1, 2, 3
PIX_PLANAR, PIX_RCT, PIX_BOTH, PIX_MED = 4, 5, 6, 7
NAMES = {0: "none", 1: "caps", 2: "words", 3: "caps+words", 4: "planar", 5: "colour transform",
         6: "colour transform+planar", 7: "prediction residual"}


def forward(data: bytes, t: int, C: int = 3, S: int = 0) -> tuple[bytes, bytes]:
    """Returns (transformed data, parameters needed to invert)."""
    if t in (TEXT_CAPS, TEXT_WORDS, TEXT_BOTH):
        mk = unused(data, 2)
        if mk is None:
            raise ValueError("no free marker bytes")
        out = data
        if t in (TEXT_CAPS, TEXT_BOTH):
            out = caps(out, mk[0])
        if t in (TEXT_WORDS, TEXT_BOTH):
            out = words(out, mk[1])
        return out, bytes(mk)
    if t == PIX_PLANAR:
        return planar(data, C), bytes([C])
    if t == PIX_RCT:
        return rct(data, C), bytes([C])
    if t == PIX_BOTH:
        return planar(rct(data, C), C), bytes([C])
    if t == PIX_MED:
        return med(data, S, C), bytes([C]) + S.to_bytes(4, "little")
    return data, b""


def inverse(data: bytes, t: int, params: bytes) -> bytes:
    if t in (TEXT_CAPS, TEXT_WORDS, TEXT_BOTH):
        out = data
        if t in (TEXT_WORDS, TEXT_BOTH):
            out = words(out, params[1], inverse=True)
        if t in (TEXT_CAPS, TEXT_BOTH):
            out = caps(out, params[0], inverse=True)
        return out
    if t == PIX_PLANAR:
        return planar(data, params[0], inverse=True)
    if t == PIX_RCT:
        return rct(data, params[0], inverse=True)
    if t == PIX_BOTH:
        return rct(planar(data, params[0], inverse=True), params[0], inverse=True)
    if t == PIX_MED:
        return med(data, int.from_bytes(params[1:5], "little"), params[0], inverse=True)
    return data
