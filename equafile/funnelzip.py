"""V10 — FunnelZip: every EquaFile lens votes on every bit (see docs/DESIGN_FUNNELZIP.md).

LISTEN  resonance of the file with itself (autocorrelation) proposes a sample width W, a channel
        step C, a row stride S and a repeat period R; binary-looking files also try the code
        filter F, text-looking files the primer P. Candidates are trial-coded on a slice and the
        cheapest wins (a few small numbers in the header).
CODE    C core (funnelzip/funnelzip.c): lenses -> funnel of mixers -> refine -> 16-bit coder.
"""
from __future__ import annotations

import ctypes
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import codefilter, transforms
from .bits import get_varint, put_varint
from .core import Encoded

ID = "v10-funnelzip"
NAME = "V10 FunnelZip (all lenses, mixed)"
SUMMARY = "Memory, word, echo, equation, resonator, grid, rhythm, motion and code lenses blended by a funnel of mixers"
MAX_INPUT = None

LENSES = {"memory": 1, "word": 2, "echo": 4, "equation": 8, "resonator": 16, "grid": 32, "funnel": 64, "refine": 128,
          "rhythm": 256, "motion": 512, "precise": 1024, "stereo": 2048, "code": 4096}
ALL = 8191
SIGNAL = LENSES["equation"] | LENSES["resonator"] | LENSES["grid"] | LENSES["motion"] | LENSES["stereo"]
TEXT = ALL & ~SIGNAL          # text: signal lenses cost time and find nothing
NO_WC = ALL & ~(LENSES["word"] | LENSES["code"])
NO_WCM = NO_WC & ~LENSES["motion"]


def mask_options(text: bool) -> list[int]:
    """Lens sets Listen tries: a lens that finds nothing in a file still costs time and adds noise to the mix."""
    if text:
        return [TEXT, TEXT & ~LENSES["code"]]
    return [ALL, ALL & ~LENSES["word"], NO_WC, NO_WCM]


def mask_cost(mask: int) -> int:
    """Header bytes a lens set costs (ALL and TEXT are one-bit presets)."""
    return 0 if mask in (ALL, TEXT) else len(varint_bytes(mask))


def varint_bytes(v: int) -> bytes:
    b = bytearray()
    put_varint(b, v)
    return bytes(b)
TRIAL = 1 << 17
CHUNK_MIN = 3 << 20      # files at least this big are split and coded on all CPU cores

_DIR = Path(__file__).resolve().parents[1] / "funnelzip"
_SRC, _LIB = _DIR / "funnelzip.c", _DIR / "libfunnelzip.dylib"
PRIME = (_DIR / "prime.bin").read_bytes() if (_DIR / "prime.bin").exists() else b""
_lib = None


def lib():
    global _lib
    if _lib is None:
        if not _LIB.exists() or _LIB.stat().st_mtime < _SRC.stat().st_mtime:
            subprocess.run(["clang", "-O3", "-shared", "-fPIC", "-o", str(_LIB), str(_SRC)], check=True)
        _lib = ctypes.CDLL(str(_LIB))
        common = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t]
        _lib.fz_compress.restype = ctypes.c_size_t
        _lib.fz_compress.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t] + common
        _lib.fz_decompress.restype = ctypes.c_int
        _lib.fz_decompress.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t] + common
    return _lib


def primer_for(t: int, params: bytes) -> bytes:
    """The primer reshaped the same way as the file (text reshapes), so its knowledge still lines up."""
    if t in (transforms.TEXT_CAPS, transforms.TEXT_WORDS, transforms.TEXT_BOTH):
        p = PRIME
        if t in (transforms.TEXT_CAPS, transforms.TEXT_BOTH):
            p = transforms.caps(p, params[0])
        if t in (transforms.TEXT_WORDS, transforms.TEXT_BOTH):
            p = transforms.words(p, params[1])
        return p[: len(p) // 12 * 12]
    return PRIME


def raw_compress(data: bytes, W: int = 1, C: int = 1, S: int = 0, mask: int = ALL, R: int = 0, prime: bool = False,
                 primer: bytes | None = None) -> bytes:
    pre = (primer if primer is not None else PRIME) if prime else b""
    buf = pre + data
    cap = len(data) + len(data) // 4 + 1024
    out = ctypes.create_string_buffer(cap)
    n = lib().fz_compress(buf, len(buf), out, cap, W, C, S, mask, R, len(pre))
    return out.raw[:n]


def raw_decompress(code: bytes, n: int, W: int = 1, C: int = 1, S: int = 0, mask: int = ALL, R: int = 0,
                   prime: bool = False, primer: bytes | None = None) -> bytes:
    pre = (primer if primer is not None else PRIME) if prime else b""
    out = ctypes.create_string_buffer(pre + bytes(max(1, n)), len(pre) + max(1, n))
    lib().fz_decompress(code, len(code), out, len(pre) + n, W, C, S, mask, R, len(pre))
    return out.raw[len(pre):len(pre) + n]


# ------------------------------------------------------------------ LISTEN
def listen(data: bytes, max_lag: int = 1 << 16) -> dict:
    """Resonance of the file with itself: where does it repeat its own shape?"""
    x = np.frombuffer(data[: 1 << 21], dtype=np.uint8).astype(np.float32)
    if len(x) < 64:
        return {"channels": [1], "strides": []}
    x -= x.mean()
    m = 1 << int(np.ceil(np.log2(2 * len(x))))
    F = np.fft.rfft(x, m)
    ac = np.fft.irfft(F * np.conj(F), m)[: min(max_lag, len(x) // 2)]
    ac = ac / (ac[0] + 1e-9)
    small = ac[1:5]
    ch = int(np.argmax(small)) + 1
    strides = []
    if len(ac) > 16:
        cand = ac[8:].copy()
        for _ in range(2):
            j = int(np.argmax(cand))
            if cand[j] < 0.2:
                break
            strides.append(j + 8)
            cand[max(0, j - 4): j + 5] = -1
    return {"channels": sorted({1, ch}), "strides": strides, "ac_small": [round(float(v), 3) for v in small]}


def long_periods(data: bytes, W: int, top: int = 2, max_bytes: int = 1 << 22) -> list[int]:
    """Resonance at long range: the distance (in bytes) at which the file repeats itself
    (a table row, a log line, a music bar, a video frame)."""
    if W == 2:
        x = np.frombuffer(data[: min(len(data), max_bytes) // 2 * 2], dtype="<i2").astype(np.float32)
    else:
        x = np.frombuffer(data[:max_bytes], dtype=np.uint8).astype(np.float32)
    if len(x) < 256:
        return []
    x -= x.mean()
    m = 1 << int(np.ceil(np.log2(2 * len(x))))
    F = np.fft.rfft(x, m)
    ac = np.fft.irfft(F * np.conj(F), m)[: len(x) // 2]
    ac /= ac[0] + 1e-9
    ac *= len(x) / (len(x) - np.arange(len(ac)))      # unbias: long lags have fewer overlapping samples
    out, cand = [], ac.copy()
    cand[:64] = -1
    for _ in range(top):
        j = int(np.argmax(cand))
        if cand[j] < 0.25:
            break
        out.append(j * W)
        cand[max(0, j - 8): j + 9] = -1
    return out


def candidates(info: dict) -> list[tuple[int, int, int]]:
    cands = [(1, 1, 0), (2, 1, 0), (2, 2, 0)]
    for c in info["channels"]:
        if c > 1:
            cands.append((1, c, 0))
        for s in info["strides"]:
            cands.append((1, c, s))
    return list(dict.fromkeys(cands))


def looks_binary(sample: bytes) -> bool:
    if not sample:
        return False
    a = np.frombuffer(sample, dtype=np.uint8)
    return float(np.mean((a < 9) | (a > 126))) > 0.15


def looks_text(sample: bytes) -> bool:
    if not sample:
        return False
    a = np.frombuffer(sample, dtype=np.uint8)
    return float(np.mean(((a >= 32) & (a < 127)) | (a == 10) | (a == 9) | (a == 13))) > 0.95


def spread(data: bytes, size: int | None = None, align: int = 12288) -> bytes:
    """A trial sample taken from the start, middle and end, so one region can't decide for the whole file.
    Pieces start on multiples of `align` (divisible by 2, 3, 4 and common row lengths) to stay in phase."""
    size = size or TRIAL
    if len(data) <= 2 * size:
        return data[:size]
    k = max(align, size // 3 // align * align)
    mid = len(data) // 2 // align * align
    end = (len(data) - k) // align * align
    return data[:k] + data[mid:mid + k] + data[end:end + k]


def _pmap(fn, items):
    """Listen's trial codings are independent: run them on all cores (the C core releases the GIL)."""
    items = list(items)
    with ThreadPoolExecutor(min(len(items), os.cpu_count() or 1)) as ex:
        return list(ex.map(fn, items))


def choose(data: bytes, mask: int = ALL) -> tuple[tuple, dict, list]:
    """Returns settings (W, C, S, R, F, P, mask), what Listen heard, and every trial (size, settings)."""
    info = listen(data)
    sl = spread(data)
    text = looks_text(sl)
    info["text"] = text
    if text and mask == ALL:
        mask = TEXT
    cands = [(1, 1, 0)] if text else candidates(info)          # text never needs 16-bit / grid layouts
    trials = sorted(zip(_pmap(lambda c: len(raw_compress(sl, *c, mask)), cands), [c + (0, 0, 0) for c in cands]))
    W, C, S = trials[0][1][:3]
    F = P = 0
    if W == 1 and looks_binary(sl):      # code lens: try the call/jump filters (with and without a stride)
        fs = (0, codefilter.ARM64, codefilter.X86, codefilter.BOTH)
        samples = {f: spread(codefilter.apply(data, f, True) if f else data) for f in fs}
        layouts = list(dict.fromkeys([(C, S), (1, 0)]))
        jobs = [(f, c, st) for f in fs for (c, st) in layouts]
        ft = sorted(zip(_pmap(lambda j: len(raw_compress(samples[j[0]], 1, j[1], j[2], mask)), jobs), jobs))
        trials += [(b, (1, c, st, 0, f, 0)) for b, (f, c, st) in ft]
        F, C, S = ft[0][1]
    if W == 1 and S == 0 and F == 0 and PRIME and text:   # primer: does prior knowledge help?
        b = len(raw_compress(sl, 1, C, 0, mask, 0, True))
        trials.append((b, (1, C, 0, 0, 0, 1)))
        P = int(b < trials[0][0])
    # last phase, one parallel batch: rhythm period x lens set
    periods = []
    if mask & LENSES["rhythm"]:
        for p in long_periods(data, W, top=2):   # a strong period may really be 2 repeats of a shorter one
            for q in (p, p // 2):
                q -= q % W
                if q >= 64 and q != S and q not in periods:
                    periods.append(q)
    info["periods"] = periods
    masks = mask_options(text) if mask in (ALL, TEXT) else [mask]
    ps = [0] + periods
    if len(ps) * len(masks) > 1:
        src = codefilter.apply(data, F, True) if F else data
        big = src[: min(len(src), max(TRIAL, 2 * max(ps) + (1 << 16)), 1 << 20)] if periods else spread(src)
        # the best lens set hardly depends on the period: periods are tried with the default set, lens sets with the
        # strongest period, all in one parallel batch (one wave on the cores instead of periods x sets)
        p0, m0 = ps[1] if periods else 0, masks[0]
        jobs = list(dict.fromkeys([(p, m0) for p in ps] + [(p0, m) for m in masks]))
        res = dict(zip(jobs, _pmap(lambda j: len(raw_compress(big, W, C, S, j[1], j[0], bool(P))) + mask_cost(j[1]), jobs)))
        trials += [(b, (W, C, S, p, F, P, m)) for (p, m), b in res.items()]
        R = min(ps, key=lambda p: res[(p, m0)])
        mask = min(masks, key=lambda m: res[(p0, m)])
    else:
        R = 0
    return (W, C, S, R, F, P, mask), info, trials


def reshaped_params(t: int, W: int, C: int, S: int, R: int) -> tuple[int, int, int, int]:
    """Planar reshapes turn C interleaved channels into C pictures of one channel each."""
    if t in (transforms.PIX_PLANAR, transforms.PIX_BOTH) and C > 1:
        return W, 1, S // C, R // C if R % C == 0 else 0
    return W, C, S, R


def choose_reshape(data: bytes, W: int, C: int, S: int, R: int, F: int, P: int, mask: int) -> tuple[int, tuple, list]:
    """Try the reversible reshapes that fit this kind of file on a sample, each with the layouts that suit it;
    keep the smallest. Returns (reshape id, (W, C, S, R) to code it with, trials)."""
    base = (W, C, S, R)
    text = not mask & SIGNAL
    if W == 1 and S > 0 and F == 0 and not text:
        jobs = [(0, base, C), (transforms.PIX_MED, base, C)]
        for c in sorted({c for c in (C, 3, 4) if c >= 3 and S % c == 0}):   # colour reshapes need the pixel's channel count
            for geo in dict.fromkeys([base, (W, c, S, R), (W, 1, S, R)]):
                jobs.append((transforms.PIX_RCT, geo, c))
            jobs.append((transforms.PIX_BOTH, (W, 1, S // c, R // c if R % c == 0 else 0), c))
    elif text and F == 0:
        jobs = [(0, base, C), (transforms.TEXT_BOTH, base, C)]
    else:
        return 0, base, []
    smp = spread(data)
    def size(job):
        t, (w, c, st, r), ch = job
        try:
            td, par = transforms.forward(smp, t, ch, S)
        except ValueError:
            return 1 << 62
        return len(raw_compress(td, w, c, st, mask, r, bool(P), primer_for(t, par) if P else None))
    res = _pmap(size, jobs)
    best = min(range(len(jobs)), key=lambda i: res[i])
    t, geo, ch = jobs[best]
    return t, (geo, ch), [(res[i], transforms.NAMES[jobs[i][0]], jobs[i][1]) for i in range(len(jobs))]


# ------------------------------------------------------------------ codec
def chunks(n: int, W: int, C: int, S: int, R: int, threads: int) -> list[tuple[int, int]]:
    """Split points on whole samples / rows / frames, so every chunk starts in phase."""
    unit = math.lcm(W * C, S or 1, R or 1)
    k = max(1, min(threads, n // max(unit, 1 << 20)))
    step = -(-n // k)
    step = -(-step // unit) * unit
    return [(a, min(n, a + step)) for a in range(0, n, step)]


def encode(data: bytes, mask: int = ALL, threads: int = 1, image_lens: bool = True, fixed: tuple | None = None) -> Encoded:
    """threads > 1 = fast mode: big files are split into chunks coded in parallel (faster, but each chunk
    learns from scratch, so files get bigger). Default: one chunk, best ratio; Listen always runs in parallel."""
    if image_lens and data[:3] == b"\xff\xd8\xff" or image_lens and data[:8] == b"\x89PNG\r\n\x1a\n":
        from . import jpeglens, pnglens   # image files: compress where the structure lives
        lens, kind = (jpeglens, 1) if data[:2] == b"\xff\xd8" else (pnglens, 2)
        body, linfo = lens.compress(data)
        out = bytearray()
        put_varint(out, len(data))
        out += bytes([0xFF, kind]) + body
        return Encoded(bytes(out), f"{['', 'JPEG', 'PNG'][kind]} lens: {linfo['path']} path", {**linfo, "chunks": 1})
    if fixed:   # the caller already knows the layout (e.g. JPEG coefficient planes): skip Listen's trials
        (W, C, S, R, F, P, mask), info, trials = fixed, {"fixed": True}, []
    else:
        (W, C, S, R, F, P, mask), info, trials = choose(data, mask) if data else ((1, 1, 0, 0, 0, 0, mask), {}, [])
    src = codefilter.apply(data, F, True) if F else data
    T, choice, ttrials = choose_reshape(src, W, C, S, R, F, P, mask) if data and not fixed else (0, None, [])
    tpar = b""
    if T:
        (geo, ch) = choice
        src, tpar = transforms.forward(src, T, ch, S)
        W, C, S, R = geo
    primer = primer_for(T, tpar) if P else None
    parts = chunks(len(src), W, C, S, R, threads) if len(src) >= CHUNK_MIN and threads > 1 else [(0, len(src))]
    with ThreadPoolExecutor(len(parts)) as ex:   # the C core releases the GIL
        codes = list(ex.map(lambda ab: raw_compress(src[ab[0]:ab[1]], W, C, S, mask, R, bool(P), primer), parts))
    out = bytearray()
    put_varint(out, len(data))
    custom = (W, C, S, R) != (1, 1, 0, 0)
    preset = 0 if mask == ALL else 0x40 if mask == TEXT else 0x80
    out.append(int(custom) | F << 1 | P << 3 | (len(parts) > 1) << 4 | (T > 0) << 5 | preset)
    if custom:
        for v in (W, C, S, R):
            put_varint(out, v)
    if preset == 0x80:
        put_varint(out, mask)
    if T:
        out += bytes([T, len(tpar)]) + tpar
        put_varint(out, len(src))     # reshapes may change the length (text markers / word codes)
    if len(parts) > 1:
        put_varint(out, len(parts))
        for (a, b), c in zip(parts, codes):
            put_varint(out, b - a)
            put_varint(out, len(c))
    for c in codes:
        out += c
    code_len = sum(map(len, codes))
    kind = "16-bit samples" if W == 2 else "bytes"
    eq = (f"listen: treat as {kind}, channel step {C}, row stride {S or 'none'} "
          f"(picked from {len(trials)} trial codings)\n"
          f"rhythm: {'repeats every ' + format(R, ',') + ' bytes' if R else 'learned from echoes'}"
          f"{'; code filter ' + ['', 'ARM64', 'x86', 'ARM64+x86'][F] if F else ''}{'; primed' if P else ''}"
          f"{'; ' + str(len(parts)) + ' chunks in parallel' if len(parts) > 1 else ''}"
          f"{'; reshaped: ' + transforms.NAMES[T] if T else ''}\n"
          f"lenses on: {', '.join(k for k, b in LENSES.items() if mask & b)}\n"
          f"{8 * code_len / max(1, len(data)):.3f} bits per byte")
    return Encoded(bytes(out), eq, {"W": W, "C": C, "S": S, "R": R, "F": F, "P": P, "mask": mask, "chunks": len(parts),
                                    "T": T, "reshape_trials": ttrials, "trials": trials, "listen": info})


def decode(payload: bytes) -> bytes:
    n, pos = get_varint(payload, 0)
    if n == 0:
        return b""
    flags = payload[pos]
    pos += 1
    if flags == 0xFF:   # image lens container (never a valid normal flag byte)
        from . import jpeglens, pnglens
        return (jpeglens if payload[pos] == 1 else pnglens).decompress(payload[pos + 1:])
    W, C, S, R = 1, 1, 0, 0
    if flags & 1:
        vals = []
        for _ in range(4):
            v, pos = get_varint(payload, pos)
            vals.append(v)
        W, C, S, R = vals
    mask = TEXT if flags & 0x40 else ALL
    if flags & 0x80:
        mask, pos = get_varint(payload, pos)
    T, tpar = 0, b""
    if flags & 0x20:
        T, k = payload[pos], payload[pos + 1]
        tpar = payload[pos + 2:pos + 2 + k]
        pos += 2 + k
        n, pos = get_varint(payload, pos)
    F, P = (flags >> 1) & 3, (flags >> 3) & 1
    primer = primer_for(T, tpar) if P else None
    if flags & 16:
        k, pos = get_varint(payload, pos)
        sizes = []
        for _ in range(k):
            a, pos = get_varint(payload, pos)
            b, pos = get_varint(payload, pos)
            sizes.append((a, b))
        jobs, off = [], pos
        for a, b in sizes:
            jobs.append((payload[off:off + b], a))
            off += b
        with ThreadPoolExecutor(len(jobs)) as ex:
            data = b"".join(ex.map(lambda j: raw_decompress(j[0], j[1], W, C, S, mask, R, bool(P), primer), jobs))
    else:
        data = raw_decompress(payload[pos:], n, W, C, S, mask, R, bool(P), primer)
    if T:
        data = transforms.inverse(data, T, tpar)
    return codefilter.apply(data, F, False) if F else data
