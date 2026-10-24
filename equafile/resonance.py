"""V9 — waves and resonance (Raaj's follow-up #2).

The file is treated as a signal x[n] (bytes, or 16-bit samples) and cut into
pieces of N samples. Per piece, the best of several wave models is kept:

V9a  waves:      x[n] ≈ Σ_k a_k·cos(ω_k n) + b_k·sin(ω_k n)
                 The ω_k are found by resonance: a bank of oscillators is driven
                 by the piece and the one that rings loudest wins (computed with
                 an FFT, which is exactly that resonator bank); its frequency is
                 then tuned to the exact peak. Its wave is subtracted and the bank
                 listens again (superposition, peeled off one wave at a time).
                 0..32 waves per piece (12 bytes each), chosen by total cost.
V9b  resonator:  x[n] ≈ Σ_k c_k·x[n-k]
                 The piece is a resonant body ringing on its own past; only small
                 kicks are needed to keep it going (linear prediction, an all-pole
                 resonant filter). Order 0..32 per piece, chosen by total cost.

What the waves cannot explain (the residual) is stored exactly and squeezed
with the V6 predictive coder, so the round trip is lossless.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz

from . import predictive
from .bits import get_varint, put_varint
from .core import Encoded

N = 4096          # samples per piece
PAD = 4           # resonator bank resolution: 4x finer than the FFT bin
KMAX = 32
ORDERS = (0, 1, 2, 4, 8, 16, 32)
Q = 12            # resonator coefficients in Q12 fixed point


# ------------------------------------------------------------------ signal <-> bytes
def to_signal(data: bytes, width: int) -> np.ndarray:
    if width == 1:
        return np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    return np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.int64)


def residual_bytes(r: np.ndarray, width: int) -> bytes:
    if width == 1:
        return (r & 0xFF).astype(np.uint8).tobytes()
    u = (r & 0xFFFF).astype(np.uint16)
    return (u & 0xFF).astype(np.uint8).tobytes() + (u >> 8).astype(np.uint8).tobytes()  # low plane, high plane


def residual_from(b: bytes, n: int, width: int) -> np.ndarray:
    if width == 1:
        return np.frombuffer(b, dtype=np.uint8).astype(np.int64)
    lo = np.frombuffer(b[:n], dtype=np.uint8).astype(np.int64)
    hi = np.frombuffer(b[n:2 * n], dtype=np.uint8).astype(np.int64)
    return lo | (hi << 8)


def wrap(v: np.ndarray, width: int) -> np.ndarray:
    """Back to the sample range (residuals are stored modulo 2^(8·width))."""
    if width == 1:
        return v & 0xFF
    return ((v + 32768) & 0xFFFF) - 32768


def est_bits(r: np.ndarray) -> float:
    """Rough coded size of a residual (Laplacian estimate), used only to choose models."""
    b = float(np.mean(np.abs(r))) if len(r) else 0.0
    return len(r) * max(0.3, np.log2(2 * np.e * b + 1e-9)) if b > 0 else len(r) * 0.3


# ------------------------------------------------------------------ resonator bank (V9a)
def resonance_spectrum(x: np.ndarray) -> np.ndarray:
    """How hard each oscillator of the bank rings when driven by x (energy per frequency).
    The Hann window gives the resonators a high Q, so one tone does not make its neighbours ring."""
    return np.abs(np.fft.rfft(x * np.hanning(len(x)), PAD * len(x))) ** 2


def settle(ws: list[float], t: np.ndarray, x: np.ndarray) -> list[float]:
    """All oscillators settle together: tune every frequency at once to best explain x
    (variable projection: amplitudes solved exactly for any set of frequencies)."""
    from scipy.optimize import least_squares

    def miss(f):
        A = np.concatenate([np.stack([np.cos(w * t), np.sin(w * t)], 1) for w in f], 1)
        coef, *_ = np.linalg.lstsq(A, x, rcond=None)
        return x - A @ coef

    sol = least_squares(miss, np.array(ws), method="lm", x_scale=2 * np.pi / (PAD * len(t)), max_nfev=60 * len(ws))
    return [float(w) for w in sol.x]


def wave_pred(waves, n0: int, n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    p = np.zeros(n)
    for w, a, b in waves:
        p += float(a) * np.cos(float(w) * t) + float(b) * np.sin(float(w) * t)
    return p


def _fit_at(w: float, t: np.ndarray, res: np.ndarray):
    A = np.stack([np.cos(w * t), np.sin(w * t)], 1)
    coef, *_ = np.linalg.lstsq(A, res, rcond=None)
    return coef, float(np.sum((res - A @ coef) ** 2))


def tune(k: int, t: np.ndarray, res: np.ndarray) -> float:
    """Sweep the ringing resonator's frequency around bank slot k to the exact peak."""
    step = 2 * np.pi / (PAD * len(t))
    lo, hi = max(0.0, (k - 1) * step), (k + 1) * step
    for _ in range(3):
        grid = np.linspace(lo, hi, 9)
        errs = [_fit_at(w, t, res)[1] for w in grid]
        j = int(np.argmin(errs))
        lo, hi = grid[max(0, j - 1)], grid[min(8, j + 1)]
    return float(grid[j])


def _joint(ws, t, x):
    """Least-squares amplitudes for all waves together (they settle as one superposition)."""
    A = np.concatenate([np.stack([np.cos(w * t), np.sin(w * t)], 1) for w in ws], 1)
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return coef.reshape(-1, 2)


def _retune(w: float, t: np.ndarray, res: np.ndarray) -> float:
    half = 2 * np.pi / (PAD * len(t))
    lo, hi = max(0.0, w - half), w + half
    for _ in range(3):
        grid = np.linspace(lo, hi, 9)
        j = int(np.argmin([_fit_at(g, t, res)[1] for g in grid]))
        lo, hi = grid[max(0, j - 1)], grid[min(8, j + 1)]
    return float(grid[j])


def fit_waves(x: np.ndarray):
    """Listen, add the loudest resonance, let all waves settle together, repeat.
    Return the best (waves, residual) by total cost."""
    n = len(x)
    t = np.arange(n, dtype=np.float64)
    xf = x.astype(np.float64)
    res, ws = xf.copy(), []
    best = (est_bits(x), [], x)
    for _ in range(KMAX):
        k = int(np.argmax(resonance_spectrum(res)[: PAD * n // 2]))
        ws.append(tune(k, t, res))
        if len(ws) <= 12:
            ws = settle(ws, t, xf)
        ws32 = [np.float32(w) for w in ws]
        coef = _joint([float(w) for w in ws32], t, xf).astype(np.float32)
        waves = [(w, a, b) for w, (a, b) in zip(ws32, coef)]
        pred = wave_pred(waves, 0, n)
        r = x - np.round(pred).astype(np.int64)
        cost = est_bits(r) + len(waves) * 96
        if cost < best[0]:
            best = (cost, waves, r)
        res = xf - pred
    return best[1], best[2]


# ------------------------------------------------------------------ resonant filter (V9b)
def lpc(x: np.ndarray, order: int) -> np.ndarray:
    if order == 0:
        return np.zeros(0, dtype=np.int64)
    xf = x.astype(np.float64) * np.hanning(len(x))
    ac = np.correlate(xf, xf, "full")[len(xf) - 1: len(xf) + order]
    if ac[0] == 0:
        return np.zeros(order, dtype=np.int64)
    ac[0] *= 1.0001
    c = solve_toeplitz(ac[:order], ac[1:order + 1])
    return np.clip(np.round(c * (1 << Q)), -32767, 32767).astype(np.int64)


def lpc_residual(x_full: np.ndarray, start: int, n: int, c: np.ndarray) -> np.ndarray:
    seg = x_full[start:start + n]
    p = len(c)
    if p == 0:
        return seg.copy()
    hist = np.concatenate([np.zeros(max(0, p - start), dtype=np.int64), x_full[max(0, start - p):start + n]])
    lags = np.lib.stride_tricks.sliding_window_view(hist, p)[:n][:, ::-1]  # row n: x[n-1] .. x[n-p]
    return seg - ((lags @ c) >> Q)


# ------------------------------------------------------------------ codec
def _encode(data: bytes, width: int, mode: str, none: bool = False):
    x = to_signal(data, width)
    head = bytearray()
    for v in (width, len(data), len(x), "ab".index(mode)):
        put_varint(head, v)
    params, res, stats = bytearray(), [], {"models": []}
    for start in range(0, len(x), N):
        seg = x[start:start + N]
        if mode == "a":
            waves, r = fit_waves(seg) if len(seg) >= 64 and not none else ([], seg.copy())
            params.append(len(waves))
            for w, a, b in waves:
                params += np.array([w, a, b], dtype="<f4").tobytes()
            stats["models"].append(len(waves))
        else:
            best = None
            for p in ((0,) if none else ORDERS):
                c = lpc(seg, p) if len(seg) > p else np.zeros(0, dtype=np.int64)
                r = lpc_residual(x, start, len(seg), c)
                cost = est_bits(r) + 16 * len(c)
                if best is None or cost < best[0]:
                    best = (cost, c, r)
            _, c, r = best
            params.append(len(c))
            params += c.astype("<i2").tobytes()
            stats["models"].append(len(c))
        res.append(r)
    r = np.concatenate(res) if res else np.zeros(0, dtype=np.int64)
    tail = data[len(x) * width:]  # odd trailing byte for width 2
    body = bytes(head) + bytes(params)
    code = predictive.compress(residual_bytes(r, width) + tail)
    out = bytearray()
    put_varint(out, len(body))
    out += body + code
    stats.update(param_bytes=len(params), residual_bytes=len(code))
    return bytes(out), stats


def decode(payload: bytes) -> bytes:
    blen, pos = get_varint(payload, 0)
    body, code = payload[pos:pos + blen], payload[pos + blen:]
    width, p = get_varint(body, 0)
    nbytes, p = get_varint(body, p)
    n, p = get_varint(body, p)
    mode, p = get_varint(body, p)
    tail_len = nbytes - n * width
    raw = predictive.decompress(code, n * width + tail_len)
    r = residual_from(raw, n, width)
    x = np.zeros(n, dtype=np.int64)
    for start in range(0, n, N):
        m = min(N, n - start)
        cnt = body[p]
        p += 1
        if mode == 0:
            waves = []
            for _ in range(cnt):
                w, a, b = np.frombuffer(body[p:p + 12], dtype="<f4")
                p += 12
                waves.append((w, a, b))
            pred = np.round(wave_pred(waves, 0, m)).astype(np.int64) if waves else np.zeros(m, dtype=np.int64)
            x[start:start + m] = wrap(r[start:start + m] + pred, width)
        else:
            c = np.frombuffer(body[p:p + 2 * cnt], dtype="<i2").astype(np.int64)
            p += 2 * cnt
            if cnt == 0:
                x[start:start + m] = wrap(r[start:start + m], width)
                continue
            cl = c.tolist()
            for i in range(start, start + m):
                acc = 0
                for j, cj in enumerate(cl, 1):
                    if i - j >= 0:
                        acc += cj * int(x[i - j])
                x[i] = int(wrap(np.array([int(r[i]) + (acc >> Q)]), width)[0])
    if width == 1:
        out = x.astype(np.uint8).tobytes()
    else:
        out = x.astype("<i2").tobytes()
    return out + raw[n * width:]


def _variant(mode: str, vid: str, name: str, summary: str):
    class V:
        ID, NAME, SUMMARY, MAX_INPUT = vid, name, summary, 1 << 21

        @staticmethod
        def encode(data: bytes, width: int | None = None) -> Encoded:
            widths = [width] if width else ([1, 2] if len(data) % 2 == 0 and len(data) <= 1 << 18 else [1])
            best = None
            for w in widths:
                for none in (False, True):  # best approach: waves, or no waves at all
                    payload, st = _encode(data, w, mode, none)
                    if best is None or len(payload) < len(best[0]):
                        best = (payload, st, w)
            payload, st, w = best
            used = sum(1 for m in st["models"] if m)
            what = "waves" if mode == "a" else "resonator taps"
            eq = (f"signal = {'bytes' if w == 1 else '16-bit samples'}, {len(st['models'])} pieces of {N}; "
                  f"{used} pieces use {what} (avg {np.mean(st['models']):.1f} per piece)\n"
                  f"wave/resonator parameters {st['param_bytes']:,} B + residual (what waves can't explain) {st['residual_bytes']:,} B")
            return Encoded(payload, eq, {**st, "width": w})

        decode = staticmethod(lambda payload: globals()["decode"](payload))
    return V


WaveVariant = _variant("a", "v9a-waves", "V9a Waves found by resonance",
                       "Resonator bank finds each piece's frequencies; superposed waves + exact residual")
ResonatorVariant = _variant("b", "v9b-resonator", "V9b Resonant filter (rings on its past)",
                            "Each piece = a resonant body x[n] = Σ c·x[n-k] + small kicks")
