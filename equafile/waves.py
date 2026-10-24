"""V7 — lossy signal equations: a file as a short sum of cosine waves.

image  pixel(x,y)   = Σ a·cos(πu(2x+1)/2W)·cos(πv(2y+1)/2H)        per tile (or one global tile)
audio  sample(t)    = Σ a·cos(πk(2t+1)/2N)                           per frame of N samples
video  value(x,y,t) = Σ a·cos(..x..)·cos(..y..)·cos(..t..)           per tile × group of frames

Only the terms whose quantised amplitude round(a/q) is non-zero are kept, so q
sets the number of terms. Amplitudes are stored as zero-run / value varints and
then (optionally) squeezed by the V6 predictive coder.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn

from . import predictive
from .bits import get_varint, put_varint, unzigzag, zigzag

KIND = {"image": 1, "audio": 2, "video": 3}


# ------------------------------------------------------------------ helpers
def rgb_to_ycc(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    return np.stack([y, cb, cr], -1)


def ycc_to_rgb(ycc: np.ndarray) -> np.ndarray:
    y, cb, cr = ycc[..., 0], ycc[..., 1] - 128, ycc[..., 2] - 128
    rgb = np.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], -1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def _pad_to(a: np.ndarray, shape) -> np.ndarray:
    pads = [(0, s - d) for d, s in zip(a.shape, shape)]
    return np.pad(a, pads, mode="edge")


def _ceil(a: int, b: int) -> int:
    return -(-a // b) * b


def _freq_order(tile) -> np.ndarray:
    """Coefficient order inside a tile: low frequencies first."""
    grids = np.meshgrid(*[np.arange(t) / t for t in tile], indexing="ij")
    return np.argsort(sum(grids).ravel(), kind="stable")


def _tiles(a: np.ndarray, tile) -> np.ndarray:
    """(D0,D1,..) -> (ntiles, prod(tile)) after a per-tile DCT."""
    nd = a.ndim
    shp = []
    for d, t in zip(a.shape, tile):
        shp += [d // t, t]
    b = a.reshape(shp).transpose(list(range(0, 2 * nd, 2)) + list(range(1, 2 * nd, 2)))
    b = dctn(b, axes=tuple(range(nd, 2 * nd)), norm="ortho")
    return b.reshape(-1, int(np.prod(tile)))


def _untiles(c: np.ndarray, shape, tile) -> np.ndarray:
    nd = len(shape)
    counts = [d // t for d, t in zip(shape, tile)]
    b = idctn(c.reshape(counts + list(tile)), axes=tuple(range(nd, 2 * nd)), norm="ortho")
    inv = np.argsort(list(range(0, 2 * nd, 2)) + list(range(1, 2 * nd, 2)))
    shp = []
    for n, t in zip(counts, tile):
        shp += [n, t]
    return b.transpose(inv).reshape(shape)


def _write_sparse(out: bytearray, q: np.ndarray) -> int:
    nz = np.flatnonzero(q)
    put_varint(out, len(nz))
    prev = -1
    for i in nz.tolist():
        put_varint(out, i - prev - 1)
        put_varint(out, zigzag(int(q[i])))
        prev = i
    return len(nz)


def _read_sparse(buf: bytes, pos: int, size: int) -> tuple[np.ndarray, int]:
    q = np.zeros(size, dtype=np.float64)
    n, pos = get_varint(buf, pos)
    idx = -1
    for _ in range(n):
        run, pos = get_varint(buf, pos)
        v, pos = get_varint(buf, pos)
        idx += run + 1
        q[idx] = unzigzag(v)
    return q, pos


def _code_channel(out: bytearray, a: np.ndarray, tile, step: float) -> int:
    padded = _pad_to(a, [_ceil(d, t) for d, t in zip(a.shape, tile)])
    c = _tiles(padded, tile)[:, _freq_order(tile)]
    q = np.round(c.T.ravel() / step).astype(np.int64)  # frequency-major: same frequency of every tile together
    return _write_sparse(out, q)


def _decode_channel(buf: bytes, pos: int, shape, tile, step: float):
    pshape = [_ceil(d, t) for d, t in zip(shape, tile)]
    ntiles, tsize = int(np.prod([p // t for p, t in zip(pshape, tile)])), int(np.prod(tile))
    flat, pos = _read_sparse(buf, pos, ntiles * tsize)
    c = np.empty((ntiles, tsize))
    c[:, _freq_order(tile)] = flat.reshape(tsize, ntiles).T * step
    a = _untiles(c, pshape, tile)
    return a[tuple(slice(0, d) for d in shape)], pos


def _finish(body: bytearray, entropy: bool) -> bytes:
    out = bytearray([1 if entropy else 0])
    if entropy:
        put_varint(out, len(body))
        out += predictive.compress(bytes(body))
    else:
        out += body
    return bytes(out)


def _open(payload: bytes) -> bytes:
    if payload[0]:
        n, pos = get_varint(payload, 1)
        return predictive.decompress(payload[pos:], n)
    return payload[1:]


def _down2(a: np.ndarray) -> np.ndarray:
    """2x2 average of the last two spatial axes (chroma subsampling)."""
    h, w = a.shape[-2:]
    a = _pad_to(a, a.shape[:-2] + (_ceil(h, 2), _ceil(w, 2)))
    return a.reshape(a.shape[:-2] + (a.shape[-2] // 2, 2, a.shape[-1] // 2, 2)).mean(axis=(-3, -1))


def _up2(a: np.ndarray, h: int, w: int) -> np.ndarray:
    return a.repeat(2, axis=-2).repeat(2, axis=-1)[..., :h, :w]


# ------------------------------------------------------------------ image
def encode_image(rgb: np.ndarray, step: float, tile: int | None = 16, entropy: bool = True):
    """tile=None -> one global equation for the whole picture."""
    h, w, _ = rgb.shape
    ycc = rgb_to_ycc(rgb) - 128.0
    body = bytearray()
    for v in (KIND["image"], h, w, tile or 0, int(round(step * 16))):
        put_varint(body, v)
    terms = 0
    for ch in range(3):
        plane = ycc[..., ch] if ch == 0 else _down2(ycc[..., ch])
        t = (tile, tile) if tile else plane.shape
        terms += _code_channel(body, plane, t, step if ch == 0 else step * 1.5)
    return _finish(body, entropy), terms


def decode_image(payload: bytes) -> np.ndarray:
    buf = _open(payload)
    pos, vals = 0, []
    for _ in range(5):
        v, pos = get_varint(buf, pos)
        vals.append(v)
    _, h, w, tile, s16 = vals
    step = s16 / 16
    planes = []
    for ch in range(3):
        shape = (h, w) if ch == 0 else (_ceil(h, 2) // 2, _ceil(w, 2) // 2)
        t = (tile, tile) if tile else shape
        a, pos = _decode_channel(buf, pos, shape, t, step if ch == 0 else step * 1.5)
        planes.append(a if ch == 0 else _up2(a, h, w))
    return ycc_to_rgb(np.stack(planes, -1) + 128.0)


# ------------------------------------------------------------------ audio
def encode_audio(x: np.ndarray, step: float, frame: int = 1024, entropy: bool = True):
    """x: int16 mono samples."""
    body = bytearray()
    for v in (KIND["audio"], len(x), frame, int(round(step * 16))):
        put_varint(body, v)
    terms = _code_channel(body, x.astype(np.float64), (frame,), step)
    return _finish(body, entropy), terms


def decode_audio(payload: bytes) -> np.ndarray:
    buf = _open(payload)
    pos, vals = 0, []
    for _ in range(4):
        v, pos = get_varint(buf, pos)
        vals.append(v)
    _, n, frame, s16 = vals
    a, _ = _decode_channel(buf, pos, (n,), (frame,), s16 / 16)
    return np.clip(np.round(a), -32768, 32767).astype(np.int16)


# ------------------------------------------------------------------ video
def encode_video(frames: np.ndarray, step: float, tile=(8, 16, 16), entropy: bool = True):
    """frames: (T, H, W, 3) uint8. One 3-D cosine equation per tile × group of frames."""
    t, h, w, _ = frames.shape
    ycc = rgb_to_ycc(frames) - 128.0
    body = bytearray()
    for v in (KIND["video"], t, h, w, *tile, int(round(step * 16))):
        put_varint(body, v)
    terms = 0
    for ch in range(3):
        vol = ycc[..., ch] if ch == 0 else _down2(ycc[..., ch])
        terms += _code_channel(body, vol, tile, step if ch == 0 else step * 1.5)
    return _finish(body, entropy), terms


def decode_video(payload: bytes) -> np.ndarray:
    buf = _open(payload)
    pos, vals = 0, []
    for _ in range(8):
        v, pos = get_varint(buf, pos)
        vals.append(v)
    _, t, h, w, tt, th, tw, s16 = vals
    tile, step = (tt, th, tw), s16 / 16
    vols = []
    for ch in range(3):
        shape = (t, h, w) if ch == 0 else (t, _ceil(h, 2) // 2, _ceil(w, 2) // 2)
        a, pos = _decode_channel(buf, pos, shape, tile, step if ch == 0 else step * 1.5)
        vols.append(a if ch == 0 else _up2(a, h, w))
    return ycc_to_rgb(np.stack(vols, -1) + 128.0)


# ------------------------------------------------------------------ metrics
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def snr(ref: np.ndarray, test: np.ndarray) -> float:
    ref, test = ref.astype(np.float64), test.astype(np.float64)
    noise = np.sum((ref - test) ** 2)
    return float("inf") if noise == 0 else 10 * np.log10(np.sum(ref ** 2) / noise)
