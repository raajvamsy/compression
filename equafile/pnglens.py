"""PNG lens: compress the pixels (where the structure lives), then rebuild the identical PNG file.

Deflate scrambles a PNG's rows and columns, so its bytes look random. We decode the pixels, FunnelZip
them, and on decode re-encode with the same PNG settings. This path is only taken if that re-encoding
reproduces the original file byte for byte (true for files written by the same encoder, e.g. PIL);
other PNGs fall back to FunnelZip on the bytes. A general version would store a re-encode recipe
(preflate/precomp-style); not built yet.
"""
from __future__ import annotations

import io

from PIL import Image

from . import funnelzip as FZ
from .bits import get_varint, put_varint

OPTIONS = ({}, {"optimize": True}, {"compress_level": 9}, {"compress_level": 6})
MODES = ("RGB", "RGBA", "L", "LA", "P", "I;16")


def rebuild(pixels: bytes, mode: str, size: tuple[int, int], opt: int, icc: bytes = b"") -> bytes:
    buf = io.BytesIO()
    extra = {"icc_profile": icc} if icc else {}
    Image.frombytes(mode, size, pixels).save(buf, "PNG", **OPTIONS[opt], **extra)
    return buf.getvalue()


def _compress(blob: bytes) -> tuple[bytes, dict]:
    try:
        im = Image.open(io.BytesIO(blob))
        if im.mode not in MODES or im.mode == "P":
            raise ValueError(f"mode {im.mode}")
        px = im.tobytes()
        icc = im.info.get("icc_profile", b"") or b""
        opt = next(i for i in range(len(OPTIONS)) if rebuild(px, im.mode, im.size, i, icc) == blob)
    except (StopIteration, ValueError, OSError) as e:
        return b"\x00" + FZ.encode(blob, image_lens=False).payload, {"path": "bytes", "why": str(e) or "no exact re-encode"}
    out = bytearray(b"\x01")
    out.append(MODES.index(im.mode))
    out.append(opt)
    put_varint(out, im.size[0])
    put_varint(out, im.size[1])
    put_varint(out, len(icc))
    out += icc
    out += FZ.encode(px, image_lens=False).payload
    return bytes(out), {"path": "pixels", "mode": im.mode, "size": im.size}


def decompress(payload: bytes) -> bytes:
    if payload[0] == 0:
        return FZ.decode(payload[1:])
    mode, opt = MODES[payload[1]], payload[2]
    w, pos = get_varint(payload, 3)
    h, pos = get_varint(payload, pos)
    n, pos = get_varint(payload, pos)
    icc = payload[pos:pos + n]
    return rebuild(FZ.decode(payload[pos + n:]), mode, (w, h), opt, icc)


def compress(blob: bytes) -> tuple[bytes, dict]:
    """Structure path, unless plain FunnelZip on the file's bytes is smaller (small or unusual files)."""
    body, info = _compress(blob)
    if info["path"] != "bytes" and len(blob) < (1 << 20):
        plain = b"\x00" + FZ.encode(blob, image_lens=False).payload
        if len(plain) < len(body):
            return plain, {"path": "bytes", "why": f"bytes path smaller than {info['path']} path"}
    return body, info
