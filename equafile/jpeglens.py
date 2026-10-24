"""JPEG lens: compress a baseline JPEG at the level where its structure lives, then rebuild the
identical file.

A JPEG is already an equation (quantized cosine coefficients per 8x8 block) written with Huffman
codes. We undo only the Huffman layer, lay the coefficients out as frequency planes (the same
frequency of neighbouring blocks side by side), FunnelZip them, and on decode rewrite the scan with
the file's own Huffman tables. The rebuilt file is checked byte for byte; anything we can't
reproduce exactly (progressive, restart markers, odd encoders) falls back to FunnelZip on the bytes.
"""
from __future__ import annotations

import os

import numpy as np

from . import funnelzip as FZ
from .bits import get_varint, put_varint

ZZ_EOB, ZZ_ZRL = 0x00, 0xF0


# ------------------------------------------------------------------ parsing
def parse(blob: bytes) -> dict:
    if blob[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    i, comps, huff, info = 2, [], {}, {"restart": 0}
    while True:
        while blob[i] != 0xFF:
            i += 1
        marker = blob[i + 1]
        if marker == 0xD8 or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            i += 2
            continue
        seg_len = int.from_bytes(blob[i + 2:i + 4], "big")
        seg = blob[i + 4:i + 2 + seg_len]
        if marker in (0xC0, 0xC1):
            info["Y"], info["X"] = int.from_bytes(seg[1:3], "big"), int.from_bytes(seg[3:5], "big")
            for k in range(seg[5]):
                cid, hv, tq = seg[6 + 3 * k], seg[7 + 3 * k], seg[8 + 3 * k]
                comps.append({"id": cid, "h": hv >> 4, "v": hv & 15, "tq": tq})
        elif marker in (0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            raise ValueError("not baseline (progressive/lossless/arithmetic)")
        elif marker == 0xC4:
            j = 0
            while j < len(seg):
                tc_th = seg[j]
                counts = list(seg[j + 1:j + 17])
                syms = list(seg[j + 17:j + 17 + sum(counts)])
                huff[(tc_th >> 4, tc_th & 15)] = (counts, syms)
                j += 17 + sum(counts)
        elif marker == 0xDD:
            info["restart"] = int.from_bytes(seg[0:2], "big")
        elif marker == 0xDA:
            ns = seg[0]
            scomps = [{"id": seg[1 + 2 * k], "td": seg[2 + 2 * k] >> 4, "ta": seg[2 + 2 * k] & 15} for k in range(ns)]
            start = i + 2 + seg_len
            j = start
            while True:   # scan ends at the first marker that is not stuffing (FF00) or a restart (FFD0-FFD7)
                if blob[j] == 0xFF and blob[j + 1] != 0x00 and not 0xD0 <= blob[j + 1] <= 0xD7:
                    break
                j += 1
            info.update(comps=comps, huff=huff, scomps=scomps, header=blob[:start], scan=blob[start:j], trailer=blob[j:])
            return info
        elif marker == 0xD9:
            raise ValueError("no scan")
        i += 2 + seg_len


def build_decode(counts, syms) -> dict:
    table, code, k = {}, 0, 0
    for length in range(1, 17):
        for _ in range(counts[length - 1]):
            table[(length, code)] = syms[k]
            k += 1
            code += 1
        code <<= 1
    return table


def build_encode(counts, syms) -> dict:
    return {sym: (length, code) for (length, code), sym in build_decode(counts, syms).items()}


def geometry(info: dict):
    comps = {c["id"]: c for c in info["comps"]}
    hmax, vmax = max(c["h"] for c in info["comps"]), max(c["v"] for c in info["comps"])
    mx, my = -(-info["X"] // (8 * hmax)), -(-info["Y"] // (8 * vmax))
    order = [comps[s["id"]] for s in info["scomps"]]
    if len(order) == 1:   # non-interleaved: blocks cover the component itself, no MCU padding
        c = order[0]
        bw, bh = -(-(-(-info["X"] * c["h"] // hmax)) // 8), -(-(-(-info["Y"] * c["v"] // vmax)) // 8)
        return [(c, bw, bh)], "single"
    return [(c, mx * c["h"], my * c["v"]) for c in order], (mx, my)


def mcus(info: dict) -> list[list[tuple[int, int, int]]]:
    """Blocks grouped by MCU (restart intervals count MCUs)."""
    geo, mode = geometry(info)
    if mode == "single":
        return [[b] for b in block_order(info)]
    per = sum(c["h"] * c["v"] for c, _, _ in geo)
    order = block_order(info)
    return [order[i:i + per] for i in range(0, len(order), per)]


def segments(info: dict) -> list[list[tuple[int, int, int]]]:
    """Restart intervals: each is coded on its own (byte-aligned, DC prediction reset)."""
    ri, ms = info["restart"], mcus(info)
    if not ri:
        return [[b for m in ms for b in m]]
    return [[b for m in ms[i:i + ri] for b in m] for i in range(0, len(ms), ri)]


def block_order(info: dict):
    """The scan order of blocks as (component index, block x, block y)."""
    geo, mode = geometry(info)
    if mode == "single":
        _, bw, bh = geo[0]
        return [(0, x, y) for y in range(bh) for x in range(bw)]
    mx, my = mode
    out = []
    for y in range(my):
        for x in range(mx):
            for ci, (c, _, _) in enumerate(geo):
                for by in range(c["v"]):
                    for bx in range(c["h"]):
                        out.append((ci, x * c["h"] + bx, y * c["v"] + by))
    return out


# ------------------------------------------------------------------ Huffman layer
def unstuff(scan: bytes) -> np.ndarray:
    b = np.frombuffer(scan, dtype=np.uint8)
    keep = np.ones(len(b), dtype=bool)
    ff = np.flatnonzero(b[:-1] == 0xFF)
    keep[ff + 1] = False
    return np.unpackbits(b[keep])


def split_scan(scan: bytes) -> list[bytes]:
    out, start, j = [], 0, 0
    while j < len(scan) - 1:
        if scan[j] == 0xFF and 0xD0 <= scan[j + 1] <= 0xD7:
            out.append(scan[start:j])
            start = j + 2
            j += 2
        else:
            j += 1
    out.append(scan[start:])
    return out


def decode_scan(info: dict) -> list[np.ndarray]:
    geo, _ = geometry(info)
    segs = segments(info)
    parts = split_scan(info["scan"])
    if len(parts) != len(segs):
        raise ValueError("restart count mismatch")
    planes = [np.zeros((bh, bw, 64), dtype=np.int32) for _, bw, bh in geo]
    for part, blocks in zip(parts, segs):
        decode_segment(info, part, blocks, planes)
    return planes


def decode_segment(info: dict, part: bytes, blocks, planes) -> None:
    geo, _ = geometry(info)
    dc_t = [build_decode(*info["huff"][(0, s["td"])]) for s in info["scomps"]]
    ac_t = [build_decode(*info["huff"][(1, s["ta"])]) for s in info["scomps"]]
    bits = unstuff(part).tolist() if part else []
    pos, pred = 0, [0] * len(geo)

    def sym(table):
        nonlocal pos
        code = 0
        for length in range(1, 17):
            code = (code << 1) | bits[pos]
            pos += 1
            s = table.get((length, code))
            if s is not None:
                return s
        raise ValueError("bad Huffman code")

    def value(size):
        nonlocal pos
        if size == 0:
            return 0
        v = 0
        for _ in range(size):
            v = (v << 1) | bits[pos]
            pos += 1
        return v if v >= 1 << (size - 1) else v - (1 << size) + 1

    for ci, x, y in blocks:
        blk = planes[ci][y, x]
        pred[ci] += value(sym(dc_t[ci]))
        blk[0] = pred[ci]
        k = 1
        while k < 64:
            rs = sym(ac_t[ci])
            if rs == ZZ_EOB:
                break
            if rs == ZZ_ZRL:
                k += 16
                continue
            k += rs >> 4
            blk[k] = value(rs & 15)
            k += 1


def encode_scan(info: dict, planes: list[np.ndarray]) -> bytes:
    parts = [encode_segment(info, planes, blocks) for blocks in segments(info)]
    out = bytearray(parts[0])
    for m, part in enumerate(parts[1:]):
        out += bytes([0xFF, 0xD0 + (m % 8)]) + part
    return bytes(out)


def encode_segment(info: dict, planes: list[np.ndarray], blocks) -> bytes:
    dc_t = [build_encode(*info["huff"][(0, s["td"])]) for s in info["scomps"]]
    ac_t = [build_encode(*info["huff"][(1, s["ta"])]) for s in info["scomps"]]
    out: list[str] = []
    pred = [0] * len(planes)

    def put(table, symbol):
        length, code = table[symbol]
        out.append(format(code, f"0{length}b"))

    def put_value(v):
        if v == 0:
            return 0, ""
        size = abs(int(v)).bit_length()
        bits = v if v > 0 else v + (1 << size) - 1
        return size, format(bits, f"0{size}b")

    for ci, x, y in blocks:
        blk = planes[ci][y, x]
        size, b = put_value(int(blk[0]) - pred[ci])
        pred[ci] = int(blk[0])
        put(dc_t[ci], size)
        out.append(b)
        nz = np.flatnonzero(blk[1:]) + 1
        last = int(nz[-1]) if len(nz) else 0
        run = 0
        for k in range(1, last + 1):
            v = int(blk[k])
            if v == 0:
                run += 1
                continue
            while run > 15:
                put(ac_t[ci], ZZ_ZRL)
                run -= 16
            size, b = put_value(v)
            put(ac_t[ci], (run << 4) | size)
            out.append(b)
            run = 0
        if last < 63:
            put(ac_t[ci], ZZ_EOB)
    s = "".join(out)
    s += "1" * (-len(s) % 8)
    raw = int(s, 2).to_bytes(len(s) // 8, "big") if s else b""
    return raw.replace(b"\xff", b"\xff\x00")


# ------------------------------------------------------------------ Huffman layer in C (same bits, ~100x faster)
def _c_args(info: dict):
    import ctypes
    geo, mode = geometry(info)
    n = len(geo)
    I4 = ctypes.c_int * 4
    h, v = I4(*[c["h"] for c, _, _ in geo]), I4(*[c["v"] for c, _, _ in geo])
    bw, bh = I4(*[g[1] for g in geo]), I4(*[g[2] for g in geo])
    tabs = []
    for tc, key in ((0, "td"), (1, "ta")):
        counts, syms = bytearray(16 * n), bytearray(256 * n)
        for ci, s in enumerate(info["scomps"]):
            c, sy = info["huff"][(tc, s[key])]
            counts[16 * ci:16 * ci + 16] = bytes(c)
            syms[256 * ci:256 * ci + len(sy)] = bytes(sy)
        tabs += [bytes(counts), bytes(syms)]
    single = mode == "single"
    mx, my = (0, 0) if single else mode
    total = sum(64 * g[1] * g[2] for g in geo)
    return (n, int(single), mx, my, info["restart"], h, v, bw, bh, *tabs), total


def decode_planes(info: dict) -> np.ndarray:
    """The scan's coefficients as one int16 array: per component, 64 frequency planes of bh x bw blocks."""
    import ctypes
    args, total = _c_args(info)
    out = np.zeros(total, dtype=np.int16)
    fn = FZ.lib().fz_jpeg_decode
    fn.restype = ctypes.c_int
    err = fn(info["scan"], ctypes.c_size_t(len(info["scan"])), *args, out.ctypes.data_as(ctypes.c_void_p))
    if err:
        raise ValueError(f"Huffman decode failed ({err})")
    return out


def encode_planes(info: dict, planes: np.ndarray) -> bytes:
    import ctypes
    args, total = _c_args(info)
    planes = np.ascontiguousarray(planes, dtype=np.int16)
    cap = planes.size * 4 + 4096
    buf = ctypes.create_string_buffer(cap)
    fn = FZ.lib().fz_jpeg_encode
    fn.restype = ctypes.c_size_t
    n = fn(planes.ctypes.data_as(ctypes.c_void_p), *args, buf, ctypes.c_size_t(cap))
    if n > cap:
        raise ValueError("scan buffer too small")
    return buf.raw[:n]


def plane_sizes(info: dict) -> list[int]:
    geo, _ = geometry(info)
    return [64 * bw * bh for _, bw, bh in geo]


# ------------------------------------------------------------------ coefficient planes <-> FunnelZip
def to_streams(planes: list[np.ndarray]) -> list[bytes]:
    """Per component: 64 frequency planes, each the blocks in raster order, as int16."""
    return [np.ascontiguousarray(p.transpose(2, 0, 1)).astype("<i2").tobytes() for p in planes]


def from_streams(streams: list[bytes], shapes: list[tuple[int, int]]) -> list[np.ndarray]:
    return [np.frombuffer(s, dtype="<i2").astype(np.int32).reshape(64, bh, bw).transpose(1, 2, 0)
            for s, (bh, bw) in zip(streams, shapes)]


def _compress(blob: bytes) -> tuple[bytes, dict]:
    """Returns (payload, info). Payload byte 0: 1 = coefficient path, 0 = plain FunnelZip fallback."""
    try:
        info = parse(blob)
        flat = decode_planes(info)
        if encode_planes(info, flat) != info["scan"]:
            raise ValueError("re-encoded scan differs")
    except (ValueError, KeyError, IndexError) as e:
        enc = FZ.encode(blob, image_lens=False)
        return b"\x00" + enc.payload, {"path": "bytes", "why": str(e)}
    out = bytearray(b"\x01")
    meta = FZ.encode(info["header"] + info["trailer"]).payload
    put_varint(out, len(info["header"]))
    put_varint(out, len(meta))
    out += meta
    from concurrent.futures import ThreadPoolExecutor
    geo, _ = geometry(info)
    cuts = np.cumsum([0] + plane_sizes(info))
    jobs, ks = [], []
    for a, (_, bw, bh) in zip(cuts[:-1], geo):
        pl = bw * bh
        comp = flat[a:a + 64 * pl]
        jobs.append((comp[:pl].astype("<i2").tobytes(), (2, 1, bw * 2, 0)))           # DC: 16-bit, one value per block
        k = groups(63 * pl)
        ks.append(k)
        for r in np.array_split(np.arange(1, 64), k):                                # AC: one byte per coefficient
            jobs.append((ac_bytes(comp[r[0] * pl:(r[-1] + 1) * pl]), (1, 1, bw, pl)))

    def code(job):   # layout is known: stride = one block row, rhythm = one frequency plane
        st, (W, C, S, R) = job
        return FZ.encode(st, image_lens=False, fixed=(W, C, S, R, 0, 0, FZ.NO_WCM)).payload
    with ThreadPoolExecutor(min(len(jobs), os.cpu_count() or 1)) as ex:   # every group of every colour in parallel
        coded = list(ex.map(code, jobs))
    sizes, j = [], 0
    for k in ks:
        out.append(k)
        for e in coded[j:j + 1 + k]:
            put_varint(out, len(e))
            out += e
        sizes.append(sum(len(e) for e in coded[j:j + 1 + k]))
        j += 1 + k
    return bytes(out), {"path": "coefficients", "blocks": [(bh, bw) for _, bw, bh in geo], "stream_bytes": sizes,
                        "groups": ks, "header_bytes": len(meta)}


def groups(n: int) -> int:
    """AC planes are coded in groups, each by its own model in its own thread: about one group per MB of
    coefficients (low and high frequencies behave differently, so separate models also compress big photos better)."""
    k = 1
    while k < 8 and (k * 2) << 20 <= n:
        k *= 2
    return k


def ac_bytes(v: np.ndarray) -> bytes:
    """AC coefficients are small: v + 128 as one byte (1..255, linear, so the signal lenses still work);
    the rare |v| > 127 is written as 0 followed by the 16-bit value."""
    v = v.astype(np.int32)
    big = (v < -127) | (v > 127)
    if not big.any():
        return (v + 128).astype(np.uint8).tobytes()
    pos = np.arange(len(v)) + 2 * (np.cumsum(big) - big)
    out = np.empty(len(v) + 2 * int(big.sum()), dtype=np.uint8)
    out[pos[~big]] = v[~big] + 128
    e, w = pos[big], v[big] & 0xFFFF
    out[e], out[e + 1], out[e + 2] = 0, w & 255, w >> 8
    return out.tobytes()


def ac_values(b: bytes) -> np.ndarray:
    a = np.frombuffer(b, dtype=np.uint8)
    esc, nxt = [], -1
    for z in np.flatnonzero(a == 0).tolist():   # a 0 inside an escape's payload is not a marker
        if z >= nxt:
            esc.append(z)
            nxt = z + 3
    if not esc:
        return a.astype(np.int16) - 128
    e = np.array(esc)
    payload = np.zeros(len(a), dtype=bool)
    payload[e + 1] = payload[e + 2] = True
    vals = a[~payload].astype(np.int16) - 128
    vals[e - 2 * np.arange(len(e))] = (a[e + 1].astype(np.uint16) | a[e + 2].astype(np.uint16) << 8).view(np.int16)
    return vals


def decompress(payload: bytes) -> bytes:
    if payload[0] == 0:
        return FZ.decode(payload[1:])
    hlen, pos = get_varint(payload, 1)
    mlen, pos = get_varint(payload, pos)
    meta = FZ.decode(payload[pos:pos + mlen])
    pos += mlen
    header, trailer = meta[:hlen], meta[hlen:]
    info = parse(header + b"\xff\xd9")   # parse tables/geometry from the stored header
    parts, kinds = [], []
    for _ in geometry(info)[0]:
        k = payload[pos]
        pos += 1
        for i in range(1 + k):
            n, pos = get_varint(payload, pos)
            parts.append(payload[pos:pos + n])
            kinds.append(i > 0)
            pos += n
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(min(len(parts), os.cpu_count() or 1)) as ex:
        streams = list(ex.map(FZ.decode, parts))
    flat = np.concatenate([ac_values(st) if ac else np.frombuffer(st, dtype="<i2") for st, ac in zip(streams, kinds)])
    return header + encode_planes(info, flat) + trailer


def compress(blob: bytes) -> tuple[bytes, dict]:
    """Structure path, unless plain FunnelZip on the file's bytes is smaller (small or unusual files)."""
    body, info = _compress(blob)
    if info["path"] != "bytes" and len(blob) < (1 << 20):
        plain = b"\x00" + FZ.encode(blob, image_lens=False).payload
        if len(plain) < len(body):
            return plain, {"path": "bytes", "why": f"bytes path smaller than {info['path']} path"}
    return body, info
