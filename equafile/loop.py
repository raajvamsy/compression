"""V2 loop, V3 seed + loop, V4 hierarchy (FunnelBrain tree).

The equation is a fixed 64-bit mixing map  f(x) = splitmix64(x)  (a bijection,
so the loop never gets stuck in a short cycle).

V2  x0 = 1, x <- f(x) repeated L times, file = top n bits of x.     store: L
V3  y = f^R(s) (R = loop count), file = top n bits of y.            store: s, R
V4  file -> 16-bit chunks -> each chunk's smallest seed -> new bit string
    -> again ...; the top level is the master equation.           store: top level + sizes
"""
from __future__ import annotations

import numpy as np

from .bits import BitReader, BitWriter, get_varint, put_varint
from .core import Encoded, TooBig

M64 = (1 << 64) - 1
GOLD = 0x9E3779B97F4A7C15
K1, K2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
ROUNDS = 4  # V3/V4 loop count


def mix(x: int) -> int:
    x = (x + GOLD) & M64
    x = ((x ^ (x >> 30)) * K1) & M64
    x = ((x ^ (x >> 27)) * K2) & M64
    return x ^ (x >> 31)


def mix_np(x: np.ndarray) -> np.ndarray:
    x = x + np.uint64(GOLD)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(K1)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(K2)
    return x ^ (x >> np.uint64(31))


def f_rounds(s: int, rounds: int = ROUNDS) -> int:
    for _ in range(rounds):
        s = mix(s)
    return s


def f_rounds_np(s: np.ndarray, rounds: int = ROUNDS) -> np.ndarray:
    for _ in range(rounds):
        s = mix_np(s)
    return s


# ---------------------------------------------------------------- V2: the loop
def loop_search(target: int, nbits: int, max_steps: int | None = None, trace: int = 0):
    """Walk x0=1 -> f -> f ... until the top nbits equal target. Returns (L, trace)."""
    x, shift, L = 1, 64 - nbits, 0
    head = []
    limit = max_steps or (1 << 62)
    while L < limit:
        x = mix(x)
        L += 1
        if len(head) < trace:
            head.append((L, x >> shift))
        if x >> shift == target:
            return L, head
    return None, head


def loop_eval(L: int, nbits: int) -> int:
    x = 1
    for _ in range(L):
        x = mix(x)
    return x >> (64 - nbits)


class LoopVariant:
    ID = "v2-loop"
    NAME = "V2 Loop: x0=1, x<-f(x), L times"
    SUMMARY = "Raaj's loop: start at 1, feed the output back in, store only the loop count L"
    MAX_INPUT = 3  # expected ~2^(8n) steps in pure Python
    MAX_BITS = 24

    @staticmethod
    def encode(data: bytes) -> Encoded:
        n = len(data) * 8
        if n == 0 or n > LoopVariant.MAX_BITS:
            raise TooBig(f"needs ~2^{n} loop steps")
        L, _ = loop_search(int.from_bytes(data, "big"), n)
        out = bytearray()
        put_varint(out, len(data))
        put_varint(out, L)
        eq = (f"x0 = 1;  x <- splitmix64(x)  repeated L = {L:,} times;  file = top {n} bits of x\n"
              f"L needs {L.bit_length()} bits to store, the file has {n} bits")
        return Encoded(bytes(out), eq, {"L": L, "L_bits": L.bit_length(), "file_bits": n})

    @staticmethod
    def decode(payload: bytes) -> bytes:
        nbytes, pos = get_varint(payload, 0)
        L, _ = get_varint(payload, pos)
        return loop_eval(L, nbytes * 8).to_bytes(nbytes, "big")


# ---------------------------------------------------------------- V3: seed + loop
def seed_search(target: int, nbits: int, rounds: int = ROUNDS, batch: int = 1 << 22,
                max_seed: int = 1 << 34) -> int | None:
    """Smallest seed s with top nbits of f^rounds(s) == target (vectorised)."""
    shift = np.uint64(64 - nbits)
    t = np.uint64(target)
    for start in range(0, max_seed, batch):
        s = np.arange(start, start + batch, dtype=np.uint64)
        hit = np.flatnonzero((f_rounds_np(s, rounds) >> shift) == t)
        if hit.size:
            return int(start + hit[0])
    return None


class SeedVariant:
    ID = "v3-seed"
    NAME = "V3 Seed + loop: f^R(s)"
    SUMMARY = "Search the start input s; store s and the loop count R"
    MAX_INPUT = 3
    MAX_BITS = 28

    @staticmethod
    def encode(data: bytes) -> Encoded:
        n = len(data) * 8
        if n == 0 or n > SeedVariant.MAX_BITS:
            raise TooBig(f"needs ~2^{n} seed trials")
        s = seed_search(int.from_bytes(data, "big"), n)
        out = bytearray()
        put_varint(out, len(data))
        put_varint(out, ROUNDS)
        put_varint(out, s)
        eq = (f"y = splitmix64^{ROUNDS}(s),  s = {s:,};  file = top {n} bits of y\n"
              f"s needs {s.bit_length()} bits, the file has {n} bits")
        return Encoded(bytes(out), eq, {"seed": s, "seed_bits": s.bit_length(), "file_bits": n})

    @staticmethod
    def decode(payload: bytes) -> bytes:
        nbytes, pos = get_varint(payload, 0)
        R, pos = get_varint(payload, pos)
        s, _ = get_varint(payload, pos)
        return (f_rounds(s, R) >> (64 - nbytes * 8)).to_bytes(nbytes, "big")


# ---------------------------------------------------------------- V4: the FunnelBrain tree
CHUNK = 16
WIDTHS = (16, 17, 19, 21)  # 2-bit prefix picks the seed width
_TABLE: np.ndarray | None = None


def _inverse_table() -> np.ndarray:
    """inv[v] = smallest seed whose equation prints the 16-bit value v."""
    global _TABLE
    if _TABLE is None:
        seeds = np.arange(1 << WIDTHS[-1], dtype=np.uint64)
        outs = (f_rounds_np(seeds) >> np.uint64(64 - CHUNK)).astype(np.int64)
        vals, first = np.unique(outs, return_index=True)
        if len(vals) != 1 << CHUNK:
            raise RuntimeError("seed table does not cover every 16-bit value")
        _TABLE = first.astype(np.int64)
    return _TABLE


def _level_up(bits: bytes, nbits: int) -> tuple[bytes, int, list[int]]:
    """One funnel level: chunk -> smallest seed per chunk -> packed seeds."""
    inv = _inverse_table()
    br, bw, seeds = BitReader(bits), BitWriter(), []
    for i in range(0, nbits, CHUNK):
        w = min(CHUNK, nbits - i)
        chunk = br.read(w) << (CHUNK - w)  # zero-pad the last chunk
        s = int(inv[chunk])
        e = next(k for k, W in enumerate(WIDTHS) if s < (1 << W))
        bw.write(e, 2)
        bw.write(s, WIDTHS[e])
        seeds.append(s)
    return bw.to_bytes(), bw.total, seeds


def _level_down(bits: bytes, child_nbits: int) -> bytes:
    br, bw = BitReader(bits), BitWriter()
    for i in range(0, child_nbits, CHUNK):
        s = br.read(WIDTHS[br.read(2)])
        v = f_rounds(s) >> (64 - CHUNK)
        w = min(CHUNK, child_nbits - i)
        bw.write(v >> (CHUNK - w), w)
    return bw.to_bytes()


class TreeVariant:
    ID = "v4-tree"
    NAME = "V4 FunnelBrain tree of sub-equations"
    SUMMARY = "16-bit chunks -> seeds -> seeds of seeds ...; master equation on top"
    MAX_INPUT = 1 << 20
    LEVELS = 1

    @staticmethod
    def build(data: bytes, levels: int):
        sizes, layers = [len(data) * 8], []
        bits, nbits = data, len(data) * 8
        for _ in range(levels):
            bits, nbits, seeds = _level_up(bits, nbits)
            sizes.append(nbits)
            layers.append(seeds)
        return bits, sizes, layers

    @staticmethod
    def encode(data: bytes, levels: int | None = None) -> Encoded:
        levels = levels or TreeVariant.LEVELS
        top, sizes, layers = TreeVariant.build(data, levels)
        out = bytearray()
        put_varint(out, levels)
        for s in sizes:
            put_varint(out, s)
        out += top
        lines = [f"level 0 (the file): {sizes[0]} bits"]
        for k, s in enumerate(sizes[1:], 1):
            seeds = layers[k - 1]
            show = ", ".join(f"{v:,}" for v in seeds[:6]) + (" …" if len(seeds) > 6 else "")
            lines.append(f"level {k}: {len(seeds)} sub-equations f^{ROUNDS}(s), s = [{show}] -> {s} bits "
                         f"({s / sizes[0]:.3f}x the file)")
        lines.append("the funnel never narrows: each seed costs more bits than the chunk it prints")
        return Encoded(bytes(out), "\n".join(lines), {"level_bits": sizes})

    @staticmethod
    def decode(payload: bytes) -> bytes:
        levels, pos = get_varint(payload, 0)
        sizes = []
        for _ in range(levels + 1):
            s, pos = get_varint(payload, pos)
            sizes.append(s)
        bits = payload[pos:]
        for k in range(levels, 0, -1):
            bits = _level_down(bits, sizes[k - 1])
        return bits[: sizes[0] // 8]
