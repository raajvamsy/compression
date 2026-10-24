"""Small byte/bit helpers shared by every equation variant."""
from __future__ import annotations


def put_varint(out: bytearray, v: int) -> None:
    """Unsigned LEB128."""
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return


def get_varint(buf: bytes, pos: int) -> tuple[int, int]:
    v = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        v |= (b & 0x7F) << shift
        if b < 0x80:
            return v, pos
        shift += 7


def zigzag(v: int) -> int:
    return (v << 1) if v >= 0 else ((-v) << 1) - 1


def unzigzag(v: int) -> int:
    return (v >> 1) if not v & 1 else -((v + 1) >> 1)


class BitWriter:
    """MSB-first bit packer (linear time)."""

    def __init__(self) -> None:
        self.out = bytearray()
        self.acc = 0
        self.n = 0  # bits held in acc
        self.total = 0

    def write(self, value: int, width: int) -> None:
        self.acc = (self.acc << width) | (value & ((1 << width) - 1))
        self.n += width
        self.total += width
        while self.n >= 8:
            self.n -= 8
            self.out.append((self.acc >> self.n) & 0xFF)
        self.acc &= (1 << self.n) - 1

    def to_bytes(self) -> bytes:
        if self.n:
            return bytes(self.out) + bytes([(self.acc << (8 - self.n)) & 0xFF])
        return bytes(self.out)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def read(self, width: int) -> int:
        end = self.pos + width
        b0, b1 = self.pos >> 3, (end + 7) >> 3
        chunk = int.from_bytes(self.data[b0:b1], "big")
        self.pos = end
        return (chunk >> (b1 * 8 - end)) & ((1 << width) - 1)


def bytes_to_bits(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)
