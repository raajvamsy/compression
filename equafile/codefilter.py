"""Code lens pre-filter: rewrite relative jump/call targets as absolute ones (reversible).

Machine code says "call the function N bytes from here", so the same call looks different at every
call site. Absolute targets turn repeated calls into repeated bytes. Same idea as xz's BCJ filters.
"""
from __future__ import annotations

import numpy as np

NONE, ARM64, X86, BOTH = 0, 1, 2, 3


def arm64(data: bytes, encode: bool) -> bytes:
    """ARM64 BL (imm26, in words) and ADRP (imm21, in 4 KiB pages), relative to the instruction,
    rewritten as absolute (same rules as xz's ARM64 BCJ filter, ADRP only within +/-512 MiB)."""
    n4 = len(data) // 4 * 4
    w = np.frombuffer(data[:n4], dtype="<u4").copy()
    pos = np.arange(len(w), dtype=np.uint32) * np.uint32(4)
    sgn = (lambda a, b: a + b) if encode else (lambda a, b: a - b)
    bl = (w >> np.uint32(26)) == np.uint32(0x25)
    new = sgn(w & np.uint32(0x03FFFFFF), pos >> np.uint32(2))
    w[bl] = (w[bl] & np.uint32(0xFC000000)) | (new[bl] & np.uint32(0x03FFFFFF))
    adrp = (w & np.uint32(0x9F000000)) == np.uint32(0x90000000)
    src = ((w >> np.uint32(29)) & np.uint32(3)) | ((w >> np.uint32(3)) & np.uint32(0x001FFFFC))
    adrp &= ((src + np.uint32(0x00020000)) & np.uint32(0x001C0000)) == 0
    dest = sgn(src, pos >> np.uint32(12))
    out = (w & np.uint32(0x9000001F)) | ((dest & np.uint32(3)) << np.uint32(29)) | ((dest & np.uint32(0x0003FFFC)) << np.uint32(3)) \
        | ((np.uint32(0) - (dest & np.uint32(0x00020000))) & np.uint32(0x00E00000))
    w[adrp] = out[adrp]
    return w.tobytes() + data[n4:]


def x86(data: bytes, encode: bool) -> bytes:
    """x86 CALL (E8) / JMP (E9) rel32, converted only for near targets (top byte 0x00/0xFF), as in xz's
    BCJ x86 filter. The result is sign-extended from bit 24, so its top byte is again 0x00/0xFF and the
    decoder can apply exactly the same test: reversible."""
    b = bytearray(data)
    i, n = 0, len(b) - 4
    while i < n:
        if b[i] in (0xE8, 0xE9) and b[i + 4] in (0x00, 0xFF):
            v = int.from_bytes(b[i + 1:i + 5], "little")
            d = (v + i + 5) if encode else (v - i - 5)
            d &= 0x01FFFFFF
            if d & 0x01000000:
                d |= 0xFE000000
            b[i + 1:i + 5] = d.to_bytes(4, "little")
            i += 5
        else:
            i += 1
    return bytes(b)


def apply(data: bytes, mode: int, encode: bool) -> bytes:
    if mode in (ARM64, BOTH):
        data = arm64(data, encode) if encode else data
    if mode in (X86, BOTH):
        data = x86(data, encode)
    if mode in (ARM64, BOTH) and not encode:
        data = arm64(data, encode)
    return data
