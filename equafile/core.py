"""Variant registry + the .eqf container.

Every lossless variant module exposes:
    ID, NAME, SUMMARY, MAX_INPUT (bytes or None)
    encode(data: bytes) -> Encoded
    decode(payload: bytes) -> bytes
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

MAGIC = b"EQF1"


@dataclass
class Encoded:
    payload: bytes          # everything the decoder needs (besides the fixed decoder code)
    equation: str           # human readable form of the equation
    info: dict = field(default_factory=dict)


class TooBig(Exception):
    """Input is beyond what this variant can search in reasonable time."""


def registry() -> dict:
    from . import funnelzip, grammar, loop, polynomial, predictive, smallloop, symbolic
    mods = [polynomial, loop.LoopVariant, loop.SeedVariant, loop.TreeVariant,
            grammar, symbolic, predictive, smallloop, funnelzip]
    return {m.ID: m for m in mods}


def pack(variant_id: str, payload: bytes) -> bytes:
    vid = variant_id.encode()
    return MAGIC + bytes([len(vid)]) + vid + payload


def unpack(blob: bytes) -> tuple[str, bytes]:
    if blob[:4] != MAGIC:
        raise ValueError("not an .eqf file")
    n = blob[4]
    return blob[5:5 + n].decode(), blob[5 + n:]


def stored_size(payload: bytes) -> int:
    """Size we report: payload + 1 byte variant tag (magic/name are fixed decoder-side constants)."""
    return len(payload) + 1


def run(variant, data: bytes) -> dict:
    """Encode, decode, verify bit-exactness. Returns a result row."""
    row = {"variant": variant.ID, "name": variant.NAME, "original": len(data)}
    if variant.MAX_INPUT is not None and len(data) > variant.MAX_INPUT:
        row.update(status="skipped", reason=f"input > {variant.MAX_INPUT} B (search cost)")
        return row
    t0 = time.perf_counter()
    try:
        enc = variant.encode(data)
    except TooBig as e:
        row.update(status="infeasible", reason=str(e))
        return row
    t1 = time.perf_counter()
    out = variant.decode(enc.payload)
    t2 = time.perf_counter()
    ok = hashlib.sha256(out).digest() == hashlib.sha256(data).digest()
    row.update(status="ok" if ok else "MISMATCH", stored=stored_size(enc.payload),
               ratio=stored_size(enc.payload) / max(1, len(data)),
               enc_s=round(t1 - t0, 4), dec_s=round(t2 - t1, 4),
               equation=enc.equation, info=enc.info)
    return row
