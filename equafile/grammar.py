"""V5a — nested rules (Re-Pair style straight-line program).

Repeatedly replace frequent symbol pairs with a new rule R = A B. Rules may use
earlier rules, so the file becomes a tree of reusable sub-equations.
Output is fixed-width packed (no entropy coding), which keeps it honest but simple.
"""
from __future__ import annotations

from collections import Counter

from .bits import BitReader, BitWriter, get_varint, put_varint
from .core import Encoded

ID = "v5-grammar"
NAME = "V5a Nested rules R = A B"
SUMMARY = "Re-Pair style grammar: sub-equations that reuse sub-equations"
MAX_INPUT = 1 << 20
MIN_COUNT = 3       # a rule costs 2 symbols, so it must replace >= 3 pairs
BATCH = 512         # disjoint pairs replaced per pass


def build(data: bytes):
    seq = list(data)
    rules: list[tuple[int, int]] = []
    nxt = 256
    while len(seq) > 1:
        counts = Counter(zip(seq, seq[1:]))
        picked, used = {}, set()
        for (a, b), c in counts.most_common():
            if c < MIN_COUNT or len(picked) >= BATCH:
                break
            if a in used or b in used:
                continue  # keep picked pairs disjoint so replacements never interfere
            picked[(a, b)] = nxt
            rules.append((a, b))
            nxt += 1
            used.update((a, b))
        if not picked:
            break
        out, i, n = [], 0, len(seq)
        while i < n:
            if i + 1 < n and (seq[i], seq[i + 1]) in picked:
                out.append(picked[(seq[i], seq[i + 1])])
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = out
    return rules, seq


def _expand_all(rules):
    exp = [bytes([i]) for i in range(256)]
    for a, b in rules:
        exp.append(exp[a] + exp[b])
    return exp


def _name(sym: int) -> str:
    if sym < 256:
        ch = chr(sym)
        return repr(ch) if ch.isprintable() else f"0x{sym:02x}"
    return f"R{sym}"


def encode(data: bytes) -> Encoded:
    rules, seq = build(data)
    nsym = 256 + len(rules)
    w = max(8, (nsym - 1).bit_length())
    out = bytearray()
    put_varint(out, len(rules))
    put_varint(out, len(seq))
    bw = BitWriter()
    for a, b in rules:
        bw.write(a, w)
        bw.write(b, w)
    for s in seq:
        bw.write(s, w)
    out += bw.to_bytes()

    exp = _expand_all(rules)
    lines = [f"{len(rules)} rules, top equation has {len(seq)} symbols, {w} bits per symbol"]
    for k, (a, b) in enumerate(rules[:12]):
        text = exp[256 + k][:24]
        lines.append(f"R{256 + k} = {_name(a)} {_name(b)}    -> {text!r}")
    if len(rules) > 12:
        lines.append("…")
    top = " ".join(_name(s) for s in seq[:16]) + (" …" if len(seq) > 16 else "")
    lines.append(f"FILE = {top}")
    return Encoded(bytes(out), "\n".join(lines), {"rules": len(rules), "top_len": len(seq), "sym_bits": w})


def decode(payload: bytes) -> bytes:
    nr, pos = get_varint(payload, 0)
    ns, pos = get_varint(payload, pos)
    w = max(8, (256 + nr - 1).bit_length())
    br = BitReader(payload[pos:])
    rules = [(br.read(w), br.read(w)) for _ in range(nr)]
    exp = _expand_all(rules)
    return b"".join(exp[br.read(w)] for _ in range(ns))
