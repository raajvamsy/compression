"""Build FunnelZip's primer: 122,880 bytes of English docs + Python code read by the model before every
primed file (like brotli's built-in 122 KB dictionary). Taken from stdlib parts that no test file uses:
pydoc_data/topics.py (English reference docs) and idlelib/*.py (code). Deterministic."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZE = 122_880          # divisible by 2, 3, 4, 8 so sample/channel alignment is kept after priming


def main() -> None:
    lib = Path(os.__file__).parent
    docs = (lib / "pydoc_data" / "topics.py").read_bytes()
    start = docs.find(b"'assert'")   # skip the file header, start inside the prose
    docs = docs[max(0, start):][: SIZE // 2]
    code = bytearray()
    for p in sorted((lib / "idlelib").glob("*.py")):
        code += p.read_bytes()
        if len(code) >= SIZE - len(docs):
            break
    prime = (docs + bytes(code))[:SIZE]
    assert len(prime) == SIZE
    out = ROOT / "funnelzip" / "prime.bin"
    out.write_bytes(prime)
    print("wrote", out.relative_to(ROOT), len(prime), "bytes")


if __name__ == "__main__":
    main()
