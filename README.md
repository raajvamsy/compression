# EquaFile

Store a file as an equation that prints it back, POC + benchmarks.

* Idea, theory, prior art, approaches: [docs/IDEA.md](docs/IDEA.md)
* Results and verdict: [docs/REPORT.md](docs/REPORT.md) (raw tables: [docs/REPORT_DATA.md](docs/REPORT_DATA.md))
* Charts: [outputs/visuals/report.html](outputs/visuals/report.html) (open in a browser)
* **FunnelZip**, the combined compressor: [docs/DESIGN_FUNNELZIP.md](docs/DESIGN_FUNNELZIP.md) · visual [outputs/visuals/funnelzip.html](outputs/visuals/funnelzip.html)
* Other visuals: [result_explained.html](outputs/visuals/result_explained.html), [small_loop_test.html](outputs/visuals/small_loop_test.html), [waves_resonance.html](outputs/visuals/waves_resonance.html)

## Setup

```bash
uv venv .venv --python 3.13 && uv pip install --python .venv/bin/python numpy pillow zstandard brotli scipy
```

## App

```bash
.venv/bin/python app/server.py
```

Open http://localhost:8765. It has three parts: an animated loop search for a short text, every equation family vs gzip/xz/brotli for any file, and an image as a sum of cosines vs JPEG/WebP at the same size.

## CLI

```bash
.venv/bin/python -m equafile show samples/text/sensor.csv --equations
.venv/bin/python -m equafile encode samples/text/counter.bin
.venv/bin/python -m equafile decode samples/text/counter.bin.eqf
.venv/bin/python -m equafile loop hi
.venv/bin/python -m equafile encode samples/audio/system_sounds.wav -v v10-funnelzip
```

## Layout

| path | what |
|---|---|
| `equafile/polynomial.py` | V1 one polynomial `f(i) = byte_i` |
| `equafile/loop.py` | V2 loop, V3 seed + loop, V4 FunnelBrain tree |
| `equafile/grammar.py` | V5a nested rules |
| `equafile/symbolic.py` | V5b segment equations `a + b·i`, repeats |
| `equafile/predictive.py` | V6 context-mixing model + arithmetic coder |
| `equafile/waves.py` | V7 cosine-series equations for image / audio / video |
| `equafile/smallloop.py` | V8 single-digit loops + simple equations |
| `equafile/resonance.py` | V9 waves found by resonance, resonant filter |
| `funnelzip/funnelzip.c`, `equafile/funnelzip.py` | V10 FunnelZip: all lenses mixed (C core, auto-built with clang) |
| `equafile/baselines.py` | gzip, bzip2, xz, zstd, brotli, PNG, JPEG, WebP, FLAC, MP3, Opus, H.264 |
| `bench/` | samples, benchmark runner, report generator |
