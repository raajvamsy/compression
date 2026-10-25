# FFIC: FunnelField Image Coder

A lossless image compressor built from ideas in FunnelBrain (a model of how a brain learns). Every pixel comes back exactly; each file carries a checksum that proves it.

## Use it

```
pip install pillow numpy
python ffic.py encode photo.png photo.ffic            # preset max: smallest files
python ffic.py encode photo.png photo.ffic --preset fast
python ffic.py decode photo.ffic photo.png
python ffic.py info   photo.ffic
python ffic.py test   photo.png                       # encode + decode + compare, both presets
python selftest.py                                    # awkward shapes x 3 modes + damaged-file check
```

The C coder (`ffic.c`) is built automatically on first use (`cc -O2 -march=native -pthread`). Input: any 8-bit RGB or grey image Pillow can read. Grey images come back grey (the copied colour layers cost about 250 bytes). Images with transparency and 16-bit images are refused rather than silently changed.

| preset | what it runs | speed (768x512 photo, one core per layer) |
|---|---|---|
| `max` | best coder + terrain balls; colour layers one after another | ~5.7 s to encode, ~5.8 s to decode |
| `fast` | the same model with the three colour layers on three threads | ~2.1 s / ~2.5 s, files ~0.05% bigger |

Decoding takes as long as encoding (the decoder runs the same model). That is the main weakness today.

## Results (real files, 20 photos: 12 Pexels crops 1024x768 + 8 Kodak 768x512)

| | total bytes | vs JPEG XL lossless, effort 9 |
|---|---|---|
| JPEG XL e9 | 10,481,322 | reference |
| FFIC `max` | 10,327,954 | **−1.46%** |
| FFIC best of `max` / pipeline / whole-colour, per photo | 10,273,344 (earlier run, without terrain balls) | **−1.98%** |

Smaller on the 12 modern photos (about −2.8%), still about 1.3% bigger on the 8 old Kodak film scans. All files decode bit-exact (`experiments/ROUNDTRIP.md`). Every experiment behind these numbers is logged in `experiments/LOG.md`; the ideas and their results are in `../docs/IDEAS_FUNNELBRAIN.md`.

## Beyond the 20 test photos

| test | best other tool | FFIC | details |
|---|---|---|---|
| 4K photos (4 x 3840x2160 crops of camera originals) | JPEG XL e9: 25,159,623 B | **23,922,822 B, −4.92%** (per photo −1.8% to −9.8%) | `experiments/BENCH4K.md` |
| Lossless video, still camera (akiyo, 30 frames) | x265 lossless: 1,569,264 B | **1,085,865 B, −30.8%** | `experiments/BENCHVIDEO.md` |
| Lossless video, handheld (foreman) | FFV1: 2,477,057 B | **2,236,692 B, −9.7%** | |
| Lossless video, fast pan (mobile) | FFV1: 4,026,264 B | **3,724,511 B, −7.5%** | |

The 4K crops come from camera JPEGs and keep their 8x8 JPEG structure, which may favour FFIC; JPEG XL's own JPEG-recompression mode does not apply to cropped pixels. FFIC is far slower than all of these tools (minutes per 4K photo, under one video frame per second), which is the main thing to fix before it is practical.

Video: `ffic file.rgb W H*frames <options> frameh=H out=clip.ffic` (frames stacked vertically, rgb24).

## How it works (short)

1. Green, then red, then blue. Each pixel is guessed by a team: simple guessers (same as left, same as above, keep the slope), learned funnels (nested by how busy, edge direction and brightness, a child funnel starts from its parent), material lines and material funnels (colour physics), and terrain balls (the decoded image as a landscape the balls roll over).
2. A referee blends the team by who was right nearby; funnels learn calmly (surprising pixels nudge them gently).
3. The leftover is coded as a few yes/no questions; each question's odds are mixed from many contexts and corrected by a second checker; a binary arithmetic coder writes the bits.
4. The decoder runs the identical model and uses the file only to settle each question.

## File format (FFIC2)

`"FFIC2\0"`, width, height (uint32), option-string length + option string (the model settings; the decoder rebuilds the same model from it), three stream lengths, the three arithmetic-coded streams (G, R, B), CRC-32 of the RGB pixels. Little-endian.

## Files

| file | what |
|---|---|
| `ffic.c` | the coder and decoder |
| `ffic.py` | command-line tool (PNG in/out, presets, info, test) |
| `selftest.py` | round-trip tests for awkward shapes and modes |
| `exp.py`, `experiments.py` | experiment harness and registry (`experiments/log.jsonl`, `LOG.md`) |
| `roundtrip.py` | real-file round trip on the 20 test photos (`experiments/ROUNDTRIP.md`) |
| `fold.py`, `fold2.py`, `slices.py`, `slabs.py`, `wavetree.py`, `wave_exact.py`, `equation.py`, `worlds*.py`, `strips.py` | research experiments (folding, slicing, waves, equations, parallel worlds, strips) |
