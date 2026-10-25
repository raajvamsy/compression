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

The C coder (`ffic.c`) is built automatically on first use (`cc -O2 -march=native -pthread`).

| input | how it is stored | comes back as |
|---|---|---|
| 8-bit RGB (PNG, TIFF, BMP, PPM, WebP lossless, ...) | one FFIC2 stream; if it has 256 colours or fewer, also tried as palette + colour-number map, smaller kept | the same RGB pixels |
| 8-bit grey | one layer (`planes=1`, 3-4x faster than coding copies) | grey |
| RGB or grey + transparency (RGBA, LA) | colour + alpha as two layers in an FFX1 container (constant alpha: one number) | the same mode, same pixels and transparency |
| palette (P), with or without transparency | palette sorted by brightness + the map of colour numbers as a grey layer | the same palette image: same numbers, palette, transparency |
| 16-bit grey (depth, elevation, scientific) | one layer coded with exactly the bits it uses (`bits=9..16`) | 16-bit grey |
| video `.y4m` (8-bit, 4:2:0 / 4:2:2 / 4:4:4 / mono) | Y, U, V layers in their native sizes, each with its frames stacked (`frameh=`), coded in parallel | a byte-identical `.y4m` |

Refused rather than silently changed: 16-bit colour, float, CMYK and other modes.

| preset | what it runs | speed (768x512 photo, one core per layer) |
|---|---|---|
| `max` | best coder + terrain balls; colour layers one after another | ~5.7 s to encode, ~5.8 s to decode |
| `fast` | the same model with the three colour layers on three threads | ~2.1 s / ~2.5 s, files ~0.05% bigger |
| `archive` | per image: 7 reversible tricks (inverted, other colour orders, mirrored, flipped) with the main brain, the best two also with the whole-colour brain; smallest kept | ~9x slower to encode; decoding as fast as max (or ~2.5x when the whole-colour brain wins) |

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

## Whole folders (like zip)

```
python ffic.py pack   photos/ photos.ffa        # every image by FFIC (4 at a time), every other file by xz
python ffic.py unpack photos.ffa restored/
```

Images keep every pixel and their metadata (colour profile, EXIF, DPI, PNG text); other files (JPEGs, documents, animated GIFs, anything FFIC cannot store exactly) keep their exact bytes. Folder of the 20 test photos as ordinary PNGs (`experiments/BENCHFOLDER.md`):

| method | archive | vs the PNG folder |
|---|---|---|
| folder of 20 PNGs | 18,053,552 B | — |
| zip -9 | 18,015,523 B | −0.2% |
| 7z LZMA2 ultra | 17,747,598 B | −1.7% |
| every photo converted to JPEG XL e9 | 10,481,322 B | −41.9% |
| `ffic.py pack` (max) | 10,329,410 B | −42.8% (47 s to pack, 50 s to unpack) |
| **`ffic.py pack --preset archive`** | **10,254,187 B** | **−43.2%**, 2.17% smaller than JPEG XL (610 s to pack, 72 s to unpack) |

## Formats benchmarks

| test | best other tool | FFIC | details |
|---|---|---|---|
| 16-bit elevation, Grand Canyon (3601x3601) | JPEG XL e9: 3,929,750 B | **3,732,532 B, −5.0%** | `experiments/BENCHDEM.md` |
| 16-bit elevation, French Alps | JPEG XL e9: 5,246,884 B | **4,895,779 B, −6.7%** | |
| 16-bit elevation, Swiss Alps | JPEG XL e9: 5,090,414 B | **4,775,794 B, −6.2%** | |
| RGBA with soft transparency (768x512) | JPEG XL e9: 332,446 B | **325,917 B, −2.0%** | `experiments/BENCHFORMATS.md` |
| palette image, 32 colours | JPEG XL e9: 81,290 B (PNG 109,623 B) | **78,409 B, −3.5%** | |
| palette + transparency | JPEG XL e9: 118,907 B | **115,300 B, −3.0%** | |
| video in native YUV 4:2:0, akiyo | x265 lossless: 588,367 B | **518,041 B, −12.0%** | |
| video in native YUV 4:2:0, foreman | x264 lossless: 1,556,770 B | **1,503,346 B, −3.4%** | |
| video in native YUV 4:2:0, mobile (fast pan) | x265 lossless: 1,972,752 B | 2,109,880 B, **+7.0% (worse)** | needs motion search |

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
