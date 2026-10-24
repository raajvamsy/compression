# EquaFile POC — results and verdict

**Date:** 2026-10-16 · Idea and theory: [IDEA.md](IDEA.md) · All numbers: [REPORT_DATA.md](REPORT_DATA.md) · Charts: [outputs/visuals/report.html](../outputs/visuals/report.html)

## TL;DR

| Question | Answer |
|---|---|
| Can every file be written as an equation and printed back exactly? | **Yes.** All 7 variants decoded every test file bit-exact (SHA-256 checked). |
| Does Raaj's loop (`x ← f(x)`, store only the loop count) make files smaller? | **No.** The loop count needs as many bits as the file; with its length it needs ~9 bits *more*. Each extra file bit doubles the search. |
| Does the FunnelBrain tree of sub-equations narrow to one master equation? | **No.** Each level grows ~16% (a seed costs ~18.5 bits per 16-bit chunk). After 5 levels the tree is 2.15× the file. |
| Do equations ever beat today's compressors? | **On structured data, yes:** a counting pattern became 6 bytes (brotli: 217). A sensor CSV beat xz by 3%. On ordinary text, code, photos and video they lose. |
| Lossy (image/audio/video)? | The image equation is about as good as **JPEG** but 2–4 dB worse than **WebP**. Video is 3–10 dB worse than **H.264**. Audio wins on SNR, but SNR is the wrong test for MP3/Opus (listening files provided). |

**Verdict:** the idea *works as a representation* but *does not work as general compression*. That is not a flaw of the POC. The counting argument guarantees it for any lossless scheme (IDEA.md §4). The parts of the intuition that do pay off already exist in compressors: equations that capture **structure**, **reusable sub-equations**, and **prediction**.

## 1. The loop and the seed (V2, V3)

`hi` is found after **L = 111,621** loop steps. That is a 17-bit number for a 16-bit file. Over 5 random targets per size:

| file bits | V2 loop count | V3 seed | V3 incl. length prefix | V3 search time |
|---:|---:|---:|---:|---:|
| 8 | 8.0 | 7.4 | 13.2 | 0.03 s |
| 16 | 16.4 | 15.4 | 23.6 | 0.04 s |
| 24 | – | 23.4 | 32.4 | 0.14 s |
| 28 | – | 29.0 | 38.0 | 3.9 s |

* Some targets get lucky and land a few bits under the file size. The decoder must also know how long the number is, and with that the average is always above break-even (Kraft inequality).
* Time doubles per bit: 28 bits took 3.9 s on a vectorised search. A 1 KB file means ~2⁸¹⁹² steps.

## 2. The FunnelBrain tree (V4)

| file | level 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| hi.txt | 16 | 21 | 37 | 54 | 76 | 91 |
| license.txt | 110,472 | 128,991 | 150,166 | 174,843 | 203,522 | 237,199 |
| random.bin | 32,768 | 38,276 | 44,661 | 52,025 | 60,556 | 70,496 |

A 16-bit chunk has 65,536 possible values, so the seed that prints it must also pick one of 65,536. Writing that choice down costs ≥ 16 bits. With the length tag it costs ~18.5. Hierarchy re-arranges bits; it cannot remove them.

## 3. Lossless: where equations win and lose (stored bytes)

| file | size | best equation | best existing | result |
|---|---:|---|---|---|
| counter.bin (0..255 ×16) | 4,096 | **6** (V5b `f(i)=i mod 256`) | 217 (brotli) | **equation 36× smaller** |
| sensor.csv | 32,918 | **6,374** (V6 predictive) | 6,564 (xz) | **equation 3% smaller** |
| sentence.txt | 77 | **57** (V5b) | 61 (brotli) | equation smaller (tiny-file header overhead only) |
| license.txt | 13,809 | 5,305 (V6) | 3,264 (brotli) | existing 38% smaller |
| typing.py | 132,718 | 36,066 (V6) | 28,616 (brotli) | existing 21% smaller |
| random.bin | 4,096 | 4,100 (V5b) | 4,100 (brotli) | tie — nothing compresses noise |
| beach photo, raw RGB | 83,460 | 48,273 (V6) | 33,720 (WebP lossless) | existing 30% smaller |
| synthetic image, raw RGB | 196,608 | 82,675 (V6) | 232 (WebP lossless) | existing far smaller |
| chord, raw PCM | 264,600 | 219,547 (V6) | 199,193 (bzip2); FLAC 48,824 | existing smaller |

Files of 1 KB or more: **2 of 9 won**. Both wins are files that really *are* equations or are very regular: a counter, and a table of slowly varying numbers. The "one polynomial" (V1) is always 1.1–2.1× the file. Its field must grow with the file, so the coefficients get wider.

## 4. Lossy

**Images.** Equal-size PSNR gap, equation minus codec:

| image | 16×16 sub-equations vs JPEG | vs WebP | one global equation vs JPEG |
|---|---:|---:|---:|
| landscape | +1.35 dB | −2.78 dB | −0.75 dB |
| wallpaper | −0.02 dB | −2.10 dB | −2.32 dB |
| beach photo | −1.27 dB | −3.82 dB | −2.07 dB |
| synthetic | +5.64 dB | −2.47 dB | −2.54 dB |

The sum-of-cosines equation is basically JPEG's own idea, so matching JPEG is expected. WebP adds prediction between blocks, and that is why it wins. Splitting into **sub-equations (tiles) beat one global equation on every image**. This is the one place where the FunnelBrain hierarchy instinct clearly helped.

**Audio.** On SNR, the equation scores +18 to +39 dB over MP3/Opus at equal size. Do not read this as a win. MP3/Opus deliberately leave ~20–30 dB of noise where ears cannot hear it (MP3 stays at 26 dB SNR even at 192 kbps). The equation uses non-overlapping 1024-sample frames, which likely clicks at low rates. **This needs a listening test.** Matched-size pairs are in `outputs/listen/` (for example `system_sounds_opus16k_19269B.wav` vs `system_sounds_equation_19246B.wav`).

**Video** (48 frames, 256×144): −3.6 dB vs H.264 on a camera pan and −10 dB on a test pattern. H.264 needs ~3–7× fewer bytes for the same PSNR because it predicts motion, which a fixed cosine equation cannot do.

## 5. How this compares with existing approaches

| Approach | Status | Relation to the POC |
|---|---|---|
| gzip / zstd / brotli / xz (LZ + entropy coding) | industry standard | They store "repeat that earlier piece" equations. V5b is a toy version. |
| Re-Pair / Sequitur grammars | research / niche | They store nested rule equations. V5a reproduces them. |
| PAQ / cmix / LLM compression | best ratios, slow | They store "predict the next bit" equations. V6 is a tiny version that already beat xz on CSV. |
| JPEG / WebP / H.264 / Opus | industry standard | They store cosine and prediction equations. V7 is roughly JPEG-level. |
| Fractal (IFS) and neural (COIN) codecs | research | Loop and learned-equation ideas, lossy only. |

## 6. Recommendation

1. **Do not pitch "any file becomes a tiny equation".** Lossless universal shrinking is provably impossible. A product or ad that claims it would be misleading and easy to debunk (Jan Sloot, πfs).
2. **Where an EquaFile-style product could be real:**
   * **Equation discovery for machine data** (sensor logs, metrics, telemetry, financial ticks). V5b/V6 already won here in a pure-Python toy. Compare next against time-series codecs (Gorilla, Chimp, Parquet delta encoding) on real datasets.
   * **Learned "equations" for media** (implicit neural representations à la COIN, or a GPU predictive model). This is the modern version of "an image is a function".
3. **The video:** if one is made now, it should be an honest explainer ("why a file can't be one magic equation, and what compressors already do instead"), not an ad. A product ad should wait for a niche win on real data.

## Reproduce

```
uv venv .venv --python 3.13 && uv pip install --python .venv/bin/python numpy pillow zstandard brotli scipy
.venv/bin/python bench/make_samples.py     # samples from files already on this Mac
.venv/bin/python bench/run_bench.py        # ~10 min, writes results/*.json
.venv/bin/python bench/make_report.py      # REPORT_DATA.md + outputs/visuals/report.html
.venv/bin/python bench/listen.py           # matched-size audio pairs
```

---

## Follow-up 1: single-digit loops + simple equations (V8)

Raaj's proposal: keep the loop count to 2–9, use only simple equations, and choose the best one per piece. Implemented in `equafile/smallloop.py` and tested with `bench/small_loop_test.py`. Visual: [outputs/visuals/small_loop_test.html](../outputs/visuals/small_loop_test.html).

* Recipe = equation (8 choices) + a + x₀ + L ∈ 2..9 = 23 bits for up to 9 bytes. All 524,288 recipes are tried at every position.
* Only **428,508** distinct 9-byte outputs exist, out of 2⁷² possible (1 in ~11 quadrillion).
* Real files: 0–38% of bytes printable; 69–86% of matches are 3-byte coincidences (save 4 bits each).
* Size: bigger than the file on 8 of 11 samples. Each re-encoding level adds ~11%. counter.bin got worse (1,314 B vs 6 B for V5b) because the L ≤ 9 cap forces 456 recipes.
* What holds up: **best approach per piece**. That is where every earlier win came from.

## Follow-up 2: waves, superposition and resonance (V9)

Raaj's proposal: treat the equations as continuous waves with frequencies, combine (superpose) them, and use resonance. Implemented in `equafile/resonance.py` and tested with `bench/resonance_test.py`. Visual: [outputs/visuals/waves_resonance.html](../outputs/visuals/waves_resonance.html).

* **V9a waves:** a resonator bank (FFT = driven oscillators) finds each 4,096-sample piece's loudest frequency. Its frequency is tuned to the exact peak, all waves settle jointly (least squares), and the wave is peeled off, up to 32 times. Waves cost 12 bytes each; the leftover is stored exactly with the V6 coder.
* **V9b resonant filter:** each piece rings on its own past, `x[n] = Σ c_k·x[n−k]` (linear prediction, order 0–32), and only the small kicks are stored.
* Both also try "no waves" and keep the smaller result (best approach per file).

| file | size | V9a waves | V9b resonant filter | best existing |
|---|---:|---:|---:|---|
| chord (3 notes) | 264,600 | **26,299** | 36,097 | FLAC −12: 46,588 |
| system sounds | 1,019,844 | 169,365 | **99,747** | FLAC −12: 139,535 |
| counter.bin | 4,096 | 1,254 | **78** | brotli: 217 |
| sensor.csv | 32,918 | 6,389 | 6,389 | xz: 6,564 |
| beach photo (raw) | 83,460 | 48,300 | 42,600 | WebP lossless: 33,720 (PNG 43,542) |
| license.txt | 13,809 | 5,314 | 5,314 | brotli: 3,264 |
| typing.py | 132,718 | 36,105 | 36,105 | brotli: 28,616 |
| random.bin | 4,096 | 4,115 | 4,115 | brotli: 4,100 |

* The resonators found the chord's notes exactly: 440.00, 554.37 and 659.25 Hz.
* Energy in the top 30 of 8,192 resonators: chord 97.4%, license.txt 36.1%, random 2.3%.
* "hi" as waves needs 5 of its 9 waves before every bit is right.
* Caveat: the FLAC win comes from coding the leftover with the slow pure-Python V6 coder. FLAC itself already uses the resonant-filter idea (linear prediction).

## Combined design: FunnelZip (V10)

Design: [DESIGN_FUNNELZIP.md](DESIGN_FUNNELZIP.md). Code: `funnelzip/funnelzip.c` (C core) + `equafile/funnelzip.py` (Listen stage, container). Benchmark: `bench/funnelzip_bench.py`. Visual: [outputs/visuals/funnelzip.html](../outputs/visuals/funnelzip.html).

Every approach becomes a **lens** that predicts the next bit; a funnel of mixers learns online which lens to trust. Equations *predict* (with a "surprise" distribution) instead of printing exactly. Listen uses the file's resonance with itself (autocorrelation) to pick sample width / channel step / row stride by trial coding.

| file | size | FunnelZip | best existing | Δ |
|---|---:|---:|---|---:|
| system_sounds.wav | 1,019,922 | **79,846** | FLAC −12: 139,535 | **−42.8%** |
| chord.wav | 264,644 | **33,965** | FLAC −12: 46,588 | −27.1% |
| sensor.csv | 32,918 | **3,159** | xz: 6,564 | −51.9% |
| counter.bin | 4,096 | **36** | brotli: 217 | −83.4% |
| landscape (raw RGB) | 786,432 | **97,407** | WebP-lossless: 119,562 | −18.5% |
| 2 MB Python source | 2,097,152 | **378,998** | brotli: 409,022 | −7.3% |
| typing.py | 132,718 | **26,622** | brotli: 28,616 | −7.0% |
| camera-pan video (raw) | 5,308,416 | **557,160** | x264-lossless: 579,083 | −3.8% |
| json_decoder.py | 12,872 | 3,136 | brotli: 3,158 | −0.7% |
| beach photo (raw RGB) | 83,460 | 34,211 | WebP-lossless: 33,720 | +1.5% |
| zsh binary | 1,361,216 | 540,467 | xz: 519,924 | +4.0% |
| wallpaper (raw RGB) | 442,368 | 167,481 | WebP-lossless: 154,976 | +8.1% |
| license.txt | 13,809 | 3,546 | brotli: 3,264 | +8.6% |
| synthetic gradient (raw) | 196,608 | 1,183 | WebP-lossless: 232 | +410% |
| random.bin | 4,096 | 4,129 | brotli: 4,100 | +0.7% |
| landscape.png (already compressed) | 183,871 | 183,887 | brotli: 183,876 | 0.0% |

All 18 files round-trip bit-exact. Speed ≈ 0.31 MB/s compress (incl. Listen trials), 0.51 MB/s decompress, single thread.

**Ablation** (size increase with one lens off): resonator +67% on sound and +89% on the counter; word/column +58% on CSV; memory +17–25% on text, code and binaries; grid +7.7% and resonator +7.4% on the photo; funnel +1–29%. The equation lens is redundant (≤ ±0.4% except on the counter), because the resonator covers it. "Memory only" (≈ V6) is +214% on sound, +114% on CSV and +836% on the counter.

**Where it loses:** flat/synthetic images (WebP's colour cache and run tricks), English text (brotli ships a built-in English dictionary), code binaries (needs a code lens), and speed.

## Follow-up: repeats of repeats (rhythm lens, 2026-10-17)

Raaj's idea: repeated content gets one equation; repeated equations get one resonance. Added to FunnelZip as the **Rhythm lens** (design §3.1):
(1) echo = one equation per repeat; (2) the distance at which repeats recur (found by long-range resonance in Listen, or learned from echo distances) = the file's rhythm, read as "the byte one period ago", which survives changed bytes; (3) for signals, the leftover of one period ago predicts this period's leftover through a learned 3-tap gain (long-term prediction).

| file | without rhythm | with rhythm | best existing |
|---|---:|---:|---|
| music_loop_clean.wav (bar × 8, volume changes) | 68,613 | **45,687** (−33%) | FLAC −12: 112,427 |
| static camera video (24 frames) | 788,515 | **690,155** (−12.5%) | x264-lossless: 752,477 |
| camera pan video (48 frames) | 557,160 (v1) | **493,957** (−11%) | x264-lossless: 579,083 |
| wallpaper (raw RGB) | 167,481 (v1) | **153,596** (−8%) | WebP-lossless: 154,976 |
| music_loop.wav (with recording noise) | 153,148 | 148,713 (−3%; near the noise floor) | FLAC −12: 169,218 |
| server.log | 123,487 | 123,399 | bzip2: 176,472 |

Listen found the music bar (66,150 bytes = 0.75 s) and the video frame (110,592 bytes) by itself. Files ≥ 10 KB beating the best existing tool: **13 of 19**. Costs: tiny files grew a few bytes (counter 36 → 43, where rhythm hurts by 17.6%), and compression slowed to ~0.12 MB/s because Listen trial-codes more periods (decompression ~0.45 MB/s).

## Follow-up: compress the file binary or the pixels?

`bench/pixels_vs_file.py`: FunnelZip on the image file's bytes vs on its decoded pixels.

| file | file | FunnelZip(file bytes) | FunnelZip(pixels) |
|---|---:|---:|---:|
| landscape.png | 183,871 | 183,887 | **95,449** (−48%) |
| beach_photo.png | 44,550 | 44,534 | **33,870** (−24%) |
| wallpaper.png | 207,776 | 207,485 | **153,596** (−26%) |
| landscape_q90.jpg | 24,498 | **23,257** (−5%) | 100,513 |
| landscape_q75.jpg | 15,335 | **13,880** (−9.5%) | 107,308 |

* **PNG:** deflate hides positions, so the file looks random; decoded pixels expose rows/columns. For these PNGs, re-encoding the pixels reproduces the identical file, so file-exact round trips work. Other encoders' PNGs would need a stored re-encode recipe (precomp/preflate approach).
* **JPEG:** the file already *is* a cosine-wave equation plus quantization; exact pixels must store JPEG's rounding noise (4–7× bigger). The right level is the quantized coefficients: Dropbox's Lepton gets ~22–23% bit-exact there. A FunnelZip coefficient lens is the next step.
* **Rule:** compress at the level where the structure lives.

## Next-steps round (FunnelZip v3, 2026-10-17)

Every "next step" from the previous round was built and measured. Details, per-step before/after and verdicts: the **"Next steps — tried"** section of [outputs/visuals/funnelzip.html](../outputs/visuals/funnelzip.html); data in `results/next_steps.json`, full table in `results/funnelzip.json`.

| step | result |
|---|---|
| 16-bit precise final stage | synthetic image −11%, 100 KB zeros 44 → 14 B |
| Motion lens (video) | camera pan −6.5% from motion alone; final −19.4% vs x264-lossless |
| Stereo | 104,650 B vs FLAC 185,351 (−43.5%); cross-channel taps −1.6% of that |
| Code lens + ARM64/x86 filters | zsh 485,935 vs xz+BCJ 486,188 (−0.1%, from +10% before) |
| Primer (122,880 B, same size as brotli's dictionary) + 1-byte header | json_decoder −14.5% vs brotli, typing −12.7%, hi.txt 5 B vs 6; license still +3.2% vs brotli |
| JPEG lens (coefficients, bit-exact rebuild, restart markers) | −24.6% to −34.3% vs the original file on PIL/Apple JPEGs; 4K photo −13.1%; beats every generic compressor on all 5 |
| PNG lens (pixels, byte-identical re-encode incl. ICC) | −23.7% to −48.4%; tiny PNG keeps the bytes path |
| Speed | Listen in parallel + per-type lens sets: text 0.28 → 0.55 MB/s, sound 0.15 → 0.29; fast mode decodes video 4× faster at +16% size; compiler flags ~3% (not used) |
| Listen fix | layout trials sample start/middle/end (wallpaper 166,642 → 153,964) |

**Totals:** 32 files, all bit-exact; best-existing beaten on **23 of 27** files ≥ 10 KB. Still losing: flat synthetic image (WebP-lossless), license.txt (+3.2% vs brotli's English dictionary), raw beach photo (+0.8% vs WebP), doc_versions (+0.4%). Slowest part: the pure-Python JPEG lens (~2 min each way on 5.6 MB).

## Reshape stage + lossless proof + speed (2026-10-17)

**Is it lossless?** Yes. `bench/lossless_proof.py` compresses and decompresses all 32 test files and compares SHA-256 and every byte: **32/32 identical, 24.7 MB checked, 0 bytes different** (`results/lossless_proof.json`). Nothing in FunnelZip is approximated. The only lossy parts of the project were the early V7 cosine-wave experiments and the MP3/JPEG/H.264 reference codecs.

**Reshape where it fails** (Raaj's "convert to the right format, reverse it on decode"): converting to "pure binary" adds nothing, because the coder already works bit by bit. Reversible *reshapes* that line up related values do help, and Listen keeps one only if it wins on a sample.

| file | before | after | best existing |
|---|---:|---:|---|
| synthetic image (raw) | 904 | **257** (prediction residual) | WebP-lossless 232 |
| beach photo (raw) | 33,997 | **30,362** (colour transform) | WebP-lossless 33,720 |
| wallpaper (raw) | 153,964 | **148,978** (colour transform) | WebP-lossless 154,976 |
| beach_photo.png / wallpaper.png / synthetic.png | 34,010 / 153,977 / 762 | **30,375 / 148,991 / 269** | the PNGs: 44,550 / 207,776 / 1,832 |
| typing.py / json_decoder.py | 24,995 / 2,699 | **24,767 / 2,687** (caps + words) | brotli 28,616 / 3,158 |
| JPEG files (known coefficient layout, planes in parallel) | q90 18,459 · Sonoma 48,184 · 4K 4,881,168 | **17,574 · 45,820 · 4,846,803** | best generic 23,509 · 67,880 · 5,441,742 |

The text reshape briefly broke round trips (it changes the length, which the header didn't store). The proof caught it, and the fix stores the reshaped length.

**Speed:** the 4K JPEG went from 127 s / 115 s to 57 s / 50 s (encode / decode). Per-file speeds are in the visual's speed table. The C core runs ~0.3–0.7 MB/s; tiny files are dominated by a fixed ~0.2–0.6 s setup.

**Totals now:** best existing tool beaten on **24 of 27** files ≥ 10 KB. Remaining: license.txt +3.2% (brotli's English dictionary), synthetic raw +25 bytes vs WebP, doc_versions +0.4%, random +0.3% (unavoidable).

## Speed round (2026-10-17)

Goal: 1+ MB/s at the same ratio. Result: **faster everywhere, and 4.4% smaller in total** over all 32 files. Still lossless: 32/32 byte-identical, 24.7 MB checked, 0 bytes different.

| one core, no Listen (`bench/speed_round.py`) | v4 size → now | v4 MB/s → now |
|---|---:|---:|
| 1 MB text | 180,415 → 179,871 | 0.77 → **1.07** |
| 1 MB sounds | 78,675 → 78,142 | 0.39 → 0.65 |
| photo pixels | 94,233 → 94,022 | 0.35 → 0.55 |
| sensor CSV | 3,117 → 3,129 | 0.61 → 0.75 |
| program binary | 518,408 → 516,721 | 0.47 → 0.61 |

| whole files (with Listen) | v4 → now | encode / decode |
|---|---:|---|
| 4K JPEG (5.6 MB) | 4,846,803 → **4,508,904** (−7.0%) | 56 s / 50 s → **5.5 s / 5.3 s** (≈1.0 MB/s) |
| Sonoma JPEG | 45,820 → 43,604 (−4.8%) | 5.5 s / 4.8 s → 2.5 s / 1.4 s |
| system sounds | 78,685 → 78,154 (−0.7%) | 3.5 s / 2.8 s → 2.3 s / 1.6 s |
| camera-pan video (5.3 MB) | 466,865 → 461,640 (−1.1%) | 19.4 s / 16.1 s → 12.9 s / 9.8 s |
| whole benchmark | 8,069,819 → 7,714,159 (−4.4%) | 0.16 / 0.22 → 0.30 / 0.58 MB/s |

What did it: JPEG Huffman in C (~6 s → 0.1 s each way, bit-identical); profile-driven engine fixes (fast exp, table-driven counters, cached surprise intervals, SIMD mixer, cheaper pixel refits); **Listen picks the lens set** (unused lenses cost time and add noise); **JPEG frequency planes in ~1 MB groups, each with its own model on its own core, AC values stored one byte each**. Details: `docs/DESIGN_FUNNELZIP.md` §3.4.

A bug caught on the way: the first table-driven counter rounded small negative steps away from zero. That cost +1.7% on 94%-zero JPEG planes. With exact rounding the planes are smaller than v4.

Not reached: 1 MB/s for sound, pixels and program code on one core (0.55–0.65). Three small files grew a little from the counter change: sensor.csv +22 B, doc_versions +8 B, server.log +126 B. Still 24 of 27 files ≥ 10 KB beat the best existing tool.
