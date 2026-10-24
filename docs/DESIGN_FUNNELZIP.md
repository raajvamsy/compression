# FunnelZip — one compressor built from every EquaFile experiment

**Status:** design + POC (2026-10-16). Builds on [IDEA.md](IDEA.md) and the results in [REPORT.md](REPORT.md).

## 1. What the experiments taught us

| Experiment | What worked | What failed | Becomes |
|---|---|---|---|
| V1–V4 loop / seed / tree | exact round trips | an exact recipe costs as many bits as the data | nothing (proven dead end) |
| V5b segment equations | `f(i)=a+b·i`, "repeat that piece" | hard choices, literals are expensive | **Equation lens**, **Echo lens** |
| V5a nested rules | reuse of earlier pieces | no entropy coding | **Echo lens** |
| V6 predictive | best on text/CSV | knows nothing about signals or grids | **Memory lens**, **Word lens**, the coder |
| V8 small loops | best choice per piece | a fixed menu of exact recipes rarely matches | equations as *predictions*, not recipes |
| V9a waves + resonance | found exact tones | 12 bytes per wave, breaks on text | **Listen** stage (resonance finds the file's period) |
| V9b resonant filter | beat FLAC on sound, PNG on photo | hard per-piece switching, side info | **Resonator lens** (learns online, no side info) |
| V7 cosine waves (lossy) | ≈ JPEG | lossy only | future lossy front end |

Two lessons decide the design:

1. **An equation should predict, not print.** In V8 an equation had to print the bytes *exactly* or it was useless. If it only *predicts* the next bit, a near miss still saves bits, and the coder pays only for the surprise. This removes the "menu too small" problem.
2. **Don't pick one equation; let all of them vote on every bit.** Choosing a method per file or per piece costs side information and is coarse. A mixer that learns, bit by bit, how much to trust each lens adapts inside a piece, costs nothing to transmit, and is never much worse than the best lens.

## 2. Architecture (FunnelBrain shape: organ lenses on a shared field)

```
file ──► LISTEN (once) ────────────────────────────────────────────► header: W, C, S (3 small numbers)
          resonance of the file with itself (autocorrelation peaks)
          → sample width W, channel step C, row stride S; trial-coded on a slice

          ┌──────────── for every bit ─────────────────────────────────────────┐
          │  LENSES (each says P(bit = 1))                                      │
          │   Memory    what followed the last 0,1,2,3,4,6 bytes                │
          │   Word      current word + previous word; column in previous line   │
          │   Echo      longest earlier repeat → its next bit                   │
          │   Equation  next sample ≈ 2·s₁ − s₂  (a + b·i, run, gradient)       │
          │   Resonator next sample ≈ Σ cₖ·s₍ₙ₋ₖ₎, coefficients learned online   │
          │   Grid      byte above (stride S), left (step C), left+above−corner  │
          │        │                                                            │
          │        ▼                                                            │
          │  FUNNEL: 3 mixers, each picks its weight set from a different       │
          │          context (bit position, echo state, sample position), then  │
          │          a final mixer blends the three (funnel of funnels)          │
          │        ▼                                                            │
          │  REFINE: two secondary estimators correct leftover bias (APM/SSE)   │
          │        ▼                                                            │
          │  CODER: binary arithmetic coding, 16-bit probability                │
          └─────────────────────────────────────────────────────────────────────┘
```

* **Resonator ⊇ waves.** A sum of K sine waves obeys an exact linear recurrence of order 2K. So the online resonator lens (order 16) can lock onto up to 8 tones by itself, with no wave parameters sent. That makes V9a's 12-byte waves unnecessary.
* **Listen = resonance on the file itself.** A raw image has a strong peak at lag = row length. A table has one at the record length, and interleaved RGB at lag 3. Candidates come from the autocorrelation peaks. Each one is trial-coded on the first 128 KB and the cheapest wins. That is Raaj's "try different approaches per problem", done by measurement.
* **No hard switching anywhere.** Every lens is always on, and the mixers learn the trust. "Best approach per piece" happens continuously, bit by bit.

## 3. Lens details

| Lens | Context / prediction | Inputs to the funnel |
|---|---|---|
| Memory | hashed orders 0, 1, 2, 3, 4, 6 + partial byte | 6 |
| Word | hash of letters in current word (+ previous word); (column, byte above in previous line) | 2 |
| Echo | 6-byte hash → last position; expected bit with learned confidence by match length | 2 |
| Equation | expected byte of `2·s₁ − s₂` and of `s₁` at sample level (W = 1 or 2) | 2 |
| Resonator | normalized LMS, order 16, on samples; for 16-bit, the high byte uses the already-known low byte | 2 |
| Grid | (above), (above, left), (left + above − corner) when S > 0 | 3 |
| Rhythm | byte one period R ago (+ how locked the beat is), and for signals the *leftover* of one period ago × learned gain | 4 |
| Motion | video (R = frame, S = row): per 16×16 block, motion estimated from the two previous frames (decoder can repeat it, nothing stored); pixel where it came from + motion-compensated gradient | 5 |
| Stereo | 2-channel audio: each channel's leftover also predicted from the other channel's leftover, 8 lags | (inside resonator) |
| Code | position in the 4-byte instruction word + the word(s) before; plus a reversible pre-filter that makes ARM64 BL/ADRP and x86 CALL/JMP targets absolute | 2 |
| Precise | final 16-bit stage: blends the refined estimate with the funnel's raw confidence (learned), 16-bit arithmetic coder | - |

All probabilities use adaptive counters (fast at first, then stable). The mixers are online logistic regressions, exactly the "learned weights" idea.

### 3.1 Repeats of repeats (Raaj's follow-up, 2026-10-17)

A hierarchy of repetition, each level one more "equation of equations":

1. **A repeat → one equation.** The Echo lens: "the next bytes are the ones after the last time these 6 bytes appeared".
2. **Repeated repeats → one resonance.** If repeats keep arriving at the same distance R (a log line, a table row, a music bar, a video frame), that distance is the file's rhythm. Listen finds R from long-range autocorrelation, trying each peak and its half and third. Without R, the lens learns it from the distances the echo lens keeps finding. The lens reads "the byte one period ago" and counts how long the beat has held. Unlike echo it survives a changed byte, because it keeps the beat instead of dropping the match.
3. **A repeated equation → its own resonance.** For signals, the resonator's leftover of one period ago predicts this period's leftover through a 3-tap learned gain (long-term prediction, as in speech codecs' pitch predictor). A bar played louder is still the same equation; only the gain changes.

The funnel's second mixer picks its weights by "is the rhythm locked?", so the rhythm is trusted only while it holds.

### 3.2 Next-steps round (2026-10-17)

* **Primer:** before a text file, the model reads a fixed 122,880-byte primer (Python docs + code, none of it in the test set). That is the same size as brotli's built-in dictionary, and the decoder reads the same primer.
* **Listen picks the lens set:** text-like files skip the signal lenses (faster, slightly smaller). Binary-like files try the code filters. All trials run in parallel and sample start/middle/end of the file (a 128 KB prefix once picked the wrong layout).
* **Image files are compressed where their structure lives:**
  * **JPEG lens:** undo only the Huffman layer → quantized cosine coefficients → 64 frequency planes per component → FunnelZip → rebuild the scan with the file's own tables (restart markers supported). The rebuilt file is verified byte-for-byte at encode time, with a fallback otherwise.
  * **PNG lens:** decode the pixels → FunnelZip → re-encode with the same settings and ICC profile, used only if byte-identical.
* **Container:** varint length + 1 flag byte for the common case (custom settings, filter, primer, chunked, image lens).
* **Fast mode (opt-in):** files ≥ 3 MB are split on frame/row/sample boundaries and coded on all cores. On video this decodes 4× faster but is 16% bigger, because each chunk learns from scratch.

### 3.3 Reshape stage (Raaj's "convert it first, undo it after", 2026-10-17)

Converting a file to "pure binary" adds nothing, because the coder already works bit by bit. What helps is a **reversible reshape** that puts related values next to each other. After Listen, FunnelZip trial-codes each reshape that fits the file on a start/middle/end sample, keeps it only if it wins, and records its id and parameters (plus the reshaped length) in the header. Decode codes back and then inverts the reshape.

| reshape | for | effect |
|---|---|---|
| colour transform (R−G, B−G) | RGB pixels | beach photo −11%, wallpaper −3% (now beats WebP-lossless) |
| colour transform + planar | RGB pixels | sometimes second best |
| prediction residual (pixel − JPEG-LS MED guess; inverse in C) | flat / synthetic pictures | synthetic 904 → ~254 B |
| caps + words (markers = unused byte values; primer reshaped the same way) | text / code | ±1%: the word/memory lenses already capture this |
| known layout for JPEG coefficient planes (W=2, stride = block row, rhythm = one frequency plane), planes in parallel | JPEG lens | ~5% smaller and ~2× faster than letting Listen guess |

Every reshape is verified by the byte-for-byte round trip (`bench/lossless_proof.py`). A length-changing text reshape briefly broke round trips until its length was stored.

### 3.4 Speed round (2026-10-17)

Goal: 1+ MB/s at the same ratio. What changed, in profile order (measured totals per file: `bench/speed_round.py` → `results/speed_round.json`):

| change | where | effect |
|---|---|---|
| `fexp()`: 2^x by a degree-6 polynomial + exponent bits, instead of `exp()` | Laplace "surprise" lenses, precise stage | exp/CDF maths was the top profile entry for sound |
| counter learning rate from a table, `2/(2n+3)` in Q24, truncated toward zero (the division it replaces) | every context counter | first try used Q16 + an arithmetic shift, which rounds negative steps away from zero. On 94%-zero JPEG planes that bias cost +1.7%; with exact rounding the planes are 0.1% *smaller* than v4 |
| surprise intervals cached per bit and narrowed after each bit; signal maths skipped when no signal lens runs | surprise lenses | fewer CDF evaluations |
| one shared input vector, 4-lane mixer dot product; third mixer skipped for non-signal lens sets | mixers | mixers + counters were the top entries for text |
| LPC refit every 1024 samples (window 2048) for pixels; audio keeps 256/4096 | resonator | `lpc_refit` was the top entry for images; audio ratio unchanged (a cheaper audio refit cost +2.9%, so audio kept its schedule) |
| JPEG Huffman decode/encode in C (`fz_jpeg_decode/encode`; restart markers, byte stuffing) | JPEG lens | 4K photo: ~6 s → 0.09 s decode, 0.14 s encode; bit-identical to the Python reference |
| **Listen picks the lens set** (text: with/without the code lens; others: all / −word / −word−code / −word−code−motion), in the same parallel batch as the rhythm-period trials | Listen | a lens that finds nothing still costs time *and* adds noise to the mix: sound −0.7%, zsh −0.4%, text −0.4%, and faster |
| **JPEG planes**: DC as 16-bit; AC as one byte each (v+128, linear, escape 0 + int16 for \|v\|>127, 0.08% of values); AC planes split into ~1 group per MB, each its own model, all groups of all colours in parallel | JPEG lens | 4K photo 4.85 MB → 4.51 MB (−7%) and 57 s → 5.5 s (1.0 MB/s); low and high frequencies behave differently, so separate models also compress better |

Tried and dropped (ratio before speed): context tables sized to the input (text +1.9% at 2^23; kept 2^24 for anything over 16 KB), 64 KB Listen trials (−16% time, zsh +3%), `-mcpu=native`/LTO (no gain), fewer byte-history orders for 16-bit data (±0.4%, +3% speed).

Where it stands: one core codes text at ~1.0 MB/s and signals (sound, pixels, code) at ~0.6 MB/s. A bitwise context-mixing model is sequential, so the remaining 2× for signals needs either fewer lenses per bit or splitting a file into independently modelled parts. That works when the parts really differ, as with JPEG frequency planes. Otherwise it costs ratio: fast mode +16% on video.

## 4. What it cannot do

It does not escape the counting argument: random data stays random (≈ +0.1%). It is a better *probability model*, so it wins where files have structure that some lens can see.

## 5. Evaluation plan

* Round trip every sample (SHA-256), plus the edge cases: empty, 1 byte, all zeros, random.
* Compare with gzip −9, bzip2 −9, xz −9e, zstd −19/−22, brotli −11, FLAC −12 (audio), PNG and WebP-lossless (images), and the best earlier EquaFile variant.
* **Ablation:** switch each lens off and measure the loss. That shows what each idea contributes.
* Speed: MB/s for encoding and decoding.

## 6. Later (not in this POC)

* **Lossy front end:** cosine-wave equation (V7) or resonator residual quantization → FunnelZip as the entropy stage.
* More lenses: stereo/multichannel, x86/ARM code (call targets), JSON/XML structure, learned dictionaries.
* GPU/neural lens (a small transformer as one more voter), which is where the LLM-compression results point.
