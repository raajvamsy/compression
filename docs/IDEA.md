# EquaFile: storing files as equations

**Status:** idea + POC (October 2026). Results are in [REPORT.md](REPORT.md).
**Author:** Raaj · **Codename:** EquaFile

---

## 1. The idea in one paragraph

Every file is a sequence of 0s and 1s. Instead of storing those bits, store an **equation** that *prints* them. If one equation is too hard to find, split the file and find a **sub-equation** for each part, then one **master equation** that prints the sub-equations (the FunnelBrain idea: big events are made of smaller events, each captured by an equation). The most compact form of this is a **loop**: start from a tiny input (say `1`), feed it to an equation, feed the output back in, and repeat for a stored **loop count**. When the loop ends you get either the file itself or a list of sub-equations (each with its own input number and loop count), which are expanded the same way until everything is the file.

What would be stored on disk is only:

```
equation id  +  start input  +  loop count   (+ the same for each sub-equation)
```

## 2. Why it is attractive

* **FunnelBrain analogy.** In FunnelBrain an event ("run, kick at speed, score") is represented by an equation, not by raw numbers. Physics does this well: `s = ut + ½at²` replaces a million position samples with three numbers. If files could be treated the same way, a movie would be a few equations.
* **The loop trick looks like free compression.** Only `(input, loop count)` is stored, so the stored size seems constant no matter how big the file is.
* **Hierarchy looks like it scales.** A 1 GB file splits into a thousand 1 MB parts, each split again, and so on. Each level seems to need only a few numbers.

## 3. The idea, formally

Let the file be a bit string `F` of length `n`.

| Variant | What is stored | Decoder |
|---|---|---|
| **V1 – one direct equation** | coefficients of `f` | `F[i] = f(i)` for every position `i` |
| **V2 – loop (Raaj's scheme)** | `f`, start input `x₀`, loop count `L` | `x ← f(x)` repeated `L` times, then `F = out(x)` |
| **V3 – seed + loop** | `f`, seed `s`, loop count `L` | same as V2, but the seed is searched, not fixed to `1` |
| **V4 – hierarchy (FunnelBrain)** | a master `(s, L)` whose output is a list of child `(s, L)` pairs, recursively | expand the tree, concatenate the leaves |
| **V5 – structural equations** | rules such as `R1 = 'h' 'i'`, `R2 = R1 R1`, `f(i) = a + b·i` | evaluate the rule tree |
| **V6 – predictive equation** | a model `P(next bit │ past bits)` + a short code | the model replays the file bit by bit |
| **V7 – lossy signal equation** | a few terms of `Σ aₖ·cos(…)` (image/audio/video) | evaluate the sum at each pixel / sample / frame |

V1–V4 are the idea exactly as stated. V5–V7 are what the idea turns into once it meets the theory below. All seven are implemented in the POC.

## 4. Theory: what can and cannot work

### 4.1 The counting (pigeonhole) argument

There are `2ⁿ` different files of `n` bits. A description that is `k` bits long can name at most `2ᵏ` different things. So if every `n`-bit file must get its own description, **some descriptions need at least `n` bits**, and on average a description cannot be shorter than the file. This holds for every scheme, including equations, loops and hierarchies, because all of them are just descriptions that a decoder expands. ([Matt Might, "Impossibly good compression"](https://matt.might.net/articles/why-infinite-or-guaranteed-file-compression-is-impossible))

### 4.2 Applied to the loop (V2/V3)

* With `f` and `x₀ = 1` fixed, the file is determined by `L` alone. `L` can take `2ᵇ` values if it has `b` bits, so it can reach at most `2ᵇ` files. To reach every `n`-bit file, `L` needs about `n` bits. **The loop count becomes the file**, just written in another form.
* The state `x` inside the loop has some fixed size `m`. An iteration on `m` bits revisits a state after at most `2ᵐ` steps, so it can only ever produce `2ᵐ` different outputs. To print any `n`-bit file, `m ≥ n`.
* **Search cost.** For a well-mixing `f`, each loop step is a lottery ticket that matches the target with probability `2⁻ⁿ`. The expected loop count is `≈ 2ⁿ`. That is 65 536 steps for `"hi"` (16 bits) and `2⁸¹⁹²` steps for a 1 KB file. There are about `2²⁶⁶` atoms in the observable universe.

### 4.3 Applied to the hierarchy (V4)

Splitting does not create information. If each leaf `(seed, loop)` costs about as many bits as the chunk it prints (4.2), the leaves together cost about `n` bits, and every extra level adds its own overhead (lengths, counts, the master equation). The tree is a lossless re-encoding, and it is slightly bigger. The POC measures this.

### 4.4 When an equation genuinely wins

An equation is short exactly when the file has **structure** that the equation family can express. Formally, the shortest program that prints a file is its **Kolmogorov complexity**. It is tiny for `0 1 2 3 … 255` or the digits of π, and ≈ `n` for random data. It is also **uncomputable** in general, so no system can find the best equation for every file. Every real compressor therefore restricts itself to an equation family it can search efficiently:

| Equation family | Real-world name |
|---|---|
| "this piece repeats that earlier piece" | LZ77 (gzip, zstd) |
| "rule = pair of rules", nested | grammar compression (Sequitur, Re-Pair) |
| "next bit follows linear recurrence" | LFSR / Berlekamp–Massey linear complexity |
| "probability of the next symbol" + arithmetic coding | PPM, CM, PAQ, LLM-based compression |
| "sum of cosine waves" | JPEG, MP3/AAC (MDCT), video codecs |
| "image = fixed point of contracting maps" | fractal (IFS) compression |
| "tiny neural net f(x,y) → colour" | implicit neural representations (COIN) |

So the idea is real, it is just old: **every compressor already stores an equation instead of the data.** The open question for this POC is whether *Raaj's specific forms* (the loop, the seed search and the hierarchy) add anything over those families.

### 4.5 Lossy is different

For images, audio and video we do not need the exact bits back, only something that looks or sounds the same. There the counting argument no longer forces `≥ n` bits, and a short equation can describe a "good enough" version. This is where the FunnelBrain analogy fits best: a physics equation of a kick does not store every atom, it stores the essence. The POC tests a pure "sum of cosines" equation against JPEG/WebP, MP3/Opus and H.264.

## 5. Prior art (research summary)

| Work | What it does | Relation to EquaFile |
|---|---|---|
| Pigeonhole / counting argument | Proves no lossless scheme shrinks all files | Bounds V1–V4 |
| [πfs](https://github.com/avar/pifs) | Stores files as offsets into π | Same as V2 with `f` = "next digit of π"; the offset is as long as the data (joke project) |
| Jan Sloot "Digital Coding System" (1990s, from general knowledge) | Claimed a full film in a few KB; never reproduced | The cautionary tale for V2/V4 |
| [Re-Pair](https://en.wikipedia.org/wiki/Re-Pair) / Sequitur | File → grammar of nested rules ("straight-line program") | Exactly V4/V5 done right: sub-equations that reuse each other |
| [Berlekamp–Massey](https://en.wikipedia.org/wiki/Berlekamp%E2%80%93Massey_algorithm) | Finds the shortest LFSR (iterated linear equation) printing a bit string | Exactly V2 with `f` linear; for random data the LFSR length is ≈ n/2 taps + n/2 seed bits = n |
| [Barnsley IFS / fractal compression](https://ntrs.nasa.gov/api/citations/19890012975/downloads/19890012975.pdf) | Image = attractor of iterated maps; loop until it converges | The closest relative of the loop idea, lossy; huge ratios only on self-similar images, slow encoding |
| [COIN (Dupont et al., 2021)](https://arxiv.org/abs/2103.03123) | Overfit a small neural net `f(x,y) → RGB`, store the weights | "Equation of an image" with a learned equation; beat JPEG at low bit-rates |
| [Language Modeling Is Compression (Delétang et al., ICLR 2024)](https://arxiv.org/abs/2309.10668) | A big predictor + arithmetic coding as a compressor | V6 at scale: Chinchilla 70B compressed ImageNet patches to 43.4% (PNG 58.5%) and LibriSpeech to 16.4% (FLAC 30.3%), but only if the model weights are not counted |
| [Paradoxical compression (NCC Group)](https://www.nccgroup.com/research/paradoxical-compression-with-verifiable-delay-functions/) | Uses slow functions to make "infinite compression" look true until checked | Shows why a demo must always verify the round trip |

**Novelty verdict before the POC.** The lossless loop/seed/hierarchy idea (V2–V4) is known not to work in general; it is the "compress anything" claim in a new form. The useful parts of the intuition (equations, hierarchy of reused sub-equations, predictive models, lossy signal equations) are well established. A product would need a niche where the data has structure that existing equation families miss.

## 6. POC plan

### 6.1 Goals
1. Build every variant V1–V7 end to end: encode → stored equation → decode → **bit-exact check** (lossless) or quality score (lossy).
2. Measure the **stored size** honestly: every byte the decoder needs (except the fixed decoder program itself) counts.
3. Benchmark against gzip, bzip2, xz, zstd, brotli (lossless) and PNG, JPEG, WebP, MP3, Opus, H.264 (lossy).
4. Test on: `"hi"`, short and long text, structured data, random data, then image, audio, video.
5. A sample app (CLI + local web UI) that shows the equation, the loop and the comparison.

### 6.2 Success criteria
* **Works:** decoded output equals input (lossless) for every variant on every file.
* **Wins:** stored size smaller than the best baseline on some realistic file type, *or* better quality at equal size (lossy).
* If it only "works" without "winning", the honest outcome is: the idea is correct as a representation, but not as compression.

### 6.3 Layout
```
equafile/          engine (one module per variant) + container format + CLI
  bits.py            varints, bit packing
  polynomial.py      V1  one equation f(i) = byte_i over GF(257)
  loop.py            V2/V3 Raaj's loop and seed search, V4 hierarchy
  grammar.py         V5a nested rules (Re-Pair style)
  symbolic.py        V5b segment equations: a + b·i, repeats, literals
  predictive.py      V6  context-mixing model + arithmetic coder
  signal.py          V7  cosine-series equations for image / audio / video
  baselines.py       existing codecs
app/               local web app (python stdlib server + one HTML page)
bench/             benchmark runner → results/ + docs/REPORT.md + outputs/visuals/report.html
```

## 7. Approaches in detail

### V1 — one direct equation
Treat byte `i` as `f(i)`. There is exactly one polynomial of degree `< n` through `n` points. Using arithmetic mod 257 (a prime just above 255) keeps it exact. `"hi"` becomes `f(x) = 104 + x` (`f(0)=104='h'`, `f(1)=105='i'`). For a real file the polynomial has `n` coefficients of 9 bits each, so it is **12.5% bigger than the file**. One equation always exists, but it is not shorter.

### V2 — the loop, exactly as described
`x₀ = 1`, `x ← f(x)` with a fixed 64-bit mixing function, stop after `L` steps, print the top `n` bits. The encoder walks the loop until the output matches the file, then stores `L`. Expected `L ≈ 2ⁿ`.

### V3 — seed + loop
Same loop with a fixed small `L` (say 4 rounds of mixing) but the start input `s` is searched instead. The search can be run in parallel, so it reaches 24–28 bits. Stored: `s` and `L`.

### V4 — hierarchy (FunnelBrain tree)
1. Cut the file into 16-bit chunks.
2. For each chunk, find the smallest seed whose loop prints it (a precomputed table makes this instant).
3. Write all seeds as one new bit string. That is level 2.
4. Repeat on level 2, level 3, … and store the master seed, the loop count and the number of levels.

Decoding expands from the master down. The POC reports the size at every level.

### V5 — structural equations
* **Grammar:** repeatedly replace the most frequent pair of symbols with a new rule (`R256 = 'e' ' '`). Rules can contain rules: a tree of sub-equations, the FunnelBrain shape that actually helps.
* **Symbolic segments:** cover the file with pieces of the form `f(i) = (a + b·i) mod 256`, "copy `len` bytes from `dist` back", or literal bytes. A counting sequence or a gradient becomes one line.

### V6 — predictive equation
The equation is a probability model: `P(next bit = 1 │ previous bytes)`, mixing what followed the last 1, 2, 3 and 4 bytes. An arithmetic coder turns good predictions into few bits. The decoder runs the same model and replays the file. This is the family behind PAQ and LLM compression.

### V7 — lossy signal equations
* **Image:** `pixel(x, y) = Σₖ aₖ · cos(π uₖ (2x+1)/2W) · cos(π vₖ (2y+1)/2H)`, keeping only the `K` strongest terms per colour channel. One global equation, and a tiled variant with one sub-equation per 16×16 tile.
* **Audio:** per 1024-sample frame, `sample(t) = Σₖ aₖ · cos(π fₖ (2t+1)/2N)` with the top `K` terms.
* **Video:** one 3-D cosine equation `f(x, y, t)` per tile and group of frames.

Quality: PSNR (image/video) and SNR (audio) against the original, at matched stored size.

## 8. Risks and what to watch
* **Hidden information.** Any "constant" the decoder uses (lookup tables, dictionaries, model weights) is part of the description. The POC counts only data that depends on the file; decoders contain no per-file data.
* **Verification.** Every lossless result is checked with SHA-256 of the decoded output.
* **Pure Python speed.** Encoders are slow and inputs are capped where noted. Sizes, not speed, are the point.

## 9. After the POC
* If a variant wins somewhere → narrow the niche, then plan the PaaS ("upload a file, get its equation") and the non-technical ad video.
* If nothing wins → keep the explainer as the deliverable ("why a file cannot be a tiny equation, and what compressors already do instead"), and pick the strongest real direction (V6/V7: learned predictive or neural equations) for the next round.
