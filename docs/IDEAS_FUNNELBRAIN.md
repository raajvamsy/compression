# FunnelBrain theory applied to image compression (FFIC prototype)

**Status:** research prototype, 2026-10-24. Code in [`ffic/`](../ffic). It measures **ideal code length** (sum of -log2 p).
There is **no bitstream and no decoder yet**, so nothing here is a verified round trip. Treat every number as "what an
arithmetic coder driven by this model should produce, minus a few hundredths of a percent".

## 1. What FunnelBrain is (read from `raajvamsy/funnel-brain`, `docs/FUNNEL_BRAIN_HYPOTHESIS.md`, `funnel/field.py`)

| FunnelBrain mechanism | In the brain | Reading for compression |
|---|---|---|
| Multi-scale transduction (windows 4 / 64 / 1024 B -> mean, spread, head, hash) | bytes become a point in concept space | a descriptor of the causal neighbourhood |
| **Funnel** (frustum attractor: position, `mouth_radius`, `depth`, `exposure_count`, fatigue) | a learned concept | a context class that owns a predictor; depth = how much experience backs it |
| `find_or_create_funnel` within `MERGE_RADIUS` | online leader clustering | classes are created on demand, not designed |
| **Spin** (wave-ball phase velocity; `wave_modulation` = frequency / amplitude / phase) | helical trajectories, bounce angles | **orientation and activity of the local signal** (the "different directions" idea) |
| **Prediction error** widens the mouth and deepens the funnel | online learning | every pixel's surprise updates weights, scale and skill |
| Fatigue / saturation | funnel fills, ball bounces out | the learning rate floors out as a funnel matures |
| **Sleep MDL**: *fission* (split a funnel whose modes disagree), meta-funnels, **nesting** (`parent_meta_id`) | offline restructuring, "keep a new cluster only if it pays for itself" | a hierarchy of contexts where **a child inherits its parent** and must earn influence |
| Parallel balls + global workspace (one winner) | many candidates, one thought | several predictors at once, blended by recent error |
| Shared field, organs as lenses | one substrate, many readers | FunnelZip's "lenses on one field" (already used there) |

What I deliberately did **not** take: the ball-bounce simulation. It is a slow way to find the nearest funnel; here a
funnel is found by direct lookup, which is where the speed comes from.

## 2. FFIC: the mapping, in code (`ffic/ffic.c`)

```
for each plane (G first, then R, then B):          <- colour is a dimension: later planes read earlier planes
  per pixel, from already-decoded pixels only:
    transduce   : activity A (8), brightness L (4), SPIN theta (8 orientations + flat)
    parallel balls : 5 fixed predictors, blended by their recent error nearby   -> "base"
    funnel path : root -> A -> (A,theta) -> (A,theta,L)     (nested; a newborn funnel copies its parent's
                  weights AND covariance; each funnel predicts; workspace weights = 1 / skill^2)
    predictor   : recursive least squares on 12 causal taps + cross-plane colour taps (17 for R, 22 for B)
    mouth       : residual scale from local error energy x activity x cross-plane surprise
    coder       : residual e = x - round(pred) as binary decisions (zero? sign? magnitude bucket, mantissa),
                  each probability = logistic mix of 6 funnel-context counters + a parametric Laplace prior
    learn       : weights, covariance, skill, mouth, counters, mixer all adapt every pixel
```

## 3. Results (20 images: 12 Pexels crops 1024x768 + 8 Kodak 768x512; reference = JXL lossless effort 9)

| Model | Size vs JXL e9 | Pexels-11 vs FunnelZip | Time, 20 images, 4 jobs in parallel |
|---|---|---|---|
| bare baseline (MED + activity context + Laplace) | +75.9% | +87.8% | 5.6 s |
| + LMS, cross-plane taps, error-energy mouth, parallel balls (`lcew`) | +13.7% | +17.0% | 6.9 s |
| + nested funnels (`ltpcew`, NLMS learner) | +15.0% | +18.2% | 8.4 s |
| + recursive least squares (`ltpcrew`) | +4.8% | +5.9% | 56.6 s |
| + binary context-mixing residual coder (`lcewb`, no RLS) | +10.2% | +13.2% | 19.7 s |
| **full model** (`ltpcrewb`) | **+1.1%** | **+1.8%** | 70.7 s (about 3.5 s / image) |
| full model, learning gated off on unsurprising pixels (`gate=0.5`) | +1.9% | +2.8% | 46.3 s (about 2.3 s / image) |
| full model, RLS only on the deepest funnel (`rl=1`) | +4.7% | +6.1% | 37.6 s |

Reference speeds on the same machine: FunnelZip about 12-17 s per image, JXL e9 about 1.3 s, JXL e7 about 0.2 s.

Per image, full model vs JXL e9: **wins** on p1054218 (-9%), p2014422 (-9%), p1770809 (-4%), ties on p1366919 and kodim20,
**loses** by 1-5% on the rest (worst: dog photo p2253275 +5%, kodim01 +5%). It beats FunnelZip on the two smoothest
Pexels images (310 kB vs 318 kB, 158 kB vs 160 kB).

## 4. What the experiments say about the theory

* **Static splitting hurts; nesting repairs it.** Splitting into up to 288 funnels by spin and level made the model *worse*
  (+29% with spin, +40% with level, against +17.8% for one funnel in that version) because every new funnel had to relearn
  its predictor (context dilution). Making the same funnels nested, with inheritance and a skill-weighted blend, gave
  +16.6% instead of +29-40%. This is the FunnelBrain rule ("a new cluster must pay for itself") confirmed on data.
  But the benefit of the extra levels themselves is small and depends on the learner: with NLMS the nested hierarchy was
  no better than a single funnel in the final code (+15.0% vs +13.7%); with RLS and the discrete coder it helps by about
  one point (+2.4% -> +1.4% in the version before the last two additions). Spin and brightness are not where the big gains are.
* **The colour dimension was the biggest single lever.** On a single G plane FFIC already matched JXL (-4% to +2%). Most
  of the early gap was R and B. Giving them G's local colour structure at the neighbouring positions took the total from
  about +18% to +4.9%.
* **A better learner mattered more than more contexts.** Recursive least squares in place of NLMS: +20% -> +19.5% alone,
  but with the discrete coder +11.7% -> +1.4%.
* **A discrete residual coder mattered on smooth images.** A parametric Laplace around a fractional prediction cannot model
  "exactly zero 80% of the time". The static conditional entropy given only (activity, energy) was *equal* to the Laplace
  cost, so the gain came from richer contexts, not from the shape (diagnostic run, dog photo).
* **Things that did not help much:** cross-plane error energy in the mouth (+4.9% -> +4.8%), ball disagreement as a
  context plus a parametric prior input (+1.4% -> +1.1%), non-causal G taps (E, S, SE, SW of an already-decoded plane:
  -0.07% for +44% time, left off), RLS on the root funnel (byte-identical with and without), and every parameter sweep
  (forgetting factor, mixer rate, counter limit, covariance init: all within 0.5 points).

## 5. Honest assessment: "a lot more and faster"

* **Faster: yes, relative to FunnelZip.** About 4x (full) to 6x (gated) faster than the current FunnelZip on the same
  images, at 1.8-2.8% larger on the Pexels subset. It is **not** faster than JXL (about 2-3x slower than e9, 10x+ slower than e7).
* **More: not yet.** The model reaches JXL e9 parity (+1.1% overall) and beats it on 3 of 20 images. It does not beat
  FunnelZip on average. The honest ceiling of this model family looks like a few percent better than JXL, because
  the parameters are flat and the easy additions are used up. Real "a lot more" needs something structurally new.
* **FunnelBrain's contribution is the design, not new mathematics.** The final system is close to known techniques
  (context mixing, recursive-least-squares prediction, hierarchical context back-off). What FunnelBrain gave was a
  coherent blueprint that I could test piece by piece, and the nesting/inheritance rule was a real, measured win.
* **Test-set caveat.** The Pexels images are JPEG-derived (CDN recompression), so they carry an 8x8 DCT block structure
  that raw camera files lack. Kodak (lossless source) is the cleaner check: there FFIC is +0% to +5% vs JXL.
* **Not done:** a real arithmetic coder and decoder (round-trip proof); float determinism between encoder and decoder
  (needs a fixed compiler/flags or fixed-point arithmetic); speed work (SIMD, fewer taps, update gating in the C core);
  8-bit grayscale / alpha / 16-bit paths.

## 5b. Experiment rounds 1-4 (full log: [ffic/experiments/LOG.md](../ffic/experiments/LOG.md), raw: `ffic/experiments/log.jsonl`)

42 logged experiments, each over the same 20 images, with code fingerprint, per-image and per-plane bytes, timing and mechanism stats.

* **Best: `stack5` = -1.20% vs JXL e9, -1.09% vs FunnelZip (Pexels), at about 6 s per image.** It beats JXL on the
  Pexels set (-2.67%) but still loses on Kodak (+1.74%).
* **Biggest single lever: AMYGDALA salience, inverted.** Learning less from surprising pixels (robust RLS, weight
  1/surprise^p): p=1 -0.89%, p=2 -1.33%, p=3 -1.46% vs the old best; clipping outliers in the skill and mouth estimators too
  adds about 0.1-0.4%. Amplifying salience (the literal amygdala) hurts +1.4%.
* **Small but real (stack):** registers (R/B funnels born from G's) -0.17%, gravity / depth pull -0.09%, continuous funnel
  field with small mouths + landing-only + sleep fission/prune about -0.1 to -0.3%, multi-scale + bounce contexts -0.1%,
  ACC probability correction -0.07%, second (surprise-selected) coder mixer -0.10%. Gains add when stacked.
* **Neutral:** hippocampal episodic recall, sleep replay, PE-driven forgetting, bounce path alone, multi-scale alone.
* **Hurts:** winner-take-all workspace (+0.6%), softer workspace (+0.6%), wide field mouths (+0.3%), curvature ball (+0.05%).
* **PFC (encoder picks the best mechanism set per image):** -0.01% beyond `stack5`, not worth the encode time.

## 5c. Colour physics: material lines and material funnels

* Pure colour-ratio constancy (R/G fixed per material) is **worse** than plain differences on both sets, even in linear light.
  The dichromatic model (body colour + light-coloured shine + camera offset) makes each material a *line* in colour space,
  R ~ a*G + b; fitted on 6 decoded neighbours it cut raw R error 12% on Kodak. In the coder (`mline=2`): -0.08% overall,
  -0.31% on Kodak, neutral on Pexels; the funnels' RLS colour taps had already learned most of it.
* **Material funnels** (`matf=1`): funnels in colour space (neighbourhood chromaticity; for B also the pixel's own R:G),
  each learning its material line across the whole image. As a ball: -0.07%. **The material's identity as a coder
  context (`mctx=1`): -0.19% more, on both sets, 18 of 20 images, B plane -0.52%.** Best so far: `matf_ctx`, -1.47% vs
  JXL e9 (Kodak +1.23%, Pexels -2.81%). Adding brightness to the landing removed most of the gain: materials are grouped
  by colour, not by how lit they are, as the physics says. Mouth size, memory length, cap and ridge barely matter.

## 5d. Gravity, time and parallel worlds

| Hypothesis | Mechanism | Result vs best single coder (`matf_ctx`, -1.47% vs JXL) |
|---|---|---|
| Gravity | each funnel's histogram of landed values acts as masses; inverse-square centre of mass near the prediction = RLS tap (`gmass`) | **-0.03%** (range/forgetting variants -0.03% to +0.01%): negligible |
| Time: coarse past | even rows first, odd rows also see the row below (`ilace`) | **+3.2% worse**: even rows lose more from distance-2 neighbours than odd rows gain |
| Time: arrow of time | encoder picks the best scan direction per image (4 options, 2 bits) | **-0.09%** for 4x encode time; 12 of 20 images prefer a flipped direction, by at most 0.43% |
| **Parallel worlds** | K complete coders run side by side; each pixel coded with their mixture, weights from evidence (`worlds.py`, `worlds_mix.c`) | 2 worlds, Bayes: 0.00%. 2 worlds, local: -0.39%. 3 worlds: -0.78%. **5 worlds, local decoherence: -1.07% -> -2.52% vs JXL e9** |

Parallel worlds is the largest gain since salience. Worlds that are individually worse (the old model is +2.6%) still add,
because they are right in different regions: weighting each world by its cost on the W, N, NW, NE pixels ("decoherence")
beats global Bayesian weighting, which settles on one world. **Caveat:** these are exact ideal code lengths of the mixture,
but a real decoder must run all K worlds and evaluate each world's probability for every candidate value (about 500 decision
nodes per world per pixel), so encode is about K x slower and decode much slower than one coder. Implementable, not cheap.

## 5e. Layers in parallel

* Colour layers coded fully independently: **+11.0% vs JXL** (12.7% bigger than linked layers). The links are essential.
* Horizontal strips, each on its own core: 2 strips +0.48% size for ~2x speed, 4 strips +1.18% for ~4x.
* **Staggered colour pipeline (`pipe=1`)**: G, R and B run on three threads at once; R waits until G is 2 rows ahead and B
  until R is, so every layer only reads finished rows and the output is deterministic (a decoder can mirror it). R/B
  funnels are born from G's funnels frozen every 32 rows, R starting after G's first 32 rows. Result on 20 images:
  **2.1x less wall time (127 s -> 60 s), +0.05% size** (-1.42% vs JXL e9). Not 3x because the layers are unequal work:
  B has the most learning inputs (22 vs 12 for G) and sets the pace. With registers off, pipeline and sequential give
  identical bytes, so the only size difference is which G funnels R/B inherit from.

## 6. Next steps, ranked by expected value

1. **Real coder + decoder + round-trip test** (must precede any claim). Small, and it fixes the numbers.
2. **FunnelBrain "sleep" for real:** MDL-gated *fission* (split a funnel only when a shadow child beats it over a window),
   plus periodic merge/prune, replacing my fixed quantisations. This is the one FunnelBrain mechanism I have not built.
   Expectation: low single digits.
3. **More balls with different timescales** (RLS with two forgetting factors, different tap sets) blended by the workspace.
4. **Speed:** gate and decimate RLS updates, SIMD the covariance update, drop taps by measured importance. Target:
   JXL-e9-class time at the current size.
5. **Use FunnelZip's strengths inside FFIC:** its exact-value hashed contexts and JPEG-block structure (8x8 lattice
   context) should help the Pexels-type images where FFIC still trails it.
6. Ideas from [IDEAS_PHYSICS.md](IDEAS_PHYSICS.md) not yet tried: coarse-to-fine two-pass context (all four sides),
   fuzzy-match (delay embedding) lens, diffusion-filled estimate as a ball.

## 7. Reproduce

```bash
cd ffic
python3 prep_data.py OUTDIR PEXELS_JPG_DIR KODAK_PNG_DIR        # raw RGB + JXL e9 reference sizes
clang -O2 -march=native -o ffic ffic.c -lm
QUIET=1 python3 bench.py OUTDIR none lcew ltpcrewb ltpcrewb:gate=0.5   # per-config totals (4 jobs in parallel)
./ffic OUTDIR/pex_p2253275.rgb 1024 768 ltpcrewb                 # one image
```
Flags: `l` learn, `t` spin, `v` level, `p` nesting, `c` colour taps, `e` energy mouth, `w` ball blend, `r` RLS,
`b` binary context-mixing coder, `f` non-causal colour taps. Options: `gate=`, `rl=` (RLS level mask), `lam=`, `lrm=`, `clim=`, `P0=`, `mu=`.

## 5f. Whole colour vs layers (tested, `ffic/experiments/LOG.md` group "whole colour")

Two ways to treat a pixel's colour as one thing instead of three separate layers. All numbers are over 20 photos, vs JXL e9.

| run | what | total | vs JXL | notes |
|---|---|---|---|---|
| `pipe` (control) | layers G→R→B, staggered threads | 10,332,754 B | −1.42% | wall 60 s |
| `inter` | pixel by pixel: G,R,B of each pixel before the next; G sees R−G, B−G of W and N | 10,362,565 B | −1.13% | worse on all 20; wall 176 s (per-pixel thread hand-off) |
| `inter_cmatch` | + "is the whole colour the same as W/N/NW/NE, and which?" — on yes, layers are free | 10,579,434 B | +0.94% | Kodak −0.6…−2.7% on 7/8; Pexels +0.03…+24% |
| `inter_cmatch_gated` | ask only where p(match) ≥ 0.55 | 10,505,799 B | +0.23% | fixes Pexels partly, loses Kodak gains |
| per photo: `pipe` or `inter_cmatch` (+1 byte) | encoder tries both | 10,283,976 B | **−1.88%** | new best total; 2× encode time |

Why: 33–76% of pixels repeat a neighbour's full colour, but on smooth high-quality photos the layered coder already codes those repeats at a fraction of a bit per layer, so a separate yes/no question costs more than it saves. On Kodak (scanned film, more texture) exact repeats are surprising to the layered coder, so the joint question wins. Pixel-by-pixel order by itself gives G nothing useful: the colour of the W/N neighbours is already implied by its own neighbours' G values plus the R/B registers.
Next if pursued: tell the layered coder "not a match" (exclude the candidate colours) on a no, instead of paying the flag on top.

### 5g. "Not a match" exclusion (tested)

After a "no", the pixel's colour is known to differ from every candidate. In G→R→B order this is exact information for B: if G and R equal a candidate's, B cannot be that candidate's B, so B's probability is renormalised over the remaining values (cost − log2(1 − P(excluded)), computed by walking the residual coder's decisions without learning, `cm_pval`). `minf=1` additionally gives the match question the G layer's own probability for the candidates' G values.

| run | alone vs JXL | Kodak vs `pipe` | best per photo with `pipe` | photos that pick it |
|---|---|---|---|---|
| `inter_cmatch` | +0.94% | −1.36% | −1.88% | 7 |
| `inter_cmatch_excl` (mexcl=1) | +0.71% | −1.70% | −1.99% | 8 |
| `inter_cmatch_excl_inf` (+ minf=1) | +0.52% | −1.75% | **−2.01%** | 8 |

Exclusion fires on 1–4% of pixels and saves 1.1–1.6 KB per photo. It does not rescue the smooth Pexels photos. There the yes/no answer is close to a coin flip (~0.8 bit per pixel), while the layered coder codes the same repeated pixels for about a tenth of a bit, and the gap is on G (p2014422: G 113 KB → 156 KB with the question). A separate question can only win where exact repeats surprise the layered coder (Kodak). CPU cost of `minf`: +20%.

## 6. Real bitstream and decoder (round trip verified)

`ffic file.rgb W H <options> out=file.ffic` now writes a real file, and `ffic -d file.ffic out.rgb` decodes it. File = header (magic, size, the option string so the decoder rebuilds the identical model) + three arithmetic-coded streams, one per colour layer (carry-less 32-bit binary coder, 16-bit probabilities). The decoder runs the same code path. Each value is decoded before anything reads it, and the threaded modes (pipe, pixel by pixel) wait on the same row/pixel hand-offs as the encoder, so the decoded result is deterministic.

What had to change for a decodable stream:
* The residual is coded as a walk down one binary tree (zero? → sign → bucket → mantissa) shared by coder, decoder, probability lookup and exclusion (`RW`/`rw_*`).
* Whole-colour match: the encoder used the pixel's own R and B to update the match model right away. Now the model learns at the start of the next G pixel, once R and B are known to the decoder too. The numbers don't change, because nothing reads that model in between.
* "Not a match" exclusion is coded for real: at every decision node the excluded values' probability mass is removed and the decision renormalised.

Round trip on all 20 photos (`ffic/roundtrip.py`, `ffic/experiments/ROUNDTRIP.md`):

| config | real file | ideal | coder overhead | vs JXL e9 | encode / decode (sum, s) | bit-exact |
|---|---|---|---|---|---|---|
| `best_now` | 10,331,759 B | 10,327,702 B | +0.039% | −1.43% | 136 / 138 | 20/20 |
| `pipe` | 10,337,138 B | 10,332,754 B | +0.042% | −1.38% | 78 / 76 | 20/20 |
| `inter_cmatch_excl_inf` | 10,541,110 B | 10,536,228 B | +0.046% | +0.57% | 267 / 273 | 20/20 |
| smallest per photo | **10,273,344 B** | | | **−1.98%** | | |

Decoding takes as long as encoding (symmetric, as in all context-mixing coders): about 7 s per photo for `best_now`, about 4 s with `pipe`.

## 7. Waves instead of pixels (wave tree, tested)

Idea: describe each colour layer with one horizontal and one vertical wave per region; split a region in four where its two waves do not fit, like a funnel splitting where it needs detail. 20 photos, sizes include the tree:

| test | what the file holds | total | vs best coder | vs JXL e9 |
|---|---|---|---|---|
| best coder (`best_now`) | surprises of the funnel guesser | 10,327,702 B | — | −1.47% |
| C. coarse wave tree as an extra guesser | surprises + 70 KB of tree | 10,385,926 B | +0.6% | −0.9% |
| A. waves first, then corrections (moderate / deep tree) | tree + per-pixel corrections | 20.15 / 19.90 MB | +95% / +93% | +92% / +90% |
| D. waves only, split until exact (`wave_exact.py`) | tree only, no corrections | 25,761,284 B | +149% | +146% |

D decodes exactly from the stream alone on 20/20 photos. Two waves almost never match a region exactly, so most of the image ends up as 1-pixel regions (e.g. the dog: 1.83 M of 2.36 M values). Waves describe large smooth shape well, but lossless coding is dominated by texture and sensor noise, which the funnel guesser handles far more cheaply.

## 8. Folding waves (tested: `ffic/fold.py`, `ffic/fold2.py`)

Fold the image in half again and again (reversible integer folds: no loss). What stays is a small stack of coarse waves; every fold leaves a fold wave that says exactly how to unfold. When the decoder unfolds, the stack is known in every direction (left, right, above, below), so the fold waves can be guessed from all sides. Measured with the same learning guesser and coder for raster and fold (ideal bits, all 3 colours; dog + kodim03):

| shape | dog | kodim03 | both |
|---|---|---|---|
| raster (left/above only) | 786,509 B | 510,869 B | — |
| averaging fold x1 / x2 / x3 / x5 | +4.4 / +5.5 / +6.0 / +6.8% | +3.6 / +5.4 / +6.4 / +7.6% | +4.1 / +5.5 / +6.2 / +7.1% |
| sharp fold (keep every other value) x1 / x5 | +2.1 / +4.4% | +2.8 / +6.4% | +2.4 / +5.2% |
| fold columns only (1 direction) | +3.6% | +0.5% | +2.4% |
| checkerboard fold | +5.9% | +0.2% | +3.7% |

Where the bits go (dog, green, sharp fold once): the fold waves cost **2.44 bits/value, 8% less than raster pixels (2.65)**, because they are guessed from all sides. But the stack costs **3.48 bits/value, 32% more**, because its own neighbours are now 2 px apart. The two cancel, and every extra fold makes the stack worse. The all-directions information is real, but in these tests it never pays for the extra cost of the stack. That matches the general rule: changing the order cannot add information, it only changes how easy each part is to guess.

### 8b. Folding the fold waves again (wavelet packets, tested)

Take the fold waves of one fold as a new image and fold them again, and so on. Same guesser and coder, green layer, sizes of the fold waves only:

| | dog | kodim03 |
|---|---|---|
| fold waves coded as an image | 201,464 B | 136,669 B |
| folded once more / twice | +9.2% / +11.1% | +5.5% / +8.3% |
| mosaic (stack + 3 fold waves as one image), folded again | 316,451 vs 296,538 B (+6.7%) | 197,185 vs 190,699 B (+3.4%) |

Every extra fold of the fold waves makes them bigger (the diagonal one by up to +15.6%). After the first fold, the fold waves are mostly edges and grain: there is no regular wave left for another fold to gather, and each fold pushes neighbours further apart. Wavelet packets help on strongly periodic textures (fingerprints, fabric), which these photos do not have.
