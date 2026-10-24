# Physics-inspired ideas for FunnelZip: what is real maths and what is only a metaphor

**Status:** research note, 2026-10-24. Nothing here is implemented or measured yet. The benchmark that would test it
(`bench/jxl_vs.py`, 12 Pexels crops vs JXL) was stopped after 9 of 12 photos; the lossless numbers it printed are not a result.

**Starting point (Raaj's idea):** today FunnelZip finds one rhythm (the row width) and predicts a pixel from three
neighbours (left, up, up-left). Treat the picture instead as **one plane / field with many dimensions** (directions,
scales, colour, time) and let the funnel mix the votes of all of them.

**Evidence tags.** `[S]` = found in the searches below, link in Sources. `[K]` = from my own background knowledge, not
searched here, check before relying on it. `[?]` = speculation / my inference.

---

## 0. The honest summary

* String theory, gravity, extra dimensions and the Bekenstein bound **have no direct compression algorithm**. The
  searches found no paper applying MERA, Kaluza-Klein or entropic gravity to image compression. They stay inspiration.
* What *does* carry over is the ordinary maths those ideas are built from: **fields on a lattice** (Ising / Markov random
  fields), **boundary determines interior** (Laplace / diffusion), **coarse-graining across scales** (wavelets, pyramids,
  renormalization), **delay embedding** (Takens), **a small set of building blocks** (sparse dictionaries, Gaussian splats),
  **tensor factorisation** (matrix product states). Each of these already exists as a compression method.
* The biggest realistic win is the first idea: **more directions and a 2D predictor**. FunnelZip's grid lens only reads
  left / up / up-left (`funnelzip/funnelzip.c`, L_GRID block). JXL's modular mode uses more neighbours (NE, NW, WW) inside
  a learned context tree `[S: JPEG XL paper]`. That is measurable headroom, not a metaphor.
* Photos have a **noise floor**: sensor noise is incompressible, so expect single-digit-percent gains on photos, larger
  gains on synthetic or textured images.
* **Benchmark caveat I should have flagged earlier:** the Pexels images are JPEG-compressed by the CDN, so their pixels carry
  an 8x8 DCT block lattice that raw camera files do not. Kodak (lossless source) is a better sanity check; use both.

---

## 1. One picture of the combined design ("FunnelField")

Every "dimension" becomes a family of lenses; the funnel (mixers) learns which dimension to trust at each pixel.

```
                 pixel (x, y, colour c) to predict
                          |
   SPACE (directions)   SCALE (micro -> macro)   COLOUR        HIDDEN STATE        ORDER / TIME
   N NE E NW NN WW      coarse pyramid level     R->G->B       online cluster id   2-pass checkerboard
   + edge-tangent       diffusion fill from      cross-channel (texture class)     previous scan / frame
   + 2D wave predictor  coarse layer             taps                              fuzzy delay-embedding match
                          \            |              |              /               /
                           +-----------+--- FUNNEL (mixers) --------+---------------+
                                              |
                                      REFINE (APM)  ->  arithmetic coder
```

---

## 2. Idea catalog

Columns: what it is, why it should help, which FunnelZip stage it touches, cost, expected gain.
Gain is my guess, not a measurement; the test plan in section 4 replaces guesses with numbers.

### A. SPACE: more directions (Raaj's first idea)

| # | Idea | Real maths / evidence | Where in FunnelZip | Cost | Expected gain |
|---|---|---|---|---|---|
| A1 | **Directional lenses**: add NE, NN, WW, anti-diagonal and hashed 2D patch contexts (e.g. the 3x3 causal neighbourhood) | JXL's tree tests N, W-WW etc.; NE/NW/WW "become useful in combination with a meta-adaptive context model" `[S]` | New inputs next to the 3 in L_GRID | small | **medium** |
| A2 | **Orientation field**: estimate local edge direction (structure tensor) and use it as a context, and predict *along* the edge ("a string": 1D curve, vibrating mode = Fourier descriptor) | Edge-enhancing diffusion steers by edge direction `[S: Weickert group]`; string-as-1D-object is the metaphor, edge tangent is the maths | Context for mixer selection + one extra predictor | small | medium |
| A3 | **2D wave predictor**: online least-squares / LMS over ~12-18 causal neighbours. A plane wave at any angle obeys a linear recurrence over a 2D neighbourhood, the 2D version of how the resonator lens already catches sums of sines in 1D | Linear-predictive lossless image coders (TMW, MRP, GLICBAWLS) `[K]` | New lens beside Resonator (L_RESONATOR) | medium, slower | **medium-high** |
| A4 | **Bidirectional / non-causal context** via a two-pass order: code a checkerboard (or every other pixel) first, then the rest see neighbours on all four sides | Markov random fields are *undirected*; autoregressive models only see the past `[S: MRF sources]`. Checkerboard context models are used in learned codecs `[K]` | Change of scan order + new lens inputs | medium | medium |
| A5 | **Scan order**: Hilbert curve instead of raster | +up to 15% in QOI-style coders `[S]`, but theory says a strong predictor may not benefit `[S]`; learned orders exist `[S: Neural Space-filling Curves]` | Preprocessing | small | low-uncertain, cheap to try |
| A6 | **Colour as a dimension**: full R->G->B cross-channel contexts (the LMS already has cross-channel taps, `lx` in the core) | Standard in lossless photo coders `[K]` | Strengthen existing taps | small | small-medium |

### B. SCALE: micro to macro (renormalization / wavelets / MERA)

| # | Idea | Evidence | Where | Cost | Gain |
|---|---|---|---|---|---|
| B1 | **Pyramid coding**: code a coarse version first; each fine pixel then has context from *all sides* at the coarse level plus causal neighbours | L3C: learned multi-scale auxiliary representations beat a fixed RGB pyramid and beat PNG/WebP/JPEG2000 `[S]`. Renormalization group = coarse-graining is the analogy; wavelets/pyramids are the maths | New "scale lens"; coarse layer coded first | medium-large | **medium** |
| B2 | **MERA / tensor-network wavelets** | MERA is wavelet-related; one talk says it may yield wavelets optimal for image compression, **unfinished, no results found** `[S]` | n/a | research | unknown, do not plan on it |
| B3 | **Self-similarity across scales** (fractal coding): predict a block from a shrunk / rotated / mirrored copy of another block | Partitioned IFS: range blocks matched to larger domain blocks via affine maps incl. reflection; encoding is slow, decoding fast `[S]` | A 2D, symmetry-aware version of the Echo lens | large search cost | low on photos, higher on textures/synthetic |

### C. FIELDS AND FORCES: gravity, diffusion, least action

| # | Idea | Evidence | Where | Cost | Gain |
|---|---|---|---|---|---|
| C1 | **Boundary determines interior** ("holographic" in spirit; precisely: a harmonic function is fixed by its boundary values). Store sparse / edge pixels, rebuild the rest by diffusion | PDE codecs keep a sparse pixel set + edge-enhancing diffusion; clearly beat JPEG at high compression, near JPEG2000; ~4% of pixels can rebuild an image `[S]` | Lossy front end; or, for lossless, a **diffusion-filled estimate from a coarse layer as one more lens input** | medium | lossy: good at high compression; lossless: small |
| C2 | **Gravity as attraction**: points falling toward centres = clustering. Concretely: vector quantisation (k-means), adaptive sampling that puts more samples where detail is, centroidal Voronoi layouts | VQ is used in GaussianImage `[S]`; adaptive triangulation in PDE codecs `[S]`. "Entropic gravity" (force emerging from information) has **no compression mapping**; metaphor only `[S: reviews found no link]` | Lossy codebooks | medium | lossy only |
| C3 | **Statistical mechanics (Ising / Potts / Gibbs)**: pixels are spins with learned couplings to neighbours | Context modelling for images is the practical form; high-order contexts cause *context dilution* and the pattern space explodes for long-range interactions `[S]`. FunnelZip already answers this by hashing plus mixing | Already the architecture; ideas: a learned "temperature" in the APM stage `[?]` | none-small | small |
| C4 | **Least action / Lagrangian prediction** for time (video, audio) | Lagrangian / Hamiltonian nets exist for video prediction (LagNetViP, LaWM); **no compression result found** `[S]` | Strengthen L_MOTION | large | video only, speculative |

### D. PARTICLES AND ATOMS: a few building blocks make everything

| # | Idea | Evidence | Where | Cost | Gain |
|---|---|---|---|---|---|
| D1 | **Dictionary atoms** (Gabor / learned, overcomplete) and matching pursuit | MP codecs beat JPEG2000 at low bitrates; overcompleteness is the reason `[S]` | Lossy front end, or an "atom bank" of predictors | medium-large | lossy: good at low rate |
| D2 | **Priming with natural-image statistics** ("the standard model": ship the few fundamental statistics, not per-file learning). FunnelZip already has a primed option (`primer_for`) | Brotli ships a built-in dictionary; same idea for pixels `[K]` | Pre-trained tables, decoder must carry them | medium | **medium on small images**, little on large |
| D3 | **Gaussian splats as particles**: 8 parameters per 2D Gaussian, vector-quantised, 1000+ FPS decode | GaussianImage: rate-distortion comparable to COIN/COIN++ `[S]` | Lossy base layer | medium | lossy; as base + lossless residual it usually *loses* to direct lossless `[?]` |
| D4 | **Hierarchy quark -> atom -> molecule** = pixels -> patches -> textures: hierarchical codebooks | Equivalent to hashed 2D patch contexts (A1) plus a patch-level Echo `[?]` | merge into A1 / B3 | - | - |

### E. EXTRA DIMENSIONS and STRINGS

| # | Idea | Evidence | Where | Cost | Gain |
|---|---|---|---|---|---|
| E1 | **Hidden state ("compact dimension")**: a variable that is not a pixel but shapes the pixels, e.g. an online-learned texture-class id used as context | Hidden states approximate MRFs inside causal context coding (Pickard fields) `[S: DTU]`; manifold hypothesis says images live on low-dimensional manifolds, neural compressors are not optimal on them `[S]` | Online clustering -> context | medium | small-medium |
| E2 | **Tensor networks / low rank**: reshape image to a tensor, truncate bond dimension (Eckart-Young) | Matrix-product-state image compression is ~20 years old, revived by quantum-inspired work; compression error roughly independent of image size in the efficient regime `[S]`; tensorisation is not systematic `[S]` | Lossy transform | medium | uncertain |
| E3 | **Graph Fourier transform**: pixels are nodes, edges cut at discontinuities, Laplacian eigenvectors are the basis | Beats DCT across sharp edges; cost is signalling the graph `[S]` | Lossy transform | large | lossy edges/synthetic |
| E4 | **String theory proper, Kaluza-Klein** | No usable compression mathematics found `[S]` | none | - | none; keep as storytelling only |

### F. TIME and ORDER

| # | Idea | Evidence | Where | Cost | Gain |
|---|---|---|---|---|---|
| F1 | **Delay embedding (Takens)**: represent each position by its delayed copies (x[n], x[n-1], x[n-S], x[n-S-1], ...), then predict from the nearest earlier neighbours in that space. This is **fuzzy matching**: today the Echo lens needs an *exact* 6-byte repeat | Delay embedding + nearest-neighbour forecasting is standard in nonlinear time-series work `[S]` | New approximate-match lens (quantised hashed patches) | medium | **medium** on textures, fur, foliage |
| F2 | **Cellular-automaton / NCA rule**: a tiny shared local rule generates texture | Self-organising texture NCAs exist; parameter counts not confirmed `[S]`. Lossy/perceptual, not exact. As a *predictor* it is just a small online-trained MLP over the neighbourhood (cmix style) `[K]` | Optional MLP lens | large, slow | small-medium, expensive |
| F3 | **Kolmogorov / MDL view**: image = short generator + noise; compression is prediction | Vitanyi-Li: compression is almost always the best strategy for prediction; K is uncomputable, practice uses probabilistic code lengths `[S]`. The Bekenstein / holographic bounds cap information per region but give no codec `[S]` | Framing and stopping rule only | - | none directly |
| F4 | **8x8 block lattice** (for JPEG-derived pixels) | Pexels CDN images are lossy JPEGs `[K]`; context on (x mod 8, y mod 8) and DCT-aware prediction would exploit it | Context input | small | gain on these test files only, not on raw photos |

---

## 3. What I would build first (recommendation)

1. **Tier 1, cheap and likely real:** A1 + A2 (more directions, edge-tangent context), A3 (2D wave predictor), A6 (colour).
2. **Tier 2, medium effort:** A4 (two-pass bidirectional), B1 (pyramid lens), F1 (fuzzy delay-embedding match), E1 (hidden state).
3. **Tier 3, mainly for a lossy mode or a research bet:** C1 diffusion fill, D1 atoms, D3 splats, B3 fractal match, E2 tensor networks.
4. **Skip as engineering, keep for the video:** string theory, Kaluza-Klein, entropic gravity, Bekenstein bound.

## 4. Test plan (replaces the guesses above)

* Fix a corpus: the 12 Pexels crops **plus** the Kodak photos (lossless source), so JPEG-lattice effects can be separated.
* Baselines: current FunnelZip, JXL lossless effort 9, WebP lossless, PNG. All pixel-exact and verified round-trip.
* Ablation: add **one** lens at a time, then cumulatively; record bytes and seconds per image. A lens that costs more
  time than it saves in bytes is dropped.
* Report per-image wins and losses, not only totals (one earlier image already lost to JXL by about 1%).
* Lossy ideas (C1, D1, D3, E2, E3) get a separate PSNR / SSIM-at-equal-size test against JXL, as in `bench/jxl_vs.py`.

## Sources

* Tensor networks / holography: [Tensor Networks from Kinematic Space (Czech et al.)](https://arxiv.org/abs/1512.01548), [Evenbly talk, wavelet-MERA for image compression (Perimeter)](https://events.perimeterinstitute.ca/event/2/contributions/19)
* Tensor-train / MPS images: [Tensor network methods for quantum-inspired image processing](https://arxiv.org/abs/2510.23089), [Tubal tensor train](https://arxiv.org/pdf/2603.10503)
* PDE / diffusion codecs: [Galic, Weickert et al., image compression with anisotropic diffusion](https://researchportal.hw.ac.uk/en/publications/image-compression-with-anisotropic-diffusion/), [Hybrid JPEG + PDE codec](https://arxiv.org/abs/2102.01138), [Mainberger thesis](https://www.mia.uni-saarland.de/Publications/mainberger-phd14.pdf)
* Gaussian splats: [GaussianImage (ECCV 2024)](https://arxiv.org/abs/2403.08551), [GaussianImage++](https://ojs.aaai.org/index.php/AAAI/article/view/37572)
* Fractal / self-similarity: [Fractal image compression overview](https://www.sfu.ca/~rpyke/335/projects/farhad/fic_essay.htm), [Denoising diffusion as partitioned IFS](https://arxiv.org/abs/2603.13069)
* MRF / Ising context modelling: [Row-centric lossless compression of Markov images](https://arxiv.org/abs/1702.08055), [Maximal entropy random walk, Ising-like models](https://arxiv.org/abs/1912.13300), [Context-adaptive coding of bi-level images (DTU)](https://orbit.dtu.dk/en/publications/context-adaptive-coding-of-bi-level-images/)
* Sparse dictionaries / matching pursuit: [Monaci et al., learned overcomplete dictionaries (EPFL)](https://infoscience.epfl.ch/record/87107/files/Monaci2004_777.pdf?version=1)
* Implicit neural / manifold: [COIN++](https://arxiv.org/abs/2201.12904), [Do neural networks compress manifolds optimally?](https://arxiv.org/abs/2205.08518)
* Multi-scale lossless: [L3C, Practical full resolution learned lossless image compression](https://arxiv.org/abs/1811.12817)
* JXL predictors / MA trees: [The JPEG XL image coding system](https://arxiv.org/abs/2506.05987), [Cloudinary: JPEG XL modular mode explained](https://cloudinary.com/blog/jpeg-xls-modular-mode-explained)
* Scan order: [Neural space-filling curves](https://sotaverified.org/papers/neural-space-filling-curves), [Scanning for compression (arXiv cs/0609049)](https://arxiv.org/abs/cs/0609049)
* Delay embedding: [Selecting embedding delays (arXiv 2302.03447)](https://arxiv.org/abs/2302.03447)
* Cellular automata: [Self-Organising Textures (Distill)](https://distill.pub/selforg/2021/textures), [Multi-texture synthesis with NCA](https://arxiv.org/abs/2407.05991)
* Physics-informed video: [LagNetViP](https://arxiv.org/abs/2010.12932), [LaWM](https://arxiv.org/abs/2605.08279)
* Kolmogorov / MDL: [Vitanyi and Li, MDL induction, Bayesianism and Kolmogorov complexity](https://arxiv.org/abs/cs/9901014)
* Graph transforms: [Variable-size symmetry-based graph Fourier transforms](https://arxiv.org/abs/2411.15824)
* Holographic / entropy bounds: [Bekenstein and the holographic principle](https://arxiv.org/abs/1212.1711)
