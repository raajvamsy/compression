# Similar problems in other fields, and how they were solved

Research notes (2026-10-25) for the ideas in `IDEAS_FUNNELBRAIN.md` sections 5–8: all-direction references, folding,
waves instead of pixels, mini groups, far-away links, splitting by difficulty. Each item: what the other field does, what
it says about our ideas, and what to try.

## 1. "Use neighbours in every direction" / folding = interpolation coding

| field | what they do | lesson for us |
|---|---|---|
| Scientific data compression (SZ3, cuSZ-i, MGARD+) | Multi-level interpolation: keep a coarse grid, then fill each finer level by interpolating from known points on **both sides** (our fold), versus the "Lorenzo" predictor that only uses already-coded neighbours (our raster). | Interpolation wins when the error allowed is loose (lossy); the causal Lorenzo predictor wins when the error bound is tight. Lossless is the tightest bound, which matches our fold result (+0.2 to +7%). Newer coders (HPEZ) try both and keep the better one per dataset. |
| Classic lossless image coding (Aiazzi et al. 1996; scanning-order theory) | Progressive / hierarchical interpolation vs optimum causal DPCM. | "Progressiveness is usually realised with a performance penalty" vs the best causal predictor; theory: with unlimited context the scan order does not change the total information. Our measurement (fold waves 8% cheaper, stack 32% dearer) is that same penalty. |
| JPEG XL | Its "squeeze" is a reversible Haar fold, used for progressive and lossy images. | The encoder turns squeeze **off by default for lossless**, the same conclusion. |
| Hierarchical prediction for colour (Kim and Cho, IEEE TIP 2014) | Chroma predicted with **lower-row** pixels too, using the already-coded luma. | Here all-directions info works: a second colour can look in every direction at a first colour that is fully decoded. This is our `f` flag (red and blue see green's right and below), never tested yet. **Test it.** |
| Video coding (B-frames, hierarchical GOP) | B-frames predict from past **and future** frames; they cost a few % of an I-frame. | Bidirectional works when the anchors are few and the rest is cheap; the bits move to the anchors, so encoders spend more on them. Same trade as our stack vs fold waves, but photos lack video's time redundancy. If we fold, keep the stack tiny and code it with the full coder. |
| Learned lossless (L3C, LC-FDNet) | Multi-scale: code a low-frequency / sub-image first, use it as the prior for the high-frequency parts. | Works with large neural models; L3C was still marginally behind FLIF. The neural nets carry the gain, not the order. |

## 2. "Store waves instead of pixels"

Lossy coders (JPEG, JPEG XL VarDCT, SZ at loose bounds) win with waves because they drop the small ones. Lossless coders that
lead benchmarks are prediction + context mixing (GraLIC tops the 2011 photo benchmark; FLIF and JPEG XL lead the 2021
format comparison). Our tests A/C/D (+0.6% to +149%) agree.

## 3. "Mini set groups" and "split by difficulty"

| field | what they do | lesson |
|---|---|---|
| MRP (Minimum Rate Predictors) | Encoder splits the image into regions/classes, optimises one linear predictor + error model per class over the **whole** image (two-pass), and sends the class map + coefficients. | Our "mini groups decided by the encoder" exists and is among the strongest classic lossless designs. Our funnels learn backward (decoder-side); MRP learns forward (encoder sends). A hybrid (encoder-sent groups, funnels learning inside) is untested. |
| DNA compression (GeCo2/3) | Many competing context models; each block uses the cheapest, or a mixer weights them by recent performance; GeCo3's neural mixer gains 2.4–7.1%. | Our referee/mixer is the same idea; a stronger mixer (small neural) is a known lever. |
| Lossless audio (OptimFROG, Monkey's Audio) | Cascades of adaptive filters (backward-adaptive, nothing sent) vs FLAC's per-frame sent coefficients. | Strongest modes are slow to decode because they are symmetric, which is our speed problem; the field accepts it or offers faster modes. |

## 4. "Links to far-away look-alikes" / "fold where it resonates"

| field | what they do | lesson |
|---|---|---|
| Example-based / template-matching lossless coding (Sumi et al.; EUSIPCO 2018; hash-based template matching) | Search the already-decoded area for neighbourhoods like the current one; use what followed them as a prediction or probability model, mixed with linear prediction. | Our "far links" idea is established and reported **competitive with state of the art**; hashing keeps the search fast. **Test it** (strongest on texture, where we trail on Kodak). |
| DNA: inverted repeats | Each context also learns from the reverse-complement strand: a "mirror fold" that matches how DNA really repeats. | Folding pays off when the data truly has that symmetry. For photos, the analogue is repetition (texture period), not mirroring. |
| Text: Burrows–Wheeler transform | A reversible reordering that groups characters by their **right-hand** context, with no side information. | A "fold" that gives right-context benefit for free exists for 1-D text, because sorting by context is reversible. There is no clean 2-D image version; the 2-D analogue is template matching. |

## What to try next (ranked by evidence)

1. **Cross-colour all-directions (`f`)**: red and blue see green on every side. Supported by Kim and Cho; zero cost; one config letter.
2. **Template-matching links** (hash search of the decoded area), as an extra guesser and context; reported competitive with state of the art.
3. **MRP-style encoder-sent groups** + funnels inside each group.
4. **Stronger mixer** (small neural mixer, GeCo3-style).
5. Folding only as an encoder choice per region (HPEZ-style "try both, keep the better"), with a tiny stack coded by the full coder.

## Sources

- SZ3: https://arxiv.org/pdf/2111.02925 · cuSZ-i: https://arxiv.org/pdf/2312.05492 · MGARD+: https://arxiv.org/pdf/2010.05872 · HPEZ: https://arxiv.org/pdf/2311.12133
- Scanning order theory: https://arxiv.org/pdf/cs/0609049 · hierarchical lossless coding: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/352E33ACD3E0F736B3979DE101090529/S2048770316000172a.pdf/lossless-image-coding-using-hierarchical-decomposition-and-recursive-partitioning.pdf · Aiazzi et al. (via CNR): https://iris.cnr.it/handle/20.500.14243/489 · Kim and Cho 2014: https://unpaywall.org/10.1109%2FTIP.2013.2293428
- JPEG XL: https://arxiv.org/pdf/2506.05987 · https://linuxmanpag.es/man/1/cjxl/ · https://github.com/libjxl/libjxl/blob/f88745497118727f861cb00887cadcb395d10f1c/doc/format_overview.md
- Video GOP / B-frames: https://en.wikipedia.org/wiki/Group_of_pictures · https://www.forasoft.com/learn/video-encoding/articles/gop-structure-ipb-frames
- Learned lossless: L3C https://arxiv.org/pdf/1811.12817 · LC-FDNet https://arxiv.org/pdf/2112.06417
- Benchmarks: https://www.imagecompression.info/gralic/LPCB-June.html · https://arxiv.org/pdf/2108.02557 · https://arxiv.org/pdf/2404.19755
- MRP / example search: https://arxiv.org/pdf/1802.04499 · https://new.eurasip.org/Proceedings/Eusipco/Eusipco2018/papers/1570438304.pdf · https://www.microsoft.com/en-us/research/publication/hash-based-line-by-line-template-matching-for-lossless-screen-image-coding/
- DNA: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7657843/ · https://sweet.ua.pt/an/publications/2008b.pdf
- Audio: https://wiki.multimedia.cx/index.php/OptimFROG · https://hydrogenaudio.org/index.php?msg=434013
- BWT: https://www.johndcook.com/blog/2024/10/15/burrows-wheeler
