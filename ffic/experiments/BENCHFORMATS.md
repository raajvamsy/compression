# Formats benchmark (lossless, real files, all decoded and compared exactly)

## Images with transparency and palette images

| image | what | PNG (optimized) | JPEG XL e9 | FFIC max | FFIC vs best other |
|---|---|---|---|---|---|
| fewcolours_rgb.png | RGB, 768x512 | 257,978 | 81,290 | 79,750 | **-1.9%** |
| la.png | LA, 768x512 | 213,686 | 138,834 | 135,341 | **-2.5%** |
| palette.png | P, 768x512 | 109,623 | 81,290 | 78,409 | **-3.5%** |
| palette_transp.png | P + transparency, 768x512 | 150,751 | 118,907 | 115,300 | **-3.0%** |
| rgba_soft.png | RGBA, 768x512 | 584,930 | 332,446 | 325,917 | **-2.0%** |

## Video in its native format (YUV 4:2:0, first 30 frames of the Xiph CIF clips, no colour conversion)

| clip | FFV1 | x264 lossless | x265 lossless | FFIC max | FFIC vs best other |
|---|---|---|---|---|---|
| akiyo | 1,596,798 (0.1 s) | 595,227 (0.2 s) | 588,367 (3.5 s) | 518,041 (10 s / 10 s) | **-12.0%** |
| foreman | 2,029,752 (0.1 s) | 1,556,770 (0.4 s) | 1,561,780 (5.8 s) | 1,503,346 (12 s / 12 s) | **-3.4%** |
| mobile | 2,998,878 (0.2 s) | 2,101,373 (0.5 s) | 1,972,752 (6.3 s) | 2,109,880 (13 s / 14 s) | **+7.0%** |

FFIC times are encode / decode seconds; other tools' times are encode seconds. FFIC video codes the Y, U and V layers in parallel (3 threads).
