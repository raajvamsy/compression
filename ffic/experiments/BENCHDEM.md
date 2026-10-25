# 16-bit benchmark: SRTM elevation tiles (lossless, real files, all decoded and compared)

Full 1-arc-second tiles, 3601 x 3601 heights (12,967,201 samples, 16-bit). PNG and JPEG XL store the same 16-bit grey image; xz is the raw 16-bit heights, compressed.

| tile | raw | xz -9e | PNG 16-bit (optimized) | JPEG XL e9 | FFIC max | FFIC vs best other |
|---|---|---|---|---|---|---|
| Grand Canyon (N36W113, 517–2807 m) | 25,934,402 | 7,308,580 | 7,056,291 | 3,929,750 | 3,732,532 (44 s / 45 s) | **-5.0%** |
| French Alps (Mont Blanc) (N45E006, 232–4797 m) | 25,934,402 | 9,871,948 | 9,240,682 | 5,246,884 | 4,895,779 (40 s / 43 s) | **-6.7%** |
| Swiss Alps (Valais) (N46E007, 398–4469 m) | 25,934,402 | 9,397,444 | 8,819,193 | 5,090,414 | 4,775,794 (42 s / 43 s) | **-6.2%** |

FFIC times: encode / decode seconds (one layer, `planes=1`, `bits=13`/`bits=12` chosen from the data).
