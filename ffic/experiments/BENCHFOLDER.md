# Folder benchmark: 20 photos as ordinary PNG files, compressed as one archive

The 20 test photos (12 Pexels crops 1024x768, 8 Kodak 768x512) saved as normal optimized PNGs in one folder, then archived. zip / tar.xz / 7z keep the PNG files byte for byte; JPEG XL and FFIC keep every pixel exactly (the PNG files they give back hold identical pixels, but the PNG bytes themselves are re-encoded).

| method | archive size | vs the PNG folder | time to pack |
|---|---|---|---|
| folder of PNGs (20 files) | 18,053,552 B | — | — |
| zip -9 | 18,015,523 B | -0.2% | 1 s |
| tar.xz -9e | 17,748,828 B | -1.7% | 11 s |
| 7z LZMA2 ultra | 17,747,598 B | -1.7% | 9 s |
| every photo -> JPEG XL e9 | 10,481,322 B | -41.9% | 38 s |
| ffic.py pack --preset archive | 10,254,187 B | -43.2% | 610 s |
| ffic.py pack (max) | 10,329,410 B | -42.8% | 47 s |

FFIC unpack: 50 s, every photo pixel-exact: yes. FFIC packs 4 photos at a time on 4 cores.

`--preset archive` (per photo: 7 reversible tricks with the main brain, the best two also with the whole-colour brain, smallest kept): 10,254,187 B, 0.73% smaller than `max` and 2.17% smaller than converting every photo to JPEG XL e9. Packing took 610 s (about 9 encodes per photo, 4 photos at a time), unpacking 72 s; every photo pixel-exact.
