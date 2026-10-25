# 4K benchmark (3840 x 2160, lossless, real files)

Centre crops of full-size Pexels originals (camera JPEGs, so the crops still carry their 8x8 JPEG structure). Every file was decoded and compared with the original.

| photo | JPEG XL e9 | FFIC max | FFIC fast |
|---|---|---|---|
| p2166711 | 4,097,603 B (13 s / 0.6 s) | 3,694,530 B, -9.84% (162 s / 148 s) | 3,701,498 B, -9.67% (60 s / 63 s) |
| p2253275 | 6,947,810 B (24 s / 1.0 s) | 6,431,658 B, -7.43% (210 s / 133 s) | 6,441,399 B, -7.29% (86 s / 83 s) |
| p3225517 | 7,519,878 B (14 s / 0.7 s) | 7,383,266 B, -1.82% (133 s / 138 s) | 7,384,978 B, -1.79% (50 s / 55 s) |
| p417074 | 6,594,332 B (15 s / 0.7 s) | 6,413,368 B, -2.74% (134 s / 134 s) | 6,421,624 B, -2.62% (48 s / 53 s) |
| **total** | **25,159,623 B** | **23,922,822 B, -4.92%** | **23,949,499 B, -4.81%** |

Times are encode / decode seconds on this machine (FFIC max and fast ran side by side, sharing 4 cores).
