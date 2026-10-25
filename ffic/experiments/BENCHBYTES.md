# Any file as an image (lossless, real streams, best layout decoded and compared)

Each file's bytes are laid out as a grey image (one byte = one pixel), or read as 16-bit numbers, at several widths; FFIC (preset max, one layer) keeps the smallest.

| file | size | zip -9 | bzip2 -9 | xz -9e | FLAC | FFIC as an image | best layout | FFIC vs best other |
|---|---|---|---|---|---|---|---|---|
| experiment_log.jsonl | 551,098 | 69,908 | 50,008 | 41,204 | — | 372,842 | bytes, width 235 | **+804.9%** |
| photo.jpg | 258,479 | 257,594 | 256,481 | 258,148 | — | 266,747 | bytes, width 256 | **+4.0%** |
| program_binary | 96,920 | 45,494 | 45,480 | 38,464 | — | 84,215 | bytes, width 48 | **+118.9%** |
| sensor_records.bin | 720,000 | 317,641 | 249,704 | 153,464 | — | 118,910 | bytes, width 12 | **-22.5%** |
| source_code.c | 80,104 | 25,705 | 23,150 | 23,424 | — | 66,671 | bytes, width 1024 | **+188.0%** |
| terrain_heights.hgt | 2,880,000 | 1,080,962 | 538,832 | 710,208 | — | 323,666 | 16-bit big-endian, width 1200 | **-39.9%** |
| voice_60s.wav | 2,880,078 | 2,216,028 | 1,915,148 | 1,972,440 | 1,501,431 | 1,941,124 | 16-bit little-endian, width 2 | **+29.3%** |

FFIC size includes 8 bytes for the original length and layout. 'best other' is the smallest of zip, bzip2, xz (and FLAC for audio).
