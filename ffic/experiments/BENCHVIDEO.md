# Lossless video benchmark (real files, all decoded and compared exactly)

First 30 frames of three Xiph test clips (352 x 288, 30 fps), converted to RGB (rgb24), so every coder gets identical frames. FFIC video mode (`frameh=288`): frames stacked into one tall image so all models keep learning across frames; each pixel also sees the previous frame (same spot + W, E, N, S, 5 taps) and two temporal balls. No motion search yet. Options: the best coder + `pipe=1 snapk=32`. Script: `benchvideo.py`, raw numbers: `benchvideo.json`.

| clip | FFV1 (archive standard) | x264 RGB lossless, veryslow | x265 lossless, veryslow | FFIC, frames independent | **FFIC video** | vs best other |
|---|---|---|---|---|---|---|
| akiyo (newsreader, still camera) | 2,157,649 B | 1,571,414 B | 1,569,264 B | 1,913,471 B | **1,085,865 B** | **−30.8%** (x265) |
| foreman (handheld, moving) | 2,477,057 B | 4,055,215 B | 4,086,002 B | 2,343,813 B | **2,236,692 B** | **−9.7%** (FFV1) |
| mobile (fast detailed pan) | 4,026,264 B | 5,316,547 B | 5,280,388 B | 3,874,368 B | **3,724,511 B** | **−7.5%** (FFV1) |

Speed for 30 frames: FFV1 and x264 under 2 s, x265 11–22 s to encode; FFIC 24–36 s to encode and the same to decode (under 1 frame per second at this small size). Caveats: the clips were filmed in YUV 4:2:0 and converted to RGB here, which makes the colour smoother than in camera-native RGB; archives usually store YUV directly, which FFIC does not support yet. Motion search (following moving objects) is not built yet; it should widen the gain on foreman and mobile.
