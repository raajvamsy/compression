# RGB layer experiment

`original.jpg` (Pexels photo 1366919) is split into 3 channel layers; each layer is split again into 4 shade bands.

- `layer_1_red/`, `layer_2_green/`, `layer_3_blue/`: `<name>_layer.png/.npy` (raw values 0-255), `_tinted.png`
  - `band_1_darkest_0-63/`, `band_2_dark_64-127/`, `band_3_light_128-191/`, `band_4_lightest_192-255/`
    - `<name>_band.png`: tinted shades, transparent where pixel is outside the band

Summing the 4 bands of a layer gives back that layer exactly.
