# RGB layer experiment

`original.jpg` (Pexels photo 1366919) is split into 3 channel layers, each split again into 2 layers by brightness (threshold 128).

- `layer_1_red/`, `layer_2_green/`, `layer_3_blue/`
  - `<name>_layer.png/.npy`: raw channel values (0-255); `_tinted.png`: same in its own color
  - `sub_1_white/`: pixels with value >= 128
  - `sub_2_black/`: pixels with value < 128
