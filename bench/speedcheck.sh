#!/bin/zsh
# speed + size check of the C core on fixed settings (no Listen), 5 file types
cd "$(dirname "$0")/.."
clang -O3 -o bench/speed_harness bench/speed_harness.c funnelzip/funnelzip.c || exit 1
for job in "samples/extra/text_1mb.txt 1 1 0 0 5575" "samples/audio/system_sounds.wav 2 1 0 682 8191" \
           "samples/extra/landscape.rgb 1 3 1536 3072 8191" "samples/text/sensor.csv 1 1 0 136 5575" \
           "samples/extra/zsh_binary 1 1 16 0 8191"; do
  printf "%-34s " "$(basename ${job%% *})"; bench/speed_harness ${=job} 1
done
