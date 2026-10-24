/* Native speed harness: compress (and optionally decompress) a file with fixed settings, N times.
 * Build: clang -O3 -o speed_harness speed_harness.c ../funnelzip/funnelzip.c
 * Use:   ./speed_harness FILE W C S R MASK REPS   (prints MB/s and size; runs long enough to profile) */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

size_t fz_compress(const uint8_t *in, size_t n, uint8_t *out, size_t cap, int W, int C, int S, int mask, size_t R, size_t skip);
int fz_decompress(const uint8_t *in, size_t len, uint8_t *out, size_t n, int W, int C, int S, int mask, size_t R, size_t skip);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage: %s FILE W C S R MASK REPS\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "rb");
    fseek(f, 0, SEEK_END); size_t n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *in = malloc(n), *out = malloc(n + n / 4 + 1024), *back = malloc(n);
    if (fread(in, 1, n, f) != n) return 1;
    fclose(f);
    int W = atoi(argv[2]), C = atoi(argv[3]), S = atoi(argv[4]), mask = atoi(argv[6]), reps = atoi(argv[7]);
    size_t R = (size_t)atol(argv[5]), m = 0;
    double t0 = now();
    for (int r = 0; r < reps; r++) m = fz_compress(in, n, out, n + n / 4 + 1024, W, C, S, mask, R, 0);
    double t1 = now();
    fz_decompress(out, m, back, n, W, C, S, mask, R, 0);
    double t2 = now();
    printf("%zu -> %zu  compress %.3f MB/s  decompress %.3f MB/s  %s\n", n, m, n * reps / 1e6 / (t1 - t0),
           n / 1e6 / (t2 - t1), memcmp(in, back, n) ? "MISMATCH" : "ok");
    return 0;
}
