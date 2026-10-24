/* Parallel worlds: exact mixture of K complete coders from their per-pixel code lengths.
 * A decoder can do the same: it runs every world, evaluates each world's probability for every candidate value, and
 * mixes them with weights computed only from already-decoded pixels.  usage: worlds_mix W H alpha beta cost1 cost2 ...
 * prints bits for: each world, Bayes (cumulative posterior), fixed-share switching (alpha), local decoherence (beta:
 * weights from the worlds' costs on the W, N, NW, NE pixels), and local x fixed-share. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(int argc, char **argv) {
    int W = atoi(argv[1]), H = atoi(argv[2]); double alpha = atof(argv[3]), beta = atof(argv[4]); int K = argc - 5;
    size_t N = (size_t)W * H; float **L = malloc(K * sizeof(float *));
    for (int k = 0; k < K; k++) { L[k] = malloc(3 * N * sizeof(float)); FILE *f = fopen(argv[5 + k], "rb"); if (!f || fread(L[k], sizeof(float), 3 * N, f) != 3 * N) { fprintf(stderr, "bad %s\n", argv[5 + k]); return 1; } fclose(f); }
    double single[16] = {0}, bayes = 0, fs = 0, loc = 0, locfs = 0;
    static const int order[3] = {1, 0, 2};
    for (int oi = 0; oi < 3; oi++) {
        int c = order[oi]; double lb[16], lf[16];                         /* log2 weights */
        for (int k = 0; k < K; k++) lb[k] = lf[k] = -log2((double)K);
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            size_t i = (size_t)c * N + (size_t)y * W + x; double p[16], mx;
            for (int k = 0; k < K; k++) { p[k] = pow(2.0, -L[k][i]); single[k] += L[k][i]; }
            #define MIX(lw, out) { double s = 0, z = 0; mx = -1e300; for (int k = 0; k < K; k++) if (lw[k] > mx) mx = lw[k]; \
                for (int k = 0; k < K; k++) { double w = pow(2.0, lw[k] - mx); z += w; s += w * p[k]; } out += -log2(s / z); }
            MIX(lb, bayes); MIX(lf, fs);
            double ll[16], lc[16];                                        /* local decoherence: neighbours' costs */
            for (int k = 0; k < K; k++) {
                double sc = 0; int n = 0;
                if (x > 0) { sc += L[k][i - 1]; n++; } if (y > 0) { sc += L[k][i - W]; n++; if (x > 0) { sc += L[k][i - W - 1]; n++; } if (x < W - 1) { sc += L[k][i - W + 1]; n++; } }
                ll[k] = -beta * sc; lc[k] = ll[k] + lf[k];
            }
            MIX(ll, loc); MIX(lc, locfs);
            for (int k = 0; k < K; k++) { lb[k] -= L[k][i]; lf[k] -= L[k][i]; }   /* evidence */
            mx = -1e300; for (int k = 0; k < K; k++) if (lf[k] > mx) mx = lf[k];
            double z = 0; for (int k = 0; k < K; k++) z += pow(2.0, lf[k] - mx);
            for (int k = 0; k < K; k++) { double w = pow(2.0, lf[k] - mx) / z; lf[k] = log2((1 - alpha) * w + alpha / K); }   /* fixed share */
            mx = -1e300; for (int k = 0; k < K; k++) if (lb[k] > mx) mx = lb[k]; for (int k = 0; k < K; k++) lb[k] -= mx;
        }
    }
    printf("worlds");
    for (int k = 0; k < K; k++) printf(" %.0f", single[k] / 8);
    printf(" bayes %.0f fixedshare %.0f local %.0f local_fs %.0f\n", bayes / 8, fs / 8, loc / 8, locfs / 8);
    return 0;
}
