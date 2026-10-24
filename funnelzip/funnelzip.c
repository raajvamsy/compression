/* FunnelZip core: lenses -> funnel of mixers -> refine (APM) -> binary arithmetic coder.
 *
 * Every lens predicts P(next bit = 1). The funnel learns online how much to trust each lens.
 * Encoder and decoder run the identical model, so only the coder's output is stored.
 * Build: clang -O3 -shared -fPIC -o libfunnelzip.dylib funnelzip.c
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef uint8_t U8;
typedef uint32_t U32;
typedef uint64_t U64;

enum { L_MEMORY = 1, L_WORD = 2, L_ECHO = 4, L_EQUATION = 8, L_RESONATOR = 16, L_GRID = 32, L_FUNNEL = 64, L_REFINE = 128,
       L_RHYTHM = 256, L_MOTION = 512, L_PRECISE = 1024, L_STEREO = 2048, L_CODE = 4096 };

#define MAXIN 32
#define TBITS 24      /* largest context table: 2^24 counters (64 MB) */
#ifndef TB_EXTRA
#define TB_EXTRA 12   /* table follows the input (2^12 counters per byte, max 2^24): tiny inputs skip a 64 MB fill, ratio unchanged */
#endif
#define MBITS 22
#define LMS_ORDER 16
#define LPC_ORDER 32
#define LPC_WIN 4096      /* audio: wide, frequent refits matter */
#define LPC_EVERY 256
#define LPC_WIN_PX 2048   /* pixels: the grid lens carries most of the load, refit less */
#define LPC_EVERY_PX 1024

/* ---------------------------------------------------------------- squash / stretch (12-bit) */
static int squash_t[4096];
static int rtab[1024];   /* rtab[n] = 2/(2n+3) in Q24: counter learning rate without a division */
static short stretch_t[4096];
static int ready = 0;

static void init_tables(void) {
    for (int i = 0; i < 4096; i++) {
        int v = (int)(4096.0 / (1.0 + exp(-(i - 2048) / 256.0)));
        squash_t[i] = v < 1 ? 1 : v > 4095 ? 4095 : v;
    }
    int pi = 0;
    for (int x = -2047; x <= 2047; x++) {
        int v = squash_t[x + 2048];
        for (int j = pi; j <= v && j < 4096; j++) stretch_t[j] = (short)x;
        pi = v + 1;
    }
    for (int j = pi; j < 4096; j++) stretch_t[j] = 2047;
    for (int i = 0; i < 1024; i++) rtab[i] = (int)((2LL << 24) / (2 * i + 3));
    ready = 1;
}
static inline int squash(int d) { return squash_t[(d > 2047 ? 2047 : d < -2047 ? -2047 : d) + 2048]; }
static inline int stretch(int p) { return stretch_t[p]; }

/* fast e^x: 2^(x*log2 e) = 2^int * 2^frac, frac by a degree-6 polynomial (relative error ~1e-9).
 * Deterministic for a given binary, so encoder and decoder agree. */
static inline double fexp(double x) {
    if (x < -700.0) return 0.0;
    if (x > 700.0) x = 700.0;
    double t = x * 1.4426950408889634, fi = floor(t), f = t - fi;
    double p = 1.0 + f * (0.6931471805599453 + f * (0.2402265069591007 + f * (0.0555041086648216 +
               f * (0.0096181291076285 + f * (0.0013333558146428 + f * 0.0001540353039338)))));
    union { double d; U64 u; } v;
    v.u = (U64)((long long)fi + 1023) << 52;
    return p * v.d;
}

static inline U32 h32(U32 x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
static inline U32 hc(U32 a, U32 b) { return h32(a * 0x9E3779B1U + b + 0x632BE5ABU); }

/* ---------------------------------------------------------------- adaptive counter: p16<<16 | n */
static inline int cnt_p12(U32 v) { return (int)(v >> 20); }
static inline void cnt_upd(U32 *s, int y, int limit) {
    U32 v = *s;
    int p = (int)(v >> 16), n = (int)(v & 0xffff);
    int q = (int)(((long long)(y ? 65535 - p : p) * rtab[n < 1023 ? n : 1023]) >> 24);
    p += y ? q : -q;   /* truncate toward zero, like the division it replaces (branch-free) */
    if (n < limit) n++;
    *s = ((U32)p << 16) | (U32)n;
}

/* ---------------------------------------------------------------- mixer */
typedef struct { float *w; int n, sets; float lr; int sel; const float *x; int pr; float dotf; } Mixer;

static void mix_init(Mixer *m, int n, int sets, float lr, float w0) {
    m->n = n; m->sets = sets; m->lr = lr; m->sel = 0;
    m->w = (float *)malloc(sizeof(float) * n * sets);
    for (int i = 0; i < n * sets; i++) m->w[i] = w0;
}
/* x = inputs already scaled to logits (shared by all mixers); 4 independent lanes so the CPU can vectorise */
static inline int mix_p(Mixer *m, const float *restrict x, int nin, int sel) {
    m->sel = sel;
    m->x = x;
    const float *restrict w = m->w + sel * m->n;
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    int i = 0;
    for (; i + 4 <= nin; i += 4) { d0 += w[i] * x[i]; d1 += w[i + 1] * x[i + 1]; d2 += w[i + 2] * x[i + 2]; d3 += w[i + 3] * x[i + 3]; }
    for (; i < nin; i++) d0 += w[i] * x[i];
    float dot = (d0 + d1) + (d2 + d3);
    m->dotf = dot;
    int d = (int)(dot * 256.0f);
    m->pr = squash(d);
    return d > 2047 ? 2047 : d < -2047 ? -2047 : d;   /* stretched output, for the next layer */
}
static inline void mix_upd(Mixer *m, int nin, int y) {
    float e0 = (float)y - m->pr / 4096.0f;
    if (e0 < 0.0025f && e0 > -0.0025f) return;   /* confident and right: skip (fx2-cmix); stops weight drift, ~2% smaller */
    float err = e0 * m->lr;
    float *restrict w = m->w + m->sel * m->n;
    const float *restrict x = m->x;
    for (int i = 0; i < nin; i++) w[i] += err * x[i];
}

/* ---------------------------------------------------------------- APM (refine) */
typedef struct { unsigned short *t; int idx; } APM;
static void apm_init(APM *a, int ctxs) {
    a->t = (unsigned short *)malloc(sizeof(unsigned short) * ctxs * 33);
    for (int c = 0; c < ctxs; c++)
        for (int j = 0; j < 33; j++) a->t[c * 33 + j] = (unsigned short)(squash((j - 16) * 128) * 16);
    a->idx = 0;
}
static inline int apm_p(APM *a, int pr, int cx) {
    int s = stretch(pr) + 2048, lo = s >> 7, w = s & 127;
    a->idx = cx * 33 + lo + (w >> 6);
    int p = (a->t[cx * 33 + lo] * (128 - w) + a->t[cx * 33 + lo + 1] * w) >> 11;
    return p < 1 ? 1 : p > 4095 ? 4095 : p;
}
static inline void apm_upd(APM *a, int y) {
    int g = (y << 16) + (y << 7) - y - y;
    a->t[a->idx] += (g - a->t[a->idx]) >> 7;
}

/* ---------------------------------------------------------------- model */
typedef struct {
    int W, C, S, mask;
    const U8 *buf; size_t pos;   /* bytes known so far (encoder: input, decoder: output) */
    int c0, bitpos; U64 c8;
    U32 *T;                      /* shared hashed table for all context lenses */
    U32 base[MAXIN], nbase[MAXIN]; int nctx; U32 tmask, mmask;   /* per-byte context bases; per-nibble blocks (16 slots = 1 cache line) */
    U32 *slot[MAXIN]; int limit[MAXIN]; int st[MAXIN]; int nin;
    /* word lens */
    U32 word, pword; size_t line_start, prev_line_start, prev_line_len;
    /* echo lens */
    U32 *mt; size_t ptr; int len; U32 echoT[64]; U32 *echo_slot;
    /* sample lenses */
    int *samp; size_t nsamp; float lw[4][LMS_ORDER]; float lx[4][8];   /* lx: cross-channel taps */ float err_eq, err_lms, err_grid;
    int pred_eq, pred_lms; int exp_eq, exp_lms; int ec_lms;
    double lpc[4][LPC_ORDER]; float *resid; int pred_lpc;   /* backward-adaptive resonator */
    int pred_grid, grid_ok; double fpred_eq, fpred_lms;
    /* rhythm lens: repeats of repeats (byte level) + repeated equation (sample level) */
    size_t R, rh_D, last_d; int d_conf, rh_agree, rh_exp; U32 rhT[16 * 2]; U32 *rh_slot;
    size_t Ps; float g[3]; float *efin; double fpred_rh; float err_rh;
    U32 resT[16 * 2 * 2]; U32 *res_slot;
    /* funnel + refine */
    Mixer m[3], fin; APM a1, a2;
    float xs[MAXIN], xf[3];   /* funnel inputs as logits, shared by the mixers */
    int sig;
    int pr_mix, pr;
    float pw[2], La, Lf; double pf; int p16;   /* precise final stage */
    struct { int A, B, mid, active; double cA, cB, cM; } sc[8];   /* surprise lenses: cumulative masses reused bit to bit */
    /* motion lens (video): per-block motion estimated from the two previous frames */
    size_t skip; int mo_ok, mo_w, mo_h, mo_bw, mo_bh; signed char *mvx, *mvy;
    int mo_pred, mo_mcg, mo_agree, mo_hist[4]; float err_mp, err_mc; U32 moT[16 * 2]; U32 *mo_slot;
} Model;

static void model_init(Model *M, const U8 *buf, size_t n, int W, int C, int S, int mask, size_t R) {
    memset(M, 0, sizeof(*M));
    M->R = R;
    M->Ps = (R > 0 && W > 0 && R % (size_t)W == 0) ? R / (size_t)W : 0;
    if (M->Ps < 2) M->Ps = 0;
    M->W = W; M->C = C < 1 ? 1 : C; M->S = S; M->mask = mask; M->buf = buf;
    M->c0 = 1;
    int nb = 0;
    while (nb < 40 && ((size_t)1 << nb) < n) nb++;
    int tb = nb + TB_EXTRA < 16 ? 16 : nb + TB_EXTRA > TBITS ? TBITS : nb + TB_EXTRA;
    int mb = nb + 1 < 12 ? 12 : nb + 1 > MBITS ? MBITS : nb + 1;
    M->tmask = ((1U << tb) - 1) & ~15U; M->mmask = (1U << mb) - 1;
    M->T = (U32 *)malloc(sizeof(U32) << tb);
    for (size_t i = 0; i < ((size_t)1 << tb); i++) M->T[i] = 32768U << 16;
    M->mt = (U32 *)calloc((size_t)1 << mb, sizeof(U32));
    for (int i = 0; i < 64; i++) M->echoT[i] = 32768U << 16;
    for (int i = 0; i < 64; i++) M->resT[i] = 32768U << 16;
    M->samp = (int *)calloc(n / (W > 0 ? W : 1) + 2, sizeof(int));
    M->resid = (float *)calloc(n / (W > 0 ? W : 1) + 2, sizeof(float));
    M->err_grid = 16;
    M->efin = (float *)calloc(n / (W > 0 ? W : 1) + 2, sizeof(float));
    for (int i = 0; i < 32; i++) M->rhT[i] = 32768U << 16;
    M->err_rh = 64;
    M->pw[0] = 1.0f; M->pw[1] = 0.0f;
    for (int i = 0; i < 32; i++) M->moT[i] = 32768U << 16;
    M->err_mp = M->err_mc = 16;
    if ((mask & L_MOTION) && R > 0 && S > 0 && W == 1 && R % (size_t)S == 0 && S % M->C == 0) {
        M->mo_ok = 1;
        M->mo_w = S / M->C; M->mo_h = (int)(R / (size_t)S);
        M->mo_bw = (M->mo_w + 15) / 16; M->mo_bh = (M->mo_h + 15) / 16;
        M->mvx = (signed char *)calloc((size_t)M->mo_bw * M->mo_bh, 1);
        M->mvy = (signed char *)calloc((size_t)M->mo_bw * M->mo_bh, 1);
    }
    for (int c = 0; c < 4; c++) { M->lw[c][0] = 1.0f; }  /* start as "next = previous" */
    M->err_eq = M->err_lms = 64;
    mix_init(&M->m[0], MAXIN, 256, 0.02f, 0.3f);
    mix_init(&M->m[1], MAXIN, 4 * 8 * 2 * 2, 0.02f, 0.3f);
    mix_init(&M->m[2], MAXIN, 2 * 16 * 8, 0.02f, 0.3f);
    mix_init(&M->fin, 3, 8, 0.005f, 0.33f);
    apm_init(&M->a1, 256);
    apm_init(&M->a2, 1 << 16);
}
static void model_free(Model *M) {
    free(M->T); free(M->mt); free(M->samp); free(M->resid); free(M->efin); free(M->mvx); free(M->mvy);
    for (int i = 0; i < 3; i++) free(M->m[i].w);
    free(M->fin.w); free(M->a1.t); free(M->a2.t);
}

static inline int byte_at(const Model *M, long i) { return i >= 0 && (size_t)i < M->pos ? M->buf[i] : 0; }
static inline int ilog2(int x) { int r = 0; while (x > 1 && r < 15) { x >>= 1; r++; } return r; }
static inline int floordiv256(int v) { return v >= 0 ? v >> 8 : -((-v + 255) >> 8); }

/* ---- surprise lens: P(next bit = 1) if the value is (prediction mu) + Laplace noise of scale b */
static inline double lap_cdf(double x, double mu, double b) {
    return x < mu ? 0.5 * fexp((x - mu) / b) : 1.0 - 0.5 * fexp(-(x - mu) / b);
}
static inline double lap_pmf(double v, double mu, double b) { return fexp(-fabs(v - mu) / b); }

/* cumulative mass of values whose current byte is < x (W=1, or the low byte of a 16-bit sample) */
static double byte_cum(const Model *M, int x, double mu, double b) {
    if (M->W == 1) return lap_cdf(x - 0.5, mu, b);
    if (b > 512) return (double)x;
    double m = 0;
    long k0 = (long)floor((mu - 8 * b) / 256.0) - 1, k1 = (long)floor((mu + 8 * b) / 256.0) + 1;
    for (long k = k0; k <= k1; k++) m += lap_cdf(x - 0.5 + 256.0 * k, mu, b);   /* constant offsets cancel in differences */
    return m;
}
/* high byte of a 16-bit sample, low byte known: mass of hb in [A,mid) and [mid,B) in one pass */
static void high_masses(const Model *M, int A, int mid, int B, double mu, double b, double *lo, double *hi) {
    int lob = M->buf[M->pos - 1];
    double c = (mu - lob) / 256.0, span = 8 * b / 256.0 + 2;
    *lo = *hi = 0;
    int h0 = (int)floor(c - span), h1 = (int)ceil(c + span);
    for (int h = h0; h <= h1; h++) {
        if (h < -128 || h > 127) continue;
        int hb = h & 255;
        if (hb < A || hb >= B) continue;
        double m = lap_pmf(256.0 * h + lob, mu, b);
        if (hb < mid) *lo += m; else *hi += m;
    }
}
static int surprise_st(Model *M, int slot, double mu, float err) {
    double b = err < 0.3f ? 0.3 : err;
    int bp = M->bitpos, pre = M->c0 - (1 << bp);
    int A = pre << (8 - bp), mid = (pre * 2 + 1) << (7 - bp), B = (pre + 1) << (8 - bp);
    double hi, lo;
    M->sc[slot].active = 0;
    if (M->W == 2 && M->pos % 2 == 1) {
        high_masses(M, A, mid, B, mu, b, &lo, &hi);
    } else {
        /* mu and b are fixed for the whole byte: the bounds A, B were already computed at the previous bit */
        if (bp == 0 || M->sc[slot].A != A || M->sc[slot].B != B) {
            M->sc[slot].A = A; M->sc[slot].B = B;
            M->sc[slot].cA = byte_cum(M, A, mu, b); M->sc[slot].cB = byte_cum(M, B, mu, b);
        }
        double cM = byte_cum(M, mid, mu, b);
        M->sc[slot].mid = mid; M->sc[slot].cM = cM; M->sc[slot].active = 1;
        lo = cM - M->sc[slot].cA; hi = M->sc[slot].cB - cM;
    }
    double all = hi + lo;
    if (!(all > 1e-300)) return 0;
    int p = (int)(4096.0 * hi / all);
    return stretch(p < 1 ? 1 : p > 4095 ? 4095 : p);
}

/* backward-adaptive LPC: refit from already-decoded samples (Levinson-Durbin), no side info */
static void lpc_refit(Model *M, size_t upto, int ch) {
    int C = M->C;
    double r[LPC_ORDER + 1] = {0};
    long start = (long)upto - (long)(M->W == 2 ? LPC_WIN : LPC_WIN_PX) * C;
    if (start < 0) start = (long)ch;
    while (start % C != ch) start++;
    for (long i = start; i < (long)upto; i += C) {
        double xi = M->samp[i];
        for (int k = 0; k <= LPC_ORDER; k++) {
            long j = i - (long)k * C;
            if (j < start) break;
            r[k] += xi * M->samp[j];
        }
    }
    if (r[0] <= 0) return;
    r[0] *= 1.0001;
    double a[LPC_ORDER + 1] = {0}, tmp[LPC_ORDER + 1], e = r[0];
    for (int i = 1; i <= LPC_ORDER; i++) {
        double acc = r[i];
        for (int j = 1; j < i; j++) acc -= a[j] * r[i - j];
        double k = acc / e;
        memcpy(tmp, a, sizeof(a));
        a[i] = k;
        for (int j = 1; j < i; j++) a[j] = tmp[j] - k * tmp[i - j];
        e *= (1 - k * k);
        if (e <= 0) break;
    }
    for (int k = 0; k < LPC_ORDER; k++) M->lpc[ch & 3][k] = a[k + 1];
}

/* expected value of the next byte from a sample prediction, given the byte position inside the sample */
static int expected_byte(const Model *M, int pred) {
    if (M->W == 1) return pred < 0 ? 0 : pred > 255 ? 255 : pred;
    int bps = (int)(M->pos % 2);
    if (bps == 0) return pred & 255;
    int lo = M->buf[M->pos - 1];
    return floordiv256(pred - lo + 128) & 255;
}

/* ---- motion lens: estimate block motion from frame f-2 -> f-1 (both already known) */
static void motion_estimate(Model *M, size_t f) {
    const U8 *a = M->buf + M->skip + (f - 1) * M->R, *b = M->buf + M->skip + (f - 2) * M->R;  /* a = newer */
    int C = M->C, S = M->S, ch = C > 1 ? 1 : 0, w = M->mo_w, h = M->mo_h;
    for (int by = 0; by < M->mo_bh; by++)
        for (int bx = 0; bx < M->mo_bw; bx++) {
            long best = -1; int bdx = 0, bdy = 0;
            for (int dy = -7; dy <= 7; dy++)
                for (int dx = -7; dx <= 7; dx++) {
                    long sad = 0; int cnt = 0;
                    for (int y = by * 16; y < by * 16 + 16 && y < h; y += 2)
                        for (int x = bx * 16; x < bx * 16 + 16 && x < w; x += 2) {
                            int sx = x - dx, sy = y - dy;
                            if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
                            sad += abs((int)a[(size_t)y * S + (size_t)x * C + ch] - (int)b[(size_t)sy * S + (size_t)sx * C + ch]);
                            cnt++;
                        }
                    if (cnt == 0) continue;
                    long score = sad * 64 / cnt + 2 * (abs(dx) + abs(dy));   /* prefer small motion on ties */
                    if (best < 0 || score < best) { best = score; bdx = dx; bdy = dy; }
                }
            M->mvx[by * M->mo_bw + bx] = (signed char)bdx;
            M->mvy[by * M->mo_bw + bx] = (signed char)bdy;
        }
}

static void motion_predict(Model *M) {
    size_t rel = M->pos - M->skip, f = rel / M->R;
    if (f == 0) { M->mo_pred = -1; return; }
    if (rel % M->R == 0) {
        if (f >= 2) motion_estimate(M, f);
        else { memset(M->mvx, 0, (size_t)M->mo_bw * M->mo_bh); memset(M->mvy, 0, (size_t)M->mo_bw * M->mo_bh); }
    }
    size_t o = rel - f * M->R;
    int y = (int)(o / (size_t)M->S), rem = (int)(o % (size_t)M->S), x = rem / M->C, c = rem % M->C;
    int bi = (y / 16) * M->mo_bw + (x / 16);
    int sx = x - M->mvx[bi], sy = y - M->mvy[bi];
    sx = sx < 0 ? 0 : sx >= M->mo_w ? M->mo_w - 1 : sx;
    sy = sy < 0 ? 0 : sy >= M->mo_h ? M->mo_h - 1 : sy;
    int mp = M->buf[M->skip + (f - 1) * M->R + (size_t)sy * M->S + (size_t)sx * M->C + c];
    int mcg = mp;
    if (x > 0) {   /* motion-compensated gradient: the left neighbour's miss carries over */
        mcg = mp + (int)M->buf[M->pos - M->C] - M->mo_hist[c & 3];
        mcg = mcg < 0 ? 0 : mcg > 255 ? 255 : mcg;
    }
    M->mo_pred = mp; M->mo_mcg = mcg;
    M->mo_hist[c & 3] = mp;
}

/* called once per byte boundary: compute every lens's context for the coming byte */
static void byte_contexts(Model *M) {
    int k = 0;
    U64 h = M->c8;
    if (M->mask & L_MEMORY) {
        static const int ord[6] = {0, 1, 2, 3, 4, 6};
        for (int i = 0; i < 6; i++) {
            U64 ctx = ord[i] == 0 ? 0 : (h & ((ord[i] == 8 ? ~0ULL : ((1ULL << (8 * ord[i])) - 1))));
            M->base[k++] = hc((U32)(ctx >> 32) ^ (U32)ctx * 31U, 0x100 + ord[i]);
        }
    }
    if (M->mask & L_WORD) {
        M->base[k++] = hc(M->word, M->pword + 0x200);
        size_t col = M->pos - M->line_start;
        int above = col < M->prev_line_len ? M->buf[M->prev_line_start + col] : 0;
        M->base[k++] = hc((U32)above | (U32)(col < 255 ? col : 255) << 8, 0x300);
    }
    if (M->mask & L_ECHO) {
        int pb = M->len > 0 ? M->buf[M->ptr] : 0;
        M->base[k++] = hc((U32)pb | (U32)(M->len < 15 ? M->len : 15) << 8, 0x400);
    }
    int ch = (int)((M->pos / M->W) % M->C);
    (void)ch;
    if (M->mask & L_EQUATION) {
        M->exp_eq = expected_byte(M, M->pred_eq);
        M->base[k++] = hc((U32)M->exp_eq | (U32)(M->pos % M->W) << 8, 0x500);
        M->base[k++] = hc((U32)M->exp_eq | (U32)ilog2((int)M->err_eq) << 8 | (U32)(M->pos % M->W) << 12, 0x501);
    }
    if (M->mask & L_RESONATOR) {
        M->exp_lms = expected_byte(M, M->pred_lms);
        M->ec_lms = ilog2((int)M->err_lms);
        M->base[k++] = hc((U32)M->exp_lms | (U32)M->ec_lms << 8 | (U32)(M->pos % M->W) << 12, 0x600);
    }
    M->grid_ok = 0;
    if ((M->mask & L_GRID) && M->S > 0) {
        long p = (long)M->pos;
        int up = byte_at(M, p - M->S), left = byte_at(M, p - M->C), ul = byte_at(M, p - M->S - M->C);
        int grad = left + up - ul; grad = grad < 0 ? 0 : grad > 255 ? 255 : grad;
        int mx = left > up ? left : up, mn = left < up ? left : up;
        M->pred_grid = ul >= mx ? mn : ul <= mn ? mx : left + up - ul;   /* MED predictor (JPEG-LS) */
        M->grid_ok = M->W == 1;
        M->base[k++] = hc((U32)up, 0x700);
        M->base[k++] = hc((U32)up | (U32)left << 8, 0x701);
        M->base[k++] = hc((U32)grad | (U32)((up - left) / 16 + 16) << 8, 0x702);
    }
    if (M->mask & L_RHYTHM) {
        size_t D = M->R > 0 ? M->R : (M->d_conf >= 1 ? M->last_d : 0);
        M->rh_D = (D > 0 && M->pos >= D) ? D : 0;
        if (M->rh_D) {
            int rb = M->buf[M->pos - M->rh_D], prev = M->pos ? M->buf[M->pos - 1] : 0;
            M->rh_exp = rb;
            M->base[k++] = hc((U32)rb | (U32)(M->rh_agree > 7 ? 7 : M->rh_agree) << 8, 0x800);
            M->base[k++] = hc((U32)rb | (U32)prev << 8, 0x801);
        } else {
            M->rh_exp = -1;
            M->base[k++] = hc(0, 0x802);
            M->base[k++] = hc(0, 0x803);
        }
    }
    if (M->mask & L_CODE) {   /* instruction words: position inside the 4-byte word + the word before */
        long p = (long)M->pos;
        U32 q = (U32)(p & 3);
        M->base[k++] = hc((U32)byte_at(M, p - 4) | q << 8, 0xA00);
        M->base[k++] = hc((U32)byte_at(M, p - 4) | (U32)byte_at(M, p - 8) << 8 | q << 16, 0xA01);
    }
    if (M->mask & L_MOTION) {
        M->mo_pred = -1;
        if (M->mo_ok) motion_predict(M);
        if (M->mo_pred >= 0) {
            M->base[k++] = hc((U32)M->mo_pred | (U32)(M->mo_agree > 7 ? 7 : M->mo_agree) << 8, 0x900);
            M->base[k++] = hc((U32)M->mo_pred | (U32)M->mo_mcg << 8, 0x901);
        } else {
            M->base[k++] = hc(0, 0x902);
            M->base[k++] = hc(0, 0x903);
        }
    }
    M->nctx = k;
    for (int i = 0; i < k; i++) {
        M->base[i] &= M->tmask;
        __builtin_prefetch(&M->T[M->base[i]], 1, 3);   /* fetch all contexts' cache lines at once, not one by one */
    }
}

static int model_p(Model *M) {
    int k = 0, c0 = M->c0;
    if (M->bitpos == 4)   /* second half of the byte: one new cache-line block per context, keyed by the high nibble */
        for (int i = 0; i < M->nctx; i++) {
            M->nbase[i] = h32(M->base[i] + (U32)c0) & M->tmask;
            __builtin_prefetch(&M->T[M->nbase[i]], 1, 3);
        }
    int part = M->bitpos < 4 ? c0 : ((1 << (M->bitpos - 4)) | (c0 & ((1 << (M->bitpos - 4)) - 1)));
    for (int i = 0; i < M->nctx; i++) {
        U32 *s = &M->T[(M->bitpos < 4 ? M->base[i] : M->nbase[i]) | (U32)part];
        M->slot[k] = s; M->limit[k] = 255;
        M->st[k++] = stretch(cnt_p12(*s));
    }
    M->echo_slot = NULL;
    if (M->mask & L_ECHO) {
        int in = 0;
        if (M->len > 0) {
            int pb = M->buf[M->ptr];
            if (((pb + 256) >> (8 - M->bitpos)) == c0) {
                int e = (pb >> (7 - M->bitpos)) & 1;
                M->echo_slot = &M->echoT[(M->len < 31 ? M->len : 31) * 2 + e];
                in = stretch(cnt_p12(*M->echo_slot));
            }
        }
        M->st[k++] = in;
    }
    M->res_slot = NULL;
    if (M->mask & L_RESONATOR) {
        int in = 0, eb = M->exp_lms;
        if (((eb + 256) >> (8 - M->bitpos)) == c0) {
            int e = (eb >> (7 - M->bitpos)) & 1;
            M->res_slot = &M->resT[(M->ec_lms * 2 + (int)(M->pos % 2)) * 2 + e];
            in = stretch(cnt_p12(*M->res_slot));
        }
        M->st[k++] = in;
    }
    M->rh_slot = NULL;
    if (M->mask & L_RHYTHM) {
        int in = 0, eb = M->rh_exp;
        if (eb >= 0 && ((eb + 256) >> (8 - M->bitpos)) == c0) {
            int e = (eb >> (7 - M->bitpos)) & 1;
            M->rh_slot = &M->rhT[(M->rh_agree > 15 ? 15 : M->rh_agree) * 2 + e];
            in = stretch(cnt_p12(*M->rh_slot));
        }
        M->st[k++] = in;
    }
    M->mo_slot = NULL;
    if (M->mask & L_MOTION) {
        int in = 0, eb = M->mo_pred;
        if (eb >= 0 && ((eb + 256) >> (8 - M->bitpos)) == c0) {
            int e = (eb >> (7 - M->bitpos)) & 1;
            M->mo_slot = &M->moT[(M->mo_agree > 15 ? 15 : M->mo_agree) * 2 + e];
            in = stretch(cnt_p12(*M->mo_slot));
        }
        M->st[k++] = in;
    }
    if (M->mask & L_EQUATION) M->st[k++] = surprise_st(M, 0, M->fpred_eq, M->err_eq);
    if (M->mask & L_RESONATOR) M->st[k++] = surprise_st(M, 1, M->fpred_lms, M->err_lms);
    if ((M->mask & L_GRID) && M->grid_ok) M->st[k++] = surprise_st(M, 2, M->pred_grid, M->err_grid);
    else if (M->mask & L_GRID) M->st[k++] = 0;
    if (M->mask & L_RHYTHM) M->st[k++] = M->Ps ? surprise_st(M, 3, M->fpred_rh, M->err_rh) : 0;
    if (M->mask & L_MOTION) {
        M->st[k++] = M->mo_pred >= 0 ? surprise_st(M, 4, M->mo_pred, M->err_mp) : 0;
        M->st[k++] = M->mo_pred >= 0 ? surprise_st(M, 5, M->mo_mcg, M->err_mc) : 0;
    }
    M->st[k++] = 256; /* bias */
    M->nin = k;

    int lenb = M->len == 0 ? 0 : M->len < 16 ? 1 : M->len < 32 ? 2 : 3;
    for (int i = 0; i < k; i++) M->xs[i] = M->st[i] * (1.0f / 256.0f);
    int s0 = mix_p(&M->m[0], M->xs, k, c0);
    int pr;
    if (M->mask & L_FUNNEL) {
        int lock = M->rh_agree >= 8 ? 1 : 0;   /* rhythm locked: trust the rhythm lens */
        int s1 = mix_p(&M->m[1], M->xs, k, ((lenb * 8 + M->bitpos) * 2 + (int)(M->pos % M->W % 2)) * 2 + lock);
        /* the third mixer chooses weights by resonator error / sample position: only useful when a signal lens runs */
        M->sig = (M->mask & (L_RESONATOR | L_EQUATION)) != 0;
        int s2 = M->sig ? mix_p(&M->m[2], M->xs, k, ((int)(M->pos % 2) * 16 + M->ec_lms) * 8 + M->bitpos) : s1;
        M->xf[0] = s0 * (1.0f / 256.0f); M->xf[1] = s1 * (1.0f / 256.0f); M->xf[2] = s2 * (1.0f / 256.0f);
        mix_p(&M->fin, M->xf, 3, M->bitpos);
        pr = M->fin.pr;
    } else {
        pr = M->m[0].pr;
    }
    M->pr_mix = pr;
    if (M->mask & L_REFINE) {
        int c1 = (int)(M->c8 & 255), c2 = (int)((M->c8 >> 8) & 255);
        pr = (apm_p(&M->a1, pr, c0) + pr * 3 + 2) >> 2;
        pr = (apm_p(&M->a2, pr, (int)(hc((U32)c1 | (U32)c2 << 8, (U32)c0) & 0xffff)) * 3 + pr + 2) >> 2;
    }
    M->pr = pr < 1 ? 1 : pr > 4095 ? 4095 : pr;
    if (M->mask & L_PRECISE) {
        /* blend the refined estimate with the funnel's raw confidence, beyond the 12-bit cap */
        M->La = stretch(M->pr) / 256.0f;
        M->Lf = (M->mask & L_FUNNEL) ? M->fin.dotf : M->m[0].dotf;
        double L = M->pw[0] * M->La + M->pw[1] * M->Lf;
        if (L > 40) L = 40; else if (L < -40) L = -40;
        M->pf = 1.0 / (1.0 + fexp(-L));
        long q = lrint(M->pf * 65536.0);
        M->p16 = (int)(q < 1 ? 1 : q > 65535 ? 65535 : q);
    } else {
        M->p16 = M->pr << 4;
    }
    return M->p16;
}

static void sample_done(Model *M) {
    size_t si = M->pos / M->W - 1;  /* index of the sample just completed */
    int v;
    if (M->W == 1) v = M->buf[M->pos - 1];
    else v = (int)(short)(M->buf[M->pos - 2] | M->buf[M->pos - 1] << 8);
    int C = M->C, ch = (int)(si % C);
    float e1 = (float)(v - M->fpred_eq), e2 = (float)(v - M->fpred_lms);
    M->err_eq = M->err_eq * 0.9f + fabsf(e1) * 0.1f;
    M->err_lms = M->err_lms * 0.9f + fabsf(e2) * 0.1f;
    /* stage 2 (NLMS) learns to predict what the resonator (LPC) misses */
    float r_now = (float)(v - M->pred_lpc);
    {
        float *w = M->lw[ch & 3], energy = 1.0f, x[LMS_ORDER];
        for (int k = 0; k < LMS_ORDER; k++) {
            long j = (long)si - (long)C * (k + 1);
            x[k] = j >= 0 ? M->resid[j] : 0.0f;
            energy += x[k] * x[k];
        }
        float xc[8] = {0}, *wx = M->lx[ch & 3];
        if (C == 2 && (M->mask & L_STEREO)) {   /* stereo: the other channel's leftover now and up to 7 steps earlier */
            for (int k = 0; k < 8; k++) {
                long j = (long)si - 1 - 2 * k;
                xc[k] = j >= 0 ? M->resid[j] : 0.0f;
                energy += xc[k] * xc[k];
            }
        }
        float pred_r = 0;
        for (int k = 0; k < 8; k++) pred_r += wx[k] * xc[k];
        for (int k = 0; k < LMS_ORDER; k++) pred_r += w[k] * x[k];
        float mu = 0.002f * (r_now - pred_r) / energy;
        for (int k = 0; k < LMS_ORDER; k++) w[k] += mu * x[k];
        for (int k = 0; k < 8; k++) wx[k] += mu * xc[k] * 4;
    }
    M->samp[si] = v;
    M->resid[si] = r_now;
    if (M->Ps) {   /* repeated equation: learn how the leftover of one period predicts the next */
        float ef = (float)(v - M->fpred_lms);
        M->err_rh = M->err_rh * 0.9f + fabsf((float)(v - M->fpred_rh)) * 0.1f;
        if (si >= M->Ps + 1) {
            float x[3], en = 1.0f, pr = 0;
            for (int k = 0; k < 3; k++) { x[k] = M->efin[si - M->Ps + k - 1]; en += x[k] * x[k]; pr += M->g[k] * x[k]; }
            float mu = 0.05f * (ef - pr) / en;
            for (int k = 0; k < 3; k++) M->g[k] += mu * x[k];
        }
        M->efin[si] = ef;
    }
    size_t every = M->W == 2 ? LPC_EVERY : LPC_EVERY_PX;
    if ((si + 1) % every == 0 || si + 1 == (size_t)(LPC_ORDER * C + 1))
        for (int c = 0; c < C && c < 4; c++) lpc_refit(M, si + 1, c);

    /* predict the next sample */
    size_t nx = si + 1;
    int nch = (int)(nx % C);
    long a = (long)nx - C, b = (long)nx - 2 * C;
    int s1 = a >= 0 ? M->samp[a] : 0, s2 = b >= 0 ? M->samp[b] : 0;
    M->pred_eq = 2 * s1 - s2;
    M->fpred_eq = M->pred_eq;
    double acc = 0;
    for (int k = 0; k < LPC_ORDER; k++) {
        long j = (long)nx - (long)C * (k + 1);
        if (j >= 0) acc += M->lpc[nch & 3][k] * M->samp[j];
    }
    if (nx <= (size_t)(LPC_ORDER * C)) acc = s1;
    float pr2 = 0, *w = M->lw[nch & 3];
    for (int k = 0; k < LMS_ORDER; k++) {
        long j = (long)nx - (long)C * (k + 1);
        if (j >= 0) pr2 += w[k] * M->resid[j];
    }
    if (C == 2 && (M->mask & L_STEREO)) {
        for (int k = 0; k < 8; k++) {
            long j = (long)nx - 1 - 2 * k;
            if (j >= 0) pr2 += M->lx[nch & 3][k] * M->resid[j];
        }
    }
    int lim = M->W == 1 ? 255 : 32767, lo = M->W == 1 ? 0 : -32768;
    long pl = lrint(acc);
    M->pred_lpc = (int)(pl < lo ? lo : pl > lim ? lim : pl);
    M->fpred_lms = acc + pr2;
    if (M->fpred_lms < lo) M->fpred_lms = lo;
    if (M->fpred_lms > lim) M->fpred_lms = lim;
    long pf = lrint(acc + pr2);
    M->pred_lms = (int)(pf < lo ? lo : pf > lim ? lim : pf);
    M->fpred_rh = M->fpred_lms;
    if (M->Ps && nx >= M->Ps + 1) {
        double add = 0;
        for (int k = 0; k < 3; k++) {
            size_t j = nx - M->Ps + k - 1;
            if (j < nx) add += M->g[k] * M->efin[j];
        }
        M->fpred_rh = M->fpred_lms + add;
        if (M->fpred_rh < lo) M->fpred_rh = lo;
        if (M->fpred_rh > lim) M->fpred_rh = lim;
    }
}

static void model_update(Model *M, int y) {
    for (int i = 0; i < M->nctx; i++) cnt_upd(M->slot[i], y, M->limit[i]);
    if (M->echo_slot) cnt_upd(M->echo_slot, y, 1023);
    if (M->res_slot) cnt_upd(M->res_slot, y, 1023);
    if (M->rh_slot) cnt_upd(M->rh_slot, y, 1023);
    if (M->mo_slot) cnt_upd(M->mo_slot, y, 1023);
    for (int i = 0; i < 6; i++)   /* narrow each surprise lens's cached interval to the half the bit chose */
        if (M->sc[i].active) {
            if (y) { M->sc[i].A = M->sc[i].mid; M->sc[i].cA = M->sc[i].cM; }
            else { M->sc[i].B = M->sc[i].mid; M->sc[i].cB = M->sc[i].cM; }
        }
    mix_upd(&M->m[0], M->nin, y);
    if (M->mask & L_FUNNEL) {
        mix_upd(&M->m[1], M->nin, y);
        if (M->sig) mix_upd(&M->m[2], M->nin, y);
        mix_upd(&M->fin, 3, y);
    }
    if (M->mask & L_REFINE) { apm_upd(&M->a1, y); apm_upd(&M->a2, y); }
    if (M->mask & L_PRECISE) {
        float err = ((float)y - (float)M->pf) * 0.002f;
        M->pw[0] += err * M->La; M->pw[1] += err * M->Lf;
    }

    M->c0 = (M->c0 << 1) | y;
    M->bitpos++;
    if (M->bitpos < 8) return;

    /* ---- byte complete */
    int c = M->c0 & 255;
    if (M->rh_D) M->rh_agree = (c == M->buf[M->pos - M->rh_D]) ? (M->rh_agree < 15 ? M->rh_agree + 1 : 15) : M->rh_agree >> 1;
    if ((M->mask & L_MOTION) && M->mo_pred >= 0) {
        M->mo_agree = (c == M->mo_pred) ? (M->mo_agree < 15 ? M->mo_agree + 1 : 15) : M->mo_agree >> 1;
        M->err_mp = M->err_mp * 0.9f + fabsf((float)(c - M->mo_pred)) * 0.1f;
        M->err_mc = M->err_mc * 0.9f + fabsf((float)(c - M->mo_mcg)) * 0.1f;
    }
    M->pos++;
    M->c0 = 1; M->bitpos = 0;
    M->c8 = (M->c8 << 8) | (U64)c;
    /* word lens */
    int lc = (c >= 'A' && c <= 'Z') ? c + 32 : c;
    if ((lc >= 'a' && lc <= 'z') || (c >= '0' && c <= '9') || c >= 128) M->word = hc(M->word, (U32)lc);
    else if (M->word) { M->pword = M->word; M->word = 0; }
    if (c == '\n') {
        M->prev_line_start = M->line_start;
        M->prev_line_len = M->pos - M->line_start;
        M->line_start = M->pos;
    }
    /* echo lens */
    if (M->len > 0 && M->buf[M->ptr] == c) { M->len++; M->ptr++; if (M->len > 65535) M->len = 65535; }
    else M->len = 0;
    if (M->pos >= 6) {
        U32 hh = (U32)(h32((U32)(M->c8 & 0xffffffffffffULL) ^ (U32)((M->c8 & 0xffffffffffffULL) >> 32) * 0x9E3779B1U) & M->mmask);
        if (M->len == 0) {
            U32 cand = M->mt[hh];
            if (cand > 0) {
                int l = 0;
                while (l < 64 && (size_t)l < cand && M->buf[cand - 1 - l] == M->buf[M->pos - 1 - l]) l++;
                if (l >= 6) {
                    M->len = l; M->ptr = cand;
                    size_t d = M->pos - cand;   /* how far back the repeat came from */
                    if (d == M->last_d) { if (M->d_conf < 7) M->d_conf++; }
                    else { M->last_d = d; M->d_conf = 0; }
                }
            }
        }
        M->mt[hh] = (U32)M->pos;
    }
    if (M->len > 0 && M->ptr >= M->pos) M->len = 0;
    /* sample lenses */
    if (M->grid_ok) M->err_grid = M->err_grid * 0.9f + fabsf((float)(c - M->pred_grid)) * 0.1f;
    if (M->pos % M->W == 0 && (M->mask & (L_RESONATOR | L_EQUATION))) sample_done(M);   /* signal maths only if a signal lens uses it */
    byte_contexts(M);
}

/* ---------------------------------------------------------------- coder */
size_t fz_compress(const U8 *in, size_t n, U8 *out, size_t cap, int W, int C, int S, int mask, size_t R, size_t skip) {
    if (!ready) init_tables();
    if (n == 0) return 0;
    Model *M = (Model *)malloc(sizeof(Model));
    model_init(M, in, n, W, C, S, mask, R);
    M->skip = skip;
    byte_contexts(M);
    U32 x1 = 0, x2 = 0xffffffffU;
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {
        int c = in[i];
        if (i < skip) {   /* primer: the model reads it, nothing is coded */
            for (int j = 7; j >= 0; j--) { model_p(M); model_update(M, (c >> j) & 1); }
            continue;
        }
        for (int j = 7; j >= 0; j--) {
            int y = (c >> j) & 1;
            U32 xmid = x1 + (U32)(((U64)(x2 - x1) * (U32)model_p(M)) >> 16);
            if (y) x2 = xmid; else x1 = xmid + 1;
            model_update(M, y);
            while (((x1 ^ x2) & 0xff000000U) == 0) {
                if (o < cap) out[o] = (U8)(x2 >> 24);
                o++;
                x1 <<= 8; x2 = (x2 << 8) | 255;
            }
        }
    }
    if (o < cap) out[o] = (U8)(x1 >> 24);
    o++;
    model_free(M); free(M);
    return o;
}

int fz_decompress(const U8 *in, size_t len, U8 *out, size_t n, int W, int C, int S, int mask, size_t R, size_t skip) {
    if (!ready) init_tables();
    if (n == 0) return 0;
    Model *M = (Model *)malloc(sizeof(Model));
    model_init(M, out, n, W, C, S, mask, R);
    M->skip = skip;
    byte_contexts(M);
    U32 x1 = 0, x2 = 0xffffffffU, x = 0;
    size_t ip = 0;
    for (int i = 0; i < 4; i++) x = (x << 8) | (ip < len ? in[ip++] : 255);
    for (size_t i = 0; i < n; i++) {
        int c = 0;
        if (i < skip) {   /* primer is already in out[] */
            for (int j = 7; j >= 0; j--) { model_p(M); model_update(M, (out[i] >> j) & 1); }
            continue;
        }
        for (int j = 0; j < 8; j++) {
            U32 xmid = x1 + (U32)(((U64)(x2 - x1) * (U32)model_p(M)) >> 16);
            int y = x <= xmid;
            if (y) x2 = xmid; else x1 = xmid + 1;
            c = (c << 1) | y;
            if (j == 7) out[i] = (U8)c;   /* the model reads the finished byte from out[] */
            model_update(M, y);
            while (((x1 ^ x2) & 0xff000000U) == 0) {
                x1 <<= 8; x2 = (x2 << 8) | 255;
                x = (x << 8) | (ip < len ? in[ip++] : 255);
            }
        }
    }
    model_free(M); free(M);
    return 0;
}

/* ---------------------------------------------------------------- trace (for the visual explainer)
 * For the first max_bits bits: every funnel input (stretched, /256 = logit) + the final P(1), per bit.
 * Returns the number of inputs per bit; rows are (nin + 2) floats: inputs..., p_final, actual bit. */
int fz_trace(const U8 *in, size_t n, int W, int C, int S, int mask, size_t R, size_t skip, float *rows, int max_bits) {
    if (!ready) init_tables();
    Model *M = (Model *)malloc(sizeof(Model));
    model_init(M, in, n, W, C, S, mask, R);
    M->skip = skip;
    byte_contexts(M);
    int nin = 0, b = 0;
    for (size_t i = 0; i < n && b < max_bits; i++) {
        for (int j = 7; j >= 0 && b < max_bits; j--, b++) {
            int y = (in[i] >> j) & 1;
            int p = model_p(M);
            nin = M->nin;
            float *r = rows + (size_t)b * (MAXIN + 2);
            for (int k = 0; k < nin; k++) r[k] = M->st[k] / 256.0f;
            r[MAXIN] = p / 65536.0f;
            r[MAXIN + 1] = (float)y;
            model_update(M, y);
        }
    }
    model_free(M); free(M);
    return nin;
}

/* ---------------------------------------------------------------- MED residual reshape (pixels)
 * forward: every byte -> (byte - MED(left, up, corner)) mod 256, using original neighbours
 * inverse: rebuild in order, so each prediction uses already-rebuilt neighbours. S = row bytes, C = channel step. */
static inline int med3(int l, int u, int ul) {
    int mx = l > u ? l : u, mn = l < u ? l : u;
    return ul >= mx ? mn : ul <= mn ? mx : l + u - ul;
}
void fz_med_forward(U8 *buf, size_t n, int S, int C) {
    for (size_t k = n; k-- > 0;) {   /* back to front: neighbours are still original */
        size_t row = k / (size_t)S, col = k % (size_t)S;
        int l = col >= (size_t)C ? buf[k - C] : 0, u = row ? buf[k - S] : 0;
        int ul = (row && col >= (size_t)C) ? buf[k - S - C] : 0;
        buf[k] = (U8)((buf[k] - med3(l, u, ul)) & 255);
    }
}
void fz_med_inverse(U8 *buf, size_t n, int S, int C) {
    for (size_t k = 0; k < n; k++) {
        size_t row = k / (size_t)S, col = k % (size_t)S;
        int l = col >= (size_t)C ? buf[k - C] : 0, u = row ? buf[k - S] : 0;
        int ul = (row && col >= (size_t)C) ? buf[k - S - C] : 0;
        buf[k] = (U8)((buf[k] + med3(l, u, ul)) & 255);
    }
}

/* ---------------------------------------------------------------- JPEG Huffman layer (baseline, restart markers)
 * Decodes a scan straight into FunnelZip's coefficient-plane layout (per component: 64 planes x bh x bw, int16),
 * and re-encodes planes into the identical scan bytes. Tables: per component 16 code counts + 256 symbols. */
typedef struct { int mincode[17], maxcode[18], valptr[17]; U8 val[256]; int ehufco[256], ehufsi[256]; } HuffT;
static void huff_build(HuffT *t, const U8 *counts, const U8 *syms) {
    int code = 0, k = 0;
    memset(t->ehufsi, 0, sizeof(t->ehufsi));
    for (int l = 1; l <= 16; l++) {
        t->valptr[l] = k; t->mincode[l] = code;
        for (int i = 0; i < counts[l - 1]; i++) {
            t->val[k] = syms[k];
            t->ehufco[syms[k]] = code; t->ehufsi[syms[k]] = l;
            k++; code++;
        }
        t->maxcode[l] = counts[l - 1] ? code - 1 : -1;
        code <<= 1;
    }
    t->maxcode[17] = 0x7fffffff;
}
typedef struct { const U8 *p; size_t n, pos; U32 acc; int bits; } BitIn;
static inline int bin_bit(BitIn *b) {
    if (b->bits == 0) {
        U8 c = b->pos < b->n ? b->p[b->pos++] : 0xFF;
        if (c == 0xFF && b->pos < b->n && b->p[b->pos] == 0x00) b->pos++;   /* stuffed 0xFF00 */
        b->acc = c; b->bits = 8;
    }
    b->bits--;
    return (b->acc >> b->bits) & 1;
}
static inline int huff_dec(BitIn *b, const HuffT *t) {
    int code = 0;
    for (int l = 1; l <= 16; l++) {
        code = (code << 1) | bin_bit(b);
        if (t->maxcode[l] >= 0 && code <= t->maxcode[l]) return t->val[t->valptr[l] + code - t->mincode[l]];
    }
    return -1;
}
static inline int recv_val(BitIn *b, int size) {
    if (!size) return 0;
    int v = 0;
    for (int i = 0; i < size; i++) v = (v << 1) | bin_bit(b);
    return v >= (1 << (size - 1)) ? v : v - (1 << size) + 1;
}
/* block scan order: for each MCU, for each component, its h x v blocks */
typedef struct { int ncomp, single, mcux, mcuy, restart; int h[4], v[4], bw[4], bh[4]; size_t off[4]; } JSpec;
#define FOR_BLOCKS(S, ...)                                                                              \
    {                                                                                                     \
        long mcu_i = 0, nmcu = (S)->single ? (long)(S)->bw[0] * (S)->bh[0] : (long)(S)->mcux * (S)->mcuy; \
        for (long m = 0; m < nmcu; m++, mcu_i++) {                                                        \
            int restart_now = (S)->restart && m > 0 && m % (S)->restart == 0;                              \
            for (int ci = 0; ci < (S)->ncomp; ci++) {                                                     \
                int hh = (S)->single ? 1 : (S)->h[ci], vv = (S)->single ? 1 : (S)->v[ci];                \
                for (int by = 0; by < vv; by++)                                                           \
                    for (int bx = 0; bx < hh; bx++) {                                                     \
                        int X = (S)->single ? (int)(m % (S)->bw[0]) : (int)(m % (S)->mcux) * hh + bx;     \
                        int Y = (S)->single ? (int)(m / (S)->bw[0]) : (int)(m / (S)->mcux) * vv + by;     \
                        __VA_ARGS__                                                                       \
                        restart_now = 0;                                                                  \
                    }                                                                                     \
            }                                                                                             \
        }                                                                                                 \
    }
static void jspec(JSpec *S, int ncomp, int single, int mcux, int mcuy, int restart, const int *h, const int *v,
                  const int *bw, const int *bh) {
    S->ncomp = ncomp; S->single = single; S->mcux = mcux; S->mcuy = mcuy; S->restart = restart;
    size_t off = 0;
    for (int c = 0; c < ncomp; c++) { S->h[c] = h[c]; S->v[c] = v[c]; S->bw[c] = bw[c]; S->bh[c] = bh[c]; S->off[c] = off; off += (size_t)64 * bw[c] * bh[c]; }
}
static const int ZZ_EOB = 0x00, ZZ_ZRL = 0xF0;
int fz_jpeg_decode(const U8 *scan, size_t n, int ncomp, int single, int mcux, int mcuy, int restart, const int *h, const int *v,
                   const int *bw, const int *bh, const U8 *dcc, const U8 *dcs, const U8 *acc, const U8 *acs, short *planes) {
    JSpec S; jspec(&S, ncomp, single, mcux, mcuy, restart, h, v, bw, bh);
    HuffT dc[4], ac[4];
    for (int c = 0; c < ncomp; c++) { huff_build(&dc[c], dcc + 16 * c, dcs + 256 * c); huff_build(&ac[c], acc + 16 * c, acs + 256 * c); }
    BitIn b = {scan, n, 0, 0, 0};
    int pred[4] = {0, 0, 0, 0};
    long seen = -1;
    int err = 0;
    FOR_BLOCKS(&S, {
        if (restart_now && ci == 0 && by == 0 && bx == 0 && seen != m) {   /* byte-align, skip RSTn, reset DC */
            b.bits = 0;
            if (b.pos + 1 < n && scan[b.pos] == 0xFF && scan[b.pos + 1] >= 0xD0 && scan[b.pos + 1] <= 0xD7) b.pos += 2;
            else err = -2;
            pred[0] = pred[1] = pred[2] = pred[3] = 0;
            seen = m;
        }
        size_t plane = (size_t)S.bw[ci] * S.bh[ci], at = S.off[ci] + (size_t)Y * S.bw[ci] + X;
        int s = huff_dec(&b, &dc[ci]);
        if (s < 0) err = -1;
        pred[ci] += recv_val(&b, s & 15);
        planes[at] = (short)pred[ci];
        for (int k = 1; k < 64;) {
            int rs = huff_dec(&b, &ac[ci]);
            if (rs < 0) { err = -1; break; }
            if (rs == ZZ_EOB) break;
            if (rs == ZZ_ZRL) { k += 16; continue; }
            k += rs >> 4;
            if (k > 63) { err = -1; break; }
            planes[at + (size_t)k * plane] = (short)recv_val(&b, rs & 15);
            k++;
        }
    });
    return err;
}
typedef struct { U8 *p; size_t cap, pos; U32 acc; int bits; } BitOut;
static inline void bout_byte(BitOut *o, U8 c) {
    if (o->pos < o->cap) o->p[o->pos] = c;
    o->pos++;
    if (c == 0xFF) { if (o->pos < o->cap) o->p[o->pos] = 0; o->pos++; }
}
static inline void bout_put(BitOut *o, U32 code, int len) {
    for (int i = len - 1; i >= 0; i--) {
        o->acc = (o->acc << 1) | ((code >> i) & 1);
        if (++o->bits == 8) { bout_byte(o, (U8)o->acc); o->acc = 0; o->bits = 0; }
    }
}
static inline void bout_pad(BitOut *o) { while (o->bits) bout_put(o, 1, 1); }
static inline void put_value(BitOut *o, const HuffT *t, int sym_extra, int v, int run, int dc) {
    int a = v < 0 ? -v : v, size = 0;
    while (a) { size++; a >>= 1; }
    int sym = dc ? size : (run << 4) | size;
    bout_put(o, (U32)t->ehufco[sym], t->ehufsi[sym]);
    (void)sym_extra;
    if (size) bout_put(o, (U32)(v > 0 ? v : v + (1 << size) - 1), size);
}
size_t fz_jpeg_encode(const short *planes, int ncomp, int single, int mcux, int mcuy, int restart, const int *h, const int *v,
                      const int *bw, const int *bh, const U8 *dcc, const U8 *dcs, const U8 *acc, const U8 *acs, U8 *out, size_t cap) {
    JSpec S; jspec(&S, ncomp, single, mcux, mcuy, restart, h, v, bw, bh);
    HuffT dc[4], ac[4];
    for (int c = 0; c < ncomp; c++) { huff_build(&dc[c], dcc + 16 * c, dcs + 256 * c); huff_build(&ac[c], acc + 16 * c, acs + 256 * c); }
    BitOut o = {out, cap, 0, 0, 0};
    int pred[4] = {0, 0, 0, 0}, rst = 0;
    long seen = -1;
    FOR_BLOCKS(&S, {
        if (restart_now && ci == 0 && by == 0 && bx == 0 && seen != m) {
            bout_pad(&o);
            if (o.pos + 1 < cap) { out[o.pos] = 0xFF; out[o.pos + 1] = (U8)(0xD0 + (rst & 7)); }
            o.pos += 2; rst++;
            pred[0] = pred[1] = pred[2] = pred[3] = 0;
            seen = m;
        }
        size_t plane = (size_t)S.bw[ci] * S.bh[ci], at = S.off[ci] + (size_t)Y * S.bw[ci] + X;
        int dcv = planes[at];
        put_value(&o, &dc[ci], 0, dcv - pred[ci], 0, 1);
        pred[ci] = dcv;
        int last = 0;
        for (int k = 63; k >= 1; k--) if (planes[at + (size_t)k * plane]) { last = k; break; }
        int run = 0;
        for (int k = 1; k <= last; k++) {
            int val = planes[at + (size_t)k * plane];
            if (!val) { run++; continue; }
            while (run > 15) { bout_put(&o, (U32)ac[ci].ehufco[ZZ_ZRL], ac[ci].ehufsi[ZZ_ZRL]); run -= 16; }
            put_value(&o, &ac[ci], 0, val, run, 0);
            run = 0;
        }
        if (last < 63) bout_put(&o, (U32)ac[ci].ehufco[ZZ_EOB], ac[ci].ehufsi[ZZ_EOB]);
    });
    bout_pad(&o);
    return o.pos;
}
