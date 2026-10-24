/* FFIC: FunnelField Image Coder, research prototype. Measures IDEAL code length (-log2 p); no bitstream yet.
 *
 * FunnelBrain mechanisms and where they live here (see docs/IDEAS_FUNNELBRAIN.md):
 *   transduction      causal-neighbourhood descriptor: activity (scale 1), level, SPIN = gradient orientation
 *   funnel            context class owning a predictor (recursive least squares), bias, exposure count (depth), skill (perf)
 *   nesting           funnels form a path root -> activity -> +spin -> +level; a child INHERITS its parent's weights AND
 *                     covariance at birth, so a new funnel never starts from nothing (this is what makes splitting affordable)
 *   parallel balls    NSUB sub-predictors predict at once; the "global workspace" blends them by recent error nearby
 *   workspace         every funnel on the path predicts; weights = 1 / skill^2, so a child gains influence only by performing better
 *   mouth             residual scale from local error energy (surprise around this pixel) x activity
 *   prediction error  weights, bias, skill and mouth adapt every pixel; the learning rate floors out (fatigue / saturation)
 *
 * usage: ffic file.rgb W H [flags]  letters: l learn  t spin  v level  p nesting  c cross  e energy-mouth  w blend  r RLS
 *        "none" = bare MED + activity context.   mu=0.01 (NLMS)  K=...  lam=0.998 (RLS forgetting)  P0=50 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define NSUB 5                    /* parallel predictors (balls) */
#define NNB 12                    /* neighbour taps (causal window) */
#define CK 4                      /* neighbours used for cross-plane taps (W N NW NE) */
#define NF 4                      /* non-causal neighbours of an already-decoded plane: E S SE SW (a decoder has the WHOLE earlier plane) */
#define NT (NNB + 2 * (CK + 1 + NF))  /* + cross-plane taps: per earlier plane, CK causal + NF future colour-gradient taps + 1 residual tap */
#define NA 8                      /* activity levels */
#define NTH 9                     /* spin states: 8 orientations + flat */
#define NL 4                      /* brightness levels */
#define NEN 10                    /* local error-energy levels */
#define NCE 6                     /* cross-plane surprise levels */
#define MAXLV 4

typedef struct { float w[NT]; float bias; int n; float perf; double P[NT * NT]; } Funnel;
typedef struct { float scale; int n; } Mouth;

static int W, H;
static int opt_l = 1, opt_t = 1, opt_v = 1, opt_p = 1, opt_c = 1, opt_e = 1, opt_w = 1, opt_r = 1, opt_b = 1;
static float MU = 0.01f, LAM = 0.998f, P0 = 50.f, GATE = 0.f; static int RLMASK = 15;
static const int ATH[NA - 1] = {3, 6, 12, 24, 48, 96, 192};
static const int TAPX[NNB] = {-1, 0, -1, 1, -2, 0, 1, -1, -2, 2, -3, 2};
static const int TAPY[NNB] = {0, -1, -1, -1, 0, -2, -2, -2, -1, -1, 0, -2};
static const int FUTX[NF] = {1, 0, 1, -1}, FUTY[NF] = {0, 1, 1, 1};
static int opt_f = 0;

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }
static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }
static inline int med(int a, int b, int c) { int mx = a > b ? a : b, mn = a < b ? a : b; return c >= mx ? mn : c <= mn ? mx : a + b - c; }
static inline double lapF(double t, double b) { return t < 0 ? 0.5 * exp(t / b) : 1.0 - 0.5 * exp(-t / b); }
static inline float emap(const float *m, int x, int y) { return (x < 0 || x >= W || y < 0) ? 0.f : m[(size_t)y * W + x]; }

#define NA_SEL 8

/* ---- residual coder: adaptive binary decisions, each predicted by mixing counters from several funnel contexts ---- */
#define NCTX 6
#define NIN (NCTX + 2)           /* mixer inputs: counters + parametric prior + bias */
#define SLOTS 19                  /* 0 zero?  1 sign  2..9 unary bucket  10..17 first mantissa bit per bucket  18 other mantissa bits */
typedef struct { float p; unsigned short n; } Ctr;
typedef struct { Ctr *t[NCTX]; int size[NCTX]; float *mw; double bits; } CM;
static float LRM = 0.004f; static int CLIM = 255;
static inline float stretchf(float p) { return logf(p / (1.f - p)); }
static inline float squashf(float x) { return 1.f / (1.f + expf(-x)); }
static void cm_init(CM *m, const int *sizes, int nsel) {
    for (int i = 0; i < NCTX; i++) { m->size[i] = sizes[i]; m->t[i] = malloc((size_t)sizes[i] * SLOTS * sizeof(Ctr)); for (size_t j = 0; j < (size_t)sizes[i] * SLOTS; j++) { m->t[i][j].p = 0.5f; m->t[i][j].n = 0; } }
    m->mw = malloc((size_t)SLOTS * nsel * NIN * sizeof(float));
    for (size_t j = 0; j < (size_t)SLOTS * nsel * NIN; j++) m->mw[j] = (j % NIN) == NCTX + 1 ? 0.f : (j % NIN) == NCTX ? 0.25f : 0.25f;
    m->bits = 0;
}
/* code one binary decision (bit) in decision slot `slot`, contexts cx[], mixer selector sel, parametric prior ppar; accumulates ideal cost */
static inline void cm_bit(CM *m, int slot, const int *cx, int sel, int bit, float ppar) {
    float st[NIN]; Ctr *c[NCTX];
    float *w = m->mw + ((size_t)slot * NA_SEL + sel) * NIN;
    float dot = 0;
    for (int i = 0; i < NCTX; i++) { c[i] = &m->t[i][(size_t)cx[i] * SLOTS + slot]; float p = c[i]->p; p = p < 0.001f ? 0.001f : p > 0.999f ? 0.999f : p; st[i] = stretchf(p); dot += w[i] * st[i]; }
    ppar = ppar < 0.001f ? 0.001f : ppar > 0.999f ? 0.999f : ppar;
    st[NCTX] = stretchf(ppar); dot += w[NCTX] * st[NCTX];
    st[NCTX + 1] = 0.3f; dot += w[NCTX + 1] * st[NCTX + 1];
    float p = squashf(dot); p = p < 1e-4f ? 1e-4f : p > 1.f - 1e-4f ? 1.f - 1e-4f : p;
    m->bits += -log2f(bit ? p : 1.f - p);
    float err = (float)bit - p;
    for (int i = 0; i < NIN; i++) w[i] += LRM * err * st[i];
    for (int i = 0; i < NCTX; i++) { float a = 1.f / (c[i]->n + 1.6f); if (a < 1.f / CLIM) a = 1.f / CLIM; c[i]->p += ((float)bit - c[i]->p) * a; if (c[i]->n < 65000) c[i]->n++; }
}
/* Laplace mass of integer residuals e in [lo,hi], centred at d (= pc - r) with scale b */
static inline double lmass(int lo, int hi, double d, double b) { return lapF(hi + 0.5 - d, b) - lapF(lo - 0.5 - d, b); }
/* code residual e (=x-r). d,b: parametric Laplace prior. r tells which signs are possible. */
static inline void cm_residual(CM *m, const int *cx, int sel, int e, int r, double d, double b) {
    const int BIG = 100000;
    double m0 = lmass(0, 0, d, b), mneg = lmass(-BIG, -1, d, b), mpos = lmass(1, BIG, d, b);
    cm_bit(m, 0, cx, sel, e == 0, (float)(m0 / (m0 + mneg + mpos + 1e-300)));
    if (e == 0) return;
    int neg = e < 0;
    if (r > 0 && r < 255) cm_bit(m, 1, cx, sel, neg, (float)(mneg / (mneg + mpos + 1e-300)));
    int mag = abs(e) - 1, k = 0; while ((2 << k) - 1 <= mag) k++;                  /* bucket k holds m in [2^k - 1, 2^(k+1) - 2] */
    #define SIDE(a, bb) (neg ? lmass(-(bb), -(a), d, b) : lmass((a), (bb), d, b))   /* mass of |e| in [a,bb] on the chosen side */
    for (int j = 0; j < k; j++) { double hi = SIDE((2 << j), BIG), all = SIDE((1 << j), BIG); cm_bit(m, 2 + (j < 8 ? j : 7), cx, sel, 1, (float)(hi / (all + 1e-300))); }
    if (k < 8) { double hi = SIDE((2 << k), BIG), all = SIDE((1 << k), BIG); cm_bit(m, 2 + k, cx, sel, 0, (float)(hi / (all + 1e-300))); }
    int rem = mag - ((1 << k) - 1), lo = 1 << k, hi2 = (2 << k) - 1;               /* |e| in [lo,hi2] */
    for (int bi = k - 1; bi >= 0; bi--) {
        int mid = (lo + hi2 + 1) / 2, bit = (rem >> bi) & 1;
        double up = SIDE(mid, hi2), all = SIDE(lo, hi2);
        cm_bit(m, bi == k - 1 ? 10 + (k < 8 ? k : 7) : 18, cx, sel, bit, (float)(up / (all + 1e-300)));
        if (bit) lo = mid; else hi2 = mid - 1;
    }
}

static void funnel_init(Funnel *f, const Funnel *parent) {
    if (parent) { *f = *parent; f->n = 0; f->perf = parent->perf * 1.25f + 0.05f; }      /* inherit weights + covariance; start a bit less trusted */
    else {
        memset(f, 0, sizeof *f); f->perf = 8.f;
        for (int i = 0; i < NT; i++) f->P[i * NT + i] = P0;
    }
}
/* learn from one observation. na = number of active taps (cross taps are absent for the first plane). */
static inline void funnel_learn(Funnel *f, const float *u, int na, float err, float un2, int rls) {
    float a = 1.f / (f->n + 1.5f); if (a < 1.f / 64) a = 1.f / 64;           /* learning rate floors out: saturation */
    f->perf += (fabsf(err) - f->perf) * (a < 1.f / 24 ? 1.f / 24 : a);
    f->bias += a * err * 0.3f;
    if (opt_l && !(GATE > 0.f && fabsf(err) < GATE && f->n > 64)) {          /* gating: a mature funnel ignores unsurprising pixels */
        if (opt_r && rls) {                                                          /* recursive least squares on the active taps (double: float loses positive-definiteness) */
            double Pu[NT], den = LAM;
            for (int i = 0; i < na; i++) { double s = 0; for (int j = 0; j < na; j++) s += f->P[i * NT + j] * u[j]; Pu[i] = s; den += u[i] * s; }
            if (!(den > 1e-6) || !isfinite(den)) {                             /* degenerate: reset this funnel's covariance, keep its weights */
                memset(f->P, 0, sizeof f->P); for (int i = 0; i < NT; i++) f->P[i * NT + i] = P0;
            } else {
                double inv = 1.0 / den, il = 1.0 / LAM; int bad = 0;
                for (int i = 0; i < na; i++) { double w = f->w[i] + Pu[i] * inv * err; f->w[i] = (float)(w > 4 ? 4 : w < -4 ? -4 : w); }
                for (int i = 0; i < na; i++) for (int j = i; j < na; j++) {
                    double v = (f->P[i * NT + j] - Pu[i] * Pu[j] * inv) * il;
                    if (i == j && (v > 1e3 || v < 0)) bad = 1;
                    f->P[i * NT + j] = f->P[j * NT + i] = v;
                }
                if (bad) { memset(f->P, 0, sizeof f->P); for (int i = 0; i < NT; i++) f->P[i * NT + i] = P0; }
            }
        } else { float g = MU * err / un2; for (int k = 0; k < na; k++) f->w[k] += g * u[k]; }
    }
    f->n++;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ffic file.rgb W H [flags]\n"); return 1; }
    W = atoi(argv[2]); H = atoi(argv[3]);
    for (int i = 4; i < argc; i++) {
        if (!strncmp(argv[i], "mu=", 3)) MU = atof(argv[i] + 3);
        else if (!strncmp(argv[i], "lam=", 4)) LAM = atof(argv[i] + 4);
        else if (!strncmp(argv[i], "P0=", 3)) P0 = atof(argv[i] + 3);
        else if (!strncmp(argv[i], "lrm=", 4)) LRM = atof(argv[i] + 4);
        else if (!strncmp(argv[i], "gate=", 5)) GATE = atof(argv[i] + 5);
        else if (!strncmp(argv[i], "rl=", 3)) RLMASK = atoi(argv[i] + 3);
        else if (!strncmp(argv[i], "clim=", 5)) CLIM = atoi(argv[i] + 5);
        else {
            opt_l = opt_t = opt_v = opt_p = opt_c = opt_e = opt_w = opt_r = opt_b = opt_f = 0;
            if (strcmp(argv[i], "none"))
                for (char *c = argv[i]; *c; c++) { switch (*c) { case 'l': opt_l = 1; break; case 't': opt_t = 1; break; case 'v': opt_v = 1; break;
                    case 'p': opt_p = 1; break; case 'c': opt_c = 1; break; case 'e': opt_e = 1; break; case 'w': opt_w = 1; break; case 'r': opt_r = 1; break; case 'b': opt_b = 1; break; case 'f': opt_f = 1; break; } }
        }
    }
    FILE *fp = fopen(argv[1], "rb"); if (!fp) { perror("open"); return 1; }
    size_t N = (size_t)W * H; unsigned char *raw = malloc(N * 3);
    if (fread(raw, 1, N * 3, fp) != N * 3) { fprintf(stderr, "short read\n"); return 1; } fclose(fp);
    int *pl[3]; float *pfin[3];
    for (int c = 0; c < 3; c++) { pl[c] = malloc(N * sizeof(int)); pfin[c] = calloc(N, sizeof(float)); for (size_t i = 0; i < N; i++) pl[c][i] = raw[i * 3 + c]; }
    static const int order[3] = {1, 0, 2};              /* G first, then R, then B */
    const int lsize[MAXLV] = {1, NA, NA * NTH, NA * NTH * NL};
    double bits[3] = {0, 0, 0}; clock_t t0 = clock(); long long born = 0;

    for (int oi = 0; oi < 3; oi++) {
        int c = order[oi]; int *P = pl[c];
        const int nact = NNB + (opt_c ? (oi >= 1 ? CK + 1 + (opt_f ? NF : 0) : 0) + (oi >= 2 ? CK + 1 + (opt_f ? NF : 0) : 0) : 0);   /* active taps */
        Funnel *lv[MAXLV]; for (int j = 0; j < MAXLV; j++) { lv[j] = malloc(lsize[j] * sizeof(Funnel)); for (int i = 0; i < lsize[j]; i++) lv[j][i].n = -1; }
        Mouth *mouth = malloc(NA * NEN * NCE * sizeof(Mouth)); for (int i = 0; i < NA * NEN * NCE; i++) { mouth[i].scale = 6.f; mouth[i].n = 0; }
        float *esub[NSUB]; for (int k = 0; k < NSUB; k++) esub[k] = calloc(N, sizeof(float));
        float *efin = calloc(N, sizeof(float));
        CM cm; { int sizes[NCTX] = {NA * NEN * NCE, NA * NTH * NL, 625 * 3, 4 * NEN * NL * 3, 9 * 9 * 5, 6 * NEN * 3 * 3}; cm_init(&cm, sizes, NA_SEL); }
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            size_t idx = (size_t)y * W + x;
            int nbv[NNB]; size_t nbi[NNB];                                        /* causal fetch with fallbacks at the borders (indices reused for cross-plane taps) */
            for (int k = 0; k < NNB; k++) {
                int xx = x + TAPX[k], yy = y + TAPY[k]; size_t ii;
                if (yy < 0) ii = x > 0 ? idx - 1 : (y > 0 ? idx - W : idx);
                else { xx = clampi(xx, 0, W - 1); ii = (yy == y && xx >= x) ? (x > 0 ? idx - 1 : (y > 0 ? idx - W : idx)) : (size_t)yy * W + xx; }
                nbi[k] = ii; nbv[k] = (ii == idx) ? 128 : P[ii];
            }
            int Wv = nbv[0], Nv = nbv[1], NWv = nbv[2], NEv = nbv[3], WWv = nbv[4], NNv = nbv[5], NNEv = nbv[6];
            int m = med(Wv, Nv, NWv);

            /* ---- parallel balls ---- */
            float sp[NSUB];
            sp[0] = (float)m;
            sp[1] = clampf((float)(Wv + Nv - NWv), 0, 255);
            sp[2] = clampf((Wv + Nv) * 0.5f + (NEv - NWv) * 0.25f, 0, 255);
            sp[3] = clampf((float)(Wv + NEv - Nv), 0, 255);
            sp[4] = clampf(Nv + (Nv - NNv) * 0.5f + (Wv - NWv) * 0.5f, 0, 255);
            float base;
            if (opt_w) {
                float sw = 0, sv = 0;
                for (int k = 0; k < NSUB; k++) {
                    float S = emap(esub[k], x - 1, y) + emap(esub[k], x, y - 1) + emap(esub[k], x - 1, y - 1) + emap(esub[k], x + 1, y - 1) + 0.5f;
                    float wk = 1.f / (S * S); sw += wk; sv += wk * sp[k];
                }
                base = sv / sw;
            } else base = sp[0];

            /* ---- transduction: activity, level, spin ---- */
            int dh = abs(Wv - WWv) + abs(Nv - NWv) + abs(Nv - NEv), dv = abs(Wv - NWv) + abs(Nv - NNv) + abs(NEv - NNEv), act = dh + dv;
            int A = 0; while (A < NA - 1 && act > ATH[A]) A++;
            int L = opt_v ? clampi((int)base >> 6, 0, NL - 1) : 0, TH = 0;
            if (opt_t) {
                float gx = (float)((Nv - NWv) + (NEv - Nv) + (Wv - WWv)), gy = (float)((Wv - NWv) + (Nv - NNv) + (NEv - NNEv));
                if (fabsf(gx) + fabsf(gy) < 9.f) TH = 8; else { float a = atan2f(gy, gx); TH = ((int)floorf((a + 3.14159265f) / 0.78539816f)) & 7; }
            }
            /* ---- nested funnels ---- */
            int key[MAXLV] = {0, A, A * NTH + TH, (A * NTH + TH) * NL + L};
            int lvidx[MAXLV], nlv = 0;
            if (opt_p) { lvidx[nlv++] = 0; lvidx[nlv++] = 1; if (opt_t) lvidx[nlv++] = 2; if (opt_t && opt_v) lvidx[nlv++] = 3; }
            else lvidx[nlv++] = opt_t ? (opt_v ? 3 : 2) : (opt_v ? 3 : 1);          /* flat partition */
            Funnel *path[MAXLV];
            for (int j = 0; j < nlv; j++) {
                int lvl = lvidx[j], k = key[lvl];
                if (!opt_t && lvl == 3) k = A * NL + L;                              /* level without spin */
                path[j] = &lv[lvl][k % lsize[lvl]];
                if (path[j]->n < 0) { funnel_init(path[j], (j > 0 && opt_p) ? path[j - 1] : NULL); path[j]->n = 0; born++; }
            }
            float u[NT]; int nu = 0;
            for (int k = 0; k < NNB; k++) u[nu++] = (float)nbv[k] - base;
            if (opt_c && oi > 0) {                                                /* colour dimension: how the already-coded planes changed here vs their neighbours */
                for (int pi = 0; pi < oi; pi++) {
                    int *Q = pl[pi == 0 ? 1 : 0]; float *Qf = pfin[pi == 0 ? 1 : 0];
                    for (int k = 0; k < CK; k++) u[nu++] = (float)(Q[idx] - Q[nbi[k]]);
                    if (opt_f) for (int k = 0; k < NF; k++) { int fx = clampi(x + FUTX[k], 0, W - 1), fy = clampi(y + FUTY[k], 0, H - 1); u[nu++] = (float)(Q[idx] - Q[(size_t)fy * W + fx]); }
                    u[nu++] = (float)Q[idx] - Qf[idx];
                }
            }
            float un2 = 1.f; for (int k = 0; k < nact; k++) un2 += u[k] * u[k];
            float pj[MAXLV], sw2 = 0.f, sv2 = 0.f;
            for (int j = 0; j < nlv; j++) {
                float d = 0.f; for (int k = 0; k < nact; k++) d += path[j]->w[k] * u[k];
                pj[j] = base + (opt_l ? d : 0.f) + path[j]->bias;
                float pe_ = path[j]->perf + 0.05f, wj = 1.f / (pe_ * pe_);
                sw2 += wj; sv2 += wj * pj[j];
            }
            float pc = clampf(sv2 / sw2, 0, 255);
            pfin[c][idx] = pc;

            /* ---- mouth ---- */
            int EN = 0;
            if (opt_e) {
                float en = emap(efin, x - 1, y) + emap(efin, x, y - 1) + 0.5f * (emap(efin, x - 1, y - 1) + emap(efin, x + 1, y - 1)) + 0.5f * (emap(efin, x - 2, y) + emap(efin, x, y - 2));
                static const float ETH[NEN - 1] = {0.6f, 1.2f, 2.f, 3.2f, 5.f, 8.f, 13.f, 22.f, 40.f};
                while (EN < NEN - 1 && en > ETH[EN]) EN++;
            }
            int CE = 0;                                                           /* how surprised were the planes already coded, right here? */
            if (opt_e && opt_c && oi > 0) {
                float ce = fabsf((float)pl[1][idx] - pfin[1][idx]); if (oi == 2) ce += fabsf((float)pl[0][idx] - pfin[0][idx]);
                static const float CTH[NCE - 1] = {0.5f, 1.5f, 3.f, 6.f, 12.f};
                while (CE < NCE - 1 && ce > CTH[CE]) CE++;
            }
            Mouth *mo = &mouth[(A * NEN + EN) * NCE + CE];
            float sb = mo->scale < 0.25f ? 0.25f : mo->scale;

            /* ---- ideal code length ---- */
            int xv = P[idx];
            double t = xv - pc, mass = lapF(t + 0.5, sb) - lapF(t - 0.5, sb), Z = lapF(255.5 - pc, sb) - lapF(-0.5 - pc, sb);
            double pr = mass / Z; if (pr < 1e-9) pr = 1e-9; if (!opt_b) bits[c] += -log2(pr);
            if (opt_b) {
                int r = (int)lrintf(pc), e = xv - r;
                #define CL2(v) (clampi((v), -2, 2) + 2)
                int cx[NCTX];
                int fr = (int)floorf((pc - r + 0.5f) * 3.999f);                    /* which side of the integer the prediction sits */
                int cfl = (opt_c && oi > 0) ? (CE >= 3 ? 2 : CE >= 1 ? 1 : 0) : 0;
                cx[0] = (A * NEN + EN) * NCE + CE;
                cx[1] = path[nlv - 1] ? (int)((path[nlv - 1] - lv[lvidx[nlv - 1]]) % (NA * NTH * NL)) : 0;
                cx[2] = (((CL2(Wv - r) * 5 + CL2(Nv - r)) * 5 + CL2(NWv - r)) * 5 + CL2(NEv - r)) * 3 + cfl;
                cx[3] = ((fr * NEN + EN) * NL + L) * 3 + cfl;
                cx[4] = (clampi((Wv - Nv) / 4, -4, 4) + 4) * 9 * 5 + (clampi((Nv - NWv) / 4, -4, 4) + 4) * 5 + (fr > 3 ? 3 : fr);
                float sd = 0, sws = 0;                                              /* parallel balls' disagreement = how unsure the workspace is */
                for (int k = 0; k < NSUB; k++) { float dd = sp[k] - pc; sd += dd * dd; sws += 1.f; }
                sd = sqrtf(sd / sws); int SQ = sd < 0.4f ? 0 : sd < 0.9f ? 1 : sd < 1.8f ? 2 : sd < 3.5f ? 3 : sd < 7.f ? 4 : 5;
                cx[5] = ((SQ * NEN + EN) * 3 + cfl) * 3 + (TH == 8 ? 0 : (TH & 1) + 1);
                cm_residual(&cm, cx, A, e, r, (double)(pc - r), (double)sb);
            }

            /* ---- prediction error: everything adapts ---- */
            float ef = fabsf(xv - pc); efin[idx] = ef;
            for (int k = 0; k < NSUB; k++) esub[k][idx] = fabsf(xv - sp[k]);
            { float a = 1.f / (mo->n + 1.5f); if (a < 1.f / 96) a = 1.f / 96; mo->scale += a * (ef - mo->scale); mo->n++; }
            for (int j = 0; j < nlv; j++) funnel_learn(path[j], u, nact, xv - pj[j], un2, (RLMASK >> (nlv - 1 - j)) & 1);   /* bit 0 = deepest funnel */
        }
        if (opt_b) bits[c] = cm.bits;
        for (int k = 0; k < NSUB; k++) free(esub[k]);
        free(efin); for (int j = 0; j < MAXLV; j++) free(lv[j]); free(mouth);
    }
    double ms = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC, tot = bits[0] + bits[1] + bits[2];
    printf("bytes=%.0f bpp=%.4f ms=%.0f funnels=%lld R=%.0f G=%.0f B=%.0f\n", tot / 8, tot / N, ms, born, bits[0] / 8, bits[1] / 8, bits[2] / 8);
    return 0;
}
