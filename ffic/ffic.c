/* FFIC: FunnelField Image Coder, research prototype. Reports the IDEAL code length (-log2 p); with out=file it also writes a real
 * arithmetic-coded bitstream, and `ffic -d file out.rgb` decodes it (the decoder runs the identical model).
 *
 * FunnelBrain mechanisms and where they live here (see docs/IDEAS_FUNNELBRAIN.md, ffic/experiments/LOG.md):
 *   transduction      causal-neighbourhood descriptor: activity (scale 1), level, SPIN = gradient orientation
 *   funnel            context class owning a predictor (recursive least squares), bias, exposure count (depth), skill (perf)
 *   nesting           funnels form a path root -> activity -> +spin -> +level; a child INHERITS its parent's weights AND
 *                     covariance at birth, so a new funnel never starts from nothing
 *   parallel balls    NSUB sub-predictors predict at once; the "global workspace" blends them by recent error nearby
 *   workspace         every funnel on the path predicts; weights = 1 / skill^ws, so a child gains influence only by performing better
 *   mouth             residual scale from local error energy (surprise around this pixel) x activity
 *   prediction error  weights, bias, skill and mouth adapt every pixel; the learning rate floors out (fatigue / saturation)
 * Experimental FunnelBrain physics (all off by default; key=value options):
 *   field=1           continuous FUNNEL FIELD: funnels are points in a multi-scale descriptor space, created on demand when a
 *                     pixel lands outside every mouth (leader clustering, as in funnel/field.py find_or_create_funnel);
 *                     prediction blends the kres nearest funnels by RESONANCE (distance to mouth x depth pull x fatigue)
 *   sleep=N           every N rows: FISSION (split a busy funnel along its widest descriptor axis) + PRUNE (free idle funnels)
 *   bounce=1          BOUNCE PATH: the funnels where the W and N pixels landed become a coder context (association = path)
 *   hippo=1           HIPPOCAMPUS: episodic table keyed by the quantised neighbourhood shape; recall = an extra ball + tap + context
 *   lampe=1           online PE reshapes geometry: a surprised funnel forgets faster (adaptive RLS forgetting)
 *   sal=+1|-1         AMYGDALA salience: amplify (+1) or damp (-1, robust) the learning weight of surprising pixels
 *                     weight = clamp(surprise, 1, salmax)^salp (or its inverse); salbias=k clips the bias update at k x skill;
 *                     salall=k clips outliers at k x usual error in the skill and mouth estimators too
 *   ws=T              global-workspace temperature: weights 1/skill^T (T=0 => winner takes all)
 *   replay=1          SLEEP REPLAY: re-learn each finished row once more (decoder can do the same)
 *   ms=1              MULTI-SCALE transduction: wide-window activity as an extra coder context (and field dimension)
 *   grav=g            GRAVITY / depth pull: workspace weight x (1 + g * log(1 + exposures)), experienced funnels attract
 *   register=1        REGISTERS: R and B grid funnels are born from G's trained funnel at the same address (SubRegister under Main)
 *   apm=1             ACC (anterior cingulate, computes PE): a secondary stage corrects each coder probability by how wrong
 *                     such probabilities usually are in this situation (error energy x ball disagreement)   apmw= apmr=
 *   mix2=1            second coder workspace selected by surprise level, blended with the first
 *   mline=1|2         COLOUR PHYSICS, material lines: under one material and light, shading moves a pixel along a line in
 *                     colour space (dichromatic model + camera offset): R ~ a*G + b. The line is fitted on the mwin decoded
 *                     neighbours (ridge mlam toward slope 1) and gives an extra ball + tap for R and B; 2 = B also from R
 *   matf=1            MATERIAL FUNNELS: R and B pixels land in funnels in colour space (neighbourhood chromaticity, and for B the
 *                     pixel's own R:G balance); each funnel learns its material line over the whole image (R ~ a*G + b,
 *                     B ~ a*G + c*R + b, decayed by mdec, ridge matlam toward the difference model), giving a ball + tap.
 *                     mrad = mouth, mmax = cap, mbw = weight of brightness in the landing, mctx=1 = material id as coder context
 *   gmass=1           GRAVITY: each funnel (field landing, else activity class) keeps a mass histogram of the values that
 *                     landed in it; the prediction's inverse-square-weighted centre of mass within +-gmr becomes an RLS tap
 *                     (gsig = softening length, glim = mass at which the histogram halves = forgetting)
 *   ilace=1           TIME AS SCALE: code even rows first (a coarser picture, its own funnels and contexts), then odd rows,
 *                     which also see the decoded row BELOW (3 extra taps + a vertical-interpolation ball)
 *   pipe=1            STAGGERED COLOUR PIPELINE: G, R and B run on three threads at once; R waits until G is lag rows ahead,
 *                     B until R is. Deterministic: a layer only reads finished rows; registers copy G's funnels frozen every snapk rows
 *   inter=1           PIXEL BY PIXEL order (G, R, B of each pixel before the next); G gets 4 colour taps from W and N
 *   minf=1            with cmatch: the match question also hears how likely the G layer already finds a candidate's G
 *   mexcl=1           with cmatch: on "not a match", B excludes the B value of every candidate whose G and R equal this pixel's
 *   cmatch=1          WHOLE-COLOUR MATCH (needs inter=1): one joint question per pixel, "same colour as which neighbour?"
 *   FFIC_PIXCOST=path  write each pixel's code length (float32, planes R,G,B) for the parallel-worlds analysis
 *   curve=1           SPIN AS CURVATURE: an extra ball that continues the local edge with its curvature (helical, not straight)
 *
 * usage: ffic file.rgb W H [flags] [key=value...]   letters: l learn t spin v level p nesting c cross e energy w blend r RLS b CM f future
 *        "none" = bare MED + activity context. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>

#define NSUB 5                    /* fixed parallel predictors (balls); +1 when hippo is on */
#define NSUBX (NSUB + 6)
#define NNB 12                    /* neighbour taps (causal window) */
#define CK 4                      /* neighbours used for cross-plane taps (W N NW NE) */
#define NF 4                      /* non-causal neighbours of an already-decoded plane: E S SE SW */
#define NX 12                     /* extra feature taps (hippocampal recall, material lines, material funnel, gravity, 3 'below' taps, 4 colour taps for G) */
#define NT (NNB + 2 * (CK + 1 + NF) + NX)
#define NA 8                      /* activity levels */
#define NTH 9                     /* spin states: 8 orientations + flat */
#define NL 4                      /* brightness levels */
#define NEN 10                    /* local error-energy levels */
#define NCE 6                     /* cross-plane surprise levels */
#define MAXLV 4
#define DIM 7                     /* funnel-field descriptor dimensions */
#define FMAXC 512
#define KMAX 4
#define HBITS 18

typedef struct {
    float w[NT]; float bias; int n; float perf; double P[NT * NT];
    float c[DIM], mouth, s1[DIM], s2[DIM]; int nsl, idle;   /* field funnels only: centre, mouth, landing stats since last sleep */
} Funnel;
typedef struct { float scale; int n; } Mouth;

static int W, H;
static int opt_l = 1, opt_t = 1, opt_v = 1, opt_p = 1, opt_c = 1, opt_e = 1, opt_w = 1, opt_r = 1, opt_b = 1, opt_f = 0;
static float MU = 0.01f, LAM = 0.998f, P0 = 50.f, GATE = 0.f; static int RLMASK = 15;
static int FIELD = 0, SLEEP = 0, BOUNCE = 0, HIPPO = 0, LAMPE = 0, REPLAY = 0, MS = 0, KRES = 3, FMAX = 192;
static int MLINE = 0, MWIN = 6; static float MLAM = 20.f;
/* material funnels: points in colour space (brightness ignored), each remembering its own material line over the whole image */
#define MMAXC 256
typedef struct { float c[3]; int n; double S[9], St[3]; } Mat;
static int MATF = 0, MMAX = 64, MCTX = 0; static float MRAD = 0.02f, MDEC = 0.999f, MATLAM = 30.f, MBW = 0.f;
static _Thread_local long long st_mat_born;
/* gravity: values the image keeps landing on act as masses; a prediction is pulled toward nearby heavy values */
static int ILACE = 0; static _Thread_local int RS = 1;   /* TIME as scale: even rows first (the coarse past), then odd rows, which also see the row below */
static int GMASS = 0, GMR = 8; static float GSIG = 2.f, GLIM = 4096.f;
/* solve (S + ridge) w = St + ridge*w0 for k regressors (+ intercept), k <= 3 */
static void mat_solve(const Mat *m, int k, const double *w0, double *w) {
    double A[3][4];
    for (int i = 0; i < k; i++) { for (int j = 0; j < k; j++) A[i][j] = m->S[i * 3 + j]; A[i][k] = m->St[i]; }
    for (int i = 0; i < k; i++) { double l = i == 0 ? 1e-3 : MATLAM; A[i][i] += l; A[i][k] += l * w0[i]; }
    for (int i = 0; i < k; i++) {                                              /* Gaussian elimination with partial pivoting */
        int p = i; for (int r = i + 1; r < k; r++) if (fabs(A[r][i]) > fabs(A[p][i])) p = r;
        if (p != i) for (int c = 0; c <= k; c++) { double t = A[i][c]; A[i][c] = A[p][c]; A[p][c] = t; }
        for (int r = 0; r < k; r++) if (r != i) { double f = A[r][i] / A[i][i]; for (int c = i; c <= k; c++) A[r][c] -= f * A[i][c]; }
    }
    for (int i = 0; i < k; i++) w[i] = A[i][k] / A[i][i];
}
static float SAL = 0.f, WST = 2.f, MERGE = 0.12f, GRAV = 0.f, SALP = 1.f, SALMAX = 4.f, SALBIAS = 0.f, SALALL = 0.f; static int REGISTER = 0, CURVE = 0;
static const int ATH[NA - 1] = {3, 6, 12, 24, 48, 96, 192};
static const int TAPX[NNB] = {-1, 0, -1, 1, -2, 0, 1, -1, -2, 2, -3, 2};
static const int TAPY[NNB] = {0, -1, -1, -1, 0, -2, -2, -2, -1, -1, 0, -2};
static const int FUTX[NF] = {1, 0, 1, -1}, FUTY[NF] = {0, 1, 1, 1};
static const int WOFX[8] = {-4, -6, 0, 0, -4, 4, -3, 3}, WOFY[8] = {0, 0, -4, -6, -4, -4, -2, -2};
static _Thread_local long long st_fission, st_prune, st_hippo_hit, st_hippo_q, st_field_born, st_replay;

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }
static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }
static inline int med(int a, int b, int c) { int mx = a > b ? a : b, mn = a < b ? a : b; return c >= mx ? mn : c <= mn ? mx : a + b - c; }
static inline double lapF(double t, double b) { return t < 0 ? 0.5 * exp(t / b) : 1.0 - 0.5 * exp(-t / b); }
static inline float emap(const float *m, int x, int y) { return (x < 0 || x >= W || y < 0) ? 0.f : m[(size_t)y * W + x]; }

#define NA_SEL 8

/* ---- residual coder: adaptive binary decisions, each predicted by mixing counters from several funnel contexts ---- */
#define NCX 10                    /* max contexts: 6 core + bounce + hippo + multi-scale + material */
#define NIN (NCX + 2)             /* mixer inputs: counters + parametric prior + bias */
#define SLOTS 19                  /* 0 zero?  1 sign  2..9 unary bucket  10..17 first mantissa bit per bucket  18 other mantissa bits */
typedef struct { float p; unsigned short n; } Ctr;
#define NSEL2 (NEN * 3)
#define NAPM (NEN * 3)
typedef struct { Ctr *t[NCX]; int size[NCX]; int ncx; float *mw, *mw2, *apm; double bits; } CM;
static float LRM = 0.004f; static int CLIM = 255; static int APM = 0, MIX2 = 0; static float APMW = 0.5f, APMR = 0.02f;
static inline float stretchf(float p) { return logf(p / (1.f - p)); }
static inline float squashf(float x) { return 1.f / (1.f + expf(-x)); }
static void cm_init(CM *m, const int *sizes, int ncx, int nsel) {
    m->ncx = ncx;
    for (int i = 0; i < ncx; i++) { m->size[i] = sizes[i]; m->t[i] = malloc((size_t)sizes[i] * SLOTS * sizeof(Ctr)); for (size_t j = 0; j < (size_t)sizes[i] * SLOTS; j++) { m->t[i][j].p = 0.5f; m->t[i][j].n = 0; } }
    m->mw = malloc((size_t)SLOTS * nsel * NIN * sizeof(float));
    for (size_t j = 0; j < (size_t)SLOTS * nsel * NIN; j++) m->mw[j] = (j % NIN) == NCX + 1 ? 0.f : 0.25f;
    m->mw2 = malloc((size_t)SLOTS * NSEL2 * NIN * sizeof(float));
    for (size_t j = 0; j < (size_t)SLOTS * NSEL2 * NIN; j++) m->mw2[j] = (j % NIN) == NCX + 1 ? 0.f : 0.25f;
    m->apm = malloc((size_t)SLOTS * NAPM * 33 * sizeof(float));              /* ACC: maps (situation, probability) -> corrected probability */
    for (size_t j = 0; j < (size_t)SLOTS * NAPM; j++) for (int b = 0; b < 33; b++) m->apm[j * 33 + b] = squashf((b - 16) * 0.5f);
    m->bits = 0;
}
static void cm_free(CM *m) { for (int i = 0; i < m->ncx; i++) free(m->t[i]); free(m->mw); free(m->mw2); free(m->apm); }
/* ---- binary arithmetic coder (carry-less, 32-bit range, 16-bit probabilities); one stream per colour layer ---- */
typedef struct { unsigned x1, x2, x; unsigned char *buf; size_t n, cap, pos; } AC;
static _Thread_local AC *g_ac;
static _Thread_local int g_acmode;                   /* 0 measure ideal length only, 1 encode, 2 decode, -1 learn only (no bits) */
static _Thread_local float g_pover = -1.f;           /* coding probability override (exclusion), model still learns normally */
static inline void ac_put(AC *a, unsigned char c) { if (a->n == a->cap) { a->cap = a->cap ? a->cap * 2 : 1 << 16; a->buf = realloc(a->buf, a->cap); } a->buf[a->n++] = c; }
static inline unsigned char ac_get(AC *a) { return a->pos < a->n ? a->buf[a->pos++] : 0; }
static void ac_start_dec(AC *a) { a->x1 = 0; a->x2 = 0xffffffffu; a->x = 0; for (int i = 0; i < 4; i++) a->x = (a->x << 8) | ac_get(a); }
static void ac_flush(AC *a) { for (int i = 0; i < 4; i++) { ac_put(a, a->x1 >> 24); a->x1 <<= 8; } }
/* code/decode one bit whose probability of being 1 is p1; returns the bit */
static inline int ac_bit(float p1, int bit) {
    if (g_acmode <= 0) return bit;
    AC *a = g_ac; unsigned p16 = (unsigned)(p1 * 65536.f); p16 = p16 < 1 ? 1 : p16 > 65535 ? 65535 : p16;
    unsigned xmid = a->x1 + (unsigned)(((unsigned long long)(a->x2 - a->x1) * p16) >> 16);
    if (g_acmode == 2) bit = a->x <= xmid;
    if (bit) a->x2 = xmid; else a->x1 = xmid + 1;
    while (((a->x1 ^ a->x2) & 0xff000000u) == 0) {
        if (g_acmode == 1) ac_put(a, a->x2 >> 24); else a->x = (a->x << 8) | ac_get(a);
        a->x1 <<= 8; a->x2 = (a->x2 << 8) | 255;
    }
    return bit;
}
/* code one binary decision (bit) in decision slot `slot`, contexts cx[], mixer selectors sel (+ sel2), ACC context actx,
   parametric prior ppar; accumulates ideal cost */
static _Thread_local int g_sel2, g_actx;
static inline int cm_bit(CM *m, int slot, const int *cx, int sel, int bit, float ppar) {
    float st[NIN]; Ctr *c[NCX]; const int nc = m->ncx;
    float *w = m->mw + ((size_t)slot * NA_SEL + sel) * NIN;
    float dot = 0;
    for (int i = 0; i < nc; i++) { c[i] = &m->t[i][(size_t)cx[i] * SLOTS + slot]; float p = c[i]->p; p = p < 0.001f ? 0.001f : p > 0.999f ? 0.999f : p; st[i] = stretchf(p); dot += w[i] * st[i]; }
    ppar = ppar < 0.001f ? 0.001f : ppar > 0.999f ? 0.999f : ppar;
    st[NCX] = stretchf(ppar); dot += w[NCX] * st[NCX];
    st[NCX + 1] = 0.3f; dot += w[NCX + 1] * st[NCX + 1];
    float *w2 = NULL, dot2 = 0;
    if (MIX2) {                                                               /* second workspace, selected by surprise level */
        w2 = m->mw2 + ((size_t)slot * NSEL2 + g_sel2) * NIN;
        for (int i = 0; i < nc; i++) dot2 += w2[i] * st[i];
        dot2 += w2[NCX] * st[NCX] + w2[NCX + 1] * st[NCX + 1];
    }
    float p1 = squashf(dot), p = MIX2 ? squashf(0.5f * (dot + dot2)) : p1;
    p = p < 1e-4f ? 1e-4f : p > 1.f - 1e-4f ? 1.f - 1e-4f : p;
    float pf = p; float *ap = NULL; int ab = 0; float af = 0;
    if (APM) {                                                                /* ACC: correct the probability by how wrong it usually is here */
        float sx = clampf(stretchf(p), -7.999f, 7.999f) * 2.f + 16.f; ab = (int)sx; af = sx - ab;
        ap = m->apm + ((size_t)slot * NAPM + g_actx) * 33;
        float pa = ap[ab] * (1.f - af) + ap[ab + 1] * af;
        pf = (1.f - APMW) * p + APMW * pa; pf = pf < 1e-4f ? 1e-4f : pf > 1.f - 1e-4f ? 1.f - 1e-4f : pf;
    }
    if (g_acmode >= 0) { float pc_ = g_pover >= 0.f ? g_pover : pf; bit = ac_bit(pc_, bit); m->bits += -log2f(bit ? pc_ : 1.f - pc_); }
    float err = (float)bit - p1;
    for (int i = 0; i < nc; i++) w[i] += LRM * err * st[i];
    w[NCX] += LRM * err * st[NCX]; w[NCX + 1] += LRM * err * st[NCX + 1];
    if (MIX2) { float e2 = (float)bit - squashf(dot2); for (int i = 0; i < nc; i++) w2[i] += LRM * e2 * st[i]; w2[NCX] += LRM * e2 * st[NCX]; w2[NCX + 1] += LRM * e2 * st[NCX + 1]; }
    if (APM) { ap[ab] += ((float)bit - ap[ab]) * APMR * (1.f - af); ap[ab + 1] += ((float)bit - ap[ab + 1]) * APMR * af; }
    for (int i = 0; i < nc; i++) { float a = 1.f / (c[i]->n + 1.6f); if (a < 1.f / CLIM) a = 1.f / CLIM; c[i]->p += ((float)bit - c[i]->p) * a; if (c[i]->n < 65000) c[i]->n++; }
    return bit;
}
/* probability that decision `slot` is 1, exactly as cm_bit would code it, without learning (for exclusion) */
static inline float cm_bitp(CM *m, int slot, const int *cx, int sel, float ppar) {
    float st[NIN]; const int nc = m->ncx; float *w = m->mw + ((size_t)slot * NA_SEL + sel) * NIN; float dot = 0, dot2 = 0;
    for (int i = 0; i < nc; i++) { float p = m->t[i][(size_t)cx[i] * SLOTS + slot].p; p = p < 0.001f ? 0.001f : p > 0.999f ? 0.999f : p; st[i] = stretchf(p); dot += w[i] * st[i]; }
    ppar = ppar < 0.001f ? 0.001f : ppar > 0.999f ? 0.999f : ppar;
    st[NCX] = stretchf(ppar); dot += w[NCX] * st[NCX]; st[NCX + 1] = 0.3f; dot += w[NCX + 1] * st[NCX + 1];
    if (MIX2) { float *w2 = m->mw2 + ((size_t)slot * NSEL2 + g_sel2) * NIN; for (int i = 0; i < nc; i++) dot2 += w2[i] * st[i]; dot2 += w2[NCX] * st[NCX] + w2[NCX + 1] * st[NCX + 1]; }
    float p = MIX2 ? squashf(0.5f * (dot + dot2)) : squashf(dot); p = p < 1e-4f ? 1e-4f : p > 1.f - 1e-4f ? 1.f - 1e-4f : p;
    if (APM) { float sx = clampf(stretchf(p), -7.999f, 7.999f) * 2.f + 16.f; int ab = (int)sx; float af = sx - ab; float *ap = m->apm + ((size_t)slot * NAPM + g_actx) * 33;
        float pa = ap[ab] * (1.f - af) + ap[ab + 1] * af; p = (1.f - APMW) * p + APMW * pa; p = p < 1e-4f ? 1e-4f : p > 1.f - 1e-4f ? 1.f - 1e-4f : p; }
    return p;
}
/* Laplace mass of integer residuals e in [lo,hi], centred at d (= pc - r) with scale b */
static inline double lmass(int lo, int hi, double d, double b) { return lapF(hi + 0.5 - d, b) - lapF(lo - 0.5 - d, b); }

/* ---- the residual as a walk down a binary tree: zero? -> sign -> unary bucket -> mantissa (same decisions for coder,
   decoder, probability lookups and exclusion) ---- */
typedef struct { int phase, neg, j, k, lo, hi2, bi, e; } RW;      /* phase 0 zero?, 1 sign, 2 bucket, 3 mantissa, 4 done */
typedef struct { int r; double d, b, m0, mneg, mpos; } RP;
#define RBIG 100000
static inline double rside(const RP *p, int neg, int a, int bb) { return neg ? lmass(-bb, -a, p->d, p->b) : lmass(a, bb, p->d, p->b); }
static inline void rw_node(const RW *w, const RP *p, int *slot, float *ppar) {
    switch (w->phase) {
    case 0: *slot = 0; *ppar = (float)(p->m0 / (p->m0 + p->mneg + p->mpos + 1e-300)); break;
    case 1: *slot = 1; *ppar = (float)(p->mneg / (p->mneg + p->mpos + 1e-300)); break;
    case 2: { double hi = rside(p, w->neg, 2 << w->j, RBIG), all = rside(p, w->neg, 1 << w->j, RBIG); *slot = 2 + (w->j < 8 ? w->j : 7); *ppar = (float)(hi / (all + 1e-300)); break; }
    default: { int mid = (w->lo + w->hi2 + 1) / 2; double up = rside(p, w->neg, mid, w->hi2), all = rside(p, w->neg, w->lo, w->hi2);
        *slot = w->bi == w->k - 1 ? 10 + (w->k < 8 ? w->k : 7) : 18; *ppar = (float)(up / (all + 1e-300)); }
    }
}
static inline int rw_bit(const RW *w, int e) {                     /* which way residual e goes at this node */
    switch (w->phase) {
    case 0: return e == 0;
    case 1: return e < 0;
    case 2: { int mag = abs(e) - 1, k = 0; while ((2 << k) - 1 <= mag) k++; return k > w->j; }
    default: return abs(e) >= (w->lo + w->hi2 + 1) / 2;
    }
}
static inline void rw_step(RW *w, const RP *p, int bit) {
    switch (w->phase) {
    case 0: if (bit) { w->phase = 4; w->e = 0; } else if (p->r > 0 && p->r < 255) w->phase = 1; else { w->neg = p->r >= 255; w->phase = 2; w->j = 0; } break;
    case 1: w->neg = bit; w->phase = 2; w->j = 0; break;
    case 2: if (bit && w->j < 7) { w->j++; break; }
        if (bit) w->j++;                                                  /* (cannot happen for 8-bit samples) */
        w->k = w->j; w->lo = 1 << w->k; w->hi2 = (2 << w->k) - 1; w->bi = w->k - 1;
        if (w->bi < 0) { w->phase = 4; w->e = w->neg ? -1 : 1; } else w->phase = 3; break;
    default: { int mid = (w->lo + w->hi2 + 1) / 2; if (bit) w->lo = mid; else w->hi2 = mid - 1;
        if (--w->bi < 0) { w->phase = 4; w->e = w->neg ? -w->lo : w->lo; } }
    }
}
static inline RP rp_make(int r, double d, double b) { RP p = {r, d, b, lmass(0, 0, d, b), lmass(-RBIG, -1, d, b), lmass(1, RBIG, d, b)}; return p; }
/* probability of the rest of residual e's path from walker state w (model is not changed) */
static double rw_rest(CM *m, const int *cx, int sel, RW w, const RP *p, int e) {
    double P = 1;
    while (w.phase != 4) { int slot; float pp; rw_node(&w, p, &slot, &pp); int bit = rw_bit(&w, e); double q = cm_bitp(m, slot, cx, sel, pp); P *= bit ? q : 1 - q; rw_step(&w, p, bit); }
    return P;
}
/* probability the residual coder gives to residual e (learns nothing) */
static double cm_pval(CM *m, const int *cx, int sel, int e, int r, double d, double b) {
    RP p = rp_make(r, d, b); RW w = {0}; return rw_rest(m, cx, sel, w, &p, e);
}
/* code (mode 1), decode (mode 2), measure only (0) or learn only (3) residual e (=x-r). Values in ex[] are known NOT to be the
   answer ("not a match"): at every node their probability mass is removed and the decision renormalised (decoder does the same). */
static int cm_residual_x(CM *m, const int *cx, int sel, int e, int r, double d, double b, const int *ex, int nex) {
    RP p = rp_make(r, d, b); RW w = {0}; int live[4], nl = 0;
    for (int i = 0; i < nex && i < 4; i++) live[nl++] = ex[i];
    int save = g_acmode; if (save == 3) g_acmode = -1;
    while (w.phase != 4) {
        int slot; float pp; rw_node(&w, &p, &slot, &pp);
        g_pover = -1.f;
        if (nl) {                                                          /* renormalise without the excluded values */
            float pf = cm_bitp(m, slot, cx, sel, pp); double x1 = 0, x0 = 0;
            for (int i = 0; i < nl; i++) { int bx = rw_bit(&w, live[i]); RW w2 = w; rw_step(&w2, &p, bx); double rest = rw_rest(m, cx, sel, w2, &p, live[i]);
                if (bx) x1 += pf * rest; else x0 += (1 - pf) * rest; }
            double den = 1 - x1 - x0, q = den > 1e-9 ? (pf - x1) / den : pf;
            g_pover = (float)(q < 1e-5 ? 1e-5 : q > 1 - 1e-5 ? 1 - 1e-5 : q);
        }
        int bit = cm_bit(m, slot, cx, sel, g_acmode == 2 ? 0 : rw_bit(&w, e), pp);
        g_pover = -1.f;
        int k = 0; for (int i = 0; i < nl; i++) if (rw_bit(&w, live[i]) == bit) live[k++] = live[i]; nl = k;   /* keep values still on this branch */
        rw_step(&w, &p, bit);
    }
    g_acmode = save;
    return w.e;
}
static inline void cm_residual(CM *m, const int *cx, int sel, int e, int r, double d, double b) { cm_residual_x(m, cx, sel, e, r, d, b, NULL, 0); }

static void funnel_init(Funnel *f, const Funnel *parent) {
    if (parent) { *f = *parent; f->n = 0; f->perf = parent->perf * 1.25f + 0.05f; }      /* inherit weights + covariance; start a bit less trusted */
    else {
        memset(f, 0, sizeof *f); f->perf = 8.f;
        for (int i = 0; i < NT; i++) f->P[i * NT + i] = P0;
    }
    f->nsl = 0; f->idle = 0; memset(f->s1, 0, sizeof f->s1); memset(f->s2, 0, sizeof f->s2);
}
static void reset_cov(Funnel *f) { memset(f->P, 0, sizeof f->P); for (int i = 0; i < NT; i++) f->P[i * NT + i] = P0; }
/* learn from one observation. na = number of active taps. */
static inline void funnel_learn(Funnel *f, const float *u, int na, float err, float un2, int rls) {
    float a = 1.f / (f->n + 1.5f); if (a < 1.f / 64) a = 1.f / 64;           /* learning rate floors out: saturation */
    float surpr = fabsf(err) / (f->perf + 0.3f);                              /* prediction error relative to this funnel's usual error */
    float ae = fabsf(err); if (SALALL > 0.f) ae = fminf(ae, SALALL * (f->perf + 0.3f));   /* skill estimate ignores outliers too */
    f->perf += (ae - f->perf) * (a < 1.f / 24 ? 1.f / 24 : a);
    f->bias += a * (SALBIAS > 0.f ? clampf(err, -SALBIAS * (f->perf + 0.3f), SALBIAS * (f->perf + 0.3f)) : err) * 0.3f;
    if (opt_l && !(GATE > 0.f && fabsf(err) < GATE && f->n > 64)) {          /* gating: a mature funnel ignores unsurprising pixels */
        if (opt_r && rls) {                                                  /* recursive least squares on the active taps */
            double lam = LAM, omega = 1.0;
            if (LAMPE) { double g = clampf(surpr * surpr, 0.25f, 4.f); lam = 1.0 - (1.0 - LAM) * g; }      /* surprise -> shorter memory */
            if (SAL != 0.f) { double s = pow(clampf(surpr, 1.f, SALMAX), SALP); omega = SAL > 0 ? s : 1.0 / s; }   /* salience weight */
            double Pu[NT], den = lam / omega;
            for (int i = 0; i < na; i++) { double s = 0; for (int j = 0; j < na; j++) s += f->P[i * NT + j] * u[j]; Pu[i] = s; den += u[i] * s; }
            if (!(den > 1e-6) || !isfinite(den)) reset_cov(f);              /* degenerate: reset covariance, keep weights */
            else {
                double inv = 1.0 / den, il = 1.0 / lam; int bad = 0;
                for (int i = 0; i < na; i++) { double w = f->w[i] + Pu[i] * inv * err; f->w[i] = (float)(w > 4 ? 4 : w < -4 ? -4 : w); }
                for (int i = 0; i < na; i++) for (int j = i; j < na; j++) {
                    double v = (f->P[i * NT + j] - Pu[i] * Pu[j] * inv) * il;
                    if (i == j && (v > 1e3 || v < 0)) bad = 1;
                    f->P[i * NT + j] = f->P[j * NT + i] = v;
                }
                if (bad) reset_cov(f);
            }
        } else { float g = MU * err / un2; for (int k = 0; k < na; k++) f->w[k] += g * u[k]; }
    }
    f->n++;
}

/* ---- funnel field: resonance, landing, sleep ---- */
static inline float dist2(const float *a, const float *b) { float s = 0; for (int i = 0; i < DIM; i++) { float d = a[i] - b[i]; s += d * d; } return s; }
static inline float resonance(const Funnel *f, float d) {                   /* as funnel/physics.py: radial falloff x depth pull x fatigue */
    float R = f->mouth * 3.f; if (d > R) return 0.f;
    float depth = logf(1.f + f->n) * 0.05f, fat = fminf(1.f, f->n * 0.00002f);
    return (1.f - d / R) * (1.f + depth) * (1.f - 0.5f * fat) + 1e-4f;
}
static void field_sleep(Funnel *fl, int *nf) {                              /* fission of busy wide funnels, prune of idle ones */
    int n0 = *nf;
    for (int i = 0; i < n0; i++) {
        Funnel *f = &fl[i];
        if (f->n < 0) continue;
        if (f->nsl == 0) { if (++f->idle >= 3 && f->n < 200) { f->n = -1; st_prune++; } continue; }
        f->idle = 0;
        if (f->nsl >= 600 && *nf < FMAX) {
            int bd = 0; float bv = -1, mean[DIM];
            for (int d = 0; d < DIM; d++) { mean[d] = f->s1[d] / f->nsl; float v = f->s2[d] / f->nsl - mean[d] * mean[d]; if (v > bv) { bv = v; bd = d; } }
            float sd = sqrtf(fmaxf(bv, 0.f));
            if (sd > f->mouth * 0.35f) {
                int slot = -1; for (int j = 0; j < *nf; j++) if (fl[j].n < 0) { slot = j; break; }
                if (slot < 0) slot = (*nf)++;
                Funnel *g = &fl[slot]; *g = *f; g->n = f->n / 2; f->n -= g->n;
                memcpy(g->c, mean, sizeof mean); memcpy(f->c, mean, sizeof mean);
                g->c[bd] += sd; f->c[bd] -= sd; g->mouth = f->mouth = fmaxf(MERGE * 0.5f, f->mouth * 0.8f);
                g->nsl = 0; g->idle = 0; memset(g->s1, 0, sizeof g->s1); memset(g->s2, 0, sizeof g->s2);
                st_fission++;
            }
        }
        f->nsl = 0; memset(f->s1, 0, sizeof f->s1); memset(f->s2, 0, sizeof f->s2);
    }
}


/* ---- shared state (read by all plane threads) and the staggered colour pipeline ---- */
static size_t N; static int *pl[3]; static float *pfin[3]; static float *pixcost;
static const int order[3] = {1, 0, 2};              /* G first, then R, then B */
static const int lsize[MAXLV] = {1, NA, NA * NTH, NA * NTH * NL};
static double bits[3]; static int nsub, icurve;
static Funnel *gl[MAXLV];                            /* sequential mode: G plane's fully trained grid funnels (Main register) */
static int PIPE = 0, LAG = 2, SNAPK = 64;            /* pipe=1: the three colour layers run at once, each LAG rows behind the previous one */
/* inter=1: PIXEL BY PIXEL instead of layer by layer: G, R, B of one pixel are coded before the next pixel, so G can use the
   previous pixels' R and B. cmatch=1: WHOLE-COLOUR MATCH: G first codes "is this pixel's whole colour equal to one of the
   W, N, NW, NE colours, and which?"; on a match the three layers cost nothing (models still learn from the pixel). */
static int INTER = 0, CMATCH = 0, MEXCL = 0; static double MTHR = 0.35; static _Thread_local long long st_excl; static unsigned char *asked; static long prog[3]; static unsigned char *matched; static signed char *midx; static double match_bits;
static _Thread_local long long st_matched; static unsigned char *freed; static int *mcand;
static inline void wait_pix(int p, long need) {
    unsigned spins = 0;
    while (__atomic_load_n(&prog[p], __ATOMIC_ACQUIRE) < need) { if (++spins > 2000) { sched_yield(); spins = 0; } }
}
typedef struct { float p; unsigned short n; } MCtr;
static MCtr mc_flag[4][128], mc_idx[256]; static float mc_w[16][6]; static int MINF = 0;
static inline float mc_p(MCtr *c) { return c->p < 0.002f ? 0.002f : c->p > 0.998f ? 0.998f : c->p; }
static inline void mc_upd(MCtr *c, int bit) { float a = 1.f / (c->n + 1.6f); if (a < 1.f / 255) a = 1.f / 255; c->p += (bit - c->p) * a; if (c->n < 65000) c->n++; }
static int MODE = 0; static AC acs[3];               /* MODE 0: ideal length only, 1: write a real bitstream, 2: decode one */
typedef struct { int on, k, nci; size_t idx, cand[4]; MCtr *cc[4]; float st[6], *w, pm; } MPend;
static MPend mpend;                                  /* match question waiting for the pixel's true colour before it learns */
static void match_learn(MPend *q) {                  /* G thread, once R and B of pixel q->idx are known (encoder and decoder alike) */
    int m = -1; size_t i = q->idx;
    for (int t = 0; t < q->k; t++) { size_t a = q->cand[t]; if (pl[0][a] == pl[0][i] && pl[1][a] == pl[1][i] && pl[2][a] == pl[2][i]) { m = t; break; } }
    matched[i] = m >= 0; midx[i] = (signed char)m;
    if (q->k > 0) { int bit = m >= 0;
        for (int t = 0; t < 6; t++) if (t < q->nci || t >= 4) q->w[t] += 0.01f * (bit - q->pm) * q->st[t];
        for (int t = 0; t < q->nci; t++) mc_upd(q->cc[t], bit); }
    q->on = 0;
}
static Funnel **snaps[4096];                         /* pipeline registers: G's grid funnels frozen every SNAPK rows */
static int rows_done[3];
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
static long long st_excl_all, tot_matched, tot_born, tot_field_born, tot_fission, tot_prune, tot_hq, tot_hh, tot_replay, tot_mat; static int tot_field_max;
static void wait_rows(int plane, int need) {         /* block until layer `plane` has finished `need` rows (coding order) */
    pthread_mutex_lock(&mu); while (rows_done[plane] < need) pthread_cond_wait(&cv, &mu); pthread_mutex_unlock(&mu);
}
static void publish_rows(int plane, int done) { pthread_mutex_lock(&mu); rows_done[plane] = done; pthread_cond_broadcast(&cv); pthread_mutex_unlock(&mu); }
static inline Funnel *register_src(int lvl, int idx, int yi) {   /* where an R/B funnel is born from (G's funnel at the same address) */
    Funnel *a = NULL;
    if (PIPE) { int si = yi / SNAPK; Funnel **sn = snaps[INTER ? si : (si > 0 ? si : 1)]; a = sn ? sn[lvl] : NULL; } else a = gl[lvl];   /* early rows use G's first snapshot */
    return (a && a[idx].n > 0) ? &a[idx] : NULL;
}

static void code_plane(int oi) {
    long long born = 0; int field_max = 0;
    g_ac = &acs[oi]; g_acmode = MODE; if (MODE == 2) ac_start_dec(g_ac); else if (MODE == 1) { g_ac->x1 = 0; g_ac->x2 = 0xffffffffu; }
        int c = order[oi]; int *P = pl[c];
        const int nml = (MLINE && oi >= 1) ? ((MLINE == 2 && oi == 2) ? 2 : 1) : 0, iml = NSUB + (HIPPO ? 1 : 0) + (CURVE ? 1 : 0);
        const int hasmat = MATF && oi >= 1, imat = iml + nml;
        const int nsubp = nsub + nml + hasmat;
        Mat *mats = hasmat ? calloc(MMAXC, sizeof(Mat)) : NULL; int nmat = 0;
        float *gmt = GMASS ? calloc(256 * 256, sizeof(float)) : NULL, *gtot = GMASS ? calloc(256, sizeof(float)) : NULL; int gkey = 0;
        const int nact = NNB + (opt_c ? (oi >= 1 ? CK + 1 + (opt_f ? NF : 0) : 0) + (oi >= 2 ? CK + 1 + (opt_f ? NF : 0) : 0) : 0) + (HIPPO ? 1 : 0) + nml + hasmat + (GMASS ? 1 : 0);   /* + 3 'below' taps on odd rows when ilace */
        Funnel *lv[MAXLV]; for (int j = 0; j < MAXLV; j++) { lv[j] = malloc(lsize[j] * sizeof(Funnel)); for (int i = 0; i < lsize[j]; i++) lv[j][i].n = -1; }
        Funnel *fld = FIELD ? malloc(FMAXC * sizeof(Funnel)) : NULL; int nfld = 0;
        Mouth *mouth = malloc(2 * NA * NEN * NCE * sizeof(Mouth)); for (int i = 0; i < 2 * NA * NEN * NCE; i++) { mouth[i].scale = 6.f; mouth[i].n = 0; }
        float *esub[NSUBX]; for (int k = 0; k < NSUBX; k++) esub[k] = calloc(N, sizeof(float));
        float *efin = calloc(N, sizeof(float)), *actm = calloc(N, sizeof(float));
        short *fid = calloc(N, sizeof(short));
        float *hmem = HIPPO ? calloc((size_t)1 << HBITS, sizeof(float)) : NULL; unsigned char *hcnt = HIPPO ? calloc((size_t)1 << HBITS, 1) : NULL;
        /* replay buffer: one row of (funnel, taps, target, base) */
        Funnel **rp_f = REPLAY ? malloc(W * sizeof(Funnel *)) : NULL; float *rp_u = REPLAY ? malloc((size_t)W * NT * sizeof(float)) : NULL;
        float *rp_t = REPLAY ? malloc(W * sizeof(float)) : NULL;
        CM cm; int ncx;
        { int sizes[NCX] = {2 * NA * NEN * NCE, NA * NTH * NL, 625 * 3, 4 * NEN * NL * 3, 9 * 9 * 5, 6 * NEN * 3 * 3}; ncx = 6;
          if (BOUNCE) sizes[ncx++] = 1 << 14;
          if (HIPPO) sizes[ncx++] = 9 * 3 * NEN;
          if (MS) sizes[ncx++] = 8 * NA * NEN;
          if (MCTX && MATF && oi >= 1) sizes[ncx++] = MMAXC * NEN;
          cm_init(&cm, sizes, ncx, NA_SEL); }
        const int HE = (H + 1) / 2;
        for (int yi = 0; yi < H; yi++) {
          const int y = ILACE ? (yi < HE ? 2 * yi : 2 * (yi - HE) + 1) : yi, pass = ILACE && yi >= HE;
          RS = (ILACE && !pass) ? 2 : 1;
          const int nsubq = nsubp + (pass ? 1 : 0), ivb = nsubp;
          if (PIPE && !INTER && oi > 0) {                                        /* earlier layers are LAG rows ahead; R starts once G's first snapshot exists */
              int need = ILACE ? H : yi + LAG; if (REGISTER && oi == 1 && need < SNAPK) need = SNAPK;
              wait_rows(oi - 1, need < H ? need : H);
          }
          for (int x = 0; x < W; x++) {
            size_t idx = (size_t)y * W + x;
            if (INTER) {                                                          /* pixel by pixel: G waits for the previous pixel's R and B */
                if (oi == 0) { wait_pix(1, idx); wait_pix(2, idx); }
                else if (oi == 1) { wait_pix(0, idx + 1); wait_pix(2, idx); }
                else { wait_pix(0, idx + 1); wait_pix(1, idx + 1); }
            }
            int nbv[NNB]; size_t nbi[NNB];                                        /* causal fetch with fallbacks at the borders */
            for (int k = 0; k < NNB; k++) {
                int xx = x + TAPX[k], yy = y + TAPY[k] * RS; size_t ii;
                if (yy < 0) ii = x > 0 ? idx - 1 : (y >= RS ? idx - (size_t)RS * W : idx);
                else { xx = clampi(xx, 0, W - 1); ii = (yy == y && xx >= x) ? (x > 0 ? idx - 1 : (y >= RS ? idx - (size_t)RS * W : idx)) : (size_t)yy * W + xx; }
                nbi[k] = ii; nbv[k] = (ii == idx) ? 128 : P[ii];
            }
            int Wv = nbv[0], Nv = nbv[1], NWv = nbv[2], NEv = nbv[3], WWv = nbv[4], NNv = nbv[5], NNEv = nbv[6];
            int m = med(Wv, Nv, NWv);

            /* ---- fixed parallel balls ---- */
            float sp[NSUBX];
            sp[0] = (float)m;
            sp[1] = clampf((float)(Wv + Nv - NWv), 0, 255);
            sp[2] = clampf((Wv + Nv) * 0.5f + (NEv - NWv) * 0.25f, 0, 255);
            sp[3] = clampf((float)(Wv + NEv - Nv), 0, 255);
            sp[4] = clampf(Nv + (Nv - NNv) * 0.5f + (Wv - NWv) * 0.5f, 0, 255);
            /* ---- hippocampus: recall what followed this neighbourhood shape before ---- */
            unsigned hkey = 0; int hhit = 0;
            if (HIPPO) {
                float ref = (Wv + Nv) * 0.5f;
                #define HQ(v) ((unsigned)(clampi((int)lrintf(((v) - ref) / 3.f), -3, 3) + 3))
                hkey = ((((((HQ(Wv) * 7 + HQ(Nv)) * 7 + HQ(NWv)) * 7 + HQ(NEv)) * 7 + HQ(WWv)) * 7 + HQ(NNv)) * 7 + HQ(NNEv));
                if (opt_c && oi > 0) { int gr = clampi((int)lrintf(pl[1][idx] - pfin[1][idx]), -3, 3) + 3; hkey = hkey * 7 + gr; }
                hkey = (hkey * 2654435761u) >> (32 - HBITS);
                hhit = hcnt[hkey] > 0; st_hippo_q++; st_hippo_hit += hhit;
                sp[NSUB] = clampf(ref + (hhit ? hmem[hkey] : 0.f), 0, 255);
            }
            if (CURVE) {                                                          /* follow the edge: N-row trend, bent by the row above's change of trend */
                int NWWv = (y > 0 && x > 1) ? P[idx - W - 2] : NWv, NNWv = (y > 1 && x > 0) ? P[idx - 2 * W - 1] : NNv;
                float tw = (float)(Wv - NWv), tw2 = (float)(NWv - NNWv), tn = (float)(Nv - NWv), tn2 = (float)(NWv - NWWv);
                sp[icurve] = clampf(0.5f * (Nv + tw + (tw - tw2) * 0.5f) + 0.5f * (Wv + tn + (tn - tn2) * 0.5f), 0, 255);
            }
            for (int q = 0; q < nml; q++) {                                       /* material line: fit this plane against an earlier one on the neighbours */
                int *Q = pl[q == 0 ? 1 : 0]; float gm = 0, rm = 0, gv = 0, cv_ = 0;
                for (int k = 0; k < MWIN; k++) { gm += Q[nbi[k]]; rm += nbv[k]; }
                gm /= MWIN; rm /= MWIN;
                for (int k = 0; k < MWIN; k++) { float dg = Q[nbi[k]] - gm; gv += dg * dg; cv_ += dg * (nbv[k] - rm); }
                float slope = (cv_ + MLAM) / (gv + MLAM);
                sp[iml + q] = clampf(rm + slope * (Q[idx] - gm), 0, 255);
            }
            int Sv = 0, SWv = 0, SEv = 0;
            if (pass) {                                                           /* the row below is already decoded (coarse past) */
                int yb = y + 1 < H ? y + 1 : y - 1;
                Sv = P[(size_t)yb * W + x]; SWv = P[(size_t)yb * W + (x > 0 ? x - 1 : x)]; SEv = P[(size_t)yb * W + (x < W - 1 ? x + 1 : x)];
                sp[ivb] = 0.5f * (Nv + Sv);
            }
            int mland = -1; float md[3] = {0, 0, 0};
            if (hasmat) {                                                         /* land in a material funnel (colour space) */
                int *Gp = pl[1], *Rp = pl[0]; float rs = 0, gs = 0, bs = 0;
                for (int k = 0; k < CK; k++) { rs += oi == 1 ? nbv[k] : Rp[nbi[k]]; gs += Gp[nbi[k]]; if (oi == 2) bs += nbv[k]; }
                if (oi == 1) md[0] = (rs + 4.f) / (rs + gs + 8.f);
                else { md[0] = (bs + 4.f) / (bs + rs + gs + 12.f); md[1] = (Rp[idx] + 2.f) / (Rp[idx] + Gp[idx] + 4.f); }
                md[2] = MBW * gs / (4.f * 255.f);
                float bd = 1e30f; for (int i = 0; i < nmat; i++) { float d = 0; for (int k = 0; k < 3; k++) { float t = md[k] - mats[i].c[k]; d += t * t; } if (d < bd) { bd = d; mland = i; } }
                if ((mland < 0 || sqrtf(bd) > MRAD) && nmat < MMAX) {            /* a new material: born as a copy of the nearest */
                    int ni = nmat++; if (mland >= 0) mats[ni] = mats[mland]; else memset(&mats[ni], 0, sizeof(Mat));
                    memcpy(mats[ni].c, md, sizeof md); mats[ni].n = 0; mland = ni; st_mat_born++;
                }
                int kk = oi == 1 ? 2 : 3; double z[3] = {1.0, (double)Gp[idx], (double)Rp[idx]}, w0[3] = {0.0, 1.0, 0.0}, w[3];
                mat_solve(&mats[mland], kk, w0, w);
                double pr = 0; for (int k = 0; k < kk; k++) pr += w[k] * z[k];
                sp[imat] = clampf((float)pr, 0, 255);
            }
            float base;
            if (opt_w) {
                float sw = 0, sv = 0;
                for (int k = 0; k < nsubq; k++) {
                    float S = emap(esub[k], x - 1, y) + emap(esub[k], x, y - RS) + emap(esub[k], x - 1, y - RS) + emap(esub[k], x + 1, y - RS) + 0.5f;
                    float wk = 1.f / (S * S); sw += wk; sv += wk * sp[k];
                }
                base = sv / sw;
            } else base = sp[0];

            /* ---- transduction: activity, level, spin, error energy, cross-plane surprise, wide activity ---- */
            int dh = abs(Wv - WWv) + abs(Nv - NWv) + abs(Nv - NEv), dv = abs(Wv - NWv) + abs(Nv - NNv) + abs(NEv - NNEv), act = dh + dv;
            actm[idx] = (float)act;
            int A = 0; while (A < NA - 1 && act > ATH[A]) A++;
            int L = opt_v ? clampi((int)base >> 6, 0, NL - 1) : 0, TH = 0;
            float gx = (float)((Nv - NWv) + (NEv - Nv) + (Wv - WWv)), gy = (float)((Wv - NWv) + (Nv - NNv) + (NEv - NNEv)), gm = fabsf(gx) + fabsf(gy);
            if (opt_t) { if (gm < 9.f) TH = 8; else { float a = atan2f(gy, gx); TH = ((int)floorf((a + 3.14159265f) / 0.78539816f)) & 7; } }
            float en = 0.f; int EN = 0;
            if (opt_e) {
                en = emap(efin, x - 1, y) + emap(efin, x, y - RS) + 0.5f * (emap(efin, x - 1, y - RS) + emap(efin, x + 1, y - RS)) + 0.5f * (emap(efin, x - 2, y) + emap(efin, x, y - 2 * RS));
                static const float ETH[NEN - 1] = {0.6f, 1.2f, 2.f, 3.2f, 5.f, 8.f, 13.f, 22.f, 40.f};
                while (EN < NEN - 1 && en > ETH[EN]) EN++;
            }
            float ce = 0.f; int CE = 0;
            if (opt_e && opt_c && oi > 0) {
                ce = fabsf((float)pl[1][idx] - pfin[1][idx]); if (oi == 2) ce += fabsf((float)pl[0][idx] - pfin[0][idx]);
                static const float CTH[NCE - 1] = {0.5f, 1.5f, 3.f, 6.f, 12.f};
                while (CE < NCE - 1 && ce > CTH[CE]) CE++;
            }
            float wide = 0.f; int AW = 0;
            if (MS || FIELD) {
                int cnt = 0; for (int k = 0; k < 8; k++) { int xx = x + WOFX[k], yy = y + WOFY[k] * RS; if (xx >= 0 && xx < W && yy >= 0 && (yy < y || xx < x)) { wide += actm[(size_t)yy * W + xx]; cnt++; } }
                wide = cnt ? wide / cnt : (float)act;
                while (AW < 7 && wide > ATH[AW]) AW++;
            }

            /* ---- nested funnels (+ optional continuous field) ---- */
            int key[MAXLV] = {0, A, A * NTH + TH, (A * NTH + TH) * NL + L};
            int lvidx[MAXLV], nlv = 0;
            if (FIELD) { lvidx[nlv++] = 0; lvidx[nlv++] = 1; }
            else if (opt_p) { lvidx[nlv++] = 0; lvidx[nlv++] = 1; if (opt_t) lvidx[nlv++] = 2; if (opt_t && opt_v) lvidx[nlv++] = 3; }
            else lvidx[nlv++] = opt_t ? (opt_v ? 3 : 2) : (opt_v ? 3 : 1);
            Funnel *path[MAXLV + KMAX]; float resw[MAXLV + KMAX];
            for (int j = 0; j < nlv; j++) {
                int lvl = lvidx[j], k = key[lvl];
                if (!opt_t && lvl == 3) k = A * NL + L;
                path[j] = &lv[lvl][k % lsize[lvl]]; resw[j] = 1.f;
                if (path[j]->n < 0) {
                    Funnel *g = (REGISTER && oi > 0) ? register_src(lvl, k % lsize[lvl], yi) : NULL;
                    funnel_init(path[j], g ? g : (j > 0 && opt_p) ? path[j - 1] : NULL); path[j]->n = 0; born++;
                    if (g) { for (int t = NNB; t < NT; t++) path[j]->w[t] = 0.f; reset_cov(path[j]); }   /* spatial weights from G; colour taps start fresh */
                }
            }
            int nlearn = nlv, nset = nlv, land = -1; float D[DIM];
            if (FIELD) {                                                          /* launch: descriptor, find resonant funnels, land or create */
                float s = fminf(1.f, gm / 30.f), ang = atan2f(gy, gx);
                D[0] = log2f(1.f + act) / 9.f; D[1] = log2f(1.f + 4.f * en) / 8.f; D[2] = log2f(1.f + wide) / 9.f;
                D[3] = 0.5f + 0.5f * s * cosf(ang); D[4] = 0.5f + 0.5f * s * sinf(ang); D[5] = base / 255.f + 2.f * pass; D[6] = log2f(1.f + 4.f * ce) / 8.f;
                int best[KMAX]; float bd[KMAX]; for (int k = 0; k < KRES; k++) { best[k] = -1; bd[k] = 1e30f; }
                for (int i = 0; i < nfld; i++) {
                    if (fld[i].n < 0) continue;
                    float d2 = dist2(D, fld[i].c);
                    if (d2 < bd[KRES - 1]) { int k = KRES - 1; while (k > 0 && d2 < bd[k - 1]) { bd[k] = bd[k - 1]; best[k] = best[k - 1]; k--; } bd[k] = d2; best[k] = i; }
                }
                float d0 = best[0] >= 0 ? sqrtf(bd[0]) : 1e9f;
                if ((best[0] < 0 || d0 > fld[best[0]].mouth) && nfld < FMAX + 64) {      /* lands in empty space: a new funnel seeds, inheriting */
                    int slot = -1; for (int j = 0; j < nfld; j++) if (fld[j].n < 0) { slot = j; break; }
                    if (slot < 0 && nfld < FMAXC) slot = nfld++;
                    if (slot >= 0) {
                        funnel_init(&fld[slot], best[0] >= 0 ? &fld[best[0]] : path[nlv - 1]); fld[slot].n = 0;
                        memcpy(fld[slot].c, D, sizeof D); fld[slot].mouth = MERGE; st_field_born++;
                        for (int k = KRES - 1; k > 0; k--) { best[k] = best[k - 1]; bd[k] = bd[k - 1]; }
                        best[0] = slot; bd[0] = 0.f;
                    }
                }
                for (int k = 0; k < KRES; k++) {
                    if (best[k] < 0) continue;
                    float r = resonance(&fld[best[k]], sqrtf(bd[k]));
                    if (k > 0 && r <= 1e-4f) continue;
                    path[nset] = &fld[best[k]]; resw[nset] = k == 0 ? fmaxf(r, 0.05f) : r; nset++;
                }
                land = best[0]; nlearn = nlv + 1;
                fid[idx] = (short)(land + 1);
                if (nfld > field_max) field_max = nfld;
            } else if (BOUNCE) fid[idx] = (short)(path[nlv - 1] - lv[lvidx[nlv - 1]] + 1);

            float u[NT]; int nu = 0;
            for (int k = 0; k < NNB; k++) u[nu++] = (float)nbv[k] - base;
            if (opt_c && oi > 0) {                                                /* colour dimension */
                for (int pi = 0; pi < oi; pi++) {
                    int *Q = pl[pi == 0 ? 1 : 0]; float *Qf = pfin[pi == 0 ? 1 : 0];
                    for (int k = 0; k < CK; k++) u[nu++] = (float)(Q[idx] - Q[nbi[k]]);
                    if (opt_f) for (int k = 0; k < NF; k++) { int fx = clampi(x + FUTX[k], 0, W - 1), fy = clampi(y + FUTY[k], 0, H - 1); u[nu++] = (float)(Q[idx] - Q[(size_t)fy * W + fx]); }
                    u[nu++] = (float)Q[idx] - Qf[idx];
                }
            }
            if (HIPPO) u[nu++] = sp[NSUB] - base;
            for (int q = 0; q < nml; q++) u[nu++] = sp[iml + q] - base;
            if (hasmat) u[nu++] = sp[imat] - base;
            if (GMASS) {                                                          /* gravity: pull toward heavy values near the prediction */
                gkey = (FIELD && land >= 0 ? land : A) & 255; const float *M = gmt + gkey * 256;
                int v0 = (int)lrintf(base); float sw = 0, sv = 0;
                for (int v = v0 - GMR; v <= v0 + GMR; v++) { if (v < 0 || v > 255 || M[v] <= 0) continue; float d = (v - base) / GSIG, w = M[v] / (1.f + d * d); sw += w; sv += w * v; }
                u[nu++] = sw > 1e-3f ? sv / sw - base : 0.f;
            }
            if (pass) { u[nu++] = Sv - base; u[nu++] = SWv - base; u[nu++] = SEv - base; }
            if (INTER && oi == 0) {                                               /* G sees its neighbours' colour: R-G and B-G at W and N */
                size_t iw = x > 0 ? idx - 1 : (y > 0 ? idx - W : idx), in_ = y > 0 ? idx - W : iw;
                if (iw == idx) { u[nu++] = 0; u[nu++] = 0; u[nu++] = 0; u[nu++] = 0; }
                else { u[nu++] = (float)(pl[0][iw] - pl[1][iw]); u[nu++] = (float)(pl[2][iw] - pl[1][iw]); u[nu++] = (float)(pl[0][in_] - pl[1][in_]); u[nu++] = (float)(pl[2][in_] - pl[1][in_]); }
            }
            const int nactp = nact + (pass ? 3 : 0) + (INTER && oi == 0 ? 4 : 0);
            float un2 = 1.f; for (int k = 0; k < nactp; k++) un2 += u[k] * u[k];
            float pj[MAXLV + KMAX], sw2 = 0.f, sv2 = 0.f;
            for (int j = 0; j < nset; j++) {                                      /* global workspace */
                float d = 0.f; for (int k = 0; k < nactp; k++) d += path[j]->w[k] * u[k];
                pj[j] = base + (opt_l ? d : 0.f) + path[j]->bias;
                float pe_ = path[j]->perf + 0.05f, wj = resw[j] / (WST == 2.f ? pe_ * pe_ : powf(pe_, WST));
                if (GRAV > 0.f) wj *= 1.f + GRAV * logf(2.f + path[j]->n);
                if (WST <= 0.f) wj = resw[j] / pe_;
                pj[j] = pj[j]; sw2 += wj; sv2 += wj * pj[j];
            }
            float pfv;
            if (WST <= 0.f) { int bj = 0; for (int j = 1; j < nset; j++) if (path[j]->perf < path[bj]->perf) bj = j; pfv = pj[bj]; }   /* winner takes all */
            else pfv = sv2 / sw2;
            float pc = clampf(pfv, 0, 255);
            pfin[c][idx] = pc;

            Mouth *mo = &mouth[((pass * NA + A) * NEN + EN) * NCE + CE];
            float sb = mo->scale < 0.25f ? 0.25f : mo->scale;

            /* ---- ideal code length ---- */
            int xv = MODE == 2 ? 0 : P[idx];                                      /* decoder: not known yet, decoded below */
            if (!opt_b) {
                double t = xv - pc, mass = lapF(t + 0.5, sb) - lapF(t - 0.5, sb), Z = lapF(255.5 - pc, sb) - lapF(-0.5 - pc, sb);
                double pr = mass / Z; if (pr < 1e-9) pr = 1e-9; bits[c] += -log2(pr);
            } else {
                int r = (int)lrintf(pc), e = xv - r; int ex[4], nex = 0;
                #define CL2(v) (clampi((v), -2, 2) + 2)
                int cx[NCX];
                int fr = (int)floorf((pc - r + 0.5f) * 3.999f);
                int cfl = (opt_c && oi > 0) ? (CE >= 3 ? 2 : CE >= 1 ? 1 : 0) : 0;
                cx[0] = ((pass * NA + A) * NEN + EN) * NCE + CE;
                cx[1] = (int)((path[nlv - 1] - lv[lvidx[nlv - 1]]) % (NA * NTH * NL));
                cx[2] = (((CL2(Wv - r) * 5 + CL2(Nv - r)) * 5 + CL2(NWv - r)) * 5 + CL2(NEv - r)) * 3 + cfl;
                cx[3] = ((fr * NEN + EN) * NL + L) * 3 + cfl;
                cx[4] = (clampi((Wv - Nv) / 4, -4, 4) + 4) * 9 * 5 + (clampi((Nv - NWv) / 4, -4, 4) + 4) * 5 + (fr > 3 ? 3 : fr);
                float sd = 0;
                for (int k = 0; k < NSUB; k++) { float dd = sp[k] - pc; sd += dd * dd; }
                sd = sqrtf(sd / NSUB); int SQ = sd < 0.4f ? 0 : sd < 0.9f ? 1 : sd < 1.8f ? 2 : sd < 3.5f ? 3 : sd < 7.f ? 4 : 5;
                cx[5] = ((SQ * NEN + EN) * 3 + cfl) * 3 + (TH == 8 ? 0 : (TH & 1) + 1);
                int ci = 6;
                if (BOUNCE) {                                                     /* where did the neighbours' balls land? */
                    unsigned fw = x > 0 ? fid[idx - 1] : 0, fn = y >= RS ? fid[idx - (size_t)RS * W] : 0, fc = fid[idx];
                    cx[ci++] = (int)((((fw * 1031u + fn) * 1031u + fc) * 2654435761u) >> 18);
                }
                if (HIPPO) cx[ci++] = ((CL2((int)lrintf(sp[NSUB] - pc)) + (hhit ? 5 : 0)) % 9 * 3 + (hhit ? (hcnt[hkey] > 3 ? 2 : 1) : 0)) * NEN + EN;
                if (MS) cx[ci++] = (AW * NA + A) * NEN + EN;
                if (MCTX && hasmat) cx[ci++] = mland * NEN + EN;
                g_sel2 = EN * 3 + cfl; g_actx = EN * 3 + (SQ < 2 ? 0 : SQ < 4 ? 1 : 2);
                if (CMATCH && oi == 0) {                                          /* whole-colour match, coded once per pixel by G */
                    if (mpend.on) match_learn(&mpend);                            /* the previous pixel is complete now: learn its true answer */
                    size_t cand[4]; int k = 0;
                    size_t nbs[4] = {x > 0 ? idx - 1 : idx, y > 0 ? idx - W : idx, (x > 0 && y > 0) ? idx - W - 1 : idx, (y > 0 && x < W - 1) ? idx - W + 1 : idx};
                    for (int q = 0; q < 4; q++) { size_t a = nbs[q]; if (a == idx) continue; int dup = 0;
                        for (int t = 0; t < k; t++) if (pl[0][cand[t]] == pl[0][a] && pl[1][cand[t]] == pl[1][a] && pl[2][cand[t]] == pl[2][a]) dup = 1;
                        if (!dup) cand[k++] = a; }
                    asked[idx] = 0; freed[idx] = 0;
                    MPend *q_ = &mpend; q_->on = 1; q_->idx = idx; q_->k = k; memcpy(q_->cand, cand, sizeof cand);
                    if (k > 0) {
                        int m = -1;                                               /* encoder only: the true answer */
                        if (MODE != 2) for (int t = 0; t < k; t++) if (pl[0][cand[t]] == pl[0][idx] && pl[1][cand[t]] == pl[1][idx] && pl[2][cand[t]] == pl[2][idx]) { m = t; break; }
                        int mW = x > 0 ? matched[idx - 1] : 0, mN = y > 0 ? matched[idx - W] : 0; float dg = 99;
                        for (int t = 0; t < k; t++) { float d = fabsf(pl[1][cand[t]] - pc); if (d < dg) dg = d; }
                        int dq = dg < 0.5f ? 0 : dg < 1.5f ? 1 : dg < 4.f ? 2 : 3, c0 = (k - 1) * 4 + mW * 2 + mN;
                        float pg = 0;                                               /* minf=1: how sure the G layer already is of a candidate's G */
                        if (MINF) { int sv[4], ns = 0; for (int t = 0; t < k; t++) { int v = pl[1][cand[t]], dup = 0; for (int u2 = 0; u2 < ns; u2++) dup |= sv[u2] == v; if (dup) continue; sv[ns++] = v;
                                pg += (float)cm_pval(&cm, cx, A, v - r, r, (double)(pc - r), (double)sb); } }
                        pg = pg < 0.001f ? 0.001f : pg > 0.999f ? 0.999f : pg;
                        int pgq = (int)(pg * 7.999f);
                        MCtr *cc[4] = {&mc_flag[0][c0], &mc_flag[1][A * NEN + EN], &mc_flag[2][(k - 1) * 4 + dq], &mc_flag[3][(k - 1) * 8 + pgq]};
                        const int nci = MINF ? 4 : 3;
                        float st[6], dot = 0; float *w = mc_w[c0];
                        for (int q = 0; q < nci; q++) { float pp = mc_p(cc[q]); st[q] = logf(pp / (1 - pp)); dot += w[q] * st[q]; }
                        if (MINF) { st[4] = logf(pg / (1 - pg)); dot += w[4] * st[4]; } else st[4] = 0;
                        st[5] = 0.3f; dot += w[5] * st[5];
                        float pm = 1.f / (1.f + expf(-dot)); pm = pm < 1e-4f ? 1e-4f : pm > 1 - 1e-4f ? 1 - 1e-4f : pm;
                        int bit = m >= 0, ask = CMATCH < 2 || pm >= MTHR, mi = -1;  /* cmatch=2: only ask where a match is expected */
                        if (ask) { bit = ac_bit(pm, bit); match_bits += -log2f(bit ? pm : 1 - pm); }
                        if (bit && ask) { for (int t = 0; t < k - 1; t++) {           /* which candidate: one yes/no per candidate */
                                int wi = x > 0 ? midx[idx - 1] + 1 : 0; MCtr *ci = &mc_idx[((t * 4 + (k - 1)) * 5 + wi) & 255];
                                float pp = mc_p(ci); int b2 = ac_bit(pp, m == t); match_bits += -log2f(b2 ? pp : 1 - pp); mc_upd(ci, b2); if (b2) { mi = t; break; } }
                            if (mi < 0) mi = k - 1; }
                        asked[idx] = ask; freed[idx] = bit && ask;
                        if (freed[idx]) { mcand[idx] = (int)cand[mi]; st_matched++; }
                        memcpy(q_->cc, cc, sizeof cc); memcpy(q_->st, st, sizeof st); q_->w = w; q_->nci = nci; q_->pm = pm;
                    }
                }
                double b0_ = cm.bits;
                if (MEXCL && oi == 2 && asked[idx] && !freed[idx]) {             /* "not a match" + G and R equal a candidate => B is not that candidate's B */
                    size_t nbs[4] = {x > 0 ? idx - 1 : idx, y > 0 ? idx - W : idx, (x > 0 && y > 0) ? idx - W - 1 : idx, (y > 0 && x < W - 1) ? idx - W + 1 : idx};
                    for (int q = 0; q < 4; q++) { size_t a = nbs[q]; if (a == idx || pl[0][a] != pl[0][idx] || pl[1][a] != pl[1][idx]) continue;
                        int v = pl[2][a] - r, dup = 0; for (int t = 0; t < nex; t++) dup |= ex[t] == v; if (!dup) ex[nex++] = v; }
                    if (nex) st_excl++;
                }
                if (CMATCH && freed[idx]) {                                       /* matched pixel: value copied, layers only learn (no bits) */
                    if (MODE == 2) xv = pl[c][mcand[idx]];
                    e = xv - r; int sv = g_acmode; g_acmode = -1; cm_residual(&cm, cx, A, e, r, (double)(pc - r), (double)sb); g_acmode = sv;
                } else { e = cm_residual_x(&cm, cx, A, e, r, (double)(pc - r), (double)sb, ex, nex); xv = r + e; }
                if (MODE == 2) P[idx] = xv;
                if (pixcost) pixcost[(size_t)c * N + idx] = (float)(cm.bits - b0_);
            }

            /* ---- prediction error: everything adapts ---- */
            float ef = fabsf(xv - pc); efin[idx] = ef;
            for (int k = 0; k < nsubq; k++) esub[k][idx] = fabsf(xv - sp[k]);
            { float a = 1.f / (mo->n + 1.5f); if (a < 1.f / 96) a = 1.f / 96; float eu = SALALL > 0.f ? fminf(ef, SALALL * (mo->scale + 0.3f)) : ef; mo->scale += a * (eu - mo->scale); mo->n++; }
            for (int j = 0; j < nlearn; j++) funnel_learn(path[j], u, nactp, xv - pj[j], un2, j >= nlv ? 1 : (RLMASK >> (nlv - 1 - j)) & 1);
            if (FIELD && land >= 0) {                                             /* landing funnel drifts toward the ball, mouth follows its spread */
                Funnel *f = &fld[land]; float a = 1.f / (f->n + 1.f); if (a < 1.f / 256) a = 1.f / 256;
                float d = sqrtf(dist2(D, f->c));
                for (int k = 0; k < DIM; k++) { f->c[k] += (D[k] - f->c[k]) * a; f->s1[k] += D[k]; f->s2[k] += D[k] * D[k]; }
                f->nsl++;
                float tgt = fmaxf(MERGE * 0.5f, 1.5f * d); f->mouth += (tgt - f->mouth) * 0.01f; f->mouth = clampf(f->mouth, MERGE * 0.5f, MERGE * 2.f);
            }
            if (GMASS) { float *M = gmt + gkey * 256; M[xv] += 1.f; if ((gtot[gkey] += 1.f) > GLIM) { for (int v = 0; v < 256; v++) M[v] *= 0.5f; gtot[gkey] *= 0.5f; } }
            if (hasmat) {                                                         /* the landing material learns this pixel; centre drifts toward it */
                Mat *mm = &mats[mland]; int kk = oi == 1 ? 2 : 3; double z[3] = {1.0, (double)pl[1][idx], (double)pl[0][idx]};
                for (int i = 0; i < 9; i++) mm->S[i] *= MDEC; for (int i = 0; i < 3; i++) mm->St[i] *= MDEC;
                for (int i = 0; i < kk; i++) { for (int j = 0; j < kk; j++) mm->S[i * 3 + j] += z[i] * z[j]; mm->St[i] += z[i] * xv; }
                mm->n++; float a = 1.f / (mm->n < 256 ? mm->n : 256); for (int k = 0; k < 3; k++) mm->c[k] += (md[k] - mm->c[k]) * a;
            }
            if (HIPPO) { float tr = (float)xv - (Wv + Nv) * 0.5f; if (!hcnt[hkey]) hmem[hkey] = tr; else hmem[hkey] += (tr - hmem[hkey]) * 0.35f; if (hcnt[hkey] < 255) hcnt[hkey]++; }
            if (INTER) __atomic_store_n(&prog[oi], (long)idx + 1, __ATOMIC_RELEASE);
            if (REPLAY) { Funnel *f = path[nlearn - 1]; rp_f[x] = f; memcpy(rp_u + (size_t)x * NT, u, sizeof(float) * nact); rp_t[x] = (float)xv - base; }
          }
          if (REPLAY) for (int x = 0; x < W; x++) {                               /* sleep replay of the finished row */
              Funnel *f = rp_f[x]; float *uu = rp_u + (size_t)x * NT, d = 0.f, n2 = 1.f;
              for (int k = 0; k < nact; k++) { d += f->w[k] * uu[k]; n2 += uu[k] * uu[k]; }
              funnel_learn(f, uu, nact, rp_t[x] - d - f->bias, n2, 1); f->n--; st_replay++;
          }
          if (FIELD && SLEEP > 0 && (yi + 1) % SLEEP == 0) field_sleep(fld, &nfld);
          if (PIPE) {
              if (oi == 0 && REGISTER && (yi + 1) % SNAPK == 0 && (yi + 1) / SNAPK < 4096) {   /* freeze G's funnels for R/B to be born from */
                  Funnel **sn = malloc(MAXLV * sizeof(Funnel *));
                  for (int j = 0; j < MAXLV; j++) { sn[j] = malloc(lsize[j] * sizeof(Funnel)); memcpy(sn[j], lv[j], lsize[j] * sizeof(Funnel)); }
                  snaps[(yi + 1) / SNAPK] = sn;
              }
              publish_rows(oi, yi + 1);
          }
        }
        if (getenv("FFIC_DUMP") && oi == 0 && fld) {                           /* explainer data: learned field + landing map of the first plane */
            char pth[512]; snprintf(pth, sizeof pth, "%s.fid", getenv("FFIC_DUMP")); FILE *fo = fopen(pth, "wb"); fwrite(fid, sizeof(short), N, fo); fclose(fo);
            snprintf(pth, sizeof pth, "%s.json", getenv("FFIC_DUMP")); fo = fopen(pth, "w"); fprintf(fo, "[");
            for (int i = 0; i < nfld; i++) { fprintf(fo, "%s{\"i\":%d,\"n\":%d,\"mouth\":%.4f,\"perf\":%.3f,\"c\":[", i ? "," : "", i, fld[i].n, fld[i].mouth, fld[i].perf);
                for (int d = 0; d < DIM; d++) fprintf(fo, "%s%.4f", d ? "," : "", fld[i].c[d]); fprintf(fo, "]}"); }
            fprintf(fo, "]\n"); fclose(fo);
        }
        if (MODE == 1) ac_flush(g_ac);
        if (opt_b) bits[c] = cm.bits + (oi == 0 ? match_bits : 0);
        cm_free(&cm);
        for (int k = 0; k < NSUBX; k++) free(esub[k]);
        free(mats); free(gmt); free(gtot); free(efin); free(actm); free(fid); free(hmem); free(hcnt); free(rp_f); free(rp_u); free(rp_t); free(fld);
        for (int j = 0; j < MAXLV; j++) { if (REGISTER && oi == 0) gl[j] = lv[j]; else if (!(REGISTER && gl[j] == lv[j])) free(lv[j]); } free(mouth);
        pthread_mutex_lock(&mu);
    tot_born += born; if (field_max > tot_field_max) tot_field_max = field_max; tot_field_born += st_field_born; tot_fission += st_fission;
    tot_prune += st_prune; tot_hq += st_hippo_q; tot_hh += st_hippo_hit; tot_replay += st_replay; tot_mat += st_mat_born; tot_matched += st_matched; st_excl_all += st_excl;
    pthread_mutex_unlock(&mu);
    if (PIPE) publish_rows(oi, H);
    if (INTER) __atomic_store_n(&prog[oi], (long)N, __ATOMIC_RELEASE);
}
static void *plane_thread(void *a) { code_plane((int)(long)a); return NULL; }

/* options: letters and key=value (the same list is stored in the file header so the decoder rebuilds the identical model) */
static int parse_opts(int argc, char **argv) {
    for (int i = 0; i < argc; i++) {
        char *a = argv[i], *eq = strchr(a, '=');
        if (eq) {
            float v = atof(eq + 1); int n = (int)(eq - a);
            #define K_(s) (n == (int)strlen(s) && !strncmp(a, s, n))
            if (K_("mu")) MU = v; else if (K_("lam")) LAM = v; else if (K_("P0")) P0 = v; else if (K_("lrm")) LRM = v; else if (K_("gate")) GATE = v;
            else if (K_("rl")) RLMASK = (int)v; else if (K_("clim")) CLIM = (int)v; else if (K_("field")) FIELD = (int)v; else if (K_("sleep")) SLEEP = (int)v;
            else if (K_("bounce")) BOUNCE = (int)v; else if (K_("hippo")) HIPPO = (int)v; else if (K_("lampe")) LAMPE = (int)v; else if (K_("sal")) SAL = v;
            else if (K_("ws")) WST = v; else if (K_("replay")) REPLAY = (int)v; else if (K_("ms")) MS = (int)v; else if (K_("merge")) MERGE = v;
            else if (K_("grav")) GRAV = v; else if (K_("mline")) MLINE = (int)v; else if (K_("matf")) MATF = (int)v; else if (K_("gmass")) GMASS = (int)v; else if (K_("ilace")) ILACE = (int)v; else if (K_("pipe")) PIPE = (int)v; else if (K_("inter")) INTER = (int)v; else if (K_("cmatch")) CMATCH = (int)v; else if (K_("mthr")) MTHR = v; else if (K_("mexcl")) MEXCL = (int)v; else if (K_("minf")) MINF = (int)v; else if (K_("lag")) LAG = clampi((int)v, 2, 1 << 20); else if (K_("snapk")) SNAPK = clampi((int)v, 1, 1 << 20); else if (K_("gmr")) GMR = clampi((int)v, 1, 64); else if (K_("gsig")) GSIG = v; else if (K_("glim")) GLIM = v; else if (K_("mmax")) MMAX = clampi((int)v, 1, MMAXC); else if (K_("mctx")) MCTX = (int)v; else if (K_("mrad")) MRAD = v; else if (K_("mdec")) MDEC = v; else if (K_("matlam")) MATLAM = v; else if (K_("mbw")) MBW = v; else if (K_("mlam")) MLAM = v; else if (K_("mwin")) MWIN = clampi((int)v, 3, NNB); else if (K_("salp")) SALP = v; else if (K_("salmax")) SALMAX = v; else if (K_("salbias")) SALBIAS = v; else if (K_("salall")) SALALL = v; else if (K_("apm")) APM = (int)v; else if (K_("mix2")) MIX2 = (int)v; else if (K_("apmw")) APMW = v; else if (K_("apmr")) APMR = v; else if (K_("register")) REGISTER = (int)v; else if (K_("curve")) CURVE = (int)v;
            else if (K_("kres")) KRES = clampi((int)v, 1, KMAX); else if (K_("fmax")) FMAX = clampi((int)v, 8, FMAXC);
            else { fprintf(stderr, "unknown option %s\n", a); return 0; }
            continue;
        }
        opt_l = opt_t = opt_v = opt_p = opt_c = opt_e = opt_w = opt_r = opt_b = opt_f = 0;
        if (strcmp(a, "none"))
            for (char *c = a; *c; c++) { switch (*c) { case 'l': opt_l = 1; break; case 't': opt_t = 1; break; case 'v': opt_v = 1; break;
                case 'p': opt_p = 1; break; case 'c': opt_c = 1; break; case 'e': opt_e = 1; break; case 'w': opt_w = 1; break; case 'r': opt_r = 1; break; case 'b': opt_b = 1; break; case 'f': opt_f = 1; break; } }
    }
    return 1;
}

int main(int argc, char **argv) {
    const char *outp = NULL, *decp = NULL; char opts[4096] = ""; char *ov[256]; int oc = 0;
    if (argc >= 4 && !strcmp(argv[1], "-d")) { decp = argv[2]; outp = argv[3]; MODE = 2; }
    else if (argc < 4) { fprintf(stderr, "usage: ffic file.rgb W H [flags] [key=value...] [out=file.ffic]\n       ffic -d file.ffic out.rgb\n"); return 1; }
    unsigned char *raw;
    if (MODE == 2) {                                                          /* header: magic, W, H, options, three stream lengths, streams */
        FILE *fi = fopen(decp, "rb"); if (!fi) { perror("open"); return 1; }
        char mg[6]; unsigned hw[2], ol, sl[3];
        if (fread(mg, 1, 6, fi) != 6 || memcmp(mg, "FFIC1", 6) || fread(hw, 4, 2, fi) != 2 || fread(&ol, 4, 1, fi) != 1 || ol >= sizeof opts) { fprintf(stderr, "not an FFIC file\n"); return 1; }
        W = (int)hw[0]; H = (int)hw[1];
        if (fread(opts, 1, ol, fi) != ol || fread(sl, 4, 3, fi) != 3) { fprintf(stderr, "truncated header\n"); return 1; } opts[ol] = 0;
        for (int i = 0; i < 3; i++) { acs[i].n = acs[i].cap = sl[i]; acs[i].buf = malloc(sl[i] + 1); if (fread(acs[i].buf, 1, sl[i], fi) != sl[i]) { fprintf(stderr, "truncated stream\n"); return 1; } }
        fclose(fi);
        for (char *t = strtok(opts, " "); t && oc < 256; t = strtok(NULL, " ")) ov[oc++] = t;
        if (!parse_opts(oc, ov)) return 1;
        N = (size_t)W * H; raw = calloc(N * 3, 1);
    } else {
        W = atoi(argv[2]); H = atoi(argv[3]);
        for (int i = 4; i < argc; i++) {
            if (!strncmp(argv[i], "out=", 4)) { outp = argv[i] + 4; MODE = 1; continue; }
            ov[oc++] = argv[i]; if (strlen(opts) + strlen(argv[i]) + 2 < sizeof opts) { if (*opts) strcat(opts, " "); strcat(opts, argv[i]); }
        }
        if (!parse_opts(oc, ov)) return 1;
        FILE *fp = fopen(argv[1], "rb"); if (!fp) { perror("open"); return 1; }
        N = (size_t)W * H; raw = malloc(N * 3);
        if (fread(raw, 1, N * 3, fp) != N * 3) { fprintf(stderr, "short read\n"); return 1; } fclose(fp);
    }
    if (MODE && !opt_b) { fprintf(stderr, "a real bitstream needs the context-mixing coder (flag b)\n"); return 1; }
    if (MODE && INTER && opt_f) { fprintf(stderr, "future taps (f) are not causal with inter=1\n"); return 1; }
    pl[0] = pl[1] = pl[2] = NULL;
    for (int c = 0; c < 3; c++) { pl[c] = malloc(N * sizeof(int)); pfin[c] = calloc(N, sizeof(float)); for (size_t i = 0; i < N; i++) pl[c][i] = raw[i * 3 + c]; }
    clock_t t0 = clock(); struct timespec w0, w1; clock_gettime(CLOCK_MONOTONIC, &w0);
    pixcost = getenv("FFIC_PIXCOST") ? calloc(3 * N, sizeof(float)) : NULL;
    nsub = NSUB + (HIPPO ? 1 : 0) + (CURVE ? 1 : 0); icurve = NSUB + (HIPPO ? 1 : 0);
    if (INTER) { PIPE = 1; ILACE = 0; if (CMATCH) { matched = calloc(N, 1); freed = calloc(N, 1); asked = calloc(N, 1); mcand = calloc(N, sizeof(int)); midx = calloc(N, 1); for (int i = 0; i < 16; i++) for (int q = 0; q < 6; q++) mc_w[i][q] = q < 4 ? 0.3f : 0; for (int a = 0; a < 4; a++) for (int i = 0; i < 128; i++) mc_flag[a][i].p = 0.3f; for (int i = 0; i < 256; i++) mc_idx[i].p = 0.5f; } }
    else CMATCH = 0;                                                      /* whole-colour match needs pixel-by-pixel order */
    if (PIPE) {                                                             /* staggered colour pipeline: one thread per layer */
        pthread_t th[3]; for (int oi = 0; oi < 3; oi++) pthread_create(&th[oi], NULL, plane_thread, (void *)(long)oi);
        for (int oi = 0; oi < 3; oi++) pthread_join(th[oi], NULL);
    } else for (int oi = 0; oi < 3; oi++) code_plane(oi);
    clock_gettime(CLOCK_MONOTONIC, &w1); double wall = (w1.tv_sec - w0.tv_sec) * 1000.0 + (w1.tv_nsec - w0.tv_nsec) / 1e6;
    long long fbytes = 0;
    if (MODE == 1) {
        FILE *fo = fopen(outp, "wb"); if (!fo) { perror("out"); return 1; }
        unsigned hw[2] = {(unsigned)W, (unsigned)H}, ol = (unsigned)strlen(opts), sl[3] = {(unsigned)acs[0].n, (unsigned)acs[1].n, (unsigned)acs[2].n};
        fwrite("FFIC1", 1, 6, fo); fwrite(hw, 4, 2, fo); fwrite(&ol, 4, 1, fo); fwrite(opts, 1, ol, fo); fwrite(sl, 4, 3, fo);
        for (int i = 0; i < 3; i++) fwrite(acs[i].buf, 1, acs[i].n, fo);
        fbytes = ftell(fo); fclose(fo);
    }
    if (MODE == 2) {
        for (int c = 0; c < 3; c++) for (size_t i = 0; i < N; i++) raw[i * 3 + c] = (unsigned char)pl[c][i];
        FILE *fo = fopen(outp, "wb"); if (!fo) { perror("out"); return 1; } fwrite(raw, 1, N * 3, fo); fclose(fo);
    }
    if (pixcost) { FILE *fo = fopen(getenv("FFIC_PIXCOST"), "wb"); fwrite(pixcost, sizeof(float), 3 * N, fo); fclose(fo); }
    double ms = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC, tot = bits[0] + bits[1] + bits[2];
    printf("file=%lld bytes=%.0f bpp=%.4f ms=%.0f wall=%.0f funnels=%lld R=%.0f G=%.0f B=%.0f field_born=%lld field_max=%d fission=%lld prune=%lld hippo_hit=%.4f replay=%lld materials=%lld matched=%.4f excl=%.4f\n",
           fbytes, tot / 8, tot / N, ms, wall, tot_born, bits[0] / 8, bits[1] / 8, bits[2] / 8, tot_field_born, tot_field_max, tot_fission, tot_prune,
           tot_hq ? (double)tot_hh / tot_hq : 0.0, tot_replay, tot_mat, (double)tot_matched / N, (double)st_excl_all / N);
    return 0;
}
