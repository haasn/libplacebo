/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <math.h>

#include "common.h"

#include <libplacebo/gamut_mapping.h>

bool pl_gamut_map_params_equal(const struct pl_gamut_map_params *a,
                               const struct pl_gamut_map_params *b)
{
    return a->function      == b->function      &&
           a->min_luma      == b->min_luma      &&
           a->max_luma      == b->max_luma      &&
           a->chroma_margin == b->chroma_margin &&
           a->lut_size_I    == b->lut_size_I    &&
           a->lut_size_C    == b->lut_size_C    &&
           a->lut_size_h    == b->lut_size_h    &&
           a->lut_stride    == b->lut_stride    &&
           pl_raw_primaries_equal(&a->input_gamut,  &b->input_gamut) &&
           pl_raw_primaries_equal(&a->output_gamut, &b->output_gamut);
}

#define FUN(params) (params->function ? *params->function : pl_gamut_map_clip)

static void noop(float *lut, const struct pl_gamut_map_params *params);
bool pl_gamut_map_params_noop(const struct pl_gamut_map_params *params)
{
    if (FUN(params).map == &noop)
        return true;

    struct pl_raw_primaries src = params->input_gamut, dst = params->output_gamut;
    bool need_map = !pl_primaries_superset(&dst, &src);
    need_map |= !pl_cie_xy_equal(&src.white, &dst.white);
    need_map |= params->chroma_margin > 1.0f;

    if (FUN(params).bidirectional) {
        need_map |= !pl_raw_primaries_equal(&dst, &src);
        need_map |= params->chroma_margin && params->chroma_margin < 1.0f;
    }

    return !need_map;
}

// For some minimal type safety, and code cleanliness
struct RGB {
    float R, G, B;
};

struct IPT {
    float I, P, T;
};

struct ICh {
    float I, C, h;
};

static inline struct ICh ipt2ich(struct IPT c)
{
    return (struct ICh) {
        .I = c.I,
        .C = sqrtf(c.P * c.P + c.T * c.T),
        .h = atan2f(c.T, c.P),
    };
}

static inline struct IPT ich2ipt(struct ICh c)
{
    return (struct IPT) {
        .I = c.I,
        .P = c.C * cosf(c.h),
        .T = c.C * sinf(c.h),
    };
}

static const float PQ_M1 = 2610./4096 * 1./4,
                   PQ_M2 = 2523./4096 * 128,
                   PQ_C1 = 3424./4096,
                   PQ_C2 = 2413./4096 * 32,
                   PQ_C3 = 2392./4096 * 32;

enum { PQ_LUT_SIZE = 256 };
static const float pq_eotf_lut[PQ_LUT_SIZE+1] = {
    0.000000e+0f, 4.337253e-8f, 1.481642e-7f, 3.125027e-7f, 5.396092e-7f, 8.338515e-7f, 1.200175e-6f, 1.643925e-6f,
    2.170776e-6f, 2.786705e-6f, 3.497978e-6f, 4.311152e-6f, 5.233072e-6f, 6.270884e-6f, 7.432037e-6f, 8.724296e-6f,
    1.015575e-5f, 1.173483e-5f, 1.347029e-5f, 1.537128e-5f, 1.744730e-5f, 1.970824e-5f, 2.216438e-5f, 2.482644e-5f,
    2.770553e-5f, 3.081323e-5f, 3.416159e-5f, 3.776310e-5f, 4.163077e-5f, 4.577811e-5f, 5.021916e-5f, 5.496852e-5f,
    6.004133e-5f, 6.545334e-5f, 7.122088e-5f, 7.736095e-5f, 8.389115e-5f, 9.082981e-5f, 9.819590e-5f, 1.060092e-4f,
    1.142900e-4f, 1.230598e-4f, 1.323404e-4f, 1.421548e-4f, 1.525267e-4f, 1.634807e-4f, 1.750423e-4f, 1.872380e-4f,
    2.000953e-4f, 2.136425e-4f, 2.279094e-4f, 2.429263e-4f, 2.587252e-4f, 2.753389e-4f, 2.928015e-4f, 3.111483e-4f,
    3.304160e-4f, 3.506427e-4f, 3.718675e-4f, 3.941314e-4f, 4.174766e-4f, 4.419468e-4f, 4.675876e-4f, 4.944458e-4f,
    5.225702e-4f, 5.520112e-4f, 5.828212e-4f, 6.150541e-4f, 6.487662e-4f, 6.840155e-4f, 7.208622e-4f, 7.593686e-4f,
    7.995992e-4f, 8.416208e-4f, 8.855027e-4f, 9.313166e-4f, 9.791366e-4f, 1.029040e-3f, 1.081105e-3f, 1.135416e-3f,
    1.192056e-3f, 1.251116e-3f, 1.312685e-3f, 1.376859e-3f, 1.443736e-3f, 1.513417e-3f, 1.586007e-3f, 1.661615e-3f,
    1.740353e-3f, 1.822339e-3f, 1.907693e-3f, 1.996539e-3f, 2.089006e-3f, 2.185230e-3f, 2.285348e-3f, 2.389503e-3f,
    2.497845e-3f, 2.610526e-3f, 2.727706e-3f, 2.849549e-3f, 2.976226e-3f, 3.107913e-3f, 3.244792e-3f, 3.387053e-3f,
    3.534891e-3f, 3.688509e-3f, 3.848116e-3f, 4.013928e-3f, 4.186172e-3f, 4.365078e-3f, 4.550888e-3f, 4.743851e-3f,
    4.944223e-3f, 5.152273e-3f, 5.368276e-3f, 5.592518e-3f, 5.825295e-3f, 6.066914e-3f, 6.317691e-3f, 6.577954e-3f,
    6.848045e-3f, 7.128314e-3f, 7.419125e-3f, 7.720856e-3f, 8.033897e-3f, 8.358652e-3f, 8.695540e-3f, 9.044993e-3f,
    9.407463e-3f, 9.783408e-3f, 1.017331e-2f, 1.057768e-2f, 1.099701e-2f, 1.143185e-2f, 1.188275e-2f, 1.235028e-2f,
    1.283503e-2f, 1.333762e-2f, 1.385869e-2f, 1.439888e-2f, 1.495889e-2f, 1.553942e-2f, 1.614119e-2f, 1.676497e-2f,
    1.741155e-2f, 1.808173e-2f, 1.877635e-2f, 1.949629e-2f, 2.024245e-2f, 2.101577e-2f, 2.181722e-2f, 2.264779e-2f,
    2.350853e-2f, 2.440052e-2f, 2.532488e-2f, 2.628276e-2f, 2.727536e-2f, 2.830393e-2f, 2.936975e-2f, 3.047416e-2f,
    3.161853e-2f, 3.280432e-2f, 3.403299e-2f, 3.530610e-2f, 3.662523e-2f, 3.799205e-2f, 3.940826e-2f, 4.087566e-2f,
    4.239608e-2f, 4.397143e-2f, 4.560371e-2f, 4.729496e-2f, 4.904732e-2f, 5.086299e-2f, 5.274428e-2f, 5.469357e-2f,
    5.671331e-2f, 5.880607e-2f, 6.097450e-2f, 6.322136e-2f, 6.554950e-2f, 6.796189e-2f, 7.046161e-2f, 7.305184e-2f,
    7.573590e-2f, 7.851722e-2f, 8.139937e-2f, 8.438606e-2f, 8.748113e-2f, 9.068855e-2f, 9.401249e-2f, 9.745722e-2f,
    1.010272e-1f, 1.047271e-1f, 1.085617e-1f, 1.125361e-1f, 1.166553e-1f, 1.209248e-1f, 1.253501e-1f, 1.299372e-1f,
    1.346920e-1f, 1.396209e-1f, 1.447302e-1f, 1.500269e-1f, 1.555179e-1f, 1.612105e-1f, 1.671124e-1f, 1.732314e-1f,
    1.795759e-1f, 1.861542e-1f, 1.929753e-1f, 2.000484e-1f, 2.073831e-1f, 2.149893e-1f, 2.228775e-1f, 2.310585e-1f,
    2.395434e-1f, 2.483439e-1f, 2.574722e-1f, 2.669409e-1f, 2.767632e-1f, 2.869526e-1f, 2.975236e-1f, 3.084908e-1f,
    3.198697e-1f, 3.316764e-1f, 3.439276e-1f, 3.566406e-1f, 3.698336e-1f, 3.835254e-1f, 3.977357e-1f, 4.124850e-1f,
    4.277946e-1f, 4.436866e-1f, 4.601843e-1f, 4.773117e-1f, 4.950940e-1f, 5.135573e-1f, 5.327289e-1f, 5.526374e-1f,
    5.733122e-1f, 5.947845e-1f, 6.170864e-1f, 6.402515e-1f, 6.643150e-1f, 6.893134e-1f, 7.152848e-1f, 7.422693e-1f,
    7.703082e-1f, 7.994449e-1f, 8.297248e-1f, 8.611951e-1f, 8.939051e-1f, 9.279063e-1f, 9.632526e-1f, 1.000000e+0f,
    1.0f, // extra padding to avoid out of bounds access
};

static inline float pq_eotf(float x)
{
    float idxf  = fminf(fmaxf(x, 0.0f), 1.0f) * (PQ_LUT_SIZE - 1);
    int ipart   = floorf(idxf);
    float fpart = idxf - ipart;
    return PL_MIX(pq_eotf_lut[ipart], pq_eotf_lut[ipart + 1], fpart);
}

static inline float pq_oetf(float x)
{
    x = powf(fmaxf(x, 0.0f), PQ_M1);
    x = (PQ_C1 + PQ_C2 * x) / (1.0f + PQ_C3 * x);
    return powf(x, PQ_M2);
}

// Helper struct containing pre-computed cached values describing a gamut
struct gamut {
    pl_matrix3x3 lms2rgb;
    pl_matrix3x3 rgb2lms;
    float min_luma, max_luma;   // pq
    float min_rgb,  max_rgb;    // 10k normalized
    struct ICh *peak_cache;     // 1-item cache for computed peaks (per hue)
};

static void get_gamuts(struct gamut *dst, struct gamut *src,
                       const struct pl_gamut_map_params *params)
{
    const float epsilon = 1e-6;
    struct gamut base = {
        .min_luma = params->min_luma,
        .max_luma = params->max_luma,
        .min_rgb  = pq_eotf(params->min_luma) - epsilon,
        .max_rgb  = pq_eotf(params->max_luma) + epsilon,
    };

    if (dst) {
        static _Thread_local struct ICh dst_cache;
        *dst = base;
        dst->lms2rgb = dst->rgb2lms = pl_ipt_rgb2lms(&params->output_gamut);
        dst->peak_cache = &dst_cache;
        pl_matrix3x3_invert(&dst->lms2rgb);
    }

    if (src) {
        static _Thread_local struct ICh src_cache;
        *src = base;
        src->lms2rgb = src->rgb2lms = pl_ipt_rgb2lms(&params->input_gamut);
        src->peak_cache = &src_cache;
        pl_matrix3x3_invert(&src->lms2rgb);
    }
}

static inline struct IPT rgb2ipt(struct RGB c, struct gamut gamut)
{
    const float L = gamut.rgb2lms.m[0][0] * c.R +
                    gamut.rgb2lms.m[0][1] * c.G +
                    gamut.rgb2lms.m[0][2] * c.B;
    const float M = gamut.rgb2lms.m[1][0] * c.R +
                    gamut.rgb2lms.m[1][1] * c.G +
                    gamut.rgb2lms.m[1][2] * c.B;
    const float S = gamut.rgb2lms.m[2][0] * c.R +
                    gamut.rgb2lms.m[2][1] * c.G +
                    gamut.rgb2lms.m[2][2] * c.B;
    const float Lp = pq_oetf(L);
    const float Mp = pq_oetf(M);
    const float Sp = pq_oetf(S);
    return (struct IPT) {
        .I = 0.4000f * Lp + 0.4000f * Mp + 0.2000f * Sp,
        .P = 4.4550f * Lp - 4.8510f * Mp + 0.3960f * Sp,
        .T = 0.8056f * Lp + 0.3572f * Mp - 1.1628f * Sp,
    };
}

static inline struct RGB ipt2rgb(struct IPT c, struct gamut gamut)
{
    const float Lp = c.I + 0.0975689f * c.P + 0.205226f * c.T;
    const float Mp = c.I - 0.1138760f * c.P + 0.133217f * c.T;
    const float Sp = c.I + 0.0326151f * c.P - 0.676887f * c.T;
    const float L = pq_eotf(Lp);
    const float M = pq_eotf(Mp);
    const float S = pq_eotf(Sp);
    return (struct RGB) {
        .R = gamut.lms2rgb.m[0][0] * L +
             gamut.lms2rgb.m[0][1] * M +
             gamut.lms2rgb.m[0][2] * S,
        .G = gamut.lms2rgb.m[1][0] * L +
             gamut.lms2rgb.m[1][1] * M +
             gamut.lms2rgb.m[1][2] * S,
        .B = gamut.lms2rgb.m[2][0] * L +
             gamut.lms2rgb.m[2][1] * M +
             gamut.lms2rgb.m[2][2] * S,
    };
}

static inline bool ingamut(struct IPT c, struct gamut gamut)
{
    const float Lp = c.I + 0.0975689f * c.P + 0.205226f * c.T;
    const float Mp = c.I - 0.1138760f * c.P + 0.133217f * c.T;
    const float Sp = c.I + 0.0326151f * c.P - 0.676887f * c.T;
    if (Lp < gamut.min_luma || Lp > gamut.max_luma ||
        Mp < gamut.min_luma || Mp > gamut.max_luma ||
        Sp < gamut.min_luma || Sp > gamut.max_luma)
    {
        // Early exit for values outside legal LMS range
        return false;
    }

    const float L = pq_eotf(Lp);
    const float M = pq_eotf(Mp);
    const float S = pq_eotf(Sp);
    struct RGB rgb = {
        .R = gamut.lms2rgb.m[0][0] * L +
             gamut.lms2rgb.m[0][1] * M +
             gamut.lms2rgb.m[0][2] * S,
        .G = gamut.lms2rgb.m[1][0] * L +
             gamut.lms2rgb.m[1][1] * M +
             gamut.lms2rgb.m[1][2] * S,
        .B = gamut.lms2rgb.m[2][0] * L +
             gamut.lms2rgb.m[2][1] * M +
             gamut.lms2rgb.m[2][2] * S,
    };
    return rgb.R >= gamut.min_rgb && rgb.R <= gamut.max_rgb &&
           rgb.G >= gamut.min_rgb && rgb.G <= gamut.max_rgb &&
           rgb.B >= gamut.min_rgb && rgb.B <= gamut.max_rgb;
}

void pl_gamut_map_generate(float *out, const struct pl_gamut_map_params *params)
{
    float *in = out;
    for (int h = 0; h < params->lut_size_h; h++) {
        for (int C = 0; C < params->lut_size_C; C++) {
            for (int I = 0; I < params->lut_size_I; I++) {
                float Ix = (float) I / (params->lut_size_I - 1);
                float Cx = (float) C / (params->lut_size_C - 1);
                float hx = (float) h / (params->lut_size_h - 1);
                struct IPT ipt = ich2ipt((struct ICh) {
                    .I = PL_MIX(params->min_luma, params->max_luma, Ix),
                    .C = PL_MIX(0.0f, 0.5f, Cx),
                    .h = PL_MIX(-M_PI, M_PI, hx),
                });
                in[0] = ipt.I;
                in[1] = ipt.P;
                in[2] = ipt.T;
                in += params->lut_stride;
            }
        }
    }

    FUN(params).map(out, params);
}

void pl_gamut_map_sample(float x[3], const struct pl_gamut_map_params *params)
{
    struct pl_gamut_map_params fixed = *params;
    fixed.lut_size_I = fixed.lut_size_C = fixed.lut_size_h = 1;
    fixed.lut_stride = 3;

    FUN(params).map(x, &fixed);
}

#define LUT_SIZE(p) (p->lut_size_I * p->lut_size_C * p->lut_size_h * p->lut_stride)
#define FOREACH_LUT(lut, C)                                                     \
    for (struct IPT *_i = (struct IPT *) lut,                                   \
                    *_end = (struct IPT *) (lut + LUT_SIZE(params)),            \
                    C;                                                          \
         _i < _end && ( C = *_i, 1 );                                           \
         *_i = C, _i = (struct IPT *) ((float *) _i + params->lut_stride))

// Something like PL_MIX(base, c, x) but follows an exponential curve, note
// that this can be used to extend 'c' outwards for x > 1
static inline struct ICh mix_exp(struct ICh c, float x, float gamma, float base)
{
    return (struct ICh) {
        .I = base + (c.I - base) * powf(x, gamma),
        .C = c.C * x,
        .h = c.h,
    };
}

// Drop gamma for colors approaching black and achromatic to avoid numerical
// instabilities, and excessive brightness boosting of grain, while also
// strongly boosting gamma for values exceeding the target peak
static inline float scale_gamma(float gamma, struct ICh ich, struct ICh peak,
                                struct gamut gamut)
{
    const float Imin = gamut.min_luma;
    const float Irel = fmaxf((ich.I - Imin) / (peak.I - Imin), 0.0f);
    return gamma * powf(Irel, 3) * fminf(ich.C / peak.C, 1.0f);
}

static const float maxDelta = 4e-3;

// Find gamut intersection using specified bounds
static inline struct ICh
desat_bounded(float I, float h, float Cmin, float Cmax, struct gamut gamut)
{
    const float maxDI = I * maxDelta;
    struct ICh res = { .I = I, .C = (Cmin + Cmax) / 2, .h = h };
    do {
        if (ingamut(ich2ipt(res), gamut)) {
            Cmin = res.C;
        } else {
            Cmax = res.C;
        }
        res.C = (Cmin + Cmax) / 2;
    } while (Cmax - Cmin > maxDI);

    return res;
}

// Finds maximally saturated in-gamut color (for given hue)
static inline struct ICh saturate(float hue, struct gamut gamut)
{
    if (gamut.peak_cache->I && fabsf(gamut.peak_cache->h - hue) < 1e-3)
        return *gamut.peak_cache;

    static const float invphi = 0.6180339887498948f;
    static const float invphi2 = 0.38196601125010515f;

    struct ICh lo = { .I = gamut.min_luma, .h = hue };
    struct ICh hi = { .I = gamut.max_luma, .h = hue };
    float de = hi.I - lo.I;
    struct ICh a = { .I = lo.I + invphi2 * de };
    struct ICh b = { .I = lo.I + invphi  * de };
    a = desat_bounded(a.I, hue, 0.0f, 0.5f, gamut);
    b = desat_bounded(b.I, hue, 0.0f, 0.5f, gamut);

    while (de > maxDelta) {
        de *= invphi;
        if (a.C > b.C) {
            hi = b;
            b = a;
            a.I = lo.I + invphi2 * de;
            a = desat_bounded(a.I, hue, lo.C - maxDelta, 0.5f, gamut);
        } else {
            lo = a;
            a = b;
            b.I = lo.I + invphi * de;
            b = desat_bounded(b.I, hue, hi.C - maxDelta, 0.5f, gamut);
        }
    }

    struct ICh peak = a.C > b.C ? a : b;
    *gamut.peak_cache = peak;
    return peak;
}

// Clip a color along the exponential curve given by `gamma`
static inline struct IPT
clip_gamma(struct IPT ipt, float gamma, struct gamut gamut)
{
    if (ipt.I <= gamut.min_luma)
        return (struct IPT) { .I = gamut.min_luma };
    if (ingamut(ipt, gamut))
        return ipt;

    struct ICh ich = ipt2ich(ipt);
    if (!gamma)
        return ich2ipt(desat_bounded(ich.I, ich.h, 0.0f, ich.C, gamut));

    const float maxDI = ich.I * maxDelta;
    struct ICh peak = saturate(ich.h, gamut);
    gamma = scale_gamma(gamma, ich, peak, gamut);
    float lo = 0.0f, hi = 1.0f, x = 0.5f;
    do {
        struct ICh test = mix_exp(ich, x, gamma, peak.I);
        if (ingamut(ich2ipt(test), gamut)) {
            lo = x;
        } else {
            hi = x;
        }
        x = (lo + hi) / 2.0f;
    } while (hi - lo > maxDI);

    return ich2ipt(mix_exp(ich, x, gamma, peak.I));
}

static const float perceptual_gamma    = 1.80f;
static const float perceptual_knee     = 0.70f;

static int cmp_float(const void *a, const void *b)
{
    float fa = *(const float*) a;
    float fb = *(const float*) b;
    return PL_CMP(fa, fb);
}

static float wrap(float h)
{
    if (h > M_PI) {
        return h - 2 * M_PI;
    } else if (h < -M_PI) {
        return h + 2 * M_PI;
    } else {
        return h;
    }
}

static void perceptual(float *lut, const struct pl_gamut_map_params *params)
{
    struct gamut dst, src;
    get_gamuts(&dst, &src, params);

    const float O = pq_eotf(params->min_luma), X = pq_eotf(params->max_luma);
    const float M = (O + X) / 2.0f;
    const struct RGB refpoints[] = {
        {X, O, O}, {O, X, O}, {O, O, X},
        {O, X, X}, {X, O, X}, {X, X, O},
        {O, X, M}, {X, O, M}, {X, M, O},
        {O, M, X}, {M, O, X}, {M, X, O},
    };

    enum {
        S = PL_ARRAY_SIZE(refpoints),
        N = S + 2, // +2 for the endpoints
    };

    struct { float hue, delta; } hueshift[N];
    for (int i = 0; i < S; i++) {
        struct ICh ich_src = ipt2ich(rgb2ipt(refpoints[i], src));
        struct ICh ich_dst = ipt2ich(rgb2ipt(refpoints[i], dst));
        hueshift[i+1].hue = ich_src.h;
        hueshift[i+1].delta = wrap(ich_dst.h - ich_src.h);
    }

    // Sort and wrap endpoints
    qsort(hueshift + 1, S, sizeof(*hueshift), cmp_float);
    hueshift[0]   = hueshift[S];
    hueshift[S+1] = hueshift[1];
    hueshift[0].hue   -= 2 * M_PI;
    hueshift[S+1].hue += 2 * M_PI;

    // Construction of cubic spline coefficients
    float dh[N], dddh[N], K[N] = {0}, tmp[N][N] = {0};
    for (int i = N - 1; i > 0; i--) {
        dh[i-1] = hueshift[i].hue - hueshift[i-1].hue;
        dddh[i] = (hueshift[i].delta - hueshift[i-1].delta) / dh[i-1];
    }
    for (int i = 1; i < N - 1; i++) {
        tmp[i][i] = 2 * (dh[i-1] + dh[i]);
        if (i != 1)
            tmp[i][i-1] = tmp[i-1][i] = dh[i-1];
        tmp[i][N-1] = 6 * (dddh[i+1] - dddh[i]);
    }
    for (int i = 1; i < N - 2; i++) {
        const float q = (tmp[i+1][i] / tmp[i][i]);
        for (int j = 1; j <= N - 1; j++)
            tmp[i+1][j] -= q * tmp[i][j];
    }
    for (int i = N - 2; i > 0; i--) {
        float sum = 0.0f;
        for (int j = i; j <= N - 2; j++)
            sum += tmp[i][j] * K[j];
        K[i] = (tmp[i][N-1] - sum) / tmp[i][i];
    }

    float prev_hue = -10.0f, prev_delta = 0.0f;
    FOREACH_LUT(lut, ipt) {

        if (ipt.I <= dst.min_luma) {
            ipt.P = ipt.T = 0.0f;
            continue;
        }

        struct ICh ich = ipt2ich(ipt);
        if (ich.C <= 1e-2f)
            continue; // Fast path for achromatic colors

        // Determine perceptual hue shift delta by interpolation of refpoints
        float delta = 0.0f;
        if (fabsf(ich.h - prev_hue) < 1e-6f) {
            delta = prev_delta;
        } else {
            for (int i = 0; i < N - 1; i++) {
                if (hueshift[i+1].hue > ich.h) {
                    pl_assert(hueshift[i].hue <= ich.h);
                    float a = (K[i+1] - K[i]) / (6 * dh[i]);
                    float b = K[i] / 2;
                    float c = dddh[i+1] - (2 * dh[i] * K[i] + K[i+1] * dh[i]) / 6;
                    float d = hueshift[i].delta;
                    float x = ich.h - hueshift[i].hue;
                    delta = ((a * x + b) * x + c) * x + d;
                    prev_delta = delta;
                    prev_hue = ich.h;
                    break;
                }
            }
        }

        const float margin = PL_DEF(params->chroma_margin, 1.0f);
        if (fabsf(delta) >= 1e-3f) {
            struct ICh src_border = desat_bounded(ich.I, ich.h, 0.0f, 0.5f, src);
            struct ICh dst_border = desat_bounded(ich.I, ich.h, 0.0f, 0.5f, dst);
            ich.h += delta * pl_smoothstep(dst_border.C, src_border.C * margin, ich.C);
        }

        // Determine intersections with source and target gamuts
        struct ICh source = saturate(ich.h, src);
        struct ICh target = saturate(ich.h, dst);
        const float gamma = scale_gamma(perceptual_gamma, ich, target, dst);

        float lo = 0.0f, x = 1.0f, hi = 1.0f / perceptual_knee + 3 * maxDelta;
        do {
            struct ICh test = mix_exp(ich, x, gamma, target.I);
            if (ingamut(ich2ipt(test), dst)) {
                lo = x;
            } else {
                hi = x;
            }
            x = (lo + hi) / 2.0f;
        } while (hi - lo > maxDelta);

        // Apply simple Mobius tone mapping curve
        const float j = PL_MIX(1.0f, perceptual_knee, ich.C / 0.5f);
        const float peak = margin * fmaxf(source.C / target.C, 1.0f);
        float xx = 1.0f / x;
        if (j < 1.0f && peak >= 1.0f) {
            const float a = -j*j * (peak - 1.0f) / (j*j - 2.0f * j + peak);
            const float b = (j*j - 2.0f * j * peak + peak) /
                            fmaxf(1e-6f, peak - 1.0f);
            const float k = (b*b + 2.0f * b*j + j*j) / (b - a);
            xx = fminf(xx, peak);
            xx = xx <= j ? xx : k * (xx + a) / (xx + b);
        }

        ipt = ich2ipt(mix_exp(ich, xx * x, gamma, target.I));
    }
}

const struct pl_gamut_map_function pl_gamut_map_perceptual = {
    .name = "perceptual",
    .description = "Perceptual soft-clip",
    .map = perceptual,
};

static void relative(float *lut, const struct pl_gamut_map_params *params)
{
    struct gamut dst;
    get_gamuts(&dst, NULL, params);

    FOREACH_LUT(lut, ipt)
        ipt = clip_gamma(ipt, perceptual_gamma, dst);
}

const struct pl_gamut_map_function pl_gamut_map_relative = {
    .name = "relative",
    .description = "Colorimetric clip",
    .map = relative,
};

static void desaturate(float *lut, const struct pl_gamut_map_params *params)
{
    struct gamut dst;
    get_gamuts(&dst, NULL, params);

    FOREACH_LUT(lut, ipt)
        ipt = clip_gamma(ipt, 0.0f, dst);
}

const struct pl_gamut_map_function pl_gamut_map_desaturate = {
    .name = "desaturate",
    .description = "Desaturating clip",
    .map = desaturate,
};

static void saturation(float *lut, const struct pl_gamut_map_params *params)
{
    struct gamut dst, src;
    get_gamuts(&dst, &src, params);

    FOREACH_LUT(lut, ipt)
        ipt = rgb2ipt(ipt2rgb(ipt, src), dst);
}

const struct pl_gamut_map_function pl_gamut_map_saturation = {
    .name = "saturation",
    .description = "Saturation mapping",
    .bidirectional = true,
    .map = saturation,
};

static void absolute(float *lut, const struct pl_gamut_map_params *params)
{
    struct gamut dst;
    get_gamuts(&dst, NULL, params);
    pl_matrix3x3 m = pl_get_adaptation_matrix(params->output_gamut.white,
                                              params->input_gamut.white);

    FOREACH_LUT(lut, ipt) {
        struct RGB rgb = ipt2rgb(ipt, dst);
        pl_matrix3x3_apply(&m, (float *) &rgb);
        ipt = rgb2ipt(rgb, dst);
        ipt = clip_gamma(ipt, perceptual_gamma, dst);
    }
}

const struct pl_gamut_map_function pl_gamut_map_absolute = {
    .name = "absolute",
    .description = "Absolute colorimetric clip",
    .map = absolute,
};

static void highlight(float *lut, const struct pl_gamut_map_params *params)
{
    struct gamut dst;
    get_gamuts(&dst, NULL, params);

    FOREACH_LUT(lut, ipt) {
        if (!ingamut(ipt, dst)) {
            ipt.I += 0.1f;
            ipt.P *= -1.2f;
            ipt.T *= -1.2f;
        }
    }
}

const struct pl_gamut_map_function pl_gamut_map_highlight = {
    .name = "highlight",
    .description = "Highlight out-of-gamut pixels",
    .map = highlight,
};

static void linear(float *lut, const struct pl_gamut_map_params *params)
{
    struct gamut dst, src;
    get_gamuts(&dst, &src, params);

    float gain = 1.0f;
    for (float hue = -M_PI; hue < M_PI; hue += 0.1f)
        gain = fminf(gain, saturate(hue, dst).C / saturate(hue, src).C);

    FOREACH_LUT(lut, ipt) {
        struct ICh ich = ipt2ich(ipt);
        ich.C *= gain;
        ipt = ich2ipt(ich);
    }
}

const struct pl_gamut_map_function pl_gamut_map_linear = {
    .name = "linear",
    .description = "Linear desaturate",
    .map = linear,
};

static void darken(float *lut, const struct pl_gamut_map_params *params)
{
    struct gamut dst, src;
    get_gamuts(&dst, &src, params);

    static const struct RGB points[6] = {
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
        {0, 1, 1}, {1, 0, 1}, {1, 1, 0},
    };

    float gain = 1.0f;
    for (int i = 0; i < PL_ARRAY_SIZE(points); i++) {
        const struct RGB p = ipt2rgb(rgb2ipt(points[i], src), dst);
        const float maxRGB = PL_MAX3(p.R, p.G, p.B);
        gain = fminf(gain, 1.0 / maxRGB);
    }

    FOREACH_LUT(lut, ipt) {
        struct RGB rgb = ipt2rgb(ipt, dst);
        rgb.R *= gain;
        rgb.G *= gain;
        rgb.B *= gain;
        ipt = rgb2ipt(rgb, dst);
        ipt = clip_gamma(ipt, perceptual_gamma, dst);
    }
}

const struct pl_gamut_map_function pl_gamut_map_darken = {
    .name = "darken",
    .description = "Darken and clip",
    .map = darken,
};

static void noop(float *lut, const struct pl_gamut_map_params *params)
{
    return;
}

const struct pl_gamut_map_function pl_gamut_map_clip = {
    .name = "clip",
    .description = "No gamut mapping (hard clip)",
    .map = noop,
};

const struct pl_gamut_map_function * const pl_gamut_map_functions[] = {
    &pl_gamut_map_clip,
    &pl_gamut_map_perceptual,
    &pl_gamut_map_relative,
    &pl_gamut_map_saturation,
    &pl_gamut_map_absolute,
    &pl_gamut_map_desaturate,
    &pl_gamut_map_darken,
    &pl_gamut_map_highlight,
    &pl_gamut_map_linear,
    NULL
};

const int pl_num_gamut_map_functions = PL_ARRAY_SIZE(pl_gamut_map_functions) - 1;

const struct pl_gamut_map_function *pl_find_gamut_map_function(const char *name)
{
    for (int i = 0; i < pl_num_gamut_map_functions; i++) {
        if (strcmp(name, pl_gamut_map_functions[i]->name) == 0)
            return pl_gamut_map_functions[i];
    }

    return NULL;
}
