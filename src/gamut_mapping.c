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

static inline float pq_eotf(float x)
{
    x = powf(fmaxf(x, 0.0f), 1.0f / PQ_M2);
    x = fmaxf(x - PQ_C1, 0.0f) / (PQ_C2 - PQ_C3 * x);
    return powf(x, 1.0f / PQ_M1);
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

static inline bool ingamut(struct RGB c, struct gamut gamut)
{
    return c.R >= gamut.min_rgb && c.R <= gamut.max_rgb &&
           c.G >= gamut.min_rgb && c.G <= gamut.max_rgb &&
           c.B >= gamut.min_rgb && c.B <= gamut.max_rgb;
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

static inline struct RGB ich2rgb(struct ICh c, struct gamut gamut)
{
    return ipt2rgb(ich2ipt(c), gamut);
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
        if (ingamut(ich2rgb(res, gamut), gamut)) {
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
    if (ingamut(ipt2rgb(ipt, gamut), gamut))
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
        if (ingamut(ich2rgb(test, gamut), gamut)) {
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

static void perceptual(float *lut, const struct pl_gamut_map_params *params)
{
    struct gamut dst, src;
    get_gamuts(&dst, &src, params);

    FOREACH_LUT(lut, ipt) {

        if (ipt.I <= dst.min_luma) {
            ipt.P = ipt.T = 0.0f;
            continue;
        }

        // Determine intersections with source and target gamuts
        struct ICh ich = ipt2ich(ipt);
        struct ICh source = saturate(ich.h, src);
        struct ICh target = saturate(ich.h, dst);
        const float gamma = scale_gamma(perceptual_gamma, ich, target, dst);

        float lo = 0.0f, x = 1.0f, hi = 1.0f / perceptual_knee + 3 * maxDelta;
        do {
            struct ICh test = mix_exp(ich, x, gamma, target.I);
            if (ingamut(ich2rgb(test, dst), dst)) {
                lo = x;
            } else {
                hi = x;
            }
            x = (lo + hi) / 2.0f;
        } while (hi - lo > maxDelta);

        // Apply simple Mobius tone mapping curve
        const float j = PL_MIX(1.0f, perceptual_knee, ich.C / 0.5f);
        const float margin = PL_DEF(params->chroma_margin, 1.0f);
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
        if (!ingamut(ipt2rgb(ipt, dst), dst)) {
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
