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

static const float PQ_M1 = 2610./4096 * 1./4,
                   PQ_M2 = 2523./4096 * 128,
                   PQ_C1 = 3424./4096,
                   PQ_C2 = 2413./4096 * 32,
                   PQ_C3 = 2392./4096 * 32;

float pl_hdr_rescale(enum pl_hdr_scaling from, enum pl_hdr_scaling to, float x)
{
    if (from == to)
        return x;
    if (!x) // micro-optimization for common value
        return x;

    // Convert input to PL_SCALE_RELATIVE
    switch (from) {
    case PL_HDR_PQ:
        x = powf(x, 1.0f / PQ_M2);
        x = fmaxf(x - PQ_C1, 0.0f) / (PQ_C2 - PQ_C3 * x);
        x = powf(x, 1.0f / PQ_M1);
        x *= 10000.0f;
        // fall through
    case PL_HDR_NITS:
        x /= PL_COLOR_SDR_WHITE;
        // fall through
    case PL_HDR_NORM:
        goto output;
    case PL_HDR_SQRT:
        x *= x;
        goto output;
    case PL_HDR_SCALING_COUNT:
        break;
    }

    pl_unreachable();

output:
    // Convert PL_SCALE_RELATIVE to output
    switch (to) {
    case PL_HDR_NORM:
        return x;
    case PL_HDR_SQRT:
        return sqrtf(x);
    case PL_HDR_NITS:
        return x * PL_COLOR_SDR_WHITE;
    case PL_HDR_PQ:
        x *= PL_COLOR_SDR_WHITE / 10000.0f;
        x = powf(x, PQ_M1);
        x = (PQ_C1 + PQ_C2 * x) / (1.0f + PQ_C3 * x);
        x = powf(x, PQ_M2);
        return x;
    case PL_HDR_SCALING_COUNT:
        break;
    }

    pl_unreachable();
}

bool pl_tone_map_params_equal(const struct pl_tone_map_params *a,
                              const struct pl_tone_map_params *b)
{
    return a->function == b->function &&
           a->param == b->param &&
           a->input_scaling == b->input_scaling &&
           a->output_scaling == b->output_scaling &&
           a->lut_size == b->lut_size &&
           a->input_min == b->input_min &&
           a->input_max == b->input_max &&
           a->output_min == b->output_min &&
           a->output_max == b->output_max &&
           pl_hdr_metadata_equal(&a->hdr, &b->hdr);
}

bool pl_tone_map_params_noop(const struct pl_tone_map_params *p)
{
    float in_min = pl_hdr_rescale(p->input_scaling, PL_HDR_NITS, p->input_min);
    float in_max = pl_hdr_rescale(p->input_scaling, PL_HDR_NITS, p->input_max);
    float out_min = pl_hdr_rescale(p->output_scaling, PL_HDR_NITS, p->output_min);
    float out_max = pl_hdr_rescale(p->output_scaling, PL_HDR_NITS, p->output_max);
    bool can_inverse = p->function == &pl_tone_map_auto || p->function->map_inverse;

    return fabs(in_min - out_min) < 1e-4 && // no BPC
           in_max < out_max + 1e-2 && // no range reduction
           (out_max < in_max + 1e-2 || !can_inverse); // no inverse tone-mapping
}

void pl_tone_map_params_infer(struct pl_tone_map_params *par)
{
    if (!par->function)
        par->function = &pl_tone_map_clip;

    if (!par->param)
        par->param = par->function->param_def;

    if (par->function == &pl_tone_map_auto) {
        float src_max = pl_hdr_rescale(par->input_scaling, PL_HDR_NORM, par->input_max);
        float dst_max = pl_hdr_rescale(par->output_scaling, PL_HDR_NORM, par->output_max);
        float ratio = src_max / dst_max;
        if ((par->hdr.ootf.num_anchors || par->hdr.scene_avg) && ratio > 1) {
            // HDR10+ metadata available: Pick SMPTE ST2094-40
            par->function = &pl_tone_map_st2094_40;
        } else if (ratio > 10) {
            // Extreme reduction: Pick spline for its quasi-linear behavior
            par->function = &pl_tone_map_spline;
        } else if (src_max < 1 + 1e-3 && dst_max < 1 + 1e-3) {
            // SDR<->SDR range conversion, use linear light stretching
            par->function = &pl_tone_map_linear;
        } else if (fmaxf(ratio, 1 / ratio) > 2 && fminf(src_max, dst_max) < 1.5f) {
            // Reasonably ranged HDR<->SDR conversion, pick BT.2446a since it
            // was designed for this task
            par->function = &pl_tone_map_bt2446a;
        } else if (ratio < 1) {
            // Small range inverse tone mapping, pick spline since BT.2446a
            // distorts colors too much
            par->function = &pl_tone_map_spline;
        } else {
            // Small range conversion (nearly no-op), pick BT.2390 because it
            // has the best asymptotic behavior (approximately linear).
            par->function = &pl_tone_map_bt2390;
        }
        par->param = par->function->param_def;
    }

    par->param = PL_CLAMP(par->param, par->function->param_min, par->function->param_max);
}

// Infer params and rescale to function scaling
static struct pl_tone_map_params fix_params(const struct pl_tone_map_params *params)
{
    struct pl_tone_map_params fixed = *params;
    pl_tone_map_params_infer(&fixed);

    const struct pl_tone_map_function *fun = params->function;
    fixed.input_scaling = fun->scaling;
    fixed.output_scaling = fun->scaling;
    fixed.input_min = pl_hdr_rescale(params->input_scaling, fun->scaling, params->input_min);
    fixed.input_max = pl_hdr_rescale(params->input_scaling, fun->scaling, params->input_max);
    fixed.output_min = pl_hdr_rescale(params->output_scaling, fun->scaling, params->output_min);
    fixed.output_max = pl_hdr_rescale(params->output_scaling, fun->scaling, params->output_max);

    return fixed;
}

#define FOREACH_LUT(lut, V)                                                     \
    for (float *_iter = lut, *_end = lut + params->lut_size, V;                 \
         _iter < _end && ( V = *_iter, 1 ); *_iter++ = V)

static void map_lut(float *lut, const struct pl_tone_map_params *params)
{
    if (params->output_max > params->input_max + 1e-4) {
        // Inverse tone-mapping
        if (params->function->map_inverse) {
            params->function->map_inverse(lut, params);
        } else {
            // Perform (perceptually linear) BPC only
            float input_min, input_max, output_min;
            input_min = pl_hdr_rescale(params->input_scaling, PL_HDR_PQ,
                                       params->input_min);
            input_max = pl_hdr_rescale(params->input_scaling, PL_HDR_PQ,
                                       params->input_max);
            output_min = pl_hdr_rescale(params->output_scaling, PL_HDR_PQ,
                                        params->output_min);
            float coeff = (input_max - output_min) / (input_max - input_min);

            FOREACH_LUT(lut, x) {
                x = pl_hdr_rescale(params->input_scaling, PL_HDR_PQ, x);
                x = coeff * (x - input_min) + output_min;
                x = pl_hdr_rescale(PL_HDR_PQ, params->output_scaling, x);
            }
        }
    } else {
        // Forward tone-mapping
        params->function->map(lut, params);
    }
}

void pl_tone_map_generate(float *out, const struct pl_tone_map_params *params)
{
    struct pl_tone_map_params fixed = fix_params(params);

    // Generate input values evenly spaced in `params->input_scaling`
    for (size_t i = 0; i < params->lut_size; i++) {
        float x = (float) i / (params->lut_size - 1);
        x = PL_MIX(params->input_min, params->input_max, x);
        out[i] = pl_hdr_rescale(params->input_scaling, fixed.function->scaling, x);
    }

    map_lut(out, &fixed);

    // Sanitize outputs and adapt back to `params->scaling`
    for (size_t i = 0; i < params->lut_size; i++) {
        float x = PL_CLAMP(out[i], fixed.output_min, fixed.output_max);
        out[i] = pl_hdr_rescale(fixed.function->scaling, params->output_scaling, x);
    }
}

float pl_tone_map_sample(float x, const struct pl_tone_map_params *params)
{
    struct pl_tone_map_params fixed = fix_params(params);
    fixed.lut_size = 1;

    x = PL_CLAMP(x, params->input_min, params->input_max);
    x = pl_hdr_rescale(params->input_scaling, fixed.function->scaling, x);
    map_lut(&x, &fixed);
    x = PL_CLAMP(x, fixed.output_min, fixed.output_max);
    x = pl_hdr_rescale(fixed.function->scaling, params->output_scaling, x);
    return x;
}

// Rescale from input-absolute to input-relative
static inline float rescale_in(float x, const struct pl_tone_map_params *params)
{
    return (x - params->input_min) / (params->input_max - params->input_min);
}

// Rescale from input-absolute to output-relative
static inline float rescale(float x, const struct pl_tone_map_params *params)
{
    return (x - params->input_min) / (params->output_max - params->output_min);
}

// Rescale from output-relative to output-absolute
static inline float rescale_out(float x, const struct pl_tone_map_params *params)
{
    return x * (params->output_max - params->output_min) + params->output_min;
}

static inline float bt1886_eotf(float x, float min, float max)
{
    const float lb = powf(min, 1/2.4f);
    const float lw = powf(max, 1/2.4f);
    return powf((lw - lb) * x + lb, 2.4f);
}

static inline float bt1886_oetf(float x, float min, float max)
{
    const float lb = powf(min, 1/2.4f);
    const float lw = powf(max, 1/2.4f);
    return (powf(x, 1/2.4f) - lb) / (lw - lb);
}

const struct pl_tone_map_function pl_tone_map_auto = {
    .name = "auto",
    .description = "Automatic selection",
};

static void noop(float *lut, const struct pl_tone_map_params *params)
{
    return;
}

const struct pl_tone_map_function pl_tone_map_clip = {
    .name = "clip",
    .description = "No tone mapping (clip)",
    .map = noop,
    .map_inverse = noop,
};

// Helper function to pick a knee point (for suitable methods) based on the
// HDR10+ brightness metadata and scene brightness average matching
static void st2094_pick_knee(float *out_src_knee, float *out_dst_knee,
                             const struct pl_tone_map_params *params)
{
    pl_assert(params->input_scaling == PL_HDR_NITS);
    pl_assert(params->output_scaling == PL_HDR_NITS);

    const float sdr_avg = PL_COLOR_SDR_WHITE / sqrtf(1000.0f); // typical contrast

    // Infer the average from scene metadata if present, or default to 10.0 as
    // this is an industry-standard value that produces good results on average
    //
    // Infer the destination from the desired output characteristics, clamping
    // the lower bound to the characteristics of a typical SDR signal
    float src_avg = PL_DEF(params->hdr.scene_avg, 10.0f);
    float dst_avg = fmaxf(sqrtf(params->output_min * params->output_max), sdr_avg);
    src_avg = PL_CLAMP(src_avg, params->input_min, params->input_max);
    dst_avg = PL_CLAMP(dst_avg, params->output_min, params->output_max);

    const float max_slope = 1.2f; // don't raise brightness too much
    dst_avg = fminf(dst_avg, src_avg * max_slope);

    *out_src_knee = fminf(src_avg, 0.8f * params->input_max);
    *out_dst_knee = fminf(sqrtf(src_avg * dst_avg), 0.8f * params->output_max);
}

// Pascal's triangle
static const uint16_t binom[17][17] = {
    {1},
    {1,1},
    {1,2,1},
    {1,3,3,1},
    {1,4,6,4,1},
    {1,5,10,10,5,1},
    {1,6,15,20,15,6,1},
    {1,7,21,35,35,21,7,1},
    {1,8,28,56,70,56,28,8,1},
    {1,9,36,84,126,126,84,36,9,1},
    {1,10,45,120,210,252,210,120,45,10,1},
    {1,11,55,165,330,462,462,330,165,55,11,1},
    {1,12,66,220,495,792,924,792,495,220,66,12,1},
    {1,13,78,286,715,1287,1716,1716,1287,715,286,78,13,1},
    {1,14,91,364,1001,2002,3003,3432,3003,2002,1001,364,91,14,1},
    {1,15,105,455,1365,3003,5005,6435,6435,5005,3003,1365,455,105,15,1},
    {1,16,120,560,1820,4368,8008,11440,12870,11440,8008,4368,1820,560,120,16,1},
};

static inline float st2094_intercept(uint8_t N, float Kx, float Ky)
{
    if (Kx <= 0 || Ky >= 1)
        return 1.0f / N;

    const float slope = Ky / Kx * (1 - Kx) / (1 - Ky);
    return fminf(slope / N, 1.0f);
}

static void st2094_40(float *lut, const struct pl_tone_map_params *params)
{
    const float D = params->output_max;

    // Allocate space for the adjusted bezier control points, plus endpoints
    float P[17], Kx, Ky, T;
    uint8_t N;

    if (params->hdr.ootf.num_anchors) {

        // Use bezier curve from metadata
        Kx = PL_CLAMP(params->hdr.ootf.knee_x, 0, 1);
        Ky = PL_CLAMP(params->hdr.ootf.knee_y, 0, 1);
        T = PL_CLAMP(params->hdr.ootf.target_luma, params->input_min, params->input_max);
        N = params->hdr.ootf.num_anchors + 1;
        pl_assert(N < PL_ARRAY_SIZE(P));
        memcpy(P + 1, params->hdr.ootf.anchors, (N - 1) * sizeof(*P));
        P[0] = 0.0f;
        P[N] = 1.0f;

    } else {

        // Missing metadata, default to simple brightness matching
        float src_knee, dst_knee;
        st2094_pick_knee(&src_knee, &dst_knee, params);
        Kx = src_knee / params->input_max;
        Ky = dst_knee / params->output_max;

        // Solve spline to match slope at knee intercept
        const float slope = Ky / Kx * (1 - Kx) / (1 - Ky);
        N = PL_CLAMP((int) ceilf(slope), 2, PL_ARRAY_SIZE(P) - 1);
        P[0] = 0.0f;
        P[1] = st2094_intercept(N, Kx, Ky);
        for (int i = 2; i <= N; i++)
            P[i] = 1.0f;
        T = D;

    }

    if (D < T) {

        // Output display darker than OOTF target, make brighter
        const float Dmin = 0.0f, u = fmaxf(0.0f, (D - Dmin) / (T - Dmin));

        // Scale down the knee point to make more room for the OOTF
        Kx *= u;
        Ky *= u;

        // Make the slope of the knee more closely approximate a clip(),
        // constrained to avoid exploding P[1]
        const float beta = N * Kx / (1 - Kx);
        const float Kxy = fminf(Kx * params->input_max / D, beta / (beta + 1));
        Ky = PL_MIX(Kxy, Ky, u);

        for (int p = 2; p <= N; p++)
            P[p] = PL_MIX(1.0f, P[p], u);

        // Make the OOTF intercept linear as D -> Dmin
        P[1] = PL_MIX(st2094_intercept(N, Kx, Ky), P[1], u);

    } else if (D > T) {

        // Output display brighter than OOTF target, make more linear
        pl_assert(params->input_max > T);
        const float w = powf(1 - (D - T) / (params->input_max - T), 1.4f);

        // Constrain the slope of the input knee to prevent it from
        // exploding and making the picture way too bright
        Ky *= T / D;

        // Make the slope of the knee more linear by solving for f(Kx) = Kx
        float Kxy = Kx * D / params->input_max;
        Ky = PL_MIX(Kxy, Ky, w);

        for (int p = 2; p < N; p++) {
            float anchor_lin = (float) p / N;
            P[p] = PL_MIX(anchor_lin, P[p], w);
        }

        // Make the OOTF intercept linear as D -> input_max
        P[1] = PL_MIX(st2094_intercept(N, Kx, Ky), P[1], w);

    }

    pl_assert(Kx >= 0 && Kx <= 1);
    pl_assert(Ky >= 0 && Ky <= 1);

    FOREACH_LUT(lut, x) {
        x = bt1886_oetf(x, params->input_min, params->input_max);
        x = bt1886_eotf(x, 0.0f, 1.0f);

        if (x <= Kx && Kx) {
            // Linear section
            x *= Ky / Kx;
        } else {
            // Bezier section
            const float t = (x - Kx) / (1 - Kx);

            x = 0; // Bn
            for (uint8_t p = 0; p <= N; p++)
                x += binom[N][p] * powf(t, p) * powf(1 - t, N - p) * P[p];

            x = Ky + (1 - Ky) * x;
        }

        x = bt1886_oetf(x, 0.0f, 1.0f);
        x = bt1886_eotf(x, params->output_min, params->output_max);
    }
}

const struct pl_tone_map_function pl_tone_map_st2094_40 = {
    .name = "st2094-40",
    .description = "SAMPTE ST 2094-40 Annex B",
    .scaling = PL_HDR_NITS,
    .map = st2094_40,
};

static void st2094_10(float *lut, const struct pl_tone_map_params *params)
{
    float src_knee, dst_knee;
    st2094_pick_knee(&src_knee, &dst_knee, params);

    const float x1 = powf(params->input_min, params->param);
    const float x3 = powf(params->input_max, params->param);
    const float x2 = powf(src_knee, params->param);

    const float y1 = params->output_min;
    const float y3 = params->output_max;
    const float y2 = dst_knee;

    const struct pl_matrix3x3 cmat = {{
        { x2*x3*(y2 - y3), x1*x3*(y3 - y1), x1*x2*(y1 - y2) },
        { x3*y3 - x2*y2,   x1*y1 - x3*y3,   x2*y2 - x1*y1   },
        { x3 - x2,         x1 - x3,         x2 - x1         },
    }};

    float coeffs[3] = { y1, y2, y3 };
    pl_matrix3x3_apply(&cmat, coeffs);

    const float k = 1.0 / (x3*y3*(x1 - x2) + x2*y2*(x3 - x1) + x1*y1*(x2 - x3));
    const float c1 = k * coeffs[0];
    const float c2 = k * coeffs[1];
    const float c3 = k * coeffs[2];

    FOREACH_LUT(lut, x) {
        float Ln = powf(x, params->param);
        x = (c1 + c2 * Ln) / (1 + c3 * Ln);
    }
}

const struct pl_tone_map_function pl_tone_map_st2094_10 = {
    .name = "st2094-10",
    .description = "SAMPTE ST 2094-10 Annex B.2",
    .scaling = PL_HDR_NITS,
    .param_desc = "Contrast",
    .param_min = 0.20,
    .param_def = 1.00,
    .param_max = 2.00,
    .map = st2094_10,
};

static void bt2390(float *lut, const struct pl_tone_map_params *params)
{
    const float minLum = rescale_in(params->output_min, params);
    const float maxLum = rescale_in(params->output_max, params);
    const float offset = params->param;
    const float ks = (1 + offset) * maxLum - offset;
    const float bp = minLum > 0 ? fminf(1 / minLum, 4) : 4;
    const float gain_inv = 1 + minLum / maxLum * powf(1 - maxLum, bp);
    const float gain = maxLum < 1 ? 1 / gain_inv : 1;

    FOREACH_LUT(lut, x) {
        x = rescale_in(x, params);

        // Piece-wise hermite spline
        if (ks < 1) {
            float tb = (x - ks) / (1 - ks);
            float tb2 = tb * tb;
            float tb3 = tb2 * tb;
            float pb = (2 * tb3 - 3 * tb2 + 1) * ks +
                       (tb3 - 2 * tb2 + tb) * (1 - ks) +
                       (-2 * tb3 + 3 * tb2) * maxLum;
            x = x < ks ? x : pb;
        }

        // Black point adaptation
        if (x < 1) {
            x += minLum * powf(1 - x, bp);
            x = gain * (x - minLum) + minLum;
        }

        x = x * (params->input_max - params->input_min) + params->input_min;
    }
}

const struct pl_tone_map_function pl_tone_map_bt2390 = {
    .name = "bt2390",
    .description = "ITU-R BT.2390 EETF",
    .scaling = PL_HDR_PQ,
    .param_desc = "Knee offset",
    .param_min = 0.50,
    .param_def = 1.00,
    .param_max = 2.00,
    .map = bt2390,
};

static void bt2446a(float *lut, const struct pl_tone_map_params *params)
{
    const float phdr = 1 + 32 * powf(params->input_max / 10000, 1/2.4f);
    const float psdr = 1 + 32 * powf(params->output_max / 10000, 1/2.4f);

    FOREACH_LUT(lut, x) {
        x = powf(rescale_in(x, params), 1/2.4f);
        x = logf(1 + (phdr - 1) * x) / logf(phdr);

        if (x <= 0.7399f) {
            x = 1.0770f * x;
        } else if (x < 0.9909f) {
            x = (-1.1510f * x + 2.7811f) * x - 0.6302f;
        } else {
            x = 0.5f * x + 0.5f;
        }

        x = (powf(psdr, x) - 1) / (psdr - 1);
        x = bt1886_eotf(x, params->output_min, params->output_max);
    }
}

static void bt2446a_inv(float *lut, const struct pl_tone_map_params *params)
{
    FOREACH_LUT(lut, x) {
        x = bt1886_oetf(x, params->input_min, params->input_max);
        x *= 255.0;
        if (x > 70) {
            x = powf(x, (2.8305e-6f * x - 7.4622e-4f) * x + 1.2528f);
        } else {
            x = powf(x, (1.8712e-5f * x - 2.7334e-3f) * x + 1.3141f);
        }
        x = powf(x / 1000, 2.4f);
        x = rescale_out(x, params);
    }
}

const struct pl_tone_map_function pl_tone_map_bt2446a = {
    .name = "bt2446a",
    .description = "ITU-R BT.2446 Method A",
    .scaling = PL_HDR_NITS,
    .map = bt2446a,
    .map_inverse = bt2446a_inv,
};

static void spline(float *lut, const struct pl_tone_map_params *params)
{
    // Normalize everything the pivot to make the math easier
    const float pivot = params->param;
    const float in_min = params->input_min - pivot;
    const float in_max = params->input_max - pivot;
    const float out_min = params->output_min - pivot;
    const float out_max = params->output_max - pivot;

    // Solve P of order 2 for:
    //  P(in_min) = out_min
    //  P'(0.0) = 1.0
    //  P(0.0) = 0.0
    const float Pa = (out_min - in_min) / (in_min * in_min);

    // Solve Q of order 3 for:
    //  Q(in_max) = out_max
    //  Q''(in_max) = 0.0
    //  Q(0.0) = 0.0
    //  Q'(0.0) = 1.0
    const float t = 2 * in_max * in_max;
    const float Qa = (in_max - out_max) / (in_max * t);
    const float Qb = -3 * (in_max - out_max) / t;

    FOREACH_LUT(lut, x) {
        x -= pivot;
        x = x > 0 ? ((Qa * x + Qb) * x + 1) * x : (Pa * x + 1) * x;
        x += pivot;
    }
}

const struct pl_tone_map_function pl_tone_map_spline = {
    .name = "spline",
    .description = "Single-pivot polynomial spline",
    .param_desc = "Pivot point",
    .param_min = 0.15, // ~1 nits
    .param_def = 0.30, // ~10 nits
    .param_max = 0.50, // ~100 nits
    .scaling = PL_HDR_PQ,
    .map = spline,
    .map_inverse = spline,
};

static void reinhard(float *lut, const struct pl_tone_map_params *params)
{
    const float peak = rescale(params->input_max, params),
                contrast = params->param,
                offset = (1.0 - contrast) / contrast,
                scale = (peak + offset) / peak;

    FOREACH_LUT(lut, x) {
        x = rescale(x, params);
        x = x / (x + offset);
        x *= scale;
        x = rescale_out(x, params);
    }
}

const struct pl_tone_map_function pl_tone_map_reinhard = {
    .name = "reinhard",
    .description = "Reinhard",
    .param_desc = "Contrast",
    .param_min = 0.001,
    .param_def = 0.50,
    .param_max = 0.99,
    .map = reinhard,
};

static void mobius(float *lut, const struct pl_tone_map_params *params)
{
    const float peak = rescale(params->input_max, params),
                j = params->param;

    // Solve for M(j) = j; M(peak) = 1.0; M'(j) = 1.0
    // where M(x) = scale * (x+a)/(x+b)
    const float a = -j*j * (peak - 1.0f) / (j*j - 2.0f * j + peak);
    const float b = (j*j - 2.0f * j * peak + peak) /
                    fmaxf(1e-6f, peak - 1.0f);
    const float scale = (b*b + 2.0f * b*j + j*j) / (b - a);

    FOREACH_LUT(lut, x) {
        x = rescale(x, params);
        x = x <= j ? x : scale * (x + a) / (x + b);
        x = rescale_out(x, params);
    }
}

const struct pl_tone_map_function pl_tone_map_mobius = {
    .name = "mobius",
    .description = "Mobius",
    .param_desc = "Knee point",
    .param_min = 0.00,
    .param_def = 0.30,
    .param_max = 0.99,
    .map = mobius,
};

static inline float hable(float x)
{
    const float A = 0.15, B = 0.50, C = 0.10, D = 0.20, E = 0.02, F = 0.30;
    return ((x * (A*x + C*B) + D*E) / (x * (A*x + B) + D*F)) - E/F;
}

static void hable_map(float *lut, const struct pl_tone_map_params *params)
{
    const float peak = params->input_max / params->output_max,
                scale = 1.0f / hable(peak);

    FOREACH_LUT(lut, x) {
        x = bt1886_oetf(x, params->input_min, params->input_max);
        x = bt1886_eotf(x, 0, peak);
        x = scale * hable(x);
        x = bt1886_oetf(x, 0, 1);
        x = bt1886_eotf(x, params->output_min, params->output_max);
    }
}

const struct pl_tone_map_function pl_tone_map_hable = {
    .name = "hable",
    .description = "Filmic tone-mapping (Hable)",
    .map = hable_map,
};

static void gamma_map(float *lut, const struct pl_tone_map_params *params)
{
    const float peak = rescale(params->input_max, params),
                cutoff = params->param,
                gamma = logf(cutoff) / logf(cutoff / peak);

    FOREACH_LUT(lut, x) {
        x = rescale(x, params);
        x = x > cutoff ? powf(x / peak, gamma) : x;
        x = rescale_out(x, params);
    }
}

const struct pl_tone_map_function pl_tone_map_gamma = {
    .name = "gamma",
    .description = "Gamma function with knee",
    .param_desc = "Knee point",
    .param_min = 0.001,
    .param_def = 0.50,
    .param_max = 1.00,
    .map = gamma_map,
};

static void linear(float *lut, const struct pl_tone_map_params *params)
{
    const float gain = params->param;

    FOREACH_LUT(lut, x) {
        x = rescale_in(x, params);
        x *= gain;
        x = rescale_out(x, params);
    }
}

const struct pl_tone_map_function pl_tone_map_linear = {
    .name = "linear",
    .description = "Perceptually linear stretch",
    .param_desc = "Exposure",
    .param_min = 0.001,
    .param_def = 1.00,
    .param_max = 10.0,
    .scaling = PL_HDR_PQ,
    .map = linear,
    .map_inverse = linear,
};

const struct pl_tone_map_function * const pl_tone_map_functions[] = {
    &pl_tone_map_auto,
    &pl_tone_map_clip,
    &pl_tone_map_st2094_40,
    &pl_tone_map_st2094_10,
    &pl_tone_map_bt2390,
    &pl_tone_map_bt2446a,
    &pl_tone_map_spline,
    &pl_tone_map_reinhard,
    &pl_tone_map_mobius,
    &pl_tone_map_hable,
    &pl_tone_map_gamma,
    &pl_tone_map_linear,
    NULL
};

const int pl_num_tone_map_functions = PL_ARRAY_SIZE(pl_tone_map_functions) - 1;

const struct pl_tone_map_function *pl_find_tone_map_function(const char *name)
{
    for (int i = 0; i < pl_num_tone_map_functions; i++) {
        if (strcmp(name, pl_tone_map_functions[i]->name) == 0)
            return pl_tone_map_functions[i];
    }

    return NULL;
}
