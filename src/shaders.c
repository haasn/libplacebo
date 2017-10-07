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
#include "bstr/bstr.h"

#include "common.h"
#include "context.h"

struct priv {
    struct bstr buffer;
    bool flexible_work_groups;
};

struct pl_shader *pl_shader_alloc(struct pl_context *ctx,
                                  const struct ra *ra)
{
    struct pl_shader *s = talloc_ptrtype(ctx, s);
    *s = (struct pl_shader) {
        .ctx = ctx,
        .ra = ra,
        .glsl = "",
        .priv = talloc_zero(s, struct priv),
    };

    return s;
}

void pl_shader_free(struct pl_shader **s)
{
    TA_FREEP(s);
}

bool pl_shader_is_compute(const struct pl_shader *s)
{
    bool ret = true;
    for (int i = 0; i < PL_ARRAY_SIZE(s->compute_work_groups); i++)
        ret &= !!s->compute_work_groups[i];
    return ret;
}

// Append a raw `struct ra_var` to the pl_shader (while making a copy of
// the variable name and data).
static void pl_shader_var(struct pl_shader *s, struct ra_var var,
                          const void *data)
{
    size_t size = ra_var_host_layout(var).size;
    int idx = s->num_variables++;
    TARRAY_GROW(s, s->variables, idx);
    TARRAY_GROW(s, s->variable_data, idx);

    var.name = talloc_strdup(s, var.name);
    s->variables[idx] = var;
    s->variable_data[idx] = talloc_memdup(s, data, size);
}

// Helpers for some of the most common variable types
static void pl_shader_var_vec3(struct pl_shader *s, const char *name,
                               const float f[3])
{
    pl_shader_var(s, (struct ra_var) {
        .name = name,
        .type = RA_VAR_FLOAT,
        .dim_v = 3,
        .dim_m = 1,
    }, f);
}

static void pl_shader_var_mat3(struct pl_shader *s, const char *name,
                               bool column_major, const float m[3][3])
{
    struct ra_var var = {
        .name = name,
        .type = RA_VAR_FLOAT,
        .dim_v = 3,
        .dim_m = 3,
    };

    if (column_major) {
        pl_shader_var(s, var, m);
    } else {
        float tmp[3][3] = {
            { m[0][0], m[0][1], m[0][2] },
            { m[1][0], m[1][1], m[1][2] },
            { m[2][0], m[2][1], m[2][2] },
        };

        pl_shader_var(s, var, tmp);
    }
}

// Append a raw `struct ra_desc` to the pl_shader (while making a copy of
// the descriptor name).
static void pl_shader_desc(struct pl_shader *s, struct ra_desc desc,
                           const void *binding)
{
    int idx = s->num_descriptors++;
    TARRAY_GROW(s, s->descriptors, idx);
    TARRAY_GROW(s, s->descriptor_bindings, idx);

    desc.name = talloc_strdup(s, desc.name);
    s->descriptors[idx] = desc;
    s->descriptor_bindings[idx] = binding;
}

static void pl_shader_append(struct pl_shader *s, const char *fmt, ...)
    PRINTF_ATTRIBUTE(2, 3);

static void pl_shader_append(struct pl_shader *s, const char *fmt, ...)
{
    struct priv *p = s->priv;

    va_list ap;
    va_start(ap, fmt);
    bstr_xappend_vasprintf(s, &p->buffer, fmt, ap);
    va_end(ap);

    // Update the GLSL shader body pointer in case the buffer got re-allocated
    s->glsl = p->buffer.start;
}

#define GLSL(...) pl_shader_append(s, __VA_ARGS__)

// Colorspace operations

// Common constants for SMPTE ST.2084 (PQ)
static const float PQ_M1 = 2610./4096 * 1./4,
                   PQ_M2 = 2523./4096 * 128,
                   PQ_C1 = 3424./4096,
                   PQ_C2 = 2413./4096 * 32,
                   PQ_C3 = 2392./4096 * 32;

// Common constants for ARIB STD-B67 (HLG)
static const float HLG_A = 0.17883277,
                   HLG_B = 0.28466892,
                   HLG_C = 0.55991073;

// Common constants for Panasonic V-Log
static const float VLOG_B = 0.00873,
                   VLOG_C = 0.241514,
                   VLOG_D = 0.598206;

// Common constants for Sony S-Log
static const float SLOG_A = 0.432699,
                   SLOG_B = 0.037584,
                   SLOG_C = 0.616596 + 0.03,
                   SLOG_P = 3.538813,
                   SLOG_Q = 0.030001,
                   SLOG_K2 = 155.0 / 219.0;

void pl_shader_linearize(struct pl_shader *s, enum pl_color_transfer trc)
{
    if (trc == PL_COLOR_TRC_LINEAR)
        return;

    // Note that this clamp may technically violate the definition of
    // ITU-R BT.2100, which allows for sub-blacks and super-whites to be
    // displayed on the display where such would be possible. That said, the
    // problem is that not all gamma curves are well-defined on the values
    // outside this range, so we ignore it and just clip anyway for sanity.
    GLSL("// pl_shader_linearize                  \n"
         "color.rgb = clamp(color.rgb, 0.0, 1.0); \n");

    switch (trc) {
    case PL_COLOR_TRC_SRGB:
        GLSL("color.rgb = mix(color.rgb * vec3(1.0/12.92),               \n"
             "                pow((color.rgb + vec3(0.055))/vec3(1.055), \n"
             "                    vec3(2.4)),                            \n"
             "                lessThan(vec3(0.04045), color.rgb));       \n");
        break;
    case PL_COLOR_TRC_BT_1886:
        GLSL("color.rgb = pow(color.rgb, vec3(2.4));\n");
        break;
    case PL_COLOR_TRC_GAMMA18:
        GLSL("color.rgb = pow(color.rgb, vec3(1.8));\n");
        break;
    case PL_COLOR_TRC_GAMMA22:
        GLSL("color.rgb = pow(color.rgb, vec3(2.2));\n");
        break;
    case PL_COLOR_TRC_GAMMA28:
        GLSL("color.rgb = pow(color.rgb, vec3(2.8));\n");
        break;
    case PL_COLOR_TRC_PRO_PHOTO:
        GLSL("color.rgb = mix(color.rgb * vec3(1.0/16.0),          \n"
             "                pow(color.rgb, vec3(1.8)),           \n"
             "                lessThan(vec3(0.03125), color.rgb)); \n");
        break;
    case PL_COLOR_TRC_PQ:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/%f));         \n"
             "color.rgb = max(color.rgb - vec3(%f), vec3(0.0))  \n"
             "             / (vec3(%f) - vec3(%f) * color.rgb); \n"
             "color.rgb = pow(color.rgb, vec3(1.0/%f));         \n"
             // PQ's output range is 0-10000, but we need it to be relative to
             // to PL_COLOR_REF_WHITE instead, so rescale
             "color.rgb *= vec3(%f);\n",
             PQ_M2, PQ_C1, PQ_C2, PQ_C3, PQ_M1, 10000 / PL_COLOR_REF_WHITE);
        break;
    case PL_COLOR_TRC_HLG:
        GLSL("color.rgb = mix(vec3(4.0) * color.rgb * color.rgb,         \n"
             "                exp((color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "                    + vec3(%f),                            \n"
             "                lessThan(vec3(0.5), color.rgb));           \n",
             HLG_C, HLG_A, HLG_B);
        break;
    case PL_COLOR_TRC_V_LOG:
        GLSL("color.rgb = mix((color.rgb - vec3(0.125)) * vec3(1.0/5.6), \n"
             "    pow(vec3(10.0), (color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "              - vec3(%f),                                  \n"
             "    lessThanEqual(vec3(0.181), color.rgb));                \n",
             VLOG_D, VLOG_C, VLOG_B);
        break;
    case PL_COLOR_TRC_S_LOG1:
        GLSL("color.rgb = pow(vec3(10.0), (color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "            - vec3(%f);                                            \n",
             SLOG_C, SLOG_A, SLOG_B);
        break;
    case PL_COLOR_TRC_S_LOG2:
        GLSL("color.rgb = mix((color.rgb - vec3(%f)) * vec3(1.0/%f),      \n"
             "    (pow(vec3(10.0), (color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "              - vec3(%f)) * vec3(1.0/%f),                   \n"
             "    lessThanEqual(vec3(%f), color.rgb));                    \n",
             SLOG_Q, SLOG_P, SLOG_C, SLOG_A, SLOG_B, SLOG_K2, SLOG_Q);
        break;
    default:
        abort();
    }

    // Rescale to prevent clipping on non-float textures
    GLSL("color.rgb *= vec3(1.0/%f);\n", pl_color_transfer_nominal_peak(trc));
}

void pl_shader_delinearize(struct pl_shader *s, enum pl_color_transfer trc)
{
    if (trc == PL_COLOR_TRC_LINEAR)
        return;

    GLSL("// pl_shader_delinearize\n"
         "color.rgb = clamp(color.rgb, 0.0, 1.0);\n"
         "color.rgb *= vec3(%f);\n", pl_color_transfer_nominal_peak(trc));

    switch (trc) {
    case PL_COLOR_TRC_SRGB:
        GLSL("color.rgb = mix(color.rgb * vec3(12.92),                    \n"
             "                vec3(1.055) * pow(color.rgb, vec3(1.0/2.4)) \n"
             "                    - vec3(0.055),                          \n"
             "                lessThanEqual(vec3(0.0031308), color.rgb)); \n");
        break;
    case PL_COLOR_TRC_BT_1886:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.4));\n");
        break;
    case PL_COLOR_TRC_GAMMA18:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/1.8));\n");
        break;
    case PL_COLOR_TRC_GAMMA22:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.2));\n");
        break;
    case PL_COLOR_TRC_GAMMA28:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.8));\n");
        break;
    case PL_COLOR_TRC_PRO_PHOTO:
        GLSL("color.rgb = mix(color.rgb * vec3(16.0),                    \n"
             "                pow(color.rgb, vec3(1.0/1.8)),             \n"
             "                lessThanEqual(vec3(0.001953), color.rgb)); \n");
        break;
    case PL_COLOR_TRC_PQ:
        GLSL("color.rgb *= vec3(1.0/%f);                         \n"
             "color.rgb = pow(color.rgb, vec3(%f));              \n"
             "color.rgb = (vec3(%f) + vec3(%f) * color.rgb)      \n"
             "             / (vec3(1.0) + vec3(%f) * color.rgb); \n"
             "color.rgb = pow(color.rgb, vec3(%f));              \n",
             10000 / PL_COLOR_REF_WHITE, PQ_M1, PQ_C1, PQ_C2, PQ_C3, PQ_M2);
        break;
    case PL_COLOR_TRC_HLG:
        GLSL("color.rgb = mix(vec3(0.5) * sqrt(color.rgb),                     \n"
             "                vec3(%f) * log(color.rgb - vec3(%f)) + vec3(%f), \n"
             "                lessThan(vec3(1.0), color.rgb));                 \n",
             HLG_A, HLG_B, HLG_C);
        break;
    case PL_COLOR_TRC_V_LOG:
        GLSL("color.rgb = mix(vec3(5.6) * color.rgb + vec3(0.125),   \n"
             "                vec3(%f) * log(color.rgb + vec3(%f))   \n"
             "                    + vec3(%f),                        \n"
             "                lessThanEqual(vec3(0.01), color.rgb)); \n",
             VLOG_C / M_LN10, VLOG_B, VLOG_D);
        break;
    case PL_COLOR_TRC_S_LOG1:
        GLSL("color.rgb = vec3(%f) * log(color.rgb + vec3(%f)) + vec3(%f);\n",
             SLOG_A / M_LN10, SLOG_B, SLOG_C);
        break;
    case PL_COLOR_TRC_S_LOG2:
        GLSL("color.rgb = mix(vec3(%f) * color.rgb + vec3(%f),                \n"
             "                vec3(%f) * log(vec3(%f) * color.rgb + vec3(%f)) \n"
             "                    + vec3(%f),                                 \n"
             "                lessThanEqual(vec3(0.0), color.rgb));           \n",
             SLOG_P, SLOG_Q, SLOG_A / M_LN10, SLOG_K2, SLOG_B, SLOG_C);
        break;
    default:
        abort();
    }
}

void pl_shader_ootf(struct pl_shader *s, enum pl_color_light light, float peak)
{
    if (light == PL_COLOR_LIGHT_DISPLAY)
        return;

    GLSL("// pl_shader_ootf      \n"
         "color.rgb *= vec3(%f); \n", peak);

    switch (light)
    {
    case PL_COLOR_LIGHT_SCENE_HLG:
        // HLG OOTF from BT.2100, assuming a reference display with a
        // peak of 1000 cd/m² -> gamma = 1.2
        GLSL("color.rgb *= vec3(%f * pow(dot(src_luma, color.rgb), 0.2));\n",
             (1000 / PL_COLOR_REF_WHITE) / pow(12, 1.2));
        break;
    case PL_COLOR_LIGHT_SCENE_709_1886:
        // This OOTF is defined by encoding the result as 709 and then decoding
        // it as 1886; although this is called 709_1886 we actually use the
        // more precise (by one decimal) values from BT.2020 instead
        GLSL("color.rgb = mix(color.rgb * vec3(4.5),                    \n"
             "                vec3(1.0993) * pow(color.rgb, vec3(0.45)) \n"
             "                             - vec3(0.0993),              \n"
             "                lessThan(vec3(0.0181), color.rgb));       \n"
             "color.rgb = pow(color.rgb, vec3(2.4));                    \n");
        break;
    case PL_COLOR_LIGHT_SCENE_1_2:
        GLSL("color.rgb = pow(color.rgb, vec3(1.2));\n");
        break;
    default:
        abort();
    }

    GLSL("color.rgb *= vec3(1.0/%f);\n", peak);
}

void pl_shader_inverse_ootf(struct pl_shader *s, enum pl_color_light light, float peak)
{
    if (light == PL_COLOR_LIGHT_DISPLAY)
        return;

    GLSL("// pl_shader_inverse_ootf\n"
         "color.rgb *= vec3(%f);\n", peak);

    switch (light)
    {
    case PL_COLOR_LIGHT_SCENE_HLG:
        GLSL("color.rgb *= vec3(1.0/%f);                                \n"
             "color.rgb /= vec3(max(1e-6, pow(dot(src_luma, color.rgb), \n"
             "                                0.2/1.2)));               \n",
             (1000 / PL_COLOR_REF_WHITE) / pow(12, 1.2));
        break;
    case PL_COLOR_LIGHT_SCENE_709_1886:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.4));                         \n"
             "color.rgb = mix(color.rgb * vec3(1.0/4.5),                         \n"
             "                pow((color.rgb + vec3(0.0993)) * vec3(1.0/1.0993), \n"
             "                    vec3(1/0.45)),                                 \n"
             "                lessThan(vec3(0.08145), color.rgb));               \n");
        break;
    case PL_COLOR_LIGHT_SCENE_1_2:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/1.2));\n");
        break;
    default:
        abort();
    }

    GLSL("color.rgb *= vec3(1.0/%f);\n", peak);
}

const struct pl_color_map_params pl_color_map_recommended_params = {
    .intent                  = PL_INTENT_RELATIVE_COLORIMETRIC,
    .tone_mapping_algo       = PL_TONE_MAPPING_MOBIUS,
    .tone_mapping_desaturate = 2.0,
    .peak_detect_frames      = 50,
};

static void pl_shader_tone_map(struct pl_shader *s, float ref_peak,
                     const struct pl_color_map_params *params)
{
    GLSL("// pl_shader_tone_map\n");

    // Desaturate the color using a coefficient dependent on the luminance
    if (params->tone_mapping_desaturate > 0) {
        GLSL("float luma = dot(dst_luma, color.rgb);                     \n"
             "float overbright = max(luma - %f, 1e-6) / max(luma, 1e-6); \n"
             "color.rgb = mix(color.rgb, vec3(luma), overbright);        \n",
             params->tone_mapping_desaturate);
    }

    // To prevent discoloration due to out-of-bounds clipping, we need to make
    // sure to reduce the value range as far as necessary to keep the entire
    // signal in range, so tone map based on the brightest component.
    GLSL("float sig = max(max(color.r, color.g), color.b); \n"
         "float sig_orig = sig;                            \n");

    if (params->peak_detect_ssbo) {
        PL_FATAL(s, "pl_shader_tone_map: peak_detect_ssbo not yet supported!\n");
        abort();
    } else {
        GLSL("const float sig_peak = %f;\n", ref_peak);
    }

    float param = params->tone_mapping_param;
    switch (params->tone_mapping_algo) {
    case PL_TONE_MAPPING_CLIP:
        GLSL("sig = %f * sig;\n", param ? param : 1.0);
        break;

    case PL_TONE_MAPPING_MOBIUS:
        GLSL("const float j = %f;                                           \n"
             // solve for M(j) = j; M(sig_peak) = 1.0; M'(j) = 1.0
             // where M(x) = scale * (x+a)/(x+b)
             "float a = -j*j * (sig_peak - 1.0) / (j*j - 2.0*j + sig_peak); \n"
             "float b = (j*j - 2.0*j*sig_peak + sig_peak) /                 \n"
             "          max(1e-6, sig_peak - 1.0);                          \n"
             "float scale = (b*b + 2.0*b*j + j*j) / (b-a);                  \n"
             "sig = mix(sig, scale * (sig + a) / (sig + b), sig > j);       \n",
             param ? param : 0.3);
        break;

    case PL_TONE_MAPPING_REINHARD: {
        float contrast = param ? param : 0.5,
              offset = (1.0 - contrast) / contrast;
        GLSL("sig = sig / (sig + %f);                   \n"
             "float scale = (sig_peak + %f) / sig_peak; \n"
             "sig *= scale;                             \n",
             offset, offset);
        break;
    }

    case PL_TONE_MAPPING_HABLE: {
        float A = 0.15, B = 0.50, C = 0.10, D = 0.20, E = 0.02, F = 0.30;
        GLSL("sig = ((sig * (%f*sig + %f)+%f)/(sig * (%f*sig + %f) + %f)) - %f;\n",
             A, C*B, D*E, A, B, D*F, E/F);
        // FIXME: make this benefit from sig_peak
        break;
    }

    case PL_TONE_MAPPING_GAMMA: {
        GLSL("const float cutoff = 0.05, gamma = 1.0/%f;                     \n"
             "float scale = pow(cutoff / sig_peak, gamma) / cutoff;          \n"
             "sig = sig > cutoff ? pow(sig / sig_peak, gamma) : scale * sig; \n",
             param ? param : 1.8);
        break;
    }

    case PL_TONE_MAPPING_LINEAR: {
        GLSL("sig = %f / sig_peak * sig;\n", param ? param : 1.0);
        break;
    }

    default:
        abort();
    }

    // Apply the computed scale factor to the color, linearly to prevent
    // discoloration
    GLSL("color.rgb *= sig / sig_orig;\n");
}

void pl_shader_color_map(struct pl_shader *s,
                         const struct pl_color_map_params *params,
                         struct pl_color_space src, struct pl_color_space dst,
                         bool prelinearized)
{
    GLSL("// pl_shader_color_map\n");
    GLSL("{\n");

    // Compute the highest encodable level
    float src_range = pl_color_transfer_nominal_peak(src.transfer),
          dst_range = pl_color_transfer_nominal_peak(dst.transfer);
    float ref_peak = src.sig_peak / dst_range;

    // Some operations need access to the video's luma coefficients, so make
    // them available
    struct pl_color_matrix rgb2xyz;
    rgb2xyz = pl_get_rgb2xyz_matrix(pl_raw_primaries_get(src.primaries));
    pl_shader_var_vec3(s, "src_luma", rgb2xyz.m[1]);
    rgb2xyz = pl_get_rgb2xyz_matrix(pl_raw_primaries_get(dst.primaries));
    pl_shader_var_vec3(s, "dst_luma", rgb2xyz.m[1]);

    // All operations from here on require linear light as a starting point,
    // so we linearize even if src.gamma == dst.gamma when one of the other
    // operations needs it
    bool need_linear = src.transfer != dst.transfer ||
                       src.primaries != dst.primaries ||
                       src_range != dst_range ||
                       src.sig_peak > dst_range ||
                       src.light != dst.light;
    bool is_linear = prelinearized;

    if (need_linear && !is_linear) {
        pl_shader_linearize(s, src.transfer);
        is_linear = true;
    }

    if (src.light != dst.light)
        pl_shader_ootf(s, src.light, pl_color_transfer_nominal_peak(src.transfer));

    // Rescale the signal to compensate for differences in the encoding range
    // and reference white level. This is necessary because of the 0-1 value
    // normalization for HDR signals.
    if (src_range != dst_range) {
        GLSL("// rescale value range;\n");
        GLSL("color.rgb *= vec3(%f);\n", src_range / dst_range);
    }

    // Adapt to the right colorspace (primaries) if necessary
    if (src.primaries != dst.primaries) {
        struct pl_raw_primaries csp_src = pl_raw_primaries_get(src.primaries),
                                csp_dst = pl_raw_primaries_get(dst.primaries);
        struct pl_color_matrix cms_matrix;
        cms_matrix = pl_get_color_mapping_matrix(csp_src, csp_dst, params->intent);
        pl_shader_var_mat3(s, "cms_matrix", false, cms_matrix.m);
        GLSL("color.rgb = cms_matrix * color.rgb;\n");
        // Since this can reduce the gamut, figure out by how much
        for (int c = 0; c < 3; c++)
            ref_peak = fmaxf(ref_peak, cms_matrix.m[c][c]);
    }

    // Tone map to prevent clipping when the source signal peak exceeds the
    // encodable range or we've reduced the gamut
    if (ref_peak > 1)
        pl_shader_tone_map(s, ref_peak, params);

    if (src.light != dst.light) {
        pl_shader_inverse_ootf(s, dst.light, pl_color_transfer_nominal_peak(dst.transfer));
    }

    // Warn for remaining out-of-gamut colors is enabled
    if (params->gamut_warning) {
        GLSL("if (any(greaterThan(color.rgb, vec3(1.01))) ||\n"
             "    any(lessThan(color.rgb, vec3(-0.01))))\n"
             "    color.rgb = vec3(1.0) - color.rgb;) // invert\n");
    }

    if (is_linear)
        pl_shader_delinearize(s, dst.transfer);

    GLSL("}\n");
}
