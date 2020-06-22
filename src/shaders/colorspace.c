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
#include "shaders.h"

void pl_shader_decode_color(struct pl_shader *sh, struct pl_color_repr *repr,
                            const struct pl_color_adjustment *params)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    GLSL("// pl_shader_decode_color \n"
         "{ \n");

    // For the non-linear color systems we need some special input handling
    // to make sure we don't accidentally screw everything up because of the
    // alpha multiplication, which only commutes with linear operations.
    bool is_nonlinear = !pl_color_system_is_linear(repr->sys);
    if (is_nonlinear && repr->alpha == PL_ALPHA_PREMULTIPLIED) {
        GLSL("color.rgb /= vec3(max(color.a, 1e-6));\n");
        repr->alpha = PL_ALPHA_INDEPENDENT;
    }

    // XYZ needs special handling due to the input gamma logic
    if (repr->sys == PL_COLOR_SYSTEM_XYZ) {
        float scale = pl_color_repr_normalize(repr);
        GLSL("color.rgb = pow(vec3(%f) * color.rgb, vec3(2.6));\n", scale);
    }

    enum pl_color_system orig_sys = repr->sys;
    struct pl_transform3x3 tr = pl_color_repr_decode(repr, params);

    ident_t cmat = sh_var(sh, (struct pl_shader_var) {
        .var  = pl_var_mat3("cmat"),
        .data = PL_TRANSPOSE_3X3(tr.mat.m),
    });

    ident_t cmat_c = sh_var(sh, (struct pl_shader_var) {
        .var  = pl_var_vec3("cmat_c"),
        .data = tr.c,
    });

    GLSL("color.rgb = %s * color.rgb + %s;\n", cmat, cmat_c);

    switch (orig_sys) {
    case PL_COLOR_SYSTEM_BT_2020_C:
        // Conversion for C'rcY'cC'bc via the BT.2020 CL system:
        // C'bc = (B'-Y'c) / 1.9404  | C'bc <= 0
        //      = (B'-Y'c) / 1.5816  | C'bc >  0
        //
        // C'rc = (R'-Y'c) / 1.7184  | C'rc <= 0
        //      = (R'-Y'c) / 0.9936  | C'rc >  0
        //
        // as per the BT.2020 specification, table 4. This is a non-linear
        // transformation because (constant) luminance receives non-equal
        // contributions from the three different channels.
        GLSL("// constant luminance conversion                                  \n"
             "color.br = color.br * mix(vec2(1.5816, 0.9936),                   \n"
             "                          vec2(1.9404, 1.7184),                   \n"
             "                          %s(lessThanEqual(color.br, vec2(0.0)))) \n"
             "           + color.gg;                                            \n",
             sh_bvec(sh, 2));
        // Expand channels to camera-linear light. This shader currently just
        // assumes everything uses the BT.2020 12-bit gamma function, since the
        // difference between 10 and 12-bit is negligible for anything other
        // than 12-bit content.
        GLSL("vec3 lin = mix(color.rgb * vec3(1.0/4.5),                        \n"
             "                pow((color.rgb + vec3(0.0993))*vec3(1.0/1.0993), \n"
             "                    vec3(1.0/0.45)),                             \n"
             "                %s(lessThanEqual(vec3(0.08145), color.rgb)));    \n",
             sh_bvec(sh, 3));
        // Calculate the green channel from the expanded RYcB, and recompress to G'
        // The BT.2020 specification says Yc = 0.2627*R + 0.6780*G + 0.0593*B
        GLSL("color.g = (lin.g - 0.2627*lin.r - 0.0593*lin.b)*1.0/0.6780;   \n"
             "color.g = mix(color.g * 4.5,                                  \n"
             "              1.0993 * pow(color.g, 0.45) - 0.0993,           \n"
             "              %s(0.0181 <= color.g));                         \n",
             sh_bvec(sh, 1));
        break;

    case PL_COLOR_SYSTEM_BT_2100_PQ:
    case PL_COLOR_SYSTEM_BT_2100_HLG: {
        // Conversion process from the spec:
        //
        // 1. L'M'S' = cmat * ICtCp
        // 2. LMS = linearize(L'M'S')  (EOTF for PQ, inverse OETF for HLG)
        // 3. RGB = lms2rgb * LMS
        //
        // After this we need to invert step 2 to arrive at non-linear RGB.
        // (It's important we keep the transfer function conversion separate
        // from the color system decoding, so we have to partially undo our
        // work here even though we will end up linearizing later on anyway)
        enum pl_color_transfer trc = orig_sys == PL_COLOR_SYSTEM_BT_2100_PQ
                                        ? PL_COLOR_TRC_PQ
                                        : PL_COLOR_TRC_HLG;

        // Inverted from the matrix in the spec, transposed to column major
        static const char *bt2100_lms2rgb = "mat3("
            "  3.43661,  -0.79133, -0.0259499, "
            " -2.50645,    1.9836, -0.0989137, "
            "0.0698454, -0.192271,    1.12486) ";

        pl_shader_linearize(sh, trc);
        GLSL("color.rgb = %s * color.rgb; \n", bt2100_lms2rgb);
        pl_shader_delinearize(sh, trc);
        break;
    }

    case PL_COLOR_SYSTEM_XYZ:
        break; // no special post-processing needed

    default:
        assert(pl_color_system_is_linear(orig_sys));
        break;
    }

    if (repr->alpha == PL_ALPHA_INDEPENDENT) {
        GLSL("color.rgb *= vec3(color.a);\n");
        repr->alpha = PL_ALPHA_PREMULTIPLIED;
    }

    // Gamma adjustment. Doing this here (in non-linear light) is technically
    // somewhat wrong, but this is just an aesthetic parameter and not really
    // meant for colorimetric precision, so we don't care too much.
    if (params && params->gamma != 1.0) {
        ident_t gamma = sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_float("gamma"),
            .data = &params->gamma,
        });
        GLSL("color.rgb = pow(color.rgb, vec3(%s)); \n", gamma);
    }

    GLSL("}\n");
}

void pl_shader_encode_color(struct pl_shader *sh,
                            const struct pl_color_repr *repr)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    GLSL("// pl_shader_encode_color \n"
         "{ \n");

    switch (repr->sys) {
    case PL_COLOR_SYSTEM_BT_2020_C:
        // Expand R'G'B' to RGB
        GLSL("vec3 lin = mix(color.rgb * vec3(1.0/4.5),                        \n"
             "                pow((color.rgb + vec3(0.0993))*vec3(1.0/1.0993), \n"
             "                    vec3(1.0/0.45)),                             \n"
             "                %s(lessThanEqual(vec3(0.08145), color.rgb)));    \n",
             sh_bvec(sh, 3));

        // Compute Yc from RGB and compress to R'Y'cB'
        GLSL("color.g = dot(vec3(0.2627, 0.6780, 0.0593), lin);     \n"
             "color.g = mix(color.g * 4.5,                          \n"
             "              1.0993 * pow(color.g, 0.45) - 0.0993,   \n"
             "              %s(0.0181 <= color.g));                 \n",
             sh_bvec(sh, 1));

        // Compute C'bc and C'rc into color.br
        GLSL("color.br = color.br - color.gg;                           \n"
             "color.br *= mix(vec2(1.0/1.5816, 1.0/0.9936),             \n"
             "                vec2(1.0/1.9404, 1.0/1.7184),             \n"
             "                %s(lessThanEqual(color.br, vec2(0.0))));  \n",
             sh_bvec(sh, 2));
        break;

    case PL_COLOR_SYSTEM_BT_2100_PQ:
    case PL_COLOR_SYSTEM_BT_2100_HLG: {
        enum pl_color_transfer trc = repr->sys == PL_COLOR_SYSTEM_BT_2100_PQ
                                        ? PL_COLOR_TRC_PQ
                                        : PL_COLOR_TRC_HLG;

        // Inverse of the matrix above
        static const char *bt2100_rgb2lms = "mat3("
            "0.412109, 0.166748, 0.024170, "
            "0.523925, 0.720459, 0.075440, "
            "0.063965, 0.112793, 0.900394) ";

        pl_shader_linearize(sh, trc);
        GLSL("color.rgb = %s * color.rgb; \n", bt2100_rgb2lms);
        pl_shader_delinearize(sh, trc);
        break;
    }

    case PL_COLOR_SYSTEM_XYZ:
        break; // no special pre-processing needed

    default:
        assert(pl_color_system_is_linear(repr->sys));
        break;
    }

    // Since this is a relatively rare operation, bypass it as much as possible
    bool skip = true;
    skip &= PL_DEF(repr->sys, PL_COLOR_SYSTEM_RGB) == PL_COLOR_SYSTEM_RGB;
    skip &= PL_DEF(repr->levels, PL_COLOR_LEVELS_PC) == PL_COLOR_LEVELS_PC;
    skip &= PL_DEF(repr->bits.sample_depth, 8) == PL_DEF(repr->bits.color_depth, 8);
    skip &= !repr->bits.bit_shift;

    if (!skip) {
        struct pl_color_repr copy = *repr;
        float xyzscale = (repr->sys == PL_COLOR_SYSTEM_XYZ)
                            ? pl_color_repr_normalize(&copy)
                            : 0.0;

        struct pl_transform3x3 tr = pl_color_repr_decode(&copy, NULL);
        pl_transform3x3_invert(&tr);

        ident_t cmat = sh_var(sh, (struct pl_shader_var) {
            .var  = pl_var_mat3("cmat"),
            .data = PL_TRANSPOSE_3X3(tr.mat.m),
        });

        ident_t cmat_c = sh_var(sh, (struct pl_shader_var) {
            .var  = pl_var_vec3("cmat_c"),
            .data = tr.c,
        });

        GLSL("color.rgb = %s * color.rgb + %s;\n", cmat, cmat_c);

        if (repr->sys == PL_COLOR_SYSTEM_XYZ)
            GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.6)) * vec3(1.0/%f); \n", xyzscale);
    }

    if (repr->alpha == PL_ALPHA_INDEPENDENT)
        GLSL("color.rgb /= vec3(max(color.a, 1e-6));\n");

    GLSL("}\n");
}

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

void pl_shader_linearize(struct pl_shader *sh, enum pl_color_transfer trc)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    if (trc == PL_COLOR_TRC_LINEAR)
        return;

    // Note that this clamp may technically violate the definition of
    // ITU-R BT.2100, which allows for sub-blacks and super-whites to be
    // displayed on the display where such would be possible. That said, the
    // problem is that not all gamma curves are well-defined on the values
    // outside this range, so we ignore it and just clamp anyway for sanity.
    GLSL("// pl_shader_linearize           \n"
         "color.rgb = max(color.rgb, 0.0); \n");

    switch (trc) {
    case PL_COLOR_TRC_SRGB:
        GLSL("color.rgb = mix(color.rgb * vec3(1.0/12.92),               \n"
             "                pow((color.rgb + vec3(0.055))/vec3(1.055), \n"
             "                    vec3(2.4)),                            \n"
             "                %s(lessThan(vec3(0.04045), color.rgb)));   \n",
             sh_bvec(sh, 3));
        break;
    case PL_COLOR_TRC_BT_1886:
        GLSL("color.rgb = pow(color.rgb, vec3(2.4));\n");
        break;
    case PL_COLOR_TRC_GAMMA18:
        GLSL("color.rgb = pow(color.rgb, vec3(1.8));\n");
        break;
    case PL_COLOR_TRC_UNKNOWN:
    case PL_COLOR_TRC_GAMMA22:
        GLSL("color.rgb = pow(color.rgb, vec3(2.2));\n");
        break;
    case PL_COLOR_TRC_GAMMA28:
        GLSL("color.rgb = pow(color.rgb, vec3(2.8));\n");
        break;
    case PL_COLOR_TRC_PRO_PHOTO:
        GLSL("color.rgb = mix(color.rgb * vec3(1.0/16.0),              \n"
             "                pow(color.rgb, vec3(1.8)),               \n"
             "                %s(lessThan(vec3(0.03125), color.rgb))); \n",
             sh_bvec(sh, 3));
        break;
    case PL_COLOR_TRC_PQ:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/%f));         \n"
             "color.rgb = max(color.rgb - vec3(%f), 0.0)        \n"
             "             / (vec3(%f) - vec3(%f) * color.rgb); \n"
             "color.rgb = pow(color.rgb, vec3(1.0/%f));         \n"
             // PQ's output range is 0-10000, but we need it to be relative to
             // to PL_COLOR_SDR_WHITE instead, so rescale
             "color.rgb *= vec3(%f);                            \n",
             PQ_M2, PQ_C1, PQ_C2, PQ_C3, PQ_M1, 10000.0 / PL_COLOR_SDR_WHITE);
        break;
    case PL_COLOR_TRC_HLG:
        GLSL("color.rgb = mix(vec3(4.0) * color.rgb * color.rgb,         \n"
             "                exp((color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "                    + vec3(%f),                            \n"
             "                %s(lessThan(vec3(0.5), color.rgb)));       \n"
             // Rescale from 0-12 to be relative to PL_COLOR_SDR_WHITE_HLG
             "color.rgb *= vec3(1.0/%f);                                 \n",
             HLG_C, HLG_A, HLG_B, sh_bvec(sh, 3), PL_COLOR_SDR_WHITE_HLG);
        break;
    case PL_COLOR_TRC_V_LOG:
        GLSL("color.rgb = mix((color.rgb - vec3(0.125)) * vec3(1.0/5.6), \n"
             "    pow(vec3(10.0), (color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "              - vec3(%f),                                  \n"
             "    %s(lessThanEqual(vec3(0.181), color.rgb)));            \n",
             VLOG_D, VLOG_C, VLOG_B, sh_bvec(sh, 3));
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
             "    %s(lessThanEqual(vec3(%f), color.rgb)));                \n",
             SLOG_Q, SLOG_P, SLOG_C, SLOG_A, SLOG_B, SLOG_K2, sh_bvec(sh, 3),
             SLOG_Q);
        break;
    default:
        abort();
    }
}

void pl_shader_delinearize(struct pl_shader *sh, enum pl_color_transfer trc)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    if (trc == PL_COLOR_TRC_LINEAR)
        return;

    GLSL("// pl_shader_delinearize         \n"
         "color.rgb = max(color.rgb, 0.0); \n");

    switch (trc) {
    case PL_COLOR_TRC_SRGB:
        GLSL("color.rgb = mix(color.rgb * vec3(12.92),                        \n"
             "                vec3(1.055) * pow(color.rgb, vec3(1.0/2.4))     \n"
             "                    - vec3(0.055),                              \n"
             "                %s(lessThanEqual(vec3(0.0031308), color.rgb))); \n",
             sh_bvec(sh, 3));
        break;
    case PL_COLOR_TRC_BT_1886:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.4));\n");
        break;
    case PL_COLOR_TRC_GAMMA18:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/1.8));\n");
        break;
    case PL_COLOR_TRC_UNKNOWN:
    case PL_COLOR_TRC_GAMMA22:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.2));\n");
        break;
    case PL_COLOR_TRC_GAMMA28:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.8));\n");
        break;
    case PL_COLOR_TRC_PRO_PHOTO:
        GLSL("color.rgb = mix(color.rgb * vec3(16.0),                        \n"
             "                pow(color.rgb, vec3(1.0/1.8)),                 \n"
             "                %s(lessThanEqual(vec3(0.001953), color.rgb))); \n",
             sh_bvec(sh, 3));
        break;
    case PL_COLOR_TRC_PQ:
        GLSL("color.rgb *= vec3(1.0/%f);                         \n"
             "color.rgb = pow(color.rgb, vec3(%f));              \n"
             "color.rgb = (vec3(%f) + vec3(%f) * color.rgb)      \n"
             "             / (vec3(1.0) + vec3(%f) * color.rgb); \n"
             "color.rgb = pow(color.rgb, vec3(%f));              \n",
             10000 / PL_COLOR_SDR_WHITE, PQ_M1, PQ_C1, PQ_C2, PQ_C3, PQ_M2);
        break;
    case PL_COLOR_TRC_HLG:
        GLSL("color.rgb *= vec3(%f);                                           \n"
             "color.rgb = mix(vec3(0.5) * sqrt(color.rgb),                     \n"
             "                vec3(%f) * log(color.rgb - vec3(%f)) + vec3(%f), \n"
             "                %s(lessThan(vec3(1.0), color.rgb)));             \n",
             PL_COLOR_SDR_WHITE_HLG, HLG_A, HLG_B, HLG_C, sh_bvec(sh, 3));
        break;
    case PL_COLOR_TRC_V_LOG:
        GLSL("color.rgb = mix(vec3(5.6) * color.rgb + vec3(0.125),       \n"
             "                vec3(%f) * log(color.rgb + vec3(%f))       \n"
             "                    + vec3(%f),                            \n"
             "                %s(lessThanEqual(vec3(0.01), color.rgb))); \n",
             VLOG_C / M_LN10, VLOG_B, VLOG_D, sh_bvec(sh, 3));
        break;
    case PL_COLOR_TRC_S_LOG1:
        GLSL("color.rgb = vec3(%f) * log(color.rgb + vec3(%f)) + vec3(%f);\n",
             SLOG_A / M_LN10, SLOG_B, SLOG_C);
        break;
    case PL_COLOR_TRC_S_LOG2:
        GLSL("color.rgb = mix(vec3(%f) * color.rgb + vec3(%f),                \n"
             "                vec3(%f) * log(vec3(%f) * color.rgb + vec3(%f)) \n"
             "                    + vec3(%f),                                 \n"
             "                %s(lessThanEqual(vec3(0.0), color.rgb)));       \n",
             SLOG_P, SLOG_Q, SLOG_A / M_LN10, SLOG_K2, SLOG_B, SLOG_C,
             sh_bvec(sh, 3));
        break;
    default:
        abort();
    }
}

const struct pl_sigmoid_params pl_sigmoid_default_params = {
    .center = 0.75,
    .slope  = 6.50,
};

void pl_shader_sigmoidize(struct pl_shader *sh,
                          const struct pl_sigmoid_params *params)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    params = PL_DEF(params, &pl_sigmoid_default_params);
    float center = PL_DEF(params->center, 0.75);
    float slope  = PL_DEF(params->slope, 6.5);

    // This function needs to go through (0,0) and (1,1), so we compute the
    // values at 1 and 0, and then scale/shift them, respectively.
    float offset = 1.0 / (1 + expf(slope * center));
    float scale  = 1.0 / (1 + expf(slope * (center - 1))) - offset;

    GLSL("// pl_shader_sigmoidize                                          \n"
         "color = clamp(color, 0.0, 1.0);                                  \n"
         "color = vec4(%f) - log(vec4(1.0) / (color * vec4(%f) + vec4(%f)) \n"
         "                         - vec4(1.0)) * vec4(%f);                \n",
         center, scale, offset, 1.0 / slope);
}

void pl_shader_unsigmoidize(struct pl_shader *sh,
                            const struct pl_sigmoid_params *params)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    // See: pl_shader_sigmoidize
    params = PL_DEF(params, &pl_sigmoid_default_params);
    float center = PL_DEF(params->center, 0.75);
    float slope  = PL_DEF(params->slope, 6.5);
    float offset = 1.0 / (1 + expf(slope * center));
    float scale  = 1.0 / (1 + expf(slope * (center - 1))) - offset;

    GLSL("// pl_shader_unsigmoidize                                           \n"
         "color = clamp(color, 0.0, 1.0);                                     \n"
         "color = vec4(%f) / (vec4(1.0) + exp(vec4(%f) * (vec4(%f) - color))) \n"
         "           - vec4(%f);                                              \n",
         1.0 / scale, slope, center, offset / scale);
}

static ident_t sh_luma_coeffs(struct pl_shader *sh, enum pl_color_primaries prim)
{
    struct pl_matrix3x3 rgb2xyz;
    rgb2xyz = pl_get_rgb2xyz_matrix(pl_raw_primaries_get(prim));
    return sh_var(sh, (struct pl_shader_var) {
        .var = pl_var_vec3("luma_coeffs"),
        .data = rgb2xyz.m[1], // RGB->Y vector
    });
}

// Applies the OOTF / inverse OOTF - including the sig_scale adaptation
static void pl_shader_ootf(struct pl_shader *sh, struct pl_color_space csp)
{
    if (csp.sig_scale != 1.0)
        GLSL("color.rgb *= vec3(%f); \n", csp.sig_scale);

    if (!csp.light || csp.light == PL_COLOR_LIGHT_DISPLAY)
        return;

    GLSL("// pl_shader_ootf                \n"
         "color.rgb = max(color.rgb, 0.0); \n");

    switch (csp.light)
    {
    case PL_COLOR_LIGHT_SCENE_HLG: {
        // HLG OOTF from BT.2100, tuned to the indicated peak
        float peak = csp.sig_peak * PL_COLOR_SDR_WHITE;
        float gamma = 1.2 + 0.42 * log10(peak / 1000.0);
        gamma = PL_MAX(gamma, 1.0);
        GLSL("color.rgb *= vec3(%f * pow(dot(%s, color.rgb), %f));\n",
             csp.sig_peak / pow(12.0 / PL_COLOR_SDR_WHITE_HLG, gamma),
             sh_luma_coeffs(sh, csp.primaries),
             gamma - 1.0);
        break;
    }
    case PL_COLOR_LIGHT_SCENE_709_1886:
        // This OOTF is defined by encoding the result as 709 and then decoding
        // it as 1886; although this is called 709_1886 we actually use the
        // more precise (by one decimal) values from BT.2020 instead
        GLSL("color.rgb = mix(color.rgb * vec3(4.5),                    \n"
             "                vec3(1.0993) * pow(color.rgb, vec3(0.45)) \n"
             "                             - vec3(0.0993),              \n"
             "                %s(lessThan(vec3(0.0181), color.rgb)));   \n"
             "color.rgb = pow(color.rgb, vec3(2.4));                    \n",
             sh_bvec(sh, 3));
        break;
    case PL_COLOR_LIGHT_SCENE_1_2:
        GLSL("color.rgb = pow(color.rgb, vec3(1.2));\n");
        break;
    default:
        abort();
    }
}

static void pl_shader_inverse_ootf(struct pl_shader *sh, struct pl_color_space csp)
{
    if (!csp.light || csp.light == PL_COLOR_LIGHT_DISPLAY)
        goto skip;

    GLSL("// pl_shader_inverse_ootf        \n"
         "color.rgb = max(color.rgb, 0.0); \n");

    switch (csp.light)
    {
    case PL_COLOR_LIGHT_SCENE_HLG: {
        float peak = csp.sig_peak * PL_COLOR_SDR_WHITE;
        float gamma = 1.2 + 0.42 * log10(peak / 1000.0);
        gamma = PL_MAX(gamma, 1.0);
        GLSL("color.rgb *= vec3(1.0/%f);                                \n"
             "color.rgb /= vec3(max(1e-6, pow(dot(%s, color.rgb),       \n"
             "                                %f)));                    \n",
             csp.sig_peak / pow(12.0 / PL_COLOR_SDR_WHITE_HLG, gamma),
             sh_luma_coeffs(sh, csp.primaries),
             (gamma - 1.0) / gamma);
        break;
    }
    case PL_COLOR_LIGHT_SCENE_709_1886:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.4));                         \n"
             "color.rgb = mix(color.rgb * vec3(1.0/4.5),                         \n"
             "                pow((color.rgb + vec3(0.0993)) * vec3(1.0/1.0993), \n"
             "                    vec3(1/0.45)),                                 \n"
             "                %s(lessThan(vec3(0.08145), color.rgb)));           \n",
             sh_bvec(sh, 3));
        break;
    case PL_COLOR_LIGHT_SCENE_1_2:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/1.2));\n");
        break;
    default:
        abort();
    }

skip:
    if (csp.sig_scale != 1.0)
        GLSL("color.rgb *= vec3(1.0 / %f); \n", csp.sig_scale);
}

const struct pl_peak_detect_params pl_peak_detect_default_params = {
    .smoothing_period       = 100.0,
    .scene_threshold_low    = 5.5,
    .scene_threshold_high   = 10.0,
    .overshoot_margin       = 0.05,
};

struct sh_peak_obj {
    const struct pl_buf *buf;
    const struct pl_buf *buf_read;
    struct pl_shader_desc desc;
    float margin;
};

static void sh_peak_uninit(const struct pl_gpu *gpu, void *ptr)
{
    struct sh_peak_obj *obj = ptr;
    pl_buf_destroy(gpu, &obj->buf);
    pl_buf_destroy(gpu, &obj->buf_read);
    *obj = (struct sh_peak_obj) {0};
}

static inline float iir_coeff(float rate)
{
    float a = 1.0 - cos(1.0 / rate);
    return sqrt(a*a + 2*a) - a;
}

bool pl_shader_detect_peak(struct pl_shader *sh,
                           struct pl_color_space csp,
                           struct pl_shader_obj **state,
                           const struct pl_peak_detect_params *params)
{
    params = PL_DEF(params, &pl_peak_detect_default_params);
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return false;

    if (!sh_try_compute(sh, 8, 8, true, 2 * sizeof(int32_t))) {
        PL_ERR(sh, "HDR peak detection requires compute shaders!");
        return false;
    }

    if (sh_glsl(sh).version < 130) {
        // uint was added in GLSL 130
        PL_ERR(sh, "HDR peak detection requires GLSL >= 130!");
        return false;
    }

    struct sh_peak_obj *obj;
    obj = SH_OBJ(sh, state, PL_SHADER_OBJ_PEAK_DETECT, struct sh_peak_obj,
                 sh_peak_uninit);
    if (!obj)
        return false;

    const struct pl_gpu *gpu = SH_GPU(sh);
    obj->margin = params->overshoot_margin;

    if (!obj->buf) {
        obj->desc = (struct pl_shader_desc) {
            .desc = {
                .name   = "PeakDetect",
                .type   = PL_DESC_BUF_STORAGE,
            },
        };

        // Note: Don't change this order, `vec2 average` being the first
        // element is hard-coded in `pl_get_detected_peak`
        bool ok = true;
        ok &= sh_buf_desc_append(obj, gpu, &obj->desc, NULL, pl_var_vec2("average"));
        ok &= sh_buf_desc_append(obj, gpu, &obj->desc, NULL, pl_var_int("frame_sum"));
        ok &= sh_buf_desc_append(obj, gpu, &obj->desc, NULL, pl_var_int("frame_max"));
        ok &= sh_buf_desc_append(obj, gpu, &obj->desc, NULL, pl_var_uint("counter"));

        if (!ok) {
            PL_ERR(sh, "HDR peak detection exhausts device limits!");
            return false;
        }

        // Create the SSBO
        size_t size = sh_buf_desc_size(&obj->desc);
        static const uint8_t zero[32] = {0};
        pl_assert(sizeof(zero) >= size);
        obj->buf = pl_buf_create(gpu, &(struct pl_buf_params) {
            .type = PL_BUF_STORAGE,
            .size = size,
            .initial_data = zero,
        });
        obj->desc.object = obj->buf;
    }

    if (!obj->buf) {
        SH_FAIL(sh, "Failed creating peak detection SSBO!");
        return false;
    }

    // Attach the SSBO and perform the peak detection logic
    obj->desc.desc.access = PL_DESC_ACCESS_READWRITE;
    obj->desc.memory = PL_MEMORY_COHERENT;
    sh_desc(sh, obj->desc);
    GLSL("// pl_shader_detect_peak \n"
         "{                        \n"
         "vec4 color_orig = color; \n");

    // Decode the color into linear light absolute scale representation
    pl_color_space_infer(&csp);
    pl_shader_linearize(sh, csp.transfer);
    pl_shader_ootf(sh, csp);

    // For performance, we want to do as few atomic operations on global
    // memory as possible, so use an atomic in shmem for the work group.
    ident_t wg_sum = sh_fresh(sh, "wg_sum"), wg_max = sh_fresh(sh, "wg_max");
    GLSLH("shared int %s;   \n", wg_sum);
    GLSLH("shared int %s;   \n", wg_max);
    GLSL("%s = 0; %s = 0;   \n"
         "barrier();        \n",
         wg_sum, wg_max);

    // Chosen to avoid overflowing on an 8K buffer
    const float log_min = 1e-3, log_scale = 400.0, sig_scale = 10000.0;

    GLSL("float sig_max = max(max(color.r, color.g), color.b);  \n"
         "float sig_log = log(max(sig_max, %f));                \n"
         "int isig_max = int(sig_max * %f);                     \n"
         "int isig_log = int(sig_log * %f);                     \n",
         log_min, sig_scale, log_scale);

    // Update the work group's shared atomics
    if (gpu->caps & PL_GPU_CAP_SUBGROUPS && false) {
        GLSL("int group_max = subgroupMax(isig_max);    \n"
             "int group_sum = subgroupAdd(isig_log);    \n"
             "if (subgroupElect()) {                    \n"
             "    atomicMax(%s, group_max);             \n"
             "    atomicAdd(%s, group_sum);             \n"
             "    memoryBarrierShared();                \n"
             "}                                         \n"
             "barrier();                                \n",
             wg_max, wg_sum);
    } else {
        GLSL("atomicMax(%s, isig_max);  \n"
             "atomicAdd(%s, isig_log);  \n"
             "memoryBarrierShared();    \n"
             "barrier();                \n",
             wg_max, wg_sum);
    }

    GLSL("color = color_orig;   \n"
         "}                     \n");

    // Have one thread per work group update the global atomics. Do this
    // at the end of the shader to avoid clobbering `average`, in case the
    // state object will be used by the same pass.
    GLSLF("// pl_shader_detect_peak                                             \n"
          "if (gl_LocalInvocationIndex == 0u) {                                 \n"
          "    int wg_avg = %s / int(gl_WorkGroupSize.x * gl_WorkGroupSize.y);  \n"
          "    atomicAdd(frame_sum, wg_avg);                                    \n"
          "    atomicMax(frame_max, %s);                                        \n"
          "    memoryBarrierBuffer();                                           \n"
          "    barrier();                                                       \n",
          wg_sum, wg_max);

    // Finally, to update the global state per dispatch, we increment a counter
    GLSLF("    uint num_wg = gl_NumWorkGroups.x * gl_NumWorkGroups.y;           \n"
          "    if (atomicAdd(counter, 1u) == num_wg - 1u) {                     \n"
          "        vec2 cur = vec2(float(frame_sum) / float(num_wg), frame_max);\n"
          "        cur *= vec2(1.0 / %f, 1.0 / %f);                             \n"
          "        cur.x = exp(cur.x);                                          \n",
          log_scale, sig_scale);

    // Set the initial value accordingly if it contains no data
    GLSLF("        if (average.y == 0.0) \n"
          "            average = cur;    \n");

    // Use an IIR low-pass filter to smooth out the detected values
    GLSLF("        average += %f * (cur - average); \n",
          iir_coeff(PL_DEF(params->smoothing_period, 100.0)));

    // Scene change hysteresis
    float log_db = 10.0 / log(10.0);
    if (params->scene_threshold_low > 0 && params->scene_threshold_high > 0) {
        GLSLF("    float delta = abs(log(cur.x / average.x));               \n"
              "    average = mix(average, cur, smoothstep(%f, %f, delta));  \n",
              params->scene_threshold_low / log_db,
              params->scene_threshold_high / log_db);
    }

    // Reset SSBO state for the next frame
    GLSLF("        frame_sum = 0;            \n"
          "        frame_max = 0;            \n"
          "        counter = 0u;             \n"
          "        memoryBarrierBuffer();    \n"
          "    }                             \n"
          "}                                 \n");

    return true;
}

bool pl_get_detected_peak(const struct pl_shader_obj *state,
                          float *out_peak, float *out_avg)
{
    if (!state || state->type != PL_SHADER_OBJ_PEAK_DETECT)
        return false;

    struct sh_peak_obj *obj = state->priv;
    const struct pl_gpu *gpu = state->gpu;

    float average[2];
    pl_assert(obj->buf->params.size >= sizeof(average));

    bool ok = pl_buf_recreate(gpu, &obj->buf_read, &(struct pl_buf_params) {
        .type = PL_BUF_TEX_TRANSFER,
        .size = sizeof(average),
        .host_readable = true,
        .memory_type = PL_BUF_MEM_HOST,
    });

    if (!ok) {
        PL_ERR(gpu, "Failed creating peak detect readback buffer");
        return false;
    }

    pl_buf_copy(gpu, obj->buf_read, 0, obj->buf, 0, sizeof(average));
    if (!pl_buf_read(gpu, obj->buf_read, 0, average, sizeof(average))) {
        PL_ERR(gpu, "Failed reading from peak detect state buffer");
        return false;
    }

    *out_avg = average[0];
    *out_peak = average[1];

    if (obj->margin > 0.0) {
        *out_peak *= 1.0 + obj->margin;
        *out_peak = PL_MIN(*out_peak, 10000 / PL_COLOR_SDR_WHITE);
    }

    return true;
}

static inline float pq_delinearize(float x)
{
    x *= PL_COLOR_SDR_WHITE / 10000.0;
    x = powf(x, PQ_M1);
    x = (PQ_C1 + PQ_C2 * x) / (1.0 + PQ_C3 * x);
    x = pow(x, PQ_M2);
    return x;
}

const struct pl_color_map_params pl_color_map_default_params = {
    .intent                 = PL_INTENT_RELATIVE_COLORIMETRIC,
    .tone_mapping_algo      = PL_TONE_MAPPING_BT_2390,
    .desaturation_strength  = 0.75,
    .desaturation_exponent  = 1.50,
    .desaturation_base      = 0.18,
    .gamut_clipping         = true,
};

static void pl_shader_tone_map(struct pl_shader *sh, struct pl_color_space src,
                               struct pl_color_space dst,
                               struct pl_shader_obj **peak_detect_state,
                               const struct pl_color_map_params *params)
{
    GLSL("// pl_shader_tone_map \n"
         "{                     \n");

    // To prevent discoloration due to out-of-bounds clipping, we need to make
    // sure to reduce the value range as far as necessary to keep the entire
    // signal in range, so tone map based on the brightest component.
    GLSL("int sig_idx = 0;                              \n"
         "if (color[1] > color[sig_idx]) sig_idx = 1;   \n"
         "if (color[2] > color[sig_idx]) sig_idx = 2;   \n"
         "float sig_max = color[sig_idx];               \n"
         "float sig_peak = %f;                          \n"
         "float sig_avg = %f;                           \n",
         src.sig_peak * src.sig_scale,
         src.sig_avg * src.sig_scale);

    // Update the variables based on values from the peak detection buffer
    if (peak_detect_state) {
        struct sh_peak_obj *obj;
        obj = SH_OBJ(sh, peak_detect_state, PL_SHADER_OBJ_PEAK_DETECT,
                     struct sh_peak_obj, sh_peak_uninit);
        if (obj && obj->buf) {
            obj->desc.desc.access = PL_DESC_ACCESS_READONLY;
            obj->desc.memory = 0;
            sh_desc(sh, obj->desc);
            GLSL("sig_avg  = average.x; \n"
                 "sig_peak = average.y; \n");

            // Apply a tiny bit of extra margin of error for overshoot to the
            // smoothed peak values, clamped to the maximum reasonable range.
            if (obj->margin > 0.0) {
                GLSL("sig_peak = min(sig_peak * %f, %f); \n",
                     1.0 + obj->margin,
                     10000 / PL_COLOR_SDR_WHITE);
            }
        }
    }

    // Rescale the input in order to bring it into a representation where 1.0
    // represents the dst_peak. This is because (almost) all of the tone
    // mapping algorithms are defined in such a way that they map to the range
    // [0.0, 1.0].
    bool need_norm = params->tone_mapping_algo != PL_TONE_MAPPING_BT_2390;
    float dst_range = dst.sig_peak * dst.sig_scale;
    if (dst_range > 1.0 && need_norm) {
        GLSL("color.rgb *= 1.0 / %f; \n"
             "sig_peak *= 1.0 / %f;  \n",
             dst_range, dst_range);
    }

    // Rename `color.rgb` to something shorter for conciseness
    GLSL("vec3 sig = color.rgb; \n"
         "vec3 sig_orig = sig;  \n");

    // Scale the signal to compensate for differences in the average brightness
    GLSL("float slope = min(%f, %f / sig_avg); \n"
         "sig *= slope;                        \n"
         "sig_peak *= slope;                   \n",
         PL_DEF(params->max_boost, 1.0), dst.sig_avg * dst.sig_scale);

    float param = params->tone_mapping_param;
    switch (params->tone_mapping_algo) {
    case PL_TONE_MAPPING_CLIP:
        GLSL("sig *= %f;\n", PL_DEF(param, 1.0));
        break;

    case PL_TONE_MAPPING_MOBIUS:
        // Mobius isn't well-defined for sig_peak <= 1.0, but the limit of
        // mobius as sig_peak -> 1.0 is a linear function, so we can just skip
        // tone-mapping in this case
        GLSL("if (sig_peak > 1.0 + 1e-6) {                                      \n"
             "    const float j = %f;                                           \n"
             // solve for M(j) = j; M(sig_peak) = 1.0; M'(j) = 1.0
             // where M(x) = scale * (x+a)/(x+b)
             "    float a = -j*j * (sig_peak - 1.0) / (j*j - 2.0*j + sig_peak); \n"
             "    float b = (j*j - 2.0*j*sig_peak + sig_peak) /                 \n"
             "              max(1e-6, sig_peak - 1.0);                          \n"
             "    float scale = (b*b + 2.0*b*j + j*j) / (b-a);                  \n"
             "    sig = mix(sig, scale * (sig + vec3(a)) / (sig + vec3(b)),     \n"
             "              %s(greaterThan(sig, vec3(j))));                     \n"
             "}                                                                 \n",
             PL_DEF(param, 0.3),
             sh_bvec(sh, 3));
        break;

    case PL_TONE_MAPPING_REINHARD: {
        float contrast = PL_DEF(param, 0.5),
              offset = (1.0 - contrast) / contrast;
        GLSL("sig = sig / (sig + vec3(%f));             \n"
             "float scale = (sig_peak + %f) / sig_peak; \n"
             "sig *= scale;                             \n",
             offset, offset);
        break;
    }

    case PL_TONE_MAPPING_HABLE: {
        float A = 0.15, B = 0.50, C = 0.10, D = 0.20, E = 0.02, F = 0.30;
        ident_t hable = sh_fresh(sh, "hable");
        GLSLH("vec3 %s(vec3 x) {                                \n"
              "    return (x * (%f*x + vec3(%f)) + vec3(%f)) /  \n"
              "           (x * (%f*x + vec3(%f)) + vec3(%f))    \n"
              "           - vec3(%f);                           \n"
              "}                                                \n",
              hable, A, C*B, D*E, A, B, D*F, E/F);
        GLSL("sig = %s(sig) / %s(vec3(sig_peak)).x;\n", hable, hable);
        break;
    }

    case PL_TONE_MAPPING_GAMMA:
        GLSL("const float cutoff = 0.05, gamma = 1.0/%f;            \n"
             "float scale = pow(cutoff / sig_peak, gamma) / cutoff; \n"
             "sig = mix(scale * sig,                                \n"
             "          pow(sig / sig_peak, vec3(gamma)),           \n"
             "          %s(greaterThan(sig, vec3(cutoff))));        \n",
             PL_DEF(param, 1.8),
             sh_bvec(sh, 3));
        break;

    case PL_TONE_MAPPING_LINEAR:
        GLSL("sig *= %f / sig_peak;\n", PL_DEF(param, 1.0));
        break;

    case PL_TONE_MAPPING_BT_2390:
        // We first need to encode both sig and sig_peak into PQ space
        GLSL("vec4 sig_pq = vec4(sig.rgb, sig_peak);                            \n"
             "sig_pq *= vec4(1.0/%f);                                           \n"
             "sig_pq = pow(sig_pq, vec4(%f));                                   \n"
             "sig_pq = (vec4(%f) + vec4(%f) * sig_pq)                           \n"
             "          / (vec4(1.0) + vec4(%f) * sig_pq);                      \n"
             "sig_pq = pow(sig_pq, vec4(%f));                                   \n",
             10000 / PL_COLOR_SDR_WHITE, PQ_M1, PQ_C1, PQ_C2, PQ_C3, PQ_M2);
        // Encode both the signal and the target brightness to be relative to
        // the source peak brightness, and figure out the target peak in this space
        GLSL("float scale = 1.0 / sig_pq.a;                                     \n"
             "sig_pq.rgb *= vec3(scale);                                        \n"
             "float maxLum = %f * scale;                                        \n",
             pq_delinearize(dst_range));
        // Apply piece-wise hermite spline
        GLSL("float ks = 1.5 * maxLum - 0.5;                                    \n"
             "vec3 tb = (sig_pq.rgb - vec3(ks)) / vec3(1.0 - ks);               \n"
             "vec3 tb2 = tb * tb;                                               \n"
             "vec3 tb3 = tb2 * tb;                                              \n"
             "vec3 pb = (2.0 * tb3 - 3.0 * tb2 + vec3(1.0)) * vec3(ks) +        \n"
             "          (tb3 - 2.0 * tb2 + tb) * vec3(1.0 - ks) +               \n"
             "          (-2.0 * tb3 + 3.0 * tb2) * vec3(maxLum);                \n"
             "sig = mix(pb, sig_pq.rgb, %s(lessThan(sig_pq.rgb, vec3(ks))));    \n",
             sh_bvec(sh, 3));
        // Convert back from PQ space to linear light
        GLSL("sig *= vec3(sig_pq.a);                                            \n"
             "sig = pow(sig, vec3(1.0/%f));                                     \n"
             "sig = max(sig - vec3(%f), 0.0) /                                  \n"
             "          (vec3(%f) - vec3(%f) * sig);                            \n"
             "sig = pow(sig, vec3(1.0/%f));                                     \n"
             "sig *= vec3(%f);                                                  \n",
             PQ_M2, PQ_C1, PQ_C2, PQ_C3, PQ_M1, 10000.0 / PL_COLOR_SDR_WHITE);
        break;

    default:
        abort();
    }

    GLSL("sig = min(sig, 1.01);                                         \n"
         "vec3 sig_lin = sig_orig * (sig[sig_idx] / sig_orig[sig_idx]); \n");

    // Mix between the per-channel tone mapped `sig` and the linear tone
    // mapped `sig_lin` based on the desaturation strength
    if (params->desaturation_strength > 0.0) {
        GLSL("float coeff = max(sig[sig_idx] - %f, 1e-6) /  \n"
             "              max(sig[sig_idx], 1.0);         \n"
             "coeff = %f * pow(coeff, %f);                  \n"
             "color.rgb = mix(sig_lin, sig, coeff);         \n",
             params->desaturation_base,
             params->desaturation_strength,
             params->desaturation_exponent);
    } else {
        GLSL("color.rgb = sig_lin; \n");
    }

    // Undo the normalization by `dst_peak`
    if (dst_range > 1.0 && need_norm)
        GLSL("color.rgb *= %f; \n", dst_range);

    GLSL("} \n");
}

void pl_shader_color_map(struct pl_shader *sh,
                         const struct pl_color_map_params *params,
                         struct pl_color_space src, struct pl_color_space dst,
                         struct pl_shader_obj **peak_detect_state,
                         bool prelinearized)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    GLSL("// pl_shader_color_map\n");
    GLSL("{\n");
    params = PL_DEF(params, &pl_color_map_default_params);

    // Default the source color space to reasonable values
    pl_color_space_infer(&src);

    // To be as conservative as possible, color mapping is disabled by default
    // except for special cases which are considered to be "sufficiently
    // different" from the source space. For primaries, this means anything
    // wide gamut; and for transfers, this means anything radically different
    // from the typical SDR curves.
    if (!dst.primaries) {
        dst.primaries = src.primaries;
        if (pl_color_primaries_is_wide_gamut(dst.primaries))
            dst.primaries = PL_COLOR_PRIM_BT_709;
    }

    if (!dst.transfer) {
        dst.transfer = src.transfer;
        if (pl_color_transfer_is_hdr(dst.transfer) ||
            dst.transfer == PL_COLOR_TRC_LINEAR)
        {
            dst.transfer = PL_COLOR_TRC_GAMMA22;
        }
    }

    // Defaults the dest average based on the source average, unless the source
    // is HDR and the destination is not
    if (!dst.sig_avg) {
        bool src_hdr = pl_color_space_is_hdr(src);
        bool dst_hdr = pl_color_space_is_hdr(dst);
        if (!(src_hdr && !dst_hdr))
            dst.sig_avg = src.sig_avg;
    }

    // Infer the remaining fields after making the above choices
    pl_color_space_infer(&dst);

    // All operations from here on require linear light as a starting point,
    // so we linearize even if src.transfer == dst.transfer when one of the other
    // operations needs it
    bool need_linear = src.transfer != dst.transfer ||
                       src.primaries != dst.primaries ||
                       src.sig_peak > dst.sig_peak ||
                       src.sig_avg != dst.sig_avg ||
                       src.sig_scale != dst.sig_scale ||
                       src.light != dst.light;
    bool need_gamut_warn = false;
    bool is_linear = prelinearized;
    if (need_linear && !is_linear) {
        pl_shader_linearize(sh, src.transfer);
        is_linear = true;
    }

    if (need_linear)
        pl_shader_ootf(sh, src);

    // Tone map to rescale the signal average/peak if needed
    if (src.sig_peak * src.sig_scale > dst.sig_peak * dst.sig_scale + 1e-6) {
        pl_shader_tone_map(sh, src, dst, peak_detect_state, params);
        need_gamut_warn = true;
    }

    // Adapt to the right colorspace (primaries) if necessary
    if (src.primaries != dst.primaries) {
        const struct pl_raw_primaries *csp_src, *csp_dst;
        csp_src = pl_raw_primaries_get(src.primaries),
        csp_dst = pl_raw_primaries_get(dst.primaries);
        struct pl_matrix3x3 cms_mat;
        cms_mat = pl_get_color_mapping_matrix(csp_src, csp_dst, params->intent);

        GLSL("color.rgb = %s * color.rgb;\n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_mat3("cms_matrix"),
            .data = PL_TRANSPOSE_3X3(cms_mat.m),
        }));

        if (!pl_primaries_superset(csp_dst, csp_src)) {
            if (params->gamut_clipping) {
                GLSL("float cmin = min(min(color.r, color.g), color.b);     \n"
                     "if (cmin < 0.0) {                                     \n"
                     "    float luma = dot(%s, color.rgb);                  \n"
                     "    float coeff = cmin / (cmin - luma);               \n"
                     "    color.rgb = mix(color.rgb, vec3(luma), coeff);    \n"
                     "}                                                     \n"
                     "float cmax = max(max(color.r, color.g), color.b);     \n"
                     "if (cmax > 1.0)                                       \n"
                     "    color.rgb /= cmax;                                \n",
                     sh_luma_coeffs(sh, dst.primaries));

            } else {
                need_gamut_warn = true;
            }
        }
    }

    // Warn for remaining out-of-gamut colors if enabled
    if (params->gamut_warning && need_gamut_warn) {
        GLSL("if (any(greaterThan(color.rgb, vec3(%f + 0.005))) ||\n"
             "    any(lessThan(color.rgb, vec3(-0.005))))\n"
             "    color.rgb = vec3(%f) - color.rgb; // invert\n",
             dst.sig_peak * dst.sig_scale, src.sig_peak * src.sig_scale);
    }

    if (need_linear)
        pl_shader_inverse_ootf(sh, dst);

    if (is_linear)
        pl_shader_delinearize(sh, dst.transfer);

    GLSL("}\n");
}

void pl_shader_cone_distort(struct pl_shader *sh, struct pl_color_space csp,
                            const struct pl_cone_params *params)
{
    if (!params || !params->cones)
        return;

    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    GLSL("// pl_shader_cone_distort\n");
    GLSL("{\n");

    pl_color_space_infer(&csp);
    pl_shader_linearize(sh, csp.transfer);

    struct pl_matrix3x3 cone_mat;
    cone_mat = pl_get_cone_matrix(params, pl_raw_primaries_get(csp.primaries));
    GLSL("color.rgb = %s * color.rgb;\n", sh_var(sh, (struct pl_shader_var) {
        .var = pl_var_mat3("cone_mat"),
        .data = PL_TRANSPOSE_3X3(cone_mat.m),
    }));

    pl_shader_delinearize(sh, csp.transfer);
    GLSL("}\n");
}

struct sh_dither_obj {
    enum pl_dither_method method;
    struct pl_shader_obj *lut;
};

static void sh_dither_uninit(const struct pl_gpu *gpu, void *ptr)
{
    struct sh_dither_obj *obj = ptr;
    pl_shader_obj_destroy(&obj->lut);
    *obj = (struct sh_dither_obj) {0};
}

static void fill_dither_matrix(void *data, const struct sh_lut_params *params)
{
    pl_assert(params->width > 0 && params->height > 0 && params->comps == 1);

    const struct sh_dither_obj *obj = params->priv;
    switch (obj->method) {
    case PL_DITHER_ORDERED_LUT:
        pl_assert(params->width == params->height);
        pl_generate_bayer_matrix(data, params->width);
        break;

    case PL_DITHER_BLUE_NOISE:
        pl_assert(params->width == params->height);
        pl_generate_blue_noise(data, params->width);
        break;

    default: abort();
    }
}

static bool dither_method_is_lut(enum pl_dither_method method)
{
    switch (method) {
    case PL_DITHER_BLUE_NOISE:
    case PL_DITHER_ORDERED_LUT:
        return true;
    case PL_DITHER_ORDERED_FIXED:
    case PL_DITHER_WHITE_NOISE:
        return false;
    default: abort();
    }
}

void pl_shader_dither(struct pl_shader *sh, int new_depth,
                      struct pl_shader_obj **dither_state,
                      const struct pl_dither_params *params)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    if (new_depth <= 0 || new_depth > 256) {
        PL_WARN(sh, "Invalid dither depth: %d.. ignoring", new_depth);
        return;
    }

    GLSL("// pl_shader_dither \n"
        "{                    \n"
        "float bias;          \n");

    params = PL_DEF(params, &pl_dither_default_params);
    if (params->lut_size < 0 || params->lut_size > 8) {
        SH_FAIL(sh, "Invalid `lut_size` specified: %d", params->lut_size);
        return;
    }

    enum pl_dither_method method = params->method;
    bool can_fixed = sh_glsl(sh).version >= 130;
    ident_t lut = NULL;
    int lut_size = 0;

    if (method == PL_DITHER_ORDERED_FIXED && !can_fixed) {
        PL_WARN(sh, "PL_DITHER_ORDERED_FIXED requires glsl version >= 130.."
                " falling back.");
        goto fallback;
    }

    if (dither_method_is_lut(method)) {
        if (!dither_state) {
            PL_WARN(sh, "LUT-based dither method specified but no dither state "
                    "object given, falling back to non-LUT based methods.");
            goto fallback;
        }

        struct sh_dither_obj *obj;
        obj = SH_OBJ(sh, dither_state, PL_SHADER_OBJ_DITHER,
                     struct sh_dither_obj, sh_dither_uninit);
        if (!obj)
            goto fallback;

        bool changed = obj->method != method;
        obj->method = method;

        lut_size = 1 << PL_DEF(params->lut_size, 6);
        lut = sh_lut(sh, &(struct sh_lut_params) {
            .object = &obj->lut,
            .type = PL_VAR_FLOAT,
            .width = lut_size,
            .height = lut_size,
            .comps = 1,
            .update = changed,
            .fill = fill_dither_matrix,
            .priv = obj,
        });
        if (!lut)
            goto fallback;
    }

    goto done;

fallback:
    method = can_fixed ? PL_DITHER_ORDERED_FIXED : PL_DITHER_WHITE_NOISE;
    // fall through

done: ;

    int size = 0;
    if (lut) {
        size = lut_size;
    } else if (method == PL_DITHER_ORDERED_FIXED) {
        size = 16; // hard-coded size
    }

    if (size) {
        // Transform the screen position to the cyclic range [0,1)
        GLSL("vec2 pos = fract(gl_FragCoord.xy * 1.0/%d.0);\n", size);

        if (params->temporal) {
            int phase = SH_PARAMS(sh).index % 8;
            float r = phase * (M_PI / 2); // rotate
            float m = phase < 4 ? 1 : -1; // mirror
            float mat[2][2] = {
                {cos(r),     -sin(r)    },
                {sin(r) * m,  cos(r) * m},
            };

            ident_t rot = sh_var(sh, (struct pl_shader_var) {
                .var  = pl_var_mat2("dither_rot"),
                .data = &mat[0][0],
                .dynamic = true,
            });
            GLSL("pos = fract(%s * pos + vec2(1.0));\n", rot);
        }
    }

    switch (method) {
    case PL_DITHER_WHITE_NOISE: {
        ident_t prng = sh_prng(sh, params->temporal, NULL);
        GLSL("bias = %s;\n", prng);
        break;
    }

    case PL_DITHER_ORDERED_FIXED:
        // Bitwise ordered dither using only 32-bit uints
        GLSL("uvec2 xy = uvec2(pos * 16.0) %% 16u;     \n"
             // Bitwise merge (morton number)
             "xy.x = xy.x ^ xy.y;                      \n"
             "xy = (xy | xy << 2) & uvec2(0x33333333); \n"
             "xy = (xy | xy << 1) & uvec2(0x55555555); \n"
             // Bitwise inversion
             "uint b = xy.x + (xy.y << 1);             \n"
             "b = (b * 0x0802u & 0x22110u) |           \n"
             "    (b * 0x8020u & 0x88440u);            \n"
             "b = 0x10101u * b;                        \n"
             "b = (b >> 16) & 0xFFu;                   \n"
             // Generate bias value
             "bias = float(b) * 1.0/256.0;             \n");
        break;

    default: // LUT-based methods
        pl_assert(lut);
        GLSL("bias = %s(ivec2(pos * %d.0));\n", lut, lut_size);
        break;
    }

    uint64_t scale = (1LLU << new_depth) - 1;
    GLSL("color = vec4(%f) * color + vec4(bias); \n"
         "color = floor(color) * vec4(%f);       \n"
         "}                                      \n",
         (float) scale, 1.0 / scale);
}

const struct pl_dither_params pl_dither_default_params = {
    .method     = PL_DITHER_BLUE_NOISE,
    .lut_size   = 6,
    .temporal   = false, // commonly flickers on LCDs
};

#ifdef PL_HAVE_LCMS

#include "lcms.h"

struct sh_3dlut_obj {
    struct pl_context *ctx;
    enum pl_rendering_intent intent;
    struct pl_3dlut_profile src, dst;
    struct pl_3dlut_result result;
    struct pl_shader_obj *lut_obj;
    bool updated; // to detect misuse of the API
    bool ok;
    ident_t lut;
};

static void sh_3dlut_uninit(const struct pl_gpu *gpu, void *ptr)
{
    struct sh_3dlut_obj *obj = ptr;
    pl_shader_obj_destroy(&obj->lut_obj);
    *obj = (struct sh_3dlut_obj) {0};
}

static void fill_3dlut(void *data, const struct sh_lut_params *params)
{
    struct sh_3dlut_obj *obj = params->priv;
    struct pl_context *ctx = obj->ctx;

    pl_assert(params->comps == 4);
    obj->ok = pl_lcms_compute_lut(ctx, obj->intent, obj->src, obj->dst, data,
                                  params->width, params->height, params->depth,
                                  &obj->result);
    if (!obj->ok)
        pl_err(ctx, "Failed computing 3DLUT!");
}

static bool color_profile_eq(const struct pl_3dlut_profile *a,
                             const struct pl_3dlut_profile *b)
{
    return pl_icc_profile_equal(&a->profile, &b->profile) &&
           pl_color_space_equal(&a->color, &b->color);
}

bool pl_3dlut_update(struct pl_shader *sh,
                     const struct pl_3dlut_profile *src,
                     const struct pl_3dlut_profile *dst,
                     struct pl_shader_obj **lut3d, struct pl_3dlut_result *out,
                     const struct pl_3dlut_params *params)
{
    params = PL_DEF(params, &pl_3dlut_default_params);
    size_t s_r = PL_DEF(params->size_r, 64),
           s_g = PL_DEF(params->size_g, 64),
           s_b = PL_DEF(params->size_b, 64);

    struct sh_3dlut_obj *obj;
    obj = SH_OBJ(sh, lut3d, PL_SHADER_OBJ_3DLUT,
                 struct sh_3dlut_obj, sh_3dlut_uninit);
    if (!obj)
        return false;

    bool changed = !color_profile_eq(&obj->src, src) ||
                   !color_profile_eq(&obj->dst, dst) ||
                   obj->intent != params->intent;

    // Update the object, since we need this information from `fill_3dlut`
    obj->ctx = sh->ctx;
    obj->intent = params->intent;
    obj->src = *src;
    obj->dst = *dst;
    obj->lut = sh_lut(sh, &(struct sh_lut_params) {
        .object = &obj->lut_obj,
        .method = SH_LUT_LINEAR,
        .type = PL_VAR_FLOAT,
        .width = s_r,
        .height = s_g,
        .depth = s_b,
        .comps = 4,
        .update = changed,
        .fill = fill_3dlut,
        .priv = obj,
    });
    if (!obj->lut || !obj->ok)
        return false;

    obj->updated = true;
    *out = obj->result;
    return true;
}

void pl_3dlut_apply(struct pl_shader *sh, struct pl_shader_obj **lut3d)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    struct sh_3dlut_obj *obj;
    obj = SH_OBJ(sh, lut3d, PL_SHADER_OBJ_3DLUT,
                 struct sh_3dlut_obj, sh_3dlut_uninit);
    if (!obj || !obj->lut || !obj->updated || !obj->ok) {
        SH_FAIL(sh, "pl_shader_3dlut called without prior pl_3dlut_update?");
        return;
    }

    GLSL("// pl_shader_3dlut\n");
    GLSL("color.rgba = %s(color.rgb);\n", obj->lut);

    obj->updated = false;
}

#endif // PL_HAVE_LCMS

const struct pl_3dlut_params pl_3dlut_default_params = {
    .intent = PL_INTENT_RELATIVE_COLORIMETRIC,
    .size_r = 64,
    .size_g = 64,
    .size_b = 64,
};
