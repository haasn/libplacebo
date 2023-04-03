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

#define _USE_MATH_DEFINES
#include <math.h>
#include "shaders.h"

#include <libplacebo/shaders/colorspace.h>

// Common constants for SMPTE ST.2084 (PQ)
static const float PQ_M1 = 2610./4096 * 1./4,
                   PQ_M2 = 2523./4096 * 128,
                   PQ_C1 = 3424./4096,
                   PQ_C2 = 2413./4096 * 32,
                   PQ_C3 = 2392./4096 * 32;

// Common constants for ARIB STD-B67 (HLG)
static const float HLG_A = 0.17883277,
                   HLG_B = 0.28466892,
                   HLG_C = 0.55991073,
                   HLG_REF = 1000.0 / PL_COLOR_SDR_WHITE;

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

void pl_shader_set_alpha(pl_shader sh, struct pl_color_repr *repr,
                         enum pl_alpha_mode mode)
{
    if (repr->alpha == PL_ALPHA_PREMULTIPLIED && mode == PL_ALPHA_INDEPENDENT) {
        GLSL("if (color.a > 1e-6)               \n"
             "    color.rgb /= vec3(color.a);   \n");
        repr->alpha = PL_ALPHA_INDEPENDENT;
    }

    if (repr->alpha == PL_ALPHA_INDEPENDENT && mode == PL_ALPHA_PREMULTIPLIED) {
        GLSL("color.rgb *= vec3(color.a); \n");
        repr->alpha = PL_ALPHA_PREMULTIPLIED;
    }
}

#ifdef PL_HAVE_DOVI
static inline void reshape_mmr(pl_shader sh, ident_t mmr, bool single,
                               int min_order, int max_order)
{
    if (single) {
        GLSL("const uint mmr_idx = 0u; \n");
    } else {
        GLSL("uint mmr_idx = uint(coeffs.y); \n");
    }

    assert(min_order <= max_order);
    if (min_order < max_order)
        GLSL("uint order = uint(coeffs.w); \n");

    GLSL("vec4 sigX;                            \n"
         "s = coeffs.x;                         \n"
         "sigX.xyz = sig.xxy * sig.yzz;         \n"
         "sigX.w = sigX.x * sig.z;              \n"
         "s += dot("$"[mmr_idx + 0].xyz, sig);  \n"
         "s += dot("$"[mmr_idx + 1], sigX);     \n",
         mmr, mmr);

    if (max_order >= 2) {
        if (min_order < 2)
            GLSL("if (order >= 2) { \n");

        GLSL("vec3 sig2 = sig * sig;                \n"
             "vec4 sigX2 = sigX * sigX;             \n"
             "s += dot("$"[mmr_idx + 2].xyz, sig2); \n"
             "s += dot("$"[mmr_idx + 3], sigX2);    \n",
             mmr, mmr);

        if (max_order == 3) {
            if (min_order < 3)
                GLSL("if (order >= 3 { \n");

            GLSL("s += dot("$"[mmr_idx + 4].xyz, sig2 * sig);   \n"
                 "s += dot("$"[mmr_idx + 5], sigX2 * sigX);     \n",
                 mmr, mmr);

            if (min_order < 3)
                GLSL("} \n");
        }

        if (min_order < 2)
            GLSL("} \n");
    }
}

static inline void reshape_poly(pl_shader sh)
{
    GLSL("s = (coeffs.z * s + coeffs.y) * s + coeffs.x; \n");
}
#endif

void pl_shader_dovi_reshape(pl_shader sh, const struct pl_dovi_metadata *data)
{
#ifdef PL_HAVE_DOVI
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0) || !data)
        return;

    sh_describe(sh, "reshaping");
    GLSL("// pl_shader_reshape                  \n"
         "{                                     \n"
         "vec3 sig;                             \n"
         "vec4 coeffs;                          \n"
         "float s;                              \n"
         "sig = clamp(color.rgb, 0.0, 1.0);     \n");

    float coeffs_data[8][4];
    float mmr_packed_data[8*6][4];

    for (int c = 0; c < 3; c++) {
        const struct pl_reshape_data *comp = &data->comp[c];
        if (!comp->num_pivots)
            continue;

        pl_assert(comp->num_pivots >= 2 && comp->num_pivots <= 9);
        GLSL("s = sig[%d]; \n", c);

        // Prepare coefficients for GPU
        bool has_poly = false, has_mmr = false, mmr_single = true;
        int mmr_idx = 0, min_order = 3, max_order = 1;
        memset(coeffs_data, 0, sizeof(coeffs_data));
        for (int i = 0; i < comp->num_pivots - 1; i++) {
            switch (comp->method[i]) {
            case 0: // polynomial
                has_poly = true;
                coeffs_data[i][3] = 0.0; // order=0 signals polynomial
                for (int k = 0; k < 3; k++)
                    coeffs_data[i][k] = comp->poly_coeffs[i][k];
                break;

            case 1:
                min_order = PL_MIN(min_order, comp->mmr_order[i]);
                max_order = PL_MAX(max_order, comp->mmr_order[i]);
                mmr_single = !has_mmr;
                has_mmr = true;
                coeffs_data[i][3] = (float) comp->mmr_order[i];
                coeffs_data[i][0] = comp->mmr_constant[i];
                coeffs_data[i][1] = (float) mmr_idx;
                for (int j = 0; j < comp->mmr_order[i]; j++) {
                    // store weights per order as two packed vec4s
                    float *mmr = &mmr_packed_data[mmr_idx][0];
                    mmr[0] = comp->mmr_coeffs[i][j][0];
                    mmr[1] = comp->mmr_coeffs[i][j][1];
                    mmr[2] = comp->mmr_coeffs[i][j][2];
                    mmr[3] = 0.0; // unused
                    mmr[4] = comp->mmr_coeffs[i][j][3];
                    mmr[5] = comp->mmr_coeffs[i][j][4];
                    mmr[6] = comp->mmr_coeffs[i][j][5];
                    mmr[7] = comp->mmr_coeffs[i][j][6];
                    mmr_idx += 2;
                }
                break;

            default:
                pl_unreachable();
            }
        }

        if (comp->num_pivots > 2) {

            // Skip the (irrelevant) lower and upper bounds
            float pivots_data[7];
            memcpy(pivots_data, comp->pivots + 1,
                   (comp->num_pivots - 2) * sizeof(pivots_data[0]));

            // Fill the remainder with a quasi-infinite sentinel pivot
            for (int i = comp->num_pivots - 2; i < PL_ARRAY_SIZE(pivots_data); i++)
                pivots_data[i] = 1e9f;

            ident_t pivots = sh_var(sh, (struct pl_shader_var) {
                .data = pivots_data,
                .var = {
                    .name = "pivots",
                    .type = PL_VAR_FLOAT,
                    .dim_v = 1,
                    .dim_m = 1,
                    .dim_a = PL_ARRAY_SIZE(pivots_data),
                },
            });

            ident_t coeffs = sh_var(sh, (struct pl_shader_var) {
                .data = coeffs_data,
                .var = {
                    .name = "coeffs",
                    .type = PL_VAR_FLOAT,
                    .dim_v = 4,
                    .dim_m = 1,
                    .dim_a = PL_ARRAY_SIZE(coeffs_data),
                },
            });

            // Efficiently branch into the correct set of coefficients
            GLSL("#define test(i) bvec4(s >= "$"[i])                \n"
                 "#define coef(i) "$"[i]                            \n"
                 "coeffs = mix(mix(mix(coef(0), coef(1), test(0)),  \n"
                 "                 mix(coef(2), coef(3), test(2)),  \n"
                 "                 test(1)),                        \n"
                 "             mix(mix(coef(4), coef(5), test(4)),  \n"
                 "                 mix(coef(6), coef(7), test(6)),  \n"
                 "                 test(5)),                        \n"
                 "             test(3));                            \n"
                 "#undef test                                       \n"
                 "#undef coef                                       \n",
                 pivots, coeffs);

        } else {

            // No need for a single pivot, just set the coeffs directly
            GLSL("coeffs = "$"; \n", sh_var(sh, (struct pl_shader_var) {
                .var = pl_var_vec4("coeffs"),
                .data = coeffs_data,
            }));

        }

        ident_t mmr = NULL_IDENT;
        if (has_mmr) {
            mmr = sh_var(sh, (struct pl_shader_var) {
                .data = mmr_packed_data,
                .var = {
                    .name = "mmr",
                    .type = PL_VAR_FLOAT,
                    .dim_v = 4,
                    .dim_m = 1,
                    .dim_a = mmr_idx,
                },
            });
        }

        if (has_mmr && has_poly) {
            GLSL("if (coeffs.w == 0.0) { \n");
            reshape_poly(sh);
            GLSL("} else { \n");
            reshape_mmr(sh, mmr, mmr_single, min_order, max_order);
            GLSL("} \n");
        } else if (has_poly) {
            reshape_poly(sh);
        } else {
            assert(has_mmr);
            GLSL("{ \n");
            reshape_mmr(sh, mmr, mmr_single, min_order, max_order);
            GLSL("} \n");
        }

        ident_t lo = sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_float("lo"),
            .data = &comp->pivots[0],
        });
        ident_t hi = sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_float("hi"),
            .data = &comp->pivots[comp->num_pivots - 1],
        });
        GLSL("color[%d] = clamp(s, "$", "$"); \n", c, lo, hi);
    }

    GLSL("} \n");
#else
    SH_FAIL(sh, "libplacebo was compiled without support for dolbyvision reshaping");
#endif
}

void pl_shader_decode_color(pl_shader sh, struct pl_color_repr *repr,
                            const struct pl_color_adjustment *params)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    sh_describe(sh, "color decoding");
    GLSL("// pl_shader_decode_color \n"
         "{ \n");

    // Do this first because the following operations are potentially nonlinear
    pl_shader_set_alpha(sh, repr, PL_ALPHA_INDEPENDENT);

    if (repr->sys == PL_COLOR_SYSTEM_XYZ ||
        repr->sys == PL_COLOR_SYSTEM_DOLBYVISION)
    {
        ident_t scale = SH_FLOAT(pl_color_repr_normalize(repr));
        GLSL("color.rgb *= vec3("$"); \n", scale);
    }

    if (repr->sys == PL_COLOR_SYSTEM_XYZ) {
        pl_shader_linearize(sh, &(struct pl_color_space) {
            .transfer = PL_COLOR_TRC_ST428,
        });
    }

    if (repr->sys == PL_COLOR_SYSTEM_DOLBYVISION)
        pl_shader_dovi_reshape(sh, repr->dovi);

    enum pl_color_system orig_sys = repr->sys;
    pl_transform3x3 tr = pl_color_repr_decode(repr, params);

    if (memcmp(&tr, &pl_transform3x3_identity, sizeof(tr))) {
        ident_t cmat = sh_var(sh, (struct pl_shader_var) {
            .var  = pl_var_mat3("cmat"),
            .data = PL_TRANSPOSE_3X3(tr.mat.m),
        });

        ident_t cmat_c = sh_var(sh, (struct pl_shader_var) {
            .var  = pl_var_vec3("cmat_c"),
            .data = tr.c,
        });

        GLSL("color.rgb = "$" * color.rgb + "$"; \n", cmat, cmat_c);
    }

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
        GLSL("// constant luminance conversion                              \n"
             "color.br = color.br * mix(vec2(1.5816, 0.9936),               \n"
             "                          vec2(1.9404, 1.7184),               \n"
             "                          lessThanEqual(color.br, vec2(0.0))) \n"
             "           + color.gg;                                        \n");
        // Expand channels to camera-linear light. This shader currently just
        // assumes everything uses the BT.2020 12-bit gamma function, since the
        // difference between 10 and 12-bit is negligible for anything other
        // than 12-bit content.
        GLSL("vec3 lin = mix(color.rgb * vec3(1.0/4.5),                        \n"
             "                pow((color.rgb + vec3(0.0993))*vec3(1.0/1.0993), \n"
             "                    vec3(1.0/0.45)),                             \n"
             "                lessThanEqual(vec3(0.08145), color.rgb));        \n");
        // Calculate the green channel from the expanded RYcB, and recompress to G'
        // The BT.2020 specification says Yc = 0.2627*R + 0.6780*G + 0.0593*B
        GLSL("color.g = (lin.g - 0.2627*lin.r - 0.0593*lin.b)*1.0/0.6780;   \n"
             "color.g = mix(color.g * 4.5,                                  \n"
             "              1.0993 * pow(color.g, 0.45) - 0.0993,           \n"
             "              0.0181 <= color.g);                             \n");
        break;

    case PL_COLOR_SYSTEM_BT_2100_PQ:;
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

        GLSL(// PQ EOTF
             "color.rgb = pow(max(color.rgb, 0.0), vec3(1.0/%f));           \n"
             "color.rgb = max(color.rgb - vec3(%f), 0.0)                    \n"
             "             / (vec3(%f) - vec3(%f) * color.rgb);             \n"
             "color.rgb = pow(color.rgb, vec3(1.0/%f));                     \n"
             // LMS matrix
             "color.rgb = mat3( 3.43661, -0.79133, -0.0259499,              \n"
             "                 -2.50645,  1.98360, -0.0989137,              \n"
             "                  0.06984, -0.192271, 1.12486) * color.rgb;   \n"
             // PQ OETF
             "color.rgb = pow(max(color.rgb, 0.0), vec3(%f));               \n"
             "color.rgb = (vec3(%f) + vec3(%f) * color.rgb)                 \n"
             "             / (vec3(1.0) + vec3(%f) * color.rgb);            \n"
             "color.rgb = pow(color.rgb, vec3(%f));                         \n",
             PQ_M2, PQ_C1, PQ_C2, PQ_C3, PQ_M1,
             PQ_M1, PQ_C1, PQ_C2, PQ_C3, PQ_M2);
        break;

    case PL_COLOR_SYSTEM_BT_2100_HLG:
        GLSL(// HLG OETF^-1
             "color.rgb = mix(vec3(4.0) * color.rgb * color.rgb,                \n"
             "                exp((color.rgb - vec3(%f)) * vec3(1.0/%f))        \n"
             "                    + vec3(%f),                                   \n"
             "                lessThan(vec3(0.5), color.rgb));                  \n"
             // LMS matrix
             "color.rgb = mat3( 3.43661, -0.79133, -0.0259499,                  \n"
             "                 -2.50645,  1.98360, -0.0989137,                  \n"
             "                  0.06984, -0.192271, 1.12486) * color.rgb;       \n"
            // HLG OETF
             "color.rgb = mix(vec3(0.5) * sqrt(color.rgb),                      \n"
             "                vec3(%f) * log(color.rgb - vec3(%f)) + vec3(%f),  \n"
             "                lessThan(vec3(1.0), color.rgb));                  \n",
             HLG_C, HLG_A, HLG_B,
             HLG_A, HLG_B, HLG_C);
        break;

    case PL_COLOR_SYSTEM_DOLBYVISION:;
#ifdef PL_HAVE_DOVI
        // Dolby Vision always outputs BT.2020-referred HPE LMS, so hard-code
        // the inverse LMS->RGB matrix corresponding to this color space.
        pl_matrix3x3 dovi_lms2rgb = {{
            { 3.06441879, -2.16597676,  0.10155818},
            {-0.65612108,  1.78554118, -0.12943749},
            { 0.01736321, -0.04725154,  1.03004253},
        }};

        pl_matrix3x3_mul(&dovi_lms2rgb, &repr->dovi->linear);
        ident_t mat = sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_mat3("lms2rgb"),
            .data = PL_TRANSPOSE_3X3(dovi_lms2rgb.m),
        });

        // PQ EOTF
        GLSL("color.rgb = pow(max(color.rgb, 0.0), vec3(1.0/%f));   \n"
             "color.rgb = max(color.rgb - vec3(%f), 0.0)            \n"
             "             / (vec3(%f) - vec3(%f) * color.rgb);     \n"
             "color.rgb = pow(color.rgb, vec3(1.0/%f));             \n",
             PQ_M2, PQ_C1, PQ_C2, PQ_C3, PQ_M1);
        // LMS matrix
        GLSL("color.rgb = "$" * color.rgb; \n", mat);
        // PQ OETF
        GLSL("color.rgb = pow(max(color.rgb, 0.0), vec3(%f));       \n"
             "color.rgb = (vec3(%f) + vec3(%f) * color.rgb)         \n"
             "             / (vec3(1.0) + vec3(%f) * color.rgb);    \n"
             "color.rgb = pow(color.rgb, vec3(%f));                 \n",
             PQ_M1, PQ_C1, PQ_C2, PQ_C3, PQ_M2);
        break;
#else
        SH_FAIL(sh, "libplacebo was compiled without support for dolbyvision reshaping");
        return;
#endif

    case PL_COLOR_SYSTEM_UNKNOWN:
    case PL_COLOR_SYSTEM_RGB:
    case PL_COLOR_SYSTEM_XYZ:
    case PL_COLOR_SYSTEM_BT_601:
    case PL_COLOR_SYSTEM_BT_709:
    case PL_COLOR_SYSTEM_SMPTE_240M:
    case PL_COLOR_SYSTEM_BT_2020_NC:
    case PL_COLOR_SYSTEM_YCGCO:
        break; // no special post-processing needed

    case PL_COLOR_SYSTEM_COUNT:
        pl_unreachable();
    }

    // Gamma adjustment. Doing this here (in non-linear light) is technically
    // somewhat wrong, but this is just an aesthetic parameter and not really
    // meant for colorimetric precision, so we don't care too much.
    if (params && params->gamma == 0) {
        // Avoid division by zero
        GLSL("color.rgb = vec3(0.0); \n");
    } else if (params && params->gamma != 1) {
        ident_t gamma = sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_float("gamma"),
            .data = &(float){ 1 / params->gamma },
        });
        GLSL("color.rgb = pow(max(color.rgb, vec3(0.0)), vec3("$")); \n", gamma);
    }

    GLSL("}\n");
}

void pl_shader_encode_color(pl_shader sh, const struct pl_color_repr *repr)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    sh_describe(sh, "color encoding");
    GLSL("// pl_shader_encode_color \n"
         "{ \n");

    switch (repr->sys) {
    case PL_COLOR_SYSTEM_BT_2020_C:
        // Expand R'G'B' to RGB
        GLSL("vec3 lin = mix(color.rgb * vec3(1.0/4.5),                        \n"
             "                pow((color.rgb + vec3(0.0993))*vec3(1.0/1.0993), \n"
             "                    vec3(1.0/0.45)),                             \n"
             "                lessThanEqual(vec3(0.08145), color.rgb));        \n");

        // Compute Yc from RGB and compress to R'Y'cB'
        GLSL("color.g = dot(vec3(0.2627, 0.6780, 0.0593), lin);     \n"
             "color.g = mix(color.g * 4.5,                          \n"
             "              1.0993 * pow(color.g, 0.45) - 0.0993,   \n"
             "              0.0181 <= color.g);                     \n");

        // Compute C'bc and C'rc into color.br
        GLSL("color.br = color.br - color.gg;                       \n"
             "color.br *= mix(vec2(1.0/1.5816, 1.0/0.9936),         \n"
             "                vec2(1.0/1.9404, 1.0/1.7184),         \n"
             "                lessThanEqual(color.br, vec2(0.0)));  \n");
        break;

    case PL_COLOR_SYSTEM_BT_2100_PQ:;
        GLSL("color.rgb = pow(max(color.rgb, 0.0), vec3(1.0/%f));           \n"
             "color.rgb = max(color.rgb - vec3(%f), 0.0)                    \n"
             "             / (vec3(%f) - vec3(%f) * color.rgb);             \n"
             "color.rgb = pow(color.rgb, vec3(1.0/%f));                     \n"
             "color.rgb = mat3(0.412109, 0.166748, 0.024170,                \n"
             "                 0.523925, 0.720459, 0.075440,                \n"
             "                 0.063965, 0.112793, 0.900394) * color.rgb;   \n"
             "color.rgb = pow(color.rgb, vec3(%f));                         \n"
             "color.rgb = (vec3(%f) + vec3(%f) * color.rgb)                 \n"
             "             / (vec3(1.0) + vec3(%f) * color.rgb);            \n"
             "color.rgb = pow(color.rgb, vec3(%f));                         \n",
             PQ_M2, PQ_C1, PQ_C2, PQ_C3, PQ_M1,
             PQ_M1, PQ_C1, PQ_C2, PQ_C3, PQ_M2);
        break;

    case PL_COLOR_SYSTEM_BT_2100_HLG:
        GLSL("color.rgb = mix(vec3(4.0) * color.rgb * color.rgb,                \n"
             "                exp((color.rgb - vec3(%f)) * vec3(1.0/%f))        \n"
             "                    + vec3(%f),                                   \n"
             "                lessThan(vec3(0.5), color.rgb));                  \n"
             "color.rgb = mat3(0.412109, 0.166748, 0.024170,                    \n"
             "                 0.523925, 0.720459, 0.075440,                    \n"
             "                 0.063965, 0.112793, 0.900394) * color.rgb;       \n"
             "color.rgb = mix(vec3(0.5) * sqrt(color.rgb),                      \n"
             "                vec3(%f) * log(color.rgb - vec3(%f)) + vec3(%f),  \n"
             "                lessThan(vec3(1.0), color.rgb));                  \n",
             HLG_C, HLG_A, HLG_B,
             HLG_A, HLG_B, HLG_C);
        break;

    case PL_COLOR_SYSTEM_DOLBYVISION:
        SH_FAIL(sh, "Cannot un-apply dolbyvision yet (no inverse reshaping)!");
        return;

    case PL_COLOR_SYSTEM_UNKNOWN:
    case PL_COLOR_SYSTEM_RGB:
    case PL_COLOR_SYSTEM_XYZ:
    case PL_COLOR_SYSTEM_BT_601:
    case PL_COLOR_SYSTEM_BT_709:
    case PL_COLOR_SYSTEM_SMPTE_240M:
    case PL_COLOR_SYSTEM_BT_2020_NC:
    case PL_COLOR_SYSTEM_YCGCO:
        break; // no special pre-processing needed

    case PL_COLOR_SYSTEM_COUNT:
        pl_unreachable();
    }

    // Since this is a relatively rare operation, bypass it as much as possible
    bool skip = true;
    skip &= PL_DEF(repr->sys, PL_COLOR_SYSTEM_RGB) == PL_COLOR_SYSTEM_RGB;
    skip &= PL_DEF(repr->levels, PL_COLOR_LEVELS_FULL) == PL_COLOR_LEVELS_FULL;
    skip &= !repr->bits.sample_depth || !repr->bits.color_depth ||
             repr->bits.sample_depth == repr->bits.color_depth;
    skip &= !repr->bits.bit_shift;

    if (!skip) {
        struct pl_color_repr copy = *repr;
        ident_t xyzscale = NULL_IDENT;
        if (repr->sys == PL_COLOR_SYSTEM_XYZ)
            xyzscale = SH_FLOAT(1.0 / pl_color_repr_normalize(&copy));

        pl_transform3x3 tr = pl_color_repr_decode(&copy, NULL);
        pl_transform3x3_invert(&tr);

        ident_t cmat = sh_var(sh, (struct pl_shader_var) {
            .var  = pl_var_mat3("cmat"),
            .data = PL_TRANSPOSE_3X3(tr.mat.m),
        });

        ident_t cmat_c = sh_var(sh, (struct pl_shader_var) {
            .var  = pl_var_vec3("cmat_c"),
            .data = tr.c,
        });

        GLSL("color.rgb = "$" * color.rgb + "$"; \n", cmat, cmat_c);

        if (repr->sys == PL_COLOR_SYSTEM_XYZ) {
            pl_shader_delinearize(sh, &(struct pl_color_space) {
                .transfer = PL_COLOR_TRC_ST428,
            });
            GLSL("color.rgb *= vec3("$"); \n", xyzscale);
        }
    }

    if (repr->alpha == PL_ALPHA_PREMULTIPLIED)
        GLSL("color.rgb *= vec3(color.a); \n");

    GLSL("}\n");
}

static ident_t sh_luma_coeffs(pl_shader sh, const struct pl_raw_primaries *prim)
{
    pl_matrix3x3 rgb2xyz;
    rgb2xyz = pl_get_rgb2xyz_matrix(prim);

    // FIXME: Cannot use `const vec3` due to glslang bug #2025
    ident_t coeffs = sh_fresh(sh, "luma_coeffs");
    GLSLH("#define "$" vec3("$", "$", "$") \n", coeffs,
          SH_FLOAT(rgb2xyz.m[1][0]), // RGB->Y vector
          SH_FLOAT(rgb2xyz.m[1][1]),
          SH_FLOAT(rgb2xyz.m[1][2]));
    return coeffs;
}

void pl_shader_linearize(pl_shader sh, const struct pl_color_space *csp)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    if (csp->transfer == PL_COLOR_TRC_LINEAR)
        return;

    float csp_min, csp_max;
    pl_color_space_nominal_luma_ex(pl_nominal_luma_params(
        .color      = csp,
        .metadata   = PL_HDR_METADATA_HDR10,
        .scaling    = PL_HDR_NORM,
        .out_min    = &csp_min,
        .out_max    = &csp_max,
    ));

    // Note that this clamp may technically violate the definition of
    // ITU-R BT.2100, which allows for sub-blacks and super-whites to be
    // displayed on the display where such would be possible. That said, the
    // problem is that not all gamma curves are well-defined on the values
    // outside this range, so we ignore it and just clamp anyway for sanity.
    GLSL("// pl_shader_linearize           \n"
         "color.rgb = max(color.rgb, 0.0); \n");

    switch (csp->transfer) {
    case PL_COLOR_TRC_SRGB:
        GLSL("color.rgb = mix(color.rgb * vec3(1.0/12.92),               \n"
             "                pow((color.rgb + vec3(0.055))/vec3(1.055), \n"
             "                    vec3(2.4)),                            \n"
             "                lessThan(vec3(0.04045), color.rgb));       \n");
        goto scale_out;
    case PL_COLOR_TRC_BT_1886: {
        const float lb = powf(csp_min, 1/2.4f);
        const float lw = powf(csp_max, 1/2.4f);
        const float a = powf(lw - lb, 2.4f);
        const float b = lb / (lw - lb);
        GLSL("color.rgb = "$" * pow(color.rgb + vec3("$"), vec3(2.4)); \n",
             SH_FLOAT(a), SH_FLOAT(b));
        return;
    }
    case PL_COLOR_TRC_GAMMA18:
        GLSL("color.rgb = pow(color.rgb, vec3(1.8));\n");
        goto scale_out;
    case PL_COLOR_TRC_GAMMA20:
        GLSL("color.rgb = pow(color.rgb, vec3(2.0));\n");
        goto scale_out;
    case PL_COLOR_TRC_UNKNOWN:
    case PL_COLOR_TRC_GAMMA22:
        GLSL("color.rgb = pow(color.rgb, vec3(2.2));\n");
        goto scale_out;
    case PL_COLOR_TRC_GAMMA24:
        GLSL("color.rgb = pow(color.rgb, vec3(2.4));\n");
        goto scale_out;
    case PL_COLOR_TRC_GAMMA26:
        GLSL("color.rgb = pow(color.rgb, vec3(2.6));\n");
        goto scale_out;
    case PL_COLOR_TRC_GAMMA28:
        GLSL("color.rgb = pow(color.rgb, vec3(2.8));\n");
        goto scale_out;
    case PL_COLOR_TRC_PRO_PHOTO:
        GLSL("color.rgb = mix(color.rgb * vec3(1.0/16.0),              \n"
             "                pow(color.rgb, vec3(1.8)),               \n"
             "                lessThan(vec3(0.03125), color.rgb));     \n");
        goto scale_out;
    case PL_COLOR_TRC_ST428:
        GLSL("color.rgb = vec3(52.37/48.0) * pow(color.rgb, vec3(2.6));\n");
        goto scale_out;
    case PL_COLOR_TRC_PQ:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/%f));         \n"
             "color.rgb = max(color.rgb - vec3(%f), 0.0)        \n"
             "             / (vec3(%f) - vec3(%f) * color.rgb); \n"
             "color.rgb = pow(color.rgb, vec3(1.0/%f));         \n"
             // PQ's output range is 0-10000, but we need it to be relative to
             // to PL_COLOR_SDR_WHITE instead, so rescale
             "color.rgb *= vec3(%f);                            \n",
             PQ_M2, PQ_C1, PQ_C2, PQ_C3, PQ_M1, 10000.0 / PL_COLOR_SDR_WHITE);
        return;
    case PL_COLOR_TRC_HLG: {
        const float y = fmaxf(1.2f + 0.42f * log10f(csp_max / HLG_REF), 1);
        const float b = sqrtf(3 * powf(csp_min / csp_max, 1 / y));
        // OETF^-1
        GLSL("color.rgb = "$" * color.rgb + vec3("$");                  \n"
             "color.rgb = mix(vec3(4.0) * color.rgb * color.rgb,        \n"
             "                exp((color.rgb - vec3(%f)) * vec3(1.0/%f))\n"
             "                    + vec3(%f),                           \n"
             "                lessThan(vec3(0.5), color.rgb));          \n",
             SH_FLOAT(1 - b), SH_FLOAT(b),
             HLG_C, HLG_A, HLG_B);
        // OOTF
        GLSL("color.rgb *= 1.0 / 12.0;                                      \n"
             "color.rgb *= "$" * pow(max(dot("$", color.rgb), 0.0), "$");   \n",
             SH_FLOAT(csp_max),
             sh_luma_coeffs(sh, pl_raw_primaries_get(csp->primaries)),
             SH_FLOAT(y - 1));
        return;
    }
    case PL_COLOR_TRC_V_LOG:
        GLSL("color.rgb = mix((color.rgb - vec3(0.125)) * vec3(1.0/5.6), \n"
             "    pow(vec3(10.0), (color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "              - vec3(%f),                                  \n"
             "    lessThanEqual(vec3(0.181), color.rgb));                \n",
             VLOG_D, VLOG_C, VLOG_B);
        return;
    case PL_COLOR_TRC_S_LOG1:
        GLSL("color.rgb = pow(vec3(10.0), (color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "            - vec3(%f);                                            \n",
             SLOG_C, SLOG_A, SLOG_B);
        return;
    case PL_COLOR_TRC_S_LOG2:
        GLSL("color.rgb = mix((color.rgb - vec3(%f)) * vec3(1.0/%f),      \n"
             "    (pow(vec3(10.0), (color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "              - vec3(%f)) * vec3(1.0/%f),                   \n"
             "    lessThanEqual(vec3(%f), color.rgb));                    \n",
             SLOG_Q, SLOG_P, SLOG_C, SLOG_A, SLOG_B, SLOG_K2, SLOG_Q);
        return;
    case PL_COLOR_TRC_LINEAR:
    case PL_COLOR_TRC_COUNT:
        break;
    }

    pl_unreachable();

scale_out:
    if (csp_max != 1 || csp_min != 0) {
        GLSL("color.rgb = "$" * color.rgb + vec3("$"); \n",
             SH_FLOAT(csp_max - csp_min), SH_FLOAT(csp_min));
    }
}

void pl_shader_delinearize(pl_shader sh, const struct pl_color_space *csp)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    if (csp->transfer == PL_COLOR_TRC_LINEAR)
        return;

    float csp_min, csp_max;
    pl_color_space_nominal_luma_ex(pl_nominal_luma_params(
        .color      = csp,
        .metadata   = PL_HDR_METADATA_HDR10,
        .scaling    = PL_HDR_NORM,
        .out_min    = &csp_min,
        .out_max    = &csp_max,
    ));

    GLSL("// pl_shader_delinearize \n");
    switch (csp->transfer) {
    case PL_COLOR_TRC_UNKNOWN:
    case PL_COLOR_TRC_SRGB:
    case PL_COLOR_TRC_LINEAR:
    case PL_COLOR_TRC_GAMMA18:
    case PL_COLOR_TRC_GAMMA20:
    case PL_COLOR_TRC_GAMMA22:
    case PL_COLOR_TRC_GAMMA24:
    case PL_COLOR_TRC_GAMMA26:
    case PL_COLOR_TRC_GAMMA28:
    case PL_COLOR_TRC_PRO_PHOTO:
    case PL_COLOR_TRC_ST428: ;
        if (csp_max != 1 || csp_min != 0) {
            GLSL("color.rgb = "$" * color.rgb + vec3("$"); \n",
                 SH_FLOAT(1 / (csp_max - csp_min)),
                 SH_FLOAT(-csp_min / (csp_max - csp_min)));
        }
        break;
    case PL_COLOR_TRC_BT_1886:
    case PL_COLOR_TRC_PQ:
    case PL_COLOR_TRC_HLG:
    case PL_COLOR_TRC_V_LOG:
    case PL_COLOR_TRC_S_LOG1:
    case PL_COLOR_TRC_S_LOG2:
        break; // scene-referred or absolute scale
    case PL_COLOR_TRC_COUNT:
        pl_unreachable();
    }

    GLSL("color.rgb = max(color.rgb, 0.0); \n");

    switch (csp->transfer) {
    case PL_COLOR_TRC_SRGB:
        GLSL("color.rgb = mix(color.rgb * vec3(12.92),                        \n"
             "                vec3(1.055) * pow(color.rgb, vec3(1.0/2.4))     \n"
             "                    - vec3(0.055),                              \n"
             "                lessThanEqual(vec3(0.0031308), color.rgb));     \n");
        return;
    case PL_COLOR_TRC_BT_1886: {
        const float lb = powf(csp_min, 1/2.4f);
        const float lw = powf(csp_max, 1/2.4f);
        const float a = powf(lw - lb, 2.4f);
        const float b = lb / (lw - lb);
        GLSL("color.rgb = pow("$" * color.rgb, vec3(1.0/2.4)) - vec3("$"); \n",
             SH_FLOAT(1.0 / a), SH_FLOAT(b));
        return;
    }
    case PL_COLOR_TRC_GAMMA18:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/1.8));\n");
        return;
    case PL_COLOR_TRC_GAMMA20:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.0));\n");
        return;
    case PL_COLOR_TRC_UNKNOWN:
    case PL_COLOR_TRC_GAMMA22:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.2));\n");
        return;
    case PL_COLOR_TRC_GAMMA24:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.4));\n");
        return;
    case PL_COLOR_TRC_GAMMA26:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.6));\n");
        return;
    case PL_COLOR_TRC_GAMMA28:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.8));\n");
        return;
    case PL_COLOR_TRC_ST428:
        GLSL("color.rgb = pow(color.rgb * vec3(48.0/52.37), vec3(1.0/2.6));\n");
        return;
    case PL_COLOR_TRC_PRO_PHOTO:
        GLSL("color.rgb = mix(color.rgb * vec3(16.0),                        \n"
             "                pow(color.rgb, vec3(1.0/1.8)),                 \n"
             "                lessThanEqual(vec3(0.001953), color.rgb));     \n");
        return;
    case PL_COLOR_TRC_PQ:
        GLSL("color.rgb *= vec3(1.0/%f);                         \n"
             "color.rgb = pow(color.rgb, vec3(%f));              \n"
             "color.rgb = (vec3(%f) + vec3(%f) * color.rgb)      \n"
             "             / (vec3(1.0) + vec3(%f) * color.rgb); \n"
             "color.rgb = pow(color.rgb, vec3(%f));              \n",
             10000 / PL_COLOR_SDR_WHITE, PQ_M1, PQ_C1, PQ_C2, PQ_C3, PQ_M2);
        return;
    case PL_COLOR_TRC_HLG: {
        const float y = fmaxf(1.2f + 0.42f * log10f(csp_max / HLG_REF), 1);
        const float b = sqrtf(3 * powf(csp_min / csp_max, 1 / y));
        // OOTF^-1
        GLSL("color.rgb *= 1.0 / "$";                                       \n"
             "color.rgb *= 12.0 * max(1e-6, pow(dot("$", color.rgb), "$")); \n",
             SH_FLOAT(csp_max),
             sh_luma_coeffs(sh, pl_raw_primaries_get(csp->primaries)),
             SH_FLOAT((1 - y) / y));
        // OETF
        GLSL("color.rgb = mix(vec3(0.5) * sqrt(color.rgb),                      \n"
             "                vec3(%f) * log(color.rgb - vec3(%f)) + vec3(%f),  \n"
             "                lessThan(vec3(1.0), color.rgb));                  \n"
             "color.rgb = "$" * color.rgb + vec3("$");                          \n",
             HLG_A, HLG_B, HLG_C,
             SH_FLOAT(1 / (1 - b)), SH_FLOAT(-b / (1 - b)));
        return;
    }
    case PL_COLOR_TRC_V_LOG:
        GLSL("color.rgb = mix(vec3(5.6) * color.rgb + vec3(0.125),       \n"
             "                vec3(%f) * log(color.rgb + vec3(%f))       \n"
             "                    + vec3(%f),                            \n"
             "                lessThanEqual(vec3(0.01), color.rgb));     \n",
             VLOG_C / M_LN10, VLOG_B, VLOG_D);
        return;
    case PL_COLOR_TRC_S_LOG1:
        GLSL("color.rgb = vec3(%f) * log(color.rgb + vec3(%f)) + vec3(%f);\n",
             SLOG_A / M_LN10, SLOG_B, SLOG_C);
        return;
    case PL_COLOR_TRC_S_LOG2:
        GLSL("color.rgb = mix(vec3(%f) * color.rgb + vec3(%f),                \n"
             "                vec3(%f) * log(vec3(%f) * color.rgb + vec3(%f)) \n"
             "                    + vec3(%f),                                 \n"
             "                lessThanEqual(vec3(0.0), color.rgb));           \n",
             SLOG_P, SLOG_Q, SLOG_A / M_LN10, SLOG_K2, SLOG_B, SLOG_C);
        return;
    case PL_COLOR_TRC_LINEAR:
    case PL_COLOR_TRC_COUNT:
        break;
    }

    pl_unreachable();
}

const struct pl_sigmoid_params pl_sigmoid_default_params = { PL_SIGMOID_DEFAULTS };

void pl_shader_sigmoidize(pl_shader sh, const struct pl_sigmoid_params *params)
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

    GLSL("// pl_shader_sigmoidize                               \n"
         "color = clamp(color, 0.0, 1.0);                       \n"
         "color = vec4("$") - vec4("$") *                       \n"
         "    log(vec4(1.0) / (color * vec4("$") + vec4("$"))   \n"
         "        - vec4(1.0));                                 \n",
         SH_FLOAT(center), SH_FLOAT(1.0 / slope),
         SH_FLOAT(scale), SH_FLOAT(offset));
}

void pl_shader_unsigmoidize(pl_shader sh, const struct pl_sigmoid_params *params)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    // See: pl_shader_sigmoidize
    params = PL_DEF(params, &pl_sigmoid_default_params);
    float center = PL_DEF(params->center, 0.75);
    float slope  = PL_DEF(params->slope, 6.5);
    float offset = 1.0 / (1 + expf(slope * center));
    float scale  = 1.0 / (1 + expf(slope * (center - 1))) - offset;

    GLSL("// pl_shader_unsigmoidize                                 \n"
         "color = clamp(color, 0.0, 1.0);                           \n"
         "color = vec4("$") /                                       \n"
         "    (vec4(1.0) + exp(vec4("$") * (vec4("$") - color)))    \n"
         "    - vec4("$");                                          \n",
         SH_FLOAT(1.0 / scale),
         SH_FLOAT(slope), SH_FLOAT(center),
         SH_FLOAT(offset / scale));
}

const struct pl_peak_detect_params pl_peak_detect_default_params = { PL_PEAK_DETECT_DEFAULTS };

static bool peak_detect_params_eq(const struct pl_peak_detect_params *a,
                                  const struct pl_peak_detect_params *b)
{
    return a->smoothing_period     == b->smoothing_period     &&
           a->scene_threshold_low  == b->scene_threshold_low  &&
           a->scene_threshold_high == b->scene_threshold_high &&
           a->minimum_peak         == b->minimum_peak         &&
           a->percentile           == b->percentile;
    // don't compare `allow_delayed` because it doesn't change measurement
}

enum {
    // How many bits to use for storing PQ data. Be careful when setting this
    // too high, as it may overflow `unsigned int` on large video sources.
    //
    // The value chosen is enough to guarantee no overflow for an 8K x 4K frame
    // consisting entirely of 100% 10k nits PQ values, with 16x16 workgroups.
    PQ_BITS     = 14,
    PQ_MAX      = (1 << PQ_BITS) - 1,

    // How many bits to use for the histogram. We bias the histogram down
    // by half the PQ range (~90 nits), effectively clumping the SDR part
    // of the image into a single histogram bin.
    HIST_BITS   = 7,
    HIST_BIAS   = 1 << (HIST_BITS - 1),
    HIST_BINS   = (1 << HIST_BITS) - HIST_BIAS,

    // Convert from histogram bin to (starting) PQ value
#define HIST_PQ(bin) (((bin) + HIST_BIAS) << (PQ_BITS - HIST_BITS))
};


pl_static_assert(PQ_BITS >= HIST_BITS);

struct peak_buf_data {
    unsigned frame_wg_count;  // number of work groups processed
    unsigned frame_sum_pq;    // sum of PQ Y values over all WGs (PQ_BITS)
    unsigned frame_max_pq;    // maximum PQ Y value among these WGs (PQ_BITS)
    unsigned frame_hist[HIST_BINS]; // always allocated, conditionally unsed
};

static const struct pl_buffer_var peak_buf_vars[] = {
#define VAR(field) {                                                            \
    .var = {                                                                    \
        .name = #field,                                                         \
        .type = PL_VAR_UINT,                                                    \
        .dim_v = 1,                                                             \
        .dim_m = 1,                                                             \
        .dim_a = sizeof(((struct peak_buf_data *) NULL)->field) /               \
                 sizeof(unsigned),                                              \
    },                                                                          \
    .layout = {                                                                 \
        .offset = offsetof(struct peak_buf_data, field),                        \
        .size   = sizeof(((struct peak_buf_data *) NULL)->field),               \
        .stride = sizeof(unsigned),                                             \
    },                                                                          \
}
    VAR(frame_wg_count),
    VAR(frame_sum_pq),
    VAR(frame_max_pq),
    VAR(frame_hist),
#undef VAR
};

struct sh_tone_map_obj {
    struct pl_tone_map_params params;
    pl_shader_obj lut;

    // Peak detection state
    struct {
        struct pl_peak_detect_params params;    // currently active parameters
        pl_buf buf;                             // pending peak detection buffer
        float avg_pq;                           // current (smoothed) values
        float max_pq;
    } peak;
};

static void sh_tone_map_uninit(pl_gpu gpu, void *ptr)
{
    struct sh_tone_map_obj *obj = ptr;
    pl_shader_obj_destroy(&obj->lut);
    pl_buf_destroy(gpu, &obj->peak.buf);
    memset(obj, 0, sizeof(*obj));
}

static inline float iir_coeff(float rate)
{
    float a = 1.0 - cos(1.0 / rate);
    return sqrt(a*a + 2*a) - a;
}

static float measure_peak(const struct peak_buf_data *data, float percentile)
{
    const float frame_max = (float) data->frame_max_pq / PQ_MAX;
    if (percentile <= 0 || percentile >= 100)
        return frame_max;

    unsigned total_pixels = 0;
    for (int i = 0; i < HIST_BINS; i++)
        total_pixels += data->frame_hist[i];
    if (!total_pixels) // no histogram data available?
        return frame_max;

    const unsigned target_pixel = ceilf(percentile / 100.0f * total_pixels);
    if (target_pixel >= total_pixels)
        return frame_max;

    unsigned sum = 0;
    for (int i = 0; i < HIST_BINS; i++) {
        const unsigned next = sum + data->frame_hist[i];
        if (next < target_pixel) {
            sum = next;
            continue;
        }

        // Upper and lower frequency boundaries of the matching histogram bin
        const unsigned count_low  = sum;      // last pixel of previous bin
        const unsigned count_high = next + 1; // first pixel of next bin
        pl_assert(count_low < target_pixel && target_pixel < count_high);

        // PQ luminance associated with count_low/high respectively
        const float pq_low  = (float) HIST_PQ(i)     / PQ_MAX;
        float pq_high       = (float) HIST_PQ(i + 1) / PQ_MAX;
        if (count_high > total_pixels) // special case for last histogram bin
            pq_high = frame_max;

        // Position of `target_pixel` inside this bin, assumes pixels are
        // equidistributed inside a histogram bin
        const float ratio = (float) (target_pixel - count_low) /
                                    (count_high - count_low);
        return PL_MIX(pq_low, pq_high, ratio);
    }

    pl_unreachable();
}

// if `force` is true, ensures the buffer is read, even if `allow_delayed`
static void update_peak_buf(pl_gpu gpu, struct sh_tone_map_obj *obj, bool force)
{
    const struct pl_peak_detect_params *params = &obj->peak.params;
    if (!obj->peak.buf)
        return;

    if (!force && params->allow_delayed && pl_buf_poll(gpu, obj->peak.buf, 0))
        return; // buffer not ready yet

    struct peak_buf_data data = {0};
    bool ok = pl_buf_read(gpu, obj->peak.buf, 0, &data, sizeof(data));
    if (ok && data.frame_wg_count > 0) {
        // Peak detection completed successfully
        pl_buf_destroy(gpu, &obj->peak.buf);
    } else {
        // No data read? Possibly this peak obj has not been executed yet
        if (params->allow_delayed) {
            PL_TRACE(gpu, "Peak detection buffer seems empty, ignoring..");
        } else if (ok) {
            PL_WARN(gpu, "Peak detection usage error: attempted detecting peak "
                    "and using detected peak in the same shader program, "
                    "but `params->allow_delayed` is false! Ignoring, but "
                    "expect incorrect output.");
        } else {
            PL_ERR(gpu, "Failed reading peak detection buffer!");
        }
        if (force)
            pl_buf_destroy(gpu, &obj->peak.buf);
        return;
    }

    float avg_pq = (float) data.frame_sum_pq / (data.frame_wg_count * PQ_MAX);
    float max_pq = measure_peak(&data, params->percentile);
    const float min_peak = PL_DEF(params->minimum_peak, 1.0f);
    max_pq = fmaxf(max_pq, pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, min_peak));

    // Set the initial value accordingly if it contains no data
    if (!obj->peak.avg_pq) {
        obj->peak.avg_pq = avg_pq;
        obj->peak.max_pq = max_pq;
    }

    // Use an IIR low-pass filter to smooth out the detected values
    const float coeff = iir_coeff(PL_DEF(params->smoothing_period, 100.0f));
    obj->peak.avg_pq += coeff * (avg_pq - obj->peak.avg_pq);
    obj->peak.max_pq += coeff * (max_pq - obj->peak.max_pq);

    // Scene change hysteresis
    if (params->scene_threshold_low > 0 && params->scene_threshold_high > 0) {
        const float log10_pq = 1e-2f; // experimentally determined approximate
        const float thresh_low = params->scene_threshold_low * log10_pq;
        const float thresh_high = params->scene_threshold_high * log10_pq;
        const float delta = fabsf(avg_pq - obj->peak.avg_pq);
        // smoothstep(thresh_low, thresh_high, delta);
        float thresh = (delta - thresh_low) / (thresh_high - thresh_low);
        thresh = PL_CLAMP(thresh, 0.0f, 1.0f);
        const float mix_coeff = thresh * thresh * (3.0f - 2.0f * thresh);
        obj->peak.avg_pq = PL_MIX(obj->peak.avg_pq, avg_pq, mix_coeff);
        obj->peak.max_pq = PL_MIX(obj->peak.max_pq, max_pq, mix_coeff);
    }
}

bool pl_shader_detect_peak(pl_shader sh, struct pl_color_space csp,
                           pl_shader_obj *state,
                           const struct pl_peak_detect_params *params)
{
    params = PL_DEF(params, &pl_peak_detect_default_params);
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return false;

    pl_gpu gpu = SH_GPU(sh);
    if (!gpu || gpu->limits.max_ssbo_size < sizeof(struct peak_buf_data)) {
        PL_ERR(sh, "HDR peak detection requires a GPU with support for at "
               "least %zu bytes of SSBO data (supported: %zu)",
               sizeof(struct peak_buf_data), gpu ? gpu->limits.max_ssbo_size : 0);
        return false;
    }

    const bool use_histogram = params->percentile > 0 && params->percentile < 100;
    size_t shmem_req = 2 * sizeof(uint32_t);
    if (use_histogram)
        shmem_req += sizeof(uint32_t[HIST_BINS]);

    if (!sh_try_compute(sh, 16, 16, true, shmem_req)) {
        PL_ERR(sh, "HDR peak detection requires compute shaders with support "
               "for at least %zu bytes of shared memory! (avail: %zu)",
               shmem_req, sh_glsl(sh).max_shmem_size);
        return false;
    }

    struct sh_tone_map_obj *obj;
    obj = SH_OBJ(sh, state, PL_SHADER_OBJ_TONE_MAP, struct sh_tone_map_obj,
                 sh_tone_map_uninit);
    if (!obj)
        return false;

    if (peak_detect_params_eq(&obj->peak.params, params)) {
        update_peak_buf(gpu, obj, true); // prevent over-writing previous frame
    } else {
        pl_reset_detected_peak(*state);
    }

    pl_assert(!obj->peak.buf);
    static const struct peak_buf_data zero = {0};
    obj->peak.buf = pl_buf_create(gpu, pl_buf_params(
        .size           = sizeof(struct peak_buf_data),
        .memory_type    = PL_BUF_MEM_DEVICE,
        .host_readable  = true,
        .storable       = true,
        .initial_data   = &zero,
    ));

    if (!obj->peak.buf) {
        SH_FAIL(sh, "Failed creating peak detection SSBO!");
        return false;
    }

    obj->peak.params = *params;

    sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name   = "PeakBuf",
            .type   = PL_DESC_BUF_STORAGE,
            .access = PL_DESC_ACCESS_READWRITE,
        },
        .memory          = PL_MEMORY_COHERENT,
        .binding.object  = obj->peak.buf,
        .buffer_vars     = (struct pl_buffer_var *) peak_buf_vars,
        .num_buffer_vars = PL_ARRAY_SIZE(peak_buf_vars),
    });

    sh_describe(sh, "peak detection");
    GLSL("// pl_shader_detect_peak                                      \n"
         "{                                                             \n"
         "const uint wg_size = gl_WorkGroupSize.x * gl_WorkGroupSize.y; \n"
         "vec4 color_orig = color;                                      \n");

    // For performance, we want to do as few atomic operations on global
    // memory as possible, so use an atomic in shmem for the work group.
    ident_t wg_sum = sh_fresh(sh, "wg_sum"),
            wg_max = sh_fresh(sh, "wg_max"),
            wg_hist = NULL_IDENT;
    GLSLH("shared uint "$", "$"; \n", wg_sum, wg_max);
    if (use_histogram) {
        wg_hist = sh_fresh(sh, "wg_hist");
        GLSLH("shared uint "$"[%u]; \n", wg_hist, HIST_BINS);
        GLSL("for (uint i = gl_LocalInvocationIndex; i < %du; i += wg_size) \n"
             "    "$"[i] = 0u;                                              \n",
             HIST_BINS, wg_hist);
    }
    GLSL($" = 0u; "$" = 0u; \n"
         "barrier();        \n",
         wg_sum, wg_max);

    // Decode color into linear light representation
    pl_color_space_infer(&csp);
    pl_shader_linearize(sh, &csp);

    // Measure luminance as N-bit PQ
    GLSL("float luma = dot("$", color.rgb);             \n"
         "luma *= %f;                                   \n"
         "luma = pow(max(luma, 0.0), %f);               \n"
         "luma = (%f + %f * luma) / (1.0 + %f * luma);  \n"
         "luma = pow(luma, %f);                         \n"
         "uint y_pq = uint(%d.0 * luma);                \n",
         sh_luma_coeffs(sh, &csp.hdr.prim),
         PL_COLOR_SDR_WHITE / 10000.0,
         PQ_M1, PQ_C1, PQ_C2, PQ_C3, PQ_M2,
         PQ_MAX);

    // Update the work group's shared atomics
    bool has_subgroups = sh_glsl(sh).subgroup_size > 0;
    if (use_histogram) {
        GLSL("int bin = (int(y_pq) >> %d) - %d; \n"
             "bin = clamp(bin, 0, %d);          \n",
             PQ_BITS - HIST_BITS, HIST_BIAS,
             HIST_BINS - 1);
        if (has_subgroups) {
            // Optimize for the very common case of identical histogram bins
            GLSL("if (subgroupAllEqual(bin)) {                  \n"
                 "    if (subgroupElect())                      \n"
                 "        atomicAdd("$"[bin], gl_SubgroupSize); \n"
                 "} else {                                      \n"
                 "    atomicAdd("$"[bin], 1u);                  \n"
                 "}                                             \n",
                 wg_hist, wg_hist);
        } else {
            GLSL("atomicAdd("$"[bin], 1u); \n", wg_hist);
        }
    }

    if (has_subgroups) {
        GLSL("uint group_sum = subgroupAdd(y_pq);   \n"
             "uint group_max = subgroupMax(y_pq);   \n"
             "if (subgroupElect()) {                \n"
             "    atomicAdd("$", group_sum);        \n"
             "    atomicMax("$", group_max);        \n"
             "}                                     \n"
             "barrier();                            \n",
             wg_sum, wg_max);
    } else {
        GLSL("atomicAdd("$", y_pq); \n"
             "atomicMax("$", y_pq); \n"
             "barrier();            \n",
             wg_sum, wg_max);
    }

    if (use_histogram) {
        GLSL("for (uint i = gl_LocalInvocationIndex; i < %du; i += wg_size) \n"
             "    atomicAdd(frame_hist[i], "$"[i]);                         \n"
             "memoryBarrierBuffer();                                        \n",
             HIST_BINS, wg_hist);
    }

    // Have one thread per work group update the global atomics
    GLSL("if (gl_LocalInvocationIndex == 0u) {          \n"
         "    atomicAdd(frame_wg_count, 1u);            \n"
         "    atomicAdd(frame_sum_pq, "$" / wg_size);   \n"
         "    atomicMax(frame_max_pq, "$");             \n"
         "    memoryBarrierBuffer();                    \n"
         "}                                             \n"
         "color = color_orig;                           \n"
         "}                                             \n",
          wg_sum, wg_max);

    return true;
}

bool pl_get_detected_hdr_metadata(const pl_shader_obj state,
                                  struct pl_hdr_metadata *out)
{
    if (!state || state->type != PL_SHADER_OBJ_TONE_MAP)
        return false;

    struct sh_tone_map_obj *obj = state->priv;
    update_peak_buf(state->gpu, obj, false);
    if (!obj->peak.avg_pq)
        return false;

    out->max_pq_y = obj->peak.max_pq;
    out->avg_pq_y = obj->peak.avg_pq;
    return true;
}

bool pl_get_detected_peak(const pl_shader_obj state,
                          float *out_peak, float *out_avg)
{
    struct pl_hdr_metadata data;
    if (!pl_get_detected_hdr_metadata(state, &data))
        return false;

    // Preserves old behavior
    *out_peak = pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NORM, data.max_pq_y);
    *out_avg  = pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NORM, data.avg_pq_y);
    return true;
}

void pl_reset_detected_peak(pl_shader_obj state)
{
    if (!state || state->type != PL_SHADER_OBJ_TONE_MAP)
        return;

    struct sh_tone_map_obj *obj = state->priv;
    pl_buf_destroy(state->gpu, &obj->peak.buf);
    memset(&obj->peak, 0, sizeof(obj->peak));
}

const struct pl_color_map_params pl_color_map_default_params = { PL_COLOR_MAP_DEFAULTS };

static void visualize_tone_map(pl_shader sh, float xmin, float xmax, float xavg,
                               float ymin, float ymax, pl_rect2df rc)
{
    if (!rc.x0 && !rc.x1)
        rc.x1 = 1.0f;
    if (!rc.y0 && !rc.y1)
        rc.y1 = 1.0f;
    if (rc.x1 == rc.x0 || rc.y1 == rc.y0)
        return;

    ident_t pos = sh_attr_vec2(sh, "tone_map_coords", &(pl_rect2df) {
        .x0 = -rc.x0         / (rc.x1 - rc.x0),
        .x1 = (1.0f - rc.x0) / (rc.x1 - rc.x0),
        .y0 = -rc.y1         / (rc.y0 - rc.y1),
        .y1 = (1.0f - rc.y1) / (rc.y0 - rc.y1),
    });

    GLSL("// Visualize tone mapping                 \n"
         "{                                         \n"
         "vec2 pos = "$";                           \n"
         "if (min(pos.x, pos.y) >= 0.0 &&           \n" // visualizer rect
         "    max(pos.x, pos.y) <= 1.0)             \n"
         "{                                         \n"
         "float xmin = "$";                         \n"
         "float xmax = "$";                         \n"
         "float xavg = "$";                         \n"
         "float ymin = "$";                         \n"
         "float ymax = "$";                         \n"
         "vec3 viz = vec3(0.5) * color.rgb;         \n"
         // PQ EOTF
         "float vv = pos.x;                         \n"
         "vv = pow(max(vv, 0.0), 1.0/%f);           \n"
         "vv = max(vv - %f, 0.0) / (%f - %f * vv);  \n"
         "vv = pow(vv, 1.0 / %f);                   \n"
         "vv *= %f;                                 \n"
         // Apply tone-mapping function
         "vv = tone_map(vv);                        \n"
         // PQ OETF
         "vv *= %f;                                 \n"
         "vv = pow(max(vv, 0.0), %f);               \n"
         "vv = (%f + %f * vv) / (1.0 + %f * vv);    \n"
         "vv = pow(vv, %f);                         \n"
         // Color based on region
         "if (pos.x < xmin || pos.x > xmax) {       \n" // outside source
         "    viz = vec3(0.0);                      \n"
         "} else if (pos.y < ymin || pos.y > ymax) {\n" // outside target
         "    if (pos.y < xmin || pos.y > xmax) {   \n" //  and also source
         "        viz = vec3(0.1, 0.1, 0.5);        \n"
         "    } else {                              \n"
         "        viz = vec3(0.4, 0.1, 0.1);        \n" //  but inside source
         "    }                                     \n"
         "} else {                                  \n" // inside domain
         "    if (abs(pos.x - pos.y) < 1e-3) {      \n" // main diagonal
         "        viz = vec3(0.2);                  \n"
         "    } else if (pos.y < vv) {              \n" // inside function
         "        viz = vec3(0.3, 1.0, 0.1);        \n"
         "        if (vv > pos.x && pos.y > pos.x)  \n" // output brighter than input
         "            viz.r = 0.7;                  \n"
         "    } else {                              \n" // outside function
         "        if (vv < pos.x && pos.y < pos.x)  \n" // output darker than input
         "            viz = vec3(0.0, 0.05, 0.1);   \n"
         "    }                                     \n"
         "    if (pos.y > xmax) {                   \n" // inverse tone-mapping region
         "        vec3 hi = vec3(0.2, 0.5, 0.8);    \n"
         "        viz = mix(viz, hi, 0.5);          \n"
         "    } else if (pos.y < xmin) {            \n" // black point region
         "        viz = mix(viz, vec3(0.0), 0.3);   \n"
         "    }                                     \n"
         "    if (xavg > 0.0 && abs(pos.x - xavg) < 1e-3)\n" // source avg brightness
         "        viz = vec3(0.5);                  \n"
         "}                                         \n"
         "color.rgb = mix(color.rgb, viz, 0.8);     \n"
         "}                                         \n"
         "}                                         \n",
         pos,
         SH_FLOAT_DYN(pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, xmin)),
         SH_FLOAT_DYN(pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, xmax)),
         SH_FLOAT_DYN(pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, xavg)),
         SH_FLOAT(pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, ymin)),
         SH_FLOAT_DYN(pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, ymax)),
         PQ_M2, PQ_C1, PQ_C2, PQ_C3, PQ_M1,
         10000.0 / PL_COLOR_SDR_WHITE,
         PL_COLOR_SDR_WHITE / 10000.0,
         PQ_M1, PQ_C1, PQ_C2, PQ_C3, PQ_M2);
}

static void fill_lut(void *data, const struct sh_lut_params *params)
{
    const struct pl_tone_map_params *lut_params = params->priv;
    assert(lut_params->lut_size == params->width);
    pl_tone_map_generate(data, lut_params);
}

static void tone_map(pl_shader sh,
                     const struct pl_color_space *src,
                     const struct pl_color_space *dst,
                     pl_shader_obj *state,
                     const struct pl_color_map_params *params)
{
    float src_min, src_max, src_avg;
    pl_color_space_nominal_luma_ex(pl_nominal_luma_params(
        .color      = src,
        .metadata   = params->metadata,
        .scaling    = PL_HDR_NORM,
        .out_min    = &src_min,
        .out_max    = &src_max,
        .out_avg    = &src_avg,
    ));

    float dst_min, dst_max;
    pl_color_space_nominal_luma_ex(pl_nominal_luma_params(
        .color      = dst,
        .metadata   = PL_HDR_METADATA_HDR10,
        .scaling    = PL_HDR_NORM,
        .out_min    = &dst_min,
        .out_max    = &dst_max,
    ));

    if (!params->inverse_tone_mapping) {
        // Never exceed the source unless requested, but still allow
        // black point adaptation
        dst_max = PL_MIN(dst_max, src_max);
    }

    // Round sufficiently similar values
    if (fabs(src_max - dst_max) < 1e-6)
        dst_max = src_max;
    if (fabs(src_min - dst_min) < 1e-6)
        dst_min = src_min;

    struct pl_tone_map_params lut_params = {
        .function = params->tone_mapping_function,
        .param = params->tone_mapping_param,
        .input_scaling = PL_HDR_SQRT,
        .output_scaling = PL_HDR_NORM,
        .lut_size = PL_DEF(params->lut_size, pl_color_map_default_params.lut_size),
        .input_min = pl_hdr_rescale(PL_HDR_NORM, PL_HDR_SQRT, src_min),
        .input_max = pl_hdr_rescale(PL_HDR_NORM, PL_HDR_SQRT, src_max),
        .input_avg = pl_hdr_rescale(PL_HDR_NORM, PL_HDR_SQRT, src_avg),
        .output_min = dst_min,
        .output_max = dst_max,
        .hdr = src->hdr,
    };

    pl_tone_map_params_infer(&lut_params);
    if (pl_tone_map_params_noop(&lut_params))
        return;

    const struct pl_tone_map_function *fun = lut_params.function;
    sh_describef(sh, "%s tone map (%.0f -> %.0f)", fun->name,
                 pl_hdr_rescale(PL_HDR_NORM, PL_HDR_NITS, src_max),
                 pl_hdr_rescale(PL_HDR_NORM, PL_HDR_NITS, dst_max));
    ident_t lut = NULL_IDENT;

    bool can_fixed = !params->force_tone_mapping_lut;
    bool is_clip = can_fixed && fun == &pl_tone_map_clip;
    bool is_linear = can_fixed && fun == &pl_tone_map_linear;

    if (state && !(is_clip || is_linear)) {
        struct sh_tone_map_obj *obj;
        obj = SH_OBJ(sh, state, PL_SHADER_OBJ_TONE_MAP, struct sh_tone_map_obj,
                     sh_tone_map_uninit);
        if (!obj)
            return;

        lut = sh_lut(sh, sh_lut_params(
            .object     = &obj->lut,
            .var_type   = PL_VAR_FLOAT,
            .lut_type   = SH_LUT_AUTO,
            .method     = SH_LUT_LINEAR,
            .width      = lut_params.lut_size,
            .comps      = 1,
            .update     = !pl_tone_map_params_equal(&lut_params, &obj->params),
            .dynamic    = src_avg > 0, // dynamic metadata was used
            .fill       = fill_lut,
            .priv       = &lut_params,
        ));
        obj->params = lut_params;
    }

    if (is_clip) {

        GLSL("#define tone_map(x) clamp((x), "$", "$") \n",
             SH_FLOAT(dst_min), SH_FLOAT_DYN(dst_max));

    } else if (is_linear) {

        const float gain = PL_DEF(lut_params.param, 1.0f);
        const float pq_src_min = pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, src_min);
        const float pq_src_max = pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, src_max);
        const float pq_dst_min = pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, dst_min);
        const float pq_dst_max = pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, dst_max);
        const float src_scale = gain / (pq_src_max - pq_src_min);
        const float dst_scale = pq_dst_max - pq_dst_min;

        ident_t linfun = sh_fresh(sh, "linear_pq");
        GLSLH("float "$"(float x) {                         \n"
             // PQ OETF
             "    x *= %f;                                  \n"
             "    x = pow(max(x, 0.0), %f);                 \n"
             "    x = (%f + %f * x) / (1.0 + %f * x);       \n"
             "    x = pow(x, %f);                           \n"
             // Stretch the input range (while clipping)
             "    x = "$" * x + "$";                        \n"
             "    x = clamp(x, 0.0, 1.0);                   \n"
             "    x = "$" * x + "$";                        \n"
             // PQ EOTF
             "    x = pow(x, 1.0 / %f);                     \n"
             "    x = max(x - %f, 0.0) / (%f - %f * x);     \n"
             "    x = pow(x, 1.0 / %f);                     \n"
             "    x *= %f;                                  \n"
             "    return x;                                 \n"
             "}                                             \n",
             linfun,
             PL_COLOR_SDR_WHITE / 10000.0,
             PQ_M1, PQ_C1, PQ_C2, PQ_C3, PQ_M2,
             SH_FLOAT_DYN(src_scale), SH_FLOAT_DYN(-src_scale * pq_src_min),
             SH_FLOAT_DYN(dst_scale), SH_FLOAT(pq_dst_min),
             PQ_M2, PQ_C1, PQ_C2, PQ_C3, PQ_M1,
             10000.0 / PL_COLOR_SDR_WHITE);

        GLSL("#define tone_map(x) ("$"(x)) \n", linfun);

    } else if (lut) {

        // Regular 1D LUT
        const float lut_range = lut_params.input_max - lut_params.input_min;
        GLSL("#define tone_map(x) ("$"("$" * sqrt(x) + "$")) \n",
             lut, SH_FLOAT_DYN(1.0f / lut_range),
             SH_FLOAT_DYN(-lut_params.input_min / lut_range));

    } else {

        // Fall back to hard-coded hable function for lack of anything better
        float A = 0.15f, B = 0.50f, C = 0.10f, D = 0.20f, E = 0.02f, F = 0.30f;
        ident_t hable = sh_fresh(sh, "hable");
        GLSLH("float "$"(float x) {                     \n"
              "    return (x * (%f*x + %f) + %f) /      \n"
              "           (x * (%f*x + %f) + %f) - %f;  \n"
              "}                                        \n",
              hable, A, C*B, D*E, A, B, D*F, E/F);

        const float scale = 1.0f / (dst_max - dst_min);
        const float peak = scale * (src_max - src_min);
        const float peak_out = ((peak * (A*peak + C*B) + D*E) /
                                (peak * (A*peak + B) + D*F)) - E/F;

        GLSL("#define tone_map(x) ("$" * "$"("$" * x + "$") + "$") \n",
             SH_FLOAT_DYN((dst_max - dst_min) / peak_out),
             hable, SH_FLOAT_DYN(scale),
             SH_FLOAT_DYN(-scale * src_min),
             SH_FLOAT(dst_min));

    }

    if (params->show_clipping) {
        GLSL("bool clip_lo = false, clip_hi = false;                \n"
             "#define cliptest(x) (clip_lo = clip_lo || x < "$",  \\\n"
             "                     clip_hi = clip_hi || x > "$")    \n",
             SH_FLOAT_DYN(src_min - 1e-6f), SH_FLOAT_DYN(src_max + 1e-6f));
    } else {
        GLSL("#define cliptest(x) \n");
    }

    enum pl_tone_map_mode mode = params->tone_mapping_mode;
    if (mode == PL_TONE_MAP_AUTO) {
        if (is_clip) {
            // No-op / clip - do this per-channel
            mode = PL_TONE_MAP_RGB;
        } else {
            // Pick this in all cases because it provides the best balance of
            // trade-offs out-of-the-box
            mode = PL_TONE_MAP_HYBRID;
        }
    }

    ident_t ct = SH_FLOAT(params->tone_mapping_crosstalk);
    GLSL("float ct_scale = 1.0 - 3.0 * "$";                 \n"
         "float ct = "$" * (color.r + color.g + color.b);   \n"
         "color.rgb = ct_scale * color.rgb + vec3(ct);      \n",
         ct, ct);

    switch (mode) {
    case PL_TONE_MAP_RGB:
        for (int c = 0; c < 3; c++) {
            GLSL("cliptest(color[%d]);             \n"
                 "color[%d] = tone_map(color[%d]); \n",
                 c, c, c);
        }
        break;

    case PL_TONE_MAP_MAX:
        GLSL("float sig_max = max(max(color.r, color.g), color.b);  \n"
             "cliptest(sig_max);                                    \n"
             "color.rgb *= tone_map(sig_max) / max(sig_max, "$");   \n",
             SH_FLOAT(dst_min));
        break;

    case PL_TONE_MAP_LUMA:
    case PL_TONE_MAP_HYBRID: {
        const struct pl_raw_primaries *prim = pl_raw_primaries_get(src->primaries);
        pl_matrix3x3 rgb2xyz = pl_get_rgb2xyz_matrix(prim);

        // Normalize X and Z by the white point
        for (int i = 0; i < 3; i++) {
            rgb2xyz.m[0][i] /= pl_cie_X(prim->white);
            rgb2xyz.m[2][i] /= pl_cie_Z(prim->white);
            rgb2xyz.m[0][i] -= rgb2xyz.m[1][i];
            rgb2xyz.m[2][i] -= rgb2xyz.m[1][i];
        }

        GLSL("vec3 xyz = "$" * color.rgb; \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_mat3("rgb2xyz"),
            .data = PL_TRANSPOSE_3X3(rgb2xyz.m),
        }));

        // Tuned to meet the desired desaturation at 1000 <-> SDR
        float ratio = logf(src_max / dst_max) / logf(1000 / PL_COLOR_SDR_WHITE);
        float coeff = src_max > dst_max ? 1 / 1.1f : 1.075f;
        float desat = (coeff - 1) * fabsf(ratio) + 1;

        GLSL("float orig = max(xyz.y, "$"); \n"
             "cliptest(xyz.y);              \n"
             "xyz.y = tone_map(xyz.y);      \n"
             "xyz.xz *= "$" * xyz.y / orig; \n",
             SH_FLOAT(dst_min),
             SH_FLOAT_DYN(desat));

        // Extra luminance correction when reducing dynamic range
        if (src_max > dst_max) {
            GLSL("xyz.y -= max("$" * xyz.x, 0.0); \n",
                 SH_FLOAT_DYN(0.1f * fabsf(ratio)));
        }

        pl_matrix3x3_invert(&rgb2xyz);
        GLSL("vec3 color_lin = "$" * xyz; \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_mat3("xyz2rgb"),
            .data = PL_TRANSPOSE_3X3(rgb2xyz.m),
        }));

        if (mode == PL_TONE_MAP_HYBRID) {
            // coeff(x) = max(a * x^-y, b * x^y)
            //   solve for coeff(dst_min) = 1, coeff(dst_max) = 1
            const float y = 2.4f;
            const float a = powf(dst_min, y);
            const float b = powf(dst_max, -y);
            GLSL("float coeff = pow(xyz.y, %f);             \n"
                 "coeff = max("$" / coeff, "$" * coeff);    \n",
                 y, SH_FLOAT(a), SH_FLOAT_DYN(b));
            for (int c = 0; c < 3; c++) {
                GLSL("if (coeff > 0.8) cliptest(color[%d]); \n"
                     "color[%d] = tone_map(color[%d]);      \n",
                     c, c, c);
            }
            GLSL("color.rgb = mix(color_lin, color.rgb, coeff); \n");
        } else {
            GLSL("color.rgb = color_lin; \n");
        }
        break;
    }

    case PL_TONE_MAP_AUTO:
    case PL_TONE_MAP_MODE_COUNT:
        pl_unreachable();
    }

    // Inverse crosstalk
    GLSL("ct = "$" * (color.r + color.g + color.b);         \n"
         "color.rgb = (color.rgb - vec3(ct)) / ct_scale;    \n",
         ct);

    if (params->show_clipping) {
        GLSL("if (clip_hi) {                                                \n"
             "    float k = dot(color.rgb, vec3(2.0 / 3.0));                \n"
             "    color.rgb = clamp(vec3(k) - color.rgb, 0.0, 1.0);         \n"
             "    float cmin = min(min(color.r, color.g), color.b);         \n"
             "    float cmax = max(max(color.r, color.g), color.b);         \n"
             "    float delta = cmax - cmin;                                \n"
             "    vec3 sat = smoothstep(cmin - 1e-6, cmax, color.rgb);      \n"
             "    const vec3 red = vec3(1.0, 0.0, 0.0);                     \n"
             "    color.rgb = mix(red, sat, smoothstep(0.0, 0.3, delta));   \n"
             "} else if (clip_lo) {                                         \n"
             "    vec3 hi = vec3(0.0, 0.3, 0.3);                            \n"
             "    color.rgb = mix(color.rgb, hi, 0.5);                      \n"
             "}                                                             \n");
    }

    if (params->visualize_lut) {
        visualize_tone_map(sh, src_min, src_max, src_avg, dst_min, dst_max,
                           params->visualize_rect);
    }

    GLSL("#undef tone_map \n"
         "#undef cliptest \n");
}

static inline bool is_identity_mat(const pl_matrix3x3 *mat)
{
    float delta = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            const float x = mat->m[i][j];
            delta += fabsf((i == j) ? (x - 1) : x);
        }
    }

    return delta < 1e-5f;
}

static void adapt_colors(pl_shader sh,
                         const struct pl_color_space *src,
                         const struct pl_color_space *dst,
                         const struct pl_color_map_params *params)
{
    bool need_reduction = pl_primaries_superset(&src->hdr.prim, &dst->hdr.prim);
    bool need_conversion = src->primaries != dst->primaries;
    if (!need_reduction && !need_conversion)
        return;

    // Main gamut adaptation matrix, respecting the desired intent
    const pl_matrix3x3 ref2ref =
        pl_get_color_mapping_matrix(&src->hdr.prim, &dst->hdr.prim, params->intent);

    float lb, lw;
    pl_color_space_nominal_luma_ex(pl_nominal_luma_params(
        .color      = dst,
        .metadata   = PL_HDR_METADATA_HDR10,
        .scaling    = PL_HDR_NORM,
        .out_min    = &lb,
        .out_max    = &lw,
    ));

    // Normalize colors to range [0-1]
    GLSL("color.rgb = "$" * color.rgb + "$"; \n",
         SH_FLOAT(1 / (lw - lb)), SH_FLOAT(-lb / (lw - lb)));

    // Convert the input colors to be represented relative to the target
    // display's mastering primaries.
    pl_matrix3x3 mat;
    mat = pl_get_color_mapping_matrix(pl_raw_primaries_get(src->primaries),
                                      &src->hdr.prim,
                                      PL_INTENT_RELATIVE_COLORIMETRIC);


    pl_matrix3x3_rmul(&ref2ref, &mat);
    if (!is_identity_mat(&mat)) {
        GLSL("color.rgb = "$" * color.rgb; \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_mat3("src2ref"),
            .data = PL_TRANSPOSE_3X3(mat.m),
        }));
    }

    enum pl_gamut_mode mode = params->gamut_mode;
    if (!need_reduction)
        mode = PL_GAMUT_CLIP;

    switch (mode) {
    case PL_GAMUT_CLIP:
        GLSL("color.rgb = clamp(color.rgb, 0.0, 1.0);           \n");
        break;

    case PL_GAMUT_WARN:
        GLSL("if (any(lessThan(color.rgb, vec3(-1e-6))) ||          \n"
             "    any(greaterThan(color.rgb, vec3(1.0 + 1e-6))))    \n"
             "{                                                     \n"
             "    float k = dot(color.rgb, vec3(2.0 / 3.0));        \n"
             "    color.rgb = clamp(vec3(k) - color.rgb, 0.0, 1.0); \n"
             "    color.rgb = sqrt(color.rgb);                      \n"
             "}                                                     \n");
        break;

    case PL_GAMUT_DARKEN: {
        float cmax = 1;
        for (int i = 0; i < 3; i++)
            cmax = PL_MAX(cmax, ref2ref.m[i][i]);
        GLSL("color.rgb *= "$"; \n", SH_FLOAT(1 / cmax));
        break;
    }

    case PL_GAMUT_DESATURATE:
        GLSL("float cmin = min(min(color.r, color.g), color.b);     \n"
             "float luma = clamp(dot("$", color.rgb), 0.0, 1.0);    \n"
             "if (cmin < 0.0 - 1e-6)                                \n"
             "    color.rgb = mix(color.rgb, vec3(luma),            \n"
             "                    -cmin / (luma - cmin));           \n"
             "float cmax = max(max(color.r, color.g), color.b);     \n"
             "if (cmax > 1.0 + 1e-6)                                \n"
             "    color.rgb = mix(color.rgb, vec3(luma),            \n"
             "                    (1.0 - cmax) / (luma - cmax));    \n",
            sh_luma_coeffs(sh, &dst->hdr.prim));
        break;

    case PL_GAMUT_MODE_COUNT:
        pl_unreachable();
    }

    // Transform the colors from the destination mastering primaries to the
    // destination nominal primaries
    mat = pl_get_color_mapping_matrix(&dst->hdr.prim,
                                      pl_raw_primaries_get(dst->primaries),
                                      PL_INTENT_RELATIVE_COLORIMETRIC);

    if (!is_identity_mat(&mat)) {
        GLSL("color.rgb = "$" * color.rgb; \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_mat3("ref2dst"),
            .data = PL_TRANSPOSE_3X3(mat.m),
        }));
    }

    // Undo normalization
    GLSL("color.rgb = "$" * color.rgb + "$"; \n",
         SH_FLOAT(lw - lb), SH_FLOAT(lb));
}

void pl_shader_color_map(pl_shader sh, const struct pl_color_map_params *params,
                         struct pl_color_space src, struct pl_color_space dst,
                         pl_shader_obj *tone_map_state,
                         bool prelinearized)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    pl_color_space_infer_map(&src, &dst);
    if (pl_color_space_equal(&src, &dst)) {
        if (prelinearized)
            pl_shader_delinearize(sh, &dst);
        return;
    }

    if (tone_map_state)
        pl_get_detected_hdr_metadata(*tone_map_state, &src.hdr);

    sh_describe(sh, "colorspace conversion");
    GLSL("// pl_shader_color_map\n");
    GLSL("{\n");
    params = PL_DEF(params, &pl_color_map_default_params);

    if (!prelinearized)
        pl_shader_linearize(sh, &src);
    tone_map(sh, &src, &dst, tone_map_state, params);
    adapt_colors(sh, &src, &dst, params);
    pl_shader_delinearize(sh, &dst);
    GLSL("}\n");
}

void pl_shader_cone_distort(pl_shader sh, struct pl_color_space csp,
                            const struct pl_cone_params *params)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;
    if (!params || !params->cones)
        return;

    sh_describe(sh, "cone distortion");
    GLSL("// pl_shader_cone_distort\n");
    GLSL("{\n");

    pl_color_space_infer(&csp);
    pl_shader_linearize(sh, &csp);

    pl_matrix3x3 cone_mat;
    cone_mat = pl_get_cone_matrix(params, pl_raw_primaries_get(csp.primaries));
    GLSL("color.rgb = "$" * color.rgb; \n", sh_var(sh, (struct pl_shader_var) {
        .var = pl_var_mat3("cone_mat"),
        .data = PL_TRANSPOSE_3X3(cone_mat.m),
    }));

    pl_shader_delinearize(sh, &csp);
    GLSL("}\n");
}
