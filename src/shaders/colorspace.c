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

static inline void reshape_mmr(pl_shader sh, ident_t mmr, bool single,
                               int min_order, int max_order)
{
    if (sh_glsl(sh).version < 130) {
        SH_FAIL(sh, "MMR reshaping requires GLSL 130+");
        return;
    }

    if (single) {
        GLSL("const uint mmr_idx = 0u; \n");
    } else {
        GLSL("uint mmr_idx = uint(coeffs.y); \n");
    }

    assert(min_order <= max_order);
    if (min_order < max_order)
        GLSL("uint order = uint(coeffs.w); \n");

    GLSL("vec4 sigX;                                            \n"
         "s = coeffs.x;                                         \n"
         "sigX.xyz = sig.xxy * sig.yzz;                         \n"
         "sigX.w = sigX.x * sig.z;                              \n"
         "s += dot(%s[mmr_idx + 0].xyz, sig);                   \n"
         "s += dot(%s[mmr_idx + 1], sigX);                      \n",
         mmr, mmr);

    if (max_order >= 2) {
        if (min_order < 2)
            GLSL("if (order >= 2) { \n");

        GLSL("vec3 sig2 = sig * sig;                            \n"
             "vec4 sigX2 = sigX * sigX;                         \n"
             "s += dot(%s[mmr_idx + 2].xyz, sig2);              \n"
             "s += dot(%s[mmr_idx + 3], sigX2);                 \n",
             mmr, mmr);

        if (max_order == 3) {
            if (min_order < 3)
                GLSL("if (order >= 3 { \n");

            GLSL("s += dot(%s[mmr_idx + 4].xyz, sig2 * sig);    \n"
                 "s += dot(%s[mmr_idx + 5], sigX2 * sigX);      \n",
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

void pl_shader_dovi_reshape(pl_shader sh, const struct pl_dovi_metadata *data)
{
    if (!data || !sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
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

        // Prepare coefficients for GPU
        bool has_poly = false, has_mmr = false, mmr_single = true;
        int mmr_idx = 0, min_order = 3, max_order = 1;
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

        // Don't mark these as dynamic, because they don't typically change
        // _that_ frequently, and we would prefer to avoid putting them in PCs
        ident_t pivots = NULL;
        if (comp->num_pivots > 2) {
            pivots = sh_var(sh, (struct pl_shader_var) {
                .data = &comp->pivots[1], // skip lower bound
                .var = {
                    .name = "pivots",
                    .type = PL_VAR_FLOAT,
                    .dim_v = 1,
                    .dim_m = 1,
                    .dim_a = comp->num_pivots - 2, // skip lower/upper bounds
                },
            });
        }

        ident_t coeffs = sh_var(sh, (struct pl_shader_var) {
            .data = coeffs_data,
            .var = {
                .name = "coeffs",
                .type = PL_VAR_FLOAT,
                .dim_v = 4,
                .dim_m = 1,
                .dim_a = comp->num_pivots - 1,
            },
        });

        ident_t mmr = NULL;
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

        // Select the right coefficient based on the pivot points
        if (comp->num_pivots > 2) {
            GLSL("coeffs = %s[%d]; \n", coeffs, comp->num_pivots - 2);
        } else {
            GLSL("coeffs = %s; \n", coeffs);
        }

        GLSL("s = sig[%d]; \n", c);
        for (int idx = comp->num_pivots - 3; idx >= 0; idx--) {
            if (comp->num_pivots > 3) {
                GLSL("coeffs = mix(coeffs, %s[%d], bvec4(s < %s[%d])); \n",
                     coeffs, idx, pivots, idx);
            } else {
                GLSL("coeffs = mix(coeffs, %s[%d], bvec4(s < %s)); \n",
                     coeffs, idx, pivots);
            }
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

        // Hard-code these as constants because they're exceptionally unlikely
        // to change from frame to frame (if they do, shoot the sample author)
        ident_t lo = SH_FLOAT(comp->pivots[0]);
        ident_t hi = SH_FLOAT(comp->pivots[comp->num_pivots - 1]);
        GLSL("color[%d] = clamp(s, %s, %s); \n", c, lo, hi);
    }

    GLSL("}\n");
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

    // XYZ needs special handling due to the input gamma logic
    if (repr->sys == PL_COLOR_SYSTEM_XYZ) {
        ident_t scale = SH_FLOAT(pl_color_repr_normalize(repr));
        GLSL("color.rgb = max(color.rgb, vec3(0.0));            \n"
             "color.rgb = pow(vec3(%s) * color.rgb, vec3(2.6)); \n",
             scale);
    }

    if (repr->sys == PL_COLOR_SYSTEM_DOLBYVISION) {
        ident_t scale = SH_FLOAT(pl_color_repr_normalize(repr));
        GLSL("color.rgb *= vec3(%s); \n", scale);
        pl_shader_dovi_reshape(sh, repr->dovi);
    }

    enum pl_color_system orig_sys = repr->sys;
    struct pl_transform3x3 tr = pl_color_repr_decode(repr, params);

    if (memcmp(&tr, &pl_transform3x3_identity, sizeof(tr))) {
        ident_t cmat = sh_var(sh, (struct pl_shader_var) {
            .var  = pl_var_mat3("cmat"),
            .data = PL_TRANSPOSE_3X3(tr.mat.m),
        });

        ident_t cmat_c = sh_var(sh, (struct pl_shader_var) {
            .var  = pl_var_vec3("cmat_c"),
            .data = tr.c,
        });

        GLSL("color.rgb = %s * color.rgb + %s;\n", cmat, cmat_c);
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

        // Inverted from the matrix in the spec, transposed to column major
        static const char *bt2100_lms2rgb = "mat3("
            "  3.43661,  -0.79133, -0.0259499, "
            " -2.50645,    1.9836, -0.0989137, "
            "0.0698454, -0.192271,    1.12486) ";

        // PQ EOTF
        GLSL("color.rgb = pow(max(color.rgb, 0.0), vec3(1.0/%f));   \n"
             "color.rgb = max(color.rgb - vec3(%f), 0.0)            \n"
             "             / (vec3(%f) - vec3(%f) * color.rgb);     \n"
             "color.rgb = pow(color.rgb, vec3(1.0/%f));             \n",
             PQ_M2, PQ_C1, PQ_C2, PQ_C3, PQ_M1);
        // LMS matrix
        GLSL("color.rgb = %s * color.rgb; \n", bt2100_lms2rgb);
        // PQ OETF
        GLSL("color.rgb = pow(max(color.rgb, 0.0), vec3(%f));       \n"
             "color.rgb = (vec3(%f) + vec3(%f) * color.rgb)         \n"
             "             / (vec3(1.0) + vec3(%f) * color.rgb);    \n"
             "color.rgb = pow(color.rgb, vec3(%f));                 \n",
             PQ_M1, PQ_C1, PQ_C2, PQ_C3, PQ_M2);
        break;

    case PL_COLOR_SYSTEM_BT_2100_HLG:
        // HLG OETF^-1
        GLSL("color.rgb = mix(vec3(4.0) * color.rgb * color.rgb,         \n"
             "                exp((color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "                    + vec3(%f),                            \n"
             "                %s(lessThan(vec3(0.5), color.rgb)));       \n",
             HLG_C, HLG_A, HLG_B, sh_bvec(sh, 3));
        // LMS matrix
        GLSL("color.rgb = %s * color.rgb; \n", bt2100_lms2rgb);
        // HLG OETF
        GLSL("color.rgb = mix(vec3(0.5) * sqrt(color.rgb),                     \n"
             "                vec3(%f) * log(color.rgb - vec3(%f)) + vec3(%f), \n"
             "                %s(lessThan(vec3(1.0), color.rgb)));             \n",
             HLG_A, HLG_B, HLG_C, sh_bvec(sh, 3));
        break;

    case PL_COLOR_SYSTEM_DOLBYVISION:;
        // Dolby Vision always outputs BT.2020-referred HPE LMS, so hard-code
        // the inverse LMS->RGB matrix corresponding to this color space.
        struct pl_matrix3x3 dovi_lms2rgb = {{
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
        GLSL("color.rgb = %s * color.rgb; \n", mat);
        // PQ OETF
        GLSL("color.rgb = pow(max(color.rgb, 0.0), vec3(%f));       \n"
             "color.rgb = (vec3(%f) + vec3(%f) * color.rgb)         \n"
             "             / (vec3(1.0) + vec3(%f) * color.rgb);    \n"
             "color.rgb = pow(color.rgb, vec3(%f));                 \n",
             PQ_M1, PQ_C1, PQ_C2, PQ_C3, PQ_M2);
        break;

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
        GLSL("color.rgb = pow(max(color.rgb, vec3(0.0)), vec3(%s)); \n", gamma);
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

    case PL_COLOR_SYSTEM_BT_2100_PQ:;
        // Inverse of the matrix above
        static const char *bt2100_rgb2lms = "mat3("
            "0.412109, 0.166748, 0.024170, "
            "0.523925, 0.720459, 0.075440, "
            "0.063965, 0.112793, 0.900394) ";

        GLSL("color.rgb = pow(max(color.rgb, 0.0), vec3(1.0/%f));   \n"
             "color.rgb = max(color.rgb - vec3(%f), 0.0)            \n"
             "             / (vec3(%f) - vec3(%f) * color.rgb);     \n"
             "color.rgb = pow(color.rgb, vec3(1.0/%f));             \n",
             PQ_M2, PQ_C1, PQ_C2, PQ_C3, PQ_M1);
        GLSL("color.rgb = %s * color.rgb; \n", bt2100_rgb2lms);
        GLSL("color.rgb = pow(color.rgb, vec3(%f));                 \n"
             "color.rgb = (vec3(%f) + vec3(%f) * color.rgb)         \n"
             "             / (vec3(1.0) + vec3(%f) * color.rgb);    \n"
             "color.rgb = pow(color.rgb, vec3(%f));                 \n",
             PQ_M1, PQ_C1, PQ_C2, PQ_C3, PQ_M2);
        break;

    case PL_COLOR_SYSTEM_BT_2100_HLG:
        GLSL("color.rgb = mix(vec3(4.0) * color.rgb * color.rgb,         \n"
             "                exp((color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "                    + vec3(%f),                            \n"
             "                %s(lessThan(vec3(0.5), color.rgb)));       \n",
             HLG_C, HLG_A, HLG_B, sh_bvec(sh, 3));
        GLSL("color.rgb = %s * color.rgb; \n", bt2100_rgb2lms);
        GLSL("color.rgb = mix(vec3(0.5) * sqrt(color.rgb),                     \n"
             "                vec3(%f) * log(color.rgb - vec3(%f)) + vec3(%f), \n"
             "                %s(lessThan(vec3(1.0), color.rgb)));             \n",
             HLG_A, HLG_B, HLG_C, sh_bvec(sh, 3));
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
        ident_t xyzscale = NULL;
        if (repr->sys == PL_COLOR_SYSTEM_XYZ)
            xyzscale = SH_FLOAT(1.0 / pl_color_repr_normalize(&copy));

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

        if (xyzscale)
            GLSL("color.rgb = pow(color.rgb, vec3(1.0/2.6)) * vec3(%s); \n", xyzscale);
    }

    if (repr->alpha == PL_ALPHA_PREMULTIPLIED)
        GLSL("color.rgb *= vec3(color.a); \n");

    GLSL("}\n");
}

static ident_t sh_luma_coeffs(pl_shader sh, enum pl_color_primaries prim)
{
    struct pl_matrix3x3 rgb2xyz;
    rgb2xyz = pl_get_rgb2xyz_matrix(pl_raw_primaries_get(prim));

    // FIXME: Cannot use `const vec3` due to glslang bug #2025
    ident_t coeffs = sh_fresh(sh, "luma_coeffs");
    GLSLH("#define %s vec3(%s, %s, %s) \n", coeffs,
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

    // Note that this clamp may technically violate the definition of
    // ITU-R BT.2100, which allows for sub-blacks and super-whites to be
    // displayed on the display where such would be possible. That said, the
    // problem is that not all gamma curves are well-defined on the values
    // outside this range, so we ignore it and just clamp anyway for sanity.
    GLSL("// pl_shader_linearize           \n"
         "color.rgb = max(color.rgb, 0.0); \n");

    float csp_min = csp->hdr.min_luma / PL_COLOR_SDR_WHITE;
    float csp_max = csp->hdr.max_luma / PL_COLOR_SDR_WHITE;
    csp_max = PL_DEF(csp_max, 1);

    switch (csp->transfer) {
    case PL_COLOR_TRC_SRGB:
        GLSL("color.rgb = mix(color.rgb * vec3(1.0/12.92),               \n"
             "                pow((color.rgb + vec3(0.055))/vec3(1.055), \n"
             "                    vec3(2.4)),                            \n"
             "                %s(lessThan(vec3(0.04045), color.rgb)));   \n",
             sh_bvec(sh, 3));
        goto scale_out;
    case PL_COLOR_TRC_BT_1886: {
        const float lb = powf(csp_min, 1/2.4f);
        const float lw = powf(csp_max, 1/2.4f);
        const float a = powf(lw - lb, 2.4f);
        const float b = lb / (lw - lb);
        GLSL("color.rgb = %s * pow(color.rgb + vec3(%s), vec3(2.4)); \n",
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
             "                %s(lessThan(vec3(0.03125), color.rgb))); \n",
             sh_bvec(sh, 3));
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
        GLSL("color.rgb = %s * color.rgb + vec3(%s);                     \n"
             "color.rgb = mix(vec3(4.0) * color.rgb * color.rgb,         \n"
             "                exp((color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "                    + vec3(%f),                            \n"
             "                %s(lessThan(vec3(0.5), color.rgb)));       \n",
             SH_FLOAT(1 - b), SH_FLOAT(b),
             HLG_C, HLG_A, HLG_B, sh_bvec(sh, 3));
        // OOTF
        GLSL("color.rgb *= 1.0 / 12.0;                                   \n"
             "color.rgb *= %s * pow(max(dot(%s, color.rgb), 0.0), %s);   \n",
             SH_FLOAT(csp_max),
             sh_luma_coeffs(sh, csp->primaries),
             SH_FLOAT(y - 1));
        return;
    }
    case PL_COLOR_TRC_V_LOG:
        GLSL("color.rgb = mix((color.rgb - vec3(0.125)) * vec3(1.0/5.6), \n"
             "    pow(vec3(10.0), (color.rgb - vec3(%f)) * vec3(1.0/%f)) \n"
             "              - vec3(%f),                                  \n"
             "    %s(lessThanEqual(vec3(0.181), color.rgb)));            \n",
             VLOG_D, VLOG_C, VLOG_B, sh_bvec(sh, 3));
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
             "    %s(lessThanEqual(vec3(%f), color.rgb)));                \n",
             SLOG_Q, SLOG_P, SLOG_C, SLOG_A, SLOG_B, SLOG_K2, sh_bvec(sh, 3),
             SLOG_Q);
        return;
    case PL_COLOR_TRC_LINEAR:
    case PL_COLOR_TRC_COUNT:
        break;
    }

    pl_unreachable();

scale_out:
    if (csp_max != 1 || csp_min != 0) {
        GLSL("color.rgb = %s * color.rgb + vec3(%s); \n",
             SH_FLOAT(csp_max - csp_min), SH_FLOAT(csp_min));
    }
}

void pl_shader_delinearize(pl_shader sh, const struct pl_color_space *csp)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    if (csp->transfer == PL_COLOR_TRC_LINEAR)
        return;

    GLSL("// pl_shader_delinearize \n");
    float csp_min = csp->hdr.min_luma / PL_COLOR_SDR_WHITE;
    float csp_max = csp->hdr.max_luma / PL_COLOR_SDR_WHITE;
    csp_max = PL_DEF(csp_max, 1);

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
    case PL_COLOR_TRC_PRO_PHOTO: ;
        if (csp_max != 1 || csp_min != 0) {
            GLSL("color.rgb = %s * color.rgb + vec3(%s); \n",
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
             "                %s(lessThanEqual(vec3(0.0031308), color.rgb))); \n",
             sh_bvec(sh, 3));
        return;
    case PL_COLOR_TRC_BT_1886: {
        const float lb = powf(csp_min, 1/2.4f);
        const float lw = powf(csp_max, 1/2.4f);
        const float a = powf(lw - lb, 2.4f);
        const float b = lb / (lw - lb);
        GLSL("color.rgb = pow(%s * color.rgb, vec3(1.0/2.4)) - vec3(%s); \n",
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
    case PL_COLOR_TRC_PRO_PHOTO:
        GLSL("color.rgb = mix(color.rgb * vec3(16.0),                        \n"
             "                pow(color.rgb, vec3(1.0/1.8)),                 \n"
             "                %s(lessThanEqual(vec3(0.001953), color.rgb))); \n",
             sh_bvec(sh, 3));
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
        GLSL("color.rgb *= 1.0 / %s;                                      \n"
             "color.rgb *= 12.0 * max(1e-6, pow(dot(%s, color.rgb), %s)); \n",
             SH_FLOAT(csp_max),
             sh_luma_coeffs(sh, csp->primaries),
             SH_FLOAT((1 - y) / y));
        // OETF
        GLSL("color.rgb = mix(vec3(0.5) * sqrt(color.rgb),                     \n"
             "                vec3(%f) * log(color.rgb - vec3(%f)) + vec3(%f), \n"
             "                %s(lessThan(vec3(1.0), color.rgb)));             \n"
             "color.rgb = %s * color.rgb + vec3(%s);                           \n",
             HLG_A, HLG_B, HLG_C, sh_bvec(sh, 3),
             SH_FLOAT(1 / (1 - b)), SH_FLOAT(-b / (1 - b)));
        return;
    }
    case PL_COLOR_TRC_V_LOG:
        GLSL("color.rgb = mix(vec3(5.6) * color.rgb + vec3(0.125),       \n"
             "                vec3(%f) * log(color.rgb + vec3(%f))       \n"
             "                    + vec3(%f),                            \n"
             "                %s(lessThanEqual(vec3(0.01), color.rgb))); \n",
             VLOG_C / M_LN10, VLOG_B, VLOG_D, sh_bvec(sh, 3));
        return;
    case PL_COLOR_TRC_S_LOG1:
        GLSL("color.rgb = vec3(%f) * log(color.rgb + vec3(%f)) + vec3(%f);\n",
             SLOG_A / M_LN10, SLOG_B, SLOG_C);
        return;
    case PL_COLOR_TRC_S_LOG2:
        GLSL("color.rgb = mix(vec3(%f) * color.rgb + vec3(%f),                \n"
             "                vec3(%f) * log(vec3(%f) * color.rgb + vec3(%f)) \n"
             "                    + vec3(%f),                                 \n"
             "                %s(lessThanEqual(vec3(0.0), color.rgb)));       \n",
             SLOG_P, SLOG_Q, SLOG_A / M_LN10, SLOG_K2, SLOG_B, SLOG_C,
             sh_bvec(sh, 3));
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

    GLSL("// pl_shader_sigmoidize                                          \n"
         "color = clamp(color, 0.0, 1.0);                                  \n"
         "color = vec4(%s) - log(vec4(1.0) / (color * vec4(%s) + vec4(%s)) \n"
         "                         - vec4(1.0)) * vec4(%s);                \n",
         SH_FLOAT(center), SH_FLOAT(scale), SH_FLOAT(offset), SH_FLOAT(1.0 / slope));
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

    GLSL("// pl_shader_unsigmoidize                                           \n"
         "color = clamp(color, 0.0, 1.0);                                     \n"
         "color = vec4(%s) / (vec4(1.0) + exp(vec4(%s) * (vec4(%s) - color))) \n"
         "           - vec4(%s);                                              \n",
         SH_FLOAT(1.0 / scale), SH_FLOAT(slope), SH_FLOAT(center), SH_FLOAT(offset / scale));
}

const struct pl_peak_detect_params pl_peak_detect_default_params = { PL_PEAK_DETECT_DEFAULTS };

struct sh_tone_map_obj {
    struct pl_tone_map_params params;
    pl_shader_obj lut;

    // Peak detection state
    pl_buf peak_buf;
    struct pl_shader_desc desc;
    float margin;
};

static void sh_tone_map_uninit(pl_gpu gpu, void *ptr)
{
    struct sh_tone_map_obj *obj = ptr;
    pl_shader_obj_destroy(&obj->lut);
    pl_buf_destroy(gpu, &obj->peak_buf);
    memset(obj, 0, sizeof(*obj));
}

static inline float iir_coeff(float rate)
{
    float a = 1.0 - cos(1.0 / rate);
    return sqrt(a*a + 2*a) - a;
}

bool pl_shader_detect_peak(pl_shader sh, struct pl_color_space csp,
                           pl_shader_obj *state,
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

    struct sh_tone_map_obj *obj;
    obj = SH_OBJ(sh, state, PL_SHADER_OBJ_TONE_MAP, struct sh_tone_map_obj,
                 sh_tone_map_uninit);
    if (!obj)
        return false;

    pl_gpu gpu = SH_GPU(sh);
    obj->margin = params->overshoot_margin;

    if (!obj->peak_buf) {
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
        struct pl_buf_params buf_params = {
            .size = size,
            .host_readable = true,
            .memory_type = PL_BUF_MEM_DEVICE,
            .storable = true,
            .initial_data = zero,
            .debug_tag = PL_DEBUG_TAG,
        };

        // Attempt creating host-readable SSBO first, suppress errors
        pl_log_level_cap(gpu->log, PL_LOG_DEBUG);
        obj->peak_buf = pl_buf_create(gpu, &buf_params);
        pl_log_level_cap(gpu->log, PL_LOG_NONE);

        if (!obj->peak_buf) {
            // Fall back to non-host-readable SSBO
            buf_params.host_readable = false;
            obj->peak_buf = pl_buf_create(gpu, &buf_params);
        }

        obj->desc.binding.object = obj->peak_buf;
    }

    if (!obj->peak_buf) {
        SH_FAIL(sh, "Failed creating peak detection SSBO!");
        return false;
    }

    // Attach the SSBO and perform the peak detection logic
    obj->desc.desc.access = PL_DESC_ACCESS_READWRITE;
    obj->desc.memory = PL_MEMORY_COHERENT;
    sh_desc(sh, obj->desc);

    sh_describe(sh, "peak detection");
    GLSL("// pl_shader_detect_peak \n"
         "{                        \n"
         "vec4 color_orig = color; \n");

    // Decode the color into linear light absolute scale representation
    pl_color_space_infer(&csp);
    pl_shader_linearize(sh, &csp);

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
    if (sh_glsl(sh).subgroup_size) {
        GLSL("int group_max = subgroupMax(isig_max);    \n"
             "int group_sum = subgroupAdd(isig_log);    \n"
             "if (subgroupElect()) {                    \n"
             "    atomicMax(%s, group_max);             \n"
             "    atomicAdd(%s, group_sum);             \n"
             "}                                         \n"
             "barrier();                                \n",
             wg_max, wg_sum);
    } else {
        GLSL("atomicMax(%s, isig_max);  \n"
             "atomicAdd(%s, isig_log);  \n"
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
          "}                                                                    \n"
          "barrier();                                                           \n",
          wg_sum, wg_max);

    // Finally, to update the global state per dispatch, we increment a counter
    GLSLF("if (gl_LocalInvocationIndex == 0u) {                                 \n"
          "    uint num_wg = gl_NumWorkGroups.x * gl_NumWorkGroups.y;           \n"
          "    if (atomicAdd(counter, 1u) == num_wg - 1u) {                     \n"
          "        vec2 cur = vec2(float(frame_sum) / float(num_wg), frame_max);\n"
          "        cur *= vec2(1.0 / %f, 1.0 / %f);                             \n"
          "        cur.x = exp(cur.x);                                          \n"
          "        cur.y = max(cur.y, %s);                                      \n",
          log_scale, sig_scale, SH_FLOAT(PL_DEF(params->minimum_peak, 1.0)));

    // Set the initial value accordingly if it contains no data
    GLSLF("        if (average.y == 0.0) \n"
          "            average = cur;    \n");

    // Use an IIR low-pass filter to smooth out the detected values
    GLSLF("        average += %s * (cur - average); \n",
          SH_FLOAT(iir_coeff(PL_DEF(params->smoothing_period, 100.0))));

    // Scene change hysteresis
    float log_db = 10.0 / log(10.0);
    if (params->scene_threshold_low > 0 && params->scene_threshold_high > 0) {
        GLSLF("    float delta = abs(log(cur.x / average.x));               \n"
              "    average = mix(average, cur, smoothstep(%s, %s, delta));  \n",
              SH_FLOAT(params->scene_threshold_low / log_db),
              SH_FLOAT(params->scene_threshold_high / log_db));
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

bool pl_get_detected_peak(const pl_shader_obj state,
                          float *out_peak, float *out_avg)
{
    if (!state || state->type != PL_SHADER_OBJ_TONE_MAP)
        return false;

    struct sh_tone_map_obj *obj = state->priv;
    pl_gpu gpu = state->gpu;
    pl_buf buf = obj->peak_buf;
    if (!buf)
        return false;

    float average[2] = {0};
    pl_assert(obj->peak_buf->params.size >= sizeof(average));

    if (buf->params.host_readable) {

        // We can read directly from the SSBO
        if (!pl_buf_read(gpu, buf, 0, average, sizeof(average))) {
            PL_ERR(gpu, "Failed reading from peak detect state buffer");
            return false;
        }

    } else {

        // We can't read directly from the SSBO, go via an intermediary
        pl_buf tmp = pl_buf_create(gpu, pl_buf_params(
            .size = sizeof(average),
            .host_readable = true,
        ));

        if (!tmp) {
            PL_ERR(gpu, "Failed creating buffer for SSBO read-back");
            return false;
        }

        pl_buf_copy(gpu, tmp, 0, buf, 0, sizeof(average));
        if (!pl_buf_read(gpu, tmp, 0, average, sizeof(average))) {
            PL_ERR(gpu, "Failed reading from SSBO read-back buffer");
            pl_buf_destroy(gpu, &tmp);
            return false;
        }
        pl_buf_destroy(gpu, &tmp);

    }

    *out_avg = average[0];
    *out_peak = average[1];

    if (obj->margin > 0.0) {
        *out_peak *= 1.0 + obj->margin;
        *out_peak = PL_MIN(*out_peak, 10000 / PL_COLOR_SDR_WHITE);
    }

    return true;
}

void pl_reset_detected_peak(pl_shader_obj state)
{
    if (!state || state->type != PL_SHADER_OBJ_TONE_MAP)
        return;

    struct sh_tone_map_obj *obj = state->priv;
    pl_buf_destroy(state->gpu, &obj->peak_buf);
}

const struct pl_color_map_params pl_color_map_default_params = { PL_COLOR_MAP_DEFAULTS };

// Get the LUT range for the dynamic tone mapping LUT
static void dynamic_lut_range(float *idx_min, float *idx_max,
                              const struct pl_tone_map_params *params)
{
    float max_peak = params->input_max;
    float min_peak = pl_hdr_rescale(params->output_scaling,
                                    params->input_scaling,
                                    params->output_max);

    // Add some headroom to avoid no-op tone mapping. (This is because
    // many curves are not good approximations of a no-op tone mapping
    // function even when tone mapping to very similar values)
    *idx_min = PL_MIX(min_peak, max_peak, 0.05f);
    *idx_max = max_peak;
}

static void fill_lut(void *data, const struct sh_lut_params *params)
{
    struct pl_tone_map_params *lut_params = params->priv;
    assert(lut_params->lut_size == params->width);
    float *lut = data;

    if (params->height) {
        // Dynamic tone-mapping, generate a LUT curve for each possible peak
        float idx_min, idx_max;
        dynamic_lut_range(&idx_min, &idx_max, lut_params);
        for (int i = 0; i < params->height; i++) {
            float x = (float) i / (params->height - 1);
            lut_params->input_max = PL_MIX(idx_min, idx_max, x);
            pl_tone_map_generate(lut, lut_params);
            lut += params->width;
        }
        lut_params->input_max = idx_max; // sanity
    } else {
        // Static tone-mapping, generate only a single curve
        pl_tone_map_generate(lut, lut_params);
    }
}

static void tone_map(pl_shader sh,
                     const struct pl_color_space *src,
                     const struct pl_color_space *dst,
                     pl_shader_obj *state,
                     const struct pl_color_map_params *params)
{
    float src_min = pl_hdr_rescale(PL_HDR_NITS, PL_HDR_NORM, src->hdr.min_luma),
          src_max = pl_hdr_rescale(PL_HDR_NITS, PL_HDR_NORM, src->hdr.max_luma),
          dst_min = pl_hdr_rescale(PL_HDR_NITS, PL_HDR_NORM, dst->hdr.min_luma),
          dst_max = pl_hdr_rescale(PL_HDR_NITS, PL_HDR_NORM, dst->hdr.max_luma);

    // Some tone mapping functions don't handle values of absolute 0 very well,
    // so clip the minimums to a very small positive value
    src_min = PL_MAX(src_min, 1e-7);
    dst_min = PL_MAX(dst_min, 1e-7);

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
        .output_min = dst_min,
        .output_max = dst_max,
    };

    enum pl_tone_map_mode mode = params->tone_mapping_mode;
    if (params->tone_mapping_algo) {
        // Backwards compatibility
        static const struct pl_tone_map_function *
        funcs[PL_TONE_MAPPING_ALGORITHM_COUNT] = {
            [PL_TONE_MAPPING_CLIP]      = &pl_tone_map_clip,
            [PL_TONE_MAPPING_MOBIUS]    = &pl_tone_map_mobius,
            [PL_TONE_MAPPING_REINHARD]  = &pl_tone_map_reinhard,
            [PL_TONE_MAPPING_HABLE]     = &pl_tone_map_hable,
            [PL_TONE_MAPPING_GAMMA]     = &pl_tone_map_gamma,
            [PL_TONE_MAPPING_LINEAR]    = &pl_tone_map_linear,
            [PL_TONE_MAPPING_BT_2390]   = &pl_tone_map_bt2390,
        };
        lut_params.function = funcs[params->tone_mapping_algo];

        // Backwards compatibility with older API, explicitly default the tone
        // mapping mode based on the previous values of desat_str etc.
        if (params->desaturation_strength == 1 && params->desaturation_exponent == 0) {
            mode = PL_DEF(mode, PL_TONE_MAP_RGB);
        } else if (params->desaturation_strength > 0) {
            mode = PL_DEF(mode, PL_TONE_MAP_HYBRID);
        } else {
            mode = PL_DEF(mode, PL_TONE_MAP_LUMA);
        }
    }

    if (pl_tone_map_params_noop(&lut_params))
        return;

    sh_describe(sh, "tone mapping");
    const struct pl_tone_map_function *fun = lut_params.function;
    struct sh_tone_map_obj *obj = NULL;
    ident_t lut = NULL;

    bool can_fixed = !params->force_tone_mapping_lut;
    bool is_noop = can_fixed && (!fun || fun == &pl_tone_map_clip);
    bool pure_bpc = can_fixed && src_max == dst_max;
    bool dynamic_peak = false;

    if (state && !(is_noop || pure_bpc)) {
        obj = SH_OBJ(sh, state, PL_SHADER_OBJ_TONE_MAP, struct sh_tone_map_obj,
                     sh_tone_map_uninit);
        if (!obj)
            return;

        // Only use dynamic peak detection for range reductions
        dynamic_peak = obj->peak_buf && src_max > dst_max;

        lut = sh_lut(sh, sh_lut_params(
            .object = &obj->lut,
            .method = SH_LUT_AUTO,
            .type = PL_VAR_FLOAT,
            .width = lut_params.lut_size,
            .height = dynamic_peak ? lut_params.lut_size : 0,
            .comps = 1,
            .linear = true,
            .update = !pl_tone_map_params_equal(&lut_params, &obj->params),
            .fill = fill_lut,
            .priv = &lut_params,
        ));
        if (!lut)
            is_noop = true;
        obj->params = lut_params;
    }

    // Hard-clamp the input values to the claimed input peak. Do this
    // per-channel to fix issues with excessively oversaturated highlights in
    // broken files that contain values outside their stated brightness range.
    GLSL("color.rgb = clamp(color.rgb, %s, %s); \n",
         SH_FLOAT(src_min), SH_FLOAT(src_max));

    if (is_noop) {

        GLSL("#define tone_map(x) clamp((x), %s, %s) \n",
             SH_FLOAT(dst_min), SH_FLOAT(dst_max));

    } else if (pure_bpc) {

        // Pure black point compensation
        const float scale = (dst_max - dst_min) / (src_max - src_min);
        GLSL("#define tone_map(x) (%s * (x) + %s) \n",
             SH_FLOAT(scale), SH_FLOAT(dst_min - scale * src_min));

    } else if (dynamic_peak) {

        // Dynamic 2D LUT
        obj->desc.desc.access = PL_DESC_ACCESS_READONLY;
        obj->desc.memory = 0;
        sh_desc(sh, obj->desc);

        float idx_min, idx_max;
        dynamic_lut_range(&idx_min, &idx_max, &lut_params);

        GLSL("const float idx_min = %s;                                     \n"
             "const float idx_max = %s;                                     \n"
             "float input_max = idx_max;                                    \n"
             "if (average.y != 0.0) {                                       \n"
             "    float sig_peak = average.y;                               \n",
             SH_FLOAT(idx_min), SH_FLOAT(idx_max));
        // Allow a tiny bit of extra overshoot for the smoothed peak
        if (obj->margin > 0)
            GLSL("sig_peak *= %s; \n", SH_FLOAT(obj->margin + 1));
        GLSL("    input_max = clamp(sqrt(sig_peak), idx_min, idx_max);      \n"
             "}                                                             \n");

        // Sample the 2D LUT from a position determined by the detected max
        GLSL("const float input_min = %s;                                   \n"
             "float scale = 1.0 / (input_max - input_min);                  \n"
             "float curve = (input_max - idx_min) / (idx_max - idx_min);    \n"
             "float base = -input_min * scale;                              \n"
             "#define tone_map(x) (%s(vec2(scale * sqrt(x) + base, curve))) \n",
             SH_FLOAT(lut_params.input_min), lut);

    } else {

        // Regular 1D LUT
        const float lut_range = lut_params.input_max - lut_params.input_min;
        GLSL("#define tone_map(x) (%s(%s * sqrt(x) + %s))   \n",
             lut, SH_FLOAT(1.0f / lut_range),
             SH_FLOAT(-lut_params.input_min / lut_range));

    }

    if (mode == PL_TONE_MAP_AUTO) {
        if (is_noop || pure_bpc || src_max == dst_max) {
            // No-op, clip, pure BPC, etc. - do this per-channel
            mode = PL_TONE_MAP_RGB;
        } else if (src_max / dst_max > 10) {
            // Extreme reduction: Pick hybrid to avoid blowing out highlights
            mode = PL_TONE_MAP_HYBRID;
        } else {
            mode = PL_TONE_MAP_LUMA;
        }
    }

    const float ct = params->tone_mapping_crosstalk;
    const struct pl_matrix3x3 crosstalk = {{
        { 1 - 2*ct, ct,       ct       },
        { ct,       1 - 2*ct, ct       },
        { ct,       ct,       1 - 2*ct },
    }};

    // PL_TONE_MAP_LUMA can do the crosstalk for free
    bool needs_ct = ct && mode != PL_TONE_MAP_LUMA;
    if (needs_ct) {
        GLSL("color.rgb = %s * color.rgb; \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_mat3("crosstalk"),
            .data = crosstalk.m, // no need to transpose, matrix is symmetric
        }));
    }

    switch (mode) {
    case PL_TONE_MAP_RGB:
        for (int c = 0; c < 3; c++)
            GLSL("color[%d] = tone_map(color[%d]); \n", c, c);
        break;

    case PL_TONE_MAP_MAX:
        GLSL("float sig_max = max(max(color.r, color.g), color.b);  \n"
             "color.rgb *= tone_map(sig_max) / max(sig_max, %s);    \n",
             SH_FLOAT(dst_min));
        break;

    case PL_TONE_MAP_HYBRID: {
        GLSL("float luma_orig = dot(%s, color.rgb);                 \n"
             "float luma_new = tone_map(luma_orig);                 \n"
             "vec3 color_lin = luma_new / luma_orig * color.rgb;    \n",
             sh_luma_coeffs(sh, src->primaries));
        for (int c = 0; c < 3; c++)
            GLSL("color[%d] = tone_map(color[%d]); \n", c, c);

        const float y = 2.4f;
        const float lb = powf(dst_min, y);
        const float lw = powf(dst_max, y);
        const float a = 1 / (lw - lb);
        const float b = -lb * a;
        GLSL("float coeff = %s * pow(luma_new, %f) + %s;            \n"
             "color.rgb = mix(color_lin, color.rgb, coeff);         \n",
             SH_FLOAT(a), y, SH_FLOAT(b));
        break;
    }

    case PL_TONE_MAP_LUMA: {
        const struct pl_raw_primaries *prim = pl_raw_primaries_get(src->primaries);
        struct pl_matrix3x3 rgb2xyz = pl_get_rgb2xyz_matrix(prim);
        pl_matrix3x3_mul(&rgb2xyz, &crosstalk);

        // Normalize X and Z by the white point
        for (int i = 0; i < 3; i++) {
            rgb2xyz.m[0][i] /= pl_cie_X(prim->white);
            rgb2xyz.m[2][i] /= pl_cie_Z(prim->white);
            rgb2xyz.m[0][i] -= rgb2xyz.m[1][i];
            rgb2xyz.m[2][i] -= rgb2xyz.m[1][i];
        }

        GLSL("vec3 xyz = %s * color.rgb; \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_mat3("rgb2xyz"),
            .data = PL_TRANSPOSE_3X3(rgb2xyz.m),
        }));

        // Tuned to meet the desired desaturation at 1000 -> SDR
        float desat = dst_max > src_max ? 1.075f : 1.1f;
        float exponent = logf(desat) / logf(1000 / PL_COLOR_SDR_WHITE);
        GLSL("float orig = max(xyz.y, %s);                      \n"
             "xyz.y = tone_map(xyz.y);                          \n"
             "xyz.xz *= pow(xyz.y / orig, %s) * xyz.y / orig;   \n",
             SH_FLOAT(dst_min), SH_FLOAT(exponent));

        // Extra luminance correction when reducing dynamic range
        if (src_max > dst_max)
            GLSL("xyz.y -= max(0.1 * xyz.x, 0.0); \n");

        pl_matrix3x3_invert(&rgb2xyz);
        GLSL("color.rgb = %s * xyz; \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_mat3("xyz2rgb"),
            .data = PL_TRANSPOSE_3X3(rgb2xyz.m),
        }));
        break;
    }

    case PL_TONE_MAP_AUTO:
    case PL_TONE_MAP_MODE_COUNT:
        pl_unreachable();
    }

    if (needs_ct) {
        const float s = 1 / (1 - 3 * ct);
        const struct pl_matrix3x3 crosstalk_inv = {{
            {-ct * s + s, -ct * s,     -ct * s     },
            {-ct * s,     -ct * s + s, -ct * s     },
            {-ct * s,     -ct * s,     -ct * s + s },
        }};
        GLSL("color.rgb = %s * color.rgb; \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_mat3("crosstalk_inv"),
            .data = crosstalk_inv.m,
        }));
    }

    GLSL("#undef tone_map \n");
}

static void adapt_colors(pl_shader sh,
                         const struct pl_color_space *src,
                         const struct pl_color_space *dst,
                         const struct pl_color_map_params *params)
{
    if (src->primaries == dst->primaries)
        return;

    const struct pl_raw_primaries *csp_src, *csp_dst;
    struct pl_matrix3x3 cms_mat;
    csp_src = pl_raw_primaries_get(src->primaries),
    csp_dst = pl_raw_primaries_get(dst->primaries);
    cms_mat = pl_get_color_mapping_matrix(csp_src, csp_dst, params->intent);

    GLSL("color.rgb = %s * color.rgb;\n", sh_var(sh, (struct pl_shader_var) {
        .var = pl_var_mat3("cms_matrix"),
        .data = PL_TRANSPOSE_3X3(cms_mat.m),
    }));

    if (pl_primaries_superset(&dst->hdr.prim, &src->hdr.prim))
        return;

    enum pl_gamut_mode mode = params->gamut_mode;
    if (params->gamut_warning)
        mode = PL_GAMUT_WARN;
    if (params->gamut_clipping)
        mode = PL_GAMUT_DESATURATE;
    if (mode == PL_GAMUT_CLIP)
        return;

    // Normalize colors to range [0-1]
    float lb = dst->hdr.min_luma / PL_COLOR_SDR_WHITE;
    float lw = dst->hdr.max_luma / PL_COLOR_SDR_WHITE;
    GLSL("color.rgb = %s * color.rgb + %s; \n",
         SH_FLOAT(1 / (lw - lb)), SH_FLOAT(-lb / (lw - lb)));

    switch (mode) {
    case PL_GAMUT_WARN:
        GLSL("if (any(lessThan(color.rgb, vec3(-0.005))) ||     \n"
             "    any(greaterThan(color.rgb, vec3(1.005))))     \n"
             "    color.rgb = vec3(1.0, 0.0, 1.0); // magenta   \n");
        break;

    case PL_GAMUT_DARKEN: {
        struct pl_matrix3x3 gamut_mat;
        gamut_mat = pl_get_color_mapping_matrix(&src->hdr.prim, &dst->hdr.prim,
                                                PL_INTENT_ABSOLUTE_COLORIMETRIC);
        float cmax = 1;
        for (int i = 0; i < 3; i++)
            cmax = PL_MAX(cmax, gamut_mat.m[i][i]);
        GLSL("color.rgb *= %s; \n", SH_FLOAT(1 / cmax));
        break;
    }

    case PL_GAMUT_DESATURATE:
        GLSL("float cmin = min(min(color.r, color.g), color.b); \n"
             "float luma = clamp(dot(%s, color.rgb), 0.0, 1.0); \n"
             "if (cmin < 0.0 - 1e-6)                            \n"
             "    color.rgb = mix(color.rgb, vec3(luma),        \n"
             "                    -cmin / (luma - cmin));       \n"
             "float cmax = max(max(color.r, color.g), color.b); \n"
             "if (cmax > 1.0 + 1e-6)                            \n"
             "    color.rgb = mix(color.rgb, vec3(luma),        \n"
             "                    (1.0 - cmax) / (luma - cmax));\n",
            sh_luma_coeffs(sh, dst->primaries));
        break;

    case PL_GAMUT_CLIP:
    case PL_GAMUT_MODE_COUNT:
        pl_unreachable();
    }

    // Undo normalization
    GLSL("color.rgb = %s * color.rgb + %s; \n",
         SH_FLOAT(lw - lb), SH_FLOAT(lb));
}

void pl_shader_color_map(pl_shader sh, const struct pl_color_map_params *params,
                         struct pl_color_space src, struct pl_color_space dst,
                         pl_shader_obj *tone_map_state,
                         bool prelinearized)
{
    pl_color_space_infer(&src);
    pl_color_space_infer_ref(&dst, &src);
    if (pl_color_space_equal(&src, &dst)) {
        if (prelinearized)
            pl_shader_delinearize(sh, &dst);
        return;
    }

    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

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
    if (!params || !params->cones)
        return;

    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    sh_describe(sh, "cone distortion");
    GLSL("// pl_shader_cone_distort\n");
    GLSL("{\n");

    pl_color_space_infer(&csp);
    pl_shader_linearize(sh, &csp);

    struct pl_matrix3x3 cone_mat;
    cone_mat = pl_get_cone_matrix(params, pl_raw_primaries_get(csp.primaries));
    GLSL("color.rgb = %s * color.rgb;\n", sh_var(sh, (struct pl_shader_var) {
        .var = pl_var_mat3("cone_mat"),
        .data = PL_TRANSPOSE_3X3(cone_mat.m),
    }));

    pl_shader_delinearize(sh, &csp);
    GLSL("}\n");
}

struct sh_dither_obj {
    enum pl_dither_method method;
    pl_shader_obj lut;
};

static void sh_dither_uninit(pl_gpu gpu, void *ptr)
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
        return;

    case PL_DITHER_BLUE_NOISE:
        pl_assert(params->width == params->height);
        pl_generate_blue_noise(data, params->width);
        return;

    case PL_DITHER_ORDERED_FIXED:
    case PL_DITHER_WHITE_NOISE:
    case PL_DITHER_METHOD_COUNT:
        return;
    }

    pl_unreachable();
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
    case PL_DITHER_METHOD_COUNT:
        break;
    }

    pl_unreachable();
}

void pl_shader_dither(pl_shader sh, int new_depth,
                      pl_shader_obj *dither_state,
                      const struct pl_dither_params *params)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    if (new_depth <= 0 || new_depth > 256) {
        PL_WARN(sh, "Invalid dither depth: %d.. ignoring", new_depth);
        return;
    }

    sh_describe(sh, "dithering");
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
        lut = sh_lut(sh, sh_lut_params(
            .object = &obj->lut,
            .type = PL_VAR_FLOAT,
            .width = lut_size,
            .height = lut_size,
            .comps = 1,
            .update = changed,
            .fill = fill_dither_matrix,
            .priv = obj,
        ));
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
        GLSL("vec2 pos = fract(gl_FragCoord.xy * 1.0/%s);\n", SH_FLOAT(size));

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
        GLSL("bias = %s.x;\n", prng);
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

    case PL_DITHER_BLUE_NOISE:
    case PL_DITHER_ORDERED_LUT:
        pl_assert(lut);
        GLSL("bias = %s(ivec2(pos * %s));\n", lut, SH_FLOAT(lut_size));
        break;

    case PL_DITHER_METHOD_COUNT:
        pl_unreachable();
    }

    uint64_t scale = (1LLU << new_depth) - 1;
    GLSL("color = vec4(%llu.0) * color + vec4(bias); \n"
         "color = floor(color) * vec4(1.0 / %llu.0); \n"
         "}                                          \n",
         (long long unsigned) scale, (long long unsigned) scale);
}

const struct pl_dither_params pl_dither_default_params = { PL_DITHER_DEFAULTS };
