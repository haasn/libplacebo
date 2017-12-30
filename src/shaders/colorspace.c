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

    GLSL("// pl_shader_decode_color\n");

    // For the non-linear color systems we need some special input handling
    // to make sure we don't accidentally screw everything up because of the
    // alpha multiplication, which only commutes with linear operations.
    bool is_nonlinear = !pl_color_system_is_linear(repr->sys);
    if (is_nonlinear && repr->alpha == PL_ALPHA_PREMULTIPLIED) {
        GLSL("color.rgb /= vec3(color.a);\n");
        repr->alpha = PL_ALPHA_INDEPENDENT;
    }

    // XYZ needs special handling due to the input gamma logic
    if (repr->sys == PL_COLOR_SYSTEM_XYZ) {
        float scale = pl_color_repr_normalize(repr);
        GLSL("color.rgb = pow(%f * color.rgb, vec3(2.6));\n", scale);
    }

    enum pl_color_system orig_sys = repr->sys;
    struct pl_transform3x3 tr = pl_color_repr_decode(repr, params);

    ident_t cmat = sh_var(sh, (struct pl_shader_var) {
        .var  = ra_var_mat3("cmat"),
        .data = PL_TRANSPOSE_3X3(tr.mat.m),
    });

    ident_t cmat_c = sh_var(sh, (struct pl_shader_var) {
        .var  = ra_var_vec3("cmat_m"),
        .data = tr.c,
    });

    GLSL("color.rgb = %s * color.rgb + %s;\n", cmat, cmat_c);

    if (orig_sys == PL_COLOR_SYSTEM_BT_2020_C) {
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
        GLSL("// constant luminance conversion                            \n"
             "color.br = color.br * mix(vec2(1.5816, 0.9936),             \n"
             "                          vec2(1.9404, 1.7184),             \n"
             "                          lessThanEqual(color.br, vec2(0))) \n"
             "           + color.gg;                                      \n"
        // Expand channels to camera-linear light. This shader currently just
        // assumes everything uses the BT.2020 12-bit gamma function, since the
        // difference between 10 and 12-bit is negligible for anything other
        // than 12-bit content.
             "color.rgb = mix(color.rgb * vec3(1.0/4.5),                       \n"
             "                pow((color.rgb + vec3(0.0993))*vec3(1.0/1.0993), \n"
             "                    vec3(1.0/0.45)),                             \n"
             "                lessThanEqual(vec3(0.08145), color.rgb));        \n"
        // Calculate the green channel from the expanded RYcB
        // The BT.2020 specification says Yc = 0.2627*R + 0.6780*G + 0.0593*B
             "color.g = (color.g - 0.2627*color.r - 0.0593*color.b)*1.0/0.6780; \n"
        // Recompress to receive the R'G'B' result, same as other systems
             "color.rgb = mix(color.rgb * vec3(4.5),                    \n"
             "                vec3(1.0993) * pow(color.rgb, vec3(0.45)) \n"
             "                   - vec3(0.0993),                        \n"
             "                lessThanEqual(vec3(0.0181), color.rgb));  \n");
    }

    if (repr->alpha == PL_ALPHA_INDEPENDENT) {
        GLSL("color.rgb *= vec3(color.a)\n");
        repr->alpha = PL_ALPHA_PREMULTIPLIED;
    }
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
             "                lessThan(vec3(0.04045), color.rgb));       \n");
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
        GLSL("color.rgb = mix(color.rgb * vec3(1.0/16.0),          \n"
             "                pow(color.rgb, vec3(1.8)),           \n"
             "                lessThan(vec3(0.03125), color.rgb)); \n");
        break;
    case PL_COLOR_TRC_PQ:
        GLSL("color.rgb = pow(color.rgb, vec3(1.0/%f));         \n"
             "color.rgb = max(color.rgb - vec3(%f), 0.0)        \n"
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
    case PL_COLOR_TRC_UNKNOWN:
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

// Applies the OOTF / inverse OOTF
static void pl_shader_ootf(struct pl_shader *sh, enum pl_color_light light,
                           ident_t luma)
{
    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    if (!light || light == PL_COLOR_LIGHT_DISPLAY)
        return;

    GLSL("// pl_shader_ootf                \n"
         "color.rgb = max(color.rgb, 0.0); \n");

    switch (light)
    {
    case PL_COLOR_LIGHT_SCENE_HLG:
        // HLG OOTF from BT.2100, assuming a reference display with a
        // peak of 1000 cd/m² -> gamma = 1.2
        GLSL("color.rgb *= vec3(%f * pow(dot(%s, color.rgb), 0.2));\n",
             (1000 / PL_COLOR_REF_WHITE) / pow(12, 1.2), luma);
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
}

static void pl_shader_inverse_ootf(struct pl_shader *sh,
                                   enum pl_color_light light, ident_t luma)
{
    if (!light || light == PL_COLOR_LIGHT_DISPLAY)
        return;

    GLSL("// pl_shader_inverse_ootf        \n"
         "color.rgb = max(color.rgb, 0.0); \n");

    switch (light)
    {
    case PL_COLOR_LIGHT_SCENE_HLG:
        GLSL("color.rgb *= vec3(1.0/%f);                                \n"
             "color.rgb /= vec3(max(1e-6, pow(dot(%s, color.rgb),       \n"
             "                                0.2/1.2)));               \n",
             (1000 / PL_COLOR_REF_WHITE) / pow(12, 1.2), luma);
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
}

const struct pl_color_map_params pl_color_map_default_params = {
    .intent                  = PL_INTENT_RELATIVE_COLORIMETRIC,
    .tone_mapping_algo       = PL_TONE_MAPPING_HABLE,
    .tone_mapping_desaturate = 0.5,
    .peak_detect_frames      = 10,
};

struct sh_peak_obj {
    const struct ra *ra;
    const struct ra_buf *buf;
};

static void sh_peak_uninit(const struct ra *ra, void *ptr)
{
    struct sh_peak_obj *obj = ptr;
    ra_buf_destroy(obj->ra, &obj->buf);
    *obj = (struct sh_peak_obj) {0};
}

static void hdr_update_peak(struct pl_shader *sh, struct pl_shader_obj **state,
                            const struct pl_color_map_params *params)
{
    if (!state)
        return;

    int frames = PL_DEF(params->peak_detect_frames, 10);
    if (frames < 1 || frames > 1000) {
        PL_ERR(sh, "Parameter peak_detect_frames must be >= 1 and <= 1000 "
               "(was %d).", frames);
        return;
    }

    struct sh_peak_obj *obj;
    obj = SH_OBJ(sh, state, PL_SHADER_OBJ_PEAK_DETECT, struct sh_peak_obj,
                 sh_peak_uninit);
    if (!obj)
        return;

    if (!sh_try_compute(sh, 8, 8, true, sizeof(uint32_t))) {
        PL_WARN(sh, "HDR peak detection requires compute shaders.. disabling");
        return;
    }

    const struct ra *ra = sh->ra;
    obj->ra = ra;

    struct ra_var idx, num, ctr, max, sum;
    idx = ra_var_uint(sh_fresh(sh, "index"));
    num = ra_var_uint(sh_fresh(sh, "number"));
    ctr = ra_var_uint(sh_fresh(sh, "counter"));
    max = ra_var_uint(sh_fresh(sh, "frames_max"));
    sum = ra_var_uint(sh_fresh(sh, "frames_sum"));
    max.dim_a = sum.dim_a = frames + 1;

    struct ra_var max_total, sum_total;
    max_total = ra_var_uint(sh_fresh(sh, "max_total"));
    sum_total = ra_var_uint(sh_fresh(sh, "sum_total"));

    // Attempt packing the peak detection SSBO
    struct ra_desc ssbo = {
        .name   = "PeakDetect",
        .type   = RA_DESC_BUF_STORAGE,
        .access = RA_DESC_ACCESS_READWRITE,
    };

    struct ra_var_layout idx_l, num_l, ctr_l, max_l, sum_l, max_tl, sum_tl;
    bool ok = true;
    ok &= ra_buf_desc_append(sh->tmp, ra, &ssbo, &idx_l, idx);
    ok &= ra_buf_desc_append(sh->tmp, ra, &ssbo, &num_l, num);
    ok &= ra_buf_desc_append(sh->tmp, ra, &ssbo, &ctr_l, ctr);
    ok &= ra_buf_desc_append(sh->tmp, ra, &ssbo, &max_l, max);
    ok &= ra_buf_desc_append(sh->tmp, ra, &ssbo, &sum_l, sum);
    ok &= ra_buf_desc_append(sh->tmp, ra, &ssbo, &max_tl, max_total);
    ok &= ra_buf_desc_append(sh->tmp, ra, &ssbo, &sum_tl, sum_total);

    if (!ok) {
        PL_WARN(sh, "HDR peak detection exhausts device limits.. disabling");
        talloc_free(ssbo.buffer_vars);
        return;
    }

    // Create the SSBO if necessary
    size_t size = ra_buf_desc_size(&ssbo);
    if (!obj->buf || obj->buf->params.size != size) {
        PL_TRACE(sh, "(Re)creating HDR peak detection SSBO");

        void *data = talloc_zero_size(NULL, size);
        ra_buf_destroy(ra, &obj->buf);
        obj->buf = ra_buf_create(ra, &(struct ra_buf_params) {
            .type = RA_BUF_STORAGE,
            .size = ra_buf_desc_size(&ssbo),
            .initial_data = data,
        });
        talloc_free(data);
    }

    if (!obj->buf) {
        PL_ERR(sh, "Failed creating peak detection SSBO!");
        return;
    }

    // Attach the SSBO and perform the peak detection logic
    sh_desc(sh, (struct pl_shader_desc) {
        .desc = ssbo,
        .object = obj->buf,
    });

    // For performance, we want to do as few atomic operations on global
    // memory as possible, so use an atomic in shmem for the work group.
    ident_t wg_sum = sh_fresh(sh, "wg_sum");
    GLSLH("shared uint %s;\n", wg_sum);
    GLSL("%s = 0;\n", wg_sum);

    // Have each thread update the work group sum with the local value
    GLSL("barrier();                     \n"
         "atomicAdd(%s, uint(sig * %f)); \n",
         wg_sum, PL_COLOR_REF_WHITE);

    // Have one thread per work group update the global atomics. We use the
    // work group average even for the global sum, to make the values slightly
    // more stable and smooth out tiny super-highlights.
    GLSL("memoryBarrierShared();                                            \n"
         "barrier();                                                        \n"
         "if (gl_LocalInvocationIndex == 0) {                               \n"
         "    uint wg_avg = %s / (gl_WorkGroupSize.x * gl_WorkGroupSize.y); \n"
         "    atomicMax(%s[%s], wg_avg);                                    \n"
         "    atomicAdd(%s[%s], wg_avg);                                    \n"
         "}                                                                 \n",
         wg_sum,
         max.name, idx.name,
         sum.name, idx.name);

    // Update the sig_peak/sig_avg from the old SSBO state
    GLSL("uint num_wg = gl_NumWorkGroups.x * gl_NumWorkGroups.y; \n"
         "if (%s > 0) {                                          \n"
         "    sig_peak = float(%s) / (%f * float(%s));           \n"
         "    sig_avg  = float(%s) / (%f * float(%s * num_wg));  \n"
         "}                                                      \n",
         num.name,
         max_total.name, PL_COLOR_REF_WHITE, num.name,
         sum_total.name, PL_COLOR_REF_WHITE, num.name);

    // Finally, to update the global state, we increment a counter per dispatch
    GLSL("memoryBarrierBuffer();                                                \n"
         "barrier();                                                            \n"
         "if (gl_LocalInvocationIndex == 0 && atomicAdd(%s, 1) == num_wg - 1) { \n"
         "    %s = 0;                                                           \n"
         // Add the current frame, then subtract and reset the next frame
         "    uint next = (%s + 1) %% %d;                                       \n"
         "    %s += %s[%s] - %s[next];                                          \n"
         "    %s += %s[%s] - %s[next];                                          \n"
         "    %s[next] = %s[next] = 0;                                          \n"
         // Update the index and count
         "    %s = next;                                                        \n"
         "    %s = min(%s + 1, %d);                                             \n"
         "    memoryBarrierBuffer();                                            \n"
         "}                                                                     \n",
         ctr.name, ctr.name,
         idx.name, frames + 1,
         max_total.name, max.name, idx.name, max.name,
         sum_total.name, sum.name, idx.name, sum.name,
         max.name, sum.name, idx.name,
         num.name, num.name, frames + 1);
}

// Average light level for SDR signals. This is equal to a signal level of 0.5
// under a typical presentation gamma of about 2.0.
static const float sdr_avg = 0.25;

static void pl_shader_tone_map(struct pl_shader *sh, struct pl_color_space src,
                               struct pl_color_space dst, ident_t luma,
                               struct pl_shader_obj **peak_detect_state,
                               const struct pl_color_map_params *params)
{
    // no-op if no tone mapping necessary
    if (src.sig_peak <= dst.sig_peak)
        return;

    GLSL("// pl_shader_tone_map\n");

    // To prevent discoloration due to out-of-bounds clipping, we need to make
    // sure to reduce the value range as far as necessary to keep the entire
    // signal in range, so tone map based on the brightest component.
    GLSL("float sig = max(max(color.r, color.g), color.b); \n"
         "float sig_peak = %f;                             \n"
         "float sig_avg = %f;                              \n",
         src.sig_peak, src.sig_avg);

    // HDR peak detection is done before scaling based on the dst.sig_peak/avg
    // in order to make the detected values stable / averageable.
    hdr_update_peak(sh, peak_detect_state, params);

    // Rescale the variables in order to bring it into a representation where
    // 1.0 represents the dst_peak. This is because all of the tone mapping
    // algorithms are defined in such a way that they map to the range [0.0, 1.0].
    if (dst.sig_peak > 1.0) {
        GLSL("sig *= 1.0/%f;      \n"
             "sig_peak *= 1.0/%f; \n",
             dst.sig_peak, dst.sig_peak);
    }

    // Desaturate the color using a coefficient dependent on the signal level
    if (params->tone_mapping_desaturate > 0) {
        GLSL("float luma = dot(%s, color.rgb);                      \n"
             "float coeff = max(sig - 0.18, 1e-6) / max(sig, 1e-6); \n"
             "coeff = pow(coeff, %f);                               \n"
             "color.rgb = mix(color.rgb, vec3(luma), coeff);        \n"
             "sig = mix(sig, luma, coeff);                          \n",
             luma, 10.0 / params->tone_mapping_desaturate);
    }

    // Store the original signal level for later re-use
    GLSL("float sig_orig = sig;\n");

    // Scale the signal to compensate for differences in the average brightness
    GLSL("float slope = min(1.0, %f / sig_avg); \n"
         "sig *= slope;                         \n"
         "sig_peak *= slope;                    \n",
         dst.sig_avg);

    float param = params->tone_mapping_param;
    switch (params->tone_mapping_algo) {
    case PL_TONE_MAPPING_CLIP:
        GLSL("sig *= %f;\n", PL_DEF(param, 1.0));
        break;

    case PL_TONE_MAPPING_MOBIUS:
        GLSL("const float j = %f;                                           \n"
             // solve for M(j) = j; M(sig_peak) = 1.0; M'(j) = 1.0
             // where M(x) = scale * (x+a)/(x+b)
             "float a = -j*j * (sig_peak - 1.0) / (j*j - 2.0*j + sig_peak); \n"
             "float b = (j*j - 2.0*j*sig_peak + sig_peak) /                 \n"
             "          max(1e-6, sig_peak - 1.0);                          \n"
             "float scale = (b*b + 2.0*b*j + j*j) / (b-a);                  \n"
             "sig = sig > j ? (scale * (sig + a) / (sig + b)) : sig;        \n",
             PL_DEF(param, 0.3));
        break;

    case PL_TONE_MAPPING_REINHARD: {
        float contrast = PL_DEF(param, 0.5),
              offset = (1.0 - contrast) / contrast;
        GLSL("sig = sig / (sig + %f);                   \n"
             "float scale = (sig_peak + %f) / sig_peak; \n"
             "sig *= scale;                             \n",
             offset, offset);
        break;
    }

    case PL_TONE_MAPPING_HABLE: {
        float A = 0.15, B = 0.50, C = 0.10, D = 0.20, E = 0.02, F = 0.30;
        ident_t hable = sh_fresh(sh, "hable");
        GLSLH("float %s(float x) {                                        \n"
              "return ((x * (%f*x + %f)+%f)/(x * (%f*x + %f) + %f)) - %f; \n"
              "}                                                          \n",
              hable, A, C*B, D*E, A, B, D*F, E/F);
        GLSL("sig = %s(sig) / %s(sig_peak);\n", hable, hable);
        break;
    }

    case PL_TONE_MAPPING_GAMMA:
        GLSL("const float cutoff = 0.05, gamma = 1.0/%f;                     \n"
             "float scale = pow(cutoff / sig_peak, gamma) / cutoff;          \n"
             "sig = sig > cutoff ? pow(sig / sig_peak, gamma) : scale * sig; \n",
             PL_DEF(param, 1.8));
        break;

    case PL_TONE_MAPPING_LINEAR:
        GLSL("sig *= %f / sig_peak;\n", PL_DEF(param, 1.0));
        break;

    default:
        abort();
    }

    // Clip the final signal to the output range and apply the difference
    // linearly to the RGB channels. (this prevents discoloration)
    GLSL("sig = min(sig, 1.0);        \n"
        "color.rgb *= sig / sig_orig; \n");
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

    // If the source light type is unknown, infer it from the transfer function.
    if (!src.light) {
        src.light = (src.transfer == PL_COLOR_TRC_HLG)
            ? PL_COLOR_LIGHT_SCENE_HLG
            : PL_COLOR_LIGHT_DISPLAY;
    }

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
            dst.transfer = PL_COLOR_TRC_GAMMA22;
    }

    // 99 times out of 100, this is what we want
    dst.light = PL_DEF(dst.light, PL_COLOR_LIGHT_DISPLAY);

    // Compute the highest encodable level
    float src_range = pl_color_transfer_nominal_peak(src.transfer),
          dst_range = pl_color_transfer_nominal_peak(dst.transfer);

    // Default the src/dst peak information based on the encodable range. For
    // the source peak, this is the safest possible value (no clipping). For
    // the dest peak, this makes full use of the available dynamic range.
    src.sig_peak = PL_DEF(src.sig_peak, src_range);
    dst.sig_peak = PL_DEF(dst.sig_peak, dst_range);

    // Defaults the signal average based on the SDR signal average.
    // Note: For HDR, this assumes well-mastered HDR content.
    src.sig_avg = PL_DEF(src.sig_avg, sdr_avg);

    // Defaults the dest average based on the source average, unless the source
    // is HDR and the destination is not, in which case fall back to SDR avg.
    if (!dst.sig_avg) {
        bool src_hdr = pl_color_transfer_is_hdr(src.transfer);
        bool dst_hdr = pl_color_transfer_is_hdr(dst.transfer);
        dst.sig_avg = src_hdr && !dst_hdr ? sdr_avg : src.sig_avg;
    }

    // All operations from here on require linear light as a starting point,
    // so we linearize even if src.gamma == dst.gamma when one of the other
    // operations needs it
    bool need_linear = src.transfer != dst.transfer ||
                       src.primaries != dst.primaries ||
                       src_range != dst_range ||
                       src.sig_peak > dst.sig_peak ||
                       src.sig_avg != dst.sig_avg ||
                       src.light != dst.light;

    // Various operations need access to the src_luma and dst_luma respectively,
    // so just always make them available if we're doing anything at all
    ident_t src_luma = NULL, dst_luma = NULL;
    if (need_linear) {
        struct pl_matrix3x3 rgb2xyz;
        rgb2xyz = pl_get_rgb2xyz_matrix(pl_raw_primaries_get(src.primaries));
        src_luma = sh_var(sh, (struct pl_shader_var) {
            .var  = ra_var_vec3("src_luma"),
            .data = rgb2xyz.m[1], // RGB->Y vector
        });
        rgb2xyz = pl_get_rgb2xyz_matrix(pl_raw_primaries_get(dst.primaries));
        dst_luma = sh_var(sh, (struct pl_shader_var) {
            .var  = ra_var_vec3("dst_luma"),
            .data = rgb2xyz.m[1], // RGB->Y vector
        });
    }

    bool is_linear = prelinearized;
    if (need_linear && !is_linear) {
        pl_shader_linearize(sh, src.transfer);
        is_linear = true;
    }

    if (src.light != dst.light)
        pl_shader_ootf(sh, src.light, src_luma);

    // Adapt to the right colorspace (primaries) if necessary
    if (src.primaries != dst.primaries) {
        const struct pl_raw_primaries *csp_src, *csp_dst;
        csp_src = pl_raw_primaries_get(src.primaries),
        csp_dst = pl_raw_primaries_get(dst.primaries);
        struct pl_matrix3x3 cms_mat;
        cms_mat = pl_get_color_mapping_matrix(csp_src, csp_dst, params->intent);
        GLSL("color.rgb = %s * color.rgb;\n", sh_var(sh, (struct pl_shader_var) {
            .var = ra_var_mat3("cms_matrix"),
            .data = PL_TRANSPOSE_3X3(cms_mat.m),
        }));
        // Since this can reduce the gamut, figure out by how much
        for (int c = 0; c < 3; c++)
            src.sig_peak = fmaxf(src.sig_peak, cms_mat.m[c][c]);
    }

    // Tone map to rescale the signal average/peak.
    pl_shader_tone_map(sh, src, dst, dst_luma, peak_detect_state, params);

    // Warn for remaining out-of-gamut colors is enabled
    if (params->gamut_warning) {
        GLSL("if (any(greaterThan(color.rgb, vec3(1.01))) ||\n"
             "    any(lessThan(color.rgb, vec3(-0.01))))\n"
             "    color.rgb = vec3(1.0) - color.rgb; // invert\n");
    }

    if (src.light != dst.light)
        pl_shader_inverse_ootf(sh, dst.light, dst_luma);

    if (is_linear)
        pl_shader_delinearize(sh, dst.transfer);

    GLSL("}\n");
}

struct sh_dither_obj {
    enum pl_dither_method method;
    struct pl_shader_obj *lut;
};

static void sh_dither_uninit(const struct ra *ra, void *ptr)
{
    struct sh_dither_obj *obj = ptr;
    pl_shader_obj_destroy(&obj->lut);
    *obj = (struct sh_dither_obj) {0};
}

static void fill_dither_matrix(void *priv, float *data, int w, int h, int d)
{
    pl_assert(w > 0 && h > 0 && d == 0);

    const struct sh_dither_obj *obj = priv;
    switch (obj->method) {
    case PL_DITHER_ORDERED_LUT:
        pl_assert(w == h);
        pl_generate_bayer_matrix(data, w);
        break;

    case PL_DITHER_BLUE_NOISE:
        pl_generate_blue_noise(data, w);
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
        PL_ERR(sh, "Invalid `lut_size` specified: %d", params->lut_size);
        return;
    }

    enum pl_dither_method method = params->method;
    ident_t lut = NULL;
    int lut_size = 0;

    if (dither_method_is_lut(method)) {
        if (!dither_state) {
            PL_TRACE(sh, "LUT-based dither method specified but no dither state "
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
        lut = sh_lut(sh, &obj->lut, SH_LUT_AUTO, lut_size, lut_size, 0,
                     changed, obj, fill_dither_matrix);
        if (!lut)
            goto fallback;
    }

    goto done;

fallback:
    if (sh->ra && sh->ra->glsl.version >= 130) {
        method = PL_DITHER_ORDERED_FIXED;
    } else {
        method = PL_DITHER_WHITE_NOISE;
    }

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
            int phase = sh->index % 8;
            float r = phase * (M_PI / 2); // rotate
            float m = phase < 4 ? 1 : -1; // mirror
            float mat[2][2] = {
                {cos(r),     -sin(r)    },
                {sin(r) * m,  cos(r) * m},
            };

            ident_t rot = sh_var(sh, (struct pl_shader_var) {
                .var  = ra_var_mat2("dither_rot"),
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
        GLSL("bias = %s(pos);\n", lut);
        break;
    }

    unsigned long long scale = (1LLU << new_depth) - 1;
    GLSL("color = vec4(%llu.0) * color + vec4(bias); \n"
         "color = floor(color) * vec4(1.0/%llu.0);   \n"
         "}                                          \n",
         scale, scale);
}

const struct pl_dither_params pl_dither_default_params = {
    .method     = PL_DITHER_BLUE_NOISE,
    .temporal   = false, // commonly flickers on LCDs
};
