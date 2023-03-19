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

#include <libplacebo/shaders/dithering.h>

const struct pl_dither_params pl_dither_default_params = { PL_DITHER_DEFAULTS };

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

static inline float approx_gamma(enum pl_color_transfer trc)
{
    switch (trc) {
    case PL_COLOR_TRC_UNKNOWN:  return 1.0f;
    case PL_COLOR_TRC_LINEAR:   return 1.0f;
    case PL_COLOR_TRC_PRO_PHOTO:return 1.8f;
    case PL_COLOR_TRC_GAMMA18:  return 1.8f;
    case PL_COLOR_TRC_GAMMA20:  return 2.0f;
    case PL_COLOR_TRC_GAMMA24:  return 2.4f;
    case PL_COLOR_TRC_GAMMA26:  return 2.6f;
    case PL_COLOR_TRC_ST428:    return 2.6f;
    case PL_COLOR_TRC_GAMMA28:  return 2.8f;

    case PL_COLOR_TRC_SRGB:
    case PL_COLOR_TRC_BT_1886:
    case PL_COLOR_TRC_GAMMA22:
        return 2.2f;

    case PL_COLOR_TRC_PQ:
    case PL_COLOR_TRC_HLG:
    case PL_COLOR_TRC_V_LOG:
    case PL_COLOR_TRC_S_LOG1:
    case PL_COLOR_TRC_S_LOG2:
        return 2.0f; // TODO: handle this better

    case PL_COLOR_TRC_COUNT: break;
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

    sh_describef(sh, "dithering (%d bits)", new_depth);
    GLSL("// pl_shader_dither \n"
        "{                    \n"
        "float bias;          \n");

    params = PL_DEF(params, &pl_dither_default_params);
    if (params->lut_size < 0 || params->lut_size > 8) {
        SH_FAIL(sh, "Invalid `lut_size` specified: %d", params->lut_size);
        return;
    }

    enum pl_dither_method method = params->method;
    ident_t lut = NULL_IDENT;
    int lut_size = 0;

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
            .object     = &obj->lut,
            .var_type   = PL_VAR_FLOAT,
            .width      = lut_size,
            .height     = lut_size,
            .comps      = 1,
            .update     = changed,
            .fill       = fill_dither_matrix,
            .priv       = obj,
        ));
        if (!lut)
            goto fallback;
    }

    goto done;

fallback:
    method = PL_DITHER_ORDERED_FIXED;
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
        GLSL("vec2 pos = fract(gl_FragCoord.xy * 1.0/"$"); \n", SH_FLOAT(size));

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
            GLSL("pos = fract("$" * pos + vec2(1.0));\n", rot);
        }
    }

    switch (method) {
    case PL_DITHER_WHITE_NOISE: {
        ident_t prng = sh_prng(sh, params->temporal, NULL);
        GLSL("bias = "$".x;\n", prng);
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
        GLSL("bias = "$"(ivec2(pos * "$"));\n", lut, SH_FLOAT(lut_size));
        break;

    case PL_DITHER_METHOD_COUNT:
        pl_unreachable();
    }

    // Scale factor for dither rounding
    GLSL("const float scale = %llu.0; \n", (1LLU << new_depth) - 1);

    const float gamma = approx_gamma(params->transfer);
    if (gamma != 1.0f && new_depth <= 4) {
        GLSL("const float gamma = "$";                  \n"
             "vec4 color_lin = pow(color, vec4(gamma)); \n",
             SH_FLOAT(gamma));

        if (new_depth == 1) {
            // Special case for bit depth 1 dithering, in this case we can just
            // ignore the low/high rounding because we know we are always
            // dithering between 0.0 and 1.0.
            GLSL("const vec4 low = vec4(0.0);           \n"
                 "const vec4 high = vec4(1.0);          \n"
                 "vec4 offset = color_lin;              \n");
        } else {
            // Linearize the low, high and current color values
            GLSL("vec4 low = floor(color * scale) / scale;  \n"
                 "vec4 high = ceil(color * scale) / scale;  \n"
                 "vec4 low_lin = pow(low, vec4(gamma));     \n"
                 "vec4 high_lin = pow(high, vec4(gamma));   \n"
                 "vec4 range = high_lin - low_lin;          \n"
                 "vec4 offset = (color_lin - low_lin) /     \n"
                 "              max(range, 1e-6);           \n");
        }

        // Mix in the correct ratio corresponding to the offset and bias
        GLSL("color = mix(low, high, greaterThan(offset, vec4(bias))); \n");
    } else {
        // Approximate each gamma segment as a straight line, this simplifies
        // the process of dithering down to a single scale and (biased) round.
        GLSL("color = scale * color + vec4(bias);   \n"
             "color = floor(color) * (1.0 / scale); \n");
    }

    GLSL("} \n");
}

/* Error diffusion code is taken from mpv, original copyright (c) 2019 Bin Jin
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

// After a (y, x) -> (y, x + y * shift) mapping, find the right most column that
// will be affected by the current column.
static int compute_rightmost_shifted_column(const struct pl_error_diffusion_kernel *k)
{
    int ret = 0;
    for (int y = 0; y <= PL_EDF_MAX_DY; y++) {
        for (int x = PL_EDF_MIN_DX; x <= PL_EDF_MAX_DX; x++) {
            if (k->pattern[y][x - PL_EDF_MIN_DX] != 0) {
                int shifted_x = x + y * k->shift;

                // The shift mapping guarantees current column (or left of it)
                // won't be affected by error diffusion.
                assert(shifted_x > 0);

                ret = PL_MAX(ret, shifted_x);
            }
        }
    }
    return ret;
}

size_t pl_error_diffusion_shmem_req(const struct pl_error_diffusion_kernel *kernel,
                                    int height)
{
    // We add PL_EDF_MAX_DY empty lines on the bottom to handle errors
    // propagated out from bottom side.
    int rows = height + PL_EDF_MAX_DY;
    int shifted_columns = compute_rightmost_shifted_column(kernel) + 1;

    // The shared memory is an array of size rows*shifted_columns. Each element
    // is a single uint for three RGB component.
    return rows * shifted_columns * sizeof(uint32_t);
}

bool pl_shader_error_diffusion(pl_shader sh, const struct pl_error_diffusion_params *params)
{
    const int width = params->input_tex->params.w, height = params->input_tex->params.h;
    const struct pl_glsl_version glsl = sh_glsl(sh);
    const struct pl_error_diffusion_kernel *kernel =
        PL_DEF(params->kernel, &pl_error_diffusion_sierra_lite);

    pl_assert(params->output_tex->params.w == width);
    pl_assert(params->output_tex->params.h == height);
    if (!sh_require(sh, PL_SHADER_SIG_NONE, width, height))
        return false;

    if (params->new_depth <= 0 || params->new_depth > 256) {
        PL_WARN(sh, "Invalid dither depth: %d.. ignoring", params->new_depth);
        return false;
    }

    // The parallel error diffusion works by applying the shift mapping first.
    // Taking the Floyd and Steinberg algorithm for example. After applying
    // the (y, x) -> (y, x + y * shift) mapping (with shift=2), all errors are
    // propagated into the next few columns, which makes parallel processing on
    // the same column possible.
    //
    //           X    7/16                X    7/16
    //    3/16  5/16  1/16   ==>    0     0    3/16  5/16  1/16

    // Figuring out the size of rectangle containing all shifted pixels.
    // The rectangle height is not changed.
    int shifted_width = width + (height - 1) * kernel->shift;

    // We process all pixels from the shifted rectangles column by column, with
    // a single global work group of size |block_size|.
    // Figuring out how many block are required to process all pixels. We need
    // this explicitly to make the number of barrier() calls match.
    int block_size = PL_MIN(glsl.max_group_threads, height);
    int blocks = PL_DIV_UP(height * shifted_width, block_size);

    // If we figure out how many of the next columns will be affected while the
    // current columns is being processed. We can store errors of only a few
    // columns in the shared memory. Using a ring buffer will further save the
    // cost while iterating to next column.
    //
    int ring_buffer_rows = height + PL_EDF_MAX_DY;
    int ring_buffer_columns = compute_rightmost_shifted_column(kernel) + 1;
    ident_t ring_buffer_size = sh_const(sh, (struct pl_shader_const) {
        .type = PL_VAR_UINT,
        .name = "ring_buffer_size",
        .data = &(unsigned) { ring_buffer_rows * ring_buffer_columns },
        .compile_time = true,
    });

    // Compute shared memory requirements and try enabling compute shader.
    size_t shmem_req = ring_buffer_rows * ring_buffer_columns * sizeof(uint32_t);
    if (!sh_try_compute(sh, block_size, 1, false, shmem_req)) {
        PL_ERR(sh, "Cannot execute error diffusion kernel: too old GPU or "
               "insufficient compute shader memory!");
        return false;
    }

    ident_t in_tex = sh_desc(sh, (struct pl_shader_desc) {
        .binding.object = params->input_tex,
        .desc = {
            .name   = "input_tex",
            .type   = PL_DESC_SAMPLED_TEX,
        },
    });

    ident_t out_img = sh_desc(sh, (struct pl_shader_desc) {
        .binding.object = params->output_tex,
        .desc = {
            .name    = "output_tex",
            .type    = PL_DESC_STORAGE_IMG,
            .access  = PL_DESC_ACCESS_WRITEONLY,
        },
    });

    sh->output = PL_SHADER_SIG_NONE;
    sh_describef(sh, "error diffusion (%s, %d bits)",
                 kernel->name, params->new_depth);

    // Defines the ring buffer in shared memory.
    GLSLH("shared uint err_rgb8["$"]; \n", ring_buffer_size);
    GLSL("// pl_shader_error_diffusion                                          \n"
         // Safeguard against accidental over-execution
         "if (gl_WorkGroupID != uvec3(0))                                       \n"
         "    return;                                                           \n"
         // Initialize the ring buffer.
         "for (uint i = gl_LocalInvocationIndex; i < "$"; i+=gl_WorkGroupSize.x)\n"
         "    err_rgb8[i] = 0u;                                                 \n"

        // Main block loop, add barrier here to have previous block all
        // processed before starting the processing of the next.
         "for (uint block_id = 0; block_id < "$"; block_id++) {                 \n"
         "barrier();                                                            \n"
        // Compute the coordinate of the pixel we are currently processing,
        // both before and after the shift mapping.
         "uint id = block_id * gl_WorkGroupSize.x + gl_LocalInvocationIndex;    \n"
         "const uint height = "$";                                              \n"
         "int y = int(id %% height), x_shifted = int(id / height);              \n"
         "int x = x_shifted - y * %d;                                           \n"
         // Proceed only if we are processing a valid pixel.
         "if (x >= 0 && x < "$") {                                              \n"
         // The index that the current pixel have on the ring buffer.
         "uint idx = uint(x_shifted * "$" + y) %% "$";                          \n"
         // Fetch the current pixel.
         "vec4 pix_orig = texelFetch("$", ivec2(x, y), 0);                      \n"
         "vec3 pix = pix_orig.rgb;                                              \n",
         ring_buffer_size,
         SH_UINT(blocks),
         SH_UINT(height),
         kernel->shift,
         SH_INT(width),
         SH_INT(ring_buffer_rows),
         ring_buffer_size,
         in_tex);

    // The dithering will quantize pixel value into multiples of 1/dither_quant.
    int dither_quant = (1 << params->new_depth) - 1;

    // We encode errors in RGB components into a single 32-bit unsigned integer.
    // The error we propagate from the current pixel is in range of
    // [-0.5 / dither_quant, 0.5 / dither_quant]. While not quite obvious, the
    // sum of all errors been propagated into a pixel is also in the same range.
    // It's possible to map errors in this range into [-127, 127], and use an
    // unsigned 8-bit integer to store it (using standard two's complement).
    // The three 8-bit unsigned integers can then be encoded into a single
    // 32-bit unsigned integer, with two 4-bit padding to prevent addition
    // operation overflows affecting other component. There are at most 12
    // addition operations on each pixel, so 4-bit padding should be enough.
    // The overflow from R component will be discarded.
    //
    // The following figure is how the encoding looks like.
    //
    //     +------------------------------------+
    //     |RRRRRRRR|0000|GGGGGGGG|0000|BBBBBBBB|
    //     +------------------------------------+
    //

    // The bitshift position for R and G component.
    const int bitshift_r = 24, bitshift_g = 12;
    // The multiplier we use to map [-0.5, 0.5] to [-127, 127].
    const int uint8_mul = 127 * 2;

    GLSL(// Add the error previously propagated into current pixel, and clear
         // it in the ring buffer.
         "uint err_u32 = err_rgb8[idx] + %uu;                                   \n"
         "pix = pix * %d.0 + vec3(int((err_u32 >> %d) & 0xFFu) - 128,           \n"
         "                        int((err_u32 >> %d) & 0xFFu) - 128,           \n"
         "                        int( err_u32        & 0xFFu) - 128) / %d.0;   \n"
         "err_rgb8[idx] = 0u;                                                   \n"
         // Write the dithered pixel.
         "vec3 dithered = round(pix);                                           \n"
         "imageStore("$", ivec2(x, y), vec4(dithered / %d.0, pix_orig.a));      \n"
         // Prepare for error propagation pass
         "vec3 err_divided = (pix - dithered) * %d.0 / %d.0;                    \n"
         "ivec3 tmp;                                                            \n",
         (128u << bitshift_r) | (128u << bitshift_g) | 128u,
         dither_quant, bitshift_r, bitshift_g, uint8_mul,
         out_img, dither_quant,
         uint8_mul, kernel->divisor);

    // Group error propagation with same weight factor together, in order to
    // reduce the number of annoying error encoding.
    for (int dividend = 1; dividend <= kernel->divisor; dividend++) {
        bool err_assigned = false;

        for (int y = 0; y <= PL_EDF_MAX_DY; y++) {
            for (int x = PL_EDF_MIN_DX; x <= PL_EDF_MAX_DX; x++) {
                if (kernel->pattern[y][x - PL_EDF_MIN_DX] != dividend)
                    continue;

                if (!err_assigned) {
                    err_assigned = true;

                    GLSL("tmp = ivec3(round(err_divided * %d.0));   \n"
                         "err_u32 = (uint(tmp.r & 0xFF) << %d) |    \n"
                         "          (uint(tmp.g & 0xFF) << %d) |    \n"
                         "           uint(tmp.b & 0xFF);            \n",
                         dividend,
                         bitshift_r, bitshift_g);
                }

                int shifted_x = x + y * kernel->shift;

                // Unlike the right border, errors propagated out from left
                // border will remain in the ring buffer. This will produce
                // visible artifacts near the left border, especially for
                // shift=3 kernels.
                if (x < 0)
                    GLSL("if (x >= %d) \n", -x);

                // Calculate the new position in the ring buffer to propagate
                // the error into.
                int ring_buffer_delta = shifted_x * ring_buffer_rows + y;
                GLSL("atomicAdd(err_rgb8[(idx + %du) %% "$"], err_u32); \n",
                     ring_buffer_delta, ring_buffer_size);
            }
        }
    }

    GLSL("}} \n"); // end of main loop + valid pixel conditional
    return true;
}
