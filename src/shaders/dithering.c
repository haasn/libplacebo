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
