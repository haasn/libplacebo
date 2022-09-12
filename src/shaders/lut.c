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
#include <ctype.h>

#include "shaders.h"

static inline bool isnumeric(char c)
{
    return (c >= '0' && c <= '9') || c == '-';
}

void pl_lut_free(struct pl_custom_lut **lut)
{
    pl_free_ptr(lut);
}

struct pl_custom_lut *pl_lut_parse_cube(pl_log log, const char *cstr, size_t cstr_len)
{
    struct pl_custom_lut *lut = pl_zalloc_ptr(NULL, lut);
    pl_str str = (pl_str) { (uint8_t *) cstr, cstr_len };
    lut->signature = pl_str_hash(str);
    int entries = 0;

    float min[3] = { 0.0, 0.0, 0.0 };
    float max[3] = { 1.0, 1.0, 1.0 };

    // Parse header
    while (str.len && !isnumeric(str.buf[0])) {
        pl_str line = pl_str_strip(pl_str_getline(str, &str));
        if (!line.len)
            continue; // skip empty line

        if (pl_str_eatstart0(&line, "TITLE")) {
            pl_info(log, "Loading LUT: %.*s", PL_STR_FMT(pl_str_strip(line)));
            continue;
        }

        if (pl_str_eatstart0(&line, "LUT_3D_SIZE")) {
            line = pl_str_strip(line);
            int size;
            if (!pl_str_parse_int(line, &size)) {
                pl_err(log, "Failed parsing dimension '%.*s'", PL_STR_FMT(line));
                goto error;
            }
            if (size <= 0 || size > 1024) {
                pl_err(log, "Invalid 3DLUT size: %dx%d%x", size, size, size);
                goto error;
            }

            lut->size[0] = lut->size[1] = lut->size[2] = size;
            entries = size * size * size;
            continue;
        }

        if (pl_str_eatstart0(&line, "LUT_1D_SIZE")) {
            line = pl_str_strip(line);
            int size;
            if (!pl_str_parse_int(line, &size)) {
                pl_err(log, "Failed parsing dimension '%.*s'", PL_STR_FMT(line));
                goto error;
            }
            if (size <= 0 || size > 65536) {
                pl_err(log, "Invalid 1DLUT size: %d", size);
                goto error;
            }

            lut->size[0] = size;
            lut->size[1] = lut->size[2] = 0;
            entries = size;
            continue;
        }

        if (pl_str_eatstart0(&line, "DOMAIN_MIN")) {
            line = pl_str_strip(line);
            if (!pl_str_parse_float(pl_str_split_char(line, ' ', &line), &min[0]) ||
                !pl_str_parse_float(pl_str_split_char(line, ' ', &line), &min[1]) ||
                !pl_str_parse_float(line, &min[2]))
            {
                pl_err(log, "Failed parsing domain: '%.*s'", PL_STR_FMT(line));
                goto error;
            }
            continue;
        }

        if (pl_str_eatstart0(&line, "DOMAIN_MAX")) {
            line = pl_str_strip(line);
            if (!pl_str_parse_float(pl_str_split_char(line, ' ', &line), &max[0]) ||
                !pl_str_parse_float(pl_str_split_char(line, ' ', &line), &max[1]) ||
                !pl_str_parse_float(line, &max[2]))
            {
                pl_err(log, "Failed parsing domain: '%.*s'", PL_STR_FMT(line));
                goto error;
            }
            continue;
        }

        if (pl_str_eatstart0(&line, "#")) {
            pl_debug(log, "Unhandled .cube comment: %.*s",
                     PL_STR_FMT(pl_str_strip(line)));
            continue;
        }

        pl_warn(log, "Unhandled .cube line: %.*s", PL_STR_FMT(pl_str_strip(line)));
    }

    if (!entries) {
        pl_err(log, "Missing LUT size specification?");
        goto error;
    }

    for (int i = 0; i < 3; i++) {
        if (max[i] - min[i] < 1e-6) {
            pl_err(log, "Invalid domain range: [%f, %f]", min[i], max[i]);
            goto error;
        }
    }

    float *data = pl_alloc(lut, sizeof(float[3]) * entries);
    lut->data = data;

    // Parse LUT body
    clock_t start = clock();
    for (int n = 0; n < entries; n++) {
        for (int c = 0; c < 3; c++) {
            static const char * const digits = "0123456789.-+e";

            // Extract valid digit sequence
            size_t len = pl_strspn(str, digits);
            pl_str entry = (pl_str) { str.buf, len };
            str.buf += len;
            str.len -= len;

            if (!entry.len) {
                if (!str.len) {
                    pl_err(log, "Failed parsing LUT: Unexpected EOF, expected "
                           "%d entries, got %d", entries * 3, n * 3 + c + 1);
                } else {
                    pl_err(log, "Failed parsing LUT: Unexpected '%c', expected "
                           "digit", str.buf[0]);
                }
                goto error;
            }

            float num;
            if (!pl_str_parse_float(entry, &num)) {
                pl_err(log, "Failed parsing float value '%.*s'", PL_STR_FMT(entry));
                goto error;
            }

            // Rescale to range 0.0 - 1.0
            *data++ = (num - min[c]) / (max[c] - min[c]);

            // Skip whitespace between digits
            str = pl_str_strip(str);
        }
    }

    str = pl_str_strip(str);
    if (str.len)
        pl_warn(log, "Extra data after LUT?... ignoring '%c'", str.buf[0]);

    pl_log_cpu_time(log, start, clock(), "parsing .cube LUT");
    return lut;

error:
    pl_free(lut);
    return NULL;
}

static void fill_lut(void *datap, const struct sh_lut_params *params)
{
    const struct pl_custom_lut *lut = params->priv;

    int dim_r = params->width;
    int dim_g = PL_DEF(params->height, 1);
    int dim_b = PL_DEF(params->depth, 1);

    float *data = datap;
    for (int b = 0; b < dim_b; b++) {
        for (int g = 0; g < dim_g; g++) {
            for (int r = 0; r < dim_r; r++) {
                size_t offset = (b * dim_g + g) * dim_r + r;
                const float *src = &lut->data[offset * 3];
                float *dst = &data[offset * 4];
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = 0.0f;
            }
        }
    }
}

void pl_shader_custom_lut(pl_shader sh, const struct pl_custom_lut *lut,
                          pl_shader_obj *lut_state)
{
    if (!lut)
        return;

    int dims;
    if (lut->size[0] > 0 && lut->size[1] > 0 && lut->size[2] > 0) {
        dims = 3;
    } else if (lut->size[0] > 0 && !lut->size[1] && !lut->size[2]) {
        dims = 1;
    } else {
        SH_FAIL(sh, "Invalid dimensions %dx%dx%d for pl_custom_lut, must be 1D "
                "or 3D!", lut->size[0], lut->size[1], lut->size[2]);
        return;
    }

    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return;

    ident_t fun = sh_lut(sh, sh_lut_params(
        .object     = lut_state,
        .var_type   = PL_VAR_FLOAT,
        .method     = SH_LUT_TETRAHEDRAL,
        .width      = lut->size[0],
        .height     = lut->size[1],
        .depth      = lut->size[2],
        .comps      = 4, // for better texel alignment
        .signature  = lut->signature,
        .fill       = fill_lut,
        .priv       = (void *) lut,
    ));

    if (!fun) {
        SH_FAIL(sh, "pl_shader_custom_lut: failed generating LUT object");
        return;
    }

    GLSL("// pl_shader_custom_lut \n");

    static const struct pl_matrix3x3 zero = {0};
    if (memcmp(&lut->shaper_in, &zero, sizeof(zero)) != 0) {
        GLSL("color.rgb = %s * color.rgb; \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_mat3("shaper_in"),
            .data = PL_TRANSPOSE_3X3(lut->shaper_in.m),
        }));
    }

    switch (dims) {
    case 1:
        sh_describe(sh, "custom 1DLUT");
        GLSL("color.rgb = vec3(%s(color.r).r, %s(color.g).g, %s(color.b).b); \n",
             fun, fun, fun);
        break;
    case 3:
        sh_describe(sh, "custom 3DLUT");
        GLSL("color.rgb = %s(color.rgb).rgb; \n", fun);
        break;
    }

    if (memcmp(&lut->shaper_out, &zero, sizeof(zero)) != 0) {
        GLSL("color.rgb = %s * color.rgb; \n", sh_var(sh, (struct pl_shader_var) {
            .var = pl_var_mat3("shaper_out"),
            .data = PL_TRANSPOSE_3X3(lut->shaper_out.m),
        }));
    }
}
