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

const struct pl_deband_params pl_deband_default_params = { PL_DEBAND_DEFAULTS };

static inline struct pl_tex_params src_params(const struct pl_sample_src *src)
{
    if (src->tex)
        return src->tex->params;

    return (struct pl_tex_params) {
        .w = src->tex_w,
        .h = src->tex_h,
    };
}

enum filter {
    NEAREST = PL_TEX_SAMPLE_NEAREST,
    LINEAR  = PL_TEX_SAMPLE_LINEAR,
    BEST,
    FASTEST,
};

// Helper function to compute the src/dst sizes and upscaling ratios
static bool setup_src(pl_shader sh, const struct pl_sample_src *src,
                      ident_t *src_tex, ident_t *pos, ident_t *size, ident_t *pt,
                      float *ratio_x, float *ratio_y, uint8_t *comp_mask,
                      float *scale, bool resizeable, const char **fn,
                      enum filter filter)
{
    enum pl_shader_sig sig;
    float src_w, src_h;
    enum pl_tex_sample_mode sample_mode;
    if (src->tex) {
        pl_fmt fmt = src->tex->params.format;
        bool can_linear = fmt->caps & PL_FMT_CAP_LINEAR;
        pl_assert(pl_tex_params_dimension(src->tex->params) == 2);
        sig = PL_SHADER_SIG_NONE;
        src_w = pl_rect_w(src->rect);
        src_h = pl_rect_h(src->rect);
        switch (filter) {
        case FASTEST:
        case NEAREST:
            sample_mode = PL_TEX_SAMPLE_NEAREST;
            break;
        case LINEAR:
            if (!can_linear) {
                SH_FAIL(sh, "Trying to use a shader that requires linear "
                        "sampling with a texture whose format (%s) does not "
                        "support PL_FMT_CAP_LINEAR", fmt->name);
                return false;
            }
            sample_mode = PL_TEX_SAMPLE_LINEAR;
            break;
        case BEST:
            sample_mode = can_linear ? PL_TEX_SAMPLE_LINEAR : PL_TEX_SAMPLE_NEAREST;
            break;
        }
    } else {
        pl_assert(src->tex_w && src->tex_h);
        sig = PL_SHADER_SIG_SAMPLER;
        src_w = src->sampled_w;
        src_h = src->sampled_h;
        if (filter == BEST || filter == FASTEST) {
            sample_mode = src->mode;
        } else {
            sample_mode = (enum pl_tex_sample_mode) filter;
            if (sample_mode != src->mode) {
                SH_FAIL(sh, "Trying to use a shader that requires a different "
                        "filter mode than the external sampler.");
                return false;
            }
        }
    }

    src_w = PL_DEF(src_w, src_params(src).w);
    src_h = PL_DEF(src_h, src_params(src).h);
    pl_assert(src_w && src_h);

    int out_w = PL_DEF(src->new_w, roundf(fabs(src_w)));
    int out_h = PL_DEF(src->new_h, roundf(fabs(src_h)));
    pl_assert(out_w && out_h);

    if (ratio_x)
        *ratio_x = out_w / fabs(src_w);
    if (ratio_y)
        *ratio_y = out_h / fabs(src_h);
    if (scale)
        *scale = PL_DEF(src->scale, 1.0);

    if (comp_mask) {
        uint8_t tex_mask = 0x0Fu;
        if (src->tex) {
            // Mask containing only the number of components in the texture
            tex_mask = (1 << src->tex->params.format->num_components) - 1;
        }

        uint8_t src_mask = src->component_mask;
        if (!src_mask)
            src_mask = (1 << PL_DEF(src->components, 4)) - 1;

        // Only actually sample components that are both requested and
        // available in the texture being sampled
        *comp_mask = tex_mask & src_mask;
    }

    if (resizeable)
        out_w = out_h = 0;
    if (!sh_require(sh, sig, out_w, out_h))
        return false;

    if (src->tex) {
        struct pl_rect2df rect = {
            .x0 = src->rect.x0,
            .y0 = src->rect.y0,
            .x1 = src->rect.x0 + src_w,
            .y1 = src->rect.y0 + src_h,
        };

        if (fn)
            *fn = sh_tex_fn(sh, src->tex->params);

        *src_tex = sh_bind(sh, src->tex, src->address_mode, sample_mode,
                           "src_tex", &rect, pos, size, pt);
    } else {
        if (size) {
            *size = sh_var(sh, (struct pl_shader_var) {
                .var = pl_var_vec2("tex_size"),
                .data = &(float[2]) { src->tex_w, src->tex_h },
            });
        }

        if (pt) {
            float sx = 1.0 / src->tex_w, sy = 1.0 / src->tex_h;
            if (src->sampler == PL_SAMPLER_RECT)
                sx = sy = 1.0;

            *pt = sh_var(sh, (struct pl_shader_var) {
                .var = pl_var_vec2("tex_pt"),
                .data = &(float[2]) { sx, sy },
            });
        }

        if (fn)
            *fn = sh_tex_fn(sh, (struct pl_tex_params) { .w = 1, .h = 1 }); // 2D

        sh->sampler_type = src->sampler;

        pl_assert(src->format);
        switch (src->format) {
        case PL_FMT_UNKNOWN:
        case PL_FMT_FLOAT:
        case PL_FMT_UNORM:
        case PL_FMT_SNORM: sh->sampler_prefix = ' '; break;
        case PL_FMT_UINT: sh->sampler_prefix = 'u'; break;
        case PL_FMT_SINT: sh->sampler_prefix = 's'; break;
        case PL_FMT_TYPE_COUNT:
            pl_unreachable();
        }

        *src_tex = "src_tex";
        *pos = "tex_coord";
    }

    return true;
}

void pl_shader_deband(pl_shader sh, const struct pl_sample_src *src,
                      const struct pl_deband_params *params)
{
    float scale;
    ident_t tex, pos, pt;
    const char *fn;
    uint8_t mask;
    char swiz[5];
    if (!setup_src(sh, src, &tex, &pos, NULL, &pt, NULL, NULL, &mask, &scale,
                   false, &fn, LINEAR))
        return;

    params = PL_DEF(params, &pl_deband_default_params);
    sh_describe(sh, "debanding");
    GLSL("vec4 color = %s(%s, %s);  \n"
         "// pl_shader_deband       \n"
         "{                         \n",
         fn, tex, pos);

    uint8_t num_comps = sh_tex_swiz(swiz, mask & ~0x8u); // ignore alpha
    pl_assert(num_comps <= 3);
    if (!num_comps) {
        GLSL("color *= %s;  \n"
             "}             \n",
             SH_FLOAT(scale));
        return;
    }

    GLSL("#define GET(X, Y) (%s(%s, %s + %s * vec2(X, Y)).%s)   \n"
         "#define T %s                                          \n",
         fn, tex, pos, pt, swiz,
         sh_float_type(num_comps));

    const char *bvec = sh_bvec(sh, num_comps);
    ident_t prng = sh_prng(sh, true, NULL);
    GLSL("T avg, diff, bound;   \n"
         "T res = color.%s;     \n"
         "vec2 d;               \n",
         swiz);

    if (params->iterations > 0) {
        ident_t radius = sh_const_float(sh, "radius", params->radius);
        ident_t threshold = sh_const_float(sh, "threshold",
                                           params->threshold / (1000 * scale));

        // For each iteration, compute the average at a given distance and
        // pick it instead of the color if the difference is below the threshold.
        for (int i = 1; i <= params->iterations; i++) {
            GLSL(// Compute a random angle and distance
                 "d = %s.xy * vec2(%d.0 * %s, %f);                  \n"
                 "d = d.x * vec2(cos(d.y), sin(d.y));               \n"
                 // Sample at quarter-turn intervals around the source pixel
                 "avg = T(0.0);                                     \n"
                 "avg += GET(+d.x, +d.y);                           \n"
                 "avg += GET(-d.x, +d.y);                           \n"
                 "avg += GET(-d.x, -d.y);                           \n"
                 "avg += GET(+d.x, -d.y);                           \n"
                 "avg *= 0.25;                                      \n"
                 // Compare the (normalized) average against the pixel
                 "diff = abs(res - avg);                            \n"
                 "bound = T(%s / %d.0);                             \n",
                 prng, i, radius, M_PI * 2,
                 threshold, i);

            if (num_comps > 1) {
                GLSL("res = mix(avg, res, %s(greaterThan(diff, bound))); \n", bvec);
            } else {
                GLSL("res = mix(avg, res, %s(diff > bound)); \n", bvec);
            }
        }
    }

    // Add some random noise to smooth out residual differences
    if (params->grain > 0) {
        // Avoid adding grain near true black
        GLSL("bound = T(\n");
        for (int c = 0; c < num_comps; c++) {
            GLSL("%c%s", c > 0 ? ',' : ' ',
                 SH_FLOAT(params->grain_neutral[c] / scale));
        }
        GLSL(");                                        \n"
             "T strength = min(abs(res - bound), %s);   \n"
             "res += strength * (T(%s) - T(0.5));       \n",
             SH_FLOAT(params->grain / (1000.0 * scale)), prng);
    }

    GLSL("color.%s = res;   \n"
         "color *= %s;      \n"
         "#undef T          \n"
         "#undef GET        \n"
         "}                 \n",
         swiz, SH_FLOAT(scale));
}

bool pl_shader_sample_direct(pl_shader sh, const struct pl_sample_src *src)
{
    float scale;
    ident_t tex, pos;
    const char *fn;
    if (!setup_src(sh, src, &tex, &pos, NULL, NULL, NULL, NULL, NULL, &scale,
                   true, &fn, BEST))
        return false;

    GLSL("// pl_shader_sample_direct          \n"
         "vec4 color = vec4(%s) * %s(%s, %s); \n",
         SH_FLOAT(scale), fn, tex, pos);
    return true;
}

bool pl_shader_sample_nearest(pl_shader sh, const struct pl_sample_src *src)
{
    float scale;
    ident_t tex, pos;
    const char *fn;
    if (!setup_src(sh, src, &tex, &pos, NULL, NULL, NULL, NULL, NULL, &scale,
                   true, &fn, NEAREST))
        return false;

    sh_describe(sh, "nearest");
    GLSL("// pl_shader_sample_nearest         \n"
         "vec4 color = vec4(%s) * %s(%s, %s); \n",
         SH_FLOAT(scale), fn, tex, pos);
    return true;
}

bool pl_shader_sample_bilinear(pl_shader sh, const struct pl_sample_src *src)
{
    float scale;
    ident_t tex, pos;
    const char *fn;
    if (!setup_src(sh, src, &tex, &pos, NULL, NULL, NULL, NULL, NULL, &scale,
                   true, &fn, LINEAR))
        return false;

    sh_describe(sh, "bilinear");
    GLSL("// pl_shader_sample_bilinear        \n"
         "vec4 color = vec4(%s) * %s(%s, %s); \n",
         SH_FLOAT(scale), fn, tex, pos);
    return true;
}

static void bicubic_calcweights(pl_shader sh, const char *t, const char *s)
{
    // Explanation of how bicubic scaling with only 4 texel fetches is done:
    //   http://www.mate.tue.nl/mate/pdfs/10318.pdf
    //   'Efficient GPU-Based Texture Interpolation using Uniform B-Splines'
    GLSL("vec4 %s = vec4(-0.5, 0.1666, 0.3333, -0.3333) * %s \n"
         "          + vec4(1, 0, -0.5, 0.5);                 \n"
         "%s = %s * %s + vec4(0.0, 0.0, -0.5, 0.5);          \n"
         "%s = %s * %s + vec4(-0.6666, 0, 0.8333, 0.1666);   \n"
         "%s.xy /= %s.zw;                                    \n"
         "%s.xy += vec2(1.0 + %s, 1.0 - %s);                 \n",
         t, s,
         t, t, s,
         t, t, s,
         t, t,
         t, s, s);
}

bool pl_shader_sample_bicubic(pl_shader sh, const struct pl_sample_src *src)
{
    ident_t tex, pos, size, pt;
    float rx, ry, scale;
    const char *fn;
    if (!setup_src(sh, src, &tex, &pos, &size, &pt, &rx, &ry, NULL, &scale,
                   true, &fn, LINEAR))
        return false;

    if (rx < 1 || ry < 1) {
        PL_TRACE(sh, "Using fast bicubic sampling when downscaling. This "
                 "will most likely result in nasty aliasing!");
    }

    sh_describe(sh, "bicubic");
    GLSL("// pl_shader_sample_bicubic                   \n"
         "vec4 color;                                   \n"
         "{                                             \n"
         "vec2 pos  = %s;                               \n"
         "vec2 pt   = %s;                               \n"
         "vec2 size = %s;                               \n"
         "vec2 fcoord = fract(pos * size + vec2(0.5));  \n",
         pos, pt, size);

    bicubic_calcweights(sh, "parmx", "fcoord.x");
    bicubic_calcweights(sh, "parmy", "fcoord.y");

    GLSL("vec4 cdelta;                              \n"
         "cdelta.xz = parmx.rg * vec2(-pt.x, pt.x); \n"
         "cdelta.yw = parmy.rg * vec2(-pt.y, pt.y); \n"
         // first y-interpolation
         "vec4 ar = %s(%s, pos + cdelta.xy);        \n"
         "vec4 ag = %s(%s, pos + cdelta.xw);        \n"
         "vec4 ab = mix(ag, ar, parmy.b);           \n"
         // second y-interpolation
         "vec4 br = %s(%s, pos + cdelta.zy);        \n"
         "vec4 bg = %s(%s, pos + cdelta.zw);        \n"
         "vec4 aa = mix(bg, br, parmy.b);           \n"
         // x-interpolation
         "color = vec4(%s) * mix(aa, ab, parmx.b);  \n"
         "}                                         \n",
         fn, tex, fn, tex, fn, tex, fn, tex, SH_FLOAT(scale));

    return true;
}

bool pl_shader_sample_oversample(pl_shader sh, const struct pl_sample_src *src,
                                 float threshold)
{
    ident_t tex, pos, size, pt;
    float rx, ry, scale;
    const char *fn;
    if (!setup_src(sh, src, &tex, &pos, &size, &pt, &rx, &ry, NULL, &scale,
                   true, &fn, LINEAR))
        return false;

    ident_t ratio = sh_var(sh, (struct pl_shader_var) {
        .var = pl_var_vec2("ratio"),
        .data = &(float[2]) { rx, ry },
    });

    // Round the position to the nearest pixel
    sh_describe(sh, "oversample");
    GLSL("// pl_shader_sample_oversample                \n"
         "vec4 color;                                   \n"
         "{                                             \n"
         "vec2 pt = %s, size = %s, pos = %s;            \n"
         "vec2 fcoord = fract(pos * size - vec2(0.5));  \n"
         "vec2 coeff = (fcoord - vec2(0.5)) * %s;       \n"
         "coeff = clamp(coeff + vec2(0.5), 0.0, 1.0);   \n",
         pt, size, pos, ratio);

    if (threshold > 0) {
        threshold = PL_MIN(threshold, 0.5f);
        ident_t thresh = sh_const_float(sh, "threshold", threshold);
        GLSL("coeff = mix(coeff, vec2(0.0),             \n"
             "    lessThan(coeff, vec2(%s)));           \n"
             "coeff = mix(coeff, vec2(1.0),             \n"
             "    greaterThan(coeff, vec2(1.0 - %s)));  \n",
             thresh, thresh);
    }

    // Compute the right output blend of colors
    GLSL("pos += (coeff - fcoord) * pt;                 \n"
         "color = vec4(%s) * %s(%s, pos);               \n"
         "}                                             \n",
         SH_FLOAT(scale), fn, tex);

    return true;
}

static void describe_filter(pl_shader sh, const struct pl_filter_config *cfg,
                            const char *stage, float rx, float ry)
{
    const char *dir;
    if (rx > 1 && ry > 1) {
        dir = "up";
    } else if (rx < 1 && ry < 1) {
        dir = "down";
    } else if (rx == 1 && ry == 1) {
        dir = "noop";
    } else {
        dir = "ana";
    }

    if (cfg->name) {
        sh_describef(sh, "%s %sscaling (%s)", stage, dir, cfg->name);
    } else if (cfg->window) {
        sh_describef(sh, "%s %sscaling (%s+%s)", stage, dir,
                     PL_DEF(cfg->kernel->name, "unknown"),
                     PL_DEF(cfg->window->name, "unknown"));
    } else {
        sh_describef(sh, "%s %sscaling (%s)", stage, dir,
                     PL_DEF(cfg->kernel->name, "unknown"));
    }
}

static bool filter_compat(pl_filter filter, float inv_scale,
                          int lut_entries, float cutoff,
                          const struct pl_filter_config *params)
{
    if (!filter)
        return false;
    if (filter->params.lut_entries != lut_entries)
        return false;
    if (fabs(filter->params.filter_scale - inv_scale) > 1e-3)
        return false;
    if (filter->params.cutoff != cutoff)
        return false;

    return pl_filter_config_eq(&filter->params.config, params);
}

// Subroutine for computing and adding an individual texel contribution
// If `in` is NULL, samples directly
// If `in` is set, takes the pixel from inX[idx] where X is the component,
// `in` is the given identifier, and `idx` must be defined by the caller
static void polar_sample(pl_shader sh, pl_filter filter, const char *fn,
                         ident_t tex, ident_t lut, ident_t cutoff, ident_t radius,
                         int x, int y, uint8_t comp_mask, ident_t in)
{
    // Since we can't know the subpixel position in advance, assume a
    // worst case scenario
    int yy = y > 0 ? y-1 : y;
    int xx = x > 0 ? x-1 : x;
    float dmax = sqrt(xx*xx + yy*yy);
    // Skip samples definitely outside the radius
    if (dmax >= filter->radius_cutoff)
        return;

    GLSL("d = length(vec2(%d.0, %d.0) - fcoord);\n", x, y);
    // Check for samples that might be skippable
    bool maybe_skippable = dmax >= filter->radius_cutoff - M_SQRT2;
    if (maybe_skippable)
        GLSL("if (d < %s) {\n", cutoff);

    // Get the weight for this pixel
    GLSL("w = %s(d * 1.0/%s); \n"
         "wsum += w;          \n",
         lut, radius);

    if (in) {
        for (uint8_t comps = comp_mask; comps;) {
            uint8_t c = __builtin_ctz(comps);
            GLSL("color[%d] += w * %s%d[idx]; \n", c, in, c);
            comps &= ~(1 << c);
        }
    } else {
        GLSL("c = %s(%s, base + pt * vec2(%d.0, %d.0)); \n",
             fn, tex, x, y);
        for (uint8_t comps = comp_mask; comps;) {
            uint8_t c = __builtin_ctz(comps);
            GLSL("color[%d] += w * c[%d]; \n", c, c);
            comps &= ~(1 << c);
        }
    }

    if (maybe_skippable)
        GLSL("}\n");
}

struct sh_sampler_obj {
    pl_filter filter;
    pl_shader_obj lut;
    pl_shader_obj pass2; // for pl_shader_sample_ortho
};

static void sh_sampler_uninit(pl_gpu gpu, void *ptr)
{
    struct sh_sampler_obj *obj = ptr;
    pl_shader_obj_destroy(&obj->lut);
    pl_shader_obj_destroy(&obj->pass2);
    pl_filter_free(&obj->filter);
    *obj = (struct sh_sampler_obj) {0};
}

static void fill_polar_lut(void *data, const struct sh_lut_params *params)
{
    const struct sh_sampler_obj *obj = params->priv;
    pl_filter filt = obj->filter;

    pl_assert(params->width == filt->params.lut_entries && params->comps == 1);
    memcpy(data, filt->weights, params->width * sizeof(float));
}

bool pl_shader_sample_polar(pl_shader sh, const struct pl_sample_src *src,
                            const struct pl_sample_filter_params *params)
{
    pl_assert(params);
    if (!params->filter.polar) {
        SH_FAIL(sh, "Trying to use polar sampling with a non-polar filter?");
        return false;
    }

    uint8_t comp_mask;
    float rx, ry, scale;
    ident_t src_tex, pos, size, pt;
    const char *fn;
    if (!setup_src(sh, src, &src_tex, &pos, &size, &pt, &rx, &ry, &comp_mask,
                   &scale, false, &fn, FASTEST))
        return false;

    struct sh_sampler_obj *obj;
    obj = SH_OBJ(sh, params->lut, PL_SHADER_OBJ_SAMPLER, struct sh_sampler_obj,
                 sh_sampler_uninit);
    if (!obj)
        return false;

    float inv_scale = 1.0 / PL_MIN(rx, ry);
    inv_scale = PL_MAX(inv_scale, 1.0);

    if (params->no_widening)
        inv_scale = 1.0;

    int lut_entries = PL_DEF(params->lut_entries, 64);
    float cutoff = PL_DEF(params->cutoff, 0.001);
    bool update = !filter_compat(obj->filter, inv_scale, lut_entries, cutoff,
                                 &params->filter);

    if (update) {
        pl_filter_free(&obj->filter);
        obj->filter = pl_filter_generate(sh->log, pl_filter_params(
            .config         = params->filter,
            .lut_entries    = lut_entries,
            .filter_scale   = inv_scale,
            .cutoff         = cutoff,
        ));

        if (!obj->filter) {
            // This should never happen, but just in case ..
            SH_FAIL(sh, "Failed initializing polar filter!");
            return false;
        }
    }

    describe_filter(sh, &params->filter, "polar", rx, ry);
    GLSL("// pl_shader_sample_polar                     \n"
         "vec4 color = vec4(0.0);                       \n"
         "{                                             \n"
         "vec2 pos = %s, size = %s, pt = %s;            \n"
         "vec2 fcoord = fract(pos * size - vec2(0.5));  \n"
         "vec2 base = pos - pt * fcoord;                \n"
         "vec2 center = base + pt * vec2(0.5);          \n"
         "float w, d, wsum = 0.0;                       \n"
         "int idx;                                      \n"
         "vec4 c;                                       \n",
         pos, size, pt);

    int bound   = ceil(obj->filter->radius_cutoff);
    int offset  = bound - 1; // padding top/left
    int padding = offset + bound; // total padding

    // Determined experimentally on modern AMD and Nvidia hardware. 32 is a
    // good tradeoff for the horizontal work group size. Apart from that,
    // just use as many threads as possible.
    const int bw = 32, bh = sh_glsl(sh).max_group_threads / bw;

    // We need to sample everything from base_min to base_max, so make sure we
    // have enough room in shmem. The extra margin on the ceilf guards against
    // floating point inaccuracy on near-integer scaling ratios.
    const float margin = 1e-5;
    int iw = (int) ceilf(bw / rx - margin) + padding + 1,
        ih = (int) ceilf(bh / ry - margin) + padding + 1;
    int sizew = iw, sizeh = ih;

    pl_gpu gpu = SH_GPU(sh);
    bool dynamic_size = SH_PARAMS(sh).dynamic_constants ||
                        !gpu || !gpu->limits.array_size_constants;
    if (dynamic_size) {
        // Overallocate the array slightly to reduce recompilation overhead
        sizew = PL_ALIGN2(sizew, 8);
        sizeh = PL_ALIGN2(sizeh, 8);
    }

    ident_t in = NULL;
    int num_comps = __builtin_popcount(comp_mask);
    int shmem_req = (sizew * sizeh * num_comps + 2) * sizeof(float);
    bool is_compute = !params->no_compute &&
                      sh_glsl(sh).compute &&
                      sh_glsl(sh).version >= 130 && // needed for round()
                      sh_try_compute(sh, bw, bh, false, shmem_req);

    // For compute shaders, which read the input texels primarily from shmem,
    // using a texture-based LUT is better. For the fragment shader fallback
    // code, which is primarily texture bound, the extra cost of LUT
    // interpolation is worth the reduction in texel fetches.
    ident_t lut = sh_lut(sh, sh_lut_params(
        .object     = &obj->lut,
        .lut_type   = is_compute ? SH_LUT_TEXTURE : SH_LUT_AUTO,
        .var_type   = PL_VAR_FLOAT,
        .width      = lut_entries,
        .comps      = 1,
        .linear     = true,
        .update     = update,
        .fill       = fill_polar_lut,
        .priv       = obj,
    ));

    if (!lut) {
        SH_FAIL(sh, "Failed initializing polar LUT!");
        return false;
    }

    ident_t cutoff_c = sh_const_float(sh, "radius_cutoff", obj->filter->radius_cutoff);
    ident_t radius_c = sh_const_float(sh, "radius", obj->filter->radius);

    if (is_compute) {

        // Compute shader kernel
        GLSL("uvec2 base_id = uvec2(0u); \n");
        if (src->rect.x0 > src->rect.x1)
            GLSL("base_id.x = gl_WorkGroupSize.x - 1u; \n");
        if (src->rect.y0 > src->rect.y1)
            GLSL("base_id.y = gl_WorkGroupSize.y - 1u; \n");

        in = sh_fresh(sh, "in");
        GLSLH("shared vec2 %s_base; \n", in);
        GLSL("if (gl_LocalInvocationID.xy == base_id)               \n"
             "    %s_base = base;                                   \n"
             "barrier();                                            \n"
             "ivec2 rel = ivec2(round((base - %s_base) * size));    \n",
             in, in);

        ident_t sizew_c = sh_const(sh, (struct pl_shader_const) {
            .type = PL_VAR_SINT,
            .compile_time = true,
            .name = "sizew",
            .data = &sizew,
        });

        ident_t sizeh_c = sh_const(sh, (struct pl_shader_const) {
            .type = PL_VAR_SINT,
            .compile_time = true,
            .name = "sizeh",
            .data = &sizeh,
        });

        ident_t iw_c = sizew_c, ih_c = sizeh_c;
        if (dynamic_size) {
            iw_c = sh_const_int(sh, "iw", iw);
            ih_c = sh_const_int(sh, "ih", ih);
        }

        // Load all relevant texels into shmem
        GLSL("for (int y = int(gl_LocalInvocationID.y); y < %s; y += %d) {  \n"
             "for (int x = int(gl_LocalInvocationID.x); x < %s; x += %d) {  \n"
             "c = %s(%s, %s_base + pt * vec2(x - %d, y - %d));              \n",
             ih_c, bh, iw_c, bw, fn, src_tex, in, offset, offset);

        for (uint8_t comps = comp_mask; comps;) {
            uint8_t c = __builtin_ctz(comps);
            GLSLH("shared float %s%d[%s * %s]; \n", in, c, sizeh_c, sizew_c);
            GLSL("%s%d[%s * y + x] = c[%d]; \n", in, c, sizew_c, c);
            comps &= ~(1 << c);
        }

        GLSL("}}                     \n"
             "barrier();             \n");

        // Dispatch the actual samples
        for (int y = 1 - bound; y <= bound; y++) {
            for (int x = 1 - bound; x <= bound; x++) {
                GLSL("idx = %s * rel.y + rel.x + %s * %d + %d; \n",
                     sizew_c, sizew_c, y + offset, x + offset);
                polar_sample(sh, obj->filter, fn, src_tex, lut, cutoff_c, radius_c,
                             x, y, comp_mask, in);
            }
        }
    } else {
        // Fragment shader sampling
        for (uint8_t comps = comp_mask; comps;) {
            uint8_t c = __builtin_ctz(comps);
            GLSL("vec4 in%d;\n", c);
            comps &= ~(1 << c);
        }

        // For maximum efficiency, we want to use textureGather() if
        // possible, rather than direct sampling. Since this is not
        // always possible/sensible, we need to possibly intermix gathering
        // with regular sampling. This requires keeping track of which
        // pixels in the next row were already gathered by the previous
        // row.
        uint32_t gathered_cur = 0x0, gathered_next = 0x0;
        const float radius2 = PL_SQUARE(obj->filter->radius_cutoff);
        const int base = bound - 1;

        if (base + bound >= 8 * sizeof(gathered_cur)) {
            SH_FAIL(sh, "Polar radius %f exceeds implementation capacity!",
                    obj->filter->radius_cutoff);
            return false;
        }

        for (int y = 1 - bound; y <= bound; y++) {
            for (int x = 1 - bound; x <= bound; x++) {
                // Skip already gathered texels
                uint32_t bit = 1llu << (base + x);
                if (gathered_cur & bit)
                    continue;

                // Using texture gathering is only more efficient than direct
                // sampling in the case where we expect to be able to use all
                // four gathered texels, without having to discard any. So
                // only do it if we suspect it will be a win rather than a
                // loss.
                int xx = x*x, xx1 = (x+1)*(x+1);
                int yy = y*y, yy1 = (y+1)*(y+1);
                bool use_gather = PL_MAX(xx, xx1) + PL_MAX(yy, yy1) < radius2;
                use_gather &= PL_MAX(x, y) <= sh_glsl(sh).max_gather_offset;
                use_gather &= PL_MIN(x, y) >= sh_glsl(sh).min_gather_offset;
                use_gather &= !src->tex || src->tex->params.format->gatherable;

                // Gathering from components other than the R channel requires
                // support for GLSL 400, which introduces the overload of
                // textureGather* that allows specifying the component.
                //
                // This is also the minimum requirement if we don't know the
                // texture format capabilities, for the sampler2D interface
                if (comp_mask != 0x1 || !src->tex)
                    use_gather &= sh_glsl(sh).version >= 400;

                if (!use_gather) {
                    // Switch to direct sampling instead
                    polar_sample(sh, obj->filter, fn, src_tex, lut, cutoff_c,
                                 radius_c, x, y, comp_mask, NULL);
                    continue;
                }

                // Gather the four surrounding texels simultaneously
                for (uint8_t comps = comp_mask; comps;) {
                    uint8_t c = __builtin_ctz(comps);
                    if (x || y) {
                        if (c) {
                            GLSL("in%d = textureGatherOffset(%s, center, "
                                 "ivec2(%d, %d), %d);\n",
                                 c, src_tex, x, y, c);
                        } else {
                            GLSL("in0 = textureGatherOffset(%s, center, "
                                 "ivec2(%d, %d));\n", src_tex, x, y);
                        }
                    } else {
                        if (c) {
                            GLSL("in%d = textureGather(%s, center, %d);\n",
                                 c, src_tex, c);
                        } else {
                            GLSL("in0 = textureGather(%s, center);\n", src_tex);
                        }
                    }
                    comps &= ~(1 << c);
                }

                // Mix in all of the points with their weights
                for (int p = 0; p < 4; p++) {
                    // The four texels are gathered counterclockwise starting
                    // from the bottom left
                    static const int xo[4] = {0, 1, 1, 0};
                    static const int yo[4] = {1, 1, 0, 0};
                    if (x+xo[p] > bound || y+yo[p] > bound)
                        continue; // next subpixel

                    GLSL("idx = %d;\n", p);
                    polar_sample(sh, obj->filter, fn, src_tex, lut, cutoff_c,
                                 radius_c, x+xo[p], y+yo[p], comp_mask, "in");
                }

                // Mark the other next row's pixels as already gathered
                gathered_next |= bit | (bit << 1);
                x++; // skip adjacent pixel
            }

            // Prepare for new row
            gathered_cur = gathered_next;
            gathered_next = 0;
        }
    }

    GLSL("color = vec4(%s / wsum) * color; \n", SH_FLOAT(scale));
    if (!(comp_mask & (1 << PL_CHANNEL_A)))
        GLSL("color.a = 1.0; \n");

    GLSL("}\n");
    return true;
}

static void fill_ortho_lut(void *data, const struct sh_lut_params *params)
{
    const struct sh_sampler_obj *obj = params->priv;
    pl_filter filt = obj->filter;
    size_t entries = filt->params.lut_entries * filt->row_stride;

    pl_assert(params->width * params->height * params->comps == entries);
    memcpy(data, filt->weights, entries * sizeof(float));
}

bool pl_shader_sample_ortho(pl_shader sh, int pass,
                            const struct pl_sample_src *src,
                            const struct pl_sample_filter_params *params)
{
    pl_assert(params);
    if (params->filter.polar) {
        SH_FAIL(sh, "Trying to use separated sampling with a polar filter?");
        return false;
    }

    pl_gpu gpu = SH_GPU(sh);
    pl_assert(gpu);

    struct pl_sample_src srcfix = *src;
    switch (pass) {
    case PL_SEP_VERT:
        srcfix.rect.x0 = 0;
        srcfix.rect.x1 = srcfix.new_w = src_params(src).w;
        break;
    case PL_SEP_HORIZ:
        srcfix.rect.y0 = 0;
        srcfix.rect.y1 = srcfix.new_h = src_params(src).h;
        break;
    }

    uint8_t comp_mask;
    float ratio[PL_SEP_PASSES], scale;
    ident_t src_tex, pos, size, pt;
    const char *fn;
    if (!setup_src(sh, &srcfix, &src_tex, &pos, &size, &pt,
                   &ratio[PL_SEP_HORIZ], &ratio[PL_SEP_VERT],
                   &comp_mask, &scale, false, &fn, FASTEST))
        return false;

    // We can store a separate sampler object per dimension, so dispatch the
    // right one. This is needed for two reasons:
    // 1. Anamorphic content can have a different scaling ratio for each
    //    dimension. In particular, you could be upscaling in one and
    //    downscaling in the other.
    // 2. After fixing the source for `setup_src`, we lose information about
    //    the scaling ratio of the other component. (Although this is only a
    //    minor reason and could easily be changed with some boilerplate)
    struct sh_sampler_obj *obj;
    obj = SH_OBJ(sh, params->lut, PL_SHADER_OBJ_SAMPLER,
                 struct sh_sampler_obj, sh_sampler_uninit);
    if (!obj)
        return false;

    if (pass != 0) {
        obj = SH_OBJ(sh, &obj->pass2, PL_SHADER_OBJ_SAMPLER,
                     struct sh_sampler_obj, sh_sampler_uninit);
        assert(obj);
    }

    float inv_scale = 1.0 / ratio[pass];
    inv_scale = PL_MAX(inv_scale, 1.0);

    if (params->no_widening)
        inv_scale = 1.0;

    int lut_entries = PL_DEF(params->lut_entries, 64);
    bool update = !filter_compat(obj->filter, inv_scale, lut_entries, 0.0,
                                 &params->filter);

    if (update) {
        pl_filter_free(&obj->filter);
        obj->filter = pl_filter_generate(sh->log, pl_filter_params(
            .config             = params->filter,
            .lut_entries        = lut_entries,
            .filter_scale       = inv_scale,
            .max_row_size       = gpu->limits.max_tex_2d_dim / 4,
            .row_stride_align   = 4,
        ));

        if (!obj->filter) {
            // This should never happen, but just in case ..
            SH_FAIL(sh, "Failed initializing separated filter!");
            return false;
        }
    }

    int N = obj->filter->row_size; // number of samples to convolve
    int width = obj->filter->row_stride / 4; // width of the LUT texture
    ident_t lut = sh_lut(sh, sh_lut_params(
        .object     = &obj->lut,
        .var_type   = PL_VAR_FLOAT,
        .width      = width,
        .height     = lut_entries,
        .comps      = 4,
        .linear     = true,
        .update     = update,
        .fill       = fill_ortho_lut,
        .priv       = obj,
    ));
    if (!lut) {
        SH_FAIL(sh, "Failed initializing separated LUT!");
        return false;
    }

    const int dir[PL_SEP_PASSES][2] = {
        [PL_SEP_HORIZ] = {1, 0},
        [PL_SEP_VERT]  = {0, 1},
    };

    static const char *names[PL_SEP_PASSES] = {
        [PL_SEP_HORIZ] = "ortho (horiz)",
        [PL_SEP_VERT]  = "ortho (vert)",
    };

    describe_filter(sh, &params->filter, names[pass], ratio[pass], ratio[pass]);
    GLSL("// pl_shader_sample_ortho                        \n"
         "vec4 color = vec4(0.0);                          \n"
         "{                                                \n"
         "vec2 pos = %s, size = %s, pt = %s;               \n"
         "vec2 dir = vec2(%d.0, %d.0);                     \n"
         "pt *= dir;                                       \n"
         "vec2 fcoord2 = fract(pos * size - vec2(0.5));    \n"
         "float fcoord = dot(fcoord2, dir);                \n"
         "vec2 base = pos - fcoord * pt - pt * vec2(%d.0); \n"
         "float weight;                                    \n"
         "vec4 ws, c;                                      \n",
         pos, size, pt,
         dir[pass][0], dir[pass][1],
         N / 2 - 1);

    bool use_ar = params->antiring > 0;
    if (use_ar) {
        GLSL("vec4 hi = vec4(0.0); \n"
             "vec4 lo = vec4(1e9); \n");
    }

    // Dispatch all of the samples
    GLSL("// scaler samples\n");
    for (int n = 0; n < N; n++) {
        // Load the right weight for this instance. For every 4th weight, we
        // need to fetch another LUT entry. Otherwise, just use the previous
        if (n % 4 == 0) {
            float denom = PL_MAX(1, width - 1); // avoid division by zero
            GLSL("ws = %s(vec2(%f, fcoord));\n", lut, (n / 4) / denom);
        }
        GLSL("weight = ws[%d];\n", n % 4);

        // Load the input texel and add it to the running sum
        GLSL("c = %s(%s, base + pt * vec2(%d.0)); \n",
             fn, src_tex, n);

        for (uint8_t comps = comp_mask; comps;) {
            uint8_t c = __builtin_ctz(comps);
            GLSL("color[%d] += weight * c[%d]; \n", c, c);
            comps &= ~(1 << c);

            if (use_ar && (n == N / 2 - 1 || n == N / 2)) {
                GLSL("lo[%d] = min(lo[%d], c[%d]); \n"
                     "hi[%d] = max(hi[%d], c[%d]); \n",
                     c, c, c, c, c, c);
            }
        }
    }

    if (use_ar) {
        GLSL("color = mix(color, clamp(color, lo, hi), %s);\n",
             sh_const_float(sh, "antiring", params->antiring));
    }

    GLSL("color *= vec4(%s);\n", SH_FLOAT(scale));
    if (!(comp_mask & (1 << PL_CHANNEL_A)))
        GLSL("color.a = 1.0; \n");

    GLSL("}\n");
    return true;
}
