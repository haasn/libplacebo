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

#include <libplacebo/colorspace.h>
#include <libplacebo/shaders/sampling.h>

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
                      ident_t *src_tex, ident_t *pos, ident_t *pt,
                      float *ratio_x, float *ratio_y, uint8_t *comp_mask,
                      float *scale, bool resizeable,
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
        pl_rect2df rect = {
            .x0 = src->rect.x0,
            .y0 = src->rect.y0,
            .x1 = src->rect.x0 + src_w,
            .y1 = src->rect.y0 + src_h,
        };

        *src_tex = sh_bind(sh, src->tex, src->address_mode, sample_mode,
                           "src_tex", &rect, pos, pt);
    } else {
        if (pt) {
            float sx = 1.0 / src->tex_w, sy = 1.0 / src->tex_h;
            if (src->sampler == PL_SAMPLER_RECT)
                sx = sy = 1.0;

            *pt = sh_var(sh, (struct pl_shader_var) {
                .var = pl_var_vec2("tex_pt"),
                .data = &(float[2]) { sx, sy },
            });
        }

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

        *src_tex = sh_fresh(sh, "src_tex");
        *pos     = sh_fresh(sh, "pos");

        GLSLH("#define "$" src_tex  \n"
              "#define "$" pos      \n",
              *src_tex, *pos);
    }

    return true;
}

void pl_shader_deband(pl_shader sh, const struct pl_sample_src *src,
                      const struct pl_deband_params *params)
{
    float scale;
    ident_t tex, pos, pt;
    uint8_t mask;
    if (!setup_src(sh, src, &tex, &pos, &pt, NULL, NULL, &mask, &scale, false, LINEAR))
        return;

    params = PL_DEF(params, &pl_deband_default_params);
    sh_describe(sh, "debanding");
    GLSL("vec4 color;                       \n"
         "// pl_shader_deband               \n"
         "{                                 \n"
         "vec2 pos = "$", pt = "$";         \n"
         "color = textureLod("$", pos, 0.0);\n",
         pos, pt, tex);

    mask &= ~0x8u; // ignore alpha channel
    uint8_t num_comps = sh_num_comps(mask);
    const char *swiz = sh_swizzle(mask);
    pl_assert(num_comps <= 3);
    if (!num_comps) {
        GLSL("color *= "$"; \n"
             "}             \n",
             SH_FLOAT(scale));
        return;
    }

    GLSL("#define GET(X, Y)                                   \\\n"
         "    (textureLod("$", pos + pt * vec2(X, Y), 0.0).%s)  \n"
         "#define T %s                                          \n",
         tex, swiz, sh_float_type(mask));

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
                 "d = "$".xy * vec2(%d.0 * "$", %f);    \n"
                 "d = d.x * vec2(cos(d.y), sin(d.y));   \n"
                 // Sample at quarter-turn intervals around the source pixel
                 "avg = T(0.0);                         \n"
                 "avg += GET(+d.x, +d.y);               \n"
                 "avg += GET(-d.x, +d.y);               \n"
                 "avg += GET(-d.x, -d.y);               \n"
                 "avg += GET(+d.x, -d.y);               \n"
                 "avg *= 0.25;                          \n"
                 // Compare the (normalized) average against the pixel
                 "diff = abs(res - avg);                \n"
                 "bound = T("$" / %d.0);                \n",
                 prng, i, radius, M_PI * 2,
                 threshold, i);

            if (num_comps > 1) {
                GLSL("res = mix(avg, res, greaterThan(diff, bound)); \n");
            } else {
                GLSL("res = mix(avg, res, diff > bound); \n");
            }
        }
    }

    // Add some random noise to smooth out residual differences
    if (params->grain > 0) {
        // Avoid adding grain near true black
        GLSL("bound = T(\n");
        for (int c = 0; c < num_comps; c++) {
            GLSL("%c"$, c > 0 ? ',' : ' ',
                 SH_FLOAT(params->grain_neutral[c] / scale));
        }
        GLSL(");                                        \n"
             "T strength = min(abs(res - bound), "$");  \n"
             "res += strength * (T("$") - T(0.5));      \n",
             SH_FLOAT(params->grain / (1000.0 * scale)), prng);
    }

    GLSL("color.%s = res;   \n"
         "color *= "$";     \n"
         "#undef T          \n"
         "#undef GET        \n"
         "}                 \n",
         swiz, SH_FLOAT(scale));
}

bool pl_shader_sample_direct(pl_shader sh, const struct pl_sample_src *src)
{
    float scale;
    ident_t tex, pos;
    if (!setup_src(sh, src, &tex, &pos, NULL, NULL, NULL, NULL, &scale, true, BEST))
        return false;

    GLSL("// pl_shader_sample_direct                            \n"
         "vec4 color = vec4("$") * textureLod("$", "$", 0.0);   \n",
         SH_FLOAT(scale), tex, pos);
    return true;
}

bool pl_shader_sample_nearest(pl_shader sh, const struct pl_sample_src *src)
{
    float scale;
    ident_t tex, pos;
    if (!setup_src(sh, src, &tex, &pos,  NULL, NULL, NULL, NULL, &scale, true, NEAREST))
        return false;

    sh_describe(sh, "nearest");
    GLSL("// pl_shader_sample_nearest                           \n"
         "vec4 color = vec4("$") * textureLod("$", "$", 0.0);   \n",
         SH_FLOAT(scale), tex, pos);
    return true;
}

bool pl_shader_sample_bilinear(pl_shader sh, const struct pl_sample_src *src)
{
    float scale;
    ident_t tex, pos;
    if (!setup_src(sh, src, &tex, &pos, NULL, NULL, NULL, NULL, &scale, true, LINEAR))
        return false;

    sh_describe(sh, "bilinear");
    GLSL("// pl_shader_sample_bilinear                          \n"
         "vec4 color = vec4("$") * textureLod("$", "$", 0.0);   \n",
         SH_FLOAT(scale), tex, pos);
    return true;
}

bool pl_shader_sample_bicubic(pl_shader sh, const struct pl_sample_src *src)
{
    ident_t tex, pos, pt;
    float rx, ry, scale;
    if (!setup_src(sh, src, &tex, &pos, &pt, &rx, &ry, NULL, &scale, true, LINEAR))
        return false;

    if (rx < 1 || ry < 1) {
        PL_TRACE(sh, "Using fast bicubic sampling when downscaling. This "
                 "will most likely result in nasty aliasing!");
    }

    // Explanation of how bicubic scaling with only 4 texel fetches is done:
    //   http://www.mate.tue.nl/mate/pdfs/10318.pdf
    //   'Efficient GPU-Based Texture Interpolation using Uniform B-Splines'

    sh_describe(sh, "bicubic");
#pragma GLSL /* pl_shader_sample_bicubic */         \
    vec4 color;                                     \
    {                                               \
    vec2 pos = $pos;                                \
    vec2 size = vec2(textureSize($tex, 0));         \
    vec2 frac  = fract(pos * size + vec2(0.5));     \
    vec2 frac2 = frac * frac;                       \
    vec2 inv   = vec2(1.0) - frac;                  \
    vec2 inv2  = inv * inv;                         \
    /* compute filter weights directly */           \
    vec2 w0 = 1.0/6.0 * inv2 * inv;                 \
    vec2 w1 = 2.0/3.0 - 0.5 * frac2 * (2.0 - frac); \
    vec2 w2 = 2.0/3.0 - 0.5 * inv2  * (2.0 - inv);  \
    vec2 w3 = 1.0/6.0 * frac2 * frac;               \
    vec4 g = vec4(w0 + w1, w2 + w3);                \
    vec4 h = vec4(w1, w3) / g + inv.xyxy;           \
    h.xy -= vec2(2.0);                              \
    /* sample four corners, then interpolate */     \
    vec4 p = pos.xyxy + $pt.xyxy * h;               \
    vec4 c00 = textureLod($tex, p.xy, 0.0);         \
    vec4 c01 = textureLod($tex, p.xw, 0.0);         \
    vec4 c0 = mix(c01, c00, g.y);                   \
    vec4 c10 = textureLod($tex, p.zy, 0.0);         \
    vec4 c11 = textureLod($tex, p.zw, 0.0);         \
    vec4 c1 = mix(c11, c10, g.y);                   \
    color = ${float:scale} * mix(c1, c0, g.x);      \
    }

    return true;
}

bool pl_shader_sample_hermite(pl_shader sh, const struct pl_sample_src *src)
{
    ident_t tex, pos, pt;
    float rx, ry, scale;
    if (!setup_src(sh, src, &tex, &pos, &pt, &rx, &ry, NULL, &scale, true, LINEAR))
        return false;

    if (rx < 1 || ry < 1) {
        PL_TRACE(sh, "Using fast hermite sampling when downscaling. This "
                 "will most likely result in nasty aliasing!");
    }

    sh_describe(sh, "hermite");
#pragma GLSL /* pl_shader_sample_hermite */              \
    vec4 color;                                          \
    {                                                    \
    vec2 pos  = $pos;                                    \
    vec2 size = vec2(textureSize($tex, 0));              \
    vec2 frac = fract(pos * size + vec2(0.5));           \
    pos += $pt * (smoothstep(0.0, 1.0, frac) - frac);    \
    color = ${float:scale} * textureLod($tex, pos, 0.0); \
    }

    return true;
}

bool pl_shader_sample_gaussian(pl_shader sh, const struct pl_sample_src *src)
{
    ident_t tex, pos, pt;
    float rx, ry, scale;
    if (!setup_src(sh, src, &tex, &pos, &pt, &rx, &ry, NULL, &scale, true, LINEAR))
        return false;

    if (rx < 1 || ry < 1) {
        PL_TRACE(sh, "Using fast gaussian sampling when downscaling. This "
                 "will most likely result in nasty aliasing!");
    }

    sh_describe(sh, "gaussian");
#pragma GLSL /* pl_shader_sample_gaussian */        \
    vec4 color;                                     \
    {                                               \
    vec2 pos  = $pos;                               \
    vec2 size = vec2(textureSize($tex, 0));         \
    vec2 off  = -fract(pos * size + vec2(0.5));     \
    vec2 off2 = -2.0 * off * off;                   \
    /* compute gaussian weights */                  \
    vec2 w0 = exp(off2 + 4.0 * off - vec2(2.0));    \
    vec2 w1 = exp(off2);                            \
    vec2 w2 = exp(off2 - 4.0 * off - vec2(2.0));    \
    vec2 w3 = exp(off2 - 8.0 * off - vec2(8.0));    \
    vec4 g = vec4(w0 + w1, w2 + w3);                \
    vec4 h = vec4(w1, w3) / g;                      \
    h.xy -= vec2(1.0);                              \
    h.zw += vec2(1.0);                              \
    g.xy /= g.xy + g.zw; /* explicitly normalize */ \
    /* sample four corners, then interpolate */     \
    vec4 p = pos.xyxy + $pt.xyxy * (h + off.xyxy);  \
    vec4 c00 = textureLod($tex, p.xy, 0.0);         \
    vec4 c01 = textureLod($tex, p.xw, 0.0);         \
    vec4 c0 = mix(c01, c00, g.y);                   \
    vec4 c10 = textureLod($tex, p.zy, 0.0);         \
    vec4 c11 = textureLod($tex, p.zw, 0.0);         \
    vec4 c1 = mix(c11, c10, g.y);                   \
    color = ${float:scale} * mix(c1, c0, g.x);      \
    }

    return true;
}

bool pl_shader_sample_oversample(pl_shader sh, const struct pl_sample_src *src,
                                 float threshold)
{
    ident_t tex, pos, pt;
    float rx, ry, scale;
    if (!setup_src(sh, src, &tex, &pos, &pt, &rx, &ry, NULL, &scale, true, LINEAR))
        return false;

    threshold = PL_CLAMP(threshold, 0.0f, 0.5f);
    sh_describe(sh, "oversample");
    #pragma GLSL /* pl_shader_sample_oversample */       \
    vec4 color;                                          \
    {                                                    \
    vec2 pos = $pos;                                     \
    vec2 size = vec2(textureSize($tex, 0));              \
    /* Round the position to the nearest pixel */        \
    vec2 fcoord = fract(pos * size - vec2(0.5));         \
    float rx = ${dynamic float:rx};                      \
    float ry = ${dynamic float:ry};                      \
    vec2 coeff = (fcoord - vec2(0.5)) * vec2(rx, ry);    \
    coeff = clamp(coeff + vec2(0.5), 0.0, 1.0);          \
    @if (threshold > 0) {                                \
        float thresh = ${float:threshold};               \
        coeff = mix(coeff, vec2(0.0),                    \
            lessThan(coeff, vec2(thresh)));              \
        coeff = mix(coeff, vec2(1.0),                    \
            greaterThan(coeff, vec2(1.0 - thresh)));     \
    @}                                                   \
                                                         \
    /* Compute the right output blend of colors */       \
    pos += (coeff - fcoord) * $pt;                       \
    color = ${float:scale} * textureLod($tex, pos, 0.0); \
    }

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

// Subroutine for computing and adding an individual texel contribution
// If `in` is NULL, samples directly
// If `in` is set, takes the pixel from inX[idx] where X is the component,
// `in` is the given identifier, and `idx` must be defined by the caller
static void polar_sample(pl_shader sh, pl_filter filter,
                         ident_t tex, ident_t lut, ident_t radius,
                         int x, int y, uint8_t comp_mask, ident_t in,
                         bool use_ar, ident_t scale)
{
    // Since we can't know the subpixel position in advance, assume a
    // worst case scenario
    int yy = y > 0 ? y-1 : y;
    int xx = x > 0 ? x-1 : x;
    float dmin = sqrt(xx*xx + yy*yy);
    // Skip samples definitely outside the radius
    if (dmin >= filter->radius)
        return;

    // Check for samples that might be skippable
    bool maybe_skippable = dmin >= filter->radius - M_SQRT2;

    // Check for samples that definitely won't contribute to anti-ringing
    const float ar_radius = filter->radius_zero;
    use_ar &= dmin < ar_radius;

#pragma GLSL                                                    \
    offset = ivec2(${const int: x}, ${const int: y});           \
    d = length(vec2(offset) - fcoord);                          \
    @if (maybe_skippable)                                       \
        if (d < $radius) {                                      \
    w = $lut(d * 1.0 / $radius);                                \
    wsum += w;                                                  \
    @if (in != NULL_IDENT) {                                    \
        @for (c : comp_mask)                                    \
            c[@c] = ${in}_@c[idx];                              \
    @} else {                                                   \
        c = textureLod($tex, base + pt * vec2(offset), 0.0);    \
    @}                                                          \
    @for (c : comp_mask)                                        \
        color[@c] += w * c[@c];                                 \
    @if (use_ar) {                                              \
        if (d <= ${const float: ar_radius}) {                   \
            @for (c : comp_mask) {                              \
                cc = vec2($scale * c[@c]);                      \
                cc.x = 1.0 - cc.x;                              \
                ww = cc + vec2(0.10);                           \
                ww = ww * ww;                                   \
                ww = ww * ww;                                   \
                ww = ww * ww;                                   \
                ww = ww * ww;                                   \
                ww = ww * ww;                                   \
                ww = w * ww;                                    \
                ar@c += ww * cc;                                \
                wwsum@c += ww;                                  \
            @}                                                  \
        }                                                       \
    @}                                                          \
    @if (maybe_skippable)                                       \
        }
}

struct sh_sampler_obj {
    pl_filter filter;
    pl_shader_obj lut;
    pl_shader_obj pass2; // for pl_shader_sample_ortho
};

#define SCALER_LUT_SIZE     256
#define SCALER_LUT_CUTOFF   1e-3f

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

    uint8_t cmask;
    float rx, ry, scalef;
    ident_t src_tex, pos, pt, scale;
    if (!setup_src(sh, src, &src_tex, &pos, &pt, &rx, &ry, &cmask, &scalef, false, FASTEST))
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
    scale = sh_const_float(sh, "scale", scalef);

    struct pl_filter_config cfg = params->filter;
    cfg.antiring = PL_DEF(cfg.antiring, params->antiring);
    cfg.blur = PL_DEF(cfg.blur, 1.0f) * inv_scale;
    bool update = !obj->filter || !pl_filter_config_eq(&obj->filter->params.config, &cfg);
    if (update) {
        pl_filter_free(&obj->filter);
        obj->filter = pl_filter_generate(sh->log, pl_filter_params(
            .config         = cfg,
            .lut_entries    = SCALER_LUT_SIZE,
            .cutoff         = SCALER_LUT_CUTOFF,
        ));

        if (!obj->filter) {
            // This should never happen, but just in case ..
            SH_FAIL(sh, "Failed initializing polar filter!");
            return false;
        }
    }

    describe_filter(sh, &cfg, "polar", rx, ry);
    GLSL("// pl_shader_sample_polar                     \n"
         "vec4 color = vec4(0.0);                       \n"
         "{                                             \n"
         "vec2 pos = "$", pt = "$";                     \n"
         "vec2 size = vec2(textureSize("$", 0));        \n"
         "vec2 fcoord = fract(pos * size - vec2(0.5));  \n"
         "vec2 base = pos - pt * fcoord;                \n"
         "vec2 center = base + pt * vec2(0.5);          \n"
         "ivec2 offset;                                 \n"
         "float w, d, wsum = 0.0;                       \n"
         "int idx;                                      \n"
         "vec4 c;                                       \n",
         pos, pt, src_tex);

    bool use_ar = cfg.antiring > 0;
    if (use_ar) {
#pragma GLSL                                                                    \
        vec2 ww, cc;                                                            \
        @for (c : cmask)                                                        \
            vec2 ar@c = vec2(0.0), wwsum@c = vec2(0.0);
    }

    int bound   = ceil(obj->filter->radius);
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

    int num_comps = __builtin_popcount(cmask);
    int shmem_req = (sizew * sizeh * num_comps + 2) * sizeof(float);
    bool is_compute = !params->no_compute && sh_glsl(sh).compute &&
                      sh_try_compute(sh, bw, bh, false, shmem_req);

    // Note: SH_LUT_LITERAL might be faster in some specific cases, but not by
    // much, and it's catastrophically slow on other platforms.
    ident_t lut = sh_lut(sh, sh_lut_params(
        .object     = &obj->lut,
        .lut_type   = SH_LUT_TEXTURE,
        .var_type   = PL_VAR_FLOAT,
        .method     = SH_LUT_LINEAR,
        .width      = SCALER_LUT_SIZE,
        .comps      = 1,
        .update     = update,
        .fill       = fill_polar_lut,
        .priv       = obj,
    ));

    if (!lut) {
        SH_FAIL(sh, "Failed initializing polar LUT!");
        return false;
    }

    ident_t radius_c = sh_const_float(sh, "radius", obj->filter->radius);
    ident_t in = sh_fresh(sh, "in");

    if (is_compute) {

        // Compute shader kernel
        GLSL("uvec2 base_id = uvec2(0u); \n");
        if (src->rect.x0 > src->rect.x1)
            GLSL("base_id.x = gl_WorkGroupSize.x - 1u; \n");
        if (src->rect.y0 > src->rect.y1)
            GLSL("base_id.y = gl_WorkGroupSize.y - 1u; \n");

        GLSLH("shared vec2 "$"_base; \n", in);
        GLSL("if (gl_LocalInvocationID.xy == base_id)               \n"
             "    "$"_base = base;                                  \n"
             "barrier();                                            \n"
             "ivec2 rel = ivec2(round((base - "$"_base) * size));   \n",
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
        GLSL("for (int y = int(gl_LocalInvocationID.y); y < "$"; y += %d) {     \n"
             "for (int x = int(gl_LocalInvocationID.x); x < "$"; x += %d) {     \n"
             "c = textureLod("$", "$"_base + pt * vec2(x - %d, y - %d), 0.0);   \n",
             ih_c, bh, iw_c, bw, src_tex, in, offset, offset);

        for (uint8_t comps = cmask; comps;) {
            uint8_t c = __builtin_ctz(comps);
            GLSLH("shared float "$"_%d["$" * "$"]; \n", in, c, sizeh_c, sizew_c);
            GLSL(""$"_%d["$" * y + x] = c[%d]; \n", in, c, sizew_c, c);
            comps &= ~(1 << c);
        }

        GLSL("}}                     \n"
             "barrier();             \n");

        // Dispatch the actual samples
        for (int y = 1 - bound; y <= bound; y++) {
            for (int x = 1 - bound; x <= bound; x++) {
                GLSL("idx = "$" * rel.y + rel.x + "$" * %d + %d; \n",
                     sizew_c, sizew_c, y + offset, x + offset);
                polar_sample(sh, obj->filter, src_tex, lut, radius_c,
                             x, y, cmask, in, use_ar, scale);
            }
        }
    } else {
        // Fragment shader sampling
        for (uint8_t comps = cmask; comps;) {
            uint8_t c = __builtin_ctz(comps);
            GLSL("vec4 "$"_%d; \n", in, c);
            comps &= ~(1 << c);
        }

        // For maximum efficiency, we want to use textureGather() if
        // possible, rather than direct sampling. Since this is not
        // always possible/sensible, we need to possibly intermix gathering
        // with regular sampling. This requires keeping track of which
        // pixels in the next row were already gathered by the previous
        // row.
        uint32_t gathered_cur = 0x0, gathered_next = 0x0;
        const float radius2 = PL_SQUARE(obj->filter->radius);
        const int base = bound - 1;

        if (base + bound >= 8 * sizeof(gathered_cur)) {
            SH_FAIL(sh, "Polar radius %f exceeds implementation capacity!",
                    obj->filter->radius);
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
                if (cmask != 0x1 || !src->tex)
                    use_gather &= sh_glsl(sh).version >= 400;

                if (!use_gather) {
                    // Switch to direct sampling instead
                    polar_sample(sh, obj->filter, src_tex, lut, radius_c,
                                 x, y, cmask, NULL_IDENT, use_ar, scale);
                    continue;
                }

                // Gather the four surrounding texels simultaneously
                for (uint8_t comps = cmask; comps;) {
                    uint8_t c = __builtin_ctz(comps);
                    if (x || y) {
                        if (c) {
                            GLSL($"_%d = textureGatherOffset("$", "
                                 "center, ivec2(%d, %d), %d); \n",
                                 in, c, src_tex, x, y, c);
                        } else {
                            GLSL($"_0 = textureGatherOffset("$", "
                                 "center, ivec2(%d, %d)); \n",
                                 in, src_tex, x, y);
                        }
                    } else {
                        if (c) {
                            GLSL($"_%d = textureGather("$", center, %d); \n",
                                 in, c, src_tex, c);
                        } else {
                            GLSL($"_0 = textureGather("$", center); \n",
                                 in, src_tex);
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
                    polar_sample(sh, obj->filter, src_tex, lut, radius_c,
                                 x+xo[p], y+yo[p], cmask, in, use_ar, scale);
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

#pragma GLSL                                                                    \
    color = $scale / wsum * color;                                              \
    @if (use_ar) {                                                              \
        @for (c : cmask) {                                                      \
            ww = ar@c / wwsum@c;                                                \
            ww.x = 1.0 - ww.x;                                                  \
            w = clamp(color[@c], ww.x, ww.y);                                   \
            w = mix(w, dot(ww, vec2(0.5)), ww.x > ww.y);                        \
            color[@c] = mix(color[@c], w, ${float:cfg.antiring});               \
        @}                                                                      \
    @}                                                                          \
    @if (!(cmask & (1 << PL_CHANNEL_A)))                                        \
        color.a = 1.0;                                                          \
    }

    return true;
}

static void fill_ortho_lut(void *data, const struct sh_lut_params *params)
{
    const struct sh_sampler_obj *obj = params->priv;
    pl_filter filt = obj->filter;

    if (filt->radius == filt->radius_zero) {
        // Main lobe covers entire radius, so all weights are positive, meaning
        // we can use the linear resampling trick
        for (int n = 0; n < SCALER_LUT_SIZE; n++) {
            const float *weights = filt->weights + n * filt->row_stride;
            float *row = (float *) data + n * filt->row_stride;
            pl_assert(filt->row_size % 2 == 0);
            for (int i = 0; i < filt->row_size; i += 2) {
                const float w0 = weights[i], w1 = weights[i+1];
                assert(w0 + w1 >= 0.0f);
                row[i] = w0 + w1;
                row[i+1] = w1 / (w0 + w1);
            }
        }
    } else {
        size_t entries = SCALER_LUT_SIZE * filt->row_stride;
        pl_assert(params->width * params->height * params->comps == entries);
        memcpy(data, filt->weights, entries * sizeof(float));
    }
}

enum {
    SEP_VERT = 0,
    SEP_HORIZ,
    SEP_PASSES
};

bool pl_shader_sample_ortho2(pl_shader sh, const struct pl_sample_src *src,
                             const struct pl_sample_filter_params *params)
{
    pl_assert(params);
    if (params->filter.polar) {
        SH_FAIL(sh, "Trying to use separated sampling with a polar filter?");
        return false;
    }

    pl_gpu gpu = SH_GPU(sh);
    pl_assert(gpu);

    uint8_t comps;
    float ratio[SEP_PASSES], scale;
    ident_t src_tex, pos, pt;
    if (!setup_src(sh, src, &src_tex, &pos, &pt,
                   &ratio[SEP_HORIZ], &ratio[SEP_VERT],
                   &comps, &scale, false, LINEAR))
        return false;


    int pass;
    if (fabs(ratio[SEP_HORIZ] - 1.0f) < 1e-6f) {
        pass = SEP_VERT;
    } else if (fabs(ratio[SEP_VERT] - 1.0f) < 1e-6f) {
        pass = SEP_HORIZ;
    } else {
        SH_FAIL(sh, "Trying to use pl_shader_sample_ortho with a "
                "pl_sample_src that requires scaling in multiple directions "
                "(rx=%f, ry=%f), this is not possible!",
                ratio[SEP_HORIZ], ratio[SEP_VERT]);
        return false;
    }

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

    struct pl_filter_config cfg = params->filter;
    cfg.antiring = PL_DEF(cfg.antiring, params->antiring);
    cfg.blur = PL_DEF(cfg.blur, 1.0f) * inv_scale;
    bool update = !obj->filter || !pl_filter_config_eq(&obj->filter->params.config, &cfg);

    if (update) {
        pl_filter_free(&obj->filter);
        obj->filter = pl_filter_generate(sh->log, pl_filter_params(
            .config             = cfg,
            .lut_entries        = SCALER_LUT_SIZE,
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
        .method     = SH_LUT_LINEAR,
        .width      = width,
        .height     = SCALER_LUT_SIZE,
        .comps      = 4,
        .update     = update,
        .fill       = fill_ortho_lut,
        .priv       = obj,
    ));
    if (!lut) {
        SH_FAIL(sh, "Failed initializing separated LUT!");
        return false;
    }

    const int dir[SEP_PASSES][2] = {
        [SEP_HORIZ] = {1, 0},
        [SEP_VERT]  = {0, 1},
    };

    static const char *names[SEP_PASSES] = {
        [SEP_HORIZ] = "ortho (horiz)",
        [SEP_VERT]  = "ortho (vert)",
    };

    describe_filter(sh, &cfg, names[pass], ratio[pass], ratio[pass]);

    float denom = PL_MAX(1, width - 1); // avoid division by zero
    bool use_ar = cfg.antiring > 0 && ratio[pass] > 1.0;
    bool use_linear = obj->filter->radius == obj->filter->radius_zero;
    use_ar &= !use_linear; // filter has no negative weights

#pragma GLSL /* pl_shader_sample_ortho */                                       \
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);                                      \
    {                                                                           \
    vec2 pos = $pos, pt = $pt;                                                  \
    vec2 size = vec2(textureSize($src_tex, 0));                                 \
    vec2 dir = vec2(${const float:dir[pass][0]}, ${const float: dir[pass][1]}); \
    pt *= dir;                                                                  \
    vec2 fcoord2 = fract(pos * size - vec2(0.5));                               \
    float fcoord = dot(fcoord2, dir);                                           \
    vec2 base = pos - fcoord * pt - pt * vec2(${const float: N / 2 - 1});       \
    vec4 ws;                                                                    \
    float off;                                                                  \
    ${vecType: comps} c, ca = ${vecType: comps}(0.0);                           \
    @if (use_ar) {                                                              \
        ${vecType: comps} hi = ${vecType: comps}(0.0);                          \
        ${vecType: comps} lo = ${vecType: comps}(1e9);                          \
    @}                                                                          \
    #pragma unroll 4                                                            \
    for (int n = 0; n < ${int: N}; n += ${const int: use_linear ? 2 : 1}) {     \
        if (n % 4 == 0)                                                         \
            ws = $lut(vec2(float(n / 4) / ${const float: denom}, fcoord));      \
        @if (use_linear) {                                                      \
            off = float(n) + ws[n % 4 + 1];                                     \
            c = textureLod($src_tex, base + pt * off, 0.0).${swizzle: comps};   \
        @} else {                                                               \
            c = textureLod($src_tex, base + pt * float(n), 0.0).${swizzle: comps}; \
        @}                                                                      \
        @if (use_ar) {                                                          \
            if (n == ${uint: N} / 2 - 1 || n == ${uint: N} / 2) {               \
                lo = min(lo, c);                                                \
                hi = max(hi, c);                                                \
            }                                                                   \
        @}                                                                      \
        ca += ws[n % 4] * c;                                                    \
    }                                                                           \
    @if (use_ar)                                                                \
        ca = mix(ca, clamp(ca, lo, hi), ${float: cfg.antiring});                \
    color.${swizzle: comps} = ${float: scale} * ca;                             \
    }

    return true;
}

const struct pl_distort_params pl_distort_default_params = { PL_DISTORT_DEFAULTS };

void pl_shader_distort(pl_shader sh, pl_tex src_tex, int out_w, int out_h,
                       const struct pl_distort_params *params)
{
    pl_assert(params);
    if (!sh_require(sh, PL_SHADER_SIG_NONE, out_w, out_h))
        return;

    const int src_w = src_tex->params.w, src_h = src_tex->params.h;
    float rx = 1.0f, ry = 1.0f;
    if (src_w > src_h) {
        ry = (float) src_h / src_w;
    } else {
        rx = (float) src_w / src_h;
    }

    // Map from texel coordinates [0,1]² to aspect-normalized representation
    const pl_transform2x2 tex2norm = {
        .mat.m = {
            { 2 * rx, 0 },
            { 0, -2 * ry },
        },
        .c = { -rx, ry },
    };

    // Map from aspect-normalized representation to canvas coords [-1,1]²
    const float sx = params->unscaled ? (float) src_w / out_w : 1.0f;
    const float sy = params->unscaled ? (float) src_h / out_h : 1.0f;
    const pl_transform2x2 norm2canvas = {
        .mat.m = {
            { sx / rx, 0 },
            { 0, sy / ry },
        },
    };

    struct pl_transform2x2 transform = params->transform;
    pl_transform2x2_mul(&transform, &tex2norm);
    pl_transform2x2_rmul(&norm2canvas, &transform);

    if (params->constrain) {
        pl_rect2df bb = pl_transform2x2_bounds(&transform, &(pl_rect2df) {
            .x1 = 1, .y1 = 1,
        });
        const float k = fmaxf(fmaxf(pl_rect_w(bb), pl_rect_h(bb)), 2.0f);
        pl_transform2x2_scale(&transform, 2.0f / k);
    };

    // Bind the canvas coordinates as [-1,1]², flipped vertically to correspond
    // to normal mathematical axis conventions
    static const pl_rect2df canvas = {
        .x0 = -1.0f, .x1 =  1.0f,
        .y0 =  1.0f, .y1 = -1.0f,
    };

    ident_t pos = sh_attr_vec2(sh, "pos", &canvas);
    ident_t pt, tex = sh_bind(sh, src_tex, params->address_mode,
                              PL_TEX_SAMPLE_LINEAR, "tex", NULL, NULL, &pt);

    // Bind the inverse of the tex2canvas transform (i.e. canvas2tex)
    pl_transform2x2_invert(&transform);
    ident_t tf = sh_var(sh, (struct pl_shader_var) {
        .var  = pl_var_mat2("tf"),
        .data = PL_TRANSPOSE_2X2(transform.mat.m),
    });

    ident_t tf_c = sh_var(sh, (struct pl_shader_var) {
        .var  = pl_var_vec2("tf_c"),
        .data = transform.c,
    });

    // See pl_shader_sample_bicubic
    sh_describe(sh, "distortion");
#pragma GLSL /* pl_shader_sample_distort */                 \
    vec4 color;                                             \
    {                                                       \
    vec2 pos = $tf * $pos + $tf_c;                          \
    vec2 pt = $pt;                                          \
    @if (params->bicubic) {                                 \
        vec2 size = vec2(textureSize($tex, 0));             \
        vec2 frac  = fract(pos * size + vec2(0.5));         \
        vec2 frac2 = frac * frac;                           \
        vec2 inv   = vec2(1.0) - frac;                      \
        vec2 inv2  = inv * inv;                             \
        vec2 w0 = 1.0/6.0 * inv2 * inv;                     \
        vec2 w1 = 2.0/3.0 - 0.5 * frac2 * (2.0 - frac);     \
        vec2 w2 = 2.0/3.0 - 0.5 * inv2  * (2.0 - inv);      \
        vec2 w3 = 1.0/6.0 * frac2 * frac;                   \
        vec4 g = vec4(w0 + w1, w2 + w3);                    \
        vec4 h = vec4(w1, w3) / g + inv.xyxy;               \
        h.xy -= vec2(2.0);                                  \
        vec4 p = pos.xyxy + pt.xyxy * h;                    \
        vec4 c00 = textureLod($tex, p.xy, 0.0);             \
        vec4 c01 = textureLod($tex, p.xw, 0.0);             \
        vec4 c0 = mix(c01, c00, g.y);                       \
        vec4 c10 = textureLod($tex, p.zy, 0.0);             \
        vec4 c11 = textureLod($tex, p.zw, 0.0);             \
        vec4 c1 = mix(c11, c10, g.y);                       \
        color = mix(c1, c0, g.x);                           \
    @} else {                                               \
        color = texture($tex, pos);                         \
    @}                                                      \
    @if (params->alpha_mode) {                              \
        vec2 border = min(pos, vec2(1.0) - pos);            \
        border = smoothstep(vec2(0.0), pt, border);         \
        @if (params->alpha_mode == PL_ALPHA_PREMULTIPLIED)  \
            color.rgba *= border.x * border.y;              \
        @else                                               \
            color.a *= border.x * border.y;                 \
    @}                                                      \
    }

}
