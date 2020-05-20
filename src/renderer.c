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
 * License along with libplacebo.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <math.h>

#include "common.h"
#include "shaders.h"
#include "dispatch.h"

enum {
    // The scalers for each plane are set up to be just the index itself
    SCALER_PLANE0 = 0,
    SCALER_PLANE1 = 1,
    SCALER_PLANE2 = 2,
    SCALER_PLANE3 = 3,

    SCALER_MAIN,
    SCALER_COUNT,
};

struct sampler {
    struct pl_shader_obj *upscaler_state;
    struct pl_shader_obj *downscaler_state;
    const struct pl_tex *sep_fbo_up;
    const struct pl_tex *sep_fbo_down;
};

struct pl_renderer {
    const struct pl_gpu *gpu;
    struct pl_context *ctx;
    struct pl_dispatch *dp;

    // Texture format to use for intermediate textures
    const struct pl_fmt *fbofmt;

    // Cached feature checks (inverted)
    bool disable_compute;       // disable the use of compute shaders
    bool disable_sampling;      // disable use of advanced scalers
    bool disable_debanding;     // disable the use of debanding shaders
    bool disable_linear_hdr;    // disable linear scaling for HDR signals
    bool disable_linear_sdr;    // disable linear scaling for SDR signals
    bool disable_blending;      // disable blending for the target/fbofmt
    bool disable_overlay;       // disable rendering overlays
    bool disable_3dlut;         // disable usage of a 3DLUT
    bool disable_peak_detect;   // disable peak detection shader
    bool disable_grain;         // disable AV1 grain code
    bool disable_hooks;         // disable user hooks / custom shaders

    // Shader resource objects and intermediate textures (FBOs)
    struct pl_shader_obj *peak_detect_state;
    struct pl_shader_obj *dither_state;
    struct pl_shader_obj *lut3d_state;
    struct pl_shader_obj *grain_state[4];
    const struct pl_tex **fbos;
    int num_fbos;
    struct sampler samplers[SCALER_COUNT];
    struct sampler *osd_samplers;
    int num_osd_samplers;
};

static void find_fbo_format(struct pl_renderer *rr)
{
    struct {
        enum pl_fmt_type type;
        int depth;
        enum pl_fmt_caps caps;
    } configs[] = {
        // Prefer floating point formats first
        {PL_FMT_FLOAT, 16, PL_FMT_CAP_LINEAR},
        {PL_FMT_FLOAT, 16, PL_FMT_CAP_SAMPLEABLE},

        // Otherwise, fall back to unorm/snorm, preferring linearly sampleable
        {PL_FMT_UNORM, 16, PL_FMT_CAP_LINEAR},
        {PL_FMT_SNORM, 16, PL_FMT_CAP_LINEAR},
        {PL_FMT_UNORM, 16, PL_FMT_CAP_SAMPLEABLE},
        {PL_FMT_SNORM, 16, PL_FMT_CAP_SAMPLEABLE},

        // As a final fallback, allow 8-bit FBO formats (for UNORM only)
        {PL_FMT_UNORM, 8, PL_FMT_CAP_LINEAR},
        {PL_FMT_UNORM, 8, PL_FMT_CAP_SAMPLEABLE},
    };

    for (int i = 0; i < PL_ARRAY_SIZE(configs); i++) {
        const struct pl_fmt *fmt;
        fmt = pl_find_fmt(rr->gpu, configs[i].type, 4, configs[i].depth, 0,
                          configs[i].caps | PL_FMT_CAP_RENDERABLE);
        if (fmt) {
            rr->fbofmt = fmt;
            break;
        }
    }

    if (!rr->fbofmt) {
        PL_WARN(rr, "Found no renderable FBO format! Most features disabled");
        return;
    }

    if (!(rr->fbofmt->caps & PL_FMT_CAP_STORABLE)) {
        PL_INFO(rr, "Found no storable FBO format; compute shaders disabled");
        rr->disable_compute = true;
    }

    if (rr->fbofmt->type != PL_FMT_FLOAT) {
        PL_INFO(rr, "Found no floating point FBO format; linear light "
                "processing disabled for HDR material");
        rr->disable_linear_hdr = true;
    }

    if (rr->fbofmt->component_depth[0] < 16) {
        PL_WARN(rr, "FBO format precision low (<16 bit); linear light "
                "processing disabled");
        rr->disable_linear_sdr = true;
    }
}

struct pl_renderer *pl_renderer_create(struct pl_context *ctx,
                                       const struct pl_gpu *gpu)
{
    struct pl_renderer *rr = talloc_ptrtype(NULL, rr);
    *rr = (struct pl_renderer) {
        .gpu  = gpu,
        .ctx = ctx,
        .dp  = pl_dispatch_create(ctx, gpu),
    };

    assert(rr->dp);
    find_fbo_format(rr);
    return rr;
}

static void sampler_destroy(struct pl_renderer *rr, struct sampler *sampler)
{
    pl_shader_obj_destroy(&sampler->upscaler_state);
    pl_shader_obj_destroy(&sampler->downscaler_state);
    pl_tex_destroy(rr->gpu, &sampler->sep_fbo_up);
    pl_tex_destroy(rr->gpu, &sampler->sep_fbo_down);
}

void pl_renderer_destroy(struct pl_renderer **p_rr)
{
    struct pl_renderer *rr = *p_rr;
    if (!rr)
        return;

    // Free all intermediate FBOs
    for (int i = 0; i < rr->num_fbos; i++)
        pl_tex_destroy(rr->gpu, &rr->fbos[i]);

    // Free all shader resource objects
    pl_shader_obj_destroy(&rr->peak_detect_state);
    pl_shader_obj_destroy(&rr->dither_state);
    pl_shader_obj_destroy(&rr->lut3d_state);
    for (int i = 0; i < PL_ARRAY_SIZE(rr->grain_state); i++)
        pl_shader_obj_destroy(&rr->grain_state[i]);

    // Free all samplers
    for (int i = 0; i < PL_ARRAY_SIZE(rr->samplers); i++)
        sampler_destroy(rr, &rr->samplers[i]);
    for (int i = 0; i < rr->num_osd_samplers; i++)
        sampler_destroy(rr, &rr->osd_samplers[i]);

    pl_dispatch_destroy(&rr->dp);
    TA_FREEP(p_rr);
}

void pl_renderer_flush_cache(struct pl_renderer *rr)
{
    pl_shader_obj_destroy(&rr->peak_detect_state);
}

const struct pl_render_params pl_render_default_params = {
    .upscaler           = &pl_filter_spline36,
    .downscaler         = &pl_filter_mitchell,
    .frame_mixer        = NULL,

    .sigmoid_params     = &pl_sigmoid_default_params,
    .peak_detect_params = &pl_peak_detect_default_params,
    .color_map_params   = &pl_color_map_default_params,
    .dither_params      = &pl_dither_default_params,
};

const struct pl_render_params pl_render_high_quality_params = {
    .upscaler           = &pl_filter_ewa_lanczos,
    .downscaler         = &pl_filter_mitchell,
    .frame_mixer        = NULL,

    .deband_params      = &pl_deband_default_params,
    .sigmoid_params     = &pl_sigmoid_default_params,
    .peak_detect_params = &pl_peak_detect_default_params,
    .color_map_params   = &pl_color_map_default_params,
    .dither_params      = &pl_dither_default_params,
};

#define FBOFMT (params->disable_fbos ? NULL : rr->fbofmt)

// Represents a "in-flight" image, which is either a shader that's in the
// process of producing some sort of image, or a texture that needs to be
// sampled from
struct img {
    // Effective texture size, always set
    int w, h;

    // Exactly *one* of these two is set:
    struct pl_shader *sh;
    const struct pl_tex *tex;

    // Current effective source area, will be sampled by the main scaler
    struct pl_rect2df rect;

    // The current effective colorspace
    struct pl_color_repr repr;
    struct pl_color_space color;
    int comps;
};

struct pass_state {
    void *tmp;

    // Pointer back to the renderer itself, for callbacks
    struct pl_renderer *rr;

    // Represents the "current" image which we're in the process of rendering.
    // This is initially set by pass_read_image, and all of the subsequent
    // rendering steps will mutate this in-place.
    struct img img;

    // Represents the "reference rect". Canonically, this is functionallity
    // equivalent to `pl_image.src_rect`, but both guaranteed to be valid, and
    // also updates as the refplane evolves (e.g. due to user hook prescalers)
    struct pl_rect2df ref_rect;

    // Cached copies of the `image` / `target` for this rendering pass,
    // corrected to make sure all rects etc. are properly defaulted/inferred.
    struct pl_image image;
    struct pl_render_target target;

    // Metadata for `rr->fbos`
    bool *fbos_used;
};

static const struct pl_tex *get_fbo(struct pass_state *pass, int w, int h)
{
    struct pl_renderer *rr = pass->rr;
    if (!rr->fbofmt)
        return NULL;

    struct pl_tex_params params = {
        .w = w,
        .h = h,
        .format = rr->fbofmt,
        .sampleable = true,
        .renderable = true,
        // Just enable what we can
        .storable   = !!(rr->fbofmt->caps & PL_FMT_CAP_STORABLE),
        .sample_mode = (rr->fbofmt->caps & PL_FMT_CAP_LINEAR)
                            ? PL_TEX_SAMPLE_LINEAR
                            : PL_TEX_SAMPLE_NEAREST,
    };

    int best_idx = -1;
    int best_diff = 0;

    // Find the best-fitting texture out of rr->fbos
    for (int i = 0; i < rr->num_fbos; i++) {
        if (pass->fbos_used[i])
            continue;

        // Orthogonal distance
        int diff = abs(rr->fbos[i]->params.w - w) +
                   abs(rr->fbos[i]->params.h - h);

        if (best_idx < 0 || diff < best_diff) {
            best_idx = i;
            best_diff = diff;
        }
    }

    // No texture found at all, add a new one
    if (best_idx < 0) {
        best_idx = rr->num_fbos;
        TARRAY_APPEND(rr, rr->fbos, rr->num_fbos, NULL);
        TARRAY_GROW(pass->tmp, pass->fbos_used, best_idx);
        pass->fbos_used[best_idx] = false;
    }

    if (!pl_tex_recreate(rr->gpu, &rr->fbos[best_idx], &params))
        return NULL;

    pass->fbos_used[best_idx] = true;
    return rr->fbos[best_idx];
}

// Forcibly convert an img to `tex`, dispatching where necessary
static const struct pl_tex *img_tex(struct pass_state *pass, struct img *img)
{
    if (img->tex) {
        pl_assert(!img->sh);
        return img->tex;
    }

    struct pl_renderer *rr = pass->rr;
    const struct pl_tex *tex = get_fbo(pass, img->w, img->h);
    if (!tex) {
        PL_ERR(rr, "Failed creating FBO texture! Disabling advanced rendering..");
        rr->fbofmt = NULL;
        pl_dispatch_abort(rr->dp, &img->sh);
        return NULL;
    }

    pl_assert(img->sh);
    bool ok = pl_dispatch_finish(rr->dp, &(struct pl_dispatch_params) {
        .shader = &img->sh,
        .target = tex,
    });

    if (!ok) {
        PL_ERR(rr, "Failed dispatching intermediate pass!");
        return NULL;
    }

    img->tex = tex;
    return img->tex;
}

// Forcibly convert an img to `sh`, sampling where necessary
static struct pl_shader *img_sh(struct pass_state *pass, struct img *img)
{
    if (img->sh) {
        pl_assert(!img->tex);
        return img->sh;
    }

    pl_assert(img->tex);
    img->sh = pl_dispatch_begin(pass->rr->dp);
    pl_shader_sample_direct(img->sh, &(struct pl_sample_src) {
        .tex = img->tex,
    });

    img->tex = NULL;
    return img->sh;
}

enum sampler_type {
    SAMPLER_DIRECT, // texture()'s built in sampling
    SAMPLER_BICUBIC, // fast bicubic scaling
    SAMPLER_COMPLEX, // complex custom filters
};

enum sampler_dir {
    SAMPLER_NOOP, // 1:1 scaling
    SAMPLER_UP,   // upscaling
    SAMPLER_DOWN, // downscaling
};

struct sampler_info {
    const struct pl_filter_config *config; // if applicable
    enum sampler_type type;
    enum sampler_dir dir;
};

static struct sampler_info sample_src_info(struct pl_renderer *rr,
                                           const struct pl_sample_src *src,
                                           const struct pl_render_params *params)
{
    struct sampler_info info;

    float rx = src->new_w / fabs(pl_rect_w(src->rect));
    float ry = src->new_h / fabs(pl_rect_h(src->rect));
    if (rx < 1.0 - 1e-6 || ry < 1.0 - 1e-6) {
        info.dir = SAMPLER_DOWN;
        info.config = params->downscaler;
    } else if (rx > 1.0 + 1e-6 || ry > 1.0 + 1e-6) {
        info.dir = SAMPLER_UP;
        info.config = params->upscaler;
    } else {
        info.dir = SAMPLER_NOOP;
        info.type = SAMPLER_DIRECT;
        info.config = NULL;
        return info;
    }

    if (!FBOFMT || rr->disable_sampling || !info.config) {
        info.type = SAMPLER_DIRECT;
    } else {
        info.type = SAMPLER_COMPLEX;

        // Try using faster replacements for GPU built-in scalers
        enum pl_tex_sample_mode sample_mode;
        if (src->tex) {
            sample_mode = src->tex->params.sample_mode;
        } else {
            sample_mode = rr->fbofmt->caps & PL_FMT_CAP_LINEAR;
        }

        bool is_linear = sample_mode == PL_TEX_SAMPLE_LINEAR;
        bool can_fast = info.config == params->upscaler ||
                        params->skip_anti_aliasing;

        if (can_fast && !params->disable_builtin_scalers) {
            if (is_linear && info.config == &pl_filter_bicubic)
                info.type = SAMPLER_BICUBIC;
            if (is_linear && info.config == &pl_filter_triangle)
                info.type = SAMPLER_DIRECT;
            if (!is_linear && info.config == &pl_filter_box)
                info.type = SAMPLER_DIRECT;
        }
    }

    return info;
}

static void dispatch_sampler(struct pass_state *pass, struct pl_shader *sh,
                             struct sampler *sampler,
                             const struct pl_render_params *params,
                             const struct pl_sample_src *src)
{
    if (!sampler)
        goto fallback;

    struct pl_renderer *rr = pass->rr;
    struct sampler_info info = sample_src_info(rr, src, params);
    struct pl_shader_obj **lut = NULL;
    const struct pl_tex **sep_fbo = NULL;
    switch (info.dir) {
    case SAMPLER_NOOP:
        goto fallback;
    case SAMPLER_DOWN:
        lut = &sampler->downscaler_state;
        sep_fbo = &sampler->sep_fbo_down;
        break;
    case SAMPLER_UP:
        lut = &sampler->upscaler_state;
        sep_fbo = &sampler->sep_fbo_up;
        break;
    }

    switch (info.type) {
    case SAMPLER_DIRECT:
        goto fallback;
    case SAMPLER_BICUBIC:
        pl_shader_sample_bicubic(sh, src);
        return;
    case SAMPLER_COMPLEX:
        break; // continue below
    }

    pl_assert(lut && sep_fbo);
    struct pl_sample_filter_params fparams = {
        .filter      = *info.config,
        .lut_entries = params->lut_entries,
        .cutoff      = params->polar_cutoff,
        .antiring    = params->antiringing_strength,
        .no_compute  = rr->disable_compute,
        .no_widening = params->skip_anti_aliasing,
        .lut         = lut,
    };

    bool ok;
    if (info.config->polar) {
        ok = pl_shader_sample_polar(sh, src, &fparams);
    } else {
        struct pl_shader *tsh = pl_dispatch_begin_ex(rr->dp, true);
        ok = pl_shader_sample_ortho(tsh, PL_SEP_VERT, src, &fparams);
        if (!ok) {
            pl_dispatch_abort(rr->dp, &tsh);
            goto done;
        }

        struct img img = {
            .sh = tsh,
            .w  = src->tex->params.w,
            .h  = src->new_h,
        };

        struct pl_sample_src src2 = *src;
        src2.tex = img_tex(pass, &img);
        src2.scale = 1.0;
        ok = src2.tex && pl_shader_sample_ortho(sh, PL_SEP_HORIZ, &src2, &fparams);
    }

done:
    if (!ok) {
        PL_ERR(rr, "Failed dispatching scaler.. disabling");
        rr->disable_sampling = true;
        goto fallback;
    }

    return;

fallback:
    // If all else fails, fall back to bilinear/nearest
    pl_shader_sample_direct(sh, src);
}

static void draw_overlays(struct pass_state *pass, const struct pl_tex *fbo,
                          const struct pl_overlay *overlays, int num,
                          struct pl_color_space color, bool use_sigmoid,
                          struct pl_transform2x2 *scale,
                          const struct pl_render_params *params)
{
    struct pl_renderer *rr = pass->rr;
    if (num <= 0 || rr->disable_overlay)
        return;

    enum pl_fmt_caps caps = fbo->params.format->caps;
    if (!rr->disable_blending && !(caps & PL_FMT_CAP_BLENDABLE)) {
        PL_WARN(rr, "Trying to draw an overlay to a non-blendable target. "
                "Alpha blending is disabled, results may be incorrect!");
        rr->disable_blending = true;
    }

    while (num > rr->num_osd_samplers) {
        TARRAY_APPEND(rr, rr->osd_samplers, rr->num_osd_samplers,
                      (struct sampler) {0});
    }

    for (int n = 0; n < num; n++) {
        const struct pl_overlay *ol = &overlays[n];
        const struct pl_plane *plane = &ol->plane;
        const struct pl_tex *tex = plane->texture;

        struct pl_rect2d rect = ol->rect;
        if (scale) {
            float v0[2] = { rect.x0, rect.y0 };
            float v1[2] = { rect.x1, rect.y1 };
            pl_transform2x2_apply(scale, v0);
            pl_transform2x2_apply(scale, v1);
            rect = (struct pl_rect2d) { v0[0], v0[1], v1[0], v1[1] };
        }

        struct pl_sample_src src = {
            .tex        = tex,
            .components = ol->mode == PL_OVERLAY_MONOCHROME ? 1 : plane->components,
            .new_w      = abs(pl_rect_w(rect)),
            .new_h      = abs(pl_rect_h(rect)),
            .rect = {
                -plane->shift_x,
                -plane->shift_y,
                tex->params.w - plane->shift_x,
                tex->params.h - plane->shift_y,
            },
        };

        struct sampler *sampler = &rr->osd_samplers[n];
        if (params->disable_overlay_sampling)
            sampler = NULL;

        struct pl_shader *sh = pl_dispatch_begin(rr->dp);
        dispatch_sampler(pass, sh, sampler, params, &src);

        GLSL("vec4 osd_color;\n");
        for (int c = 0; c < src.components; c++) {
            if (plane->component_mapping[c] < 0)
                continue;
            GLSL("osd_color[%d] = color[%d];\n", plane->component_mapping[c],
                 tex->params.format->sample_order[c]);
        }

        switch (ol->mode) {
        case PL_OVERLAY_NORMAL:
            GLSL("color = osd_color;\n");
            break;
        case PL_OVERLAY_MONOCHROME:
            GLSL("color.a = osd_color[0];\n");
            GLSL("color.rgb = %s;\n", sh_var(sh, (struct pl_shader_var) {
                .var  = pl_var_vec3("base_color"),
                .data = &ol->base_color,
                .dynamic = true,
            }));
            break;
        default: abort();
        }

        struct pl_color_repr repr = ol->repr;
        pl_shader_decode_color(sh, &repr, NULL);
        pl_shader_color_map(sh, params->color_map_params, ol->color, color,
                            NULL, false);

        if (use_sigmoid)
            pl_shader_sigmoidize(sh, params->sigmoid_params);

        static const struct pl_blend_params blend_params = {
            .src_rgb = PL_BLEND_SRC_ALPHA,
            .dst_rgb = PL_BLEND_ONE_MINUS_SRC_ALPHA,
            .src_alpha = PL_BLEND_ONE,
            .dst_alpha = PL_BLEND_ONE_MINUS_SRC_ALPHA,
        };

        const struct pl_blend_params *blend = &blend_params;
        if (rr->disable_blending)
            blend = NULL;

        bool ok = pl_dispatch_finish(rr->dp, &(struct pl_dispatch_params) {
            .shader = &sh,
            .target = fbo,
            .rect   = rect,
            .blend_params = blend,
        });

        if (!ok) {
            PL_ERR(rr, "Failed rendering overlay texture!");
            rr->disable_overlay = true;
            return;
        }
    }
}

static const struct pl_tex *get_hook_tex(void *priv, int width, int height)
{
    struct pass_state *pass = priv;

    return get_fbo(pass, width, height);
}

// Returns if any hook was applied (even if there were errors)
static bool pass_hook(struct pass_state *pass, struct img *img,
                      enum pl_hook_stage stage,
                      const struct pl_render_params *params)
{
    struct pl_renderer *rr = pass->rr;
    if (!rr->fbofmt || rr->disable_hooks)
        return false;

    bool ret = false;

    for (int n = 0; n < params->num_hooks; n++) {
        const struct pl_hook *hook = params->hooks[n];
        if (!(hook->stages & stage))
            continue;

        PL_TRACE(rr, "Dispatching hook %d stage 0x%x", n, stage);
        struct pl_hook_params hparams = {
            .gpu = rr->gpu,
            .dispatch = rr->dp,
            .get_tex = get_hook_tex,
            .priv = pass,
            .stage = stage,
            .rect = img->rect,
            .repr = img->repr,
            .color = img->color,
            .components = img->comps,
            .src_rect = pass->ref_rect,
            .dst_rect = pass->target.dst_rect,
        };

        // TODO: Add some sort of `test` API function to the hooks that allows
        // us to skip having to touch the `img` state at all for no-ops

        switch (hook->input) {
        case PL_HOOK_SIG_NONE:
            break;

        case PL_HOOK_SIG_TEX: {
            hparams.tex = img_tex(pass, img);
            if (!hparams.tex) {
                PL_ERR(rr, "Failed dispatching shader prior to hook!");
                goto error;
            }
            break;
        }

        case PL_HOOK_SIG_COLOR:
            hparams.sh = img_sh(pass, img);
            break;

        default: abort();
        }

        struct pl_hook_res res = hook->hook(hook->priv, &hparams);
        if (res.failed) {
            PL_ERR(rr, "Failed executing hook, disabling");
            goto error;
        }

        bool resizable = pl_hook_stage_resizable(stage);
        switch (res.output) {
        case PL_HOOK_SIG_NONE:
            break;

        case PL_HOOK_SIG_TEX:
            if (!resizable) {
                if (res.tex->params.w != img->w ||
                    res.tex->params.h != img->h ||
                    !pl_rect2d_eq(res.rect, img->rect))
                {
                    PL_ERR(rr, "User hook tried resizing non-resizable stage!");
                    goto error;
                }
            }

            *img = (struct img) {
                .tex = res.tex,
                .repr = res.repr,
                .color = res.color,
                .comps = res.components,
                .rect = res.rect,
                .w = res.tex->params.w,
                .h = res.tex->params.h,
            };
            break;

        case PL_HOOK_SIG_COLOR:
            if (!resizable) {
                if (res.sh->output_w != img->w ||
                    res.sh->output_h != img->h ||
                    !pl_rect2d_eq(res.rect, img->rect))
                {
                    PL_ERR(rr, "User hook tried resizing non-resizable stage!");
                    goto error;
                }
            }

            *img = (struct img) {
                .sh = res.sh,
                .repr = res.repr,
                .color = res.color,
                .comps = res.components,
                .rect = res.rect,
                .w = res.sh->output_w,
                .h = res.sh->output_h,
            };
            break;

        default: abort();
        }

        // a hook was performed successfully
        ret = true;
    }

    return ret;

error:
    rr->disable_hooks = true;

    // Make sure the state remains as valid as possible, even if the resulting
    // shaders might end up nonsensical, to prevent segfaults
    if (!img->tex && !img->sh)
        img->sh = pl_dispatch_begin(rr->dp);
    return ret;
}

// `deband_src` results
enum {
    DEBAND_NOOP = 0, // no debanding was performing
    DEBAND_NORMAL,   // debanding was performed, the plane should still be scaled
    DEBAND_SCALED,   // debanding took care of scaling as well
};

static int deband_src(struct pass_state *pass, struct pl_shader *psh,
                      struct pl_sample_src *psrc,
                      const struct pl_image *image,
                      const struct pl_render_params *params)
{
    struct pl_renderer *rr = pass->rr;
    if (rr->disable_debanding || !params->deband_params)
        return DEBAND_NOOP;

    if (psrc->tex->params.sample_mode != PL_TEX_SAMPLE_LINEAR) {
        PL_WARN(rr, "Debanding requires uploaded textures to be linearly "
                "sampleable (params.sample_mode = PL_TEX_SAMPLE_LINEAR)! "
                "Disabling debanding..");
        rr->disable_debanding = true;
        return DEBAND_NOOP;
    }

    // The debanding shader can replace direct GPU sampling
    bool deband_scales = sample_src_info(rr, psrc, params).type == SAMPLER_DIRECT;

    struct pl_shader *sh = psh;
    struct pl_sample_src *src = psrc;
    struct pl_sample_src fixed;
    if (!deband_scales) {
        // Only sample/deband the relevant cut-out, but round it to the nearest
        // integer to avoid doing fractional scaling
        fixed = *src;
        fixed.rect.x0 = floorf(fixed.rect.x0);
        fixed.rect.y0 = floorf(fixed.rect.y0);
        fixed.rect.x1 = ceilf(fixed.rect.x1);
        fixed.rect.y1 = ceilf(fixed.rect.y1);
        fixed.new_w = pl_rect_w(fixed.rect);
        fixed.new_h = pl_rect_h(fixed.rect);
        src = &fixed;

        if (fixed.new_w == psrc->new_w &&
            fixed.new_h == psrc->new_h &&
            pl_rect2d_eq(fixed.rect, psrc->rect))
        {
            // If there's nothing left to be done (i.e. we're already rendering
            // an exact integer crop without scaling), also skip the scalers
            deband_scales = true;
        } else {
            sh = pl_dispatch_begin_ex(rr->dp, true);
        }
    }

    // Divide the deband grain scale by the effective current colorspace nominal
    // peak, to make sure the output intensity of the grain is as independent
    // of the source as possible, even though it happens this early in the
    // process (well before any linearization / output adaptation)
    struct pl_deband_params dparams = *params->deband_params;
    float scale = pl_color_transfer_nominal_peak(image->color.transfer)
                * image->color.sig_scale;
    dparams.grain /= scale;

    pl_shader_deband(sh, src, &dparams);

    if (deband_scales)
        return DEBAND_SCALED;

    struct img img = {
        .sh = sh,
        .w  = src->new_w,
        .h  = src->new_h,
    };

    const struct pl_tex *new = img_tex(pass, &img);
    if (!new) {
        PL_ERR(rr, "Failed dispatching debanding shader.. disabling debanding!");
        rr->disable_debanding = true;
        return DEBAND_NOOP;
    }

    // Update the original pl_sample_src to point to the new texture
    psrc->tex = new;
    psrc->rect.x0 -= src->rect.x0;
    psrc->rect.y0 -= src->rect.y0;
    psrc->rect.x1 -= src->rect.x0;
    psrc->rect.y1 -= src->rect.y0;
    psrc->scale = 1.0;
    return DEBAND_NORMAL;
}

static void hdr_update_peak(struct pass_state *pass,
                            const struct pl_render_params *params)
{
    struct pl_renderer *rr = pass->rr;
    if (!params->peak_detect_params || !pl_color_space_is_hdr(pass->img.color))
        goto cleanup;

    if (rr->disable_compute || rr->disable_peak_detect)
        goto cleanup;

    if (!FBOFMT && !params->allow_delayed_peak_detect) {
        PL_WARN(rr, "Disabling peak detection because "
                "`allow_delayed_peak_detect` is false, but lack of FBOs "
                "forces the result to be delayed.");
        rr->disable_peak_detect = true;
        goto cleanup;
    }

    bool ok = pl_shader_detect_peak(img_sh(pass, &pass->img), pass->img.color,
                                    &rr->peak_detect_state,
                                    params->peak_detect_params);
    if (!ok) {
        PL_WARN(rr, "Failed creating HDR peak detection shader.. disabling");
        rr->disable_peak_detect = true;
        goto cleanup;
    }

    return;

cleanup:
    // No peak detection required or supported, so clean up the state to avoid
    // confusing it with later frames where peak detection is enabled again
    pl_shader_obj_destroy(&rr->peak_detect_state);
}

// Plane 'type', ordered by incrementing priority
enum plane_type {
    PLANE_ALPHA,
    PLANE_CHROMA,
    PLANE_LUMA,
    PLANE_RGB,
    PLANE_XYZ,
};

static const char *plane_type_names[] = {
    [PLANE_ALPHA]   = "alpha",
    [PLANE_CHROMA]  = "chroma",
    [PLANE_LUMA]    = "luma",
    [PLANE_RGB]     = "rgb",
    [PLANE_XYZ]     = "xyz",
};

struct plane_state {
    enum plane_type type;
    struct pl_plane plane;
    struct img img; // for per-plane shaders
};

static void log_plane_info(struct pl_renderer *rr, const struct plane_state *st)
{
    const struct pl_plane *plane = &st->plane;
    PL_TRACE(rr, "    Type: %s", plane_type_names[st->type]);

    switch (plane->components) {
    case 0:
        PL_TRACE(rr, "    Components: (none)");
        break;
    case 1:
        PL_TRACE(rr, "    Components: {%d}",
                 plane->component_mapping[0]);
        break;
    case 2:
        PL_TRACE(rr, "    Components: {%d %d}",
                 plane->component_mapping[0],
                 plane->component_mapping[1]);
        break;
    case 3:
        PL_TRACE(rr, "    Components: {%d %d %d}",
                 plane->component_mapping[0],
                 plane->component_mapping[1],
                 plane->component_mapping[2]);
        break;
    case 4:
        PL_TRACE(rr, "    Components: {%d %d %d %d}",
                 plane->component_mapping[0],
                 plane->component_mapping[1],
                 plane->component_mapping[2],
                 plane->component_mapping[3]);
        break;
    }

    PL_TRACE(rr, "    Rect: {%f %f} -> {%f %f}",
             st->img.rect.x0, st->img.rect.y0, st->img.rect.x1, st->img.rect.y1);

    PL_TRACE(rr, "    Bits: %d (used) / %d (sampled), shift %d",
             st->img.repr.bits.color_depth,
             st->img.repr.bits.sample_depth,
             st->img.repr.bits.bit_shift);
}

static enum plane_type detect_plane_type(const struct plane_state *st)
{
    const struct pl_plane *plane = &st->plane;

    if (pl_color_system_is_ycbcr_like(st->img.repr.sys)) {
        int t = -1;
        for (int c = 0; c < plane->components; c++) {
            switch (plane->component_mapping[c]) {
            case PL_CHANNEL_Y: t = PL_MAX(t, PLANE_LUMA); continue;
            case PL_CHANNEL_A: t = PL_MAX(t, PLANE_ALPHA); continue;

            case PL_CHANNEL_CB:
            case PL_CHANNEL_CR:
                t = PL_MAX(t, PLANE_CHROMA);
                continue;

            default: continue;
            }
        }

        pl_assert(t >= 0);
        return t;
    }

    // Extra test for exclusive / separated alpha plane
    if (plane->components == 1 && plane->component_mapping[0] == PL_CHANNEL_A)
        return PLANE_ALPHA;

    switch (st->img.repr.sys) {
    case PL_COLOR_SYSTEM_UNKNOWN: // fall through to RGB
    case PL_COLOR_SYSTEM_RGB: return PLANE_RGB;
    case PL_COLOR_SYSTEM_XYZ: return PLANE_XYZ;
    default: abort();
    }
}

// Returns true if grain was applied
static bool plane_av1_grain(struct pass_state *pass, int plane_idx,
                            struct plane_state *st,
                            const struct pl_tex *ref_tex,
                            const struct pl_image *image,
                            const struct pl_render_params *params)
{
    struct pl_renderer *rr = pass->rr;
    if (rr->disable_grain)
        return false;

    struct img *img = &st->img;
    struct pl_plane *plane = &st->plane;
    struct pl_color_repr repr = st->img.repr;
    struct pl_av1_grain_params grain_params = {
        .data = image->av1_grain,
        .luma_tex = ref_tex,
        .repr = &repr,
        .components = plane->components,
    };

    for (int c = 0; c < plane->components; c++)
        grain_params.component_mapping[c] = plane->component_mapping[c];

    if (!pl_needs_av1_grain(&grain_params))
        return false;

    if (!FBOFMT) {
        PL_ERR(rr, "AV1 grain required but no renderable format available.. "
              "disabling!");
        rr->disable_grain = true;
        return false;
    }

    grain_params.tex = img_tex(pass, img);
    if (!grain_params.tex)
        return false;

    struct pl_shader *sh = pl_dispatch_begin_ex(rr->dp, true);
    if (!pl_shader_av1_grain(sh, &rr->grain_state[plane_idx], &grain_params)) {
        pl_dispatch_abort(rr->dp, &sh);
        rr->disable_grain = true;
        return false;
    }

    if (!img_tex(pass, img)) {
        PL_ERR(rr, "Failed applying AV1 grain.. disabling!");
        rr->disable_grain = true;
        return false;
    }

    return true;
}

// Returns true if any user hooks were executed
static bool plane_user_hooks(struct pass_state *pass, struct plane_state *st,
                             const struct pl_render_params *params)
{
    static const enum pl_hook_stage plane_stages[] = {
        [PLANE_ALPHA]   = PL_HOOK_ALPHA_INPUT,
        [PLANE_CHROMA]  = PL_HOOK_CHROMA_INPUT,
        [PLANE_LUMA]    = PL_HOOK_LUMA_INPUT,
        [PLANE_RGB]     = PL_HOOK_RGB_INPUT,
        [PLANE_XYZ]     = PL_HOOK_XYZ_INPUT,
    };

    return pass_hook(pass, &st->img, plane_stages[st->type], params);
}

// This scales and merges all of the source images, and initializes pass->img.
static bool pass_read_image(struct pl_renderer *rr, struct pass_state *pass,
                            const struct pl_render_params *params)
{
    struct pl_image *image = &pass->image;
    struct pl_shader *sh = pl_dispatch_begin_ex(rr->dp, true);
    sh_require(sh, PL_SHADER_SIG_NONE, 0, 0);

    // Initialize the color to black
    const char *neutral = "0.0, 0.0, 0.0";
    if (pl_color_system_is_ycbcr_like(image->repr.sys))
        neutral = "0.0, 0.5, 0.5";

    GLSL("vec4 color = vec4(%s, 1.0);            \n"
         "// pass_read_image                     \n"
         "{                                      \n"
         "vec4 tmp;                              \n",
         neutral);

    // First of all, we have to pick a "reference" plane for alignment.
    // This should ideally be the plane that most closely matches the target
    // image size
    struct plane_state planes[4];
    struct plane_state *ref = NULL;

    // Do a first pass to figure out each plane's type and find the ref plane
    pl_assert(image->num_planes < PL_ARRAY_SIZE(planes));
    for (int i = 0; i < image->num_planes; i++) {
        struct plane_state *st = &planes[i];
        *st = (struct plane_state) {
            .plane = image->planes[i],
            .img = {
                .w = image->planes[i].texture->params.w,
                .h = image->planes[i].texture->params.h,
                .tex = image->planes[i].texture,
                .repr = image->repr,
                .color = image->color,
                .comps = image->planes[i].components,
            },
        };

        st->type = detect_plane_type(st);
        pl_assert(st->type >= 0);
        switch (st->type) {
        case PLANE_RGB:
        case PLANE_LUMA:
        case PLANE_XYZ:
            ref = st;
            break;
        default: break;
        }
    }

    if (!ref) {
        PL_ERR(rr, "Image contains no LUMA, RGB or XYZ planes?");
        return false;
    }

    // Original ref texture, even after preprocessing
    const struct pl_tex *ref_tex = ref->plane.texture;

    // At this point in time we can finally infer src_rect to ensure it's valid
    if (!pl_rect_w(image->src_rect)) {
        image->src_rect.x0 = 0;
        image->src_rect.x1 = ref_tex->params.w;
    }

    if (!pl_rect_h(image->src_rect)) {
        image->src_rect.y0 = 0;
        image->src_rect.y1 = ref_tex->params.h;
    }

    // Guess the source primaries based on resolution
    if (!image->color.primaries) {
        image->color.primaries = pl_color_primaries_guess(ref_tex->params.w,
                                                          ref_tex->params.h);
    }

    pass->ref_rect = image->src_rect;

    // Do a second pass to compute the rc of each plane
    for (int i = 0; i < image->num_planes; i++) {
        struct plane_state *st = &planes[i];
        float rx = ref_tex->params.w / st->plane.texture->params.w,
              ry = ref_tex->params.h / st->plane.texture->params.h;

        // Only accept integer scaling ratios. This accounts for the fact that
        // fractionally subsampled planes get rounded up to the nearest integer
        // size, which we want to discard.
        float rrx = rx >= 1 ? roundf(rx) : 1.0 / roundf(1.0 / rx),
              rry = ry >= 1 ? roundf(ry) : 1.0 / roundf(1.0 / ry);

        float sx = st->plane.shift_x,
              sy = st->plane.shift_y;

        st->img.rect = (struct pl_rect2df) {
            .x0 = image->src_rect.x0 / rrx - sx / rx,
            .y0 = image->src_rect.y0 / rry - sy / ry,
            .x1 = image->src_rect.x1 / rrx - sx / rx,
            .y1 = image->src_rect.y1 / rry - sy / ry,
        };

        if (st == ref) {
            // Make sure st->rc == src_rect
            pl_assert(rrx == 1 && rry == 1 && sx == 0 && sy == 0);
        }

        PL_TRACE(rr, "Plane %d:", i);
        log_plane_info(rr, st);

        // Perform AV1 grain synthesis if needed. Do this first because it
        // requires unmodified plane sizes, and also because it's closer to the
        // intent of the spec (which is to apply synthesis effectively during
        // decoding)

        if (plane_av1_grain(pass, i, st, ref_tex, image, params)) {
            PL_TRACE(rr, "After AV1 grain:");
            log_plane_info(rr, st);
        }

        if (plane_user_hooks(pass, st, params)) {
            PL_TRACE(rr, "After user hooks:");
            log_plane_info(rr, st);
        }
    }

    // For quality reasons, explicitly drop subpixel offsets from the ref rect
    // and re-add them as part of `pass->img.rect`, always rounding towards 0
    float off_x = ref->img.rect.x0 - truncf(ref->img.rect.x0),
          off_y = ref->img.rect.y0 - truncf(ref->img.rect.y0);

    bool has_alpha = false;
    for (int i = 0; i < image->num_planes; i++) {
        struct plane_state *st = &planes[i];
        const struct pl_plane *plane = &st->plane;

        float scale_x = pl_rect_w(st->img.rect) / pl_rect_w(ref->img.rect),
              scale_y = pl_rect_h(st->img.rect) / pl_rect_h(ref->img.rect);

        struct pl_sample_src src = {
            .tex        = img_tex(pass, &st->img),
            .components = plane->components,
            .scale      = pl_color_repr_normalize(&st->img.repr),
            .new_w      = ref->img.w,
            .new_h      = ref->img.h,
            .rect = {
                st->img.rect.x0 - scale_x * off_x,
                st->img.rect.y0 - scale_y * off_y,
                st->img.rect.x1 - scale_x * off_x,
                st->img.rect.y1 - scale_y * off_y,
            },
        };

        if (!src.tex)
            src.tex = plane->texture; // Sanity fallback

        PL_TRACE(rr, "Aligning plane %d: {%f %f %f %f} -> {%f %f %f %f}",
                i, st->img.rect.x0, st->img.rect.y0,
                st->img.rect.x1, st->img.rect.y1,
                src.rect.x0, src.rect.y0,
                src.rect.x1, src.rect.y1);

        struct pl_shader *psh = pl_dispatch_begin_ex(rr->dp, true);
        if (deband_src(pass, psh, &src, image, params) != DEBAND_SCALED)
            dispatch_sampler(pass, psh, &rr->samplers[i], params, &src);

        ident_t sub = sh_subpass(sh, psh);
        if (!sub) {
            // Can't merge shaders, so instead force FBO indirection here
            struct img inter_img = {
                .sh = psh,
                .w = st->img.w,
                .h = st->img.h,
            };

            const struct pl_tex *inter_tex = img_tex(pass, &inter_img);
            if (!inter_tex) {
                PL_ERR(rr, "Failed dispatching subpass for plane.. disabling "
                       "all plane shaders");
                rr->disable_sampling = true;
                rr->disable_debanding = true;
                rr->disable_grain = true;
                pl_dispatch_abort(rr->dp, &sh);
                return false;
            }

            psh = pl_dispatch_begin_ex(rr->dp, true);
            pl_shader_sample_direct(psh, &(struct pl_sample_src) {
                .tex = inter_tex,
            });

            sub = sh_subpass(sh, psh);
            pl_assert(sub);
        }

        GLSL("tmp = %s();\n", sub);
        for (int c = 0; c < src.components; c++) {
            if (plane->component_mapping[c] < 0)
                continue;
            GLSL("color[%d] = tmp[%d];\n", plane->component_mapping[c],
                 src.tex->params.format->sample_order[c]);

            has_alpha |= plane->component_mapping[c] == PL_CHANNEL_A;
        }

        // we don't need it anymore
        pl_dispatch_abort(rr->dp, &psh);
    }

    GLSL("}\n");

    pass->img = (struct img) {
        .sh     = sh,
        .w      = ref->img.w,
        .h      = ref->img.h,
        .repr   = ref->img.repr,
        .color  = image->color,
        .comps  = has_alpha ? 4 : 3,
        .rect   = {
            off_x,
            off_y,
            off_x + pl_rect_w(ref->img.rect),
            off_y + pl_rect_h(ref->img.rect),
        },
    };

    // Update the reference rect to our adjusted image coordinates
    pass->ref_rect = pass->img.rect;

    pass_hook(pass, &pass->img, PL_HOOK_NATIVE,  params);

    // Convert the image colorspace
    pl_shader_decode_color(img_sh(pass, &pass->img), &pass->img.repr,
                           params->color_adjustment);

    pass_hook(pass, &pass->img, PL_HOOK_RGB, params);

    // HDR peak detection, do this as early as possible
    hdr_update_peak(pass, params);
    return true;
}

static bool pass_scale_main(struct pl_renderer *rr, struct pass_state *pass,
                            const struct pl_render_params *params)
{
    if (!FBOFMT) {
        PL_TRACE(rr, "Skipping main scaler (no FBOs)");
        return true;
    }

    const struct pl_image *image = &pass->image;
    const struct pl_render_target *target = &pass->target;

    struct img *img = &pass->img;
    struct pl_sample_src src = {
        .components = img->comps,
        .new_w      = abs(pl_rect_w(target->dst_rect)),
        .new_h      = abs(pl_rect_h(target->dst_rect)),
        .rect       = img->rect,
    };

    bool need_fbo = image->num_overlays > 0;
    need_fbo |= rr->peak_detect_state && !params->allow_delayed_peak_detect;

    struct sampler_info info = sample_src_info(rr, &src, params);
    bool use_sigmoid = info.dir == SAMPLER_UP && params->sigmoid_params;
    bool use_linear  = use_sigmoid || info.dir == SAMPLER_DOWN;

    // We need to enable the full rendering pipeline if there are any user
    // shaders / hooks that might depend on it.
    uint64_t scaling_hooks = PL_HOOK_PRE_OVERLAY | PL_HOOK_PRE_KERNEL |
                             PL_HOOK_POST_KERNEL;
    uint64_t linear_hooks = PL_HOOK_LINEAR | PL_HOOK_SIGMOID;

    for (int i = 0; i < params->num_hooks; i++) {
        if (params->hooks[i]->stages & (scaling_hooks | linear_hooks)) {
            need_fbo = true;
            if (params->hooks[i]->stages & linear_hooks)
                use_linear = true;
            if (params->hooks[i]->stages & PL_HOOK_SIGMOID)
                use_sigmoid = true;
        }
    }

    if (info.dir == SAMPLER_NOOP && !need_fbo) {
        pl_assert(src.new_w == img->w && src.new_h == img->h);
        PL_TRACE(rr, "Skipping main scaler (would be no-op)");
        return true;
    }

    if (info.type == SAMPLER_DIRECT && !need_fbo) {
        PL_TRACE(rr, "Skipping main scaler (free sampling)");
        return true;
    }

    // Hard-disable both sigmoidization and linearization when required
    if (params->disable_linear_scaling || rr->disable_linear_sdr)
        use_sigmoid = use_linear = false;

    // Avoid sigmoidization for HDR content because it clips to [0,1]
    if (pl_color_transfer_is_hdr(img->color.transfer)) {
        use_sigmoid = false;
        // Also disable linearization if necessary
        if (rr->disable_linear_hdr)
            use_linear = false;
    }

    if (use_linear) {
        pl_shader_linearize(img_sh(pass, img), img->color.transfer);
        img->color.transfer = PL_COLOR_TRC_LINEAR;
        pass_hook(pass, img, PL_HOOK_LINEAR, params);
    }

    if (use_sigmoid) {
        pl_shader_sigmoidize(img_sh(pass, img), params->sigmoid_params);
        pass_hook(pass, img, PL_HOOK_SIGMOID, params);
    }

    pass_hook(pass, img, PL_HOOK_PRE_OVERLAY, params);

    img->tex = img_tex(pass, img);
    if (!img->tex)
        return false;

    // Draw overlay on top of the intermediate image if needed, accounting
    // for possible stretching needed due to mismatch between the ref and src
    struct pl_transform2x2 tf = pl_transform2x2_identity;
    if (!pl_rect2d_eq(img->rect, image->src_rect)) {
        float rx = pl_rect_w(img->rect) / pl_rect_w(image->src_rect),
              ry = pl_rect_w(img->rect) / pl_rect_w(image->src_rect);

        tf = (struct pl_transform2x2) {
            .mat = {{{ rx, 0.0 }, { 0.0, ry }}},
            .c = {
                img->rect.x0 - image->src_rect.x0 * rx,
                img->rect.y0 - image->src_rect.y0 * ry
            },
        };
    }

    draw_overlays(pass, img->tex, image->overlays, image->num_overlays,
                  img->color, use_sigmoid, &tf, params);

    pass_hook(pass, img, PL_HOOK_PRE_KERNEL, params);

    src.tex = img_tex(pass, img);
    struct pl_shader *sh = pl_dispatch_begin_ex(rr->dp, true);
    dispatch_sampler(pass, sh, &rr->samplers[SCALER_MAIN], params, &src);
    *img = (struct img) {
        .sh     = sh,
        .w      = src.new_w,
        .h      = src.new_h,
        .repr   = img->repr,
        .rect   = { 0, 0, src.new_w, src.new_h },
        .color  = img->color,
        .comps  = img->comps,
    };

    pass_hook(pass, img, PL_HOOK_POST_KERNEL, params);

    if (use_sigmoid)
        pl_shader_unsigmoidize(img_sh(pass, img), params->sigmoid_params);

    pass_hook(pass, img, PL_HOOK_SCALED, params);
    return true;
}

static bool pass_output_target(struct pl_renderer *rr, struct pass_state *pass,
                               const struct pl_render_params *params)
{
    const struct pl_image *image = &pass->image;
    const struct pl_render_target *target = &pass->target;
    const struct pl_tex *fbo = target->fbo;
    struct img *img = &pass->img;
    struct pl_shader *sh = img_sh(pass, img);

    // Color management
    bool prelinearized = false;
    struct pl_color_space ref = image->color;
    assert(ref.primaries == pass->img.color.primaries);
    assert(ref.light == pass->img.color.light);
    if (pass->img.color.transfer == PL_COLOR_TRC_LINEAR)
        prelinearized = true;

    bool use_3dlut = image->profile.data || target->profile.data ||
                     params->force_3dlut;
    if (rr->disable_3dlut)
        use_3dlut = false;

#ifdef PL_HAVE_LCMS

    if (use_3dlut) {
        struct pl_3dlut_profile src = {
            .color = ref,
            .profile = image->profile,
        };

        struct pl_3dlut_profile dst = {
            .color = target->color,
            .profile = target->profile,
        };

        struct pl_3dlut_result res;
        bool ok = pl_3dlut_update(sh, &src, &dst, &rr->lut3d_state, &res,
                                  params->lut3d_params);
        if (!ok) {
            rr->disable_3dlut = true;
            use_3dlut = false;
            goto fallback;
        }

        // current -> 3DLUT in
        pl_shader_color_map(sh, params->color_map_params, ref, res.src_color,
                            &rr->peak_detect_state, prelinearized);
        // 3DLUT in -> 3DLUT out
        pl_3dlut_apply(sh, &rr->lut3d_state);
        // 3DLUT out -> target
        pl_shader_color_map(sh, params->color_map_params, res.dst_color,
                            target->color, NULL, false);
    }

fallback:

#else // !PL_HAVE_LCMS

    if (use_3dlut) {
        PL_WARN(rr, "An ICC profile was set, but libplacebo is built without "
                "support for LittleCMS! Disabling..");
        rr->disable_3dlut = true;
        use_3dlut = false;
    }

#endif

    if (!use_3dlut) {
        // current -> target
        pl_shader_color_map(sh, params->color_map_params, ref, target->color,
                            &rr->peak_detect_state, prelinearized);
    }

    // Apply color blindness simulation if requested
    if (params->cone_params)
        pl_shader_cone_distort(sh, target->color, params->cone_params);

    bool is_comp = pl_shader_is_compute(sh);
    if (is_comp && !fbo->params.storable) {
        if (!img_tex(pass, &pass->img)) {
            PL_ERR(rr, "Failed dispatching compute shader to intermediate FBO?");
            return false;
        }
    }

    pl_shader_encode_color(img_sh(pass, img), &target->repr);
    pass_hook(pass, &pass->img, PL_HOOK_OUTPUT, params);
    // FIXME: What if this ends up being a compute shader?? Should we do the
    // is_compute unredirection *after* encode_color, or will that fuck up
    // the bit depth?

    // FIXME: Technically we should try dithering before bit shifting if we're
    // going to be encoding to a low bit depth, since the caller might end up
    // discarding the extra bits. Ideally, we would pull the `bit_shift` out
    // of the `target->repr` and apply it separately after dithering.

    if (params->dither_params) {
        // Just assume the first component's depth is canonical. This works
        // in practice, since for cases like rgb565 we want to use the lower
        // depth anyway. Plus, every format has at least one component.
        int fmt_depth = fbo->params.format->component_depth[0];
        int depth = PL_DEF(target->repr.bits.sample_depth, fmt_depth);

        // Ignore dithering for >16-bit FBOs, since it's pretty pointless
        if (depth <= 16 || params->force_dither) {
            pl_shader_dither(img_sh(pass, img), depth, &rr->dither_state,
                             params->dither_params);
        }
    }

    sh = img_sh(pass, img);
    pl_assert(fbo->params.renderable);
    bool ok = pl_dispatch_finish(rr->dp, &(struct pl_dispatch_params) {
        .shader = &sh,
        .target = fbo,
        .rect   = target->dst_rect,
    });

    *img = (struct img) {0};
    return ok;
}

static void fix_rects(struct pl_image *image, struct pl_render_target *target)
{
    if ((!target->dst_rect.x0 && !target->dst_rect.x1) ||
        (!target->dst_rect.y0 && !target->dst_rect.y1))
    {
        target->dst_rect = (struct pl_rect2d) {
            0, 0, target->fbo->params.w, target->fbo->params.h,
        };
    }

    // We always want to prefer flipping in the dst_rect over flipping in
    // the src_rect. They're functionally equivalent either way.
    if (image->src_rect.x0 > image->src_rect.x1) {
        PL_SWAP(image->src_rect.x0, image->src_rect.x1);
        PL_SWAP(target->dst_rect.x0, target->dst_rect.x1);
    }

    if (image->src_rect.y0 > image->src_rect.y1) {
        PL_SWAP(image->src_rect.y0, image->src_rect.y1);
        PL_SWAP(target->dst_rect.y0, target->dst_rect.y1);
    }
}

bool pl_render_image(struct pl_renderer *rr, const struct pl_image *pimage,
                     const struct pl_render_target *ptarget,
                     const struct pl_render_params *params)
{
    params = PL_DEF(params, &pl_render_default_params);

    struct pass_state pass = {
        .tmp = talloc_new(NULL),
        .rr = rr,
        .image = *pimage,
        .target = *ptarget,
    };

    pass.fbos_used = talloc_zero_array(pass.tmp, bool, rr->num_fbos);

    struct pl_image *image = &pass.image;
    struct pl_render_target *target = &pass.target;

    fix_rects(image, target);
    pl_color_space_infer(&image->color);
    pl_color_space_infer(&target->color);

    // As a special case, don't infer the image primaries just yet, since
    // that's done in a resolution-dependent way in pass_read_image
    if (!pimage->color.primaries)
        image->color.primaries = PL_COLOR_PRIM_UNKNOWN;

    // TODO: output caching
    pl_dispatch_reset_frame(rr->dp);

    for (int i = 0; i < params->num_hooks; i++) {
        if (params->hooks[i]->reset)
            params->hooks[i]->reset(params->hooks[i]->priv);
    }

    if (!pass_read_image(rr, &pass, params))
        goto error;

    if (!pass_scale_main(rr, &pass, params))
        goto error;

    if (!pass_output_target(rr, &pass, params))
        goto error;

    // If we don't have FBOs available, simulate the on-image overlays at
    // this stage
    if (image->num_overlays > 0 && !FBOFMT) {
        float rx = pl_rect_w(target->dst_rect) / pl_rect_w(image->src_rect),
              ry = pl_rect_h(target->dst_rect) / pl_rect_h(image->src_rect);

        struct pl_transform2x2 scale = {
            .mat = {{{ rx, 0.0 }, { 0.0, ry }}},
            .c = {
                target->dst_rect.x0 - image->src_rect.x0 * rx,
                target->dst_rect.y0 - image->src_rect.y0 * ry
            },
        };

        draw_overlays(&pass, target->fbo, image->overlays, image->num_overlays,
                      target->color, false, &scale, params);
    }

    // Draw the final output overlays
    draw_overlays(&pass, target->fbo, target->overlays, target->num_overlays,
                  target->color, false, NULL, params);

    talloc_free(pass.tmp);
    return true;

error:
    pl_dispatch_abort(rr->dp, &pass.img.sh);
    talloc_free(pass.tmp);
    PL_ERR(rr, "Failed rendering image!");
    return false;
}

void pl_render_target_from_swapchain(struct pl_render_target *out_target,
                                     const struct pl_swapchain_frame *frame)
{
    const struct pl_tex *fbo = frame->fbo;
    *out_target = (struct pl_render_target) {
        .fbo = fbo,
        .dst_rect = { 0, 0, fbo->params.w, fbo->params.h },
        .repr = frame->color_repr,
        .color = frame->color_space,
    };

    if (frame->flipped)
        PL_SWAP(out_target->dst_rect.y0, out_target->dst_rect.y1);
}
