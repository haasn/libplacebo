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
#include "filters.h"
#include "shaders.h"
#include "dispatch.h"

struct cached_frame {
    uint64_t signature;
    uint64_t params_hash; // for detecting `pl_render_params` changes
    struct pl_color_space color;
    struct pl_icc_profile profile;
    pl_tex tex;
    int comps;
    bool evict; // for garbage collection
};

struct sampler {
    pl_shader_obj upscaler_state;
    pl_shader_obj downscaler_state;
};

struct osd_vertex {
    float pos[2];
    float coord[2];
    float color[4];
};

struct pl_renderer {
    pl_gpu gpu;
    pl_dispatch dp;
    pl_log log;

    // Cached feature checks (inverted)
    bool disable_fbos;          // disable the use of FBOs (skip fbofmt probing)
    bool disable_sampling;      // disable use of advanced scalers
    bool disable_debanding;     // disable the use of debanding shaders
    bool disable_blending;      // disable blending for the target/fbofmt
    bool disable_overlay;       // disable rendering overlays
    bool disable_icc;           // disable usage of ICC profiles
    bool disable_peak_detect;   // disable peak detection shader
    bool disable_grain;         // disable film grain code
    bool disable_hooks;         // disable user hooks / custom shaders
    bool disable_mixing;        // disable frame mixing

    // Shader resource objects and intermediate textures (FBOs)
    pl_shader_obj tone_map_state;
    pl_shader_obj dither_state;
    pl_shader_obj icc_state;
    pl_shader_obj grain_state[4];
    pl_shader_obj lut_state[3];
    PL_ARRAY(pl_tex) fbos;
    struct sampler sampler_main;
    struct sampler samplers_src[4];
    struct sampler samplers_dst[4];
    bool peak_detect_active;

    // Temporary storage for vertex/index data
    PL_ARRAY(struct osd_vertex) osd_vertices;
    PL_ARRAY(uint16_t) osd_indices;
    struct pl_vertex_attrib osd_attribs[3];

    // Frame cache (for frame mixing / interpolation)
    PL_ARRAY(struct cached_frame) frames;
    PL_ARRAY(pl_tex) frame_fbos;
};

enum {
    // Index into `lut_state`
    LUT_IMAGE,
    LUT_TARGET,
    LUT_PARAMS,
};

pl_renderer pl_renderer_create(pl_log log, pl_gpu gpu)
{
    pl_renderer rr = pl_alloc_ptr(NULL, rr);
    *rr = (struct pl_renderer) {
        .gpu  = gpu,
        .log = log,
        .dp  = pl_dispatch_create(log, gpu),
        .osd_attribs = {
            {
                .name = "pos",
                .offset = offsetof(struct osd_vertex, pos),
                .fmt = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 2),
            }, {
                .name = "coord",
                .offset = offsetof(struct osd_vertex, coord),
                .fmt = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 2),
            }, {
                .name = "osd_color",
                .offset = offsetof(struct osd_vertex, color),
                .fmt = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 4),
            }
        },
    };

    assert(rr->dp);
    return rr;
}

static void sampler_destroy(pl_renderer rr, struct sampler *sampler)
{
    pl_shader_obj_destroy(&sampler->upscaler_state);
    pl_shader_obj_destroy(&sampler->downscaler_state);
}

void pl_renderer_destroy(pl_renderer *p_rr)
{
    pl_renderer rr = *p_rr;
    if (!rr)
        return;

    // Free all intermediate FBOs
    for (int i = 0; i < rr->fbos.num; i++)
        pl_tex_destroy(rr->gpu, &rr->fbos.elem[i]);
    for (int i = 0; i < rr->frames.num; i++)
        pl_tex_destroy(rr->gpu, &rr->frames.elem[i].tex);
    for (int i = 0; i < rr->frame_fbos.num; i++)
        pl_tex_destroy(rr->gpu, &rr->frame_fbos.elem[i]);

    // Free all shader resource objects
    pl_shader_obj_destroy(&rr->tone_map_state);
    pl_shader_obj_destroy(&rr->dither_state);
    pl_shader_obj_destroy(&rr->icc_state);
    for (int i = 0; i < PL_ARRAY_SIZE(rr->lut_state); i++)
        pl_shader_obj_destroy(&rr->lut_state[i]);
    for (int i = 0; i < PL_ARRAY_SIZE(rr->grain_state); i++)
        pl_shader_obj_destroy(&rr->grain_state[i]);

    // Free all samplers
    sampler_destroy(rr, &rr->sampler_main);
    for (int i = 0; i < PL_ARRAY_SIZE(rr->samplers_src); i++)
        sampler_destroy(rr, &rr->samplers_src[i]);
    for (int i = 0; i < PL_ARRAY_SIZE(rr->samplers_dst); i++)
        sampler_destroy(rr, &rr->samplers_dst[i]);

    pl_dispatch_destroy(&rr->dp);
    pl_free_ptr(p_rr);
}

size_t pl_renderer_save(pl_renderer rr, uint8_t *out_cache)
{
    return pl_dispatch_save(rr->dp, out_cache);
}

void pl_renderer_load(pl_renderer rr, const uint8_t *cache)
{
    pl_dispatch_load(rr->dp, cache);
}

void pl_renderer_flush_cache(pl_renderer rr)
{
    for (int i = 0; i < rr->frames.num; i++)
        pl_tex_destroy(rr->gpu, &rr->frames.elem[i].tex);
    rr->frames.num = 0;

    pl_reset_detected_peak(rr->tone_map_state);
    rr->peak_detect_active = false;
}

const struct pl_render_params pl_render_fast_params = { PL_RENDER_DEFAULTS };
const struct pl_render_params pl_render_default_params = {
    PL_RENDER_DEFAULTS
    .upscaler           = &pl_filter_spline36,
    .downscaler         = &pl_filter_mitchell,
    .sigmoid_params     = &pl_sigmoid_default_params,
    .dither_params      = &pl_dither_default_params,
};

const struct pl_render_params pl_render_high_quality_params = {
    PL_RENDER_DEFAULTS
    .upscaler           = &pl_filter_ewa_lanczos,
    .downscaler         = &pl_filter_mitchell,
    .sigmoid_params     = &pl_sigmoid_default_params,
    .peak_detect_params = &pl_peak_detect_default_params,
    .dither_params      = &pl_dither_default_params,
    .deband_params      = &pl_deband_default_params,
};

// This is only used as a sentinel, to use the GLSL implementation
static double oversample(const struct pl_filter_function *k, double x)
{
    pl_unreachable();
}

static const struct pl_filter_function oversample_kernel = {
    .weight = oversample,
    .tunable = {true},
    .params = {0.0},
};

const struct pl_filter_config pl_filter_oversample = {
    .kernel = &oversample_kernel,
};

const struct pl_filter_preset pl_frame_mixers[] = {
    { "none",           NULL,                       "No frame mixing" },
    { "oversample",     &pl_filter_oversample,      "Oversample (AKA SmoothMotion)" },
    { "mitchell_clamp", &pl_filter_mitchell_clamp,  "Cubic spline (clamped)" },
    {0}
};

const int pl_num_frame_mixers = PL_ARRAY_SIZE(pl_frame_mixers) - 1;

const struct pl_filter_preset pl_scale_filters[] = {
    {"none",                NULL,                   "Built-in sampling"},
    {"oversample",          &pl_filter_oversample,  "Oversample (Aspect-preserving NN)"},
    COMMON_FILTER_PRESETS,
    {0}
};

const int pl_num_scale_filters = PL_ARRAY_SIZE(pl_scale_filters) - 1;

// Represents a "in-flight" image, which is either a shader that's in the
// process of producing some sort of image, or a texture that needs to be
// sampled from
struct img {
    // Effective texture size, always set
    int w, h;

    // Recommended format (falls back to fbofmt otherwise), only for shaders
    pl_fmt fmt;

    // Exactly *one* of these two is set:
    pl_shader sh;
    pl_tex tex;

    // Current effective source area, will be sampled by the main scaler
    struct pl_rect2df rect;

    // The current effective colorspace
    struct pl_color_repr repr;
    struct pl_color_space color;
    int comps;
};

// Plane 'type', ordered by incrementing priority
enum plane_type {
    PLANE_INVALID = 0,
    PLANE_ALPHA,
    PLANE_CHROMA,
    PLANE_LUMA,
    PLANE_RGB,
    PLANE_XYZ,
};

static inline enum plane_type detect_plane_type(const struct pl_plane *plane,
                                                const struct pl_color_repr *repr)
{
    if (pl_color_system_is_ycbcr_like(repr->sys)) {
        int t = PLANE_INVALID;
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

        pl_assert(t);
        return t;
    }

    // Extra test for exclusive / separated alpha plane
    if (plane->components == 1 && plane->component_mapping[0] == PL_CHANNEL_A)
        return PLANE_ALPHA;

    switch (repr->sys) {
    case PL_COLOR_SYSTEM_UNKNOWN: // fall through to RGB
    case PL_COLOR_SYSTEM_RGB: return PLANE_RGB;
    case PL_COLOR_SYSTEM_XYZ: return PLANE_XYZ;

    // For the switch completeness check
    case PL_COLOR_SYSTEM_BT_601:
    case PL_COLOR_SYSTEM_BT_709:
    case PL_COLOR_SYSTEM_SMPTE_240M:
    case PL_COLOR_SYSTEM_BT_2020_NC:
    case PL_COLOR_SYSTEM_BT_2020_C:
    case PL_COLOR_SYSTEM_BT_2100_PQ:
    case PL_COLOR_SYSTEM_BT_2100_HLG:
    case PL_COLOR_SYSTEM_DOLBYVISION:
    case PL_COLOR_SYSTEM_YCGCO:
    case PL_COLOR_SYSTEM_COUNT:
        break;
    }

    pl_unreachable();
}

struct pass_state {
    void *tmp;
    pl_renderer rr;
    const struct pl_render_params *params;
    struct pl_render_info info; // for info callback

    // Represents the "current" image which we're in the process of rendering.
    // This is initially set by pass_read_image, and all of the subsequent
    // rendering steps will mutate this in-place.
    struct img img;

    // Represents the "reference rect". Canonically, this is functionally
    // equivalent to `image.crop`, but also updates as the refplane evolves
    // (e.g. due to user hook prescalers)
    struct pl_rect2df ref_rect;

    // Integer version of `target.crop`. Semantically identical.
    struct pl_rect2d dst_rect;

    // Logical end-to-end rotation
    pl_rotation rotation;

    // Cached copies of the `image` / `target` for this rendering pass,
    // corrected to make sure all rects etc. are properly defaulted/inferred.
    struct pl_frame image;
    struct pl_frame target;

    // Some extra plane metadata, inferred from `planes`
    enum plane_type src_type[4];
    int src_ref, dst_ref; // index into `planes`

    // Metadata for `rr->fbos`
    pl_fmt fbofmt[5];
    bool *fbos_used;

};

static void find_fbo_format(struct pass_state *pass)
{
    const struct pl_render_params *params = pass->params;
    pl_renderer rr = pass->rr;
    if (params->disable_fbos || rr->disable_fbos || pass->fbofmt[4])
        return;

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

    pl_fmt fmt = NULL;
    for (int i = 0; i < PL_ARRAY_SIZE(configs); i++) {
        if (params->force_low_bit_depth_fbos && configs[i].depth > 8)
            continue;

        fmt = pl_find_fmt(rr->gpu, configs[i].type, 4, configs[i].depth, 0,
                          PL_FMT_CAP_RENDERABLE | configs[i].caps);
        if (!fmt)
            continue;

        pass->fbofmt[4] = fmt;

        // Probe the right variant for each number of channels, falling
        // back to the next biggest format
        for (int c = 1; c < 4; c++) {
            pass->fbofmt[c] = pl_find_fmt(rr->gpu, configs[i].type, c,
                                        configs[i].depth, 0, fmt->caps);
            pass->fbofmt[c] = PL_DEF(pass->fbofmt[c], pass->fbofmt[c+1]);
        }
        return;
    }

    PL_WARN(rr, "Found no renderable FBO format! Most features disabled");
    rr->disable_fbos = true;
}

static void info_callback(void *priv, const struct pl_dispatch_info *dinfo)
{
    struct pass_state *pass = priv;
    const struct pl_render_params *params = pass->params;
    if (!params->info_callback)
        return;

    pass->info.pass = dinfo;
    params->info_callback(params->info_priv, &pass->info);
    if (pass->info.stage == PL_RENDER_STAGE_FRAME)
        pass->info.index++;
}

static pl_tex get_fbo(struct pass_state *pass, int w, int h, pl_fmt fmt,
                      int comps, pl_debug_tag debug_tag)
{
    pl_renderer rr = pass->rr;
    comps = PL_DEF(comps, 4);
    fmt = PL_DEF(fmt, pass->fbofmt[comps]);
    if (!fmt)
        return NULL;

    struct pl_tex_params params = {
        .w          = w,
        .h          = h,
        .format     = fmt,
        .sampleable = true,
        .renderable = true,
        .blit_src   = fmt->caps & PL_FMT_CAP_BLITTABLE,
        .storable   = fmt->caps & PL_FMT_CAP_STORABLE,
        .debug_tag  = debug_tag,
    };

    int best_idx = -1;
    int best_diff = 0;

    // Find the best-fitting texture out of rr->fbos
    for (int i = 0; i < rr->fbos.num; i++) {
        if (pass->fbos_used[i])
            continue;

        // Orthogonal distance, with penalty for format mismatches
        int diff = abs(rr->fbos.elem[i]->params.w - w) +
                   abs(rr->fbos.elem[i]->params.h - h) +
                   ((rr->fbos.elem[i]->params.format != fmt) ? 1000 : 0);

        if (best_idx < 0 || diff < best_diff) {
            best_idx = i;
            best_diff = diff;
        }
    }

    // No texture found at all, add a new one
    if (best_idx < 0) {
        best_idx = rr->fbos.num;
        PL_ARRAY_APPEND(rr, rr->fbos, NULL);
        pl_grow(pass->tmp, &pass->fbos_used, rr->fbos.num * sizeof(bool));
        pass->fbos_used[best_idx] = false;
    }

    if (!pl_tex_recreate(rr->gpu, &rr->fbos.elem[best_idx], &params))
        return NULL;

    pass->fbos_used[best_idx] = true;
    return rr->fbos.elem[best_idx];
}

// Forcibly convert an img to `tex`, dispatching where necessary
static pl_tex _img_tex(struct pass_state *pass, struct img *img, pl_debug_tag tag)
{
    if (img->tex) {
        pl_assert(!img->sh);
        return img->tex;
    }

    pl_renderer rr = pass->rr;
    pl_tex tex = get_fbo(pass, img->w, img->h, img->fmt, img->comps, tag);
    img->fmt = NULL;

    if (!tex) {
        PL_ERR(rr, "Failed creating FBO texture! Disabling advanced rendering..");
        memset(pass->fbofmt, 0, sizeof(pass->fbofmt));
        pl_dispatch_abort(rr->dp, &img->sh);
        rr->disable_fbos = true;
        return NULL;
    }

    pl_assert(img->sh);
    bool ok = pl_dispatch_finish(rr->dp, pl_dispatch_params(
        .shader = &img->sh,
        .target = tex,
    ));

    if (!ok) {
        PL_ERR(rr, "Failed dispatching intermediate pass!");
        img->sh = pl_dispatch_begin(rr->dp);
        return NULL;
    }

    img->tex = tex;
    return img->tex;
}

#define img_tex(pass, img) _img_tex(pass, img, PL_DEBUG_TAG)

// Forcibly convert an img to `sh`, sampling where necessary
static pl_shader img_sh(struct pass_state *pass, struct img *img)
{
    if (img->sh) {
        pl_assert(!img->tex);
        return img->sh;
    }

    pl_assert(img->tex);
    img->sh = pl_dispatch_begin(pass->rr->dp);
    pl_shader_sample_direct(img->sh, pl_sample_src( .tex = img->tex ));

    img->tex = NULL;
    return img->sh;
}

enum sampler_type {
    SAMPLER_DIRECT,  // pick based on texture caps
    SAMPLER_NEAREST, // direct sampling, force nearest
    SAMPLER_BICUBIC, // fast bicubic scaling
    SAMPLER_COMPLEX, // complex custom filters
    SAMPLER_OVERSAMPLE,
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
    enum sampler_dir dir_sep[2];
};

static struct sampler_info sample_src_info(struct pass_state *pass,
                                           const struct pl_sample_src *src)
{
    const struct pl_render_params *params = pass->params;
    struct sampler_info info = {0};
    pl_renderer rr = pass->rr;

    float rx = src->new_w / fabsf(pl_rect_w(src->rect));
    if (rx < 1.0 - 1e-6) {
        info.dir_sep[0] = SAMPLER_DOWN;
    } else if (rx > 1.0 + 1e-6) {
        info.dir_sep[0] = SAMPLER_UP;
    }

    float ry = src->new_h / fabsf(pl_rect_h(src->rect));
    if (ry < 1.0 - 1e-6) {
        info.dir_sep[1] = SAMPLER_DOWN;
    } else if (ry > 1.0 + 1e-6) {
        info.dir_sep[1] = SAMPLER_UP;
    }

    // We use PL_MAX so downscaling overrides upscaling when choosing scalers
    info.dir = PL_MAX(info.dir_sep[0], info.dir_sep[1]);
    switch (info.dir) {
    case SAMPLER_DOWN:
        info.config = params->downscaler;
        break;
    case SAMPLER_UP:
        info.config = params->upscaler;
        break;
    case SAMPLER_NOOP:
        info.type = SAMPLER_NEAREST;
        return info;
    }

    if (!pass->fbofmt[4] || rr->disable_sampling || !info.config) {
        info.type = SAMPLER_DIRECT;
    } else if (info.config->kernel->weight == oversample) {
        info.type = SAMPLER_OVERSAMPLE;
    } else {
        info.type = SAMPLER_COMPLEX;

        // Try using faster replacements for GPU built-in scalers
        pl_fmt texfmt = src->tex ? src->tex->params.format : pass->fbofmt[4];
        bool can_linear = texfmt->caps & PL_FMT_CAP_LINEAR;
        bool can_fast = info.dir == SAMPLER_UP || params->skip_anti_aliasing;

        if (can_fast && !params->disable_builtin_scalers) {
            if (can_linear && info.config == &pl_filter_bicubic)
                info.type = SAMPLER_BICUBIC;
            if (can_linear && info.config == &pl_filter_bilinear)
                info.type = SAMPLER_DIRECT;
            if (info.config == &pl_filter_nearest)
                info.type = can_linear ? SAMPLER_NEAREST : SAMPLER_DIRECT;
        }
    }

    return info;
}

static void dispatch_sampler(struct pass_state *pass, pl_shader sh,
                             struct sampler *sampler, pl_tex target_tex,
                             const struct pl_sample_src *src)
{
    const struct pl_render_params *params = pass->params;
    if (!sampler)
        goto fallback;

    pl_renderer rr = pass->rr;
    struct sampler_info info = sample_src_info(pass, src);
    pl_shader_obj *lut = NULL;
    switch (info.dir) {
    case SAMPLER_NOOP:
        goto fallback;
    case SAMPLER_DOWN:
        lut = &sampler->downscaler_state;
        break;
    case SAMPLER_UP:
        lut = &sampler->upscaler_state;
        break;
    }

    switch (info.type) {
    case SAMPLER_DIRECT:
        goto fallback;
    case SAMPLER_NEAREST:
        pl_shader_sample_nearest(sh, src);
        return;
    case SAMPLER_OVERSAMPLE:
        pl_shader_sample_oversample(sh, src, info.config->kernel->params[0]);
        return;
    case SAMPLER_BICUBIC:
        pl_shader_sample_bicubic(sh, src);
        return;
    case SAMPLER_COMPLEX:
        break; // continue below
    }

    pl_assert(lut);
    struct pl_sample_filter_params fparams = {
        .filter      = *info.config,
        .lut_entries = params->lut_entries,
        .cutoff      = params->polar_cutoff,
        .antiring    = params->antiringing_strength,
        .no_widening = params->skip_anti_aliasing,
        .lut         = lut,
    };

    if (target_tex) {
        fparams.no_compute = !target_tex->params.storable;
    } else {
        fparams.no_compute = !(pass->fbofmt[4]->caps & PL_FMT_CAP_STORABLE);
    }

    bool ok;
    if (info.config->polar) {
        // Polar samplers are always a single function call
        ok = pl_shader_sample_polar(sh, src, &fparams);
    } else if (info.dir_sep[0] && info.dir_sep[1]) {
        // Scaling is needed in both directions
        pl_shader tsh = pl_dispatch_begin(rr->dp);
        ok = pl_shader_sample_ortho(tsh, PL_SEP_VERT, src, &fparams);
        if (!ok) {
            pl_dispatch_abort(rr->dp, &tsh);
            goto done;
        }

        struct img img = {
            .sh = tsh,
            .w  = src->tex->params.w,
            .h  = src->new_h,
            .comps = src->components,
        };

        struct pl_sample_src src2 = *src;
        src2.tex = img_tex(pass, &img);
        src2.scale = 1.0;
        ok = src2.tex && pl_shader_sample_ortho(sh, PL_SEP_HORIZ, &src2, &fparams);
    } else if (info.dir_sep[0]) {
        // Scaling is needed only in the horizontal direction
        ok = pl_shader_sample_ortho(sh, PL_SEP_HORIZ, src, &fparams);
    } else {
        // Scaling is needed only in the vertical direction
        pl_assert(info.dir_sep[1]);
        ok = pl_shader_sample_ortho(sh, PL_SEP_VERT, src, &fparams);
    }

done:
    if (!ok) {
        PL_ERR(rr, "Failed dispatching scaler.. disabling");
        rr->disable_sampling = true;
        goto fallback;
    }

    return;

fallback:
    // If all else fails, fall back to auto sampling
    pl_shader_sample_direct(sh, src);
}

static void swizzle_color(pl_shader sh, int comps, const int comp_map[4],
                          bool force_alpha)
{
    ident_t orig = sh_fresh(sh, "orig_color");
    GLSL("vec4 %s = color;                  \n"
         "color = vec4(0.0, 0.0, 0.0, 1.0); \n", orig);

    static const int def_map[4] = {0, 1, 2, 3};
    comp_map = PL_DEF(comp_map, def_map);

    for (int c = 0; c < comps; c++) {
        if (comp_map[c] >= 0)
            GLSL("color[%d] = %s[%d]; \n", c, orig, comp_map[c]);
    }

    if (force_alpha)
        GLSL("color.a = %s.a; \n", orig);
}

// `scale` adapts from `pass->dst_rect` to the plane being rendered to
static void draw_overlays(struct pass_state *pass, pl_tex fbo,
                          int comps, const int comp_map[4],
                          const struct pl_overlay *overlays, int num,
                          struct pl_color_space color, struct pl_color_repr repr,
                          const struct pl_transform2x2 *output_shift)
{
    const struct pl_render_params *params = pass->params;
    pl_renderer rr = pass->rr;
    if (num <= 0 || rr->disable_overlay)
        return;

    enum pl_fmt_caps caps = fbo->params.format->caps;
    if (!rr->disable_blending && !(caps & PL_FMT_CAP_BLENDABLE)) {
        PL_WARN(rr, "Trying to draw an overlay to a non-blendable target. "
                "Alpha blending is disabled, results may be incorrect!");
        rr->disable_blending = true;
    }

    const struct pl_frame *image = pass->src_ref >= 0 ? &pass->image : NULL;
    struct pl_transform2x2 src_to_dst;
    if (image) {
        float rx = pl_rect_w(pass->dst_rect) / pl_rect_w(image->crop);
        float ry = pl_rect_h(pass->dst_rect) / pl_rect_h(image->crop);
        src_to_dst = (struct pl_transform2x2) {
            .mat.m = {{ rx, 0 }, { 0, ry }},
            .c = {
                pass->dst_rect.x0 - rx * image->crop.x0,
                pass->dst_rect.y0 - ry * image->crop.y0,
            },
        };

        if (pass->rotation % PL_ROTATION_180 == PL_ROTATION_90) {
            PL_SWAP(src_to_dst.c[0], src_to_dst.c[1]);
            src_to_dst.mat = (struct pl_matrix2x2) {{{ 0, ry }, { rx, 0 }}};
        }
    }

    const struct pl_frame *target = &pass->target;
    struct pl_rect2df dst_crop = target->crop;
    pl_rect2df_rotate(&dst_crop, -pass->rotation);
    pl_rect2df_normalize(&dst_crop);

    for (int n = 0; n < num; n++) {
        struct pl_overlay ol = overlays[n];
        struct pl_overlay_part fallback;
        if (!ol.tex) {
            // Backwards compatibility
            ol.tex = ol.plane.texture;
            ol.parts = &fallback;
            ol.num_parts = 1;
            fallback = (struct pl_overlay_part) {
                .src = {
                    .x0 = -ol.plane.shift_x,
                    .y0 = -ol.plane.shift_y,
                    .x1 = ol.tex->params.w - ol.plane.shift_x,
                    .y1 = ol.tex->params.h - ol.plane.shift_y,
                },
                .dst = {
                    .x0 = ol.rect.x0,
                    .y0 = ol.rect.y0,
                    .x1 = ol.rect.x1,
                    .y1 = ol.rect.y1,
                },
                .color = {
                    ol.base_color[0],
                    ol.base_color[1],
                    ol.base_color[2],
                    1.0,
                },
            };
        }

        if (!ol.num_parts)
            continue;

        if (!ol.coords) {
            ol.coords = overlays == target->overlays
                            ? PL_OVERLAY_COORDS_DST_FRAME
                            : PL_OVERLAY_COORDS_SRC_FRAME;
        }

        struct pl_transform2x2 tf = pl_transform2x2_identity;
        switch (ol.coords) {
            case PL_OVERLAY_COORDS_SRC_CROP:
                if (!image)
                    continue;
                tf.c[0] = image->crop.x0;
                tf.c[1] = image->crop.y0;
                // fall through
            case PL_OVERLAY_COORDS_SRC_FRAME:
                if (!image)
                    continue;
                pl_transform2x2_rmul(&src_to_dst, &tf);
                break;
            case PL_OVERLAY_COORDS_DST_CROP:
                tf.c[0] = dst_crop.x0;
                tf.c[1] = dst_crop.y0;
                break;
            case PL_OVERLAY_COORDS_DST_FRAME:
                break;
            case PL_OVERLAY_COORDS_AUTO:
            case PL_OVERLAY_COORDS_COUNT:
                pl_unreachable();
        }

        if (output_shift)
            pl_transform2x2_rmul(output_shift, &tf);

        // Construct vertex/index buffers
        rr->osd_vertices.num = 0;
        rr->osd_indices.num = 0;
        for (int i = 0; i < ol.num_parts; i++) {
            const struct pl_overlay_part *part = &ol.parts[i];

#define EMIT_VERT(x, y)                                                         \
            do {                                                                \
                float pos[2] = { part->dst.x, part->dst.y };                    \
                pl_transform2x2_apply(&tf, pos);                                \
                PL_ARRAY_APPEND(rr, rr->osd_vertices, (struct osd_vertex) {     \
                    .pos = {                                                    \
                        2.0 * (pos[0] / fbo->params.w) - 1.0,                   \
                        2.0 * (pos[1] / fbo->params.h) - 1.0,                   \
                    },                                                          \
                    .coord = {                                                  \
                        part->src.x / ol.tex->params.w,                         \
                        part->src.y / ol.tex->params.h,                         \
                    },                                                          \
                    .color = {                                                  \
                        part->color[0], part->color[1],                         \
                        part->color[2], part->color[3],                         \
                    },                                                          \
                });                                                             \
            } while (0)

            int idx_base = rr->osd_vertices.num;
            EMIT_VERT(x0, y0); // idx 0: top left
            EMIT_VERT(x1, y0); // idx 1: top right
            EMIT_VERT(x0, y1); // idx 2: bottom left
            EMIT_VERT(x1, y1); // idx 3: bottom right
            PL_ARRAY_APPEND(rr, rr->osd_indices, idx_base + 0);
            PL_ARRAY_APPEND(rr, rr->osd_indices, idx_base + 1);
            PL_ARRAY_APPEND(rr, rr->osd_indices, idx_base + 2);
            PL_ARRAY_APPEND(rr, rr->osd_indices, idx_base + 2);
            PL_ARRAY_APPEND(rr, rr->osd_indices, idx_base + 1);
            PL_ARRAY_APPEND(rr, rr->osd_indices, idx_base + 3);
        }

        // Draw parts
        pl_shader sh = pl_dispatch_begin(rr->dp);
        ident_t tex = sh_desc(sh, (struct pl_shader_desc) {
            .desc = {
                .name = "osd_tex",
                .type = PL_DESC_SAMPLED_TEX,
            },
            .binding = {
                .object = ol.tex,
                .sample_mode = (ol.tex->params.format->caps & PL_FMT_CAP_LINEAR)
                    ? PL_TEX_SAMPLE_LINEAR
                    : PL_TEX_SAMPLE_NEAREST,
            },
        });

        sh_describe(sh, "overlay");
        GLSL("// overlay \n");

        switch (ol.mode) {
        case PL_OVERLAY_NORMAL:
            GLSL("vec4 color = %s(%s, coord); \n",
                 sh_tex_fn(sh, ol.tex->params), tex);
            break;
        case PL_OVERLAY_MONOCHROME:
            GLSL("vec4 color = osd_color; \n");
            break;
        case PL_OVERLAY_MODE_COUNT:
            pl_unreachable();
        };

        sh->res.output = PL_SHADER_SIG_COLOR;
        pl_shader_decode_color(sh, &ol.repr, NULL);
        pl_shader_color_map(sh, params->color_map_params, ol.color, color,
                            NULL, false);

        bool premul = repr.alpha == PL_ALPHA_PREMULTIPLIED;
        pl_shader_encode_color(sh, &repr);
        if (ol.mode == PL_OVERLAY_MONOCHROME) {
            GLSL("color.%s *= %s(%s, coord).r; \n",
                 premul ? "rgba" : "a",
                 sh_tex_fn(sh, ol.tex->params), tex);
        }

        swizzle_color(sh, comps, comp_map, true);

        struct pl_blend_params blend_params = {
            .src_rgb = premul ? PL_BLEND_ONE : PL_BLEND_SRC_ALPHA,
            .src_alpha = PL_BLEND_ONE,
            .dst_rgb = PL_BLEND_ONE_MINUS_SRC_ALPHA,
            .dst_alpha = PL_BLEND_ONE_MINUS_SRC_ALPHA,
        };

        bool ok = pl_dispatch_vertex(rr->dp, pl_dispatch_vertex_params(
            .shader = &sh,
            .target = fbo,
            .blend_params = rr->disable_blending ? NULL : &blend_params,
            .vertex_stride = sizeof(struct osd_vertex),
            .num_vertex_attribs = ol.mode == PL_OVERLAY_NORMAL ? 2 : 3,
            .vertex_attribs = rr->osd_attribs,
            .vertex_position_idx = 0,
            .vertex_coords = PL_COORDS_NORMALIZED,
            .vertex_type = PL_PRIM_TRIANGLE_LIST,
            .vertex_count = rr->osd_indices.num,
            .vertex_data = rr->osd_vertices.elem,
            .index_data = rr->osd_indices.elem,
        ));

        if (!ok) {
            PL_ERR(rr, "Failed rendering overlays!");
            rr->disable_overlay = true;
            return;
        }
    }
}

static pl_tex get_hook_tex(void *priv, int width, int height)
{
    struct pass_state *pass = priv;

    return get_fbo(pass, width, height, NULL, 4, PL_DEBUG_TAG);
}

// Returns if any hook was applied (even if there were errors)
static bool pass_hook(struct pass_state *pass, struct img *img,
                      enum pl_hook_stage stage)
{
    const struct pl_render_params *params = pass->params;
    pl_renderer rr = pass->rr;
    if (!pass->fbofmt[4] || rr->disable_hooks)
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
            .orig_repr = &pass->image.repr,
            .orig_color = &pass->image.color,
            .components = img->comps,
            .src_rect = pass->ref_rect,
            .dst_rect = pass->dst_rect,
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

        case PL_HOOK_SIG_COUNT:
            pl_unreachable();
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

        case PL_HOOK_SIG_COUNT:
            pl_unreachable();
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

static int deband_src(struct pass_state *pass, pl_shader psh,
                      struct pl_sample_src *psrc)
{
    const struct pl_render_params *params = pass->params;
    const struct pl_frame *image = &pass->image;
    pl_renderer rr = pass->rr;
    if (rr->disable_debanding || !params->deband_params || !pass->fbofmt[4])
        return DEBAND_NOOP;

    if (!(psrc->tex->params.format->caps & PL_FMT_CAP_LINEAR)) {
        PL_WARN(rr, "Debanding requires uploaded textures to be linearly "
                "sampleable (params.sample_mode = PL_TEX_SAMPLE_LINEAR)! "
                "Disabling debanding..");
        rr->disable_debanding = true;
        return DEBAND_NOOP;
    }

    bool deband_scales = false;

    pl_shader sh = psh;
    struct pl_sample_src src = *psrc;
    // Only sample/deband the relevant cut-out, but round it to the nearest
    // integer to avoid doing fractional scaling
    pl_rect2df_normalize(&src.rect);
    src.rect.x0 = floorf(src.rect.x0);
    src.rect.y0 = floorf(src.rect.y0);
    src.rect.x1 = ceilf(src.rect.x1);
    src.rect.y1 = ceilf(src.rect.y1);
    src.new_w = pl_rect_w(src.rect);
    src.new_h = pl_rect_h(src.rect);

    if (src.new_w == psrc->new_w &&
        src.new_h == psrc->new_h &&
        pl_rect2d_eq(src.rect, psrc->rect))
    {
        // If there's nothing left to be done (i.e. we're already rendering
        // an exact integer crop without scaling), also skip the scalers
        deband_scales = true;
    } else {
        sh = pl_dispatch_begin_ex(rr->dp, true);
    }

    // Divide the deband grain scale by the effective current colorspace nominal
    // peak, to make sure the output intensity of the grain is as independent
    // of the source as possible, even though it happens this early in the
    // process (well before any linearization / output adaptation)
    struct pl_deband_params dparams = *params->deband_params;
    dparams.grain /= image->color.hdr.max_luma / PL_COLOR_SDR_WHITE;

    pl_shader_deband(sh, &src, &dparams);

    if (deband_scales)
        return DEBAND_SCALED;

    struct img img = {
        .sh = sh,
        .w  = src.new_w,
        .h  = src.new_h,
        .comps = src.components,
    };

    pl_tex new = img_tex(pass, &img);
    if (!new) {
        PL_ERR(rr, "Failed dispatching debanding shader.. disabling debanding!");
        rr->disable_debanding = true;
        return DEBAND_NOOP;
    }

    // Update the original pl_sample_src to point to the new texture
    psrc->tex = new;
    psrc->rect.x0 -= src.rect.x0;
    psrc->rect.y0 -= src.rect.y0;
    psrc->rect.x1 -= src.rect.x0;
    psrc->rect.y1 -= src.rect.y0;
    psrc->scale = 1.0;
    return DEBAND_NORMAL;
}

static void hdr_update_peak(struct pass_state *pass)
{
    const struct pl_render_params *params = pass->params;
    pl_renderer rr = pass->rr;
    if (!params->peak_detect_params || !pl_color_space_is_hdr(&pass->img.color))
        goto cleanup;

    if (rr->disable_peak_detect)
        goto cleanup;

    if (pass->fbofmt[4] && !(pass->fbofmt[4]->caps & PL_FMT_CAP_STORABLE))
        goto cleanup;

    if (pass->img.color.hdr.max_luma <= pass->target.color.hdr.max_luma + 1e-6)
        goto cleanup; // no adaptation needed

    if (params->lut && params->lut_type == PL_LUT_CONVERSION)
        goto cleanup; // LUT handles tone mapping

    if (!pass->fbofmt[4] && !params->allow_delayed_peak_detect) {
        PL_WARN(rr, "Disabling peak detection because "
                "`allow_delayed_peak_detect` is false, but lack of FBOs "
                "forces the result to be delayed.");
        rr->disable_peak_detect = true;
        goto cleanup;
    }

    bool ok = pl_shader_detect_peak(img_sh(pass, &pass->img), pass->img.color,
                                    &rr->tone_map_state, params->peak_detect_params);
    if (!ok) {
        PL_WARN(rr, "Failed creating HDR peak detection shader.. disabling");
        rr->disable_peak_detect = true;
        goto cleanup;
    }

    rr->peak_detect_active = true;
    return;

cleanup:
    // No peak detection required or supported, so clean up the state to avoid
    // confusing it with later frames where peak detection is enabled again
    pl_reset_detected_peak(rr->tone_map_state);
    rr->peak_detect_active = false;
}

struct plane_state {
    enum plane_type type;
    struct pl_plane plane;
    struct img img; // for per-plane shaders
    float plane_w, plane_h; // logical plane dimensions
};

static const char *plane_type_names[] = {
    [PLANE_INVALID] = "invalid",
    [PLANE_ALPHA]   = "alpha",
    [PLANE_CHROMA]  = "chroma",
    [PLANE_LUMA]    = "luma",
    [PLANE_RGB]     = "rgb",
    [PLANE_XYZ]     = "xyz",
};

static void log_plane_info(pl_renderer rr, const struct plane_state *st)
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

// Returns true if grain was applied
static bool plane_film_grain(struct pass_state *pass, int plane_idx,
                             struct plane_state *st,
                             const struct plane_state *ref,
                             const struct pl_frame *image)
{
    pl_renderer rr = pass->rr;
    if (rr->disable_grain)
        return false;

    struct img *img = &st->img;
    struct pl_plane *plane = &st->plane;
    struct pl_color_repr repr = st->img.repr;
    struct pl_film_grain_params grain_params = {
        .data = image->film_grain,
        .luma_tex = ref->plane.texture,
        .repr = &repr,
        .components = plane->components,
    };

    switch (image->film_grain.type) {
    case PL_FILM_GRAIN_NONE: return false;
    case PL_FILM_GRAIN_H274: break;
    case PL_FILM_GRAIN_AV1:
        grain_params.luma_tex = ref->plane.texture;
        for (int c = 0; c < ref->plane.components; c++) {
            if (ref->plane.component_mapping[c] == PL_CHANNEL_Y)
                grain_params.luma_comp = c;
        }
        break;
    default: pl_unreachable();
    }

    for (int c = 0; c < plane->components; c++)
        grain_params.component_mapping[c] = plane->component_mapping[c];

    if (!pl_needs_film_grain(&grain_params))
        return false;

    if (!pass->fbofmt[plane->components]) {
        PL_ERR(rr, "Film grain required but no renderable format available.. "
              "disabling!");
        rr->disable_grain = true;
        return false;
    }

    grain_params.tex = img_tex(pass, img);
    if (!grain_params.tex)
        return false;

    img->sh = pl_dispatch_begin_ex(rr->dp, true);
    if (!pl_shader_film_grain(img->sh, &rr->grain_state[plane_idx], &grain_params)) {
        pl_dispatch_abort(rr->dp, &img->sh);
        rr->disable_grain = true;
        return false;
    }

    img->tex = NULL;
    if (!img_tex(pass, img)) {
        PL_ERR(rr, "Failed applying film grain.. disabling!");
        pl_dispatch_abort(rr->dp, &img->sh);
        img->tex = grain_params.tex;
        rr->disable_grain = true;
        return false;
    }

    img->repr = repr;
    return true;
}

static const enum pl_hook_stage plane_hook_stages[] = {
    [PLANE_ALPHA]   = PL_HOOK_ALPHA_INPUT,
    [PLANE_CHROMA]  = PL_HOOK_CHROMA_INPUT,
    [PLANE_LUMA]    = PL_HOOK_LUMA_INPUT,
    [PLANE_RGB]     = PL_HOOK_RGB_INPUT,
    [PLANE_XYZ]     = PL_HOOK_XYZ_INPUT,
};

static enum pl_lut_type guess_frame_lut_type(const struct pl_frame *frame,
                                             bool reversed)
{
    if (!frame->lut)
        return PL_LUT_UNKNOWN;
    if (frame->lut_type)
        return frame->lut_type;

    enum pl_color_system sys_in = frame->lut->repr_in.sys;
    enum pl_color_system sys_out = frame->lut->repr_out.sys;
    if (reversed)
        PL_SWAP(sys_in, sys_out);

    if (sys_in == PL_COLOR_SYSTEM_RGB && sys_out == sys_in)
        return PL_LUT_NORMALIZED;

    if (sys_in == frame->repr.sys && sys_out == PL_COLOR_SYSTEM_RGB)
        return PL_LUT_CONVERSION;

    // Unknown, just fall back to the default
    return PL_LUT_NATIVE;
}

static pl_fmt merge_fmt(pl_renderer rr, const struct img *a, const struct img *b)
{
    pl_fmt fmta = a->tex ? a->tex->params.format : a->fmt;
    pl_fmt fmtb = b->tex->params.format;
    pl_assert(fmta && fmtb);
    if (fmta->type != fmtb->type)
        return NULL;

    int num_comps = PL_MIN(4, a->comps + b->comps);
    int min_depth = PL_MAX(a->repr.bits.sample_depth, b->repr.bits.sample_depth);

    // Only return formats that support all relevant caps of both formats
    const enum pl_fmt_caps mask = PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_LINEAR;
    enum pl_fmt_caps req_caps = (fmta->caps & mask) | (fmtb->caps & mask);

    return pl_find_fmt(rr->gpu, fmta->type, num_comps, min_depth, 0, req_caps);
}

// Applies a series of rough heuristics to figure out whether we expect any
// performance gains from plane merging. This is basically a series of checks
// for operations that we *know* benefit from merged planes
static bool want_merge(struct pass_state *pass,
                       const struct plane_state *st,
                       const struct plane_state *ref)
{
    const struct pl_render_params *params = pass->params;
    const pl_renderer rr = pass->rr;
    if (!pass->fbofmt[4])
        return false;

    // Debanding
    if (!rr->disable_debanding && params->deband_params)
        return true;

    // Other plane hooks, which are generally nontrivial
    enum pl_hook_stage stage = plane_hook_stages[st->type];
    for (int i = 0; i < params->num_hooks; i++) {
        if (params->hooks[i]->stages & stage)
            return true;
    }

    // Non-trivial scaling
    struct pl_sample_src src = {
        .new_w = ref->img.w,
        .new_h = ref->img.h,
        .rect = {
            .x1 = st->img.w,
            .y1 = st->img.h,
        },
    };

    struct sampler_info info = sample_src_info(pass, &src);
    if (info.type == SAMPLER_COMPLEX)
        return true;

    // Film grain synthesis, can be merged for compatible channels, saving on
    // redundant sampling of the grain/offset textures
    struct pl_film_grain_params grain_params = {
        .data = pass->image.film_grain,
        .repr = (struct pl_color_repr *) &st->img.repr,
        .components = st->plane.components,
    };

    for (int c = 0; c < st->plane.components; c++)
        grain_params.component_mapping[c] = st->plane.component_mapping[c];

    if (!rr->disable_grain && pl_needs_film_grain(&grain_params))
        return true;

    return false;
}

// This scales and merges all of the source images, and initializes pass->img.
static bool pass_read_image(struct pass_state *pass)
{
    const struct pl_render_params *params = pass->params;
    struct pl_frame *image = &pass->image;
    pl_renderer rr = pass->rr;

    struct plane_state planes[4];
    struct plane_state *ref = &planes[pass->src_ref];

    for (int i = 0; i < image->num_planes; i++) {
        planes[i] = (struct plane_state) {
            .type = detect_plane_type(&image->planes[i], &image->repr),
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
    }

    // Original ref texture, even after preprocessing
    pl_tex ref_tex = ref->plane.texture;

    // Merge all compatible planes into 'combined' shaders
    for (int i = 0; i < image->num_planes; i++) {
        struct plane_state *sti = &planes[i];
        if (!sti->type)
            continue;
        if (!want_merge(pass, sti, ref))
            continue;

        for (int j = i+1; j < image->num_planes; j++) {
            struct plane_state *stj = &planes[j];
            bool merge = sti->type == stj->type &&
                         sti->img.w == stj->img.w &&
                         sti->img.h == stj->img.h &&
                         sti->plane.shift_x == stj->plane.shift_x &&
                         sti->plane.shift_y == stj->plane.shift_y;
            if (!merge)
                continue;

            pl_fmt fmt = merge_fmt(rr, &sti->img, &stj->img);
            if (!fmt)
                continue;

            PL_TRACE(rr, "Merging plane %d into plane %d", j, i);
            pl_shader sh = sti->img.sh;
            if (!sh) {
                sh = sti->img.sh = pl_dispatch_begin_ex(pass->rr->dp, true);
                sh_describe(sh, "merging planes");
                GLSL("vec4 tmp; \n");
                pl_shader_sample_direct(sh, pl_sample_src( .tex = sti->img.tex ));
                sti->img.tex = NULL;
            }

            pl_shader psh = pl_dispatch_begin_ex(pass->rr->dp, true);
            pl_shader_sample_direct(psh, pl_sample_src( .tex = stj->img.tex ));

            ident_t sub = sh_subpass(sh, psh);
            pl_dispatch_abort(rr->dp, &psh);
            if (!sub)
                break; // skip merging

            GLSL("tmp = %s(); \n", sub);
            for (int jc = 0; jc < stj->img.comps; jc++) {
                int map = stj->plane.component_mapping[jc];
                if (!map)
                    continue;
                int ic = sti->img.comps++;
                pl_assert(ic < 4);
                GLSL("color[%d] = tmp[%d]; \n", ic, jc);
                sti->plane.components = sti->img.comps;
                sti->plane.component_mapping[ic] = map;
            }

            sti->img.fmt = fmt;
            *stj = (struct plane_state) {0};
        }

        if (!img_tex(pass, &sti->img)) {
            PL_ERR(rr, "Failed dispatching plane merging shader, disabling FBOs!");
            memset(pass->fbofmt, 0, sizeof(pass->fbofmt));
            rr->disable_fbos = true;
            return false;
        }
    }

    // Compute the sampling rc of each plane
    for (int i = 0; i < image->num_planes; i++) {
        struct plane_state *st = &planes[i];
        if (!st->type)
            continue;

        float rx = (float) st->plane.texture->params.w / ref_tex->params.w,
              ry = (float) st->plane.texture->params.h / ref_tex->params.h;

        // Only accept integer scaling ratios. This accounts for the fact that
        // fractionally subsampled planes get rounded up to the nearest integer
        // size, which we want to discard.
        float rrx = rx >= 1 ? roundf(rx) : 1.0 / roundf(1.0 / rx),
              rry = ry >= 1 ? roundf(ry) : 1.0 / roundf(1.0 / ry);

        float sx = st->plane.shift_x,
              sy = st->plane.shift_y;

        st->img.rect = (struct pl_rect2df) {
            .x0 = (image->crop.x0 - sx) * rrx,
            .y0 = (image->crop.y0 - sy) * rry,
            .x1 = (image->crop.x1 - sx) * rrx,
            .y1 = (image->crop.y1 - sy) * rry,
        };

        st->plane_w = ref_tex->params.w * rrx;
        st->plane_h = ref_tex->params.h * rry;

        PL_TRACE(rr, "Plane %d:", i);
        log_plane_info(rr, st);

        // Perform film grain synthesis if needed. Do this first because it
        // requires unmodified plane sizes, and also because it's closer to the
        // intent of the spec (which is to apply synthesis effectively during
        // decoding)

        if (plane_film_grain(pass, i, st, ref, image)) {
            PL_TRACE(rr, "After film grain:");
            log_plane_info(rr, st);
        }

        if (pass_hook(pass, &st->img, plane_hook_stages[st->type])) {
            PL_TRACE(rr, "After user hooks:");
            log_plane_info(rr, st);
        }

        // Update the conceptual width/height after applying plane shaders
        st->img.w = roundf(pl_rect_w(st->img.rect));
        st->img.h = roundf(pl_rect_h(st->img.rect));
    }

    pl_shader sh = pl_dispatch_begin_ex(rr->dp, true);
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

    // For quality reasons, explicitly drop subpixel offsets from the ref rect
    // and re-add them as part of `pass->img.rect`, always rounding towards 0.
    // Additionally, drop anamorphic subpixel mismatches.
    float off_x = ref->img.rect.x0 - truncf(ref->img.rect.x0),
          off_y = ref->img.rect.y0 - truncf(ref->img.rect.y0),
          stretch_x = roundf(pl_rect_w(ref->img.rect)) / pl_rect_w(ref->img.rect),
          stretch_y = roundf(pl_rect_h(ref->img.rect)) / pl_rect_h(ref->img.rect);

    for (int i = 0; i < image->num_planes; i++) {
        struct plane_state *st = &planes[i];
        const struct pl_plane *plane = &st->plane;
        if (!st->type)
            continue;

        float scale_x = pl_rect_w(st->img.rect) / pl_rect_w(ref->img.rect),
              scale_y = pl_rect_h(st->img.rect) / pl_rect_h(ref->img.rect),
              base_x = st->img.rect.x0 - scale_x * off_x,
              base_y = st->img.rect.y0 - scale_y * off_y;

        struct pl_sample_src src = {
            .tex        = st->img.tex,
            .components = plane->components,
            .address_mode = plane->address_mode,
            .scale      = pl_color_repr_normalize(&st->img.repr),
            .new_w      = ref->img.w,
            .new_h      = ref->img.h,
            .rect = {
                base_x,
                base_y,
                base_x + stretch_x * pl_rect_w(st->img.rect),
                base_y + stretch_y * pl_rect_h(st->img.rect),
            },
        };

        if (plane->flipped) {
            src.rect.y0 = st->plane_h - src.rect.y0;
            src.rect.y1 = st->plane_h - src.rect.y1;
        }

        PL_TRACE(rr, "Aligning plane %d: {%f %f %f %f} -> {%f %f %f %f}%s",
                 i, st->img.rect.x0, st->img.rect.y0,
                 st->img.rect.x1, st->img.rect.y1,
                 src.rect.x0, src.rect.y0,
                 src.rect.x1, src.rect.y1,
                 plane->flipped ? " (flipped) " : "");

        pl_shader psh = pl_dispatch_begin_ex(rr->dp, true);
        if (deband_src(pass, psh,  &src) != DEBAND_SCALED)
            dispatch_sampler(pass, psh, &rr->samplers_src[i], NULL, &src);

        ident_t sub = sh_subpass(sh, psh);
        if (!sub) {
            // Can't merge shaders, so instead force FBO indirection here
            struct img inter_img = {
                .sh = psh,
                .w = ref->img.w,
                .h = ref->img.h,
                .comps = src.components,
            };

            pl_tex inter_tex = img_tex(pass, &inter_img);
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
            pl_shader_sample_direct(psh, pl_sample_src( .tex = inter_tex ));
            sub = sh_subpass(sh, psh);
            pl_assert(sub);
        }

        GLSL("tmp = %s();\n", sub);
        for (int c = 0; c < src.components; c++) {
            if (plane->component_mapping[c] < 0)
                continue;
            GLSL("color[%d] = tmp[%d];\n", plane->component_mapping[c], c);
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
        .comps  = ref->img.repr.alpha ? 4 : 3,
        .rect   = {
            off_x,
            off_y,
            off_x + pl_rect_w(ref->img.rect) / stretch_x,
            off_y + pl_rect_h(ref->img.rect) / stretch_y,
        },
    };

    // Update the reference rect to our adjusted image coordinates
    pass->ref_rect = pass->img.rect;

    pass_hook(pass, &pass->img, PL_HOOK_NATIVE);

    // Apply LUT logic and colorspace conversion
    enum pl_lut_type lut_type = guess_frame_lut_type(image, false);
    sh = img_sh(pass, &pass->img);
    bool needs_conversion = true;

    if (lut_type == PL_LUT_NATIVE || lut_type == PL_LUT_CONVERSION) {
        // Fix bit depth normalization before applying LUT
        float scale = pl_color_repr_normalize(&pass->img.repr);
        GLSL("color *= vec4(%s); \n", SH_FLOAT(scale));
        pl_shader_set_alpha(sh, &pass->img.repr, PL_ALPHA_INDEPENDENT);
        pl_shader_custom_lut(sh, image->lut, &rr->lut_state[LUT_IMAGE]);

        if (lut_type == PL_LUT_CONVERSION) {
            pass->img.repr.sys = PL_COLOR_SYSTEM_RGB;
            pass->img.repr.levels = PL_COLOR_LEVELS_FULL;
            needs_conversion = false;
        }
    }

    if (needs_conversion)
        pl_shader_decode_color(sh, &pass->img.repr, params->color_adjustment);
    if (lut_type == PL_LUT_NORMALIZED)
        pl_shader_custom_lut(sh, image->lut, &rr->lut_state[LUT_IMAGE]);

    // Pre-multiply alpha channel before the rest of the pipeline, to avoid
    // bleeding colors from transparent regions into non-transparent regions
    pl_shader_set_alpha(sh, &pass->img.repr, PL_ALPHA_PREMULTIPLIED);

    pass_hook(pass, &pass->img, PL_HOOK_RGB);
    sh = NULL;

    // HDR peak detection, do this as early as possible
    hdr_update_peak(pass);
    return true;
}

static bool pass_scale_main(struct pass_state *pass)
{
    const struct pl_render_params *params = pass->params;
    pl_renderer rr = pass->rr;

    pl_fmt fbofmt = pass->fbofmt[pass->img.comps];
    if (!fbofmt) {
        PL_TRACE(rr, "Skipping main scaler (no FBOs)");
        return true;
    }

    struct img *img = &pass->img;
    struct pl_sample_src src = {
        .components = img->comps,
        .new_w      = abs(pl_rect_w(pass->dst_rect)),
        .new_h      = abs(pl_rect_h(pass->dst_rect)),
        .rect       = img->rect,
    };

    const struct pl_frame *image = &pass->image;
    bool need_fbo = image->num_overlays > 0;
    need_fbo |= rr->peak_detect_active && !params->allow_delayed_peak_detect;

    // Force FBO indirection if this shader is non-resizable
    int out_w, out_h;
    if (img->sh && pl_shader_output_size(img->sh, &out_w, &out_h))
        need_fbo |= out_w != src.new_w || out_h != src.new_h;

    struct sampler_info info = sample_src_info(pass, &src);
    bool use_sigmoid = info.dir == SAMPLER_UP && params->sigmoid_params;
    bool use_linear  = info.dir == SAMPLER_DOWN;

    // We need to enable the full rendering pipeline if there are any user
    // shaders / hooks that might depend on it.
    uint64_t scaling_hooks = PL_HOOK_PRE_KERNEL | PL_HOOK_POST_KERNEL;
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
        img->w = src.new_w;
        img->h = src.new_h;
        PL_TRACE(rr, "Skipping main scaler (free sampling)");
        return true;
    }

    // Hard-disable both sigmoidization and linearization when required
    if (params->disable_linear_scaling || fbofmt->component_depth[0] < 16)
        use_sigmoid = use_linear = false;

    // Avoid sigmoidization for HDR content because it clips to [0,1]
    if (pl_color_space_is_hdr(&img->color)) {
        use_sigmoid = false;
        if (fbofmt->type != PL_FMT_FLOAT)
            use_linear = false; // linear HDR needs out-of-range signals
    }

    if (use_linear || use_sigmoid) {
        pl_shader_linearize(img_sh(pass, img), &img->color);
        img->color.transfer = PL_COLOR_TRC_LINEAR;
        pass_hook(pass, img, PL_HOOK_LINEAR);
    }

    if (use_sigmoid) {
        pl_shader_sigmoidize(img_sh(pass, img), params->sigmoid_params);
        pass_hook(pass, img, PL_HOOK_SIGMOID);
    }

    pass_hook(pass, img, PL_HOOK_PRE_KERNEL);

    src.tex = img_tex(pass, img);
    if (!src.tex)
        return false;

    pl_shader sh = pl_dispatch_begin_ex(rr->dp, true);
    dispatch_sampler(pass, sh, &rr->sampler_main, NULL, &src);
    *img = (struct img) {
        .sh     = sh,
        .w      = src.new_w,
        .h      = src.new_h,
        .repr   = img->repr,
        .rect   = { 0, 0, src.new_w, src.new_h },
        .color  = img->color,
        .comps  = img->comps,
    };

    pass_hook(pass, img, PL_HOOK_POST_KERNEL);

    if (use_sigmoid)
        pl_shader_unsigmoidize(img_sh(pass, img), params->sigmoid_params);

    pass_hook(pass, img, PL_HOOK_SCALED);
    return true;
}

#define CLEAR_COL(params)                                                       \
    (float[4]) {                                                                \
        (params)->background_color[0],                                          \
        (params)->background_color[1],                                          \
        (params)->background_color[2],                                          \
        1.0 - (params)->background_transparency,                                \
    }

static bool pass_output_target(struct pass_state *pass)
{
    const struct pl_render_params *params = pass->params;
    const struct pl_frame *image = &pass->image;
    const struct pl_frame *target = &pass->target;
    pl_renderer rr = pass->rr;

    struct img *img = &pass->img;
    pl_shader sh = img_sh(pass, img);

    // Color management
    bool prelinearized = false;
    bool need_conversion = true;
    assert(image->color.primaries == img->color.primaries);
    if (img->color.transfer == PL_COLOR_TRC_LINEAR) {
        if (img->repr.alpha == PL_ALPHA_PREMULTIPLIED) {
            // Very annoying edge case: since prelinerization happens with
            // premultiplied alpha, but color mapping happens with independent
            // alpha, we need to go back to non-linear representation *before*
            // alpha mode conversion, to avoid distortion
            img->color.transfer = image->color.transfer;
            pl_shader_delinearize(sh, &img->color);
        } else {
            prelinearized = true;
        }
    }

    // Do all processing in independent alpha, to avoid nonlinear distortions
    pl_shader_set_alpha(sh, &img->repr, PL_ALPHA_INDEPENDENT);

    bool need_icc = !params->ignore_icc_profiles &&
                    (image->profile.data || target->profile.data) &&
                    !pl_icc_profile_equal(&image->profile, &target->profile);

    if (params->force_icc_lut || params->force_3dlut)
        need_icc |= !pl_color_space_equal(&image->color, &target->color);
    need_icc &= !rr->disable_icc;

    if (params->lut) {
        struct pl_color_space lut_in = params->lut->color_in;
        struct pl_color_space lut_out = params->lut->color_out;
        switch (params->lut_type) {
        case PL_LUT_UNKNOWN:
        case PL_LUT_NATIVE:
            pl_color_space_merge(&lut_in, &image->color);
            pl_color_space_merge(&lut_out, &image->color);
            break;
        case PL_LUT_CONVERSION:
            pl_color_space_merge(&lut_in, &image->color);
            pl_color_space_merge(&lut_out, &target->color);
            // Conversion LUT the highest priority
            need_icc = false;
            need_conversion = false;
            break;
        case PL_LUT_NORMALIZED:
            if (!prelinearized) {
                // PL_LUT_NORMALIZED wants linear input data
                pl_shader_linearize(sh, &img->color);
                img->color.transfer = PL_COLOR_TRC_LINEAR;
                prelinearized = true;
            }
            pl_color_space_merge(&lut_in, &img->color);
            pl_color_space_merge(&lut_out, &img->color);
            break;
        }

        pl_shader_color_map(sh, params->color_map_params, image->color, lut_in,
                            NULL, prelinearized);

        if (params->lut_type == PL_LUT_NORMALIZED) {
            GLSLF("color.rgb *= vec3(1.0/%s); \n",
                  SH_FLOAT(pl_color_transfer_nominal_peak(lut_in.transfer)));
        }

        pl_shader_custom_lut(sh, params->lut, &rr->lut_state[LUT_PARAMS]);

        if (params->lut_type == PL_LUT_NORMALIZED) {
            GLSLF("color.rgb *= vec3(%s); \n",
                  SH_FLOAT(pl_color_transfer_nominal_peak(lut_out.transfer)));
        }

        if (params->lut_type != PL_LUT_CONVERSION) {
            pl_shader_color_map(sh, params->color_map_params, lut_out, img->color,
                                NULL, false);
        }
    }

#ifdef PL_HAVE_LCMS

    if (need_icc) {
        struct pl_icc_color_space src = {
            .color = image->color,
            .profile = image->profile,
        };

        struct pl_icc_color_space dst = {
            .color = target->color,
            .profile = target->profile,
        };

        if (params->ignore_icc_profiles)
            src.profile = dst.profile = (struct pl_icc_profile) {0};

        struct pl_icc_result res;
        bool ok = pl_icc_update(sh, &src, &dst, &rr->icc_state, &res,
                                PL_DEF(params->icc_params, params->lut3d_params));
        if (!ok) {
            rr->disable_icc = true;
            goto fallback;
        }

        // current -> ICC in
        pl_shader_color_map(sh, params->color_map_params, image->color,
                            res.src_color, &rr->tone_map_state, prelinearized);
        // ICC in -> ICC out
        pl_icc_apply(sh, &rr->icc_state);
        // ICC out -> target
        pl_shader_color_map(sh, params->color_map_params, res.dst_color,
                            target->color, NULL, false);

        need_conversion = false;
    }

fallback:

#else // !PL_HAVE_LCMS

    if (need_icc) {
        PL_WARN(rr, "An ICC profile was set, but libplacebo is built without "
                "support for LittleCMS! Disabling..");
        rr->disable_icc = true;
    }

#endif

    if (need_conversion) {
        // current -> target
        pl_shader_color_map(sh, params->color_map_params, image->color,
                            target->color, &rr->tone_map_state, prelinearized);
    }

    // Apply color blindness simulation if requested
    if (params->cone_params)
        pl_shader_cone_distort(sh, target->color, params->cone_params);

    enum pl_lut_type lut_type = guess_frame_lut_type(target, true);
    if (lut_type == PL_LUT_NORMALIZED || lut_type == PL_LUT_CONVERSION)
        pl_shader_custom_lut(sh, target->lut, &rr->lut_state[LUT_TARGET]);

    bool need_blend = params->blend_against_tiles || !target->repr.alpha;
    if (img->comps == 4 && need_blend) {
        if (params->blend_against_tiles) {
            static const float zero[2][3] = {0};
            const float (*color)[3] = params->tile_colors;
            if (memcmp(color, zero, sizeof(zero)) == 0)
                color = pl_render_default_params.tile_colors;
            int size = PL_DEF(params->tile_size, pl_render_default_params.tile_size);
            GLSLH("#define bg_tile_a vec3(%s, %s, %s) \n",
                  SH_FLOAT(color[0][0]), SH_FLOAT(color[0][1]), SH_FLOAT(color[0][2]));
            GLSLH("#define bg_tile_b vec3(%s, %s, %s) \n",
                  SH_FLOAT(color[1][0]), SH_FLOAT(color[1][1]), SH_FLOAT(color[1][2]));
            GLSL("%s tile = lessThan(fract(gl_FragCoord.xy * %s), vec2(0.5));   \n"
                 "vec3 bg_color = tile.x == tile.y ? bg_tile_a : bg_tile_b;     \n",
                 sh_bvec(sh, 2), SH_FLOAT(1.0 / size));
        } else {
            GLSLH("#define bg_color vec3(%s, %s, %s) \n",
                  SH_FLOAT(params->background_color[0]),
                  SH_FLOAT(params->background_color[1]),
                  SH_FLOAT(params->background_color[2]));
        }

        pl_assert(img->repr.alpha != PL_ALPHA_PREMULTIPLIED);
        GLSL("color = vec4(mix(bg_color, color.rgb, color.a), 1.0); \n");
        img->repr.alpha = PL_ALPHA_UNKNOWN;
        img->comps = 3;
    }

    // Apply the color scale separately, after encoding is done, to make sure
    // that the intermediate FBO (if any) has the correct precision.
    struct pl_color_repr repr = target->repr;
    float scale = pl_color_repr_normalize(&repr);
    if (lut_type != PL_LUT_CONVERSION)
        pl_shader_encode_color(sh, &repr);
    if (lut_type == PL_LUT_NATIVE) {
        pl_shader_custom_lut(sh, target->lut, &rr->lut_state[LUT_TARGET]);
        pl_shader_set_alpha(sh, &img->repr, PL_ALPHA_PREMULTIPLIED);
    }

    // Rotation handling
    struct pl_rect2d dst_rect = pass->dst_rect;
    if (pass->rotation % PL_ROTATION_180 == PL_ROTATION_90) {
        PL_SWAP(dst_rect.x0, dst_rect.y0);
        PL_SWAP(dst_rect.x1, dst_rect.y1);
        PL_SWAP(img->w, img->h);
        sh->transpose = true;
    }

    pass_hook(pass, img, PL_HOOK_OUTPUT);
    sh = NULL;

    const struct pl_plane *ref = &target->planes[pass->dst_ref];
    bool flipped_x = dst_rect.x1 < dst_rect.x0,
         flipped_y = dst_rect.y1 < dst_rect.y0;

    if (!params->skip_target_clearing && pl_frame_is_cropped(target))
        pl_frame_clear_rgba(rr->gpu, target, CLEAR_COL(params));

    for (int p = 0; p < target->num_planes; p++) {
        const struct pl_plane *plane = &target->planes[p];
        float rx = (float) plane->texture->params.w / ref->texture->params.w,
              ry = (float) plane->texture->params.h / ref->texture->params.h;

        // Only accept integer scaling ratios. This accounts for the fact
        // that fractionally subsampled planes get rounded up to the
        // nearest integer size, which we want to over-render.
        float rrx = rx >= 1 ? roundf(rx) : 1.0 / roundf(1.0 / rx),
              rry = ry >= 1 ? roundf(ry) : 1.0 / roundf(1.0 / ry);
        float sx = plane->shift_x, sy = plane->shift_y;

        struct pl_rect2df plane_rectf = {
            .x0 = (dst_rect.x0 - sx) * rrx,
            .y0 = (dst_rect.y0 - sy) * rry,
            .x1 = (dst_rect.x1 - sx) * rrx,
            .y1 = (dst_rect.y1 - sy) * rry,
        };

        // Normalize to make the math easier
        pl_rect2df_normalize(&plane_rectf);

        // Round the output rect
        int rx0 = floorf(plane_rectf.x0), ry0 = floorf(plane_rectf.y0),
            rx1 =  ceilf(plane_rectf.x1), ry1 =  ceilf(plane_rectf.y1);

        PL_TRACE(rr, "Subsampled target %d: {%f %f %f %f} -> {%d %d %d %d}",
                 p, plane_rectf.x0, plane_rectf.y0,
                 plane_rectf.x1, plane_rectf.y1,
                 rx0, ry0, rx1, ry1);

        if (target->num_planes > 1) {

            // Planar output, so we need to sample from an intermediate FBO
            struct pl_sample_src src = {
                .tex        = img_tex(pass, img),
                .new_w      = rx1 - rx0,
                .new_h      = ry1 - ry0,
                .rect = {
                    .x0 = (rx0 - plane_rectf.x0) / rrx,
                    .x1 = (rx1 - plane_rectf.x0) / rrx,
                    .y0 = (ry0 - plane_rectf.y0) / rry,
                    .y1 = (ry1 - plane_rectf.y0) / rry,
                },
            };

            if (!src.tex) {
                PL_ERR(rr, "Output requires multiple planes, but FBOs are "
                       "unavailable. This combination is unsupported.");
                return false;
            }

            PL_TRACE(rr, "Sampling %dx%d img aligned from {%f %f %f %f}",
                     pass->img.w, pass->img.h,
                     src.rect.x0, src.rect.y0,
                     src.rect.x1, src.rect.y1);

            for (int c = 0; c < plane->components; c++) {
                if (plane->component_mapping[c] < 0)
                    continue;
                src.component_mask |= 1 << plane->component_mapping[c];
            }

            sh = pl_dispatch_begin(rr->dp);
            dispatch_sampler(pass, sh, &rr->samplers_dst[p], plane->texture, &src);

        } else {

            // Single plane, so we can directly re-use the img shader unless
            // it's incompatible with the FBO capabilities
            bool is_comp = pl_shader_is_compute(img_sh(pass, img));
            if (is_comp && !plane->texture->params.storable) {
                if (!img_tex(pass, img)) {
                    PL_ERR(rr, "Rendering requires compute shaders, but output "
                           "is not storable, and FBOs are unavailable. This "
                           "combination is unsupported.");
                    return false;
                }
            }

            sh = img_sh(pass, img);
            img->sh = NULL;

        }

        if (params->dither_params) {
            // Ignore dithering for > 16-bit outputs by default, since it makes
            // little sense to do so (and probably just adds errors)
            int depth = target->repr.bits.color_depth;
            if (depth && (depth < 16 || params->force_dither))
                pl_shader_dither(sh, depth, &rr->dither_state, params->dither_params);
        }

        GLSL("color *= vec4(1.0 / %s); \n", SH_FLOAT(scale));
        swizzle_color(sh, plane->components, plane->component_mapping, false);

        struct pl_rect2d plane_rect = {
            .x0 = flipped_x ? rx1 : rx0,
            .x1 = flipped_x ? rx0 : rx1,
            .y0 = flipped_y ? ry1 : ry0,
            .y1 = flipped_y ? ry0 : ry1,
        };

        struct pl_transform2x2 tscale = {
            .mat = {{{ rrx, 0.0 }, { 0.0, rry }}},
            .c = { -sx, -sy },
        };

        if (plane->flipped) {
            int plane_h = rry * ref->texture->params.h;
            plane_rect.y0 = plane_h - plane_rect.y0;
            plane_rect.y1 = plane_h - plane_rect.y1;
            tscale.mat.m[1][1] = -tscale.mat.m[1][1];
            tscale.c[1] += plane->texture->params.h;
        }

        bool ok = pl_dispatch_finish(rr->dp, pl_dispatch_params(
            .shader = &sh,
            .target = plane->texture,
            .blend_params = params->blend_params,
            .rect = plane_rect,
        ));

        if (!ok)
            return false;

        if (pass->info.stage != PL_RENDER_STAGE_BLEND) {
            draw_overlays(pass, plane->texture, plane->components,
                          plane->component_mapping, image->overlays,
                          image->num_overlays, target->color, target->repr,
                          &tscale);
        }

        draw_overlays(pass, plane->texture, plane->components,
                      plane->component_mapping, target->overlays,
                      target->num_overlays, target->color, target->repr,
                      &tscale);
    }

    *img = (struct img) {0};
    return true;
}

#define require(expr)                                                           \
  do {                                                                          \
      if (!(expr)) {                                                            \
          PL_ERR(rr, "Validation failed: %s (%s:%d)",                           \
                  #expr, __FILE__, __LINE__);                                   \
          pl_log_stack_trace(rr->log, PL_LOG_ERR);                              \
          return false;                                                         \
      }                                                                         \
  } while (0)

#define validate_plane(plane, param)                                            \
  do {                                                                          \
      require((plane).texture);                                                 \
      require((plane).texture->params.param);                                   \
      require((plane).components > 0 && (plane).components <= 4);               \
      for (int c = 0; c < (plane).components; c++) {                            \
          require((plane).component_mapping[c] >= PL_CHANNEL_NONE &&            \
                  (plane).component_mapping[c] <= PL_CHANNEL_A);                \
      }                                                                         \
  } while (0)

#define validate_overlay(overlay)                                               \
  do {                                                                          \
      require(!(overlay).tex ^ !(overlay).plane.texture);                       \
      if ((overlay).tex) {                                                      \
          require((overlay).tex->params.sampleable);                            \
          require((overlay).num_parts >= 0);                                    \
          for (int n = 0; n < (overlay).num_parts; n++) {                       \
              const struct pl_overlay_part *p = &(overlay).parts[n];            \
              require(pl_rect_w(p->dst) && pl_rect_h(p->dst));                  \
          }                                                                     \
      } else {                                                                  \
          require((overlay).num_parts == 0);                                    \
          require((overlay).plane.texture->params.sampleable);                  \
          require(pl_rect_w((overlay).rect) && pl_rect_h((overlay).rect));      \
      }                                                                         \
  } while (0)

// Perform some basic validity checks on incoming structs to help catch invalid
// API usage. This is not an exhaustive check. In particular, enums are not
// bounds checked. This is because most functions accepting enums already
// abort() in the default case, and because it's not the intent of this check
// to catch all instances of memory corruption - just common logic bugs.
static bool validate_structs(pl_renderer rr,
                             const struct pl_frame *image,
                             const struct pl_frame *target)
{
    // Rendering to/from a frame with no planes is technically allowed, but so
    // pointless that it's more likely to be a user error worth catching.
    require(image->num_planes > 0 && image->num_planes <= PL_MAX_PLANES);
    require(target->num_planes > 0 && target->num_planes <= PL_MAX_PLANES);
    for (int i = 0; i < image->num_planes; i++)
        validate_plane(image->planes[i], sampleable);
    for (int i = 0; i < target->num_planes; i++)
        validate_plane(target->planes[i], renderable);

    float src_w = pl_rect_w(image->crop), src_h = pl_rect_h(image->crop);
    float dst_w = pl_rect_w(target->crop), dst_h = pl_rect_h(target->crop);
    require(!src_w == !src_h);
    require(!dst_w == !dst_h);

    require(image->num_overlays >= 0);
    require(target->num_overlays >= 0);
    for (int i = 0; i < image->num_overlays; i++)
        validate_overlay(image->overlays[i]);
    for (int i = 0; i < target->num_overlays; i++)
        validate_overlay(target->overlays[i]);

    return true;
}

// returns index
static int frame_ref(const struct pl_frame *frame)
{
    pl_assert(frame->num_planes);
    for (int i = 0; i < frame->num_planes; i++) {
        switch (detect_plane_type(&frame->planes[i], &frame->repr)) {
        case PLANE_RGB:
        case PLANE_LUMA:
        case PLANE_XYZ:
            return i;
        case PLANE_CHROMA:
        case PLANE_ALPHA:
            continue;
        case PLANE_INVALID:
            pl_unreachable();
        }
    }

    return 0;
}

static void fix_refs_and_rects(struct pass_state *pass)
{
    struct pl_frame *image = &pass->image;
    struct pl_frame *target = &pass->target;
    pass->src_ref = frame_ref(image);
    pass->dst_ref = frame_ref(target);

    // Fix the rendering rects
    struct pl_rect2df *src = &image->crop, *dst = &target->crop;
    pl_tex src_ref = image->planes[pass->src_ref].texture;
    pl_tex dst_ref = target->planes[pass->dst_ref].texture;
    int dst_w = dst_ref->params.w, dst_h = dst_ref->params.h;

    if ((!src->x0 && !src->x1) || (!src->y0 && !src->y1)) {
        src->x1 = src_ref->params.w;
        src->y1 = src_ref->params.h;
    };

    if ((!dst->x0 && !dst->x1) || (!dst->y0 && !dst->y1)) {
        dst->x1 = dst_w;
        dst->y1 = dst_h;
    }

    // Compute end-to-end rotation
    pass->rotation = pl_rotation_normalize(image->rotation - target->rotation);
    pl_rect2df_rotate(dst, -pass->rotation); // normalize by counter-rotating
    if (pass->rotation % PL_ROTATION_180 == PL_ROTATION_90)
        PL_SWAP(dst_w, dst_h);

    // Keep track of whether the end-to-end rendering is flipped
    bool flipped_x = (src->x0 > src->x1) != (dst->x0 > dst->x1),
         flipped_y = (src->y0 > src->y1) != (dst->y0 > dst->y1);

    // Normalize both rects to make the math easier
    pl_rect2df_normalize(src);
    pl_rect2df_normalize(dst);

    // Round the output rect and clip it to the framebuffer dimensions
    float rx0 = roundf(PL_MAX(dst->x0, 0.0)),
          ry0 = roundf(PL_MAX(dst->y0, 0.0)),
          rx1 = roundf(PL_MIN(dst->x1, dst_w)),
          ry1 = roundf(PL_MIN(dst->y1, dst_h));

    // Adjust the src rect corresponding to the rounded crop
    float scale_x = pl_rect_w(*src) / pl_rect_w(*dst),
          scale_y = pl_rect_h(*src) / pl_rect_h(*dst),
          base_x = src->x0,
          base_y = src->y0;

    src->x0 = base_x + (rx0 - dst->x0) * scale_x;
    src->x1 = base_x + (rx1 - dst->x0) * scale_x;
    src->y0 = base_y + (ry0 - dst->y0) * scale_y;
    src->y1 = base_y + (ry1 - dst->y0) * scale_y;

    // Update dst_rect to the rounded values and re-apply flip if needed. We
    // always do this in the `dst` rather than the `src`` because this allows
    // e.g. polar sampling compute shaders to work.
    *dst = (struct pl_rect2df) {
        .x0 = flipped_x ? rx1 : rx0,
        .y0 = flipped_y ? ry1 : ry0,
        .x1 = flipped_x ? rx0 : rx1,
        .y1 = flipped_y ? ry0 : ry1,
    };

    // Copies of the above, for convenience
    pass->ref_rect = *src;
    pass->dst_rect = (struct pl_rect2d) {
        dst->x0, dst->y0, dst->x1, dst->y1,
    };
}

static void fix_color_space(struct pl_frame *frame)
{
    pl_tex tex = frame->planes[frame_ref(frame)].texture;

    // If the primaries are not known, guess them based on the resolution
    if (!frame->color.primaries)
        frame->color.primaries = pl_color_primaries_guess(tex->params.w, tex->params.h);

    pl_color_space_infer(&frame->color);

    // For UNORM formats, we can infer the sampled bit depth from the texture
    // itself. This is ignored for other format types, because the logic
    // doesn't really work out for them anyways, and it's best not to do
    // anything too crazy unless the user provides explicit details.
    struct pl_bit_encoding *bits = &frame->repr.bits;
    if (!bits->sample_depth && tex->params.format->type == PL_FMT_UNORM) {
        // Just assume the first component's depth is canonical. This works in
        // practice, since for cases like rgb565 we want to use the lower depth
        // anyway. Plus, every format has at least one component.
        bits->sample_depth = tex->params.format->component_depth[0];

        // If we don't know the color depth, assume it spans the full range of
        // the texture. Otherwise, clamp it to the texture depth.
        bits->color_depth = PL_DEF(bits->color_depth, bits->sample_depth);
        bits->color_depth = PL_MIN(bits->color_depth, bits->sample_depth);

        // If the texture depth is higher than the known color depth, assume
        // the colors were left-shifted.
        bits->bit_shift += bits->sample_depth - bits->color_depth;
    }
}

static bool pass_infer_state(struct pass_state *pass)
{
    struct pl_frame *image = &pass->image;
    struct pl_frame *target = &pass->target;
    if (!validate_structs(pass->rr, image, target))
        return false;

    fix_refs_and_rects(pass);
    fix_color_space(image);
    find_fbo_format(pass);

    // Infer the target color space info based on the image's
    pl_color_space_infer_ref(&target->color, &image->color);
    fix_color_space(target);

    // Detect the presence of an alpha channel in the frames and explicitly
    // default the alpha mode in this case, so we can use it to detect whether
    // or not to strip the alpha channel during rendering.
    //
    // Note the different defaults for the image and target, because files
    // are usually independent but windowing systems usually expect
    // premultiplied. (We also premultiply for internal rendering, so this
    // way of doing it avoids a possible division-by-zero path!)
    if (!image->repr.alpha) {
        for (int i = 0; i < image->num_planes; i++) {
            const struct pl_plane *plane = &image->planes[i];
            for (int c = 0; c < plane->components; c++) {
                if (plane->component_mapping[c] == PL_CHANNEL_A)
                    image->repr.alpha = PL_ALPHA_INDEPENDENT;
            }
        }
    }

    if (!target->repr.alpha) {
        for (int i = 0; i < target->num_planes; i++) {
            const struct pl_plane *plane = &target->planes[i];
            for (int c = 0; c < plane->components; c++) {
                if (plane->component_mapping[c] == PL_CHANNEL_A)
                    target->repr.alpha = PL_ALPHA_PREMULTIPLIED;
            }
        }
    }

    return true;
}

static bool draw_empty_overlays(pl_renderer rr,
                                const struct pl_frame *ptarget,
                                const struct pl_render_params *params)
{
    if (!params->skip_target_clearing)
        pl_frame_clear_rgba(rr->gpu, ptarget, CLEAR_COL(params));

    if (!ptarget->num_overlays)
        return true;

    struct pass_state pass = {
        .rr = rr,
        .params = params,
        .target = *ptarget,
        .info.stage = PL_RENDER_STAGE_FRAME,
        .src_ref = -1,
    };

    struct pl_frame *target = &pass.target;
    require(target->num_planes > 0 && target->num_planes <= PL_MAX_PLANES);
    for (int i = 0; i < target->num_planes; i++)
        validate_plane(target->planes[i], renderable);
    require(target->num_overlays >= 0);
    for (int i = 0; i < target->num_overlays; i++)
        validate_overlay(target->overlays[i]);
    fix_color_space(target);
    pass.dst_ref = frame_ref(target);

    pl_dispatch_callback(rr->dp, &pass, info_callback);
    pl_dispatch_reset_frame(rr->dp);

    pl_tex ref = target->planes[pass.dst_ref].texture;
    for (int p = 0; p < target->num_planes; p++) {
        const struct pl_plane *plane = &target->planes[p];
        // Math replicated from `pass_output_target`
        float rx = (float) plane->texture->params.w / ref->params.w,
              ry = (float) plane->texture->params.h / ref->params.h;
        float rrx = rx >= 1 ? roundf(rx) : 1.0 / roundf(1.0 / rx),
              rry = ry >= 1 ? roundf(ry) : 1.0 / roundf(1.0 / ry);
        float sx = plane->shift_x, sy = plane->shift_y;

        struct pl_transform2x2 tscale = {
            .mat = {{{ rrx, 0.0 }, { 0.0, rry }}},
            .c = { -sx, -sy },
        };

        draw_overlays(&pass, plane->texture, plane->components,
                      plane->component_mapping, target->overlays,
                      target->num_overlays, target->color, target->repr,
                      &tscale);
    }

    return true;
}

bool pl_render_image(pl_renderer rr, const struct pl_frame *pimage,
                     const struct pl_frame *ptarget,
                     const struct pl_render_params *params)
{
    params = PL_DEF(params, &pl_render_default_params);
    pl_dispatch_mark_dynamic(rr->dp, params->dynamic_constants);
    if (!pimage)
        return draw_empty_overlays(rr, ptarget, params);

    struct pass_state pass = {
        .rr = rr,
        .params = params,
        .image = *pimage,
        .target = *ptarget,
        .info.stage = PL_RENDER_STAGE_FRAME,
    };

    if (!pass_infer_state(&pass))
        return false;

    pass.tmp = pl_tmp(NULL),
    pass.fbos_used = pl_calloc(pass.tmp, rr->fbos.num, sizeof(bool));

    pl_dispatch_callback(rr->dp, &pass, info_callback);
    pl_dispatch_reset_frame(rr->dp);

    for (int i = 0; i < params->num_hooks; i++) {
        if (params->hooks[i]->reset)
            params->hooks[i]->reset(params->hooks[i]->priv);
    }

    if (!pass_read_image(&pass))
        goto error;

    if (!pass_scale_main(&pass))
        goto error;

    if (!pass_output_target(&pass))
        goto error;

    pl_free(pass.tmp);
    return true;

error:
    pl_dispatch_abort(rr->dp, &pass.img.sh);
    pl_free(pass.tmp);
    PL_ERR(rr, "Failed rendering image!");
    return false;
}

static uint64_t render_params_hash(const struct pl_render_params *params_orig)
{
    struct pl_render_params params = *params_orig;
    uint64_t hash = 0;

#define HASH_PTR(ptr)                                                           \
    do {                                                                        \
        if (ptr) {                                                              \
            pl_hash_merge(&hash, pl_mem_hash(ptr, sizeof(*ptr)));               \
            ptr = NULL;                                                         \
        }                                                                       \
    } while (0)

#define HASH_FILTER(scaler)                                                     \
    do {                                                                        \
        if (scaler) {                                                           \
            struct pl_filter_config filter = *scaler;                           \
            HASH_PTR(filter.kernel);                                            \
            HASH_PTR(filter.window);                                            \
            pl_hash_merge(&hash, pl_mem_hash(&filter, sizeof(filter)));         \
            scaler = NULL;                                                      \
        }                                                                       \
    } while (0)

    HASH_FILTER(params.upscaler);
    HASH_FILTER(params.downscaler);

    HASH_PTR(params.deband_params);
    HASH_PTR(params.sigmoid_params);
    HASH_PTR(params.color_adjustment);
    HASH_PTR(params.peak_detect_params);
    HASH_PTR(params.color_map_params);
    HASH_PTR(params.dither_params);

    // Hash all hooks
    for (int i = 0; i < params.num_hooks; i++) {
        const struct pl_hook *hook = params.hooks[i];
        if (hook->stages == PL_HOOK_OUTPUT)
            continue; // ignore hooks only relevant to pass_output_target
        pl_hash_merge(&hash, pl_mem_hash(hook, sizeof(*hook)));
    }
    params.hooks = NULL;

    // Hash the LUT by only looking at the signature
    if (params.lut) {
        pl_hash_merge(&hash, params.lut->signature);
        params.lut = NULL;
    }

#define CLEAR(field) field = (__typeof__(field)) {0}

    // Clear out fields only relevant to pl_render_image_mix
    CLEAR(params.frame_mixer);
    CLEAR(params.preserve_mixing_cache);
    CLEAR(params.skip_caching_single_frame);
    memset(params.background_color, 0, sizeof(params.background_color));
    CLEAR(params.background_transparency);
    CLEAR(params.skip_target_clearing);
    CLEAR(params.blend_against_tiles);
    memset(params.tile_colors, 0, sizeof(params.tile_colors));
    CLEAR(params.tile_size);

    // Clear out fields only relevant to pass_output_target
    CLEAR(params.blend_params);
    CLEAR(params.cone_params);
    CLEAR(params.icc_params);
    CLEAR(params.lut3d_params);
    CLEAR(params.ignore_icc_profiles);
    CLEAR(params.force_icc_lut);
    CLEAR(params.force_3dlut);
    CLEAR(params.force_dither);
    CLEAR(params.dynamic_constants);
    CLEAR(params.allow_delayed_peak_detect);

    pl_hash_merge(&hash, pl_mem_hash(&params, sizeof(params)));
    return hash;
}

#define MAX_MIX_FRAMES 16

bool pl_render_image_mix(pl_renderer rr, const struct pl_frame_mix *images,
                         const struct pl_frame *ptarget,
                         const struct pl_render_params *params)
{
    if (!images->num_frames)
        return pl_render_image(rr, NULL, ptarget, params);

    params = PL_DEF(params, &pl_render_default_params);
    uint64_t params_hash = render_params_hash(params);
    pl_dispatch_mark_dynamic(rr->dp, params->dynamic_constants);

    require(images->num_frames >= 1);
    for (int i = 0; i < images->num_frames - 1; i++)
        require(images->timestamps[i] <= images->timestamps[i+1]);

    // As the canonical reference, find the nearest neighbour frame
    const struct pl_frame *refimg = images->frames[0];
    float best = fabsf(images->timestamps[0]);
    for (int i = 1; i < images->num_frames; i++) {
        float dist = fabsf(images->timestamps[i]);
        if (dist < best) {
            refimg = images->frames[i];
            best = dist;
            continue;
        } else {
            break;
        }
    }

    struct pass_state pass = {
        .rr = rr,
        .params = params,
        .image = *refimg,
        .target = *ptarget,
        .info.stage = PL_RENDER_STAGE_BLEND,
    };

    if (rr->disable_mixing)
        goto fallback;
    if (!pass_infer_state(&pass))
        return false;
    if (!pass.fbofmt[4])
        goto fallback;

    int out_w = abs(pl_rect_w(pass.dst_rect)),
        out_h = abs(pl_rect_h(pass.dst_rect));

    int fidx = 0;
    struct cached_frame frames[MAX_MIX_FRAMES];
    float weights[MAX_MIX_FRAMES];
    float wsum = 0.0;
    pass.tmp = pl_tmp(NULL);

    // Garbage collect the cache by evicting all frames from the cache that are
    // not determined to still be required
    for (int i = 0; i < rr->frames.num; i++)
        rr->frames.elem[i].evict = true;

    // Traverse the input frames and determine/prepare the ones we need
    for (int i = 0; i < images->num_frames; i++) {
        uint64_t sig = images->signatures[i];
        float pts = images->timestamps[i];
        PL_TRACE(rr, "Considering image with signature 0x%llx, pts %f",
                 (unsigned long long) sig, pts);

        // Combining images with different rotations is basically unfeasible
        if (pl_rotation_normalize(images->frames[i]->rotation - refimg->rotation)) {
            PL_TRACE(rr, "  -> Skipping: incompatible rotation");
            continue;
        }

        float weight;
        const struct pl_filter_config *mixer = params->frame_mixer;
        bool single_frame = !mixer || images->num_frames == 1;
        if (single_frame) {

            // Only render the refimg, ignore others
            if (images->frames[i] == refimg) {
                weight = 1.0;
            } else {
                PL_TRACE(rr, "  -> Skipping: no frame mixer");
                continue;
            }

        // For backwards compatibility, treat !kernel as oversample
        } else if (!mixer->kernel || mixer->kernel->weight == oversample) {

            // Compute the visible interval [pts, end] of this frame
            float end = i+1 < images->num_frames ? images->timestamps[i+1] : INFINITY;
            if (pts > images->vsync_duration || end < 0.0) {
                PL_TRACE(rr, "  -> Skipping: no intersection with vsync");
                continue;
            } else {
                pts = PL_MAX(pts, 0.0);
                end = PL_MIN(end, images->vsync_duration);
                pl_assert(end >= pts);
            }

            // Weight is the fraction of vsync interval that frame is visible
            weight = (end - pts) / images->vsync_duration;
            PL_TRACE(rr, "  -> Frame [%f, %f] intersects [%f, %f] = weight %f",
                     pts, end, 0.0, images->vsync_duration, weight);

            if (weight < mixer->kernel->params[0]) {
                PL_TRACE(rr, "     (culling due to threshold)");
                weight = 0.0;
            }

        } else {

            if (fabsf(pts) >= mixer->kernel->radius) {
                PL_TRACE(rr, "  -> Skipping: outside filter radius (%f)",
                         mixer->kernel->radius);
                continue;
            }

            // Weight is directly sampled from the filter
            weight = pl_filter_sample(mixer, pts);
            PL_TRACE(rr, "  -> Filter offset %f = weight %f", pts, weight);

        }

        struct cached_frame *f = NULL;
        for (int j = 0; j < rr->frames.num; j++) {
            if (rr->frames.elem[j].signature == sig) {
                f = &rr->frames.elem[j];
                f->evict = false;
                break;
            }
        }

        // Skip frames with negligible contributions. Do this after the loop
        // above to make sure these frames don't get evicted just yet, and
        // also exclude the reference image from this optimization to ensure
        // that we always have at least one frame.
        const float cutoff = 1e-3;
        if (fabsf(weight) <= cutoff && images->frames[i] != refimg) {
            PL_TRACE(rr, "   -> Skipping: weight (%f) below threshold (%f)",
                     weight, cutoff);
            continue;
        }

        bool skip_cache = single_frame && params->skip_caching_single_frame;
        if (!f && skip_cache) {
            PL_TRACE(rr, "Single frame not found in cache, bypassing");
            goto fallback;
        }

        if (!f) {
            // Signature does not exist in the cache at all yet,
            // so grow the cache by this entry.
            PL_ARRAY_GROW(rr, rr->frames);
            f = &rr->frames.elem[rr->frames.num++];
            *f = (struct cached_frame) {
                .signature = sig,
                .color = images->frames[i]->color,
                .profile = images->frames[i]->profile,
            };
        }

        // Check to see if we can blindly reuse this cache entry. This is the
        // case if either the params are compatible, or the user doesn't care
        bool strict_reuse = !params->preserve_mixing_cache || skip_cache;
        bool can_reuse = f->tex;
        if (can_reuse && strict_reuse) {
            can_reuse = f->tex->params.w == out_w &&
                        f->tex->params.h == out_h &&
                        f->params_hash == params_hash;
        }

        if (!can_reuse && skip_cache) {
            PL_TRACE(rr, "Single frame cache entry invalid, bypassing");
            goto fallback;
        }

        if (!can_reuse) {
            // If we can't reuse the entry, we need to re-render this frame
            PL_TRACE(rr, "  -> Cached texture missing or invalid.. (re)creating");
            if (!f->tex) {
                if (PL_ARRAY_POP(rr->frame_fbos, &f->tex))
                    pl_tex_invalidate(rr->gpu, f->tex);
            }

            bool ok = pl_tex_recreate(rr->gpu, &f->tex, pl_tex_params(
                .w = out_w,
                .h = out_h,
                .format = pass.fbofmt[4],
                .sampleable = true,
                .renderable = true,
                .blit_dst = pass.fbofmt[4]->caps & PL_FMT_CAP_BLITTABLE,
                .storable = pass.fbofmt[4]->caps & PL_FMT_CAP_STORABLE,
            ));

            if (!ok) {
                PL_ERR(rr, "Could not create intermediate texture for "
                       "frame mixing.. disabling!");
                rr->disable_mixing = true;
                goto fallback;
            }

            struct pass_state inter_pass = {
                .rr = rr,
                .tmp = pass.tmp,
                .params = pass.params,
                .fbos_used = pl_calloc(pass.tmp, rr->fbos.num, sizeof(bool)),
                .image = *images->frames[i],
                .target = *ptarget,
                .info.stage = PL_RENDER_STAGE_FRAME,
            };

            // Render a single frame up to `pass_output_target`
            memcpy(inter_pass.fbofmt, pass.fbofmt, sizeof(pass.fbofmt));
            if (!pass_infer_state(&inter_pass))
                goto error;

            pl_dispatch_callback(rr->dp, &inter_pass, info_callback);
            pl_dispatch_reset_frame(rr->dp);
            for (int n = 0; n < params->num_hooks; n++) {
                if (params->hooks[n]->reset)
                    params->hooks[n]->reset(params->hooks[n]->priv);
            }

            if (!pass_read_image(&inter_pass))
                goto error;
            if (!pass_scale_main(&inter_pass))
                goto error;

            pl_assert(inter_pass.img.w == out_w &&
                      inter_pass.img.h == out_h);

            if (inter_pass.img.tex) {
                struct pl_tex_blit_params blit = {
                    .src = inter_pass.img.tex,
                    .dst = f->tex,
                };

                if (blit.src->params.blit_src && blit.dst->params.blit_dst) {
                    pl_tex_blit(rr->gpu, &blit);
                } else {
                    pl_tex_blit_raster(rr->gpu, rr->dp, &blit);
                }
            } else {
                ok = pl_dispatch_finish(rr->dp, pl_dispatch_params(
                    .shader = &inter_pass.img.sh,
                    .target = f->tex,
                ));
                if (!ok)
                    goto error;
            }

            float sx = out_w / pl_rect_w(inter_pass.dst_rect),
                  sy = out_h / pl_rect_h(inter_pass.dst_rect);

            struct pl_transform2x2 shift = {
                .mat.m = {{ sx, 0, }, { 0, sy, }},
                .c = {
                    -sx * inter_pass.dst_rect.x0,
                    -sy * inter_pass.dst_rect.y0
                },
            };

            if (inter_pass.rotation % PL_ROTATION_180 == PL_ROTATION_90) {
                PL_SWAP(shift.mat.m[0][0], shift.mat.m[0][1]);
                PL_SWAP(shift.mat.m[1][0], shift.mat.m[1][1]);
            }

            draw_overlays(&inter_pass, f->tex, inter_pass.img.comps, NULL,
                          inter_pass.image.overlays,
                          inter_pass.image.num_overlays,
                          inter_pass.img.color,
                          inter_pass.img.repr,
                          &shift);

            f->params_hash = params_hash;
            f->color = inter_pass.img.color;
            f->comps = inter_pass.img.comps;
            pl_assert(inter_pass.img.repr.alpha != PL_ALPHA_INDEPENDENT);
        }

        pl_assert(fidx < MAX_MIX_FRAMES);
        frames[fidx] = *f;
        weights[fidx] = weight;
        wsum += weight;
        fidx++;
    }

    // Evict the frames we *don't* need
    for (int i = 0; i < rr->frames.num; ) {
        if (rr->frames.elem[i].evict) {
            PL_TRACE(rr, "Evicting frame with signature %llx from cache",
                     (unsigned long long) rr->frames.elem[i].signature);
            PL_ARRAY_APPEND(rr, rr->frame_fbos, rr->frames.elem[i].tex);
            PL_ARRAY_REMOVE_AT(rr->frames, i);
            continue;
        } else {
            i++;
        }
    }

    // Sample and mix the output color
    pl_dispatch_callback(rr->dp, &pass, info_callback);
    pl_dispatch_reset_frame(rr->dp);
    pass.info.index = fidx;

    pl_shader sh = pl_dispatch_begin(rr->dp);
    sh_describe(sh, "frame mixing");
    sh->res.output = PL_SHADER_SIG_COLOR;
    sh->output_w = out_w;
    sh->output_h = out_h;

    GLSL("vec4 color;                   \n"
         "// pl_render_image_mix        \n"
         "{                             \n"
         "vec4 mix_color = vec4(0.0);   \n");

    // Mix in the image color space, but using the transfer function of
    // (arbitrarily) the latest rendered frame. This avoids unnecessary ping
    // ponging between linear and nonlinear light when combining linearly
    // scaled images with frame mixing.
    struct pl_color_space mix_color = pass.image.color;
    pl_assert(fidx > 0);
    mix_color.transfer = frames[fidx - 1].color.transfer;

    int comps = 0;
    for (int i = 0; i < fidx; i++) {
        const struct pl_tex_params *tpars = &frames[i].tex->params;

        // Use linear sampling if desired and possible
        enum pl_tex_sample_mode sample_mode = PL_TEX_SAMPLE_NEAREST;
        if ((tpars->w != out_w || tpars->h != out_h) &&
            (tpars->format->caps & PL_FMT_CAP_LINEAR))
        {
            sample_mode = PL_TEX_SAMPLE_LINEAR;
        }

        ident_t pos, tex = sh_bind(sh, frames[i].tex, PL_TEX_ADDRESS_CLAMP,
                                   sample_mode, "frame", NULL, &pos, NULL, NULL);

        GLSL("color = %s(%s, %s); \n", sh_tex_fn(sh, *tpars), tex, pos);

        // Note: This ignores differences in ICC profile, which we decide to
        // just simply not care about. Doing that properly would require
        // converting between different image profiles, and the headache of
        // finagling that state is just not worth it because this is an
        // exceptionally unlikely hypothetical.
        pl_shader_color_map(sh, NULL, frames[i].color, mix_color, NULL, false);

        ident_t weight = "1.0";
        if (weights[i] != wsum) { // skip loading weight for nearest neighbour
            weight = sh_var(sh, (struct pl_shader_var) {
                .var = pl_var_float("weight"),
                .data = &(float){ weights[i] / wsum },
                .dynamic = true,
            });
        }

        GLSL("mix_color += %s * color; \n", weight);
        comps = PL_MAX(comps, frames[i].comps);
    }

    GLSL("color = mix_color; \n"
         "}                  \n");

    // Dispatch this to the destination
    pass.fbos_used = pl_calloc(pass.tmp, rr->fbos.num, sizeof(bool));
    pass.img = (struct img) {
        .sh = sh,
        .w = out_w,
        .h = out_h,
        .comps = comps,
        .color = mix_color,
        .repr = {
            .sys = PL_COLOR_SYSTEM_RGB,
            .levels = PL_COLOR_LEVELS_PC,
            .alpha = comps >= 4 ? PL_ALPHA_PREMULTIPLIED : PL_ALPHA_UNKNOWN,
        },
    };

    for (int i = 0; i < params->num_hooks; i++) {
        if (params->hooks[i]->reset)
            params->hooks[i]->reset(params->hooks[i]->priv);
    }

    if (!pass_output_target(&pass))
        goto fallback;

    pl_free(pass.tmp);
    return true;

error:
    PL_ERR(rr, "Could not render image for frame mixing.. disabling!");
    rr->disable_mixing = true;
    // fall through

fallback:
    pl_free(pass.tmp);
    return pl_render_image(rr, refimg, ptarget, params);


}

void pl_frame_set_chroma_location(struct pl_frame *frame,
                                  enum pl_chroma_location chroma_loc)
{
    pl_tex ref = frame->planes[frame_ref(frame)].texture;

    if (ref) {
        // Texture dimensions are already known, so apply the chroma location
        // only to subsampled planes
        int ref_w = ref->params.w, ref_h = ref->params.h;

        for (int i = 0; i < frame->num_planes; i++) {
            struct pl_plane *plane = &frame->planes[i];
            pl_tex tex = plane->texture;
            bool subsampled = tex->params.w < ref_w || tex->params.h < ref_h;
            if (subsampled)
                pl_chroma_location_offset(chroma_loc, &plane->shift_x, &plane->shift_y);
        }
    } else {
        // Texture dimensions are not yet known, so apply the chroma location
        // to all chroma planes, regardless of subsampling
        for (int i = 0; i < frame->num_planes; i++) {
            struct pl_plane *plane = &frame->planes[i];
            if (detect_plane_type(plane, &frame->repr) == PLANE_CHROMA)
                pl_chroma_location_offset(chroma_loc, &plane->shift_x, &plane->shift_y);
        }
    }
}

void pl_frame_from_swapchain(struct pl_frame *out_frame,
                             const struct pl_swapchain_frame *frame)
{
    pl_tex fbo = frame->fbo;
    int num_comps = fbo->params.format->num_components;
    if (!frame->color_repr.alpha)
        num_comps = PL_MIN(num_comps, 3);

    *out_frame = (struct pl_frame) {
        .num_planes = 1,
        .planes = {{
            .texture = fbo,
            .flipped = frame->flipped,
            .components = num_comps,
            .component_mapping = {0, 1, 2, 3},
        }},
        .crop = { 0, 0, fbo->params.w, fbo->params.h },
        .repr = frame->color_repr,
        .color = frame->color_space,
    };
}

bool pl_frame_is_cropped(const struct pl_frame *frame)
{
    int x0 = roundf(PL_MIN(frame->crop.x0, frame->crop.x1)),
        y0 = roundf(PL_MIN(frame->crop.y0, frame->crop.y1)),
        x1 = roundf(PL_MAX(frame->crop.x0, frame->crop.x1)),
        y1 = roundf(PL_MAX(frame->crop.y0, frame->crop.y1));

    pl_tex ref = frame->planes[frame_ref(frame)].texture;
    pl_assert(ref);

    if (!x0 && !x1)
        x1 = ref->params.w;
    if (!y0 && !y1)
        y1 = ref->params.h;

    return x0 > 0 || y0 > 0 || x1 < ref->params.w || y1 < ref->params.h;
}

void pl_frame_clear_rgba(pl_gpu gpu, const struct pl_frame *frame,
                         const float rgba[4])
{
    struct pl_color_repr repr = frame->repr;
    struct pl_transform3x3 tr = pl_color_repr_decode(&repr, NULL);
    pl_transform3x3_invert(&tr);

    float encoded[3] = { rgba[0], rgba[1], rgba[2] };
    pl_transform3x3_apply(&tr, encoded);

    float mult = frame->repr.alpha == PL_ALPHA_PREMULTIPLIED ? rgba[3] : 1.0;
    for (int p = 0; p < frame->num_planes; p++) {
        const struct pl_plane *plane =  &frame->planes[p];
        float clear[4] = { 0.0, 0.0, 0.0, rgba[3] };
        for (int c = 0; c < plane->components; c++) {
            int ch = plane->component_mapping[c];
            if (ch >= 0 && ch < 3)
                clear[c] = mult * encoded[plane->component_mapping[c]];
        }

        pl_tex_clear(gpu, plane->texture, clear);
    }
}
