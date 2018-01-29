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

enum {
    // The scalers for each plane are set up to be just the index itself
    SCALER_PLANE0 = 0,
    SCALER_PLANE1 = 1,
    SCALER_PLANE2 = 2,
    SCALER_PLANE3 = 3,

    SCALER_MAIN,
    SCALER_COUNT,
};

// Canonical plane order aliases
enum {
    PLANE_R = 0,
    PLANE_G = 1,
    PLANE_B = 2,
    PLANE_A = 3,
    PLANE_COUNT,

    // aliases for other systems
    PLANE_Y    = PLANE_R,
    PLANE_CB   = PLANE_G,
    PLANE_CR   = PLANE_B,

    PLANE_CIEX = PLANE_R,
    PLANE_CIEY = PLANE_G,
    PLANE_CIEZ = PLANE_B,
};

struct sampler {
    struct pl_shader_obj *upscaler_state;
    struct pl_shader_obj *downscaler_state;
    const struct ra_tex *sep_fbo_up;
    const struct ra_tex *sep_fbo_down;
};

struct pl_renderer {
    const struct ra *ra;
    struct pl_context *ctx;
    struct pl_dispatch *dp;

    // Texture format to use for intermediate textures
    const struct ra_fmt *fbofmt;

    // Cached feature checks (inverted)
    bool disable_compute;    // disable the use of compute shaders
    bool disable_sampling;   // disable use of advanced scalers
    bool disable_debanding;  // disable the use of debanding shaders
    bool disable_linear_hdr; // disable linear scaling for HDR signals
    bool disable_linear_sdr; // disable linear scaling for SDR signals
    bool disable_blending;   // disable blending for the target/fbofmt
    bool disable_overlay;    // disable rendering overlays

    // Shader resource objects and intermediate textures (FBOs)
    struct pl_shader_obj *peak_detect_state;
    struct pl_shader_obj *dither_state;
    const struct ra_tex *main_scale_fbo;
    const struct ra_tex *deband_fbos[PLANE_COUNT];
    struct sampler samplers[SCALER_COUNT];
    struct sampler *osd_samplers;
    int num_osd_samplers;
};

static void find_fbo_format(struct pl_renderer *rr)
{
    struct {
        enum ra_fmt_type type;
        int depth;
        enum ra_fmt_caps caps;
    } configs[] = {
        // Prefer floating point formats first
        {RA_FMT_FLOAT, 16, RA_FMT_CAP_LINEAR},
        {RA_FMT_FLOAT, 16, RA_FMT_CAP_SAMPLEABLE},

        // Otherwise, fall back to unorm/snorm, preferring linearly sampleable
        {RA_FMT_UNORM, 16, RA_FMT_CAP_LINEAR},
        {RA_FMT_SNORM, 16, RA_FMT_CAP_LINEAR},
        {RA_FMT_UNORM, 16, RA_FMT_CAP_SAMPLEABLE},
        {RA_FMT_SNORM, 16, RA_FMT_CAP_SAMPLEABLE},

        // As a final fallback, allow 8-bit FBO formats (for UNORM only)
        {RA_FMT_UNORM, 8, RA_FMT_CAP_LINEAR},
        {RA_FMT_UNORM, 8, RA_FMT_CAP_SAMPLEABLE},
    };

    for (int i = 0; i < PL_ARRAY_SIZE(configs); i++) {
        const struct ra_fmt *fmt;
        fmt = ra_find_fmt(rr->ra, configs[i].type, 4, configs[i].depth, 0,
                          configs[i].caps | RA_FMT_CAP_RENDERABLE);
        if (fmt) {
            rr->fbofmt = fmt;
            break;
        }
    }

    if (!rr->fbofmt) {
        PL_WARN(rr, "Found no renderable FBO format! Most features disabled");
        return;
    }

    if (!(rr->fbofmt->caps & RA_FMT_CAP_STORABLE)) {
        PL_INFO(rr, "Found no storable FBO format; compute shaders disabled");
        rr->disable_compute = true;
    }

    if (rr->fbofmt->type != RA_FMT_FLOAT) {
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
                                       const struct ra *ra)
{
    struct pl_renderer *rr = talloc_ptrtype(NULL, rr);
    *rr = (struct pl_renderer) {
        .ra  = ra,
        .ctx = ctx,
        .dp  = pl_dispatch_create(ctx, ra),
    };

    assert(rr->dp);
    find_fbo_format(rr);
    return rr;
}

static void sampler_destroy(struct pl_renderer *rr, struct sampler *sampler)
{
    pl_shader_obj_destroy(&sampler->upscaler_state);
    pl_shader_obj_destroy(&sampler->downscaler_state);
    ra_tex_destroy(rr->ra, &sampler->sep_fbo_up);
    ra_tex_destroy(rr->ra, &sampler->sep_fbo_down);
}

void pl_renderer_destroy(struct pl_renderer **p_rr)
{
    struct pl_renderer *rr = *p_rr;
    if (!rr)
        return;

    // Free all intermediate FBOs
    ra_tex_destroy(rr->ra, &rr->main_scale_fbo);
    for (int i = 0; i < PL_ARRAY_SIZE(rr->deband_fbos); i++)
        ra_tex_destroy(rr->ra, &rr->deband_fbos[i]);

    // Free all shader resource objects
    pl_shader_obj_destroy(&rr->peak_detect_state);
    pl_shader_obj_destroy(&rr->dither_state);

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
    // TODO
}

const struct pl_render_params pl_render_default_params = {
    .upscaler         = &pl_filter_spline36,
    .downscaler       = &pl_filter_mitchell,
    .frame_mixer      = NULL,

    .deband_params    = &pl_deband_default_params,
    .sigmoid_params   = &pl_sigmoid_default_params,
    .color_map_params = &pl_color_map_default_params,
    .dither_params    = &pl_dither_default_params,
};

// Represents a "in-flight" image, which is a shader that's in the process of
// producing some sort of image
struct img {
    struct pl_shader *sh;
    int w, h;

    // Accumulated texture offset, which will need to be accounted for by
    // the main scaler.
    float offx, offy;

    // The current effective colorspace
    struct pl_color_repr repr;
    struct pl_color_space color;
    int comps;
};

static const struct ra_tex *finalize_img(struct pl_renderer *rr,
                                         struct img *img,
                                         const struct ra_fmt *fmt,
                                         const struct ra_tex **tex)
{
    pl_assert(fmt);

    if (*tex) {
        const struct ra_tex_params *cur = &(*tex)->params;
        if (cur->w == img->w && cur->h == img->h && cur->format == fmt)
            goto resized;
    }

    PL_INFO(rr, "Resizing intermediate FBO texture: %dx%d", img->w, img->h);

    ra_tex_destroy(rr->ra, tex);
    *tex = ra_tex_create(rr->ra, &(struct ra_tex_params) {
        .w = img->w,
        .h = img->h,
        .format = fmt,
        .sampleable = true,
        .renderable = true,
        // Just enable what we can
        .storable   = !!(fmt->caps & RA_FMT_CAP_STORABLE),
        .sample_mode = (fmt->caps & RA_FMT_CAP_LINEAR)
                            ? RA_TEX_SAMPLE_LINEAR
                            : RA_TEX_SAMPLE_NEAREST,
    });

    if (!*tex) {
        PL_ERR(rr, "Failed creating FBO texture! Disabling advanced rendering..");
        rr->fbofmt = NULL;
        pl_dispatch_abort(rr->dp, &img->sh);
        return NULL;
    }


resized:
    if (!pl_dispatch_finish(rr->dp, &img->sh, *tex, NULL, NULL)) {
        PL_ERR(rr, "Failed dispatching intermediate pass!");
        return NULL;
    }

    return *tex;
}

struct pass_state {
    // Represents the "current" image which we're in the process of rendering.
    // This is initially set by pass_read_image, and all of the subsequent
    // rendering steps will mutate this in-place.
    struct img cur_img;
};

static void dispatch_sampler(struct pl_renderer *rr, struct pl_shader *sh,
                             float ratio, struct sampler *sampler,
                             const struct pl_render_params *params,
                             const struct pl_sample_src *src)
{
    if (!rr->fbofmt || rr->disable_sampling || !sampler)
        goto fallback;

    const struct pl_filter_config *config = NULL;
    bool is_linear = src->tex->params.sample_mode == RA_TEX_SAMPLE_LINEAR;
    struct pl_shader_obj **lut;
    const struct ra_tex **sep_fbo;

    if (ratio > 1.0) {
        config = params->upscaler;
        lut = &sampler->upscaler_state;
        sep_fbo = &sampler->sep_fbo_up;
    } else if (ratio < 1.0) {
        config = params->downscaler;
        lut = &sampler->downscaler_state;
        sep_fbo = &sampler->sep_fbo_down;
    } else { // ratio == 1.0
        goto direct;
    }

    if (!config)
        goto fallback;

    // Try using faster replacements for GPU built-in scalers
    bool can_fast = ratio > 1.0 || params->skip_anti_aliasing;
    if (can_fast && !params->disable_builtin_scalers) {
        if (is_linear && config == &pl_filter_bicubic)
            goto fallback; // the bicubic check will succeed
        if (is_linear && config == &pl_filter_triangle)
            goto direct;
        if (!is_linear && config == &pl_filter_box)
            goto direct;
    }

    struct pl_sample_filter_params fparams = {
        .filter      = *config,
        .lut_entries = params->lut_entries,
        .cutoff      = params->polar_cutoff,
        .antiring    = params->antiringing_strength,
        .no_compute  = rr->disable_compute,
        .no_widening = params->skip_anti_aliasing,
        .lut         = lut,
    };

    bool ok;
    if (config->polar) {
        ok = pl_shader_sample_polar(sh, src, &fparams);
    } else {
        struct pl_shader *tsh = pl_dispatch_begin(rr->dp);
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
        src2.tex = finalize_img(rr, &img, rr->fbofmt, sep_fbo);
        ok = pl_shader_sample_ortho(sh, PL_SEP_HORIZ, &src2, &fparams);
    }

done:
    if (!ok) {
        PL_ERR(rr, "Failed dispatching scaler.. disabling");
        rr->disable_sampling = true;
        goto fallback;
    }

    return;

fallback:
    // Use bicubic sampling if supported
    if (rr->fbofmt && is_linear) {
        pl_shader_sample_bicubic(sh, src);
        return;
    }

direct:
    // If all else fails, fall back to bilinear/nearest
    pl_shader_sample_direct(sh, src);
}

static void deband_plane(struct pl_renderer *rr, struct pl_plane *plane,
                         const struct ra_tex **fbo,
                         const struct pl_render_params *params)
{
    if (!rr->fbofmt || rr->disable_debanding || !params->deband_params)
        return;

    const struct ra_tex *tex = plane->texture;
    if (tex->params.sample_mode != RA_TEX_SAMPLE_LINEAR) {
        PL_WARN(rr, "Debanding requires uploaded textures to be linearly "
                "sampleable (params.sample_mode = RA_TEX_SAMPLE_LINEAR)! "
                "Disabling debanding..");
        rr->disable_debanding = true;
        return;
    }

    struct pl_shader *sh = pl_dispatch_begin(rr->dp);
    pl_shader_deband(sh, tex, params->deband_params);

    struct img img = {
        .sh = sh,
        .w  = tex->params.w,
        .h  = tex->params.h,
    };

    const struct ra_tex *new = finalize_img(rr, &img, rr->fbofmt, fbo);
    if (!new) {
        PL_ERR(rr, "Failed dispatching debanding shader.. disabling debanding!");
        rr->disable_debanding = true;
        return;
    }

    plane->texture = new;
}

// This scales and merges all of the source images, and initializes the cur_img.
static bool pass_read_image(struct pl_renderer *rr, struct pass_state *pass,
                            const struct pl_image *image,
                            const struct pl_render_params *params)
{
    struct pl_shader *sh = pl_dispatch_begin(rr->dp);
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
    struct pl_plane planes[4];
    const struct pl_plane *refplane = NULL; // points to one of `planes`
    int best_diff, best_off;

    pl_assert(image->num_planes < PLANE_COUNT);
    for (int i = 0; i < image->num_planes; i++) {
        struct pl_plane *plane = &planes[i];
        *plane = image->planes[i];

        const struct ra_tex *tex = plane->texture;
        int diff = PL_MAX(abs(tex->params.w - image->width),
                          abs(tex->params.h - image->height));
        int off = PL_MAX(plane->shift_x, plane->shift_y);

        if (!refplane || diff < best_diff || (diff == best_diff && off < best_off)) {
            refplane = plane;
            best_diff = diff;
            best_off = off;
        }
    }

    if (!refplane) {
        PL_ERR(rr, "Image contains no planes?");
        return false;
    }

    float target_w = refplane->texture->params.w,
          target_h = refplane->texture->params.h;
    bool has_alpha = false;

    for (int i = 0; i < image->num_planes; i++) {
        struct pl_shader *psh = pl_dispatch_begin(rr->dp);
        struct pl_plane *plane = &planes[i];
        deband_plane(rr, plane, &rr->deband_fbos[i], params);

        // Compute the source shift/scale relative to the reference size
        float pw = plane->texture->params.w,
              ph = plane->texture->params.h,
              rx = target_w / pw,
              ry = target_h / ph,
              sx = plane->shift_x - refplane->shift_x,
              sy = plane->shift_y - refplane->shift_y;

        // Only accept integer scaling ratios. This accounts for the fact
        // that fractionally subsampled planes get rounded up to the nearest
        // integer size, which we want to discard.
        float rrx = rx >= 1 ? roundf(rx) : 1.0 / roundf(1.0 / rx),
              rry = ry >= 1 ? roundf(ry) : 1.0 / roundf(1.0 / ry),
              rpw = target_w / rrx,
              rph = target_h / rry;

        struct pl_sample_src src = {
            .tex        = plane->texture,
            .components = plane->components,
            .new_w      = target_w,
            .new_h      = target_h,
            .rect       = {
                -sx / rx,
                -sy / ry,
                rpw - sx / rx,
                rph - sy / ry,
            },
        };

        // FIXME: in theory, we could reuse the debanding result from
        // `deband_plane` if available, instead of having to dispatch a no-op
        // shader, for trivial sampling cases (no scaling or shifting)
        dispatch_sampler(rr, psh, PL_MIN(rx, ry), &rr->samplers[i], params, &src);

        ident_t sub = sh_subpass(sh, psh);
        if (!sub) {
            PL_ERR(sh, "Failed dispatching subpass for plane.. disabling "
                   "scalers");
            rr->disable_sampling = true;
            pl_dispatch_abort(rr->dp, &psh);
            pl_dispatch_abort(rr->dp, &sh);

            // FIXME: instead of erroring here, instead render out to a cache
            // FBO and sample from that instead
            return false;
        }

        GLSL("tmp = %s();\n", sub);
        for (int c = 0; c < src.components; c++) {
            if (plane->component_mapping[c] < 0)
                continue;
            GLSL("color[%d] = tmp[%d];\n", plane->component_mapping[c],
                 plane->texture->params.format->sample_order[c]);

            has_alpha |= plane->component_mapping[c] == PLANE_A;
        }

        // we don't need it anymore
        pl_dispatch_abort(rr->dp, &psh);
    }

    pass->cur_img = (struct img) {
        .sh     = sh,
        .w      = target_w,
        .h      = target_h,
        .offx   = refplane->shift_x,
        .offy   = refplane->shift_y,
        .repr   = image->repr,
        .color  = image->color,
        .comps  = has_alpha ? 4 : 3,
    };

    // Convert the image colorspace
    pl_shader_decode_color(sh, &pass->cur_img.repr, params->color_adjustment);
    GLSL("}\n");
    return true;
}

static bool pass_scale_main(struct pl_renderer *rr, struct pass_state *pass,
                            const struct pl_image *image,
                            const struct pl_render_target *target,
                            const struct pl_render_params *params)
{
    struct img *img = &pass->cur_img;
    int target_w = abs(pl_rect_w(target->dst_rect)),
        target_h = abs(pl_rect_h(target->dst_rect));

    float rx = target_w / fabs(pl_rect_w(image->src_rect)),
          ry = target_h / fabs(pl_rect_h(image->src_rect));

    if (rx == 1.0 && ry == 1.0 && !img->offx && !img->offy) {
        PL_TRACE(rr, "Skipping main scaler (would be no-op)");
        return true;
    }

    if (!rr->fbofmt) {
        PL_TRACE(rr, "Skipping main scaler (no FBOs)");
        return true;
    }

    bool downscaling = rx < 1.0 || ry < 1.0;
    bool upscaling = !downscaling && (rx > 1.0 || ry > 1.0);

    bool use_sigmoid = upscaling && params->sigmoid_params;
    bool use_linear  = use_sigmoid || downscaling;

    // Hard-disable both sigmoidization and linearization when requested
    if (params->disable_linear_scaling)
        use_sigmoid = use_linear = false;

    // Avoid sigmoidization for HDR content because it clips to [0,1]
    if (pl_color_transfer_is_hdr(img->color.transfer))
        use_sigmoid = false;

    if (use_linear) {
        pl_shader_linearize(img->sh, img->color.transfer);
        img->color.transfer = PL_COLOR_TRC_LINEAR;
    }

    if (use_sigmoid)
        pl_shader_sigmoidize(img->sh, params->sigmoid_params);

    struct pl_sample_src src = {
        .tex        = finalize_img(rr, img, rr->fbofmt, &rr->main_scale_fbo),
        .components = img->comps,
        .new_w      = target_w,
        .new_h      = target_h,
        .rect = {
            image->src_rect.x0 - img->offx,
            image->src_rect.y0 - img->offy,
            image->src_rect.x1 - img->offx,
            image->src_rect.y1 - img->offy,
        },
    };

    // TODO: render overlay on top of the image

    if (!src.tex)
        return false;

    float ratio = PL_MIN(fabs(rx), fabs(ry));
    struct pl_shader *sh = pl_dispatch_begin(rr->dp);
    dispatch_sampler(rr, sh, ratio, &rr->samplers[SCALER_MAIN], params, &src);
    pass->cur_img = (struct img) {
        .sh     = sh,
        .w      = target_w,
        .h      = target_h,
        .repr   = img->repr,
        .color  = img->color,
        .comps  = img->comps,
    };

    if (use_sigmoid)
        pl_shader_unsigmoidize(sh, params->sigmoid_params);

    return true;
}

static bool pass_output_target(struct pl_renderer *rr, struct pass_state *pass,
                               const struct pl_render_target *target,
                               const struct pl_render_params *params)
{
    const struct ra_tex *fbo = target->fbo;

    // Color management
    struct pl_shader *sh = pass->cur_img.sh;
    pl_shader_color_map(sh, params->color_map_params, pass->cur_img.color,
                        target->color, &rr->peak_detect_state, false);
    pl_shader_encode_color(sh, &target->repr);

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
        if (depth <= 16)
            pl_shader_dither(sh, depth, &rr->dither_state, params->dither_params);
    }

    bool is_comp = pl_shader_is_compute(sh);
    if (is_comp && !fbo->params.storable) {
        // TODO: force caching
        abort();
    }

    pl_assert(fbo->params.renderable);
    return pl_dispatch_finish(rr->dp, &pass->cur_img.sh, fbo,
                              &target->dst_rect, NULL);
}

static void pass_draw_overlay(struct pl_renderer *rr,
                              const struct pl_render_target *target,
                              const struct pl_render_params *params)
{
    if (!target->num_overlays || rr->disable_overlay)
        return;

    enum ra_fmt_caps caps = target->fbo->params.format->caps;
    if (!rr->disable_blending && !(caps & RA_FMT_CAP_BLENDABLE)) {
        PL_WARN(rr, "Trying to draw an overlay to a non-blendable target. "
                "Alpha blending is disabled, results may be incorrect!");
        rr->disable_blending = true;
    }

    while (target->num_overlays > rr->num_osd_samplers) {
        TARRAY_APPEND(rr, rr->osd_samplers, rr->num_osd_samplers,
                      (struct sampler) {0});
    }

    for (int n = 0; n < target->num_overlays; n++) {
        const struct pl_overlay *ol = &target->overlays[n];
        const struct pl_plane *plane = &ol->plane;
        const struct ra_tex *tex = plane->texture;

        struct pl_sample_src src = {
            .tex        = tex,
            .components = ol->mode == PL_OVERLAY_MONOCHROME ? 1 : plane->components,
            .new_w      = abs(pl_rect_w(ol->rect)),
            .new_h      = abs(pl_rect_h(ol->rect)),
            .rect = {
                -plane->shift_x,
                -plane->shift_y,
                tex->params.w - plane->shift_x,
                tex->params.h - plane->shift_y,
            },
        };

        float rx = (float) src.new_w / src.tex->params.w,
              ry = (float) src.new_h / src.tex->params.h;
        float ratio = PL_MIN(fabs(rx), fabs(ry));

        struct sampler *sampler = &rr->osd_samplers[n];
        if (params->disable_overlay_sampling)
            sampler = NULL;

        struct pl_shader *sh = pl_dispatch_begin(rr->dp);
        dispatch_sampler(rr, sh, ratio, sampler, params, &src);

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
                .var  = ra_var_vec3("base_color"),
                .data = &ol->base_color,
                .dynamic = true,
            }));
            break;
        default: abort();
        }

        struct pl_color_repr repr = ol->repr;
        pl_shader_decode_color(sh, &repr, NULL);
        pl_shader_color_map(sh, params->color_map_params, ol->color,
                            target->color, NULL, false);

        static const struct ra_blend_params blend_params = {
            .src_rgb = RA_BLEND_SRC_ALPHA,
            .dst_rgb = RA_BLEND_ONE_MINUS_SRC_ALPHA,
            .src_alpha = RA_BLEND_ONE,
            .dst_alpha = RA_BLEND_ONE_MINUS_SRC_ALPHA,
        };

        const struct ra_blend_params *blend = &blend_params;
        if (rr->disable_blending)
            blend = NULL;

        if (!pl_dispatch_finish(rr->dp, &sh, target->fbo, &ol->rect, blend)) {
            PL_ERR(rr, "Failed rendering overlay texture!");
            rr->disable_overlay = true;
            return;
        }
    }
}

static void fix_rects(struct pl_image *image, struct pl_render_target *target)
{
    pl_assert(image->width && image->height);

    // Initialize the rects to the full size if missing
    if ((!image->src_rect.x0 && !image->src_rect.x1) ||
        (!image->src_rect.y0 && !image->src_rect.y1))
    {
        image->src_rect = (struct pl_rect2df) {
            0, 0, image->width, image->height,
        };
    }

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

    struct pl_image image = *pimage;
    struct pl_render_target target = *ptarget;
    fix_rects(&image, &target);

    // TODO: output caching
    pl_dispatch_reset_frame(rr->dp);

    struct pass_state pass = {0};
    if (!pass_read_image(rr, &pass, &image, params))
        goto error;

    if (!pass_scale_main(rr, &pass, &image, &target, params))
        goto error;

    if (!pass_output_target(rr, &pass, &target, params))
        goto error;

    pass_draw_overlay(rr, &target, params);
    return true;

error:
    pl_dispatch_abort(rr->dp, &pass.cur_img.sh);
    PL_ERR(rr, "Failed rendering image!");
    return false;
}

void pl_render_target_from_swapchain(struct pl_render_target *out_target,
                                     const struct ra_swapchain_frame *frame)
{
    const struct ra_tex *fbo = frame->fbo;
    *out_target = (struct pl_render_target) {
        .fbo = fbo,
        .dst_rect = { 0, 0, fbo->params.w, fbo->params.h },
        .repr = frame->color_repr,
        .color = frame->color_space,
    };
}
