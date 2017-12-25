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

    // aliases for other systems
    PLANE_Y    = PLANE_R,
    PLANE_CB   = PLANE_G,
    PLANE_CR   = PLANE_B,

    PLANE_CIEX = PLANE_R,
    PLANE_CIEY = PLANE_G,
    PLANE_CIEZ = PLANE_B,
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
    bool disable_linear_hdr; // disable linear scaling for HDR signals
    bool disable_linear_sdr; // disable linear scaling for SDR signals

    // Shader resource objects
    struct pl_shader_obj *peak_detect_state;
    struct pl_shader_obj *dither_state;
    struct pl_shader_obj *upscaler_state; // shared since the LUT is static
    struct pl_shader_obj *downscaler_state[SCALER_COUNT];

    // Intermediate textures (FBOs)
    const struct ra_tex *main_scale_fbo;
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

void pl_renderer_destroy(struct pl_renderer **p_rr)
{
    struct pl_renderer *rr = *p_rr;
    if (!rr)
        return;

    // Free all intermediate FBOs
    ra_tex_destroy(rr->ra, &rr->main_scale_fbo);

    // Free all shader resource objects
    pl_shader_obj_destroy(&rr->peak_detect_state);
    pl_shader_obj_destroy(&rr->dither_state);
    pl_shader_obj_destroy(&rr->upscaler_state);
    for (int i = 0; i < PL_ARRAY_SIZE(rr->downscaler_state); i++)
        pl_shader_obj_destroy(&rr->downscaler_state[i]);

    pl_dispatch_destroy(&rr->dp);
    TA_FREEP(p_rr);
}

void pl_renderer_flush_cache(struct pl_renderer *rr)
{
    // TODO
}

const struct pl_render_params pl_render_default_params = {
    .upscaler         = NULL, // XXX: only until separated works
    .downscaler       = NULL,
    .frame_mixer      = NULL,

    .deband_params    = &pl_deband_default_params,
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

struct pass_state {
    // Represents the "current" image which we're in the process of rendering.
    // This is initially set by pass_read_image, and all of the subsequent
    // rendering steps will mutate this in-place.
    struct img cur_img;
};

static void dispatch_sampler(struct pl_renderer *rr, struct pl_shader *sh,
                             float ratio, int idx,
                             const struct pl_render_params *params,
                             const struct pl_sample_src *src)
{
    if (!rr->fbofmt || rr->disable_sampling)
        goto fallback;

    const struct pl_filter_config *config = NULL;
    struct pl_shader_obj **lut;

    if (ratio > 1.0) {
        config = params->upscaler;
        lut = &rr->upscaler_state;
    } else if (ratio < 1.0) {
        config = params->downscaler;
        lut = &rr->downscaler_state[idx];
    } else { // ratio == 1.0
        goto direct;
    }

    if (!config)
        goto fallback;

    if (config->polar) {
        bool r = pl_shader_sample_polar(sh, src, &(struct pl_sample_polar_params) {
            .filter      = *config,
            .lut_entries = params->lut_entries,
            .lut         = lut,
            .no_compute  = rr->disable_compute,
            .no_widening = params->skip_anti_aliasing,
        });

        if (!r) {
            PL_ERR(rr, "Failed dispatching (polar) scaler.. disabling");
            rr->disable_sampling = true;
            goto fallback;
        }
        return;

    } else { // non-polar
        // TODO
        abort();
    }

fallback:
    // Use bicubic sampling if supported
    if (rr->fbofmt && src->tex->params.sample_mode == RA_TEX_SAMPLE_LINEAR) {
        pl_shader_sample_bicubic(sh, src);
        return;
    }

direct:
    // If all else fails, fall back to bilinear/nearest
    pl_shader_sample_direct(sh, src);
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
    const struct pl_plane *refplane = NULL;
    int best_diff, best_off;

    for (int i = 0; i < image->num_planes; i++) {
        const struct pl_plane *plane = &image->planes[i];
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
        const struct pl_plane *plane = &image->planes[i];
        struct pl_shader *psh = pl_dispatch_begin(rr->dp);
        pl_assert(refplane);

        // Compute the source shift/scale relative to the reference size
        float pw = plane->texture->params.w,
              ph = plane->texture->params.h,
              rx = target_w / pw,
              ry = target_h / ph,
              sx = plane->shift_x - refplane->shift_x,
              sy = plane->shift_y - refplane->shift_y;

        struct pl_sample_src src = {
            .tex        = plane->texture,
            .components = plane->components,
            .new_w      = target_w,
            .new_h      = target_h,
            .rect       = {
                sx / rx,
                sy / ry,
                sx / rx,
                sy / ry,
            },
        };

        dispatch_sampler(rr, psh, PL_MIN(rx, ry), i, params, &src);

        ident_t sub = sh_subpass(sh, psh);
        if (!sub) {
            PL_ERR(sh, "Failed dispatching subpass for plane.. disabling "
                   "scalers");
            rr->disable_sampling = true;
            pl_dispatch_abort(rr->dp, psh);
            pl_dispatch_abort(rr->dp, sh);

            // FIXME: instead of erroring here, instead render out to a cache
            // FBO and sample from that instead
            return false;
        }

        GLSL("tmp = %s();\n", sub);
        for (int c = 0; c < src.components; c++) {
            GLSL("color[%d] = tmp[%d];\n", plane->component_mapping[i],
                 plane->texture->params.format->sample_order[i]);

            has_alpha |= plane->component_mapping[i] == PLANE_A;
        }

        // we don't need it anymore
        pl_dispatch_abort(rr->dp, psh);
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

static const struct ra_tex *finalize_img(struct pl_renderer *rr,
                                         struct img *img,
                                         const struct ra_fmt *fmt,
                                         const struct ra_tex **tex)
{
    pl_assert(fmt);

    if (*tex) {
        const struct ra_tex_params *cur = &(*tex)->params;
        if (cur->w == img->w && cur->h == img->h && cur->format == fmt)
            return *tex;
    }

    PL_INFO(rr, "Resizing texture: %dx%d", img->w, img->h);

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
        return NULL;
    }

    if (!pl_dispatch_finish(rr->dp, img->sh, *tex, NULL)) {
        PL_ERR(rr, "Failed dispatching intermediate pass!");
        return NULL;
    }

    return *tex;
}

static bool pass_scale_main(struct pl_renderer *rr, struct pass_state *pass,
                            const struct pl_image *image,
                            const struct pl_render_target *target,
                            const struct pl_render_params *params)
{
    struct img *img = &pass->cur_img;
    float target_w = fabs(pl_rect_w(target->dst_rect)),
          target_h = fabs(pl_rect_h(target->dst_rect));

    float rx = target_w / fabs(pl_rect_w(image->src_rect)),
          ry = target_w / fabs(pl_rect_h(image->src_rect));

    if (rx == 1.0 && ry == 1.0 && !img->offx && !img->offy) {
        PL_TRACE(rr, "Skipping main scaler (would be no-op)");
        return true;
    }

    if (!rr->fbofmt) {
        PL_TRACE(rr, "Skipping main scaler (no FBOs)");
        return true;
    }

    // TODO: linearization/sigmoidization

    struct pl_sample_src src = {
        .tex        = finalize_img(rr, img, rr->fbofmt, &rr->main_scale_fbo),
        .components = img->comps,
        .new_w      = target_w,
        .new_h      = target_h,
        .rect = {
            image->src_rect.x0 + img->offx,
            image->src_rect.y0 + img->offy,
            image->src_rect.x1 + img->offx,
            image->src_rect.y1 + img->offy,
        },
    };

    if (!src.tex)
        return false;

    struct pl_shader *sh = pl_dispatch_begin(rr->dp);
    dispatch_sampler(rr, sh, PL_MIN(fabs(rx), fabs(ry)), SCALER_MAIN, params, &src);
    pass->cur_img = (struct img) {
        .sh     = sh,
        .w      = target_w,
        .h      = target_h,
        .repr   = img->repr,
        .color  = img->color,
        .comps  = img->comps,
    };

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

    if (params->dither_params) {
        // Just assume the first component's depth is canonical. This works
        // in practice, since for cases like rgb565 we want to use the lower
        // depth anyway. Plus, every format has at least one component.
        int depth = fbo->params.format->component_depth[0];

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
    return pl_dispatch_finish(rr->dp, sh, fbo, &target->dst_rect);
}

bool pl_render_image(struct pl_renderer *rr, const struct pl_image *image,
                     const struct pl_render_target *target,
                     const struct pl_render_params *params)
{
    params = PL_DEF(params, &pl_render_default_params);

    // TODO: validate pl_image correctness
    // TODO: output caching
    pl_dispatch_reset_frame(rr->dp);

    struct pass_state pass = {0};
    if (!pass_read_image(rr, &pass, image, params))
        goto error;

    if (!pass_scale_main(rr, &pass, image, target, params))
        goto error;

    if (!pass_output_target(rr, &pass, target, params))
        goto error;

    return true;

error:
    PL_ERR(rr, "Failed rendering image!");
    return false;
}
