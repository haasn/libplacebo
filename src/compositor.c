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
#include <float.h>

#include "common.h"
#include "log.h"

#include <libplacebo/compositor.h>

struct priv {
    pl_log log;
    pl_gpu gpu;

    // Used for the final composition step
    pl_renderer rr;
    pl_fmt fmt;
    pl_tex tex;

    // List of pointers, sorted by z_index
    PL_ARRAY(pl_input) sorted;

    // For pl_compositor_status
    PL_ARRAY(struct pl_input_status) input_status;
    PL_ARRAY(pl_input) visible; // subset of `sorted`

    // For pl_compositor_get_errors()
    PL_ARRAY(uint64_t) disabled_hooks;
    enum pl_render_error errors;

    // Temporary info for info_callback()
    pl_input active_input;
    void (*info_callback)(void *priv, const struct pl_render_info *info);
    void *info_priv;
};

struct priv_input {
    int index; // index into pl_compositor.inputs[]
    enum pl_queue_status status;
    struct pl_frame_mix mix;

    // Mutable copies of the frames in `mix`
    PL_ARRAY(struct pl_frame) frames;

    struct pl_rect2df cached_src_rect;
    struct pl_rect2df cached_dst_rect;

    // Rounded / resolved values
    double start_pts;
    double stop_pts;
};

static const struct pl_render_params *input_params(pl_input input)
{
    return PL_DEF(input->params, &pl_render_default_params);
}

static const struct pl_render_params *canvas_params(pl_compositor comp)
{
    return PL_DEF(comp->params, &pl_render_default_params);
}

// Effective clear modes, accounting for deprecated fields
static enum pl_clear_mode get_border(const struct pl_render_params *par)
{
    return par->skip_target_clearing ? PL_CLEAR_SKIP : par->border;
}

static enum pl_clear_mode get_background(const struct pl_render_params *par)
{
    if (par->blend_against_tiles)
        return PL_CLEAR_TILES;
    return par->skip_target_clearing ? PL_CLEAR_SKIP : par->background;
}

// Overrides all fields of `par` taken from the canvas params, for rendering
// either the lowest visible input (`base`) or any input on top of it. When
// not `direct`, inputs are rendered onto the intermediate canvas instead.
static void set_canvas_params(struct pl_render_params *par,
                              const struct pl_render_params *canvas,
                              bool base, bool direct)
{
    par->background = base ? get_background(canvas) : PL_CLEAR_SKIP;
    par->border     = base ? get_border(canvas) : PL_CLEAR_SKIP;
    memcpy(par->background_color, canvas->background_color,
           sizeof(par->background_color));
    par->background_transparency = canvas->background_transparency;
    memcpy(par->tile_colors, canvas->tile_colors, sizeof(par->tile_colors));
    par->tile_size            = canvas->tile_size;
    par->blur_radius          = canvas->blur_radius;
    par->skip_target_clearing = false;
    par->blend_against_tiles  = false;

    if (!base) {
        par->blend_params = &pl_alpha_overlay;
    } else {
        par->blend_params = direct ? canvas->blend_params : NULL;
    }

    // Dithering only makes sense when rendering to the target frame
    par->dither_params   = direct ? canvas->dither_params : NULL;
    par->error_diffusion = direct ? canvas->error_diffusion : NULL;
    par->force_dither    = direct && canvas->force_dither;
    par->disable_dither_gamma_correction
        = canvas->disable_dither_gamma_correction;
}

// Render params for rendering the canvas onto the target frame
static void get_canvas_params(struct pl_render_params *par, pl_compositor comp)
{
    *par = *canvas_params(comp);

    // Image processing only applies to the inputs
    par->frame_mixer        = NULL;
    par->deband_params      = NULL;
    par->sigmoid_params     = NULL;
    par->color_adjustment   = NULL;
    par->peak_detect_params = NULL;
    par->cone_params        = NULL;
    par->deinterlace_params = NULL;
    par->distort_params     = NULL;
    par->hooks              = NULL;
    par->num_hooks          = 0;
    par->lut                = NULL;
    par->corner_rounding    = 0.0f;
    par->transparency       = 0.0f;
}

static void info_callback(void *priv, const struct pl_render_info *info)
{
    pl_compositor comp = priv;
    struct priv *p = PL_PRIV(comp);
    if (p->info_callback)
        p->info_callback(p->info_priv, info);
    comp->info_callback(comp->info_priv, p->active_input, info);
}

static void set_info_callback(struct pl_render_params *par,
                              pl_compositor comp, pl_input input)
{
    if (!comp->info_callback)
        return;

    struct priv *p = PL_PRIV(comp);
    p->active_input = input;
    p->info_callback = par->info_callback;
    p->info_priv = par->info_priv;

    par->info_callback = info_callback;
    par->info_priv = comp;
}

void pl_compositor_options_set(pl_compositor comp, pl_options opts)
{
    comp->params = &opts->params;
    for (int i = 0; i < comp->num_inputs; i++)
        comp->inputs[i]->params = &opts->params;
}

static void input_uninit(pl_input input)
{
    pl_renderer_destroy(&input->renderer);
    pl_queue_destroy(&input->queue);
}

void pl_compositor_destroy(pl_compositor *pcomp)
{
    pl_compositor comp = *pcomp;
    if (!comp)
        return;

    struct priv *p = PL_PRIV(comp);
    for (int i = 0; i < comp->num_inputs; i++)
        input_uninit(comp->inputs[i]);

    if (p->gpu) {
        pl_tex_destroy(p->gpu, &p->tex);
        pl_renderer_destroy(&p->rr);
    }

    pl_free((void *) comp);
    *pcomp = NULL;
}

pl_compositor pl_compositor_create(pl_log log)
{
    pl_compositor comp = pl_zalloc_obj(NULL, comp, struct priv);
    struct priv *p = PL_PRIV(comp);
    p->log = log;
    return comp;
}

bool pl_compositor_init(pl_compositor comp, pl_gpu gpu)
{
    struct priv *p = PL_PRIV(comp);
    if (p->gpu) {
        PL_ERR(p, "Trying to initialize a compositor that was already "
               "initialized!");
        return false;
    }

    p->gpu = gpu;
    p->rr = pl_renderer_create(p->log, gpu);

    const enum pl_fmt_caps caps = PL_FMT_CAP_BLENDABLE | PL_FMT_CAP_BLITTABLE;
    p->fmt = pl_find_fmt(p->gpu, PL_FMT_FLOAT, 4, 16, 0, caps);

    // Retroactively initialize any inputs created before the compositor
    for (int i = 0; i < comp->num_inputs; i++) {
        pl_input input = comp->inputs[i];
        input->queue    = pl_queue_create(gpu);
        input->renderer = pl_renderer_create(p->log, gpu);
    }
    return true;
}

pl_input pl_compositor_input_create(pl_compositor comp)
{
    struct priv *p = PL_PRIV(comp);

    pl_input input = pl_zalloc_obj(comp, input, struct priv_input);
    struct priv_input *pi = PL_PRIV(input);
    pi->index = comp->num_inputs;
    PL_ARRAY_APPEND_RAW(comp, comp->inputs, comp->num_inputs, input);
    PL_ARRAY_APPEND(comp, p->sorted, input);

    input->start_pts  =  0.0;
    input->stop_pts   =  DBL_MAX;
    input->start_mode = PL_PTS_ROUND_NEAREST;
    input->stop_mode  = PL_PTS_ROUND_NEAREST;
    input->seek_pts   =  0.0;
    input->src_rect   = (struct pl_rect2df) { 0, 0, 1, 1 };
    input->dst_rect   = (struct pl_rect2df) { 0, 0, 1, 1 };
    input->focus      = (pl_vec2df) { 0.5f, 0.5f };

    if (p->gpu) {
        input->queue      = pl_queue_create(p->gpu);
        input->renderer   = pl_renderer_create(p->log, p->gpu);
    }

    pi->status    = PL_QUEUE_OK;
    pi->start_pts = pi->stop_pts = NAN;
    return input;
}

void pl_compositor_input_destroy(pl_compositor comp, pl_input *pinput)
{
    pl_input input = *pinput;
    if (!input)
        return;

    struct priv *p = PL_PRIV(comp);
    const int idx = ((struct priv_input *) PL_PRIV(input))->index;
    pl_assert(idx < comp->num_inputs && comp->inputs[idx] == input);

    PL_ARRAY_REMOVE_AT_RAW(comp->inputs, comp->num_inputs, idx);
    for (int i = idx; i < comp->num_inputs; i++)
        ((struct priv_input *) PL_PRIV(comp->inputs[i]))->index = i;

    for (int i = 0; i < p->sorted.num; i++) {
        if (p->sorted.elem[i] == input) {
            PL_ARRAY_REMOVE_AT(p->sorted, i);
            break;
        }
    }

    input_uninit(input);
    pl_free((void *) input);
    *pinput = NULL;
}

// Return whether `pts` has passed the corresponding output frame boundary
static bool pts_passed(double pts, enum pl_pts_round_mode mode,
                       const struct pl_queue_params *params)
{
    const double pts_out = params->pts;
    const float vsync = params->vsync_duration;
    if (!vsync)
        mode = PL_PTS_ROUND_UP;

    const double eps = 1e-6; // compensate for double/float mismatch in API
    switch (mode) {
    case PL_PTS_ROUND_UP:      return pts_out >= pts - eps;
    case PL_PTS_ROUND_NEAREST: return pts_out > pts - vsync / 2 + eps;
    case PL_PTS_ROUND_DOWN:    return pts_out > pts - vsync + eps;
    case PL_PTS_ROUND_COUNT:   break;
    }

    pl_unreachable();
}

struct get_frame_priv {
    pl_input input;
    void *priv_orig;
};

static enum pl_queue_status get_frame(struct pl_source_frame *out_frame,
                                      const struct pl_queue_params *params)
{
    const struct get_frame_priv *p = params->priv;
    struct pl_queue_params qparams = *params;
    qparams.priv = p->priv_orig;
    return p->input->get_frame(p->input, out_frame, &qparams);
}

enum pl_queue_status pl_input_update(pl_input input,
                                     struct pl_input_status *out_status,
                                     const struct pl_queue_params *params)
{
    if (!input->queue) {
        if (out_status)
            *out_status = (struct pl_input_status) {0};
        return PL_QUEUE_OK; // no frames to display
    }

    struct priv_input *pi = PL_PRIV(input);
    const double pts = params->pts;
    if (pi->status == PL_QUEUE_EOF || pi->status == PL_QUEUE_ERR)
        goto done;

    const bool started = pts_passed(input->start_pts, input->start_mode, params);
    const bool stopped = pts_passed(input->stop_pts,  input->stop_mode,  params);

    struct get_frame_priv gp = { input, params->priv };
    struct pl_queue_params qparams = *params;
    qparams.get_frame = input->get_frame ? get_frame : NULL;
    qparams.priv = &gp;
    qparams.radius = pl_frame_mix_radius(input_params(input));

    if (stopped) {
        pi->status = PL_QUEUE_EOF;
    } else if (isnan(pi->start_pts) && !started) {
        // Not yet visible, but advance the queue along the input's own
        // timeline anyway, to discard any frames from before `seek_pts`.
        // The queue starts out at PTS 0, and can't go backwards from there.
        qparams.pts = pts - input->start_pts + input->seek_pts;
        qparams.pts = fmax(qparams.pts, PL_MIN(0.0, input->seek_pts));
        qparams.timeout = 0;
        pi->status = pl_queue_update(input->queue, NULL, &qparams);
    } else {
        if (isnan(pi->start_pts))
            pi->start_pts = pts;

        qparams.pts = pts - pi->start_pts + input->seek_pts;
        pi->status = pl_queue_update(input->queue, &pi->mix, &qparams);

        // Make local mutable copies of the frame structs and rewrite `mix`
        PL_ARRAY_RESIZE(input, pi->frames, pi->mix.num_frames);
        pi->frames.num = pi->mix.num_frames;

        for (int i = 0; i < pi->mix.num_frames; i++) {
            struct pl_frame *frame = &pi->frames.elem[i];
            *frame = *pi->mix.frames[i];
            pi->mix.frames[i] = frame;

            // Apply input-level colorspace overrides
            struct pl_color_repr  repr  = input->repr;
            struct pl_color_space color = input->color;
            pl_color_repr_merge(&repr, &frame->repr);
            pl_color_space_merge(&color, &frame->color);
            frame->repr  = repr;
            frame->color = color;
            if (input->chroma_location)
                pl_frame_set_chroma_location(frame, input->chroma_location);
            // frame->rotation and frame->crop are updated by render_input()
        }
    }

    if (pi->status == PL_QUEUE_EOF || pi->status == PL_QUEUE_ERR) {
        pl_renderer_reset(input->renderer);
        pl_queue_reset(input->queue);
        pl_queue_push(input->queue, NULL); // discard any further frames
        pi->stop_pts = pts;
    }

done:;
    bool active = pi->status == PL_QUEUE_OK || pi->status == PL_QUEUE_MORE;
    active &= pi->mix.num_frames > 0;

    if (out_status) {
        *out_status = (struct pl_input_status) {
            .status    = pi->status,
            .mix       = active ? &pi->mix : NULL,
            .pts       = active ? pts - pi->start_pts + input->seek_pts : NAN,
            .pts_drift = active ? pl_queue_pts_offset(input->queue) : 0.0,
            .start_pts = pi->start_pts,
            .stop_pts  = pi->stop_pts,
        };
    }

    return pi->status;
}

static bool alloc_inter_fbo(struct priv *p, int width, int height)
{
    if (!p->fmt)
        goto error;

    bool ok = pl_tex_recreate(p->gpu, &p->tex, pl_tex_params(
        .format     = p->fmt,
        .w          = width,
        .h          = height,
        .blit_dst   = true,
        .renderable = true,
        .sampleable = true,
        .storable   = p->fmt->caps & PL_FMT_CAP_STORABLE,
    ));
    if (!ok)
        goto error;

    return true;

error:
    if (!(p->errors & PL_RENDER_ERR_FBO)) {
        PL_WARN(p, "Failed to create a linear texture for compositing "
                "inputs, falling back to non-linear blending.");
        p->errors |= PL_RENDER_ERR_FBO;
    }
    return false;
}

// Resolve unspecified crop values to the full frame size
static pl_rect2df frame_crop(const struct pl_frame *frame)
{
    pl_rect2df crop = frame->crop;
    if (!frame->num_planes)
        return crop;

    int w, h;
    pl_frame_size(frame, &w, &h);
    if (!crop.x0 && !crop.x1)
        crop.x1 = w;
    if (!crop.y0 && !crop.y1)
        crop.y1 = h;
    return crop;
}

// Get the logical, SAR-adjusted, rotated size of a frame
static void logical_size(float *out_w, float *out_h, const float sar,
                         const struct pl_frame *frame)
{
    const pl_rect2df crop = frame_crop(frame);
    float w = fabsf(pl_rect_w(crop)), h = fabsf(pl_rect_h(crop));
    if (sar > 1.0f)
        w *= sar;
    else if (sar > 0.0f && sar < 1.0f)
        h /= sar;

    if (frame->rotation % PL_ROTATION_180 == PL_ROTATION_90)
        PL_SWAP(w, h);

    *out_w = w;
    *out_h = h;
}

static pl_rect2df mix_rect(pl_rect2df rc, pl_rect2df f)
{
    return (pl_rect2df) {
        .x0 = PL_MIX(rc.x0, rc.x1, f.x0),
        .y0 = PL_MIX(rc.y0, rc.y1, f.y0),
        .x1 = PL_MIX(rc.x0, rc.x1, f.x1),
        .y1 = PL_MIX(rc.y0, rc.y1, f.y1),
    };
}

static inline void mirror(float *a, float *b, float lo, float hi)
{
    *a = lo + hi - *a;
    *b = lo + hi - *b;
}

void pl_input_place(pl_input input,
                    const struct pl_frame *image,
                    const struct pl_frame *target,
                    struct pl_rect2df *src_rect,
                    struct pl_rect2df *dst_rect)
{
    enum pl_fit_mode mode = input->fit;
    float src_sar = image->pixel_aspect_ratio;
    float dst_sar = target->pixel_aspect_ratio;
    if (mode == PL_FIT_RAW)
        src_sar = dst_sar = 1.0f;

    float image_w, image_h, canvas_w, canvas_h;
    logical_size(&image_w,  &image_h,  src_sar, image);
    logical_size(&canvas_w, &canvas_h, dst_sar, target);

    if (input->rotation % PL_ROTATION_180 == PL_ROTATION_90)
        PL_SWAP(image_w, image_h);

    // Perform placement as though both rects were unflipped, i.e. in their
    // mirrored frames of reference, and mirror the result back at the end
    pl_rect2df src_crop = input->src_rect, dst_crop = input->dst_rect;
    const bool flip_sx = src_crop.x0 > src_crop.x1;
    const bool flip_sy = src_crop.y0 > src_crop.y1;
    const bool flip_dx = dst_crop.x0 > dst_crop.x1;
    const bool flip_dy = dst_crop.y0 > dst_crop.y1;
    pl_rect2df_normalize(&src_crop);
    pl_rect2df_normalize(&dst_crop);

    if (mode == PL_FIT_RAW) {
        // Snap both crops to whole pixels, so the image can be placed at its
        // exact size, mapping pixels onto pixels
        src_crop = (pl_rect2df) {
            .x0 = roundf(src_crop.x0 * image_w) / image_w,
            .y0 = roundf(src_crop.y0 * image_h) / image_h,
            .x1 = roundf(src_crop.x1 * image_w) / image_w,
            .y1 = roundf(src_crop.y1 * image_h) / image_h,
        };

        dst_crop = (pl_rect2df) {
            .x0 = roundf(dst_crop.x0 * canvas_w) / canvas_w,
            .y0 = roundf(dst_crop.y0 * canvas_h) / canvas_h,
            .x1 = roundf(dst_crop.x1 * canvas_w) / canvas_w,
            .y1 = roundf(dst_crop.y1 * canvas_h) / canvas_h,
        };
    }

    const float src_w = image_w  * pl_rect_w(src_crop);
    const float src_h = image_h  * pl_rect_h(src_crop);
    const float dst_w = canvas_w * pl_rect_w(dst_crop);
    const float dst_h = canvas_h * pl_rect_h(dst_crop);

    if (mode == PL_FIT_SCALE_DOWN) {
        if (src_w > dst_w || src_h > dst_h)
            mode = PL_FIT_CONTAIN;
        else
            mode = PL_FIT_PLACE;
    }

    const float src_dar = src_w / src_h;
    const float dst_dar = dst_w / dst_h;
    *src_rect = (struct pl_rect2df) { 0, 0, 1, 1 };
    *dst_rect = (struct pl_rect2df) { 0, 0, 1, 1 };

    switch (mode) {
    case PL_FIT_FILL:
        break;
    case PL_FIT_PLACE:
    case PL_FIT_RAW:
        pl_rect2df_stretch(dst_rect, src_w / dst_w, src_h / dst_h);
        break;
    case PL_FIT_CONTAIN:
        pl_rect2df_aspect_set(dst_rect, src_dar / dst_dar, 0.0f);
        break;
    case PL_FIT_COVER:
        pl_rect2df_aspect_set(dst_rect, src_dar / dst_dar, 1.0f);
        break;
    case PL_FIT_SCALE_DOWN:
    case PL_FIT_MODE_COUNT:
        pl_unreachable();
    }

    // Virtual placed width/height on the canvas
    const float vw = pl_rect_w(*dst_rect), vh = pl_rect_h(*dst_rect);

    // Align the smaller of the two rects according to the focal point.
    // Unscaled images are instead always centered on the focus point of the
    // canvas, even if this places them (partially) outside of it.
    const bool unscaled = input->fit == PL_FIT_PLACE || input->fit == PL_FIT_RAW;
    const pl_vec2df focus = input->focus;
    if (vw > 1.0f && !unscaled) {
        dst_rect->x0 = PL_CLAMP(0.5f - focus.x * vw, 1.0f - vw, 0.0f);
    } else {
        dst_rect->x0 = focus.x - vw / 2;
        if (!unscaled)
            dst_rect->x0 = PL_CLAMP(dst_rect->x0, 0.0f, 1.0f - vw);
    }
    if (vh > 1.0f && !unscaled) {
        dst_rect->y0 = PL_CLAMP(0.5f - focus.y * vh, 1.0f - vh, 0.0f);
    } else {
        dst_rect->y0 = focus.y - vh / 2;
        if (!unscaled)
            dst_rect->y0 = PL_CLAMP(dst_rect->y0, 0.0f, 1.0f - vh);
    }
    dst_rect->x1 = dst_rect->x0 + vw;
    dst_rect->y1 = dst_rect->y0 + vh;

    // Rebase relative to the specified crop
    *src_rect = mix_rect(src_crop, *src_rect);
    *dst_rect = mix_rect(dst_crop, *dst_rect);

    if (mode == PL_FIT_RAW) {
        // The size is already exact, so only the offset needs rounding
        const float dx = roundf(dst_rect->x0 * canvas_w) / canvas_w - dst_rect->x0;
        const float dy = roundf(dst_rect->y0 * canvas_h) / canvas_h - dst_rect->y0;
        dst_rect->x0 += dx;
        dst_rect->x1 += dx;
        dst_rect->y0 += dy;
        dst_rect->y1 += dy;
    }

    // Clip the final result to `dst_rect`
    const float sx = pl_rect_w(*src_rect) / pl_rect_w(*dst_rect);
    const float sy = pl_rect_h(*src_rect) / pl_rect_h(*dst_rect);
    if (dst_rect->x0 < dst_crop.x0) {
        src_rect->x0 += (dst_crop.x0 - dst_rect->x0) * sx;
        dst_rect->x0 = dst_crop.x0;
    }
    if (dst_rect->x1 > dst_crop.x1) {
        src_rect->x1 -= (dst_rect->x1 - dst_crop.x1) * sx;
        dst_rect->x1 = dst_crop.x1;
    }
    if (dst_rect->y0 < dst_crop.y0) {
        src_rect->y0 += (dst_crop.y0 - dst_rect->y0) * sy;
        dst_rect->y0 = dst_crop.y0;
    }
    if (dst_rect->y1 > dst_crop.y1) {
        src_rect->y1 -= (dst_rect->y1 - dst_crop.y1) * sy;
        dst_rect->y1 = dst_crop.y1;
    }

    // Nothing remains of an image placed entirely outside of `dst_rect`
    if (dst_rect->x0 >= dst_rect->x1 || dst_rect->y0 >= dst_rect->y1) {
        *src_rect = (pl_rect2df) { src_crop.x0, src_crop.y0, src_crop.x0, src_crop.y0 };
        *dst_rect = (pl_rect2df) { dst_crop.x0, dst_crop.y0, dst_crop.x0, dst_crop.y0 };
    }

    // Re-apply any flips, by mirroring the result within the respective crop
    if (flip_sx)
        mirror(&src_rect->x0, &src_rect->x1, src_crop.x0, src_crop.x1);
    if (flip_sy)
        mirror(&src_rect->y0, &src_rect->y1, src_crop.y0, src_crop.y1);
    if (flip_dx)
        mirror(&dst_rect->x0, &dst_rect->x1, dst_crop.x0, dst_crop.x1);
    if (flip_dy)
        mirror(&dst_rect->y0, &dst_rect->y1, dst_crop.y0, dst_crop.y1);
}

// Apply a rect in the frame's displayed (rotated) frame of reference to its
// crop, which is in the frame's unrotated frame of reference
static void set_crop(struct pl_frame *frame, pl_rect2df rect)
{
    // Rotating clockwise moves the bottom left corner to the top left, so
    // map each step back via (x, y) -> (y, 1 - x)
    for (int i = 0; i < pl_rotation_normalize(frame->rotation); i++) {
        rect = (pl_rect2df) {
            .x0 = rect.y0,
            .y0 = 1.0f - rect.x1,
            .x1 = rect.y1,
            .y1 = 1.0f - rect.x0,
        };
    }

    frame->crop = mix_rect(frame_crop(frame), rect);
}

// Use first visible input as overall size reference
static const struct pl_frame *ref_frame(const struct pl_frame_mix *mix)
{
    for (int i = 0; i < mix->num_frames; i++) {
        if (i+1 == mix->num_frames || mix->timestamps[i+1] > 0)
            return mix->frames[i];
    }
    return NULL;
}

static bool render_input(pl_compositor comp, pl_input input,
                         const struct pl_frame *ptarget,
                         const struct pl_render_params *params)
{
    struct priv *p = PL_PRIV(comp);
    struct priv_input *pi = PL_PRIV(input);
    const struct pl_frame *ref = ref_frame(&pi->mix);
    pl_assert(ref);

    struct pl_rect2df src_rect, dst_rect;
    pl_input_place(input, ref, ptarget, &src_rect, &dst_rect);

    PL_TRACE(p, "Placed src (%.3f, %.3f) -> (%.3f, %.3f), dst (%.3f, %.3f) -> (%.3f, %.3f)",
             src_rect.x0, src_rect.y0, src_rect.x1, src_rect.y1,
             dst_rect.x0, dst_rect.y0, dst_rect.x1, dst_rect.y1);

    if (!pl_rect2d_eq(src_rect, pi->cached_src_rect) ||
        !pl_rect2d_eq(dst_rect, pi->cached_dst_rect))
    {
        PL_DEBUG(p, "Flushing render cache due to updated crop");
        pl_renderer_flush_cache(input->renderer);
        pi->cached_src_rect = src_rect;
        pi->cached_dst_rect = dst_rect;

    }

    struct pl_frame target = *ptarget;
    set_crop(&target, dst_rect);
    for (int i = 0; i < pi->frames.num; i++) {
        struct pl_frame *frame = &pi->frames.elem[i];
        frame->rotation += input->rotation;
        set_crop(frame, src_rect);
    }

    return pl_render_image_mix(input->renderer, &pi->mix, &target, params);
}

// Whether any part of the image is placed onto the canvas, if known
static bool on_canvas(pl_input input, const struct pl_frame *image,
                      const struct pl_frame *target)
{
    if (!target)
        return true;

    struct pl_rect2df src, dst;
    pl_input_place(input, image, target, &src, &dst);
    return pl_rect_w(dst) && pl_rect_h(dst);
}

static bool covers_canvas(pl_input input, const struct pl_frame *image,
                          const struct pl_frame *target)
{
    struct pl_rect2df src, dst;
    if (!target)
        goto fallback;

    pl_input_place(input, image, target, &src, &dst);
    pl_rect2df_normalize(&dst);

    const pl_rect2df crop = frame_crop(target);
    const float cw = fabsf(pl_rect_w(crop));
    const float ch = fabsf(pl_rect_h(crop));
    return roundf(dst.x0 * cw) <= 0.0f && roundf(dst.y0 * ch) <= 0.0f &&
           roundf(dst.x1 * cw) >= cw   && roundf(dst.y1 * ch) >= ch;

fallback:
    switch (input->fit) {
    case PL_FIT_FILL:
    case PL_FIT_COVER:
        dst = input->dst_rect;
        pl_rect2df_normalize(&dst);
        return dst.x0 <= 0.0f && dst.y0 <= 0.0f &&
               dst.x1 >= 1.0f && dst.y1 >= 1.0f;
    default:
        return false;
    }
}

static bool is_opaque(pl_input input, const struct pl_frame *image)
{
    const struct pl_render_params *par = input_params(input);
    const struct pl_distort_params *distort = par->distort_params;

    return image->repr.alpha == PL_ALPHA_NONE && !par->corner_rounding &&
           par->transparency <= 0.0f &&
           (!distort || !memcmp(distort, &pl_distort_default_params, sizeof(*distort)));
}

static int cmp_z_index(const void *pa, const void *pb)
{
    const pl_input a = *(const pl_input *) pa;
    const pl_input b = *(const pl_input *) pb;
    if (a->z_index != b->z_index)
        return PL_CMP(a->z_index, b->z_index);

    /* Break ties by attachment order; later entries obscuring earlier */
    const struct priv_input *ia = PL_PRIV(a), *ib = PL_PRIV(b);
    return PL_CMP(ia->index, ib->index);
}

// Updates all inputs, and the list of visible inputs (p->visible)
static void update_inputs(pl_compositor comp, struct pl_compositor_status *status,
                          const struct pl_frame *target,
                          const struct pl_queue_params *params)
{
    struct priv *p = PL_PRIV(comp);
    pl_assert(p->sorted.num == comp->num_inputs);
    qsort(p->sorted.elem, p->sorted.num, sizeof(pl_input), cmp_z_index);

    PL_ARRAY_RESIZE(comp, p->input_status, comp->num_inputs);
    p->input_status.num = comp->num_inputs;
    p->visible.num = 0;

    bool eof = true, more = false, err = false;
    for (int i = 0; i < p->sorted.num; i++) {
        pl_input input = p->sorted.elem[i];
        struct priv_input *pi = PL_PRIV(input);
        struct pl_input_status *in_status = &p->input_status.elem[pi->index];
        switch (pl_input_update(input, in_status, params)) {
        case PL_QUEUE_EOF:
            continue;
        case PL_QUEUE_MORE:
            more = true;
            // fall through
        case PL_QUEUE_OK:
            eof = false;
            if (!pi->mix.num_frames)
                continue;
            break;
        case PL_QUEUE_ERR:
            PL_ERR(p, "Received error while updating input %d", pi->index);
            err = true;
            continue;
        }

        const struct pl_frame *cur = pl_frame_mix_nearest(&pi->mix);
        pl_assert(cur);
        if (!on_canvas(input, cur, target))
            continue;

        // Fully transparent inputs still need to be rendered if they are
        // the base layer, since they imply the shape of the bg/border
        const struct pl_render_params *par = input_params(input);
        if (p->visible.num && par->transparency >= 1.0f)
            continue;

        // If an opaque frame covers the entire output, disregard lower layers
        if (covers_canvas(input, cur, target) && is_opaque(input, cur)) {
            // Earlier inputs will not be rendered
            for (int j = 0; j < p->visible.num; j++)
                pl_renderer_flush_cache(p->visible.elem[j]->renderer);
            p->visible.num = 0;
        }

        PL_ARRAY_APPEND(comp, p->visible, input);
    }

    *status = (struct pl_compositor_status) {
        .status       = err  ? PL_QUEUE_ERR  :
                        more ? PL_QUEUE_MORE :
                        eof  ? PL_QUEUE_EOF  : PL_QUEUE_OK,
        .input_status = p->input_status.elem,
        .visible      = p->visible.elem,
        .num_visible  = p->visible.num,
    };

    if (p->visible.num) {
        const struct priv_input *pi = PL_PRIV(p->visible.elem[0]);
        status->base = pl_frame_mix_nearest(&pi->mix); // match renderer.c semantics
    }
}

bool pl_compositor_render(pl_compositor comp, const struct pl_frame *ptarget,
                          const struct pl_queue_params *params,
                          struct pl_compositor_status *out_status)
{
    struct priv *p = PL_PRIV(comp);
    if (!p->gpu) {
        PL_ERR(p, "Trying to call pl_compositor_render() on an uninitialized compositor!");
        return false;
    }

    struct pl_frame target = *ptarget;
    target.repr  = comp->repr;
    target.color = comp->color;
    pl_color_repr_merge(&target.repr, &ptarget->repr);
    pl_color_space_merge(&target.color, &ptarget->color);
    if (comp->chroma_location)
        pl_frame_set_chroma_location(&target, comp->chroma_location);

    struct pl_compositor_status status;
    update_inputs(comp, &status, &target, params);
    if (out_status)
        *out_status = status;
    if (status.status == PL_QUEUE_ERR)
        return false;

    struct pl_render_params cpar;
    get_canvas_params(&cpar, comp);
    if (!status.num_visible) {
        set_info_callback(&cpar, comp, NULL);
        return pl_render_image(p->rr, NULL, &target, &cpar);
    }

    // Compute normalized rotated output dimensions
    int target_width, target_height;
    pl_frame_size(&target, &target_width, &target_height);

    // The linear light intermediate can only be skipped for opaque single inputs
    const bool trivial = status.num_visible == 1 &&
                         is_opaque(status.visible[0], status.base);

    const struct pl_render_params *canvas = canvas_params(comp);
    const bool skip_border = get_border(canvas) == PL_CLEAR_SKIP;
    bool use_inter = !trivial && !canvas->disable_fbos &&
                     !comp->disable_linear_composition &&
                     alloc_inter_fbo(p, target_width, target_height);

    struct pl_frame orig_target = target;
    if (use_inter) {
        // Ensure the border is well-defined
        if (skip_border)
            pl_tex_clear(p->gpu, p->tex, (float[4]) {0});

        target = (struct pl_frame) {
            .num_planes = 1,
            .planes[0] = {
                .components = 4,
                .component_mapping = {0, 1, 2, 3},
                .texture = p->tex,
            },
            .crop  = orig_target.crop,
            .color = orig_target.color,
            .repr  = pl_color_repr_rgb,
            .rotation = orig_target.rotation,
            .pixel_aspect_ratio = orig_target.pixel_aspect_ratio,
        };

        target.color.transfer = PL_COLOR_TRC_LINEAR; // composite in linear light
        target.repr.alpha     = PL_ALPHA_PREMULTIPLIED;
    }

    for (int i = 0; i < status.num_visible; i++) {
        pl_input input = status.visible[i];
        struct pl_render_params ipar = *input_params(input);
        set_canvas_params(&ipar, canvas, i == 0, !use_inter);
        set_info_callback(&ipar, comp, input);

        if (!render_input(comp, input, &target, &ipar)) {
            struct priv_input *pi = PL_PRIV(input);
            PL_ERR(p, "Failed rendering input %d", pi->index);
            return false;
        }

        // This is required to get correct blending onto YUV target buffers.
        target.repr.alpha = PL_ALPHA_INDEPENDENT;
    }

    if (use_inter) {
        // Blit the linear intermediate image to the output frame
        target.crop = orig_target.crop = (pl_rect2df) {0};
        target.repr.alpha = PL_ALPHA_PREMULTIPLIED;

        // The base layer already drew the background, in linear light
        cpar.background = PL_CLEAR_SKIP;
        cpar.blend_against_tiles = false;
        if (skip_border && !cpar.blend_params) {
            // Blend the canvas onto the target to preserve the border
            cpar.blend_params = &pl_alpha_overlay;
            orig_target.repr.alpha = PL_ALPHA_INDEPENDENT;
        }

        set_info_callback(&cpar, comp, NULL);
        return pl_render_image(p->rr, &target, &orig_target, &cpar);
    }

    return true;
}

struct pl_compositor_status pl_compositor_update(pl_compositor comp,
                                    const struct pl_queue_params *params,
                                    const struct pl_canvas_size *canvas)
{
    static const struct pl_canvas_size default_canvas = {0};
    canvas = PL_DEF(canvas, &default_canvas);

    const struct pl_frame target_virtual = {
        .crop.x1  = canvas->width,
        .crop.y1  = canvas->height,
        .rotation = canvas->rotation,
        .pixel_aspect_ratio = PL_DEF(canvas->pixel_aspect_ratio, 1.0f),
    };

    const bool has_size = canvas->width && canvas->height;
    struct pl_compositor_status status;
    update_inputs(comp, &status, has_size ? &target_virtual : NULL, params);
    return status;
}

const struct pl_frame *pl_compositor_current_frame(pl_compositor comp,
                                    const struct pl_queue_params *params,
                                    const struct pl_canvas_size *canvas)
{
    return pl_compositor_update(comp, params, canvas).base;
}

static enum pl_render_error get_errors_accum(pl_compositor comp, pl_renderer rr)
{
    if (!rr)
        return 0;

    struct priv *p = PL_PRIV(comp);
    struct pl_render_errors errors = pl_renderer_get_errors(rr);
    PL_ARRAY(uint64_t) disabled = {
        .elem = (uint64_t *) errors.disabled_hooks,
        .num = errors.num_disabled_hooks,
    };
    PL_ARRAY_CONCAT(comp, p->disabled_hooks, disabled);
    return errors.errors;
}

struct pl_render_errors pl_compositor_get_errors(pl_compositor comp)
{
    struct priv *p = PL_PRIV(comp);
    struct pl_render_errors ret = { .errors = p->errors };

    p->disabled_hooks.num = 0;
    ret.errors |= get_errors_accum(comp, p->rr);
    for (int i = 0; i < comp->num_inputs; i++)
        ret.errors |= get_errors_accum(comp, comp->inputs[i]->renderer);

    ret.disabled_hooks = p->disabled_hooks.elem;
    ret.num_disabled_hooks = p->disabled_hooks.num;
    return ret;
}

void pl_compositor_reset_errors(pl_compositor comp,
                                const struct pl_render_errors *errors)
{
    struct priv *p = PL_PRIV(comp);
    if (!p->rr) {
        pl_assert(!p->errors);
        return;
    }

    p->errors = errors ? (p->errors & ~errors->errors) : 0;

    pl_renderer_reset_errors(p->rr, errors);
    for (int i = 0; i < comp->num_inputs; i++)
        pl_renderer_reset_errors(comp->inputs[i]->renderer, errors);
}
