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

#ifndef LIBPLACEBO_COMPOSITOR_H_
#define LIBPLACEBO_COMPOSITOR_H_

#include <libplacebo/config.h>
#include <libplacebo/options.h>
#include <libplacebo/renderer.h>
#include <libplacebo/utils/frame_queue.h>

PL_API_BEGIN

enum pl_fit_mode {
    PL_FIT_CONTAIN = 0, // Largest size contained within the canvas, preserving aspect
    PL_FIT_COVER,       // Smallest size that covers the canvas, preserving aspect
    PL_FIT_FILL,        // Image is stretched to the exact output size, ignoring aspect
    PL_FIT_SCALE_DOWN,  // Image scaled only as needed to fit inside the canvas, preserving aspect
    PL_FIT_PLACE,       // No fit mode, source is unscaled (as-is)
    PL_FIT_RAW,         // Like PL_FIT_PLACE, but rounding to exact pixels
    PL_FIT_MODE_COUNT,
};

// Note: PL_PTS_ROUND_DOWN and PL_PTS_ROUND_NEAREST require knowing the output
// frame rate. If `pl_queue_params.vsync_duration` is not set, they are
// ignored and treated as PL_PTS_ROUND_UP instead.
enum pl_pts_round_mode {
    PL_PTS_ROUND_NEAREST = 0, // Round to the nearest output frame (ties round up)
    PL_PTS_ROUND_UP,          // Round up to the nearest output frame
    PL_PTS_ROUND_DOWN,        // Round down to the nearest output frame
    PL_PTS_ROUND_COUNT,
};

// Thread-safety: Unsafe
typedef struct pl_input_t {
    // Output timestamps at which the input becomes visible and invisible again,
    // in the same timebase as `pl_queue_params.pts`.
    double start_pts; // inclusive bound, defaults to 0.0
    double stop_pts;  // exclusive bound, defaults to DBL_MAX

    // PTS rounding mode for the above, applied when they do not exactly align
    // with a requested output frame. Defaults to PL_PTS_ROUND_NEAREST.
    enum pl_pts_round_mode start_mode;
    enum pl_pts_round_mode stop_mode;

    // Position within the input's own timeline (i.e. the same timebase as
    // `pl_source_frame.pts`) to display at `start_pts`. Analogous to seeking
    // the input before playing it.
    double seek_pts;

    // Color space / representation overrides. Any values set here override
    // the values returned by the `get_frame` callback / `pl_queue_push()`,
    // respectively.
    //
    // Note: the lifetime of `repr.dovi` must remain valid for as long as this
    // input is attached.
    struct pl_color_repr repr;
    struct pl_color_space color;

    // Chroma location override. If set, replaces the chroma location of the
    // input's frames (i.e. the `shift_x/y` of their subsampled planes).
    enum pl_chroma_location chroma_location;

    // Compositing order. Inputs with a higher `z_index` are drawn on top of,
    // and may obscure, inputs with a lower one. Ties are broken by the order
    // in which the inputs were attached, with later inputs drawn on top.
    int z_index;

    // Extra end-to-end rotation applied to the input. This is applied *after*
    // any frame-level rotation metadata, but before determining the image
    // placement.
    pl_rotation rotation;

    // Controls how the (logical, aspect-corrected, rotated) image is placed
    // within the output frame. The fit is applied after `src_rect` and
    // `dst_rect` constrain the image/canvas.
    enum pl_fit_mode fit;

    // Source and destination rects, relative to the frame's own frame-level
    // crop, represented in the range [0, 1]. Both are expressed in the
    // displayed (rotated) frame of reference - that is, after both the
    // frame-level rotation and `rotation` for the source, and after the
    // canvas rotation for the destination. Defaults to {0, 0, 1, 1}.
    pl_rect2df src_rect;
    pl_rect2df dst_rect;

    // Focus point, deciding the alignment wherever the placed image is larger
    // or smaller than the canvas: the smaller of the two is centered on the
    // focus point of the larger, as far as possible without exceeding it.
    //
    // For PL_FIT_PLACE and PL_FIT_RAW, the image is instead always centered
    // on the focus point of the canvas, without such a limit. A focus point
    // outside of [0, 1] can thus move the image partially or entirely off the
    // canvas, without affecting its size.
    //
    // Defaults to {0.5, 0.5}, i.e. center alignment.
    pl_vec2df focus;

    // Frame retrieval callback for this input. If provided, this will be
    // called by the compositor to request additional frames. `params` is
    // a mirror of the parameters passed to `pl_input_update()`, but in the
    // input's own timebase.
    //
    // The user can alternatively push frames using pl_queue_push() on `queue`,
    // whenever pl_input_update() returns PL_QUEUE_MORE.
    enum pl_queue_status (*get_frame)(struct pl_input_t *input,
                                      struct pl_source_frame *out_frame,
                                      const struct pl_queue_params *params);

    // Queue of frames for this input. Should be fed with frames by the user
    // when not using `get_frame`. In this case, `pl_input_update` can
    // be used to query which inputs need additional frames.
    //
    // NULL before a compositor is initialized.
    pl_queue queue;

    // Renderer used to render frames for this input. NULL before a compositor
    // is initialized.
    pl_renderer renderer;

    // Render parameters used for rendering this input. Note that some
    // options are ignored; see `pl_compositor.params` for more details.
    //
    // Optional; defaults to `&pl_render_default_params` if unset.
    const struct pl_render_params *params;

    // Extra user data. Ignored by libplacebo.
    void *user_data;
} *pl_input;

// Thread-safety: Unsafe
typedef struct pl_compositor_t {
    // Inputs, in the order they were attached. Read-only; may be re-allocated
    // by `pl_compositor_input_create()`.
    pl_input *inputs;
    int num_inputs;

    // Parameters to use for the final composited output path. This affects
    // conversion to the target colorspace and format, as well as subsampling,
    // dithering, blending and so on, as well as the background/border clearing.
    //
    // Optional; defaults to `&pl_render_default_params` if unset.
    const struct pl_render_params *params;

    // Color space / representation overrides for the output. Any values set
    // here override the values of the target frame passed to
    // `pl_compositor_render()`.
    struct pl_color_repr repr;
    struct pl_color_space color;

    // Chroma location override for the output, see `pl_input.chroma_location`.
    enum pl_chroma_location chroma_location;

    // Disables linear light composition, rendering all inputs directly onto
    // the target frame instead. Intended for debugging purposes only.
    //
    // Note: Linear light composition is only used when necessary.
    bool disable_linear_composition;

    // Optional callback invoked for every shader pass. `input` is set to the
    // active input, or NULL for the final composition output pass.
    void (*info_callback)(void *priv, pl_input input,
                          const struct pl_render_info *info);
    void *info_priv;
} *pl_compositor;

// Create a new compositor with no inputs.
PL_API pl_compositor pl_compositor_create(pl_log log);

// Initialize a compositor. After this function returns, pl_input.queue and
// pl_input.renderer will be available for all inputs, including ones added
// after pl_compositor_init() was called.
//
// Returns an error on failure.
PL_API bool pl_compositor_init(pl_compositor comp, pl_gpu gpu);

// Destroy a compositor and all of its inputs.
PL_API void pl_compositor_destroy(pl_compositor *comp);

// Create a new input and initialize it to an empty state. The input is
// appended to `pl_compositor.inputs`.
PL_API pl_input pl_compositor_input_create(pl_compositor comp);

// Detach and destroy an input, removing it from `pl_compositor.inputs`.
PL_API void pl_compositor_input_destroy(pl_compositor comp, pl_input *input);

// Helper to set `pl_compositor.params`, and the `params` of all currently
// attached inputs to `opts->params`. `opts` must remain valid for as long
// as it is in use.
//
// Note: This should be called again after attaching new inputs.
PL_API void pl_compositor_options_set(pl_compositor comp, pl_options opts);

// Extra metadata returned by `pl_input_update()`.
struct pl_input_status {
    // The status of the input's queue, as returned by `pl_input_update()`.
    enum pl_queue_status status;

    // The frame mix for this `params->pts`, or NULL if the input is not
    // currently visible - either because it has not yet started, has already
    // stopped, or has no frames available for this timestamp. When non-NULL,
    // it is guaranteed to contain at least one frame.
    //
    // The lifetime of this pointer is only guaranteed until the next call to
    // `pl_input_update()` or `pl_compositor_render()`.
    //
    // Note: The mix may include input frames from outside [start_pts, stop_pts),
    // as required by the configured frame mixer's radius.
    const struct pl_frame_mix *mix;

    // Effective position within the input's own timeline, or NAN if `mix` is
    // NULL. Not subject to any drift compensation.
    double pts;

    // PTS drift/offset, as reported by pl_queue_pts_offset(), or 0.0 if
    // `mix` is NULL.
    double pts_drift;

    // Resolved scheduling values, after rounding, or NAN while still unknown:
    double start_pts; // always valid if `mix` is non-NULL
    double stop_pts;  // always valid once `mix` has become NULL again
};

// Attempt updating the input's frame queue for the target timestamp. Calling
// this is optional if using `pl_input.get_frame`, but can still be used to
// signal EOF / error conditions, or query the queue state at a given point
// in time using the (optional) `out_state` parameter.
//
// Before the input starts, its queue is still advanced along its own timeline,
// as if it had already started. This discards frames from before `seek_pts`,
// and returns PL_QUEUE_MORE as soon as more frames are needed to get there.
// Once the input stops, its queue is reset and any further frames are discarded.
//
// Note: `params->get_frame` and `params->radius` are ignored, in favor of
// `pl_input.get_frame` and the radius of the input's frame mixer.
PL_API enum pl_queue_status pl_input_update(pl_input input,
                                    struct pl_input_status *out_status,
                                    const struct pl_queue_params *params);

// Extra metadata returned by `pl_compositor_render()`.
struct pl_compositor_status {
    // Combined status of all inputs: PL_QUEUE_ERR if any input failed,
    // otherwise PL_QUEUE_MORE if any input needs more frames, PL_QUEUE_EOF if
    // all inputs (if any) have reached EOF, and PL_QUEUE_OK otherwise.
    enum pl_queue_status status;

    // The status of every input, as returned by `pl_input_update()`, in the
    // same order as `pl_compositor.inputs`.
    const struct pl_input_status *input_status;

    // The inputs that were rendered, in compositing order (bottom to top).
    // This excludes inputs that are not currently visible, as well as inputs
    // fully obscured by an opaque input covering the entire canvas.
    const pl_input *visible;
    int num_visible;

    // The current reference frame of the lowest visible input (`visible[0]`),
    // which the border is blurred from, or NULL if there are no visible inputs.
    // Equivalent to the frame returned by `pl_compositor_current_frame()`.
    const struct pl_frame *base;

    // Note: The lifetime of the above pointers is only guaranteed until the
    // next call to any `pl_compositor_*` or `pl_input_*` function.
};

// Note that this still succeeds if any inputs are not ready, but the result
// will simply not include those inputs; or will contain partially blended
// results. Calling this function requires that the compositor was initialized
// (see pl_compositor_init).
//
// It's recommended to first call `pl_input_update` on all inputs to ensure
// that all inputs are fully satisfied before attempting to render, unless
// using `pl_input.get_frame`. Alternatively, `out_status` (optional) can be
// used to check whether this was the case.
PL_API bool pl_compositor_render(pl_compositor comp, const struct pl_frame *target,
                                 const struct pl_queue_params *params,
                                 struct pl_compositor_status *out_status);

// Input placement logic. Computes the adjusted src and dst rects for a
// given input/output, as crop-relative rectangles in the same (rotated)
// frames of reference as `pl_input.src_rect` and `pl_input.dst_rect`. Both
// rects are empty if the image is placed entirely outside of `dst_rect`.
//
// Note: Fields besides `crop`, `pixel_aspect_ratio` and `rotation` are ignored.
PL_API void pl_input_place(pl_input input,
                           const struct pl_frame *image,
                           const struct pl_frame *target,
                           struct pl_rect2df *out_src,
                           struct pl_rect2df *out_dst);

struct pl_canvas_size {
    // see `pl_frame` for the meaning of these fields
    float width;
    float height;
    float pixel_aspect_ratio;
    pl_rotation rotation;
};

#define pl_canvas_size(...) (&(struct pl_canvas_size) { __VA_ARGS__ })

// Update all inputs for the current instance in time, exactly like
// `pl_compositor_render()` would, but without rendering anything. This can
// be used to e.g. provide more frames to inputs that need them, before
// rendering with the same `params`.
//
// Note: If the canvas size is not specified, the result may include inputs
// that would normally be obscured by a higher layer, or placed off-canvas.
PL_API struct pl_compositor_status pl_compositor_update(pl_compositor comp,
                                    const struct pl_queue_params *params,
                                    const struct pl_canvas_size *canvas);

// Returns the "current" reference frame (visible base layer) for the current
// instance in time, as determined by `pl_compositor_update()`.
PL_API const struct pl_frame *pl_compositor_current_frame(pl_compositor comp,
                                    const struct pl_queue_params *params,
                                    const struct pl_canvas_size *canvas);

// See `pl_renderer_get_errors` and `pl_renderer_reset_errors`.
PL_API struct pl_render_errors pl_compositor_get_errors(pl_compositor comp);
PL_API void pl_compositor_reset_errors(pl_compositor comp,
                                       const struct pl_render_errors *errors);

// A scene is a high-level wrapper around pl_compositor, that can be
// automatically initialized from a JSON description.
typedef const struct pl_scene_t {
    // Note: The scene owns `pl_compositor.params`, as well as the `params` of
    // every input. Inputs may not be added to or removed from the compositor.
    pl_compositor comp;
} *pl_scene;

// Parse a scene graph from a provided JSON object. Returns NULL on errors.
//
// This does not require a GPU. The resulting compositor is uninitialized, and
// must be initialized with `pl_compositor_init` before rendering.
//
// See `docs/json.md` for more information.
PL_API pl_scene pl_scene_create_json(pl_log log, const char *json, size_t len);

// Destroys a scene, including its compositor and all of its inputs.
PL_API void pl_scene_destroy(pl_scene *scene);

// Update the compositor and input parameters of `scene` to reflect the
// current instance in time (output-referred). Has no effect on scenes without
// animations.
PL_API void pl_scene_update(pl_scene scene, double pts);

// Helper function to both update a scene and render it.
static inline bool pl_scene_render(pl_scene scene, const struct pl_frame *target,
                                   const struct pl_queue_params *params,
                                   struct pl_compositor_status *out_status)
{
    pl_scene_update(scene, params->pts);
    return pl_compositor_render(scene->comp, target, params, out_status);
}

PL_API_END

#endif // LIBPLACEBO_COMPOSITOR_H_
