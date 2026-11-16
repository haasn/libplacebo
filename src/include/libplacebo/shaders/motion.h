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

#ifndef LIBPLACEBO_SHADERS_MOTION_H_
#define LIBPLACEBO_SHADERS_MOTION_H_

// Motion estimation between a pair of frames.

#include <libplacebo/colorspace.h>
#include <libplacebo/dispatch.h>
#include <libplacebo/shaders.h>

PL_API_BEGIN

// Which direction(s) of flow to compute.
enum pl_motion_direction {
    PL_MOTION_FORWARD   = 1 << 0, // A -> B
    PL_MOTION_BACKWARD  = 1 << 1, // B -> A
    PL_MOTION_BOTH = PL_MOTION_FORWARD | PL_MOTION_BACKWARD,
};

// Additional motion estimation outputs to compute, besides the
// flow vectors themselves.
enum pl_motion_feature {
    PL_MOTION_GLOBAL    = 1 << 0,   // pl_motion_get_global()
    PL_MOTION_QUALITY   = 1 << 1,   // pl_motion_get_quality()
};

// Tunable parameters. Kept separate from the per-call inputs so that callers
// can reuse unchanging configurations.
//
// Changing any of these between calls implicitly discards motion estimation
// state, as if `pl_motion_reset` had been called.
struct pl_motion_estimate_params {
    // Side length, in pixels, of the square blocks the search operates on.
    // Must be even. Smaller blocks track finer detail at a superlinear cost.
    int block_size_coarse;

    // Side length, in pixels, of the squares that the final flow fields may
    // be refined to. Must be in the range [1, block_size_coarse]. This is a
    // hint; the actual resolution of the flow field may be coarser than this,
    // depending on implementation details.
    int block_size_fine;

    // Number of levels in the coarse-to-fine pyramid, including the full
    // resolution level. 0 picks a value from the input resolution and
    // `block_size_coarse`.
    int pyramid_levels;

    // Search radius in pixels. `coarse` applies to only the coarsest level of
    // the image pyramid. `fine` applies to all other levels of the image pyramid.
    // It is intended that `coarse` > `fine`, since that would allow the coarsest
    // level to detect large motion. If `pyramid_levels` == 1, we use `search_radius_fine` on the only image pyramid level.
    int search_radius_coarse;
    int search_radius_fine;
};

#define PL_MOTION_ESTIMATE_DEFAULTS                                            \
    .block_size_coarse      = 8,                                               \
    .block_size_fine        = 1,                                               \
    .pyramid_levels         = 0,                                               \
    .search_radius_coarse   = 4,                                               \
    .search_radius_fine     = 1,

#define pl_motion_estimate_params(...)                                         \
    (&(struct pl_motion_estimate_params) {                                     \
        PL_MOTION_ESTIMATE_DEFAULTS __VA_ARGS__})

PL_API extern const struct pl_motion_estimate_params
    pl_motion_estimate_default_params;

// Returns the number of blocks `pl_motion_estimate` searches at full
// resolution, for frames of dimensions `w` x `h`. Purely informative.
PL_API int pl_motion_num_blocks(int w, int h,
                                const struct pl_motion_estimate_params *params);

// One frame of the pair to estimate motion between.
struct pl_motion_estimate_frame {
    pl_tex tex;   // must be sampleable
    uint64_t sig; // must uniquely identify this frame
    struct pl_color_repr repr;
    struct pl_color_space color;
};

// Per-call inputs to pl_motion_estimate()
struct pl_motion_estimate_input {
    // Must be the same GPU `dispatch` was created with.
    pl_gpu gpu;

    // Used to allocate and execute every internal pass.
    pl_dispatch dispatch;

    // The frame pair to estimate motion between.
    struct pl_motion_estimate_frame a; // earlier frame
    struct pl_motion_estimate_frame b; // later frame

    // Direction(s) of flow to compute. Must be non-zero.
    enum pl_motion_direction direction;

    // Optional extra set of features to compute.
    enum pl_motion_feature features;

    // Persistent state. This is owned by the caller and must be freed using
    // `pl_shader_obj_destroy`. May point at NULL on first use.
    pl_shader_obj *state;
};

#define pl_motion_estimate_input(...)                                          \
    (&(struct pl_motion_estimate_input) {__VA_ARGS__})

// Estimate motion between `input->a` and `input->b`.
//
// This invalidates any prior flow field contained in `input->state`. However,
// if `input->a` matches the prior call's `input->b`, information is re-used
// from the previous output to improve the quality of the flow vectors.
//
// Returns false on failure.
PL_API bool pl_motion_estimate(const struct pl_motion_estimate_input *input,
                               const struct pl_motion_estimate_params *params);

// Discard all temporal state. After this call, the `uint64_t sig` associated
// with each subsequent frame may alias those of frames before the reset.
PL_API void pl_motion_reset(pl_shader_obj state);

// Returns the flow field for a single direction, or NULL. The returned
// texture is owned by the state object and must not be modified by the caller,
// nor used after another `pl_motion_estimate` or `pl_motion_reset` call.
//
// This returns a float-valued texture such that `vec2(mv.x, mv.y)` is the
// displacement in pixels from the earlier frame to the later frame. (Or
// from the later frame to the earlier frame, for PL_MOTION_BACKWARD)
//
// In other words, if the field holds `d` at position `p`, then the content
// at source frame location `p` corresponds to `p + d` in the target frame.
//
// This texture may have a size that differs from the inputs themselves.
// However, the granularity is guaranteed to lie somewhere in the range
// [block_size_fine, block_size_coarse]. `tex->params` provides the exact
// dimensions and format.
PL_API pl_tex pl_motion_get_flow(pl_shader_obj state,
                                 enum pl_motion_direction dir);

// Returns the quality of the flow field of a single direction, or NULL. The
// returned texture is owned by the state object.
//
// This is a single component float texture in the range [0, 1], with a value
// of 1.0 corresponding to high quality and 0.0 meaning low quality vectors.
//
// See pl_motion_get_flow(), PL_MOTION_QUALITY.
PL_API pl_tex pl_motion_get_quality(pl_shader_obj state,
                                    enum pl_motion_direction dir);

// Returns the buffer holding the global motion vector, or NULL. The returned
// buffer is owned by the state object.
//
// The corresponding (PL_DESC_BUF_STORAGE) layout is:
//
//     layout(std430) buffer {
//         vec2  global_mv;         // offset 0
//         float global_confidence; // offset 8
//     };
//
// See pl_motion_get_flow(), PL_MOTION_GLOBAL.
PL_API pl_buf pl_motion_get_global_buf(pl_shader_obj state,
                                       enum pl_motion_direction dir);

// Helper to read and access the last global motion buffer from the GPU.
// Returns false if no global motion is available.
//
// Note: This will block until the result is available, and should therefore
// be avoided except as a debugging aid.
PL_API bool pl_motion_get_global(pl_shader_obj state,
                                 enum pl_motion_direction dir,
                                 float *out_x, float *out_y,
                                 float *out_confidence);

// Algorithm used for frame interpolation when enabled.
enum pl_frame_interp_algo {
    // A modified frame mixing algorithm that offsets a sample by global
    // motion.
    PL_FRAME_INTERP_GLOBAL_MOTION = 0,

    // Debugging aid: instead of interpolating, draws the forward motion
    // vectors between the two frames with the Middlebury color wheel, on top
    // of the earlier frame. Vectors of low quality fade to gray.
    PL_FRAME_INTERP_VISUALIZE,

    // Debugging aid: instead of interpolating, draws the quality of the
    // forward motion vectors (see `pl_motion_get_quality`), from black for
    // the lowest to white for the highest.
    PL_FRAME_INTERP_QUALITY,

    PL_FRAME_INTERP_ALGO_COUNT,
};

// Tunable parameters. Kept separate from the per-call inputs so that callers
// can reuse unchanging configurations.
struct pl_frame_interp_params {
    // Frame interpolation algorithm to use.
    enum pl_frame_interp_algo algo;

    // Motion estimation parameters.
    struct pl_motion_estimate_params motion;
};

#define PL_FRAME_INTERP_DEFAULTS                                               \
    .algo = PL_FRAME_INTERP_GLOBAL_MOTION,                                     \
    .motion = {PL_MOTION_ESTIMATE_DEFAULTS},

#define pl_frame_interp_params(...)                                            \
    (&(struct pl_frame_interp_params) {PL_FRAME_INTERP_DEFAULTS __VA_ARGS__})

PL_API extern const struct pl_frame_interp_params
    pl_frame_interp_default_params;

// Per-call inputs.
struct pl_frame_interp_input {
    // Must be the same GPU `dispatch` was created with.
    pl_gpu gpu;

    // Used to allocate and execute motion estimation passes.
    pl_dispatch dispatch;

    // Motion estimation state, as for `pl_motion_estimate_input.state`. After
    // a successful call, the motion estimation results can be retrieved from
    // this state.
    pl_shader_obj *state;

    // The pair of frames to synthesize an intermediate frame between. These
    // are directly sampled when generating the intermediate frame.
    struct pl_motion_estimate_frame a; // earlier frame
    struct pl_motion_estimate_frame b; // later frame

    // Position of the synthesized frame between `a` and `b`, in [0, 1]. 0.0
    // corresponds to `a`, 1.0 corresponds to `b`.
    float t;

    // Color space to emit the synthesized frame in. Both frames are converted
    // to this color space before being combined, so this should generally be
    // a linear light space.
    struct pl_color_space out_color;
};

#define pl_frame_interp_input(...)                                             \
    (&(struct pl_frame_interp_input) {__VA_ARGS__})

// Synthesize the frame at `input->t` between `input->a` and `input->b`,
// estimating motion between the two frames as needed. Repeated calls for the
// same pair of frames reuse the motion estimate.
//
// Appends a shader writing `vec4 color` in `input->out_color`, with
// premultiplied alpha. The output size matches that of the input textures.
//
// Returns false on failure. Unless `sh` itself is unsuitable for appending a
// sampling operation, it is left untouched in this case.
PL_API bool pl_shader_frame_interp(pl_shader sh,
                                   const struct pl_frame_interp_input *input,
                                   const struct pl_frame_interp_params *params);

PL_API_END

#endif // LIBPLACEBO_SHADERS_MOTION_H_
