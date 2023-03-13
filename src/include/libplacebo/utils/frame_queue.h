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

#ifndef LIBPLACEBO_FRAME_QUEUE_H
#define LIBPLACEBO_FRAME_QUEUE_H

#include <libplacebo/renderer.h>
#include <libplacebo/shaders/deinterlacing.h>

PL_API_BEGIN

// An abstraction layer for automatically turning a conceptual stream of
// (frame, pts) pairs, as emitted by a decoder or filter graph, into a
// `pl_frame_mix` suitable for `pl_render_image_mix`.
//
// This API ensures that minimal work is performed (e.g. only mapping frames
// that are actually required), while also satisfying the requirements
// of any configured frame mixer.
//
// Thread-safety: Safe
typedef struct pl_queue_t *pl_queue;

enum pl_queue_status {
    PL_QUEUE_OK,       // success
    PL_QUEUE_EOF,      // no more frames are available
    PL_QUEUE_MORE,     // more frames needed, but not (yet) available
    PL_QUEUE_ERR = -1, // some unknown error occurred while retrieving frames
};

struct pl_source_frame {
    // The frame's presentation timestamp, in seconds relative to the first
    // frame. These must be monotonically increasing for subsequent frames.
    // To implement a discontinuous jump, users must explicitly reset the
    // frame queue with `pl_queue_reset` and restart from PTS 0.0.
    float pts;

    // The frame's duration. This is not needed in normal scenarios, as the
    // FPS can be inferred from the `pts` values themselves. Providing it
    // only helps initialize the value for initial frames, which can smooth
    // out the interpolation weights. Its use is also highly recommended
    // when displaying interlaced frames. (Optional)
    float duration;

    // If set to something other than PL_FIELD_NONE, this source frame is
    // marked as interlaced. It will be split up into two separate frames
    // internally, and exported to the resulting `pl_frame_mix` as a pair of
    // fields, referencing the corresponding previous and next frames. The
    // first field will have the same PTS as `pts`, and the second field will
    // be inserted at the timestamp `pts + duration/2`.
    //
    // Note: As a result of FPS estimates being unreliable around streams with
    // mixed FPS (or when mixing interlaced and progressive frames), it's
    // highly recommended to always specify a valid `duration` for interlaced
    // frames.
    enum pl_field first_field;

    // Abstract frame data itself. To allow mapping frames only when they're
    // actually needed, frames use a lazy representation. The provided
    // callbacks will be invoked to interface with it.
    void *frame_data;

    // This will be called to map the frame to the GPU, only if needed.
    //
    // `tex` is a pointer to an array of 4 texture objects (or NULL), which
    // *may* serve as backing storage for the texture being mapped. These are
    // intended to be recreated by `map`, e.g. using `pl_tex_recreate` or
    // `pl_upload_plane` as appropriate. They will be managed internally by
    // `pl_queue` and destroyed at some unspecified future point in time.
    //
    // Note: If `map` fails, it will not be retried, nor will `discard` be run.
    // The user should clean up state in this case.
    bool (*map)(pl_gpu gpu, pl_tex *tex, const struct pl_source_frame *src,
                struct pl_frame *out_frame);

    // If present, this will be called on frames that are done being used by
    // `pl_queue`. This may be useful to e.g. unmap textures backed by external
    // APIs such as hardware decoders. (Optional)
    void (*unmap)(pl_gpu gpu, struct pl_frame *frame, const struct pl_source_frame *src);

    // This function will be called for frames that are deemed unnecessary
    // (e.g. never became visible) and should instead be cleanly freed.
    // (Optional)
    void (*discard)(const struct pl_source_frame *src);
};

// Create a new, empty frame queue.
//
// It's highly recommended to fully render a single frame with `pts == 0.0`,
// and flush the GPU pipeline with `pl_gpu_finish`, prior to starting the timed
// playback loop.
pl_queue pl_queue_create(pl_gpu gpu);
void pl_queue_destroy(pl_queue *queue);

// Explicitly clear the queue. This is essentially equivalent to destroying
// and recreating the queue, but preserves any internal memory allocations.
//
// Note: Calling `pl_queue_reset` may block, if another thread is currently
// blocked on a different `pl_queue_*` call.
void pl_queue_reset(pl_queue queue);

// Explicitly push a frame. This is an alternative way to feed the frame queue
// with incoming frames, the other method being the asynchronous callback
// specified as `pl_queue_params.get_frame`. Both methods may be used
// simultaneously, although providing `get_frame` is recommended since it
// avoids the risk of the queue underrunning.
//
// When no more frames are available, call this function with `frame == NULL`
// to indicate EOF and begin draining the frame queue.
void pl_queue_push(pl_queue queue, const struct pl_source_frame *frame);

// Variant of `pl_queue_push` that blocks while the queue is judged
// (internally) to be "too full". This is useful for asynchronous decoder loops
// in order to prevent the queue from exhausting available RAM if frames are
// decoded significantly faster than they're displayed.
//
// The given `timeout` parameter specifies how long to wait before giving up,
// in nanoseconds. Returns false if this timeout was reached.
bool pl_queue_push_block(pl_queue queue, uint64_t timeout,
                         const struct pl_source_frame *frame);

struct pl_queue_params {
    // The PTS of the frame that will be rendered. This should be set to the
    // timestamp (in seconds) of the next vsync, relative to the initial frame.
    //
    // These must be monotonically increasing. To implement a discontinuous
    // jump, users must explicitly reset the frame queue with `pl_queue_reset`
    // and restart from PTS 0.0.
    float pts;

    // The radius of the configured mixer. This should be set to the value
    // as returned by `pl_frame_mix_radius`.
    float radius;

    // The estimated duration of a vsync, in seconds. This will only be used as
    // a hint, the true value will be estimated by comparing `pts` timestamps
    // between calls to `pl_queue_update`. (Optional)
    float vsync_duration;

    // If the difference between the (estimated) vsync duration and the
    // (measured) frame duration is smaller than this threshold, silently
    // disable interpolation and switch to ZOH semantics instead.
    //
    // For example, a value of 0.01 allows the FPS to differ by up to 1%
    // without being interpolated. Note that this will result in a continuous
    // phase drift unless also compensated for by the user, which will
    // eventually resulted in a dropped or duplicated frame. (Though this can
    // be preferable to seeing that same phase drift result in a temporally
    // smeared image)
    float interpolation_threshold;

    // Specifies how long `pl_queue_update` will wait for frames to become
    // available, in nanoseconds, before giving up and returning with
    // QUEUE_MORE.
    //
    // If `get_frame` is provided, this value is ignored by `pl_queue` and
    // should instead be interpreted by the provided callback.
    uint64_t timeout;

    // This callback will be used to pull new frames from the decoder. It may
    // block if needed. The user is responsible for setting appropriate time
    // limits and/or returning and interpreting QUEUE_MORE as sensible.
    //
    // Providing this callback is entirely optional. Users can instead choose
    // to manually feed the frame queue with new frames using `pl_queue_push`.
    enum pl_queue_status (*get_frame)(struct pl_source_frame *out_frame,
                                      const struct pl_queue_params *params);
    void *priv;
};

#define pl_queue_params(...) (&(struct pl_queue_params) { __VA_ARGS__ })

// Advance the frame queue's internal state to the target timestamp. Any frames
// which are no longer needed (i.e. too far in the past) are automatically
// unmapped and evicted. Any future frames which are needed to fill the queue
// must either have been pushed in advance, or will be requested using the
// provided `get_frame` callback. If you call this on `out_mix == NULL`, the
// queue state will advance, but no frames will be mapped.
//
// This function may return with PL_QUEUE_MORE, in which case the user may wish
// to ensure more frames are available and then re-run this function with the
// same parameters. In this case, `out_mix` is still written to, but it may be
// incomplete (or even contain no frames at all). Additionally, when the source
// contains interlaced frames (see `pl_source_frame.first_field`), this
// function may return with PL_QUEUE_MORE if a frame is missing references to
// a future frame.
//
// The resulting mix of frames in `out_mix` will represent the neighbourhood of
// the target timestamp, and can be passed to `pl_render_image_mix` as-is.
//
// Note: `out_mix` will only remain valid until the next call to
// `pl_queue_update` or `pl_queue_reset`.
enum pl_queue_status pl_queue_update(pl_queue queue, struct pl_frame_mix *out_mix,
                                     const struct pl_queue_params *params);

PL_API_END

#endif // LIBPLACEBO_FRAME_QUEUE_H
