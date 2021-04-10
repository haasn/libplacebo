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

#include "common.h"
#include "context.h"

struct cache_entry {
    const struct pl_tex *tex[4];
};

struct entry {
    struct cache_entry cache;
    struct pl_source_frame src;
    struct pl_frame frame;
    uint64_t signature;
    bool mapped;
    bool ok;
};

// Hard limits for vsync timing validity
#define MIN_FPS 10
#define MAX_FPS 200

// Limits for FPS estimation state
#define MAX_SAMPLES 32
#define MIN_SAMPLES 8

struct pool {
    float samples[MAX_SAMPLES];
    float estimate;
    float sum;
    int idx;
    int num;
    int total;
};

struct pl_queue {
    const struct pl_gpu *gpu;
    struct pl_context *ctx;

    // Frame queue and state
    PL_ARRAY(struct entry) queue;
    uint64_t signature;
    bool eof;

    // Average vsync/frame fps estimation state
    struct pool vps, fps;
    float reported_vps;
    float reported_fps;
    float prev_pts;

    // Storage for temporary arrays
    PL_ARRAY(uint64_t) tmp_sig;
    PL_ARRAY(float) tmp_ts;
    PL_ARRAY(const struct pl_frame *) tmp_frame;

    // Queue of GPU objects to reuse
    PL_ARRAY(struct cache_entry) cache;
};

struct pl_queue *pl_queue_create(const struct pl_gpu *gpu)
{
    struct pl_queue *p = pl_alloc_ptr(NULL, p);
    *p = (struct pl_queue) {
        .gpu = gpu,
        .ctx = gpu->ctx,
    };

    return p;
}

static inline void unmap_frame(struct pl_queue *p, struct entry *entry)
{
    if (!entry->mapped && entry->src.discard) {
        PL_TRACE(p, "Discarding unused frame with PTS %f", entry->src.pts);
        entry->src.discard(&entry->src);
    }

    if (entry->mapped && entry->ok && entry->src.unmap) {
        PL_TRACE(p, "Unmapping frame with PTS %f", entry->src.pts);
        entry->src.unmap(p->gpu, &entry->frame, &entry->src);
    }
}


void pl_queue_destroy(struct pl_queue **queue)
{
    struct pl_queue *p = *queue;
    if (!p)
        return;

    for (int n = 0; n < p->queue.num; n++) {
        struct entry *entry = &p->queue.elem[n];
        unmap_frame(p, entry);
        for (int i = 0; i < PL_ARRAY_SIZE(entry->cache.tex); i++)
            pl_tex_destroy(p->gpu, &entry->cache.tex[i]);
    }

    for (int n = 0; n < p->cache.num; n++) {
        for (int i = 0; i < PL_ARRAY_SIZE(p->cache.elem[n].tex); i++)
            pl_tex_destroy(p->gpu, &p->cache.elem[n].tex[i]);
    }

    pl_free(p);
    *queue = NULL;
}

static inline void cull_entry(struct pl_queue *p, struct entry *entry)
{
    unmap_frame(p, entry);

    // Recycle non-empty texture cache entries
    static const struct cache_entry null_cache = {0};
    if (memcmp(&entry->cache, &null_cache, sizeof(null_cache)) != 0) {
        for (int i = 0; i < PL_ARRAY_SIZE(entry->cache.tex); i++) {
            if (entry->cache.tex[i])
                pl_tex_invalidate(p->gpu, entry->cache.tex[i]);
        }
        PL_ARRAY_APPEND(p, p->cache, entry->cache);
    }
}

void pl_queue_reset(struct pl_queue *p)
{
    for (int i = 0; i < p->queue.num; i++)
        cull_entry(p, &p->queue.elem[i]);

    *p = (struct pl_queue) {
        .gpu = p->gpu,
        .ctx = p->ctx,

        // Explicitly preserve allocations
        .queue.elem = p->queue.elem,
        .tmp_sig.elem = p->tmp_sig.elem,
        .tmp_ts.elem = p->tmp_ts.elem,
        .tmp_frame.elem = p->tmp_frame.elem,

        // Reuse GPU object cache entirely
        .cache = p->cache,
    };
}

static inline float delta(float old, float new)
{
    return fabs((new - old) / PL_MIN(new, old));
}

static inline void update_estimate(struct pool *pool, float cur)
{
    if (pool->num) {
        static const float max_delta = 0.3;
        if (delta(pool->sum / pool->num, cur) > max_delta) {
            pool->sum = 0.0;
            pool->num = pool->idx = 0;
        }
    }

    if (pool->num++ == MAX_SAMPLES) {
        pool->sum -= pool->samples[pool->idx];
        pool->num--;
    }

    pool->sum += pool->samples[pool->idx] = cur;
    pool->idx = (pool->idx + 1) % MAX_SAMPLES;
    pool->total++;

    if (pool->total < MIN_SAMPLES || pool->num >= MIN_SAMPLES)
        pool->estimate = pool->sum / pool->num;
}

void pl_queue_push(struct pl_queue *p, const struct pl_source_frame *src)
{
    if (!src) {
        PL_TRACE(p, "Received EOF, draining frame queue...");
        p->eof = true;
        return;
    }

    PL_TRACE(p, "Received new frame with PTS %f", src->pts);

    if (p->queue.num) {
        float prev_pts = p->queue.elem[p->queue.num - 1].src.pts;
        float delta = src->pts - prev_pts;
        if (delta < 0.0) {
            PL_WARN(p, "Backwards source PTS jump %f -> %f, discarding frame...",
                    prev_pts, src->pts);
            if (src->discard)
                src->discard(src);
            return;
        } else if (p->fps.estimate && delta > 10.0 * p->fps.estimate) {
            PL_WARN(p, "Discontinuous source PTS jump %f -> %f", prev_pts, src->pts);
        } else {
            update_estimate(&p->fps, delta);
        }
    } else if (src->pts != 0) {
        PL_DEBUG(p, "First frame received with non-zero PTS %f", src->pts);
    }

    struct cache_entry cache = {0};
    PL_ARRAY_POP(p->cache, &cache);
    PL_ARRAY_APPEND(p, p->queue, (struct entry) {
        .signature = p->signature++,
        .cache = cache,
        .src = *src,
    });
}

static void report_estimates(struct pl_queue *p)
{
    if (p->fps.total >= MIN_SAMPLES && p->vps.total >= MIN_SAMPLES) {
        if (p->reported_fps && p->reported_vps) {
            // Only re-report the estimates if they've changed considerably
            // from the previously reported values
            static const float report_delta = 0.3;
            float delta_fps = delta(p->reported_fps, p->fps.estimate);
            float delta_vps = delta(p->reported_vps, p->vps.estimate);
            if (delta_fps < report_delta && delta_vps < report_delta)
                return;
        }

        PL_INFO(p, "Estimated source FPS: %.3f, display FPS: %.3f",
                1.0 / p->fps.estimate, 1.0 / p->vps.estimate);

        p->reported_fps = p->fps.estimate;
        p->reported_vps = p->vps.estimate;
    }
}

static enum pl_queue_status get_frame(struct pl_queue *p,
                                      const struct pl_queue_params *params)
{
    if (p->eof)
        return QUEUE_EOF;

    if (!params->get_frame)
        return QUEUE_MORE;

    struct pl_source_frame src;
    enum pl_queue_status ret;
    switch ((ret = params->get_frame(&src, params))) {
    case QUEUE_OK:
        pl_queue_push(p, &src);
        break;
    case QUEUE_EOF:
        pl_queue_push(p, NULL);
        break;
    default: break;
    }

    return ret;
}

static bool map_frame(struct pl_queue *p, struct entry *entry)
{
    if (!entry->mapped) {
        PL_TRACE(p, "Mapping frame with PTS %f", entry->src.pts);
        entry->mapped = true;
        entry->ok = entry->src.map(p->gpu, entry->cache.tex,
                                   &entry->src, &entry->frame);
    }

    return entry->ok;
}

// Advance the queue as needed to make sure idx 0 is the last frame before
// `pts`, and idx 1 is the first frame after `pts` (unless this is the last).
//
// Returns QUEUE_OK only if idx 0 is still legal under ZOH semantics.
static enum pl_queue_status advance(struct pl_queue *p, float pts,
                                    const struct pl_queue_params *params)
{
    // Cull all frames except the last frame before `pts`
    int culled = 0;
    for (int i = 1; i < p->queue.num; i++) {
        if (p->queue.elem[i].src.pts <= pts) {
            cull_entry(p, &p->queue.elem[i - 1]);
            culled++;
        }
    }
    PL_ARRAY_REMOVE_RANGE(p->queue, 0, culled);

    // Keep adding new frames until we find one in the future, or EOF
    while (p->queue.num < 2) {
        enum pl_queue_status ret;
        switch ((ret = get_frame(p, params))) {
        case QUEUE_ERR:
        case QUEUE_MORE:
            return ret;
        case QUEUE_EOF:
            if (!p->queue.num)
                return ret;
            goto done;
        case QUEUE_OK:
            if (p->queue.num > 1 && p->queue.elem[1].src.pts <= pts) {
                cull_entry(p, &p->queue.elem[0]);
                PL_ARRAY_REMOVE_AT(p->queue, 0);
            }
            continue;
        }
    }

done:
    if (p->eof && p->queue.num == 1 && p->fps.estimate) {
        // Last frame is held for an extra `p->fps.estimate` duration,
        // afterwards this function just returns EOF.
        //
        // Note that if `p->fps.estimate` is not available, then we're
        // displaying a source that only has a single frame, in which case we
        // most likely just want to repeat it forever. (Not a perfect
        // heuristic, but w/e)
        if (p->queue.elem[0].src.pts + p->fps.estimate < pts) {
            cull_entry(p, &p->queue.elem[0]);
            p->queue.num = 0;
            return QUEUE_EOF;
        }
    }

    pl_assert(p->queue.num);
    return QUEUE_OK;
}

// Present a single frame as appropriate for `pts`
static enum pl_queue_status nearest(struct pl_queue *p, struct pl_frame_mix *mix,
                                    const struct pl_queue_params *params)
{
    enum pl_queue_status ret;
    if ((ret = advance(p, params->pts, params)))
        return ret;

    struct entry *entry = &p->queue.elem[0];
    if (!map_frame(p, entry))
        return QUEUE_ERR;

    // Return a mix containing only this single frame
    p->tmp_sig.num = p->tmp_ts.num = p->tmp_frame.num = 0;
    PL_ARRAY_APPEND(p, p->tmp_sig, entry->signature);
    PL_ARRAY_APPEND(p, p->tmp_frame, &entry->frame);
    PL_ARRAY_APPEND(p, p->tmp_ts, 0.0);
    *mix = (struct pl_frame_mix) {
        .num_frames = 1,
        .frames = p->tmp_frame.elem,
        .signatures = p->tmp_sig.elem,
        .timestamps = p->tmp_ts.elem,
        .vsync_duration = 1.0,
    };

    PL_TRACE(p, "Showing single frame with PTS %f for target PTS %f",
             entry->src.pts, params->pts);

    report_estimates(p);
    return QUEUE_OK;
}

// Special case of `interpolate` for radius = 0, in which case we need exactly
// the previous frame and the following frame
static enum pl_queue_status oversample(struct pl_queue *p, struct pl_frame_mix *mix,
                                       const struct pl_queue_params *params)
{
    enum pl_queue_status ret;
    if ((ret = advance(p, params->pts, params)))
        return ret;

    // Can't oversample with only a single frame, fall back to ZOH semantics
    if (p->queue.num < 2 || p->queue.elem[0].src.pts > params->pts)
        return nearest(p, mix, params);

    struct entry *entries[2] = { &p->queue.elem[0], &p->queue.elem[1] };
    pl_assert(entries[0]->src.pts <= params->pts);
    pl_assert(entries[1]->src.pts >= params->pts);

    // Returning a mix containing both of these two frames
    p->tmp_sig.num = p->tmp_ts.num = p->tmp_frame.num = 0;
    for (int i = 0; i < 2; i++) {
        if (!map_frame(p, entries[i]))
            return QUEUE_ERR;

        float ts = (entries[i]->src.pts - params->pts) / p->fps.estimate;
        PL_ARRAY_APPEND(p, p->tmp_sig, entries[i]->signature);
        PL_ARRAY_APPEND(p, p->tmp_frame, &entries[i]->frame);
        PL_ARRAY_APPEND(p, p->tmp_ts, ts);
    }

    *mix = (struct pl_frame_mix) {
        .num_frames = 2,
        .frames = p->tmp_frame.elem,
        .signatures = p->tmp_sig.elem,
        .timestamps = p->tmp_ts.elem,
        .vsync_duration = p->vps.estimate / p->fps.estimate,
    };

    PL_TRACE(p, "Oversampling 2 frames for target PTS %f:", params->pts);
    for (int i = 0; i < mix->num_frames; i++)
        PL_TRACE(p, "    id %"PRIu64" ts %f", mix->signatures[i], mix->timestamps[i]);

    report_estimates(p);
    return QUEUE_OK;
}

// Present a mixture of frames, relative to the vsync ratio
static enum pl_queue_status interpolate(struct pl_queue *p,
                                        struct pl_frame_mix *mix,
                                        const struct pl_queue_params *params)
{
    // No FPS estimate available, possibly source contains only a single file,
    // or this is the first frame to be rendered. Fall back to ZOH semantics.
    if (!p->fps.estimate)
        return nearest(p, mix, params);

    // No radius information, special case in which we only need the previous
    // and next frames.
    if (!params->radius)
        return oversample(p, mix, params);

    float min_pts = params->pts - params->radius * p->fps.estimate,
          max_pts = params->pts + params->radius * p->fps.estimate;

    enum pl_queue_status ret;
    if ((ret = advance(p, min_pts, params)))
        return ret;

    // Keep adding new frames until we've covered the range we care about
    pl_assert(p->queue.num);
    while (p->queue.elem[p->queue.num - 1].src.pts < max_pts) {
        switch ((ret = get_frame(p, params))) {
        case QUEUE_ERR:
        case QUEUE_MORE:
            return ret;
        case QUEUE_EOF:
            goto done;
        case QUEUE_OK:
            continue;
        }
    }

done: ;

    // Construct a mix object representing the current queue state, starting at
    // the last frame before `min_pts` to make sure there's a fallback frame
    // available for ZOH semantics.
    p->tmp_sig.num = p->tmp_ts.num = p->tmp_frame.num = 0;
    for (int i = 0; i < p->queue.num; i++) {
        struct entry *entry = &p->queue.elem[i];
        if (entry->src.pts > max_pts)
            break;
        if (!map_frame(p, entry))
            return QUEUE_ERR;

        float ts = (entry->src.pts - params->pts) / p->fps.estimate;
        PL_ARRAY_APPEND(p, p->tmp_sig, entry->signature);
        PL_ARRAY_APPEND(p, p->tmp_frame, &entry->frame);
        PL_ARRAY_APPEND(p, p->tmp_ts, ts);
    }

    *mix = (struct pl_frame_mix) {
        .num_frames = p->tmp_frame.num,
        .frames = p->tmp_frame.elem,
        .signatures = p->tmp_sig.elem,
        .timestamps = p->tmp_ts.elem,
        .vsync_duration = p->vps.estimate / p->fps.estimate,
    };

    pl_assert(mix->num_frames);
    PL_TRACE(p, "Showing mix of %d frames for target PTS %f:",
             mix->num_frames, params->pts);
    for (int i = 0; i < mix->num_frames; i++)
        PL_TRACE(p, "    id %"PRIu64" ts %f", mix->signatures[i], mix->timestamps[i]);

    report_estimates(p);
    return QUEUE_OK;
}

static bool prefill(struct pl_queue *p, const struct pl_queue_params *params)
{
    int min_frames = 2 * ceilf(params->radius);
    min_frames = PL_MAX(min_frames, 2);

    while (p->queue.num < min_frames) {
        switch (get_frame(p, params)) {
        case QUEUE_ERR:
            return false;
        case QUEUE_EOF:
        case QUEUE_MORE:
            return true;
        case QUEUE_OK:
            continue;
        }
    }

    // In the most likely case, the first few frames will all be required. So
    // force-map them all to initialize GPU state on initial rendering. This is
    // better than the alternative of missing the cache later, when timing is
    // more relevant.
    for (int i = 0; i < min_frames; i++) {
        if (!map_frame(p, &p->queue.elem[i]))
            return false;
    }

    return true;
}

static inline void default_estimate(struct pool *pool, float val)
{
    if (!pool->estimate && isnormal(val) && val > 0.0)
        pool->estimate = val;
}

enum pl_queue_status pl_queue_update(struct pl_queue *p,
                                     struct pl_frame_mix *out_mix,
                                     const struct pl_queue_params *params)
{
    default_estimate(&p->fps, params->frame_duration);
    default_estimate(&p->vps, params->vsync_duration);

    float delta = params->pts - p->prev_pts;
    if (delta < 0.0) {

        // This is a backwards PTS jump. This is something we can handle
        // semi-gracefully, but only if we haven't culled past the current
        // frame yet.
        if (p->queue.num && p->queue.elem[0].src.pts > params->pts) {
            PL_ERR(p, "Requested PTS %f is lower than the oldest frame "
                   "PTS %f. This is not supported, PTS must be monotonically "
                   "increasing! Please use `pl_queue_reset` to reset the frame "
                   "queue on discontinuous PTS jumps.",
                   params->pts, p->queue.elem[0].src.pts);
            return QUEUE_ERR;
        }

    } else if (delta > 1.0) {

        // A jump of more than a second is probably the result of a
        // discontinuous jump after a suspend. To prevent this from exploding
        // the FPS estimate, treat this as a new frame.
        PL_TRACE(p, "Discontinuous target PTS jump %f -> %f, ignoring...",
                 p->prev_pts, params->pts);

    } else if (delta > 0) {

        update_estimate(&p->vps, params->pts - p->prev_pts);

    }

    p->prev_pts = params->pts;

    // As a special case, prefill the queue if this is the first frame
    if (!params->pts && !p->queue.num) {
        if (!prefill(p, params))
            return QUEUE_ERR;
    }

    // Ignore unrealistically high or low FPS, common near start of playback
    static const float max_vsync = 1.0 / MIN_FPS;
    static const float min_vsync = 1.0 / MAX_FPS;
    if (p->vps.estimate > min_vsync && p->vps.estimate < max_vsync) {
        // We know the vsync duration, so construct an interpolation mix
        return interpolate(p, out_mix, params);
    } else {
        // We don't know the vsync duration (yet), so just point-sample the
        // nearest (zero-order-hold) frame
        return nearest(p, out_mix, params);
    }
}
