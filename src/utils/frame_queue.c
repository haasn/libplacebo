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

#include <errno.h>
#include <math.h>

#include "common.h"
#include "log.h"
#include "pl_thread.h"

struct cache_entry {
    pl_tex tex[4];
};

struct entry {
    pl_rc_t rc;
    float pts;
    struct cache_entry cache;
    struct pl_source_frame src;
    struct pl_frame frame;
    uint64_t signature;
    bool mapped;
    bool ok;

    // for interlaced frames
    enum pl_field field;
    struct entry *primary;
    struct entry *prev, *next;
    bool dirty;
};

// Hard limits for vsync timing validity
#define MIN_FPS 10
#define MAX_FPS 200

// Limits for FPS estimation state
#define MAX_SAMPLES 32
#define MIN_SAMPLES 8

// Stickiness to prevent `interpolation_threshold` oscillation
#define THRESHOLD_MAX_RATIO 0.3
#define THRESHOLD_FRAMES 5

// Maximum number of not-yet-mapped frames to allow queueing in advance
#define PREFETCH_FRAMES 2

struct pool {
    float samples[MAX_SAMPLES];
    float estimate;
    float sum;
    int idx;
    int num;
    int total;
};

struct pl_queue_t {
    pl_gpu gpu;
    pl_log log;

    // For multi-threading, we use two locks. The `lock_weak` guards the queue
    // state itself. The `lock_strong` has a bigger scope and should be held
    // for the duration of any functions that expect the queue state to
    // remain more or less valid (with the exception of adding new members).
    //
    // In particular, `pl_queue_reset` and `pl_queue_update` will take
    // the strong lock, while `pl_queue_push_*` will only take the weak
    // lock.
    pl_mutex lock_strong;
    pl_mutex lock_weak;
    pl_cond wakeup;

    // Frame queue and state
    PL_ARRAY(struct entry *) queue;
    uint64_t signature;
    int threshold_frames;
    bool want_frame;
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

pl_queue pl_queue_create(pl_gpu gpu)
{
    pl_queue p = pl_alloc_ptr(NULL, p);
    *p = (struct pl_queue_t) {
        .gpu = gpu,
        .log = gpu->log,
    };

    pl_mutex_init(&p->lock_strong);
    pl_mutex_init(&p->lock_weak);
    PL_CHECK_ERR(pl_cond_init(&p->wakeup));
    return p;
}

static void recycle_cache(pl_queue p, struct cache_entry *cache, bool recycle)
{
    bool has_textures = false;
    for (int i = 0; i < PL_ARRAY_SIZE(cache->tex); i++) {
        if (!cache->tex[i])
            continue;

        has_textures = true;
        if (recycle) {
            pl_tex_invalidate(p->gpu, cache->tex[i]);
        } else {
            pl_tex_destroy(p->gpu, &cache->tex[i]);
        }
    }

    if (recycle && has_textures)
        PL_ARRAY_APPEND(p, p->cache, *cache);

    memset(cache, 0, sizeof(*cache)); // sanity
}

static void entry_deref(pl_queue p, struct entry **pentry, bool recycle)
{
    struct entry *entry = *pentry;
    *pentry = NULL;
    if (!entry || !pl_rc_deref(&entry->rc))
        return;

    if (!entry->mapped && entry->src.discard) {
        PL_TRACE(p, "Discarding unused frame id %"PRIu64" with PTS %f",
                 entry->signature, entry->src.pts);
        entry->src.discard(&entry->src);
    }

    if (entry->mapped && entry->ok && entry->src.unmap) {
        PL_TRACE(p, "Unmapping frame id %"PRIu64" with PTS %f",
                 entry->signature, entry->src.pts);
        entry->src.unmap(p->gpu, &entry->frame, &entry->src);
    }

    recycle_cache(p, &entry->cache, recycle);
    pl_free(entry);
}

static struct entry *entry_ref(struct entry *entry)
{
    pl_rc_ref(&entry->rc);
    return entry;
}

static void entry_cull(pl_queue p, struct entry *entry, bool recycle)
{
    // Forcibly clean up references to prev/next frames, even if `entry` has
    // remaining refs pointing at it. This is to prevent cyclic references.
    entry_deref(p, &entry->primary, recycle);
    entry_deref(p, &entry->prev, recycle);
    entry_deref(p, &entry->next, recycle);
    entry_deref(p, &entry, recycle);
}

void pl_queue_destroy(pl_queue *queue)
{
    pl_queue p = *queue;
    if (!p)
        return;

    for (int n = 0; n < p->queue.num; n++)
        entry_cull(p, p->queue.elem[n], false);
    for (int n = 0; n < p->cache.num; n++) {
        for (int i = 0; i < PL_ARRAY_SIZE(p->cache.elem[n].tex); i++)
            pl_tex_destroy(p->gpu, &p->cache.elem[n].tex[i]);
    }

    pl_cond_destroy(&p->wakeup);
    pl_mutex_destroy(&p->lock_weak);
    pl_mutex_destroy(&p->lock_strong);
    pl_free(p);
    *queue = NULL;
}

void pl_queue_reset(pl_queue p)
{
    pl_mutex_lock(&p->lock_strong);
    pl_mutex_lock(&p->lock_weak);

    for (int i = 0; i < p->queue.num; i++)
        entry_cull(p, p->queue.elem[i], false);

    *p = (struct pl_queue_t) {
        .gpu = p->gpu,
        .log = p->log,

        // Reuse lock objects
        .lock_strong = p->lock_strong,
        .lock_weak = p->lock_weak,
        .wakeup = p->wakeup,

        // Explicitly preserve allocations
        .queue.elem = p->queue.elem,
        .tmp_sig.elem = p->tmp_sig.elem,
        .tmp_ts.elem = p->tmp_ts.elem,
        .tmp_frame.elem = p->tmp_frame.elem,

        // Reuse GPU object cache entirely
        .cache = p->cache,
    };

    pl_cond_signal(&p->wakeup);
    pl_mutex_unlock(&p->lock_weak);
    pl_mutex_unlock(&p->lock_strong);
}

static inline float delta(float old, float new)
{
    return fabs((new - old) / PL_MIN(new, old));
}

static inline void default_estimate(struct pool *pool, float val)
{
    if (!pool->estimate && isnormal(val) && val > 0.0)
        pool->estimate = val;
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

static void queue_push(pl_queue p, const struct pl_source_frame *src)
{
    if (p->eof && !src)
        return; // ignore duplicate EOF

    if (p->eof && src) {
        PL_INFO(p, "Received frame after EOF signaled... discarding frame!");
        if (src->discard)
            src->discard(src);
        return;
    }

    pl_cond_signal(&p->wakeup);

    if (!src) {
        PL_TRACE(p, "Received EOF, draining frame queue...");
        p->eof = true;
        p->want_frame = false;
        return;
    }

    // Update FPS estimates if possible/reasonable
    default_estimate(&p->fps, src->first_field ? src->duration / 2 : src->duration);
    if (p->queue.num) {
        float last_pts = p->queue.elem[p->queue.num - 1]->pts;
        float delta = src->pts - last_pts;
        if (delta < 0.0) {
            PL_DEBUG(p, "Backwards source PTS jump %f -> %f", last_pts, src->pts);
        } else if (p->fps.estimate && delta > 10.0 * p->fps.estimate) {
            PL_DEBUG(p, "Discontinuous source PTS jump %f -> %f", last_pts, src->pts);
        } else {
            update_estimate(&p->fps, delta);
        }
    } else if (src->pts != 0) {
        PL_DEBUG(p, "First frame received with non-zero PTS %f", src->pts);
    }

    struct entry *entry = pl_alloc_ptr(NULL, entry);
    *entry = (struct entry) {
        .signature = p->signature++,
        .pts = src->pts,
        .src = *src,
    };
    pl_rc_init(&entry->rc);
    PL_ARRAY_POP(p->cache, &entry->cache);
    PL_TRACE(p, "Added new frame id %"PRIu64" with PTS %f",
             entry->signature, entry->pts);

    // Insert new entry into the correct spot in the queue, sorted by PTS
    for (int i = p->queue.num;; i--) {
        if (i == 0 || p->queue.elem[i - 1]->pts <= entry->pts) {
            if (src->first_field == PL_FIELD_NONE) {
                // Progressive
                PL_ARRAY_INSERT_AT(p, p->queue, i, entry);
                break;
            } else {
                // Interlaced
                struct entry *prev = i > 0 ? p->queue.elem[i - 1] : NULL;
                struct entry *next = i < p->queue.num ? p->queue.elem[i] : NULL;
                struct entry *entry2 = pl_zalloc_ptr(NULL, entry2);
                if (next) {
                    entry2->pts = (entry->pts + next->pts) / 2;
                } else if (src->duration) {
                    entry2->pts = entry->pts + src->duration / 2;
                } else if (p->fps.estimate) {
                    entry2->pts = entry->pts + p->fps.estimate;
                } else {
                    PL_ERR(p, "Frame with PTS %f specified as interlaced, but "
                           "no FPS information known yet! Please specify a "
                           "valid `pl_source_frame.duration`. Treating as "
                           "progressive...", src->pts);
                    PL_ARRAY_INSERT_AT(p, p->queue, i, entry);
                    pl_free(entry2);
                    break;
                }

                entry->field = src->first_field;
                entry2->primary = entry_ref(entry);
                entry2->field = pl_field_other(entry->field);
                entry2->signature = p->signature++;

                PL_TRACE(p, "Added second field id %"PRIu64" with PTS %f",
                         entry2->signature, entry2->pts);

                // Link previous/next frames
                if (prev) {
                    entry->prev = entry_ref(PL_DEF(prev->primary, prev));
                    entry2->prev = entry_ref(PL_DEF(prev->primary, prev));
                    // Retroactively re-link the previous frames that should
                    // be referencing this frame
                    for (int j = i - 1; j >= 0; --j) {
                        struct entry *e = p->queue.elem[j];
                        if (e != prev && e != prev->primary)
                            break;
                        entry_deref(p, &e->next, true);
                        e->next = entry_ref(entry);
                        if (e->dirty) { // reset signature to signal change
                            e->signature = p->signature++;
                            e->dirty = false;
                        }
                    }
                }

                if (next) {
                    entry->next = entry_ref(PL_DEF(next->primary, next));
                    entry2->next = entry_ref(PL_DEF(next->primary, next));
                    for (int j = i; j < p->queue.num; j++) {
                        struct entry *e = p->queue.elem[j];
                        if (e != next && e != next->primary)
                            break;
                        entry_deref(p, &e->prev, true);
                        e->prev = entry_ref(entry);
                        if (e->dirty) {
                            e->signature = p->signature++;
                            e->dirty = false;
                        }
                    }
                }

                PL_ARRAY_INSERT_AT(p, p->queue, i, entry);
                PL_ARRAY_INSERT_AT(p, p->queue, i+1, entry2);
                break;
            }
        }
    }

    p->want_frame = false;
}

void pl_queue_push(pl_queue p, const struct pl_source_frame *frame)
{
    pl_mutex_lock(&p->lock_weak);
    queue_push(p, frame);
    pl_mutex_unlock(&p->lock_weak);
}

static inline bool entry_mapped(struct entry *entry)
{
    return entry->mapped || (entry->primary && entry->primary->mapped);
}

static bool queue_has_room(pl_queue p)
{
    if (p->want_frame)
        return true;

    // Examine the queue tail
    for (int i = p->queue.num - 1; i >= 0; i--) {
        if (entry_mapped(p->queue.elem[i]))
            return true;
        if (p->queue.num - i >= PREFETCH_FRAMES)
            return false;
    }

    return true;
}

bool pl_queue_push_block(pl_queue p, uint64_t timeout,
                         const struct pl_source_frame *frame)
{
    pl_mutex_lock(&p->lock_weak);
    if (!timeout || !frame || p->eof)
        goto skip_blocking;

    while (!queue_has_room(p) && !p->eof) {
        if (pl_cond_timedwait(&p->wakeup, &p->lock_weak, timeout) == ETIMEDOUT) {
            pl_mutex_unlock(&p->lock_weak);
            return false;
        }
    }

skip_blocking:

    queue_push(p, frame);
    pl_mutex_unlock(&p->lock_weak);
    return true;
}

static void report_estimates(pl_queue p)
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

// note: may add more than one frame, since it releases the lock
static enum pl_queue_status get_frame(pl_queue p, const struct pl_queue_params *params)
{
    if (p->eof)
        return PL_QUEUE_EOF;

    if (!params->get_frame) {
        if (!params->timeout)
            return PL_QUEUE_MORE;

        p->want_frame = true;
        pl_cond_signal(&p->wakeup);

        while (p->want_frame) {
            if (pl_cond_timedwait(&p->wakeup, &p->lock_weak, params->timeout) == ETIMEDOUT)
                return PL_QUEUE_MORE;
        }

        return p->eof ? PL_QUEUE_EOF : PL_QUEUE_OK;
    }

    // Don't hold the weak mutex while calling into `get_frame`, to allow
    // `pl_queue_push` to run concurrently while we're waiting for frames
    pl_mutex_unlock(&p->lock_weak);

    struct pl_source_frame src;
    enum pl_queue_status ret;
    switch ((ret = params->get_frame(&src, params))) {
    case PL_QUEUE_OK:
        pl_queue_push(p, &src);
        break;
    case PL_QUEUE_EOF:
        pl_queue_push(p, NULL);
        break;
    case PL_QUEUE_MORE:
    case PL_QUEUE_ERR:
        break;
    }

    pl_mutex_lock(&p->lock_weak);
    return ret;
}

static inline bool map_frame(pl_queue p, struct entry *entry)
{
    if (!entry->mapped) {
        PL_TRACE(p, "Mapping frame id %"PRIu64" with PTS %f",
                 entry->signature, entry->pts);
        entry->mapped = true;
        entry->ok = entry->src.map(p->gpu, entry->cache.tex,
                                   &entry->src, &entry->frame);
        if (!entry->ok)
            PL_ERR(p, "Failed mapping frame id %"PRIu64" with PTS %f",
                   entry->signature, entry->pts);
    }

    return entry->ok;
}

static bool map_entry(pl_queue p, struct entry *entry)
{
    bool ok = map_frame(p, entry->primary ? entry->primary : entry);
    if (entry->prev)
        ok &= map_frame(p, entry->prev);
    if (entry->next)
        ok &= map_frame(p, entry->next);
    if (!ok)
        return false;

    if (entry->primary)
        entry->frame = entry->primary->frame;

    if (entry->field) {
        entry->frame.field = entry->field;
        entry->frame.first_field = PL_DEF(entry->primary, entry)->src.first_field;
        entry->frame.prev = entry->prev ? &entry->prev->frame : NULL;
        entry->frame.next = entry->next ? &entry->next->frame : NULL;
        entry->dirty = true;
    }

    return true;
}

static bool entry_complete(struct entry *entry)
{
    return entry->field ? !!entry->next : true;
}

// Advance the queue as needed to make sure idx 0 is the last frame before
// `pts`, and idx 1 is the first frame after `pts` (unless this is the last).
//
// Returns PL_QUEUE_OK only if idx 0 is still legal under ZOH semantics.
static enum pl_queue_status advance(pl_queue p, float pts,
                                    const struct pl_queue_params *params)
{
    // Cull all frames except the last frame before `pts`
    int culled = 0;
    for (int i = 1; i < p->queue.num; i++) {
        if (p->queue.elem[i]->pts <= pts) {
            entry_cull(p, p->queue.elem[i - 1], true);
            culled++;
        }
    }
    PL_ARRAY_REMOVE_RANGE(p->queue, 0, culled);

    // Keep adding new frames until we find one in the future, or EOF
    enum pl_queue_status ret = PL_QUEUE_OK;
    while (p->queue.num < 2) {
        switch ((ret = get_frame(p, params))) {
        case PL_QUEUE_ERR:
            return ret;
        case PL_QUEUE_EOF:
            if (!p->queue.num)
                return ret;
            goto done;
        case PL_QUEUE_MORE:
        case PL_QUEUE_OK:
            while (p->queue.num > 1 && p->queue.elem[1]->pts <= pts) {
                entry_cull(p, p->queue.elem[0], true);
                PL_ARRAY_REMOVE_AT(p->queue, 0);
            }
            if (ret == PL_QUEUE_MORE)
                return ret;
            continue;
        }
    }

    if (!entry_complete(p->queue.elem[1])) {
        switch (get_frame(p, params)) {
        case PL_QUEUE_ERR:
            return PL_QUEUE_ERR;
        case PL_QUEUE_MORE:
            ret = PL_QUEUE_MORE;
            // fall through
        case PL_QUEUE_EOF:
        case PL_QUEUE_OK:
            goto done;
        }
    }

done:
    if (p->eof && p->queue.num == 1) {
        if (p->queue.elem[0]->pts == 0.0 || !p->fps.estimate) {
            // If the last frame has PTS 0.0, or we have no FPS estimate, then
            // this is probably a single-frame file, in which case we want to
            // extend the ZOH to infinity, rather than returning. Not a perfect
            // heuristic, but w/e
            return PL_QUEUE_OK;
        }

        // Last frame is held for an extra `p->fps.estimate` duration,
        // afterwards this function just returns EOF.
        if (p->queue.elem[0]->pts + p->fps.estimate < pts) {
            entry_cull(p, p->queue.elem[0], true);
            p->queue.num = 0;
            return PL_QUEUE_EOF;
        }
    }

    pl_assert(p->queue.num);
    return ret;
}

static inline enum pl_queue_status point(pl_queue p, struct pl_frame_mix *mix,
                                         const struct pl_queue_params *params)
{
    // Find closest frame (nearest neighbour semantics)
    pl_assert(p->queue.num);
    struct entry *entry = p->queue.elem[0];
    double best = fabs(entry->pts - params->pts);
    for (int i = 1; i < p->queue.num; i++) {
        double dist = fabs(p->queue.elem[i]->pts - params->pts);
        if (dist < best) {
            entry = p->queue.elem[i];
            best = dist;
            continue;
        } else {
            break;
        }
    }

    if (!map_entry(p, entry))
        return PL_QUEUE_ERR;

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

    PL_TRACE(p, "Showing single frame id %"PRIu64" with PTS %f for target PTS %f",
             entry->signature, entry->pts, params->pts);

    report_estimates(p);
    return PL_QUEUE_OK;
}

// Present a single frame as appropriate for `pts`
static enum pl_queue_status nearest(pl_queue p, struct pl_frame_mix *mix,
                                    const struct pl_queue_params *params)
{
    enum pl_queue_status ret;
    switch ((ret = advance(p, params->pts, params))) {
    case PL_QUEUE_ERR:
    case PL_QUEUE_EOF:
        return ret;
    case PL_QUEUE_OK:
        break;
    case PL_QUEUE_MORE:
        if (!p->queue.num) {
            if (mix)
                *mix = (struct pl_frame_mix) {0};
            return ret;
        }
        break;
    }

    if (!mix)
        return PL_QUEUE_OK;

    return point(p, mix, params);
}

// Special case of `interpolate` for radius = 0, in which case we need exactly
// the previous frame and the following frame
static enum pl_queue_status oversample(pl_queue p, struct pl_frame_mix *mix,
                                       const struct pl_queue_params *params)
{
    enum pl_queue_status ret;
    switch ((ret = advance(p, params->pts, params))) {
    case PL_QUEUE_ERR:
    case PL_QUEUE_EOF:
        return ret;
    case PL_QUEUE_OK:
        break;
    case PL_QUEUE_MORE:
        if (!p->queue.num) {
            if (mix)
                *mix = (struct pl_frame_mix) {0};
            return ret;
        }
        break;
    }

    if (!mix)
        return PL_QUEUE_OK;

    // Can't oversample with only a single frame, fall back to point sampling
    if (p->queue.num < 2 || p->queue.elem[0]->pts > params->pts) {
        if (point(p, mix, params) != PL_QUEUE_OK)
            return PL_QUEUE_ERR;
        return ret;
    }

    struct entry *entries[2] = { p->queue.elem[0], p->queue.elem[1] };
    pl_assert(entries[0]->pts <= params->pts);
    pl_assert(entries[1]->pts >= params->pts);

    // Returning a mix containing both of these two frames
    p->tmp_sig.num = p->tmp_ts.num = p->tmp_frame.num = 0;
    for (int i = 0; i < 2; i++) {
        if (!map_entry(p, entries[i]))
            return PL_QUEUE_ERR;
        float ts = (entries[i]->pts - params->pts) / p->fps.estimate;
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
    return ret;
}

// Present a mixture of frames, relative to the vsync ratio
static enum pl_queue_status interpolate(pl_queue p, struct pl_frame_mix *mix,
                                        const struct pl_queue_params *params)
{
    // No FPS estimate available, possibly source contains only a single frame,
    // or this is the first frame to be rendered. Fall back to point sampling.
    if (!p->fps.estimate)
        return nearest(p, mix, params);

    // Silently disable interpolation if the ratio dips lower than the
    // configured threshold
    float ratio = fabs(p->fps.estimate / p->vps.estimate - 1.0);
    if (ratio < params->interpolation_threshold) {
        if (!p->threshold_frames) {
            PL_INFO(p, "Detected fps ratio %.4f below threshold %.4f, "
                    "disabling interpolation",
                    ratio, params->interpolation_threshold);
        }

        p->threshold_frames = THRESHOLD_FRAMES + 1;
        return nearest(p, mix, params);
    } else if (ratio < THRESHOLD_MAX_RATIO && p->threshold_frames > 1) {
        p->threshold_frames--;
        return nearest(p, mix, params);
    } else {
        if (p->threshold_frames) {
            PL_INFO(p, "Detected fps ratio %.4f exceeds threshold %.4f, "
                    "re-enabling interpolation",
                    ratio, params->interpolation_threshold);
        }
        p->threshold_frames = 0;
    }

    // No radius information, special case in which we only need the previous
    // and next frames.
    if (!params->radius)
        return oversample(p, mix, params);

    float min_pts = params->pts - params->radius * p->fps.estimate,
          max_pts = params->pts + params->radius * p->fps.estimate;

    enum pl_queue_status ret;
    switch ((ret = advance(p, min_pts, params))) {
    case PL_QUEUE_ERR:
    case PL_QUEUE_EOF:
        return ret;
    case PL_QUEUE_MORE:
        goto done;
    case PL_QUEUE_OK:
        break;
    }

    // Keep adding new frames until we've covered the range we care about
    pl_assert(p->queue.num);
    while (p->queue.elem[p->queue.num - 1]->pts < max_pts) {
        switch ((ret = get_frame(p, params))) {
        case PL_QUEUE_ERR:
            return ret;
        case PL_QUEUE_MORE:
        case PL_QUEUE_EOF:
            goto done;
        case PL_QUEUE_OK:
            continue;
        }
    }

    if (!entry_complete(p->queue.elem[p->queue.num - 1])) {
        ret = get_frame(p, params);
        if (ret == PL_QUEUE_ERR)
            return ret;
    }

done: ;

    if (!mix)
        return PL_QUEUE_OK;

    // Construct a mix object representing the current queue state, starting at
    // the last frame before `min_pts` to make sure there's a fallback frame
    // available for ZOH semantics.
    p->tmp_sig.num = p->tmp_ts.num = p->tmp_frame.num = 0;
    for (int i = 0; i < p->queue.num; i++) {
        struct entry *entry = p->queue.elem[i];
        if (entry->pts > max_pts)
            break;
        if (!map_entry(p, entry))
            return PL_QUEUE_ERR;
        float ts = (entry->pts - params->pts) / p->fps.estimate;
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

    PL_TRACE(p, "Showing mix of %d frames for target PTS %f:",
             mix->num_frames, params->pts);
    for (int i = 0; i < mix->num_frames; i++)
        PL_TRACE(p, "    id %"PRIu64" ts %f", mix->signatures[i], mix->timestamps[i]);

    report_estimates(p);
    return ret;
}

static bool prefill(pl_queue p, const struct pl_queue_params *params)
{
    int min_frames = 2 * ceilf(params->radius);
    min_frames = PL_MAX(min_frames, PREFETCH_FRAMES);

    while (p->queue.num < min_frames) {
        switch (get_frame(p, params)) {
        case PL_QUEUE_ERR:
            return false;
        case PL_QUEUE_EOF:
        case PL_QUEUE_MORE:
            return true;
        case PL_QUEUE_OK:
            continue;
        }
    }

    // In the most likely case, the first few frames will all be required. So
    // force-map them all to initialize GPU state on initial rendering. This is
    // better than the alternative of missing the cache later, when timing is
    // more relevant.
    for (int i = 0; i < min_frames; i++) {
        if (!map_entry(p, p->queue.elem[i]))
            return false;
    }

    return true;
}

enum pl_queue_status pl_queue_update(pl_queue p, struct pl_frame_mix *out_mix,
                                     const struct pl_queue_params *params)
{
    pl_mutex_lock(&p->lock_strong);
    pl_mutex_lock(&p->lock_weak);
    default_estimate(&p->fps, params->frame_duration);
    default_estimate(&p->vps, params->vsync_duration);

    float delta = params->pts - p->prev_pts;
    if (delta < 0.0) {

        // This is a backwards PTS jump. This is something we can handle
        // semi-gracefully, but only if we haven't culled past the current
        // frame yet.
        if (p->queue.num && p->queue.elem[0]->pts > params->pts) {
            PL_ERR(p, "Requested PTS %f is lower than the oldest frame "
                   "PTS %f. This is not supported, PTS must be monotonically "
                   "increasing! Please use `pl_queue_reset` to reset the frame "
                   "queue on discontinuous PTS jumps.",
                   params->pts, p->queue.elem[0]->pts);
            pl_mutex_unlock(&p->lock_weak);
            pl_mutex_unlock(&p->lock_strong);
            return PL_QUEUE_ERR;
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
        if (!prefill(p, params)) {
            pl_mutex_unlock(&p->lock_weak);
            pl_mutex_unlock(&p->lock_strong);
            return PL_QUEUE_ERR;
        }
    }

    // Ignore unrealistically high or low FPS, common near start of playback
    static const float max_vsync = 1.0 / MIN_FPS;
    static const float min_vsync = 1.0 / MAX_FPS;
    enum pl_queue_status ret;

    if (p->vps.estimate > min_vsync && p->vps.estimate < max_vsync) {
        // We know the vsync duration, so construct an interpolation mix
        ret = interpolate(p, out_mix, params);
    } else {
        // We don't know the vsync duration (yet), so just point-sample
        ret = nearest(p, out_mix, params);
    }

    pl_cond_signal(&p->wakeup);
    pl_mutex_unlock(&p->lock_weak);
    pl_mutex_unlock(&p->lock_strong);
    return ret;
}
