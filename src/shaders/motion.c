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

#include <stdlib.h>
#include <string.h>

#include "shaders.h"
#include "shaders/motion.h"

#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/motion.h>
#include <libplacebo/shaders/sampling.h>

const struct pl_motion_estimate_params pl_motion_estimate_default_params = {
    PL_MOTION_ESTIMATE_DEFAULTS
};

// One texture per level of an image pyramid, coarsest level last.
typedef PL_ARRAY(pl_tex) tex_pyramid_t;

// Persistent state and scratch space behind a PL_SHADER_OBJ_MOTION shader object.
struct sh_motion_obj {
    // --- Persistent across frames

    // Signatures of the frame pair the persistent state below describes, i.e.
    // the ones given to the most recent `pl_motion_estimate` call. `sig_b`
    // also identifies the frame `pyramid_prev` was built from.
    uint64_t sig_a, sig_b;

    // Luma pyramid of the *previous* call's frame `b`, reused as this call's
    // frame `a` pyramid. Only holds a usable pyramid if `pyramid_valid`.
    tex_pyramid_t pyramid_prev;
    bool pyramid_valid;

    // Block granular flow fields from each level of the image pyramid
    // searched by the most recent motion estimation invocation. Used to seed
    // the next motion estimation search.
    //
    // `fwd.elem[0]` and `bwd.elem[0]` are what `pl_motion_get_flow` would
    // return.
    // `valid` determines if `fwd` and `bwd` contain useful content.
    struct {
        tex_pyramid_t fwd;
        tex_pyramid_t bwd;
        enum pl_motion_direction valid; // directions actually computed
    } flow;

    // Dimensions of `flow.fwd.elem[0]` (number of blocks to cover the full
    // resolution).
    int grid_w, grid_h;

    // Global motion of each direction, as an SSBO laid out as
    // `global_buf_data`. Not allocated until `input->estimate_global` is set.
    // `valid` determines if `fwd` and `bwd` hold the global motion of the
    // directions in `flow.valid`. `hist` is scratch for computation.
    struct {
        pl_buf fwd;
        pl_buf bwd;
        pl_buf hist;
        bool valid;
    } global;

    // LUTs for the search candidate ordering.
    // `cand_lut[0]` holds the fine search radius LUT and `cand_lut[1]` holds
    // the coarse one.
    pl_shader_obj cand_lut[2];

    // --- Valid only for the duration of one pl_motion_estimate call
    struct {
        tex_pyramid_t levels; // luma pyramid of this call's frame `b`
        tex_pyramid_t raw;    // per-level search output
        tex_pyramid_t filtered; // per-level search output after median
                                // filtering
    } scratch;

    struct pl_motion_estimate_params params;
    int w, h;
};

// Don't reduce image pyramid to a size that is too small to recover motion.
#define MIN_COARSE_BLOCKS 8

// "Block" will be used to refer to a block of pixels that we group together
// for the motion search.
// "Tile" will be used to refer to a set of blocks that one workgroup operates
// on.
// `MOTION_BPGX` and `MOTION_BPGY` define the tile dimensions.
//
// TODO: benchmark the impacts of varying shape and size of tile to configure
// texel reuse and memory access patterns. Also needs to be revisited when we
// start the motion search at different locations for each block depending on
// the temporal state.
#define MOTION_BPGX 2
#define MOTION_BPGY 5

// Use a sufficiently large workgroup size of
// (MOTION_MEDIAN_WORKGROUP_DIM, MOTION_MEDIAN_WORKGROUP_DIM) for median
// filtering shader pass.
#define MOTION_MEDIAN_WORKGROUP_DIM 16

// Total number of bins considered when creating the histogram of global
// motion. NOTE: if the median motion is outside of [-`MOTION_GLOBAL_BINS/2`,
// `MOTION_GLOBAL_BINS-1`], the global motion is clamped.
#define MOTION_GLOBAL_BINS 256

// The shader that creates the histogram of motion uses a workgroup with the
// dimensions (`MOTION_GLOBAL_BINS`, `MOTION_GLOBAL_BINS`)
#define MOTION_GLOBAL_WORKGROUP_DIM 16
pl_static_assert(MOTION_GLOBAL_WORKGROUP_DIM *MOTION_GLOBAL_WORKGROUP_DIM ==
                 MOTION_GLOBAL_BINS);

// The different seed candidates that the motion search will consider, in the
// order that breaks ties.
enum motion_seed_kind {
    MOTION_SEED_ZERO,
    MOTION_SEED_PARENT,
    MOTION_SEED_PARENT_LEFT,
    MOTION_SEED_PARENT_RIGHT,
    MOTION_SEED_PARENT_UP,
    MOTION_SEED_PARENT_DOWN,
    MOTION_SEED_PRIOR,
    MOTION_SEED_NUM,
};

// See `motion_seeds_plan`.
struct motion_seeds {
    int rank[MOTION_SEED_NUM];
    int num;
};

// Plans which of the `motion_seed_kind` a given level of the motion search
// pyramid will consider. `rank[k]` == -1 when candidate `k` will not be
// considered, `rank[k]` == index of candidate `k` in ranking of seeds that
// will be considered.
static struct motion_seeds motion_seeds_plan(bool has_parent, bool has_prior)
{
    struct motion_seeds seeds = {0};
    for (int i = 0; i < MOTION_SEED_NUM; i++)
        seeds.rank[i] = -1;

    seeds.rank[MOTION_SEED_ZERO] = seeds.num++;

    if (has_parent) {
        seeds.rank[MOTION_SEED_PARENT] = seeds.num++;
        seeds.rank[MOTION_SEED_PARENT_LEFT] = seeds.num++;
        seeds.rank[MOTION_SEED_PARENT_RIGHT] = seeds.num++;
        seeds.rank[MOTION_SEED_PARENT_UP] = seeds.num++;
        seeds.rank[MOTION_SEED_PARENT_DOWN] = seeds.num++;
    }

    if (has_prior)
        seeds.rank[MOTION_SEED_PRIOR] = seeds.num++;

    return seeds;
}

// Constants that will be used for motion estimation. Varies depending on
// device capabilities and user selected parameters.
struct motion_geometry {
    int block_size;    // dimensions of square block size
    int radius;        // search radius, in pixels of the current image pyramid
                       // level
    int bpgx, bpgy;    // dimensions of tile
    int threads_block; // threads per block in the tile

    // The seed candidates every block of this level scores before searching.
    struct motion_seeds seeds;

    // "Candidate" will be used to refer to a motion vector (a vector of (dx, dy)
    // with 0 <= dx,dy <= radius) that is considered for a given block of pixels.
    // The ordering of the `(2 * radius + 1)^2` motion candidates in the
    // search radius, for tie breaking order.
    // For i in [0, `(2 * radius + 1)^2`), i is the ordering for tie breaking,
    // and (cand[2*i], cand[2*i+1]) represents (dx, dy) of the corresponding
    // motion search candidate.
    const int *cand;
};

// The size and stride of the various contents of shared memory that a motion
// search shader will use.
// "Window" will be used to refer to all of the pixels that are possible
// motion vector candidates for a given block.
// TODO: revisit strides for bank conflicts.
struct motion_shmem_layout {
    int win;          // dimensions of square window
    int win_stride;   // row stride inside a search window
    int block_stride; // stride between different windows of a tile
    size_t size;
};

// TODO: consider if we can pack multiple pixels into a single shared memory bank.
static struct motion_shmem_layout
motion_compute_shmem_layout(int block_size, int radius, int bpgx, int bpgy,
                            int nseed)
{
    const int win = block_size + 2 * radius;
    const int win_stride = win | 1;
    const int block_stride = (win * win_stride) | 1;
    const int blocks = bpgx * bpgy;

    // When `nseed` == 1, that is the case when the only seed candidate is
    // the zero vector. We do not need to score any candidates in this case.
    const size_t score = nseed > 1 ? (size_t) blocks * nseed : 0;

    const size_t words =
        (size_t) blocks * PL_SQUARE(block_size) // reference tile
        + (size_t) blocks * block_stride        // search windows
        + 2 * (size_t) blocks                   // cost and rank
        + 2 * (size_t) blocks                   // seed
        + score;                                // seed candidate scores

    return (struct motion_shmem_layout) {
        .win = win,
        .win_stride = win_stride,
        .block_stride = block_stride,
        .size = words * sizeof(uint32_t),
    };
}

// Fills the `out` array with `(2 * radius + 1)^2` many candidate displacements
// of the search window. Candidate i will always represent the displacement at
// (out[2*i], out[2*i + 1]), and the index i will be used to make the tie
// breaking order deterministic and favoring smaller displacements.
//
// We measure the magnitude of displacements using Chebyshev distance.
static void cand_fill(int radius, int *out)
{
    int num = 0;
    for (int ring = 0; ring <= radius; ring++) {
        for (int dy = -ring; dy <= ring; dy++) {
            for (int dx = -ring; dx <= ring; dx++) {
                const int adx = abs(dx), ady = abs(dy);
                if (PL_MAX(adx, ady) != ring)
                    continue; // everything closer has already been written

                out[2 * num + 0] = dx;
                out[2 * num + 1] = dy;
                num++;
            }
        }
    }

    pl_assert(num == PL_SQUARE(2 * radius + 1)); // correct number of elements
    pl_assert(!out[0] && !out[1]); // first element of `out` is (0,0)
}

// Selects an appropriate shader geometry (grid and workgroup dimensions) for
// the motion estimation. Iteratively reduces the tile size that each workgroup
// will process until a size is found that fits in shared memory.
static bool motion_plan(pl_gpu gpu, void *alloc, int block_size, int radius,
                        struct motion_seeds seeds, int grid_w, int grid_h,
                        struct motion_geometry *out)
{
    const int ncand = PL_SQUARE(2 * radius + 1);
    const int max_group = gpu->glsl.max_group_threads;
    const int max_x =
        gpu->glsl.max_group_size[0]; // TODO: check if this is limiting factor
    const int max_threads = PL_MIN(max_group, max_x);

    // This should not occur for any reasonable radius choice.
    const int threads_block = PL_MIN(ncand, max_threads);

    // This should not occur for any reasonable resolution of image.
    int bpgx = PL_MIN(MOTION_BPGX, grid_w);
    int bpgy = PL_MIN(MOTION_BPGY, grid_h);

    while (bpgx > 1 || bpgy > 1) {
        const struct motion_shmem_layout shmem = motion_compute_shmem_layout(
            block_size, radius, bpgx, bpgy, seeds.num);

        if (bpgx * bpgy * threads_block <= max_threads &&
            shmem.size <= gpu->glsl.max_shmem_size)
            break;

        if (bpgx >= bpgy)
            bpgx--;
        else
            bpgy--;
    }

    const struct motion_shmem_layout shmem =
        motion_compute_shmem_layout(block_size, radius, bpgx, bpgy, seeds.num);

    if (bpgx * bpgy * threads_block > max_threads) {
        PL_ERR(gpu,
               "pl_motion_estimate: search radius %d needs %d threads per work "
               "group, but this GPU supports at most %d!",
               radius, bpgx * bpgy * threads_block, max_threads);
        return false;
    }

    if (shmem.size > gpu->glsl.max_shmem_size) {
        PL_ERR(gpu,
               "pl_motion_estimate: block size %d with search radius %d needs "
               "%zu bytes of shared memory per work group, but this GPU "
               "provides only %zu!",
               block_size, radius, shmem.size, gpu->glsl.max_shmem_size);
        return false;
    }

    int *cand = pl_calloc(alloc, 2 * ncand, sizeof(*cand));
    cand_fill(radius, cand);

    *out = (struct motion_geometry) {
        .block_size = block_size,
        .radius = radius,
        .bpgx = bpgx,
        .bpgy = bpgy,
        .threads_block = threads_block,
        .seeds = seeds,
        .cand = cand,
    };

    return true;
}

// Destroys every level of `pyramid`, along with the array holding them.
static void pyramid_destroy(pl_gpu gpu, tex_pyramid_t *pyramid)
{
    for (int i = 0; i < pyramid->num; i++)
        pl_tex_destroy(gpu, &pyramid->elem[i]);
    pl_free_ptr(&pyramid->elem);
    pyramid->num = 0;
}

// Number of levels before degenerating into a single pixel.
static int levels_supported(int w, int h)
{
    int levels = 1;
    while (w > 1 || h > 1) {
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        levels++;
    }
    return levels;
}

// Number of levels to use in the absence of a user provided value.
static int levels_auto(int w, int h, int block_size)
{
    const int min_size = MIN_COARSE_BLOCKS * block_size;
    int levels = 1;
    while (PL_MIN((w + 1) / 2, (h + 1) / 2) >= min_size) {
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        levels++;
    }
    return levels;
}

static int pyramid_num_levels(int w, int h,
                              const struct pl_motion_estimate_params *params)
{
    if (!params->pyramid_levels)
        return levels_auto(w, h, params->block_size);

    return PL_MIN(params->pyramid_levels, levels_supported(w, h));
}

// Allocate or resize `pyramid` to hold exactly `levels` textures, level `i`
// being the input resolution halved `i` times (rounding up).
static bool pyramid_recreate(pl_gpu gpu, void *alloc, tex_pyramid_t *pyramid,
                             int w, int h, int levels, pl_fmt fmt)
{
    // If size of the frame reduced or block size increased since previous motion
    // estimate, destroy the pyramid allocations that would be unused.
    for (int i = levels; i < pyramid->num; i++)
        pl_tex_destroy(gpu, &pyramid->elem[i]);

    // If size of the frame increased since previous motion estimate, or this
    // is a cold start, prepare the indices to be allocated by `pl_tex_recreate`.
    PL_ARRAY_RESIZE(alloc, *pyramid, levels);
    for (int i = pyramid->num; i < levels; i++)
        pyramid->elem[i] = NULL;
    pyramid->num = levels;

    for (int i = 0; i < levels; i++) {
        bool ok = pl_tex_recreate(
            gpu, &pyramid->elem[i],
            pl_tex_params(.w = w, .h = h, .format = fmt, .sampleable = true,
                          .renderable = true,
                          .storable = fmt->caps & PL_FMT_CAP_STORABLE,
                          .debug_tag = PL_DEBUG_TAG, ));

        if (!ok) {
            PL_ERR(gpu,
                   "pl_motion_estimate: failed creating %dx%d pyramid "
                   "level!",
                   w, h);
            return false;
        }

        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }

    return true;
}

// Format for motion vector fields, holding `vec3(mv.x, mv.y, confidence)`,
// with an empty fourth element for padding. mv.x and mv.y are whole number
// floats. Prefer 16 bit floats but fall back to 32 bits.
static pl_fmt flow_fmt(pl_gpu gpu)
{
    static const enum pl_fmt_caps caps = PL_FMT_CAP_STORABLE |
                                         PL_FMT_CAP_LINEAR;

    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 0, caps);
    if (fmt)
        return fmt;

    return pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 0, caps);
}

// Similar to `pyramid_recreate`, except does not invalidate the contents of
// `pyramid` when the dimensions and levels of the flow pyramid has not changed
// since the previous call.
static bool flow_pyramid_recreate(pl_gpu gpu, void *alloc,
                                  tex_pyramid_t *pyramid, int w, int h,
                                  int levels, int block_size, pl_fmt fmt)
{
    bool reusable = pyramid->num == levels;
    for (int i = 0, lw = w, lh = h; reusable && i < levels; i++) {
        pl_tex tex = pyramid->elem[i];
        reusable = tex && tex->params.format == fmt &&
                   tex->params.w == PL_DIV_UP(lw, block_size) &&
                   tex->params.h == PL_DIV_UP(lh, block_size);
        lw = (lw + 1) / 2;
        lh = (lh + 1) / 2;
    }

    if (reusable)
        return true;

    // Free levels that are now unused.
    for (int i = levels; i < pyramid->num; i++)
        pl_tex_destroy(gpu, &pyramid->elem[i]);

    // Clear levels that will be reused.
    PL_ARRAY_RESIZE(alloc, *pyramid, levels);
    for (int i = pyramid->num; i < levels; i++)
        pyramid->elem[i] = NULL;
    pyramid->num = levels;

    for (int i = 0; i < levels; i++) {
        // Use PL_DIV_UP to account for trailing partial blocks.
        const int grid_w = PL_DIV_UP(w, block_size);
        const int grid_h = PL_DIV_UP(h, block_size);

        bool ok = pl_tex_recreate(
            gpu, &pyramid->elem[i],
            pl_tex_params(.w = grid_w, .h = grid_h, .format = fmt,
                          .sampleable = true, .storable = true,
                          .debug_tag = PL_DEBUG_TAG, ));

        if (!ok) {
            PL_ERR(gpu, "pl_motion_estimate: failed creating %dx%d flow field!",
                   grid_w, grid_h);
            return false;
        }

        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }

    return true;
}

// Downscale `src` by one pyramid level into `dst` using the Burt-Adelson REDUCE
// filter, i.e. convolution with the separable binomial kernel with a weighting
// of [1 4 6 4 1]/16. `dst` must measure exactly `src` halved in each dimension,
// rounding up.
static bool pyramid_reduce(pl_dispatch dispatch, pl_tex src, pl_tex dst)
{
    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "pyramid REDUCE %dx%d -> %dx%d", src->params.w,
                 src->params.h, dst->params.w, dst->params.h);
    sh->output = PL_SHADER_SIG_COLOR;

    // Align the output texel (i,j) on the center of source texel (2i,2j).
    const pl_rect2df rect = {
        .x0 = -0.5f,
        .y0 = -0.5f,
        .x1 = 2.0f * dst->params.w - 0.5f,
        .y1 = 2.0f * dst->params.h - 0.5f,
    };

    ident_t pos, pt;
    ident_t tex = sh_bind(sh, src, PL_TEX_ADDRESS_MIRROR, PL_TEX_SAMPLE_LINEAR,
                          "src", &rect, &pos, &pt);
    if (!tex) {
        pl_dispatch_abort(dispatch, &sh);
        return false;
    }

    // clang-format off
#pragma GLSL /* pyramid_reduce */                                               \
    vec4 color;                                                                 \
    {                                                                           \
        vec2 c = $pos;                                                          \
        vec2 d = 1.2 * $pt;                                                     \
        float sum  = 36.0 * textureLod($tex, c, 0.0).r;                         \
        sum += 30.0 * textureLod($tex, c - vec2(d.x, 0.0), 0.0).r;              \
        sum += 30.0 * textureLod($tex, c + vec2(d.x, 0.0), 0.0).r;              \
        sum += 30.0 * textureLod($tex, c - vec2(0.0, d.y), 0.0).r;              \
        sum += 30.0 * textureLod($tex, c + vec2(0.0, d.y), 0.0).r;              \
        sum += 25.0 * textureLod($tex, c - d, 0.0).r;                           \
        sum += 25.0 * textureLod($tex, c + d, 0.0).r;                           \
        sum += 25.0 * textureLod($tex, c - vec2(d.x, -d.y), 0.0).r;             \
        sum += 25.0 * textureLod($tex, c + vec2(d.x, -d.y), 0.0).r;             \
        color = vec4(sum * (1.0 / 256.0), 0.0, 0.0, 1.0);                       \
    }
    // clang-format on

    return pl_dispatch_finish(dispatch, pl_dispatch_params(
        .shader = &sh,
        .target = dst,
    ));
}

// pyramid->elem[0] must contain the luma plane of the corresponding frame.
static bool pyramid_build(pl_gpu gpu, pl_dispatch dispatch,
                          const tex_pyramid_t *pyramid)
{
    for (int i = 1; i < pyramid->num; i++) {
        if (!pyramid_reduce(dispatch, pyramid->elem[i-1], pyramid->elem[i])) {
            PL_ERR(gpu, "pl_motion_estimate: failed reducing pyramid level "
                   "%d!", i);
            return false;
        }
    }

    return true;
}

static void motion_uninit(pl_gpu gpu, void *ptr)
{
    struct sh_motion_obj *obj = ptr;
    pyramid_destroy(gpu, &obj->pyramid_prev);
    pyramid_destroy(gpu, &obj->scratch.levels);
    pyramid_destroy(gpu, &obj->scratch.raw);
    pyramid_destroy(gpu, &obj->scratch.filtered);
    pyramid_destroy(gpu, &obj->flow.fwd);
    pyramid_destroy(gpu, &obj->flow.bwd);
    pl_buf_destroy(gpu, &obj->global.fwd);
    pl_buf_destroy(gpu, &obj->global.bwd);
    pl_buf_destroy(gpu, &obj->global.hist);
    for (int i = 0; i < PL_ARRAY_SIZE(obj->cand_lut); i++)
        pl_shader_obj_destroy(&obj->cand_lut[i]);
    memset(obj, 0, sizeof(*obj));
}

// Drop everything that only makes sense in the context of a particular frame
// sequence. Invalidates but does not release allocated GPU buffers.
static void invalidate_temporal(pl_gpu gpu, struct sh_motion_obj *obj,
                                const char *reason)
{
    PL_DEBUG(gpu, "pl_motion: discarding temporal state: %s", reason);

    obj->sig_a = obj->sig_b = 0;
    obj->flow.valid = 0;
    obj->global.valid = false;
    obj->pyramid_valid = false;

    pl_buf_destroy(gpu, &obj->global.fwd);
    pl_buf_destroy(gpu, &obj->global.bwd);
    pl_buf_destroy(gpu, &obj->global.hist);
}

int pl_motion_num_blocks(int w, int h,
                         const struct pl_motion_estimate_params *params)
{
    const int bs = params->block_size;
    if (w <= 0 || h <= 0 || bs <= 0)
        return 0;

    // Must match the grid computed by `pl_motion_estimate`.
    return PL_DIV_UP(w, bs) * PL_DIV_UP(h, bs);
}

void pl_motion_reset(pl_shader_obj state)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    if (!obj)
        return;

    invalidate_temporal(state->gpu, obj, "reset by application");
}

pl_tex pl_motion_get_flow(pl_shader_obj state, enum pl_motion_direction dir)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    if (!obj || !(obj->flow.valid & dir))
        return NULL;

    switch (dir) {
    case PL_MOTION_FORWARD:
        return obj->flow.fwd.elem[0];
    case PL_MOTION_BACKWARD:
        return obj->flow.bwd.elem[0];
    default:
        break;
    }

    return NULL; // `dir` cannot name `PL_MOTION_BOTH`, we can only return
                 // one flow field.
}

ident_t sh_motion_flow(pl_shader sh, pl_shader_obj state,
                       enum pl_motion_direction dir)
{
    pl_tex flow = pl_motion_get_flow(state, dir);
    if (!flow)
        return NULL_IDENT;

    return sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name = "motion_flow",
            .type = PL_DESC_SAMPLED_TEX,
        },
        .binding = {
            .object      = flow,
            .sample_mode = PL_TEX_SAMPLE_NEAREST,
        },
    });
}

pl_buf pl_motion_get_global_buf(pl_shader_obj state,
                                enum pl_motion_direction dir)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    if (!obj || !obj->global.valid || !(obj->flow.valid & dir))
        return NULL;

    switch (dir) {
    case PL_MOTION_FORWARD:
        return obj->global.fwd;
    case PL_MOTION_BACKWARD:
        return obj->global.bwd;
    default:
        break;
    }

    return NULL; // `dir` cannot name `PL_MOTION_BOTH`, we can only return
                 // one global motion vector.
}

struct global_buf_data {
    float global_mv[2];
    float global_confidence;
};

const struct pl_buffer_var
    pl_motion_global_buf_vars[PL_MOTION_BACKWARD + 1]
                             [PL_MOTION_GLOBAL_BUF_NUM_VARS] = {
#define VAR(field, suffix, dim) {                                               \
    .var = {                                                                    \
        .name = #field "_" #suffix,                                             \
        .type = PL_VAR_FLOAT,                                                   \
        .dim_v = dim,                                                           \
        .dim_m = 1,                                                             \
        .dim_a = 1,                                                             \
    },                                                                          \
    .layout = {                                                                 \
        .offset = offsetof(struct global_buf_data, field),                      \
        .size   = sizeof(((struct global_buf_data *) NULL)->field),             \
        .stride = sizeof(((struct global_buf_data *) NULL)->field),             \
    },                                                                          \
}
    // NOTE: index 0 is empty.
    [PL_MOTION_FORWARD] = {
        VAR(global_mv, fwd, 2),
        VAR(global_confidence, fwd, 1),
    },
    [PL_MOTION_BACKWARD] = {
        VAR(global_mv, bwd, 2),
        VAR(global_confidence, bwd, 1),
    },
#undef VAR
};

ident_t sh_motion_global(pl_shader sh, pl_shader_obj state,
                         enum pl_motion_direction dir)
{
    pl_buf global = pl_motion_get_global_buf(state, dir);
    if (!global)
        return NULL_IDENT;

    pl_assert(global->params.size >= sizeof(struct global_buf_data));

    sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name   = "global_motion_desc",
            .type   = PL_DESC_BUF_STORAGE,
            .access = PL_DESC_ACCESS_READONLY,
        },
        .binding.object  = global,
        .buffer_vars =
            (struct pl_buffer_var *) pl_motion_global_buf_vars[dir],
        .num_buffer_vars = PL_MOTION_GLOBAL_BUF_NUM_VARS,
    });

    ident_t gm = sh_fresh(sh, "global_motion");
    // clang-format off
#pragma GLSLP /* sh_motion_global */                                            \
    @if (dir == PL_MOTION_FORWARD)                                              \
    #define $gm vec3(global_mv_fwd, global_confidence_fwd)                      \
    @else                                                                       \
    #define $gm vec3(global_mv_bwd, global_confidence_bwd)                      \

    // clang-format on
    return gm;
}

bool pl_motion_get_global(pl_shader_obj state, enum pl_motion_direction dir,
                          float *out_x, float *out_y, float *out_confidence)
{
    pl_buf global = pl_motion_get_global_buf(state, dir);
    if (!global)
        return false;

    struct global_buf_data data;
    if (!pl_buf_read(state->gpu, global, 0, &data, sizeof(data)))
        return false;

    if (out_x)
        *out_x = data.global_mv[0];
    if (out_y)
        *out_y = data.global_mv[1];
    if (out_confidence)
        *out_confidence = data.global_confidence;
    return true;
}

static bool params_eq(const struct pl_motion_estimate_params *a,
                      const struct pl_motion_estimate_params *b)
{
    return a->block_size == b->block_size &&
           a->pyramid_levels == b->pyramid_levels &&
           a->search_radius_coarse == b->search_radius_coarse &&
           a->search_radius_fine == b->search_radius_fine;
}

static bool validate(pl_gpu gpu, const struct pl_motion_estimate_input *input,
                     const struct pl_motion_estimate_params *params)
{
    if (!gpu->glsl.compute) {
        PL_ERR(gpu, "pl_motion_estimate: requires compute shader support!");
        return false;
    }

    if (!input->a.tex || !input->b.tex) {
        PL_ERR(gpu, "pl_motion_estimate: missing input texture!");
        return false;
    }

    const struct pl_tex_params *a = &input->a.tex->params;
    const struct pl_tex_params *b = &input->b.tex->params;
    if (a->w != b->w || a->h != b->h) {
        PL_ERR(gpu, "pl_motion_estimate: input dimensions differ (%dx%d vs "
               "%dx%d)!", a->w, a->h, b->w, b->h);
        return false;
    }

    if (!a->sampleable || !b->sampleable) {
        PL_ERR(gpu, "pl_motion_estimate: input textures must be sampleable!");
        return false;
    }

    if (input->a.sig == input->b.sig) {
        PL_ERR(gpu, "pl_motion_estimate: a.sig and b.sig must identify "
               "distinct frames (both 0x%llx)!",
               (unsigned long long) input->a.sig);
        return false;
    }

    if (input->direction & ~(unsigned) PL_MOTION_BOTH || !input->direction) {
        PL_ERR(gpu, "pl_motion_estimate: invalid direction %d!",
               (int) input->direction);
        return false;
    }

    if (params->block_size < 1) {
        PL_ERR(gpu, "pl_motion_estimate: block_size must be positive!");
        return false;
    }

    if (params->block_size > 64) {
        PL_ERR(gpu, "pl_motion_estimate: block_size must be <= 64!");
        return false;
    }

    if (params->pyramid_levels < 0) {
        PL_ERR(gpu, "pl_motion_estimate: pyramid_levels must be non-negative!");
        return false;
    }

    if (params->search_radius_coarse < 0 || params->search_radius_fine < 0) {
        PL_ERR(gpu, "pl_motion_estimate: search radii must be non-negative!");
        return false;
    }

    if (params->search_radius_coarse > 64 || params->search_radius_fine > 64) {
        PL_ERR(gpu, "pl_motion_estimate: search radii must be <= 64!");
        return false;
    }

    return true;
}

// Convert an input frame to luma.
static bool convert_to_luma(pl_dispatch dispatch,
                            const struct pl_motion_estimate_frame *frame,
                            pl_tex out)
{
    struct pl_color_repr repr = frame->repr;
    struct pl_color_space color = frame->color;
    pl_color_space_infer(&color);

    pl_shader sh = pl_dispatch_begin(dispatch);
    if (!pl_shader_sample_direct(sh, pl_sample_src( .tex = frame->tex ))) {
        pl_dispatch_abort(dispatch, &sh);
        return false;
    }

    // Decode frame to RGB if required. `pl_shader_decode_color_ex` also ensures
    // `PL_ALPHA_INDEPENDENT`, which is required by `pl_shader_extract_features`.
    pl_shader_decode_color_ex(sh, pl_color_decode_args( .repr = &repr ));

    // Convert to luma (IPT intensity). `pl_shader_extract_features` linearizes
    // for us.
    pl_shader_extract_features(sh, color);

    return pl_dispatch_finish(dispatch, pl_dispatch_params(
        .shader = &sh,
        .target = out,
    ));
}

static void cand_lut_fill(void *data, const struct sh_lut_params *params)
{
    const struct motion_geometry *geom = params->priv;
    memcpy(data, geom->cand, 2 * (size_t) params->width * sizeof(int));
}

// All of the parameters for a motion search shader for one level of the image
// pyramid.
struct motion_search {
    pl_tex ref;                         // reference frame luma
    pl_tex tgt;                         // target frame luma
    pl_tex parent; // flow field of the previous motion search image pyramid
                   // level, or NULL at the coarsest level
    pl_tex prior;  // flow field produced at the same image pyramid level by
                   // the motion search invocation on the previous frame pair,
                   // or NULL when there is none available
    pl_tex out_mv;                      // block granular flow field
    const struct motion_geometry *geom; // compile time constants
    pl_shader_obj *cand_lut;
};

// Dispatches the motion search shader for every block of `p->out_mv`.
//
// Uses the sum of squared distances (SSD) to compute the most likely
// motion vector that describes the motion of a block of pixels from one
// frame to the next.
static bool motion_search_pass(pl_gpu gpu, pl_dispatch dispatch,
                               const struct motion_search *p)
{
    const struct motion_geometry *geom = p->geom;
    const int bs      = geom->block_size;
    const int r       = geom->radius;
    const int bpgx    = geom->bpgx, bpgy = geom->bpgy;
    const int bpg     = bpgx * bpgy;
    const int tpb     = geom->threads_block;
    const int threads = bpg * tpb;
    const int ncand   = PL_SQUARE(2 * r + 1);
    const int nseed = geom->seeds.num;

    // Total size of tile.
    const int ref_w = bpgx * bs, ref_h = bpgy * bs;

    const struct motion_shmem_layout shmem =
        motion_compute_shmem_layout(bs, r, bpgx, bpgy, nseed);
    const int win = shmem.win;
    const int win_stride = shmem.win_stride;
    const int block_stride = shmem.block_stride;

    const int tex_w = p->ref->params.w, tex_h = p->ref->params.h;
    const int grid_w = p->out_mv->params.w, grid_h = p->out_mv->params.h;

    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh,
                 "exhaustive motion search (%dx%d blocks, radius %d, %d seed "
                 "candidates)",
                 bs, bs, r, nseed);
    sh->output = PL_SHADER_SIG_NONE;

    if (!sh_try_compute(sh, threads, 1, false, shmem.size)) {
        PL_ERR(gpu,
               "pl_motion_estimate: cannot create motion search shader, needs "
               "%d threads and %zu bytes of shared memory per work group!",
               threads, shmem.size);
        goto error;
    }

    PL_DEBUG(gpu, "pl_motion_estimate: searching %dx%d -> %dx%d blocks: block "
             "size %d, radius %d, %d candidates, %dx%d block tile, %d threads, "
             "%zu bytes of shared memory",
             tex_w, tex_h, grid_w, grid_h, bs, r, ncand, bpgx, bpgy, threads,
             shmem.size);

    // The rank of a candidate is its index in the LUT, derived from its thread
    // index, so the reduction can minimize the rank to resolve ties
    // deterministically.
    ident_t cand = sh_lut(sh, sh_lut_params(
        .object   = p->cand_lut,
        .var_type = PL_VAR_SINT,
        .lut_type = SH_LUT_UNIFORM,
        .method   = SH_LUT_NONE,
        .width    = ncand,
        .comps    = 2,
        .fill     = cand_lut_fill,
        .priv     = (void *) geom,
    ));

    if (!cand) {
        PL_ERR(gpu, "pl_motion_estimate: failed uploading candidate LUT!");
        goto error;
    }

    // Describe the textures for the reference frames and motion vector output.
    ident_t ref = sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name = "ref",
            .type = PL_DESC_SAMPLED_TEX,
        },
        .binding = {
            .object      = p->ref,
            .sample_mode = PL_TEX_SAMPLE_NEAREST,
        },
    });

    ident_t tgt = sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name = "tgt",
            .type = PL_DESC_SAMPLED_TEX,
        },
        .binding = {
            .object      = p->tgt,
            .sample_mode = PL_TEX_SAMPLE_NEAREST,
        },
    });

    ident_t out = sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name   = "out_mv",
            .type   = PL_DESC_STORAGE_IMG,
            .access = PL_DESC_ACCESS_WRITEONLY,
        },
        .binding.object = p->out_mv,
    });

    // Bind the flow field of the previous motion search image pyramid level,
    // unless this is the coarsest (first) level of the motion search.
    const bool has_parent = p->parent != NULL;
    const int parent_w = has_parent ? p->parent->params.w : 0;
    const int parent_h = has_parent ? p->parent->params.h : 0;
    const int rank_parent = geom->seeds.rank[MOTION_SEED_PARENT];
    const int rank_left = geom->seeds.rank[MOTION_SEED_PARENT_LEFT];
    const int rank_right = geom->seeds.rank[MOTION_SEED_PARENT_RIGHT];
    const int rank_up = geom->seeds.rank[MOTION_SEED_PARENT_UP];
    const int rank_down = geom->seeds.rank[MOTION_SEED_PARENT_DOWN];
    ident_t parent = NULL_IDENT;
    if (has_parent) {
        parent = sh_desc(sh, (struct pl_shader_desc) {
            .desc = {
                .name = "parent",
                .type = PL_DESC_SAMPLED_TEX,
            },
            .binding = {
                .object      = p->parent,
                .sample_mode = PL_TEX_SAMPLE_NEAREST,
            },
        });
    }

    // Bind the flow field of the same motion search image pyramid level from
    // the previous motion search invocation on the previous frame pair. Don't
    // bind anything when there is no usable prior available (upon a cold
    // start, after a reset, or a change in motion estimation direction).
    const bool has_prior = p->prior != NULL;
    const int prior_w = has_prior ? p->prior->params.w : 0;
    const int prior_h = has_prior ? p->prior->params.h : 0;
    const int rank_prior = geom->seeds.rank[MOTION_SEED_PRIOR];
    ident_t prior = NULL_IDENT;
    if (has_prior) {
        prior = sh_desc(sh, (struct pl_shader_desc) {
            .desc = {
                .name = "prior",
                .type = PL_DESC_SAMPLED_TEX,
            },
            .binding = {
                .object      = p->prior,
                .sample_mode = PL_TEX_SAMPLE_NEAREST,
            },
        });
    }

    // clang-format off
#pragma GLSLH /* motion_search_pass */                                          \
    /* Shared memory allocations and helper functions.                          \
     * TODO: see if we can convert to int8 luma beforehand, or if we can pack   \
     * multiple texels into a single load or shared memory bank. */             \
                                                                                \
    /* All pixels of the tile in the reference frame. */                        \
    shared uint sRef[${const int: ref_w * ref_h}];                              \
    /* Search windows in the target frame for each block of the tile. */        \
    shared uint sTgt[${const int: bpg * block_stride}];                         \
    /* Minimum cost of all candidates for each block. */                        \
    shared uint best_cost[${const int: bpg}];                                   \
    /* Deterministic tie breaking rank. */                                      \
    shared uint best_rank[${const int: bpg}];                                   \
    /* Displacement into the target image that each block's search window will  \
     * be centered upon or seeded with. */                                      \
    shared ivec2 s_seed[${const int: bpg}];                                     \
    /* Avoid using any shared memory if there is only one seed candidate. */    \
    @if (nseed > 1) {                                                           \
        /* SSD of each seed candidate of each block of the tile. */             \
        shared uint s_score[${const int: bpg * nseed}];                         \
    @}                                                                          \
                                                                                \
    uint luma8b(float x) {                                                      \
        return uint(round(clamp(x, 0.0, 1.0) * 255.0));                         \
    }                                                                           \
                                                                                \
    @if (has_parent) {                                                          \
        /* Use the winning motion search candidate from the previous motion     \
         * search pyramid level. We assume that the resolution of the image     \
         * pyramid could be perfectly halved at each level, and that block      \
         * (x,y) in the parent level corresponds to blocks (i,j) in the current \
         * level for i in {2*x, 2*x+1}, j in {2*y, 2*y+1}. */                   \
        ivec2 parent_seed(ivec2 blk) {                                          \
            ivec2 p = clamp(blk / 2, ivec2(0),                                  \
                            ivec2(${const int: parent_w},                       \
                                  ${const int: parent_h}) - 1);                 \
            /* Multiply by two to account for change in scale between pyramid   \
             * levels. */                                                       \
            return 2 * ivec2(texelFetch($parent, p, 0).xy);                     \
        }                                                                       \
    @}                                                                          \
                                                                                \
    @if (has_prior) {                                                           \
        /* Use the winning motion search candidate that was found by the        \
         * corresponding block in the same motion search pyramid level in the   \
         * invocation of the motion estimation on the previous frame pair. */   \
        ivec2 prior_seed(ivec2 blk) {                                           \
            ivec2 p = clamp(blk, ivec2(0),                                      \
                            ivec2(${const int: prior_w},                        \
                                  ${const int: prior_h}) - 1);                  \
            return ivec2(texelFetch($prior, p, 0).xy);                          \
        }                                                                       \
    @}                                                                          \
                                                                                \
    /* Returns the seed candidate that a thread `i` is responsible for          \
     * scoring. */                                                              \
    ivec2 seed_candidate(ivec2 blk, int i) {                                    \
        @if (has_parent) {                                                      \
            if (i == ${const int: rank_parent})                                 \
                return parent_seed(blk);                                        \
            if (i == ${const int: rank_left})                                   \
                return parent_seed(blk + ivec2(-1, 0));                         \
            if (i == ${const int: rank_right})                                  \
                return parent_seed(blk + ivec2(1, 0));                          \
            if (i == ${const int: rank_up})                                     \
                return parent_seed(blk + ivec2(0, -1));                         \
            if (i == ${const int: rank_down})                                   \
                return parent_seed(blk + ivec2(0, 1));                          \
        @}                                                                      \
        @if (has_prior) {                                                       \
            if (i == ${const int: rank_prior})                                  \
                return prior_seed(blk);                                         \
        @}                                                                      \
        return ivec2(0);                                                        \
    }                                                                           \
                                                                                \
    @if (nseed > 1) {                                                           \
        /* Sum of squared differences between reference block `b` and the       \
         * corresponding block in the target frame, but displaced by `d`. */    \
        uint seed_score(int b, ivec2 blk, ivec2 d) {                            \
            ivec2 size = ivec2(${const int: tex_w}, ${const int: tex_h});       \
            int bx = b % ${const int: bpgx};                                    \
            int by = b / ${const int: bpgx};                                    \
            uint ssd = 0u;                                                      \
            for (int i = 0; i < ${const int: bs}; i++) {                        \
                int ref_row = (by * ${const int: bs} + i) * ${const int: ref_w} \
                            + bx * ${const int: bs};                            \
                for (int j = 0; j < ${const int: bs}; j++) {                    \
                    ivec2 pos = clamp(blk * ${const int: bs} + ivec2(j, i) + d, \
                                      ivec2(0), size - 1);                      \
                    int diff = int(sRef[ref_row + j])                           \
                             - int(luma8b(texelFetch($tgt, pos, 0).r));         \
                    ssd += uint(diff * diff);                                   \
                }                                                               \
            }                                                                   \
            return ssd;                                                         \
        }                                                                       \
    @}
    // clang-format on

    // clang-format off
#pragma GLSL /* motion_search_pass */                                           \
    uint tid = gl_LocalInvocationIndex;                                         \
    int b  = int(tid) / ${const int: tpb};  /* block within the tile */         \
    int c0 = int(tid) % ${const int: tpb};  /* first candidate of thread */     \
    /* The block of pixels within the window this thread will search. */        \
    int bx = b % ${const int: bpgx};                                            \
    int by = b / ${const int: bpgx};                                            \
    ivec2 tile = ivec2(gl_WorkGroupID.xy) * ivec2(${const int: bpgx},           \
                                                  ${const int: bpgy});          \
    /* The index of this thread's block of pixels within the grid of blocks. */ \
    ivec2 blk  = tile + ivec2(bx, by);                                          \
    ivec2 size = ivec2(${const int: tex_w}, ${const int: tex_h});               \
                                                                                \
    if (tid < ${const uint: bpg}) {                                             \
        best_cost[tid] = 0xFFFFFFFFu;                                           \
        best_rank[tid] = 0xFFFFFFFFu;                                           \
    }                                                                           \
                                                                                \
    /* Load block from reference image into shared memory. We clamp coordinates \
     * and pad partial blocks that extend past the image boundaries. */         \
    for (uint i = tid; i < ${const uint: ref_w * ref_h};                        \
         i += ${const uint: threads})                                           \
    {                                                                           \
        ivec2 elem = ivec2(int(i) % ${const int: ref_w},                        \
                           int(i) / ${const int: ref_w});                       \
        ivec2 ref_pos = clamp(tile * ${const int: bs} + elem, ivec2(0),         \
                              size - 1);                                        \
        sRef[i] = luma8b(texelFetch($ref, ref_pos, 0).r);                       \
    }                                                                           \
                                                                                \
    /* Select the displacement that block `b`s search window will be centered   \
     * on, saved to shared memory for the rest of the block to use. */          \
    ivec2 seed = ivec2(0);                                                      \
    @if (nseed > 1) {                                                           \
        barrier();  /* reference tile must be available in shared memory */     \
                                                                                \
        for (int i = c0; i < ${const int: nseed};                               \
             i += ${const int: tpb})                                            \
        {                                                                       \
            s_score[b * ${const int: nseed} + i] =                              \
                seed_score(b, blk, seed_candidate(blk, i));                     \
        }                                                                       \
        barrier();                                                              \
                                                                                \
        /* Pick the seed candidate with the lowest SSD. */                      \
        if (c0 == 0) {                                                          \
            int best_seed = 0;                                                  \
            uint best_ssd = s_score[b * ${const int: nseed}];                   \
            for (int i = 1; i < ${const int: nseed}; i++) {                     \
                uint seed_ssd = s_score[b * ${const int: nseed} + i];           \
                if (seed_ssd < best_ssd) {                                      \
                    best_seed = i;                                              \
                    best_ssd = seed_ssd;                                        \
                }                                                               \
            }                                                                   \
            seed = seed_candidate(blk, best_seed);                              \
            s_seed[b] = seed;                                                   \
        }                                                                       \
    @} else {                                                                   \
        if (c0 == 0) {                                                          \
            seed = seed_candidate(blk, 0);                                      \
            s_seed[b] = seed;                                                   \
        }                                                                       \
    @}                                                                          \
    barrier();                                                                  \
                                                                                \
    /* Load block from target image into shared memory. Each block can have a   \
     * unique search window based on `s_seed`, so account for that. Pad rows    \
     * of the search window in shared memory to avoid bank conflicts. */        \
    for (uint i = tid; i < ${const uint: bpg * win * win_stride};               \
         i += ${const uint: threads})                                           \
    {                                                                           \
        int win_idx = int(i) / ${const int: win * win_stride};                  \
        int j       = int(i) % ${const int: win * win_stride};                  \
        int win_x   = j % ${const int: win_stride};                             \
        int win_y   = j / ${const int: win_stride};                             \
        if (win_x < ${const int: win}) {                                        \
            ivec2 win_blk = tile + ivec2(win_idx % ${const int: bpgx},          \
                                         win_idx / ${const int: bpgx});         \
            ivec2 origin = win_blk * ${const int: bs} + s_seed[win_idx]         \
                         - ${const int: r};                                     \
            ivec2 tgt_pos = clamp(origin + ivec2(win_x, win_y), ivec2(0),       \
                                  size - 1);                                    \
            sTgt[win_idx * ${const int: block_stride}                           \
                 + win_y * ${const int: win_stride} + win_x]                    \
                = luma8b(texelFetch($tgt, tgt_pos, 0).r);                       \
        }                                                                       \
    }                                                                           \
    barrier();  /* make sure all elements are loaded into shared memory */      \
                                                                                \
    /* Compute the sum of squared differences (SSD) of all candidates of each   \
     * block. One thread loads all elements of a given candidate. */            \
    uint best = 0xFFFFFFFFu, rank = 0xFFFFFFFFu;                                \
    for (int c = c0; c < ${const int: ncand}; c += ${const int: tpb}) {         \
        ivec2 cand_offset = $cand(c);                                           \
        /* Change the candidate offset into coordinates within the window */    \
        cand_offset += ${const int: r};                                         \
        uint ssd = 0u;                                                          \
        for (int i = 0; i < ${const int: bs}; i++) {                            \
            int ref_row = (by * ${const int: bs} + i) * ${const int: ref_w}     \
                        + bx * ${const int: bs};                                \
            int tgt_row = b * ${const int: block_stride}                        \
                        + (cand_offset.y + i) * ${const int: win_stride}        \
                        + cand_offset.x;                                        \
            for (int j = 0; j < ${const int: bs}; j++) {                        \
                int diff = int(sRef[ref_row + j]) - int(sTgt[tgt_row + j]);     \
                ssd += uint(diff * diff);                                       \
            }                                                                   \
        }                                                                       \
        /* For when a thread computes SSD of multiple candidates. */            \
        /* Strictly lower, so the order of candidates breaks ties. */           \
        if (ssd < best) {                                                       \
            best = ssd;                                                         \
            rank = uint(c);                                                     \
        }                                                                       \
    }                                                                           \
                                                                                \
    /* Reduce over SSD cost. */                                                 \
    atomicMin(best_cost[b], best);                                              \
    barrier();                                                                  \
    /* Reduce over rank for deterministic minimum */                            \
    if (best == best_cost[b])                                                   \
        atomicMin(best_rank[b], rank);                                          \
    barrier();                                                                  \
    if (c0 == 0 && blk.x < ${const int: grid_w} &&                              \
        blk.y < ${const int: grid_h})                                           \
        imageStore($out, blk,                                                   \
                   vec4(vec2(seed + $cand(int(best_rank[b]))), 1.0, 0.0));
    // clang-format on

    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader        = &sh,
        .dispatch_size = {
            PL_DIV_UP(grid_w, bpgx),
            PL_DIV_UP(grid_h, bpgy),
            1,
        },
    ));

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// Replace all block's motion vectors with the component-wise median of the
// surrounding 3x3 neighbourhood of blocks.
static bool motion_median_pass(pl_gpu gpu, pl_dispatch dispatch, pl_tex in,
                               pl_tex out)
{
    const int grid_w = out->params.w, grid_h = out->params.h;

    int bw = MOTION_MEDIAN_WORKGROUP_DIM;
    int bh = MOTION_MEDIAN_WORKGROUP_DIM;

    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "motion vector median filter (3x3, %dx%d blocks)", grid_w,
                 grid_h);
    sh->output = PL_SHADER_SIG_NONE;

    if (!sh_try_compute(sh, bw, bh, false, 0)) {
        PL_ERR(gpu,
               "pl_motion_estimate: cannot create motion vector median shader, "
               "needs %dx%d threads per work group!",
               bw, bh);
        goto error;
    }

    ident_t src = sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name = "flow_in",
            .type = PL_DESC_SAMPLED_TEX,
        },
        .binding = {
            .object      = in,
            .sample_mode = PL_TEX_SAMPLE_NEAREST,
        },
    });

    ident_t dst = sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name   = "flow_out",
            .type   = PL_DESC_STORAGE_IMG,
            .access = PL_DESC_ACCESS_WRITEONLY,
        },
        .binding.object = out,
    });

    // clang-format off
#pragma GLSLH /* motion_median_pass */                                          \
    /* Branchless, element-wise comparator to use in the sorting network. */    \
    void sort2(inout vec2 a, inout vec2 b) {                                    \
        vec2 lo = min(a, b);                                                    \
        b = max(a, b);                                                          \
        a = lo;                                                                 \
    }                                                                           \
                                                                                \
    /* Sort 9 elements residing in registers using a sorting network. Sorting   \
     * network from https://bertdobbelaere.github.io/sorting_networks.html or   \
     * SorterHunter. */                                                         \
    void sort9(inout vec2 a, inout vec2 b, inout vec2 c, inout vec2 d,          \
               inout vec2 e, inout vec2 f, inout vec2 g, inout vec2 h,          \
               inout vec2 i) {                                                  \
        sort2(c,f); sort2(a,d); sort2(e,i); sort2(b,h);                         \
        sort2(a,h); sort2(c,e); sort2(f,g); sort2(d,i);                         \
        sort2(a,c); sort2(b,d); sort2(e,f); sort2(h,i);                         \
        sort2(b,e); sort2(f,h); sort2(d,g);                                     \
        sort2(c,e); sort2(a,b); sort2(d,f); sort2(g,i);                         \
        sort2(c,d); sort2(e,f); sort2(g,h);                                     \
        sort2(b,c); sort2(d,e); sort2(f,g);                                     \
    }                                                                           \
    // clang-format on

    // clang-format off
#pragma GLSL /* motion_median_pass */                                           \
    ivec2 grid = ivec2(${const int: grid_w}, ${const int: grid_h});             \
    ivec2 blk  = ivec2(gl_GlobalInvocationID.xy);                               \
    if (all(lessThan(blk, grid))) {                                             \
        /* The 3x3 neighbourhood around this block, clamped. */                 \
        vec2 mv[9];                                                             \
        for (int i = 0; i < 9; i++) {                                           \
            ivec2 p = clamp(blk + ivec2(i % 3, i / 3) - 1, ivec2(0), grid - 1); \
            mv[i] = texelFetch($src, p, 0).xy;                                  \
        }                                                                       \
                                                                                \
        sort9(mv[0], mv[1], mv[2], mv[3], mv[4], mv[5], mv[6], mv[7], mv[8]);   \
                                                                                \
        /* Take median of sorted elements. The confidence is always 1. */       \
        imageStore($dst, blk, vec4(mv[4], 1.0, 0.0));                           \
    }
    // clang-format on

    return pl_dispatch_compute(
        dispatch, pl_dispatch_compute_params(.shader = &sh,
                                             .dispatch_size = {
                                                 PL_DIV_UP(grid_w, bw),
                                                 PL_DIV_UP(grid_h, bh),
                                                 1,
                                             }, ));

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// Two contiguous histograms describing the global motion in the frame.
// One histogram for each component of the motion vector (x,y).
static const struct pl_buffer_var global_hist_var = {
    .var = {
        .name  = "global_hist",
        .type  = PL_VAR_UINT,
        .dim_v = 1,
        .dim_m = 1,
        .dim_a = 2 * MOTION_GLOBAL_BINS,
    },
    .layout = {
        .offset = 0,
        .size   = 2 * MOTION_GLOBAL_BINS * sizeof(uint32_t),
        .stride = sizeof(uint32_t),
    },
};

// Reusable helper to recreate global motion and histogram scratch buffers.
static bool global_buf_recreate(pl_gpu gpu, pl_buf *buf, size_t size,
                                bool host_readable)
{
    if (*buf)
        return true;

    // .initial_data doesn't care if it is larger than .size
    static const uint32_t zero[2 * MOTION_GLOBAL_BINS] = { 0 };
    pl_assert(size <= sizeof(zero));
    *buf = pl_buf_create(
        gpu, pl_buf_params(.size = size, .storable = true,
                           .host_readable = host_readable,
                           .memory_type = host_readable ? PL_BUF_MEM_AUTO
                                                        : PL_BUF_MEM_DEVICE,
                           .initial_data = zero, .debug_tag = PL_DEBUG_TAG, ));

    if (!*buf) {
        PL_ERR(gpu,
               "pl_motion_estimate: failed creating %zu byte global "
               "motion buffer!",
               size);
        return false;
    }

    return true;
}

// Create a histogram of motion from the blocks of `flow` into `hist`.
// NOTE: if the median motion is outside of [-`MOTION_GLOBAL_BINS/2`,
// `MOTION_GLOBAL_BINS-1`], the global motion is clamped.
static bool motion_global_hist_pass(pl_gpu gpu, pl_dispatch dispatch,
                                    pl_tex flow, pl_buf hist)
{
    const int grid_w = flow->params.w, grid_h = flow->params.h;
    const int dim = MOTION_GLOBAL_WORKGROUP_DIM;
    const size_t shmem = 2 * MOTION_GLOBAL_BINS * sizeof(uint32_t);

    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "global motion histogram (%dx%d blocks)", grid_w, grid_h);
    sh->output = PL_SHADER_SIG_NONE;

    if (!sh_try_compute(sh, dim, dim, false, shmem)) {
        PL_ERR(gpu,
               "pl_motion_estimate: cannot create global motion histogram "
               "shader, needs %dx%d threads and %zu bytes of shared memory "
               "per work group!",
               dim, dim, shmem);
        goto error;
    }

    ident_t src = sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name = "flow_in",
            .type = PL_DESC_SAMPLED_TEX,
        },
        .binding = {
            .object      = flow,
            .sample_mode = PL_TEX_SAMPLE_NEAREST,
        },
    });

    sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name   = "global_hist_buf",
            .type   = PL_DESC_BUF_STORAGE,
            .access = PL_DESC_ACCESS_READWRITE,
        },
        .binding.object  = hist,
        .buffer_vars     = (struct pl_buffer_var *) &global_hist_var,
        .num_buffer_vars = 1,
    });

    // clang-format off
#pragma GLSLH /* motion_global_hist_pass */                                     \
    shared uint s_hist[2][${const int: MOTION_GLOBAL_BINS}];

#pragma GLSL /* motion_global_hist_pass */                                      \
    /* One thread per bin. */                                                   \
    int tid = int(gl_LocalInvocationIndex);                                     \
    s_hist[0][tid] = s_hist[1][tid] = 0u; /* zero shared memory */              \
    barrier();                                                                  \
                                                                                \
    ivec2 grid = ivec2(${const int: grid_w}, ${const int: grid_h});             \
    ivec2 blk = ivec2(gl_GlobalInvocationID.xy);                                \
    if (all(lessThan(blk, grid))) {                                             \
        vec2 mv = texelFetch($src, blk, 0).xy;                                  \
        /* Shift mv in [-128, 128] to bin index in [0, 256). */                 \
        ivec2 bin = ivec2(round(mv)) + ${const int: MOTION_GLOBAL_BINS / 2};    \
        bin = clamp(bin, ivec2(0), ivec2(${const int: MOTION_GLOBAL_BINS - 1}));\
        atomicAdd(s_hist[0][bin.x], 1u);                                        \
        atomicAdd(s_hist[1][bin.y], 1u);                                        \
    }                                                                           \
    barrier();                                                                  \
                                                                                \
    for (int c = 0; c < 2; c++) {                                               \
        if (s_hist[c][tid] != 0u) /* prefer branching over unnecessary atomic */\
            atomicAdd(global_hist[c * ${const int: MOTION_GLOBAL_BINS} + tid],  \
                      s_hist[c][tid]);                                          \
    }
    // clang-format on

    return pl_dispatch_compute(
        dispatch, pl_dispatch_compute_params(.shader = &sh,
                                             .dispatch_size = {
                                                 PL_DIV_UP(grid_w, dim),
                                                 PL_DIV_UP(grid_h, dim),
                                                 1,
                                             }, ));

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// Reads the median from the histogram. Needs to be in a separate shader because
// the shader that builds the histogram uses many blocks.
// TODO: estimate a confidence. It is always 1 for now. One approach would be
// using the proportion of bins in the histogram that are close to the median.
static bool motion_global_median_pass(pl_gpu gpu, pl_dispatch dispatch,
                                      pl_buf hist, int num, pl_buf out,
                                      enum pl_motion_direction dir)
{
    const size_t shmem = 2 * MOTION_GLOBAL_BINS * sizeof(uint32_t);

    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "global motion median (%d blocks)", num);
    sh->output = PL_SHADER_SIG_NONE;

    if (!sh_try_compute(sh, MOTION_GLOBAL_BINS, 1, false, shmem)) {
        PL_ERR(gpu,
               "pl_motion_estimate: cannot create global motion median "
               "shader, needs %d threads and %zu bytes of shared memory per "
               "work group!",
               MOTION_GLOBAL_BINS, shmem);
        goto error;
    }

    sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name   = "global_hist_buf",
            .type   = PL_DESC_BUF_STORAGE,
            .access = PL_DESC_ACCESS_READWRITE,
        },
        .binding.object  = hist,
        .buffer_vars     = (struct pl_buffer_var *) &global_hist_var,
        .num_buffer_vars = 1,
    });

    sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name   = "global_motion_out",
            .type   = PL_DESC_BUF_STORAGE,
            .access = PL_DESC_ACCESS_WRITEONLY,
        },
        .binding.object  = out,
        .buffer_vars =
            (struct pl_buffer_var *) pl_motion_global_buf_vars[dir],
        .num_buffer_vars = PL_MOTION_GLOBAL_BUF_NUM_VARS,
    });

    // clang-format off
#pragma GLSLH /* motion_global_median_pass */                                   \
    shared uint s_hist[2][${const int: MOTION_GLOBAL_BINS}];

#pragma GLSL /* motion_global_median_pass */                                    \
    /* One thread per bin for loading and clearing the histogram. */            \
    int tid = int(gl_LocalInvocationIndex);                                     \
    for (int c = 0; c < 2; c++) {                                               \
        int g = c * ${const int: MOTION_GLOBAL_BINS} + tid;                     \
        s_hist[c][tid] = global_hist[g];                                        \
        global_hist[g] = 0u;                                                    \
    }                                                                           \
    barrier();                                                                  \
                                                                                \
    /* Single-threaded prefix sum to find the median. */                        \
    if (tid < 2) {                                                              \
        uint sum = 0u;                                                          \
        int i = 0;                                                              \
        for (; i < ${const int: MOTION_GLOBAL_BINS}; i++) {                     \
            sum += s_hist[tid][i];                                              \
            if (sum > ${const uint: num / 2})                                   \
                break;                                                          \
        }                                                                       \
        /* Shift bin index in [0, 256) to mv in [-128, 128). */                 \
        float mv = float(i - ${const int: MOTION_GLOBAL_BINS / 2});             \
        @if (dir == PL_MOTION_FORWARD) {                                        \
            global_mv_fwd[tid] = mv;                                            \
            if (tid == 0)                                                       \
                global_confidence_fwd = 1.0;                                    \
        @} else {                                                               \
            global_mv_bwd[tid] = mv;                                            \
            if (tid == 0)                                                       \
                global_confidence_bwd = 1.0;                                    \
        @}                                                                      \
    }
    // clang-format on

    return pl_dispatch_compute(
        dispatch, pl_dispatch_compute_params(.shader = &sh,
                                             .dispatch_size = { 1, 1, 1 }, ));

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// Estimates the global motion of `dir` from its flow field.
static bool estimate_global(pl_gpu gpu, pl_dispatch dispatch,
                            struct sh_motion_obj *obj,
                            enum pl_motion_direction dir)
{
    const bool fwd = dir == PL_MOTION_FORWARD;
    pl_tex flow = fwd ? obj->flow.fwd.elem[0] : obj->flow.bwd.elem[0];
    pl_buf *global = fwd ? &obj->global.fwd : &obj->global.bwd;

    return global_buf_recreate(gpu, &obj->global.hist,
                               2 * MOTION_GLOBAL_BINS * sizeof(uint32_t),
                               false) &&
           global_buf_recreate(gpu, global, sizeof(struct global_buf_data),
                               true) &&
           motion_global_hist_pass(gpu, dispatch, flow, obj->global.hist) &&
           motion_global_median_pass(gpu, dispatch, obj->global.hist,
                                     flow->params.w * flow->params.h, *global,
                                     dir);
}

bool pl_motion_estimate(const struct pl_motion_estimate_input *input,
                        const struct pl_motion_estimate_params *params)
{
    if (!input || !input->gpu || !input->dispatch || !input->state)
        return false;

    params = PL_DEF(params, &pl_motion_estimate_default_params);

    pl_gpu gpu = input->gpu;
    struct sh_motion_obj *obj =
        sh_alloc_obj(gpu, input->state, PL_SHADER_OBJ_MOTION,
                     sizeof(struct sh_motion_obj), motion_uninit);
    if (!obj)
        return false;

    if (!validate(gpu, input, params)) {
        invalidate_temporal(gpu, obj, "invalid input");
        return false;
    }

    const int w = input->a.tex->params.w;
    const int h = input->a.tex->params.h;
    const char *changed = NULL;
    if (w != obj->w || h != obj->h) {
        changed = "input dimensions changed";
    } else if (!params_eq(&obj->params, params)) {
        changed = "parameters changed";
    }

    if (changed) {
        invalidate_temporal(gpu, obj, changed);
        obj->params = *params;
        obj->w = w;
        obj->h = h;
    }

    if (obj->sig_a == input->a.sig && obj->sig_b == input->b.sig &&
        (obj->flow.valid & input->direction) == input->direction)
    {
        // Same frame pair as the previous call, at most the global motion is
        // left to compute.
        if (!input->estimate_global || obj->global.valid)
            return true;

        for (int i = 0; i < 2; i++) {
            const enum pl_motion_direction dir = 1 << i;
            if ((obj->flow.valid & dir) &&
                !estimate_global(gpu, input->dispatch, obj, dir))
                goto error;
        }

        obj->global.valid = true;
        return true;
    }

    if (obj->sig_b != input->a.sig) {
        invalidate_temporal(
            gpu, obj,
            "frame signature of first frame not equal to frame signature "
            "of second frame from previous call");
    }

    obj->sig_a = input->a.sig;
    obj->sig_b = input->b.sig;

    const int levels = pyramid_num_levels(w, h, params);
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 16, 0,
                             PL_FMT_CAP_LINEAR | PL_FMT_CAP_RENDERABLE);
    if (!fmt) {
        PL_ERR(gpu, "pl_motion_estimate: no renderable, linearly filterable "
                    "single component float format available!");
        goto error;
    }

    // The search covers the trailing partial blocks as well, hence ceiling
    // division.
    obj->grid_w = PL_DIV_UP(w, params->block_size);
    obj->grid_h = PL_DIV_UP(h, params->block_size);

    pl_fmt mv_fmt = flow_fmt(gpu);
    if (!mv_fmt) {
        PL_ERR(gpu, "pl_motion_estimate: no storable, linearly filterable "
                    "two component float format available!");
        goto error;
    }

    // These are no-ops as long as the motion search shape stays the same.
    // Otherwise, they invalidate the no longer usable temporal prior that resides
    // in the `obj->flow.fwd` and `obj->flow.bwd`.
    if (input->direction & PL_MOTION_FORWARD) {
        if (!flow_pyramid_recreate(gpu, obj, &obj->flow.fwd, w, h, levels,
                                   params->block_size, mv_fmt))
            goto error;
    }

    if (input->direction & PL_MOTION_BACKWARD) {
        if (!flow_pyramid_recreate(gpu, obj, &obj->flow.bwd, w, h, levels,
                                   params->block_size, mv_fmt))
            goto error;
    }

    if (!flow_pyramid_recreate(gpu, obj, &obj->scratch.raw, w, h, levels,
                               params->block_size, mv_fmt))
        goto error;

    if (!flow_pyramid_recreate(gpu, obj, &obj->scratch.filtered, w, h, levels,
                               params->block_size, mv_fmt))
        goto error;

    // Upon cold start or after reset, build the image pyramid for frame `a`,
    // use cached pyramid otherwise.
    if (!obj->pyramid_valid) {
        if (!pyramid_recreate(gpu, obj, &obj->pyramid_prev, w, h, levels, fmt))
            goto error;
        if (!convert_to_luma(input->dispatch, &input->a,
                             obj->pyramid_prev.elem[0]))
        {
            PL_ERR(gpu, "pl_motion_estimate: failed converting frame a to "
                        "luma!");
            goto error;
        }
        if (!pyramid_build(gpu, input->dispatch, &obj->pyramid_prev))
            goto error;
    }

    if (!pyramid_recreate(gpu, obj, &obj->scratch.levels, w, h, levels, fmt))
        goto error;
    if (!convert_to_luma(input->dispatch, &input->b,
                         obj->scratch.levels.elem[0]))
    {
        PL_ERR(gpu, "pl_motion_estimate: failed converting frame b to luma!");
        goto error;
    }
    if (!pyramid_build(gpu, input->dispatch, &obj->scratch.levels))
        goto error;

    static const enum pl_motion_direction dirs[] = {
        PL_MOTION_FORWARD,
        PL_MOTION_BACKWARD,
    };

    void *tmp = pl_tmp(NULL);
    bool ok = true;
    for (int d = 0; d < PL_ARRAY_SIZE(dirs); d++) {
        if (!(input->direction & dirs[d]))
            continue;

        const bool fwd = dirs[d] == PL_MOTION_FORWARD;
        tex_pyramid_t *field = fwd ? &obj->flow.fwd : &obj->flow.bwd;

        const bool has_prior = obj->flow.valid & dirs[d];

        for (int i = levels - 1; i >= 0; i--) {
            const bool coarsest = levels > 1 && i == levels - 1;
            const int radius = coarsest ? params->search_radius_coarse
                                        : params->search_radius_fine;

            pl_tex raw = obj->scratch.raw.elem[i];
            pl_tex out = obj->scratch.filtered.elem[i];

            pl_tex a = obj->pyramid_prev.elem[i];
            pl_tex b = obj->scratch.levels.elem[i];

            // Use the parent motion search to seed this levels search, for
            // all levels but the coarsest. The parent is the motion field
            // after median filtering is ran.
            pl_tex parent =
                i < levels - 1 ? obj->scratch.filtered.elem[i + 1] : NULL;

            // Seed the motion search using the motion estimate at the
            // corresponding level of the motion search pyramid from the
            // previous frame pair that the motion estimation was invoked
            // upon.
            pl_tex prior = has_prior ? field->elem[i] : NULL;

            struct motion_geometry geom;
            ok = motion_plan(gpu, tmp, params->block_size, radius,
                             motion_seeds_plan(parent != NULL, prior != NULL),
                             out->params.w, out->params.h, &geom);
            if (!ok)
                break;

            ok = motion_search_pass(gpu, input->dispatch,
                                    &(struct motion_search) {
                                        .ref = fwd ? a : b,
                                        .tgt = fwd ? b : a,
                                        .parent = parent,
                                        .prior = prior,
                                        .out_mv = raw,
                                        .geom = &geom,
                                        .cand_lut = &obj->cand_lut[coarsest],
                                    });
            if (!ok)
                break;

            ok = motion_median_pass(gpu, input->dispatch, raw, out);
            if (!ok)
                break;
        }

        if (!ok)
            break;

        // NOTE: we assume that the prior forward motion field is safe to be
        // overwritten by the backward motion search.
        //
        // NOTE: motion estimation that succeeds on the forward motion
        // estimation call but fails on the backward motion estimation call
        // may leave sh_motion_obj in an incomplete state. we assume that if
        // the backward direction fails, the caller is not interested in the
        // forward direction results.
        PL_SWAP(*field, obj->scratch.filtered);

        if (input->estimate_global) {
            ok = estimate_global(gpu, input->dispatch, obj, dirs[d]);
            if (!ok)
                break;
        }
    }
    pl_free(tmp);

    if (!ok)
        goto error;

    obj->flow.valid = input->direction;
    obj->global.valid = input->estimate_global;

    // Cache frame `b`'s luma pyramid, use reuse `a` buffers as scratch.
    PL_SWAP(obj->pyramid_prev, obj->scratch.levels);
    obj->pyramid_valid = true;

    return true;

error:
    invalidate_temporal(gpu, obj, "estimation failed");
    return false;
}
