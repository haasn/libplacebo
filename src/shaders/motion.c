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
#include <stdlib.h>
#include <string.h>

#include "shaders.h"
#include "shaders/motion.h"
#include "shaders/pyramid.h"

#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/motion.h>
#include <libplacebo/shaders/sampling.h>

const struct pl_motion_estimate_params pl_motion_estimate_default_params = {
    PL_MOTION_ESTIMATE_DEFAULTS
};

const struct pl_frame_interp_params pl_frame_interp_default_params = {
    PL_FRAME_INTERP_DEFAULTS
};

enum {
    FWD = 0,
    BWD = 1,
    NUM_DIRS = 2,
};

static int dir_index(enum pl_motion_direction dir)
{
    switch (dir) {
    case PL_MOTION_FORWARD:  return FWD;
    case PL_MOTION_BACKWARD: return BWD;
    default:                 return -1;
    }
}

// Persistent state and scratch space behind a PL_SHADER_OBJ_MOTION shader object.
struct sh_motion_obj {
    // --- Current motion estimation state metadata
    struct pl_motion_estimate_params params;

    // Current frame pair signatures and dimensions
    uint64_t sig_prev, sig_next;
    int frame_w, frame_h;

    // Directions and features the state is set up for
    enum pl_motion_direction direction;
    enum pl_motion_feature features;

    // Which directions of `dir` are currently valid
    enum pl_motion_direction valid;

    // --- Outputs for the current frame pair, in each direction
    struct motion_dir {
        pl_tex_pyramid coarse; // coarse flow field (blocks), after smoothing
        pl_tex_pyramid fine;   // fine flow field (pixels), or {0}
        pl_tex gain;           // gain (1x1 texture)
        pl_tex quality;        // flow quality estimate, or NULL
        pl_buf global_flow;    // global flow buffer, or NULL
    } dir[NUM_DIRS];

    // --- Temporary buffers used only within a motion estimation call
    struct {
        pl_tex_pyramid input; // Image pyramid of the input features
        pl_tex_pyramid raw;   // Raw motion vectors (before smoothing)
        pl_buf hist;          // Histogram for global motion estimation
    } scratch;

    // --- Other cached / persistent GPU objects

    // LUTs for the motion search candidate ordering
    pl_shader_obj cand_lut[2];

    // Luma pyramid of the *previous* call's frame `b`, reused as this call's
    // frame `a` pyramid. Only holds a usable pyramid if `pyramid_valid`.
    pl_tex_pyramid pyramid_prev;
    bool pyramid_valid;
};

// Binds `tex` for use with `texelFetch`.
static ident_t bind_tex(pl_shader sh, const char *name, pl_tex tex)
{
    return sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name = name,
            .type = PL_DESC_SAMPLED_TEX,
        },
        .binding = {
            .object      = tex,
            .sample_mode = PL_TEX_SAMPLE_NEAREST,
        },
    });
}

static ident_t bind_tex_linear(pl_shader sh, const char *name, pl_tex tex,
                               ident_t *out_pos, ident_t *out_pt)
{
    return sh_bind(sh, tex, PL_TEX_ADDRESS_CLAMP, PL_TEX_SAMPLE_LINEAR, name,
                   NULL, out_pos, out_pt);
}

// Binds `tex` as a write-only storage image.
static ident_t bind_img(pl_shader sh, const char *name, pl_tex tex)
{
    return sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name   = name,
            .type   = PL_DESC_STORAGE_IMG,
            .access = PL_DESC_ACCESS_WRITEONLY,
        },
        .binding.object = tex,
    });
}

// Binds `buf` as a storage buffer holding `vars`.
static void bind_buf(pl_shader sh, const char *name, pl_buf buf,
                     enum pl_desc_access access,
                     const struct pl_buffer_var *vars, int num_vars)
{
    sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name   = name,
            .type   = PL_DESC_BUF_STORAGE,
            .access = access,
        },
        .binding.object  = buf,
        .buffer_vars     = (struct pl_buffer_var *) vars,
        .num_buffer_vars = num_vars,
    });
}

// Begins a compute shader with work groups of `bw` x `bh` threads and `shmem`
// bytes of shared memory, described by `fmt`. Returns NULL and aborts the
// shader if the GPU cannot run it.
PL_PRINTF(6, 7)
static pl_shader pass_begin(pl_gpu gpu, pl_dispatch dispatch, int bw, int bh,
                            size_t shmem, const char *fmt, ...)
{
    pl_shader sh = pl_dispatch_begin(dispatch);
    sh->output = PL_SHADER_SIG_NONE;

    va_list ap;
    va_start(ap, fmt);
    sh_vdescribef(sh, fmt, ap);
    va_end(ap);

    if (!sh_try_compute(sh, bw, bh, false, shmem)) {
        pl_assert(sh->info->steps.num == 1);
        PL_ERR(gpu, "pl_motion_estimate: cannot run %s, it needs %dx%d threads "
                    "and %zu bytes of shared memory per work group!",
               sh->info->steps.elem[0], bw, bh, shmem);
        pl_dispatch_abort(dispatch, &sh);
        return NULL;
    }

    return sh;
}

// Partial sums of a compute shader in shared memory: `n` sums of the type
// `type` with the components `comps`, for each of `groups` groups of `parts`
// threads.
//
// Defines `void put(int group, int part, int i, T v)`, storing what thread
// `part` of a group contributes to its sum `i`, and
// `T total(int group, int i)`, adding up the contributions after a barrier.
//
// Needs `groups * parts * n` values of shared memory, which the caller
// must account for.
static void sh_partial_sums(pl_shader sh, enum pl_var_type type, uint8_t comps,
                            int groups, int parts, int n, ident_t *out_put,
                            ident_t *out_total)
{
    ident_t buf   = sh_fresh(sh, "partial_sums");
    ident_t put   = sh_fresh(sh, "put_sum");
    ident_t total = sh_fresh(sh, "total_sum");

    // clang-format off
#pragma GLSLH /* sh_partial_sums */                                             \
    @switch (type) {                                                            \
    @case PL_VAR_FLOAT:                                                         \
        #define T ${vecType: comps}                                             \
        @break;                                                                 \
    @case PL_VAR_SINT:                                                          \
        #define T ${ivecType: comps}                                            \
        @break;                                                                 \
    @default:                                                                   \
        #define T ${uvecType: comps}                                            \
        @break;                                                                 \
    @}                                                                          \
    shared T $buf[${const int: groups * parts * n}];                            \
    void $put(int group, int part, int i, T v) {                                \
        $buf[(group * ${const int: n} + i) * ${const int: parts} + part] = v;   \
    }                                                                           \
    T $total(int group, int i) {                                                \
        int base = (group * ${const int: n} + i) * ${const int: parts};         \
        T sum = $buf[base];                                                     \
        for (int k = 1; k < ${const int: parts}; k++)                           \
            sum += $buf[base + k];                                              \
        return sum;                                                             \
    }                                                                           \
    #undef T
    // clang-format on

    *out_put   = put;
    *out_total = total;
}

// Don't reduce image pyramid to a size that is too small to recover motion.
#define MIN_COARSE_BLOCKS 3

// "Block" will be used to refer to a block of pixels that we group together
// for the motion search.
// "Tile" will be used to refer to a set of blocks that one workgroup operates
// on.
// `MOTION_BPGX` and `MOTION_BPGY` define the tile dimensions. The threads of
// a tile wait for each other at every barrier, so small tiles idle less. 7
// blocks of the default 3x3 search are 63 threads, which is one wave on AMD.
#define MOTION_BPGX 1
#define MOTION_BPGY 7

// Relative change in brightness between two frames below which
// `motion_gain_pass` reports none. Compensating a change this small breaks
// more exact matches of unchanged content than it fixes.
#define MOTION_GAIN_DEADZONE 0.005f

// Maximum number of subblock refinement passes, see refine_plan().
#define MOTION_MAX_REFINE_PASSES 2

// Number of rings of far candidates of `motion_refine_4x4_pass`: the vectors of
// the blocks 2 and 4 blocks away along either axis. They bring back the
// motion of thin objects, which the smoothing removes from the block field
// around them.
#define MOTION_REFINE_4X4_RINGS 2

// Margin by which the vector of a neighbouring block must match a pixel better
// than the vector of its own block for `motion_refine_1x1_pass` to take it, as
// a mean squared error in 8 bit code values.
#define MOTION_REFINE_1X1_MARGIN 1.0f

// Side length, in pixels, of the squares that `motion_quality_pass` averages
// the quality of the vectors over.
#define MOTION_QUALITY_SUB 2

// Mismatch between the 3x3 window around a pixel and where its vector leads,
// relative to the texture of the window, from which the quality of the vector
// starts to fall, and at which it is gone.
#define MOTION_QUALITY_COST_LO 0.6f
#define MOTION_QUALITY_COST_HI 1.2f

// Texture added to that of the window, as a standard deviation in 8 bit code
// values: the level of noise below which a mismatch means nothing.
#define MOTION_QUALITY_NOISE 6.0f

// Distance, in pixels, of a vector from the median of the 3x3 vectors around
// it at which its quality falls to 1/e.
#define MOTION_QUALITY_OUTLIER 4.0f

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
    MOTION_SEED_PARENT_X, // the parent's horizontal neighbour nearest the block
    MOTION_SEED_PARENT_Y, // the parent's vertical neighbour nearest the block
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
        seeds.rank[MOTION_SEED_PARENT_X] = seeds.num++;
        seeds.rank[MOTION_SEED_PARENT_Y] = seeds.num++;
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
    bool subpixel;     // refine the vectors to a fraction of a pixel
    bool zero_mean;    // compare the blocks without their means

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
                            int nseed, bool subpixel, bool zero_mean)
{
    // The window of a sub-pixel search also covers the pixels around the
    // candidates, whose cost the fit needs.
    const int win = block_size + 2 * (radius + subpixel);
    const int win_stride = win | 1;
    const int block_stride = (win * win_stride) | 1;
    const int blocks = bpgx * bpgy;

    // When `nseed` == 1, that is the case when the only seed candidate is
    // the zero vector. We do not need to score any candidates in this case.
    // Otherwise, every thread of a block scores a part of each seed (at most
    // one per pixel).
    const int parts = PL_MIN(PL_SQUARE(2 * radius + 1), PL_SQUARE(block_size));
    // The zero-mean SSD also needs the sums of the differences.
    const size_t score = nseed > 1 ? (size_t) blocks * nseed * parts
                                     * (zero_mean ? 2 : 1) : 0;

    const size_t words =
        (size_t) blocks * PL_SQUARE(block_size) // reference tile
        + (size_t) blocks * block_stride        // search windows
        + 2 * (size_t) blocks                   // cost and rank
        + (subpixel ? 4 : 0) * (size_t) blocks  // cost around the winner
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
                        struct motion_seeds seeds, bool subpixel,
                        bool zero_mean, int grid_w, int grid_h,
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
            block_size, radius, bpgx, bpgy, seeds.num, subpixel, zero_mean);

        if (bpgx * bpgy * threads_block <= max_threads &&
            shmem.size <= gpu->glsl.max_shmem_size)
            break;

        if (bpgx >= bpgy)
            bpgx--;
        else
            bpgy--;
    }

    const struct motion_shmem_layout shmem = motion_compute_shmem_layout(
        block_size, radius, bpgx, bpgy, seeds.num, subpixel, zero_mean);

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
        .subpixel = subpixel,
        .zero_mean = zero_mean,
        .seeds = seeds,
        .cand = cand,
    };

    return true;
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
        return levels_auto(w, h, params->block_size_coarse);

    return PL_MIN(params->pyramid_levels, pl_pyramid_max_levels(w, h));
}

// Format for motion vector fields, holding `vec2(mv.x, mv.y)`. Prefer 16 bit
// floats but fall back to 32 bits.
static pl_fmt flow_fmt(pl_gpu gpu)
{
    static const enum pl_fmt_caps caps = PL_FMT_CAP_STORABLE |
                                         PL_FMT_CAP_LINEAR;

    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 2, 16, 0, caps);
    if (fmt)
        return fmt;

    return pl_find_fmt(gpu, PL_FMT_FLOAT, 2, 32, 0, caps);
}

static void motion_uninit(pl_gpu gpu, void *ptr)
{
    struct sh_motion_obj *obj = ptr;
    pl_pyramid_destroy(gpu, &obj->pyramid_prev);
    pl_pyramid_destroy(gpu, &obj->scratch.input);
    pl_pyramid_destroy(gpu, &obj->scratch.raw);
    for (int i = 0; i < NUM_DIRS; i++) {
        pl_pyramid_destroy(gpu, &obj->dir[i].coarse);
        pl_pyramid_destroy(gpu, &obj->dir[i].fine);
        pl_tex_destroy(gpu, &obj->dir[i].gain);
        pl_tex_destroy(gpu, &obj->dir[i].quality);
        pl_buf_destroy(gpu, &obj->dir[i].global_flow);
    }
    pl_buf_destroy(gpu, &obj->scratch.hist);
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

    obj->sig_prev = obj->sig_next = 0;
    obj->valid = 0;
    obj->pyramid_valid = false;
}

int pl_motion_num_blocks(int w, int h,
                         const struct pl_motion_estimate_params *params)
{
    const int bs = params->block_size_coarse;
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
    // `dir` cannot name `PL_MOTION_BOTH`, we can only return one flow field.
    const int d = dir_index(dir);
    if (!obj || d < 0 || !(obj->valid & dir))
        return NULL;

    const pl_tex_pyramid *fine = &obj->dir[d].fine;
    return fine->num ? fine->elem[0] : obj->dir[d].coarse.elem[0];
}

pl_tex pl_motion_get_quality(pl_shader_obj state, enum pl_motion_direction dir)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    const int d = dir_index(dir);
    if (!obj || d < 0 || !(obj->features & PL_MOTION_QUALITY) ||
        !(obj->valid & dir))
        return NULL;

    return obj->dir[d].quality;
}

ident_t sh_motion_flow(pl_shader sh, pl_shader_obj state,
                       enum pl_motion_direction dir)
{
    pl_tex flow = pl_motion_get_flow(state, dir);
    if (!flow)
        return NULL_IDENT;

    return bind_tex(sh, "motion_flow", flow);
}

pl_buf pl_motion_get_global_buf(pl_shader_obj state,
                                enum pl_motion_direction dir)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    // `dir` cannot name `PL_MOTION_BOTH`, we can only return one global
    // motion vector.
    const int d = dir_index(dir);
    if (!obj || d < 0 || !(obj->features & PL_MOTION_GLOBAL) ||
        !(obj->valid & dir))
        return NULL;

    return obj->dir[d].global_flow;
}

struct global_buf_data {
    float global_mv[2];
    float global_confidence;
};

// Names differ per direction, so that both can be bound to one shader
#define GLOBAL_BUF_NUM_VARS 2
static const struct pl_buffer_var
    global_buf_vars[PL_MOTION_BACKWARD + 1][GLOBAL_BUF_NUM_VARS] = {
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

    bind_buf(sh, "global_motion_desc", global, PL_DESC_ACCESS_READONLY,
             global_buf_vars[dir], GLOBAL_BUF_NUM_VARS);

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
    return a->block_size_coarse == b->block_size_coarse &&
           a->block_size_fine == b->block_size_fine &&
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

    if (params->block_size_coarse < 1) {
        PL_ERR(gpu, "pl_motion_estimate: block_size_coarse must be positive!");
        return false;
    }

    if (params->block_size_coarse > 64) {
        PL_ERR(gpu, "pl_motion_estimate: block_size_coarse must be <= 64!");
        return false;
    }

    if (params->block_size_coarse % 2) {
        PL_ERR(gpu, "pl_motion_estimate: block_size_coarse must be even!");
        return false;
    }

    if (params->block_size_fine < 1 ||
        params->block_size_fine > params->block_size_coarse)
    {
        PL_ERR(gpu, "pl_motion_estimate: block_size_fine must be in "
               "[1, block_size_coarse]!");
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

// Returns the (float) gain factor needed to compensate for brightness changes
// between the target and the reference frame.
static ident_t get_target_gain(pl_shader sh, pl_tex gain_tex)
{
    ident_t gain = sh_fresh(sh, "gain");
    if (gain_tex) {
        ident_t tex = bind_tex(sh, "gain_tex", gain_tex);
        GLSL("float "$" = texelFetch("$", ivec2(0), 0).x;", gain, tex);
    } else {
        GLSL("float "$" = 1.0;", gain);
    }

    return gain;
}

// Measures the change in brightness from `ref` to `tgt`, as on a fade, into
// the 1x1 texture `gain`: the factor that brings `tgt` to the brightness of
// `ref`. It is the ratio of the mean luma of the blocks of `ref` to that of
// the blocks in `tgt` that their vectors in `flow` lead to.
//
// Comparing the content that the vectors matched, instead of the means of the
// frames, keeps content that moves into or out of the frame from counting as
// a change in brightness. For the same reason, blocks whose vectors leave the
// frame are skipped, and the others are weighted by how well their textures
// match once their means are removed.
static bool motion_gain_pass(pl_gpu gpu, pl_dispatch dispatch, int bs,
                             pl_tex ref, pl_tex tgt, pl_tex flow, pl_tex gain)
{
    const int tex_w = ref->params.w, tex_h = ref->params.h;
    const int grid_w = flow->params.w, grid_h = flow->params.h;
    const int threads = 64;
    const float fbs = bs;

    // SSD of a block at which a block without any texture counts half: a
    // mean squared error of 4 code values (8 bit) per pixel.
    const float flat = 4.0f * bs * bs / (255.0f * 255.0f);

    pl_shader sh = pass_begin(gpu, dispatch, threads, 1,
                              threads * sizeof(float[2]),
                              "motion brightness gain (%dx%d blocks)", grid_w,
                              grid_h);
    if (!sh)
        return false;

    ident_t ref_id  = bind_tex(sh, "ref", ref);
    ident_t flow_id = bind_tex(sh, "flow", flow);
    ident_t gain_id = bind_img(sh, "gain", gain);
    ident_t tgt_id  = bind_tex_linear(sh, "tgt", tgt, NULL, NULL);
    if (!tgt_id) {
        pl_dispatch_abort(dispatch, &sh);
        return false;
    }

    // Weighted sums of the block means of `ref` and `tgt`
    ident_t put_means, total_means;
    sh_partial_sums(sh, PL_VAR_FLOAT, 0x3, 1, threads, 1, &put_means,
                    &total_means);

    // clang-format off
#pragma GLSL /* motion_gain_pass */                                             \
    int tid = int(gl_LocalInvocationIndex);                                     \
    vec2 size = vec2(${const int: tex_w}, ${const int: tex_h});                 \
    vec2 sum = vec2(0.0);                                                       \
    /* A single work group shares all blocks among its threads. */              \
    for (int i = tid; i < ${const int: grid_w * grid_h};                        \
         i += ${const int: threads})                                            \
    {                                                                           \
        ivec2 blk = ivec2(i % ${const int: grid_w}, i / ${const int: grid_w});  \
        vec2 mv = texelFetch($flow_id, blk, 0).xy;                              \
        vec2 org = vec2(blk * ${const int: bs});                                \
        if (any(lessThan(org + mv, vec2(0.0))) ||                               \
            any(greaterThan(org + mv + ${const float: fbs}, size)) ||           \
            any(greaterThan(org + ${const float: fbs}, size)))                  \
            continue;                                                           \
                                                                                \
        vec2 mean = vec2(0.0);                                                  \
        for (int j = 0; j < ${const int: bs * bs}; j++) {                       \
            vec2 pos = org + vec2(j % ${const int: bs}, j / ${const int: bs});  \
            mean.x += texelFetch($ref_id, ivec2(pos), 0).r;                     \
            mean.y += textureLod($tgt_id, (pos + 0.5 + mv) / size, 0.0).r;      \
        }                                                                       \
        mean /= ${const float: fbs * fbs};                                      \
                                                                                \
        /* How far the textures of the two blocks differ, relative to how much  \
         * texture there is. */                                                 \
        float ssd = 0.0, energy = 0.0;                                          \
        for (int j = 0; j < ${const int: bs * bs}; j++) {                       \
            vec2 pos = org + vec2(j % ${const int: bs}, j / ${const int: bs});  \
            float r = texelFetch($ref_id, ivec2(pos), 0).r - mean.x;            \
            float t = textureLod($tgt_id, (pos + 0.5 + mv) / size, 0.0).r       \
                    - mean.y;                                                   \
            ssd += (r - t) * (r - t);                                           \
            energy += r * r + t * t;                                            \
        }                                                                       \
        sum += mean / (1.0 + ssd / (0.1 * energy + ${const float: flat}));      \
    }                                                                           \
                                                                                \
    $put_means(0, tid, 0, sum);                                                 \
    barrier();                                                                  \
    if (tid == 0) {                                                             \
        sum = $total_means(0, 0);                                               \
        float change = sum.x / max(sum.y, 1e-6) - 1.0;                          \
        change = sign(change) *                                                 \
                 max(abs(change) - ${const float: MOTION_GAIN_DEADZONE}, 0.0);  \
        imageStore($gain_id, ivec2(0), vec4(1.0 + change));                     \
    }
    // clang-format on

    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader        = &sh,
        .dispatch_size = { 1, 1, 1 },
    ));
}

static void cand_lut_fill(void *data, const struct sh_lut_params *params)
{
    const struct motion_geometry *geom = params->priv;
    memcpy(data, geom->cand, 2 * (size_t) params->width * sizeof(int));
}

enum motion_prior {
    MOTION_PRIOR_SAME,   // the same direction, of the previous frame pair
    MOTION_PRIOR_NEGATE, // the opposite direction, from the same frame
    MOTION_PRIOR_INVERT, // the opposite direction, from the other frame
};

// All of the parameters for a motion search shader for one level of the image
// pyramid.
struct motion_search {
    pl_tex ref;                         // reference frame luma
    pl_tex tgt;                         // target frame luma
    pl_tex parent; // flow field of the previous motion search image pyramid
                   // level, or NULL at the coarsest level
    pl_tex prior;  // flow field of the same image pyramid level to take the
                   // temporal prior from, or NULL when there is none available
    enum motion_prior prior_kind;
    pl_tex gain;   // change in brightness from `ref` to `tgt` to compensate
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
    const bool subpixel = geom->subpixel;
    const int wr = r + subpixel; // radius of the search window

    // Total size of tile.
    const int ref_w = bpgx * bs, ref_h = bpgy * bs;

    const bool zero_mean = geom->zero_mean;
    const float block_px = bs * bs;

    const struct motion_shmem_layout shmem =
        motion_compute_shmem_layout(bs, r, bpgx, bpgy, nseed, subpixel,
                                    zero_mean);
    const int win = shmem.win;
    const int win_stride = shmem.win_stride;
    const int block_stride = shmem.block_stride;

    const int tex_w = p->ref->params.w, tex_h = p->ref->params.h;
    const int grid_w = p->out_mv->params.w, grid_h = p->out_mv->params.h;

    pl_shader sh = pass_begin(gpu, dispatch, threads, 1, shmem.size,
                              "exhaustive motion search (%dx%d blocks, radius "
                              "%d, %d seed candidates)", bs, bs, r, nseed);
    if (!sh)
        return false;

    // Fetch the luma 2x2 texels at a time: the cost of the search is mostly
    // that of its texture fetches.
    const int units = bs * bs / 4; // fetches per block
    const int parts = PL_MIN(tpb, units); // threads scoring a block's seeds

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
    ident_t ref = bind_tex(sh, "ref", p->ref);
    ident_t tgt = bind_tex(sh, "tgt", p->tgt);
    ident_t ref_quad = sh_tex_quad(sh, 0, ref, &p->ref->params);
    ident_t tgt_quad = sh_tex_quad(sh, 0, tgt, &p->tgt->params);
    ident_t out = bind_img(sh, "out_mv", p->out_mv);
    ident_t gain = get_target_gain(sh, p->gain);

    // Bind the flow field of the previous motion search image pyramid level,
    // unless this is the coarsest (first) level of the motion search.
    const bool has_parent = p->parent != NULL;
    const int parent_w = has_parent ? p->parent->params.w : 0;
    const int parent_h = has_parent ? p->parent->params.h : 0;
    const int rank_parent = geom->seeds.rank[MOTION_SEED_PARENT];
    const int rank_x = geom->seeds.rank[MOTION_SEED_PARENT_X];
    const int rank_y = geom->seeds.rank[MOTION_SEED_PARENT_Y];
    ident_t parent = NULL_IDENT;
    if (has_parent) {
        parent = bind_tex(sh, "parent", p->parent);
    }

    // Bind the flow field of the same motion search image pyramid level that
    // provides the temporal prior. Don't bind anything when there is no usable
    // prior available (upon a cold start, after a reset, or a change in motion
    // estimation direction).
    const bool has_prior = p->prior != NULL;
    const int prior_w = has_prior ? p->prior->params.w : 0;
    const int prior_h = has_prior ? p->prior->params.h : 0;
    const int rank_prior = geom->seeds.rank[MOTION_SEED_PRIOR];
    const bool prior_neg = p->prior_kind != MOTION_PRIOR_SAME;
    const bool prior_inv = p->prior_kind == MOTION_PRIOR_INVERT;
    const float fbs = bs;
    ident_t prior = NULL_IDENT;
    if (has_prior) {
        prior = bind_tex(sh, "prior", p->prior);
    }

    // Partial SSD of each seed candidate of each block of the tile, one per
    // thread of the block, and of the differences themselves for the
    // zero-mean SSD. Not needed if there is only a single candidate.
    ident_t put_sum2 = NULL_IDENT, total_sum2 = NULL_IDENT;
    ident_t put_sum = NULL_IDENT, total_sum = NULL_IDENT;
    if (nseed > 1) {
        sh_partial_sums(sh, PL_VAR_UINT, 0x1, bpg, parts, nseed, &put_sum2,
                        &total_sum2);
        if (zero_mean) {
            sh_partial_sums(sh, PL_VAR_SINT, 0x1, bpg, parts, nseed, &put_sum,
                            &total_sum);
        }
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
    @if (subpixel) {                                                            \
        /* Cost one pixel before and after the winner of each block, on x and   \
         * then on y. */                                                        \
        shared uint near_cost[${const int: 4 * bpg}];                           \
        uint luma8b(float x) {                                                  \
            return uint(round(clamp(x, 0.0, 1.0) * 255.0));                     \
        }                                                                       \
    @}                                                                          \
    uvec4 luma8b(vec4 x) {                                                      \
        return uvec4(round(clamp(x, 0.0, 1.0) * 255.0));                        \
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
        /* Use the vector of the corresponding block in the same motion search  \
         * pyramid level of the prior field, see `motion_prior`. */             \
        ivec2 prior_seed(ivec2 blk) {                                           \
            ivec2 p = clamp(blk, ivec2(0),                                      \
                            ivec2(${const int: prior_w},                        \
                                  ${const int: prior_h}) - 1);                  \
            vec2 mv = texelFetch($prior, p, 0).xy;                              \
            @if (prior_inv) {                                                   \
                /* The prior field maps its blocks onto the frame of this       \
                 * search. Find the block that lands on this one, i.e. solve    \
                 * `pos = q + mv(q)` for `q`, by fixed point iteration. */      \
                vec2 pos = (vec2(blk) + 0.5) * ${const float: fbs};             \
                for (int i = 0; i < 2; i++) {                                   \
                    ivec2 q = ivec2(floor((pos - mv) / ${const float: fbs}));   \
                    q = clamp(q, ivec2(0), ivec2(${const int: prior_w},         \
                                                 ${const int: prior_h}) - 1);   \
                    mv = texelFetch($prior, q, 0).xy;                           \
                }                                                               \
            @}                                                                  \
            @if (prior_neg)                                                     \
                mv = -mv;                                                       \
            return ivec2(round(mv));                                            \
        }                                                                       \
    @}                                                                          \
                                                                                \
    /* Sum of the (squared) differences between `a` and `b`. */                 \
    void sqdiff(uvec4 a, uvec4 b, inout int sum, inout uint sum2) {             \
        ivec4 d = ivec4(a) - ivec4(b);                                          \
        uvec4 d2 = uvec4(d * d);                                                \
        sum  +=  d.x +  d.y +  d.z +  d.w;                                      \
        sum2 += d2.x + d2.y + d2.z + d2.w;                                      \
    }                                                                           \
                                                                                \
    /* The SSD of a block without the difference of the two blocks' means,      \
     * from the sum of its differences: sum((d - mean(d))^2). */                \
    uint zero_mean_ssd(uint ssd, int sum) {                                     \
        float dc = float(sum) * float(sum) / ${const float: block_px};          \
        return ssd - min(ssd, uint(dc));                                        \
    }                                                                           \
                                                                                \
    /* Sum of squared differences between reference block `b` and the block     \
     * at `offset` within its search window in shared memory. */                \
    uint window_ssd(int b, ivec2 offset) {                                      \
        int bx = b % ${const int: bpgx};                                        \
        int by = b / ${const int: bpgx};                                        \
        uint ssd = 0u;                                                          \
        int sum = 0;                                                            \
        for (int i = 0; i < ${const int: bs}; i++) {                            \
            int ref_row = (by * ${const int: bs} + i) * ${const int: ref_w}     \
                        + bx * ${const int: bs};                                \
            int tgt_row = b * ${const int: block_stride}                        \
                        + (offset.y + i) * ${const int: win_stride}             \
                        + offset.x;                                             \
            for (int j = 0; j < ${const int: bs}; j++) {                        \
                int diff = int(sRef[ref_row + j]) - int(sTgt[tgt_row + j]);     \
                ssd += uint(diff * diff);                                       \
                sum += diff;                                                    \
            }                                                                   \
        }                                                                       \
        @if (zero_mean)                                                         \
            ssd = zero_mean_ssd(ssd, sum);                                      \
        return ssd;                                                             \
    }                                                                           \
                                                                                \
    /* Returns seed candidate `i` of block `blk`. */                            \
    ivec2 seed_candidate(ivec2 blk, int i) {                                    \
        @if (has_parent) {                                                      \
            if (i == ${const int: rank_parent})                                 \
                return parent_seed(blk);                                        \
            /* The adjacent block towards the centre of the parent has the same \
             * parent, so only the one on the opposite side has a different one \
             * to seed with. */                                                 \
            ivec2 side = 2 * (blk & 1) - 1;                                     \
            if (i == ${const int: rank_x})                                      \
                return parent_seed(blk + ivec2(side.x, 0));                     \
            if (i == ${const int: rank_y})                                      \
                return parent_seed(blk + ivec2(0, side.y));                     \
        @}                                                                      \
        @if (has_prior) {                                                       \
            if (i == ${const int: rank_prior})                                  \
                return prior_seed(blk);                                         \
        @}                                                                      \
        return ivec2(0);                                                        \
    }
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
    /* Score the seed candidates of this thread's block. Each thread of the     \
     * block sums a share of its pixels, read from the textures, so that none   \
     * has to wait for another. */                                              \
    ivec2 seed = ivec2(0);                                                      \
    @if (nseed > 1) {                                                           \
        ivec2 sd[${const int: nseed}];                                          \
        int psum[${const int: nseed}];                                          \
        uint psum2[${const int: nseed}];                                        \
        for (int i = 0; i < ${const int: nseed}; i++) {                         \
            sd[i] = seed_candidate(blk, i);                                     \
            psum[i]  = 0;                                                       \
            psum2[i] = 0u;                                                      \
        }                                                                       \
        for (int j = c0; j < ${const int: units}; j += ${const int: tpb}) {     \
            ivec2 pos = blk * ${const int: bs}                                  \
                      + 2 * ivec2(j % ${const int: bs / 2},                     \
                                  j / ${const int: bs / 2});                    \
            uvec4 ref_px = luma8b($ref_quad(pos));                              \
            for (int i = 0; i < ${const int: nseed}; i++)                       \
                sqdiff(ref_px, luma8b($gain * $tgt_quad(pos + sd[i])),          \
                       psum[i], psum2[i]);                                      \
        }                                                                       \
        if (c0 < ${const int: parts}) {                                         \
            for (int i = 0; i < ${const int: nseed}; i++) {                     \
                $put_sum2(b, c0, i, psum2[i]);                                  \
                @if (zero_mean)                                                 \
                    $put_sum(b, c0, i, psum[i]);                                \
            }                                                                   \
        }                                                                       \
    @}                                                                          \
                                                                                \
    /* Load block from reference image into shared memory. We clamp coordinates \
     * and pad partial blocks that extend past the image boundaries. */         \
    for (uint i = tid; i < ${const uint: ref_w * ref_h / 4};                    \
         i += ${const uint: threads})                                           \
    {                                                                           \
        ivec2 elem = 2 * ivec2(int(i) % ${const int: ref_w / 2},                \
                               int(i) / ${const int: ref_w / 2});               \
        uvec4 px = luma8b($ref_quad(tile * ${const int: bs} + elem));           \
        int j = elem.y * ${const int: ref_w} + elem.x;                          \
        sRef[j] = px.x;                                                         \
        sRef[j + 1] = px.y;                                                     \
        sRef[j + ${const int: ref_w}] = px.z;                                   \
        sRef[j + ${const int: ref_w + 1}] = px.w;                               \
    }                                                                           \
                                                                                \
    /* Pick the seed candidate with the lowest SSD, the first one on ties, to   \
     * center the search window of the block on. */                             \
    @if (nseed > 1) {                                                           \
        barrier();                                                              \
        uint best_ssd = 0xFFFFFFFFu;                                            \
        for (int i = 0; i < ${const int: nseed}; i++) {                         \
            uint ssd = $total_sum2(b, i);                                       \
            @if (zero_mean)                                                     \
                ssd = zero_mean_ssd(ssd, $total_sum(b, i));                     \
            if (ssd < best_ssd) {                                               \
                best_ssd = ssd;                                                 \
                seed = sd[i];                                                   \
            }                                                                   \
        }                                                                       \
    @}                                                                          \
                                                                                \
    /* Load the search window of this thread's block from the target image into \
     * shared memory. Pad rows of the search window in shared memory to avoid   \
     * bank conflicts. */                                                       \
    ivec2 origin = blk * ${const int: bs} + seed - ${const int: wr};            \
    int win_base = b * ${const int: block_stride};                              \
    for (int i = c0; i < ${const int: win * win / 4}; i += ${const int: tpb}) { \
        ivec2 elem = 2 * ivec2(i % ${const int: win / 2},                       \
                               i / ${const int: win / 2});                      \
        uvec4 px = luma8b($gain * $tgt_quad(origin + elem));                    \
        int j = win_base + elem.y * ${const int: win_stride} + elem.x;          \
        sTgt[j] = px.x;                                                         \
        sTgt[j + 1] = px.y;                                                     \
        sTgt[j + ${const int: win_stride}] = px.z;                              \
        sTgt[j + ${const int: win_stride + 1}] = px.w;                          \
    }                                                                           \
    barrier();  /* make sure all elements are loaded into shared memory */      \
                                                                                \
    /* Compute the sum of squared differences (SSD) of all candidates of each   \
     * block. One thread loads all elements of a given candidate. */            \
    uint best = 0xFFFFFFFFu, rank = 0xFFFFFFFFu;                                \
    for (int c = c0; c < ${const int: ncand}; c += ${const int: tpb}) {         \
        ivec2 cand_offset = $cand(c);                                           \
        /* Change the candidate offset into coordinates within the window */    \
        cand_offset += ${const int: wr};                                        \
        uint ssd = window_ssd(b, cand_offset);                                  \
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
    ivec2 winner = $cand(int(best_rank[b]));                                    \
    @if (subpixel) {                                                            \
        for (int i = c0; i < 4; i += ${const int: tpb}) {                       \
            ivec2 adj = winner;                                                 \
            adj[i / 2] += 2 * (i % 2) - 1;                                      \
            near_cost[4 * b + i] = window_ssd(b, adj + ${const int: wr});       \
        }                                                                       \
        barrier();                                                              \
    @}                                                                          \
    if (c0 == 0 && blk.x < ${const int: grid_w} &&                              \
        blk.y < ${const int: grid_h})                                           \
    {                                                                           \
        vec2 mv = vec2(seed + winner);                                          \
        @if (subpixel) {                                                        \
            /* Move to the vertex of the parabola through the cost of the       \
             * winner and of its two neighbours, separately for each axis. It   \
             * is within half a pixel if the winner is the lowest of the three, \
             * which the neighbours outside of the search need not respect. */  \
            vec2 prev = vec2(near_cost[4 * b], near_cost[4 * b + 2]);           \
            vec2 next = vec2(near_cost[4 * b + 1], near_cost[4 * b + 3]);       \
            vec2 curv = prev + next - vec2(2.0 * float(best_cost[b]));          \
            vec2 off = 0.5 * (prev - next) / max(curv, vec2(1.0));              \
            mv += clamp(off, vec2(-0.5), vec2(0.5));                            \
        @}                                                                      \
        imageStore($out, blk, vec4(mv, 0.0, 0.0));                              \
    }
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

// Replace every block's motion vector with one derived from the surrounding 3x3
// neighbourhood of blocks, to suppress vectors that the search got wrong.
static bool motion_smooth_pass(pl_gpu gpu, pl_dispatch dispatch, int bs,
                               pl_tex ref, pl_tex tgt, pl_tex gain_tex,
                               pl_tex in, pl_tex out)
{
    const int grid_w = out->params.w, grid_h = out->params.h;
    const int tex_w = ref->params.w, tex_h = ref->params.h;

    // A work group covers `gx` x `gy` blocks, with one thread per row. Every
    // other row of a block tells the candidates apart as well as all of them,
    // so each thread takes half of an even row.
    const int gx = PL_CLAMP(64 / bs, 1, 8), gy = 4;
    const int span = bs / 2;
    const int bw = gx * bs, bh = gy;
    const size_t shmem = bw * bh * 8 * sizeof(float);

    pl_shader sh = pass_begin(gpu, dispatch, bw, bh, shmem,
                              "motion vector smoothing (3x3, %dx%d blocks)",
                              grid_w, grid_h);
    if (!sh)
        return false;

    ident_t tex = bind_tex(sh, "flow_in", in);
    ident_t dst = bind_img(sh, "flow_out", out);
    ident_t load_block = sh_tex_load_block(sh, 3, 3, 0x3, tex, &in->params);
    ident_t ref_id = bind_tex(sh, "ref", ref);
    ident_t gain = get_target_gain(sh, gain_tex);

    // The vectors are fractional
    ident_t tgt_pt, tgt_id = bind_tex_linear(sh, "tgt", tgt, NULL, &tgt_pt);
    if (!tgt_id)
        goto error;

    // SSD of each neighbour over each row of each block of the group
    ident_t put_sum, total_sum;
    sh_partial_sums(sh, PL_VAR_FLOAT, 0x1, gx * gy, bs, 8, &put_sum, &total_sum);

    // clang-format off
#pragma GLSL /* motion_smooth_pass */                                           \
    ivec2 grid = ivec2(${const int: grid_w}, ${const int: grid_h});             \
    ivec2 size = ivec2(${const int: tex_w}, ${const int: tex_h});               \
    ivec2 tid  = ivec2(gl_LocalInvocationID.xy);                                \
    int row    = tid.x % ${const int: bs};                                      \
    ivec2 blk  = ivec2(gl_WorkGroupID.xy) *                                     \
                 ivec2(${const int: gx}, ${const int: gy}) +                    \
                 ivec2(tid.x / ${const int: bs}, tid.y);                        \
    int group  = tid.y * ${const int: gx} + tid.x / ${const int: bs};           \
                                                                                \
    /* The 3x3 neighbourhood around this block, clamped. */                     \
    vec2 mv[9];                                                                 \
    $load_block(blk - 1, mv);                                                   \
                                                                                \
    /* Take the vector of the neighbour that matches the block best. Every      \
     * thread matches its share of the block against the vectors of the 8       \
     * neighbours. The block's own vector is left out: the search already       \
     * preferred it over every vector it tried, usually including those of      \
     * the neighbours, so it would win exactly where its match is               \
     * ambiguous. */                                                            \
    float ssd[8];                                                               \
    for (int k = 0; k < 8; k++)                                                 \
        ssd[k] = 0.0;                                                           \
    for (int j = 0; j < ${const int: span}; j++) {                              \
        ivec2 off = ivec2(j + (row & 1) * ${const int: span}, row & ~1);        \
        ivec2 pos = min(blk * ${const int: bs} + off, size - 1);                \
        float r = texelFetch($ref_id, pos, 0).r;                                \
        vec2 c = vec2(pos) + vec2(0.5);                                         \
        for (int k = 0; k < 8; k++) {                                           \
            vec2 v = mv[k < 4 ? k : k + 1].xy;                                  \
            float t = $gain * textureLod($tgt_id, (c + v) * $tgt_pt, 0.0).r;    \
            float d = r - t;                                                    \
            ssd[k] += d * d;                                                    \
        }                                                                       \
    }                                                                           \
    for (int k = 0; k < 8; k++)                                                 \
        $put_sum(group, row, k, ssd[k]);                                        \
    barrier();                                                                  \
                                                                                \
    if (all(lessThan(blk, grid)) && row == 0) {                                 \
        /* Without any neighbour inside the grid, keep the block's vector. */   \
        vec2 best = mv[4].xy;                                                   \
        float best_ssd = 1e30;                                                  \
        for (int k = 0; k < 8; k++) {                                           \
            int i = k < 4 ? k : k + 1;                                          \
            ivec2 p = blk + ivec2(i % 3, i / 3) - 1;                            \
            float sum = $total_sum(group, k);                                   \
            bool inside = all(equal(p, clamp(p, ivec2(0), grid - 1)));          \
            if (inside && sum < best_ssd) {                                     \
                best = mv[i].xy;                                                \
                best_ssd = sum;                                                 \
            }                                                                   \
        }                                                                       \
                                                                                \
        imageStore($dst, blk, vec4(best, 0.0, 0.0));                            \
    }
    // clang-format on

    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = {
            PL_DIV_UP(grid_w, gx),
            PL_DIV_UP(grid_h, gy),
            1,
        },
    ));

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// Writes a vector for every 4x4 sub-block of `out`: of the vectors of the 3x3
// blocks around it in `blocks` (of `bs` pixels squared), and of
// `MOTION_REFINE_4X4_RINGS` rings of blocks further away, the one that best
// matches the sub-block and a border of one pixel around it. The vector of the
// sub-block's own block is preferred by a margin of one code value (8 bit).
static bool motion_refine_4x4_pass(pl_gpu gpu, pl_dispatch dispatch, int bs,
                                   pl_tex ref, pl_tex tgt, pl_tex gain_tex,
                                   pl_tex blocks, pl_tex out)
{
    const int w = ref->params.w, h = ref->params.h;
    const int grid_w = blocks->params.w, grid_h = blocks->params.h;
    const int sub_w = out->params.w, sub_h = out->params.h;
    const int sub = 4, win = sub + 2;
    const int ncand = 9 + 4 * MOTION_REFINE_4X4_RINGS;

    pl_shader sh = pass_begin(gpu, dispatch, 8, 8, 0,
                              "motion vector refinement (4x4, %dx%d)", sub_w,
                              sub_h);
    if (!sh)
        return false;

    ident_t src = bind_tex(sh, "blocks", blocks);
    ident_t dst = bind_img(sh, "sub_out", out);
    ident_t ref_id = bind_tex(sh, "ref", ref);
    ident_t tgt_id = bind_tex(sh, "tgt", tgt);
    ident_t load_block = sh_tex_load_block(sh, win, win, 0x1, ref_id,
                                           &ref->params);
    ident_t block_ssd = sh_block_ssd(sh, win, win, 0, tgt_id, &tgt->params);
    ident_t gain = get_target_gain(sh, gain_tex);

    // clang-format off
#pragma GLSL /* motion_refine_4x4_pass */                                       \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 grid = ivec2(${const int: grid_w}, ${const int: grid_h});             \
    ivec2 sb = ivec2(gl_GlobalInvocationID.xy);                                 \
    if (all(lessThan(sb, ivec2(${const int: sub_w}, ${const int: sub_h})))) {   \
        /* Origin of the sub-block in the frame, and the block its centre is    \
         * in. */                                                               \
        ivec2 org = sb * ${const int: sub};                                     \
        ivec2 blk = min(org + ${const int: sub / 2}, size - 1)                  \
                  / ${const int: bs};                                           \
        float ref_px[${const int: win * win}];                                  \
        $load_block(org - 1, ref_px);                                           \
                                                                                \
        /* The own block first, then its neighbours, then the far rings. */     \
        const ivec2 cand[${const int: ncand}] = ivec2[${const int: ncand}](     \
            ivec2( 0,  0),                                                      \
            ivec2(-1, -1), ivec2( 0, -1), ivec2( 1, -1), ivec2(-1,  0),         \
            ivec2( 1,  0), ivec2(-1,  1), ivec2( 0,  1), ivec2( 1,  1)          \
        @for (r < MOTION_REFINE_4X4_RINGS) {                                    \
            , ivec2(0, -(2 << @r)), ivec2(-(2 << @r), 0),                       \
              ivec2(2 << @r, 0), ivec2(0, 2 << @r)                              \
        @}                                                                      \
        );                                                                      \
                                                                                \
        /* Costs are mean squared errors over the window, times 255. */         \
        const float margin = ${const float: 1.0f / 255.0f};                     \
        const float scale = ${const float: 255.0f / (win * win)};               \
        vec2 best = vec2(0.0), own = vec2(0.0), last = vec2(0.0);               \
        float best_cost = 1e9;                                                  \
        for (int k = 0; k < ${const int: ncand}; k++) {                         \
            /* The others pay the margin: none can beat what is within it. */   \
            if (k > 0 && best_cost <= margin)                                   \
                break;                                                          \
                                                                                \
            ivec2 p = clamp(blk + cand[k], ivec2(0), grid - 1);                 \
            vec2 mv = texelFetch($src, p, 0).xy;                                \
                                                                                \
            /* Most candidates repeat the own, the previous or the best one. */ \
            vec2 d0 = mv - own, d1 = mv - last, d2 = mv - best;                 \
            if (k > 0 && (dot(d0, d0) < 0.01 || dot(d1, d1) < 0.01 ||           \
                          dot(d2, d2) < 0.01))                                  \
                continue;                                                       \
            if (k == 0)                                                         \
                own = mv;                                                       \
            last = mv;                                                          \
                                                                                \
            /* A candidate is dropped as soon as it has lost. */                \
            float pay = k == 0 ? 0.0 : margin;                                  \
            float limit = (best_cost - pay) / scale;                            \
            float cost = pay + scale * $block_ssd(ref_px, vec2(org - 1) + mv,   \
                                                  $gain, limit);                \
            if (cost < best_cost) {                                             \
                best_cost = cost;                                               \
                best = mv;                                                      \
            }                                                                   \
        }                                                                       \
        imageStore($dst, sb, vec4(best, 0.0, 0.0));                             \
    }
    // clang-format on

    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh,
        .width  = sub_w,
        .height = sub_h,
    ));
}

// Writes a vector for every pixel of `out`, which has the dimensions of `ref`:
// the vector of the pixel's block in `flow` (of `bs` pixels squared), or that
// of one of the 8 neighbouring blocks if it matches the 3x3 pixels around the
// pixel better by more than `MOTION_REFINE_1X1_MARGIN`.
static bool motion_refine_1x1_pass(pl_gpu gpu, pl_dispatch dispatch, int bs,
                                   pl_tex ref, pl_tex tgt, pl_tex gain_tex,
                                   pl_tex flow, pl_tex out)
{
    const int w = out->params.w, h = out->params.h;
    const int grid_w = flow->params.w, grid_h = flow->params.h;

    // SSD over the 9 pixels of the window, in units of the luma textures
    const float margin = MOTION_REFINE_1X1_MARGIN * 9.0f / (255.0f * 255.0f);

    pl_shader sh = pass_begin(gpu, dispatch, 16, 16, 0,
                              "motion vector refinement (1x1, %dx%d)", w, h);
    if (!sh)
        return false;

    ident_t src = bind_tex(sh, "flow_in", flow);
    ident_t dst = bind_img(sh, "flow_out", out);
    ident_t ref_id = bind_tex(sh, "ref", ref);
    ident_t tgt_id = bind_tex(sh, "tgt", tgt);
    ident_t load_block = sh_tex_load_block(sh, 3, 3, 0x1, ref_id, &ref->params);
    ident_t block_ssd = sh_block_ssd(sh, 3, 3, 0, tgt_id, &tgt->params);
    ident_t gain = get_target_gain(sh, gain_tex);

    // clang-format off
#pragma GLSL /* motion_refine_1x1_pass */                                       \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 grid = ivec2(${const int: grid_w}, ${const int: grid_h});             \
    ivec2 pos  = ivec2(gl_GlobalInvocationID.xy);                               \
    if (all(lessThan(pos, size))) {                                             \
        ivec2 blk = pos / ${const int: bs};                                     \
        vec2 org = vec2(pos - 1);                                               \
        float ref_px[9];                                                        \
        $load_block(pos - 1, ref_px);                                           \
                                                                                \
        /* The vector of the pixel's own block comes first. The others pay the  \
         * margin, so none of them can win once it matches within that. */      \
        vec2 cand[9];                                                           \
        cand[0] = texelFetch($src, blk, 0).xy;                                  \
        vec2 best = cand[0];                                                    \
        float best_ssd = $block_ssd(ref_px, org + best, $gain, 1e30);           \
        if (best_ssd > ${const float: margin}) {                                \
            for (int k = 1; k < 9; k++) {                                       \
                int i = k <= 4 ? k - 1 : k;                                     \
                ivec2 p = clamp(blk + ivec2(i % 3, i / 3) - 1, ivec2(0),        \
                                grid - 1);                                      \
                cand[k] = texelFetch($src, p, 0).xy;                            \
                /* A vector equal to an earlier one cannot win. */              \
                bool dup = false;                                               \
                for (int j = 0; j < k; j++)                                     \
                    dup = dup || cand[k] == cand[j];                            \
                if (dup)                                                        \
                    continue;                                                   \
                float ssd = ${const float: margin}                              \
                          + $block_ssd(ref_px, org + cand[k], $gain,            \
                                       best_ssd - ${const float: margin});      \
                if (ssd < best_ssd) {                                           \
                    best = cand[k];                                             \
                    best_ssd = ssd;                                             \
                }                                                               \
            }                                                                   \
        }                                                                       \
        imageStore($dst, pos, vec4(best, 0.0, 0.0));                            \
    }
    // clang-format on

    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh,
        .width  = w,
        .height = h,
    ));
}

// Returns the number of refine passes needed, and the block size of each
static int refine_plan(const struct pl_motion_estimate_params *params,
                       int plan[MOTION_MAX_REFINE_PASSES])
{
    int num = 0;
    if (params->block_size_fine <= 1 && params->block_size_coarse > 1)
        plan[num++] = 1;
    if (params->block_size_fine <= 4 && params->block_size_coarse > 4)
        plan[num++] = 4;
    pl_static_assert(MOTION_MAX_REFINE_PASSES >= 2);
    pl_assert(num <= MOTION_MAX_REFINE_PASSES);
    return num;
}

// Allocate or resize `pyramid` to hold a field for each of the `num` block
// sizes of `sizes`, at the same index, for frames of `w` x `h`.
static bool refine_pyramid_recreate(pl_gpu gpu, void *alloc,
                                    pl_tex_pyramid *pyramid, int w, int h,
                                    const int sizes[], int num, pl_fmt fmt)
{
    pl_pyramid_resize(gpu, alloc, pyramid, num);
    for (int i = 0; i < num; i++) {
        const int fw = PL_DIV_UP(w, sizes[i]);
        const int fh = PL_DIV_UP(h, sizes[i]);
        bool ok = pl_tex_recreate(gpu, &pyramid->elem[i], pl_tex_params(
            .w          = fw,
            .h          = fh,
            .format     = fmt,
            .sampleable = true,
            .storable   = true,
        ));
        if (!ok) {
            PL_ERR(gpu, "pl_motion_estimate: failed creating %dx%d refined "
                        "flow field!", fw, fh);
            return false;
        }
    }

    return true;
}

// Refines `blocks`, of `bs` pixels squared, into the levels of `fine`, of the
// block sizes of `sizes`, from the coarsest.
static bool refine_flow(pl_gpu gpu, pl_dispatch dispatch, const int sizes[],
                        pl_tex_pyramid *fine, int bs, pl_tex ref, pl_tex tgt,
                        pl_tex gain, pl_tex blocks)
{
    pl_tex in = blocks;
    for (int i = fine->num - 1; i >= 0; i--) {
        bool ok = false;
        switch (sizes[i]) {
        case 1:
            ok = motion_refine_1x1_pass(gpu, dispatch, bs, ref, tgt, gain, in,
                                        fine->elem[i]);
            break;
        case 4:
            ok = motion_refine_4x4_pass(gpu, dispatch, bs, ref, tgt, gain, in,
                                        fine->elem[i]);
            break;
        }
        if (!ok)
            return false;
        in = fine->elem[i];
        bs = sizes[i];
    }

    return true;
}

// Estimates the quality of `flow`. The quality of a vector scales with the
// consistency between the 3x3 window around its pixel and where it leads,
// and with the consistency of the vector and its neighbors.
//
// A mismatch means less the more texture the window has, so it is taken
// relative to that.
static bool motion_quality_pass(pl_gpu gpu, pl_dispatch dispatch, int bs,
                                pl_tex ref, pl_tex tgt, pl_tex gain_tex,
                                pl_tex flow, pl_tex out)
{
    const int w = ref->params.w, h = ref->params.h;
    const int out_w = out->params.w, out_h = out->params.h;
    const int sub = MOTION_QUALITY_SUB, win = sub + 2;
    const int grid_w = flow->params.w, grid_h = flow->params.h;
    const float noise = 9.0f * PL_SQUARE(MOTION_QUALITY_NOISE / 255.0f);
    const float outlier = 1.0f / PL_SQUARE(MOTION_QUALITY_OUTLIER);

    pl_shader sh = pass_begin(gpu, dispatch, 16, 16, 0,
                              "motion vector quality (%dx%d)", out_w, out_h);
    if (!sh)
        return false;

    ident_t src = bind_tex(sh, "flow", flow);
    ident_t dst = bind_img(sh, "quality_out", out);
    ident_t ref_id = bind_tex(sh, "ref", ref);
    ident_t tgt_id = bind_tex(sh, "tgt", tgt);
    ident_t load_block = sh_tex_load_block(sh, win, win, 0x1, ref_id,
                                           &ref->params);
    ident_t block_ssd = sh_block_ssd(sh, 3, 3, 0, tgt_id, &tgt->params);
    ident_t load_vectors = sh_tex_load_block(sh, 4, 4, 0x3, src, &flow->params);
    ident_t median9 = sh_median(sh, 9, 0x3);
    ident_t gain = get_target_gain(sh, gain_tex);

    // clang-format off
#pragma GLSL /* motion_quality_pass */                                          \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 grid = ivec2(${const int: grid_w}, ${const int: grid_h});             \
    ivec2 sq   = ivec2(gl_GlobalInvocationID.xy);                               \
    if (all(lessThan(sq, ivec2(${const int: out_w}, ${const int: out_h})))) {   \
        /* The pixels of the square, and those around it. */                    \
        ivec2 org = sq * ${const int: sub};                                     \
        float luma[${const int: win * win}];                                    \
        $load_block(org - 1, luma);                                             \
                                                                                \
        /* The vectors of the pixels of the square, which are at most 2x2,      \
         * and those around them. */                                            \
        ivec2 blk_org = org / ${const int: bs} - 1;                             \
        vec2 flow[16];                                                          \
        $load_vectors(blk_org, flow);                                           \
                                                                                \
        float sum = 0.0, num = 0.0;                                             \
        float ref_px[9], texture, ssd, d, q;                                    \
        ivec2 pos, blk;                                                         \
        vec2 mv, to, around[9];                                                 \
        @for (y < sub) {                                                        \
        @for (x < sub) {                                                        \
            pos = org + ivec2(@x, @y);                                          \
            if (all(lessThan(pos, size))) {                                     \
                /* Spelled out for fields with a vector per pixel, to index     \
                 * the array with constants. */                                 \
                @if (bs == 1) {                                                 \
                    blk = ivec2(@x + 1, @y + 1);                                \
                @} else {                                                       \
                    blk = pos / ${const int: bs} - blk_org;                     \
                @}                                                              \
                @for (j < 9) {                                                  \
                    around[@j] = flow[(blk.y + @j / 3 - 1) * 4                  \
                                      + blk.x + @j % 3 - 1];                    \
                @}                                                              \
                mv = around[4];                                                 \
                d = distance(mv, $median9(around));                             \
                q = exp(-d * d * ${const float: outlier});                      \
                                                                                \
                texture = ${const float: noise};                                \
                @for (j < 9) {                                                  \
                    ref_px[@j] = luma[(@y + @j / 3) * ${const int: win}         \
                                      + @x + @j % 3];                           \
                    d = ref_px[@j] - luma[(@y + 1) * ${const int: win}          \
                                          + @x + 1];                            \
                    texture += d * d;                                           \
                @}                                                              \
                ssd = $block_ssd(ref_px, vec2(pos - 1) + mv, $gain, 1e30);      \
                                                                                \
                /* Where a vector leaves the frame there is nothing to match    \
                 * it against. */                                               \
                to = vec2(pos) + 0.5 + mv;                                      \
                if (all(greaterThanEqual(to, vec2(0.0))) &&                     \
                    all(lessThanEqual(to, vec2(size))))                         \
                {                                                               \
                    q *= 1.0 - smoothstep(                                      \
                        ${const float: MOTION_QUALITY_COST_LO},                 \
                        ${const float: MOTION_QUALITY_COST_HI},                 \
                        sqrt(ssd / texture));                                   \
                }                                                               \
                                                                                \
                sum += q;                                                       \
                num += 1.0;                                                     \
            }                                                                   \
        @}                                                                      \
        @}                                                                      \
        imageStore($dst, sq, vec4(sum / num, 0.0, 0.0, 0.0));                   \
    }
    // clang-format on

    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh,
        .width  = out_w,
        .height = out_h,
    ));
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

// Reusable helper to create global motion and histogram scratch buffers.
static bool global_buf_create(pl_gpu gpu, pl_buf *buf, size_t size,
                              bool host_readable)
{
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

    pl_shader sh = pass_begin(gpu, dispatch, dim, dim, shmem,
                              "global motion histogram (%dx%d blocks)", grid_w,
                              grid_h);
    if (!sh)
        return false;

    ident_t src = bind_tex(sh, "flow_in", flow);
    bind_buf(sh, "global_hist_buf", hist, PL_DESC_ACCESS_READWRITE,
             &global_hist_var, 1);

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

    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh,
        .width  = grid_w,
        .height = grid_h,
    ));
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

    pl_shader sh = pass_begin(gpu, dispatch, MOTION_GLOBAL_BINS, 1, shmem,
                              "global motion median (%d blocks)", num);
    if (!sh)
        return false;

    bind_buf(sh, "global_hist_buf", hist, PL_DESC_ACCESS_READWRITE,
             &global_hist_var, 1);
    bind_buf(sh, "global_motion_out", out, PL_DESC_ACCESS_WRITEONLY,
             global_buf_vars[dir], GLOBAL_BUF_NUM_VARS);

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

    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader        = &sh,
        .dispatch_size = { 1, 1, 1 },
    ));
}

// Estimates the global motion of `dir` from its flow field.
static bool estimate_global(pl_gpu gpu, pl_dispatch dispatch,
                            struct sh_motion_obj *obj,
                            enum pl_motion_direction dir)
{
    const int d = dir_index(dir);
    pl_tex flow = obj->dir[d].coarse.elem[0];
    pl_buf global = obj->dir[d].global_flow;

    return motion_global_hist_pass(gpu, dispatch, flow, obj->scratch.hist) &&
           motion_global_median_pass(gpu, dispatch, obj->scratch.hist,
                                     flow->params.w * flow->params.h, global,
                                     dir);
}

// Estimates the quality of the vectors of every direction in
// `obj->valid`, from the luma of the frames they were estimated between.
static bool estimate_quality(pl_gpu gpu, pl_dispatch dispatch,
                             struct sh_motion_obj *obj, pl_tex luma_a,
                             pl_tex luma_b)
{
    int refine[MOTION_MAX_REFINE_PASSES];
    const int num_refine = refine_plan(&obj->params, refine);
    const int bs = num_refine ? refine[0] : obj->params.block_size_coarse;
    const bool has_gain =
        pyramid_num_levels(obj->frame_w, obj->frame_h, &obj->params) > 1;

    for (int d = 0; d < NUM_DIRS; d++) {
        const enum pl_motion_direction dir = 1 << d;
        if (!(obj->valid & dir))
            continue;

        const bool fwd = d == FWD;
        const pl_tex_pyramid *fine = &obj->dir[d].fine;
        const pl_tex_pyramid *blocks = &obj->dir[d].coarse;
        bool ok = motion_quality_pass(gpu, dispatch, bs,
                                      fwd ? luma_a : luma_b,
                                      fwd ? luma_b : luma_a,
                                      has_gain ? obj->dir[d].gain : NULL,
                                      num_refine ? fine->elem[0]
                                                 : blocks->elem[0],
                                      obj->dir[d].quality);
        if (!ok)
            return false;
    }

    return true;
}

// Creates all GPU objects needed for this motion estimation call. Whenever the
// dimensions, parameters, directions or features change, this recreates the
// entire state from scratch, discarding the temporal state.
static bool motion_setup(pl_gpu gpu, struct sh_motion_obj *obj, int w, int h,
                         const struct pl_motion_estimate_params *params,
                         enum pl_motion_direction direction,
                         enum pl_motion_feature features)
{
    if (w == obj->frame_w && h == obj->frame_h &&
        direction == obj->direction && features == obj->features &&
        params_eq(&obj->params, params))
        return true;

    PL_DEBUG(gpu, "pl_motion: (re)initializing motion state");
    motion_uninit(gpu, obj);
    obj->params = *params;
    obj->frame_w = w;
    obj->frame_h = h;
    obj->direction = direction;
    obj->features = features;

    const int levels = pyramid_num_levels(w, h, params);
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 16, 0,
                             PL_FMT_CAP_LINEAR | PL_FMT_CAP_RENDERABLE);
    if (!fmt) {
        PL_ERR(gpu, "pl_motion_estimate: no renderable, linearly filterable "
                    "single component float format available!");
        goto error;
    }

    pl_fmt mv_fmt = flow_fmt(gpu);
    if (!mv_fmt) {
        PL_ERR(gpu, "pl_motion_estimate: no storable, linearly filterable "
                    "two component float format available!");
        goto error;
    }

    // The search covers the trailing partial blocks as well
    const int grid_w = PL_DIV_UP(w, params->block_size_coarse);
    const int grid_h = PL_DIV_UP(h, params->block_size_coarse);

    const struct pl_tex_params *input_params = pl_tex_params(
        .w          = w,
        .h          = h,
        .format     = fmt,
        .sampleable = true,
        .renderable = true,
        .storable   = fmt->caps & PL_FMT_CAP_STORABLE,
    );

    const struct pl_tex_params *flow_params = pl_tex_params(
        .w          = grid_w,
        .h          = grid_h,
        .format     = mv_fmt,
        .sampleable = true,
        .storable   = true,
    );

    for (int d = 0; d < NUM_DIRS; d++) {
        if (!(direction & (1 << d)))
            continue;
        if (!pl_pyramid_recreate(gpu, obj, &obj->dir[d].coarse, levels,
                                 flow_params))
            goto error;
    }

    if (!pl_pyramid_recreate(gpu, obj, &obj->scratch.raw, levels, flow_params))
        goto error;

    int refine[MOTION_MAX_REFINE_PASSES];
    const int num_refine = refine_plan(params, refine);
    for (int d = 0; d < NUM_DIRS; d++) {
        if (!(direction & (1 << d)))
            continue;
        if (!refine_pyramid_recreate(gpu, obj, &obj->dir[d].fine, w, h,
                                     refine, num_refine, mv_fmt))
            goto error;
    }

    pl_fmt gain_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 32, 32, PL_FMT_CAP_STORABLE);

    for (int i = 0; i < NUM_DIRS; i++) {
        if (!(direction & (1 << i)))
            continue;
        bool ok = pl_tex_recreate(gpu, &obj->dir[i].gain, pl_tex_params(
            .w          = 1,
            .h          = 1,
            .format     = PL_DEF(gain_fmt, mv_fmt),
            .sampleable = true,
            .storable   = true,
        ));

        if (!ok) {
            PL_ERR(gpu, "pl_motion_estimate: failed creating gain texture!");
            goto error;
        }
    }

    if (!pl_pyramid_recreate(gpu, obj, &obj->pyramid_prev,  levels, input_params) ||
        !pl_pyramid_recreate(gpu, obj, &obj->scratch.input, levels, input_params))
        goto error;

    if (features & PL_MOTION_QUALITY) {
        pl_fmt qfmt = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 8, 8,
                                  PL_FMT_CAP_STORABLE | PL_FMT_CAP_LINEAR);
        if (!qfmt) {
            PL_ERR(gpu, "pl_motion_estimate: no storable, linearly filterable "
                        "single component format available!");
            goto error;
        }

        const int qw = PL_DIV_UP(w, MOTION_QUALITY_SUB);
        const int qh = PL_DIV_UP(h, MOTION_QUALITY_SUB);
        for (int d = 0; d < NUM_DIRS; d++) {
            if (!(direction & (1 << d)))
                continue;
            bool ok = pl_tex_recreate(gpu, &obj->dir[d].quality, pl_tex_params(
                .w          = qw,
                .h          = qh,
                .format     = qfmt,
                .sampleable = true,
                .storable   = true,
            ));
            if (!ok) {
                PL_ERR(gpu, "pl_motion_estimate: failed creating %dx%d quality "
                            "texture!", qw, qh);
                goto error;
            }
        }
    }

    if (features & PL_MOTION_GLOBAL) {
        if (!global_buf_create(gpu, &obj->scratch.hist,
                               2 * MOTION_GLOBAL_BINS * sizeof(uint32_t),
                               false))
            goto error;

        for (int d = 0; d < NUM_DIRS; d++) {
            if (!(direction & (1 << d)))
                continue;
            if (!global_buf_create(gpu, &obj->dir[d].global_flow,
                                   sizeof(struct global_buf_data), true))
                goto error;
        }
    }

    return true;

error:
    motion_uninit(gpu, obj);
    return false;
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
    if (!motion_setup(gpu, obj, w, h, params, input->direction,
                      input->features))
        return false;

    if (obj->sig_prev == input->a.sig && obj->sig_next == input->b.sig &&
        (obj->valid & input->direction) == input->direction)
    {
        // Same frame pair as the previous call
        return true;
    }

    if (obj->sig_next != input->a.sig) {
        invalidate_temporal(
            gpu, obj,
            "frame signature of first frame not equal to frame signature "
            "of second frame from previous call");
    }

    obj->sig_prev = input->a.sig;
    obj->sig_next = input->b.sig;

    const int levels = pyramid_num_levels(w, h, params);
    int refine[MOTION_MAX_REFINE_PASSES];
    const int num_refine = refine_plan(params, refine);

    // Upon cold start or after reset, build the image pyramid for frame `a`,
    // use cached pyramid otherwise.
    if (!obj->pyramid_valid) {
        if (!convert_to_luma(input->dispatch, &input->a,
                             obj->pyramid_prev.elem[0]))
        {
            PL_ERR(gpu, "pl_motion_estimate: failed converting frame a to "
                        "luma!");
            goto error;
        }
        if (!pl_pyramid_build(gpu, input->dispatch, &obj->pyramid_prev))
            goto error;
    }

    if (!convert_to_luma(input->dispatch, &input->b,
                         obj->scratch.input.elem[0]))
    {
        PL_ERR(gpu, "pl_motion_estimate: failed converting frame b to luma!");
        goto error;
    }
    if (!pl_pyramid_build(gpu, input->dispatch, &obj->scratch.input))
        goto error;

    void *tmp = pl_tmp(NULL);
    bool ok = true;
    for (int d = 0; d < NUM_DIRS; d++) {
        const enum pl_motion_direction dir = 1 << d;
        if (!(input->direction & dir))
            continue;

        pl_tex_pyramid *field = &obj->dir[d].coarse;

        enum motion_prior prior_kind = MOTION_PRIOR_SAME;
        pl_tex_pyramid *prior_field = field;
        if (d == FWD && (obj->valid & PL_MOTION_BACKWARD)) {
            prior_kind = MOTION_PRIOR_NEGATE;
            prior_field = &obj->dir[BWD].coarse;
        } else if (d == BWD && (input->direction & PL_MOTION_FORWARD)) {
            prior_kind = MOTION_PRIOR_INVERT;
            prior_field = &obj->dir[FWD].coarse;
        }

        const bool has_prior = prior_kind != MOTION_PRIOR_SAME ||
                               (obj->valid & dir);

        for (int i = levels - 1; i >= 0; i--) {
            // The top level has no vectors of a coarser level to start from,
            // so it compares the blocks without their means: a change in
            // brightness between the frames, as on a fade, otherwise outweighs
            // the match of the content. Its vectors then measure that change,
            // which the levels below compensate.
            const bool top = i == levels - 1;
            pl_tex gain = top ? NULL : obj->dir[d].gain;
            const bool coarsest = levels > 1 && top;
            const int radius = coarsest ? params->search_radius_coarse
                                        : params->search_radius_fine;

            pl_tex raw = obj->scratch.raw.elem[i];
            pl_tex out = field->elem[i];

            // The frames in the order of the direction
            pl_tex luma[2] = {
                obj->pyramid_prev.elem[i],
                obj->scratch.input.elem[i],
            };

            // Use the parent motion search to seed this levels search, for
            // all levels but the coarsest. The parent is the motion field
            // after smoothing.
            pl_tex parent =
                i < levels - 1 ? field->elem[i + 1] : NULL;

            // Seed the motion search using the prior at the corresponding
            // level of the motion search pyramid.
            pl_tex prior = has_prior ? prior_field->elem[i] : NULL;

            struct motion_geometry geom;
            ok = motion_plan(gpu, tmp, params->block_size_coarse, radius,
                             motion_seeds_plan(parent != NULL, prior != NULL),
                             i == 0, top, out->params.w, out->params.h,
                             &geom);
            if (!ok)
                break;

            const struct motion_search search = {
                .cand_lut = &obj->cand_lut[coarsest],
                .ref    = luma[d],
                .tgt    = luma[!d],
                .parent = parent,
                .prior  = prior,
                .prior_kind = prior_kind,
                .gain   = gain,
                .out_mv = raw,
                .geom   = &geom,
            };

            ok = motion_search_pass(gpu, input->dispatch, &search);
            if (!ok)
                break;

            ok = motion_smooth_pass(gpu, input->dispatch,
                                    params->block_size_coarse, luma[d],
                                    luma[!d], gain, raw, out);
            if (!ok)
                break;

            if (top && i > 0) {
                ok = motion_gain_pass(gpu, input->dispatch,
                                      params->block_size_coarse, luma[d],
                                      luma[!d], out, obj->dir[d].gain);
                if (!ok)
                    break;
            }
        }

        if (!ok)
            break;

        if (num_refine) {
            pl_tex luma[2] = {
                obj->pyramid_prev.elem[0],
                obj->scratch.input.elem[0],
            };
            // The gain is that of the levels below the top one, if any
            ok = refine_flow(gpu, input->dispatch, refine, &obj->dir[d].fine,
                             params->block_size_coarse, luma[d], luma[!d],
                             levels > 1 ? obj->dir[d].gain : NULL,
                             field->elem[0]);
            if (!ok)
                break;
        }

        // NOTE: motion estimation that succeeds on the forward motion
        // estimation call but fails on the backward motion estimation call
        // may leave sh_motion_obj in an incomplete state. we assume that if
        // the backward direction fails, the caller is not interested in the
        // forward direction results.

        if (input->features & PL_MOTION_GLOBAL) {
            ok = estimate_global(gpu, input->dispatch, obj, dir);
            if (!ok)
                break;
        }
    }
    pl_free(tmp);

    if (!ok)
        goto error;

    obj->valid = input->direction;

    if (input->features & PL_MOTION_QUALITY) {
        if (!estimate_quality(gpu, input->dispatch, obj,
                              obj->pyramid_prev.elem[0],
                              obj->scratch.input.elem[0]))
            goto error;
    }


    // Cache frame `b`'s luma pyramid, use reuse `a` buffers as scratch.
    PL_SWAP(obj->pyramid_prev, obj->scratch.input);
    obj->pyramid_valid = true;

    return true;

error:
    // Don't leave behind partially updated or allocated objects
    motion_uninit(gpu, obj);
    return false;
}

// Converts `color`, as sampled from `frame`, to `out_color` with premultiplied
// alpha.
static void interp_convert(pl_shader sh,
                           const struct pl_motion_estimate_frame *frame,
                           const struct pl_color_space *out_color)
{
    struct pl_color_repr repr = frame->repr;
    struct pl_color_space csp = frame->color;
    pl_color_space_infer(&csp);
    if (!pl_color_space_equal(&csp, out_color)) {
        pl_shader_set_alpha(sh, &repr, PL_ALPHA_INDEPENDENT);
        pl_shader_color_map_ex(sh, NULL, pl_color_map_args(csp, *out_color));
    }
    pl_shader_set_alpha(sh, &repr, PL_ALPHA_PREMULTIPLIED);
}

// Mixes the two frames, each sampled at the position offset by the global
// motion, scaled by the frame's distance in time to the intermediate frame.
static bool frame_interp_global(pl_shader sh,
                                const struct pl_frame_interp_input *input,
                                const struct pl_color_space *out_color)
{
    pl_shader_obj state = *input->state;
    if (!pl_motion_get_global_buf(state, PL_MOTION_FORWARD)) {
        PL_ERR(sh, "pl_shader_frame_interp: no global motion estimate "
               "available!");
        return false;
    }

    const struct pl_tex_params *tpars = &input->a.tex->params;
    if (!sh_require(sh, PL_SHADER_SIG_NONE, tpars->w, tpars->h))
        return false;

    sh_describe(sh, "frame interpolation (global motion)");
    ident_t gm = sh_motion_global(sh, state, PL_MOTION_FORWARD);
    pl_assert(gm);

    const float t = PL_CLAMP(input->t, 0.0f, 1.0f);
    const struct pl_motion_estimate_frame *frames[NUM_DIRS] = {
        &input->a, &input->b
    };

    // clang-format off
#pragma GLSL /* pl_shader_frame_interp (global motion) */                       \
    vec4 color;                                                                 \
    {                                                                           \
    vec4 mix_color = vec4(0.0);                                                 \
    vec4 mix_clamped = vec4(0.0);  /* fallback for when global motion offset is \
                                      out of frame for both frames */           \
    float mix_inside = 0.0;  /* sum of weights of the pixels with valid global  \
                                motion offset */                                \
    vec2 gm_pos;  /* pixel position offset by global motion, scaled by t */     \
    float gm_inside;  /* 0 if gm_pos is out of bounds, 1 otherwise */
    // clang-format on

    for (int i = 0; i < NUM_DIRS; i++) {
        pl_tex tex = frames[i]->tex;
        enum pl_tex_sample_mode sample_mode = PL_TEX_SAMPLE_NEAREST;
        if (tex->params.format->caps & PL_FMT_CAP_LINEAR)
            sample_mode = PL_TEX_SAMPLE_LINEAR; // offsets are fractional

        ident_t pos, pt, frame = sh_bind(sh, tex, PL_TEX_ADDRESS_CLAMP,
                                         sample_mode, "frame", NULL, &pos, &pt);
        if (!frame)
            return false;

        // A pixel in the intermediate frame finds its corresponding pixel in
        // frame `a` at (x,y) - t*gm, and in frame `b` at (x,y) + (1-t)*gm.
        const float offset = i == 0 ? -t : 1.0f - t;
        ident_t weight = SH_FLOAT_DYN(i == 0 ? 1.0f - t : t);

        // clang-format off
#pragma GLSL /* pl_shader_frame_interp (global motion) */                       \
        gm_pos = $pos + ${dynamic float: offset} * $gm.xy * $pt;                \
        gm_inside = all(greaterThanEqual(gm_pos, vec2(0.0))) &&                 \
                    all(lessThanEqual(gm_pos, vec2(1.0))) ? 1.0 : 0.0;          \
        color = textureLod($frame, gm_pos, 0.0);
        // clang-format on

        interp_convert(sh, frames[i], out_color);

        // clang-format off
#pragma GLSL /* pl_shader_frame_interp (global motion) */                       \
        mix_clamped += $weight * color;                                         \
        /* Discard pixels that are out of frame. */                             \
        mix_color += gm_inside * $weight * color;                               \
        mix_inside += gm_inside * $weight;
        // clang-format on
    }

    // clang-format off
#pragma GLSL /* pl_shader_frame_interp (global motion) */                       \
    /* Reweight contributions of pixels to only consider those that are in      \
     * frame. */                                                                \
    color = mix_inside > 0.0 ? mix_color / mix_inside : mix_clamped;            \
    }
    // clang-format on

    return true;
}

// Draws the forward flow with the Middlebury color wheel, desaturated towards
// gray where its quality is low
static bool frame_interp_visualize(pl_shader sh,
                                   const struct pl_frame_interp_input *input,
                                   const struct pl_color_space *out_color)
{
    const struct pl_tex_params *tpars = &input->a.tex->params;
    if (!sh_require(sh, PL_SHADER_SIG_NONE, tpars->w, tpars->h))
        return false;

    ident_t flow = sh_motion_flow(sh, *input->state, PL_MOTION_FORWARD);
    if (!flow) {
        PL_ERR(sh, "pl_shader_frame_interp: no flow field available!");
        return false;
    }

    pl_tex qtex = pl_motion_get_quality(*input->state, PL_MOTION_FORWARD);
    if (!qtex) {
        PL_ERR(sh, "pl_shader_frame_interp: no flow quality available!");
        return false;
    }

    ident_t quality = bind_tex_linear(sh, "motion_quality", qtex, NULL, NULL);
    ident_t pos, frame = sh_bind(sh, input->a.tex, PL_TEX_ADDRESS_CLAMP,
                                 PL_TEX_SAMPLE_NEAREST, "frame", NULL, &pos,
                                 NULL);
    if (!frame)
        return false;

    sh_describe(sh, "frame interpolation (visualize motion)");

    // clang-format off
#pragma GLSL /* pl_shader_frame_interp (visualize motion) */                    \
    vec4 color;                                                                 \
    {                                                                           \
    color = textureLod($frame, $pos, 0.0);
    // clang-format on

    interp_convert(sh, &input->a, out_color);

    // The wheel is piecewise linear between red, yellow, green, cyan, blue and
    // magenta, with segment lengths 15, 6, 4, 11, 13, 6. As in the original
    // implementation, the hue spans only [0, 54], leaving the last segment
    // incomplete.
    // clang-format off
#pragma GLSL /* pl_shader_frame_interp (visualize motion) */                    \
    vec4 frame_color = color;                                                   \
    const vec3 R = vec3(1.0, 0.0, 0.0), Y = vec3(1.0, 1.0, 0.0),                \
               G = vec3(0.0, 1.0, 0.0), C = vec3(0.0, 1.0, 1.0),                \
               B = vec3(0.0, 0.0, 1.0), M = vec3(1.0, 0.0, 1.0);                \
    vec2 uv = textureLod($flow, $pos, 0.0).xy;                                  \
    float rad = length(uv);                                                     \
    vec3 col = vec3(1.0);                                                       \
    if (rad > 0.0) {                                                            \
        float fk = (atan(-uv.y, -uv.x) / ${const float: M_PI} + 1.0) * 27.0;    \
        if (fk < 15.0)      col = mix(R, Y, fk / 15.0);                         \
        else if (fk < 21.0) col = mix(Y, G, (fk - 15.0) / 6.0);                 \
        else if (fk < 25.0) col = mix(G, C, (fk - 21.0) / 4.0);                 \
        else if (fk < 36.0) col = mix(C, B, (fk - 25.0) / 11.0);                \
        else if (fk < 49.0) col = mix(B, M, (fk - 36.0) / 13.0);                \
        else                col = mix(M, R, (fk - 49.0) / 6.0);                 \
        col = vec3(1.0) - rad / (rad + 4.0) * (vec3(1.0) - col);                \
    }                                                                           \
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));                        \
    col = mix(vec3(luma), col, textureLod($quality, $pos, 0.0).x);              \
    color = vec4(col, 1.0);
    // clang-format on

    struct pl_color_space csp = pl_color_space_srgb;
    pl_color_space_infer(&csp);
    pl_shader_color_map_ex(sh, NULL, pl_color_map_args(csp, *out_color));

    // clang-format off
#pragma GLSL /* pl_shader_frame_interp (visualize motion) */                    \
    color = mix(frame_color, color, 0.95);                                      \
    }
    // clang-format on

    return true;
}

static bool frame_interp_quality(pl_shader sh,
                                 const struct pl_frame_interp_input *input,
                                 const struct pl_color_space *out_color)
{
    const struct pl_tex_params *tpars = &input->a.tex->params;
    if (!sh_require(sh, PL_SHADER_SIG_NONE, tpars->w, tpars->h))
        return false;

    pl_tex qtex = pl_motion_get_quality(*input->state, PL_MOTION_FORWARD);
    if (!qtex) {
        PL_ERR(sh, "pl_shader_frame_interp: no flow quality available!");
        return false;
    }

    ident_t pos, quality = bind_tex_linear(sh, "motion_quality", qtex, &pos,
                                           NULL);
    if (!quality)
        return false;

    sh_describe(sh, "frame interpolation (visualize quality)");

    // clang-format off
#pragma GLSL /* pl_shader_frame_interp (visualize quality) */                   \
    float q = textureLod($quality, $pos, 0.0).x;                                \
    vec4 color = vec4(vec3(q), 1.0);
    // clang-format on

    struct pl_color_space csp = pl_color_space_srgb;
    pl_color_space_infer(&csp);
    pl_shader_color_map_ex(sh, NULL, pl_color_map_args(csp, *out_color));
    return true;
}

bool pl_shader_frame_interp(pl_shader sh,
                            const struct pl_frame_interp_input *input,
                            const struct pl_frame_interp_params *params)
{
    params = PL_DEF(params, &pl_frame_interp_default_params);
    if (!input || !input->state)
        return false;

    struct pl_motion_estimate_params me_params = params->motion;
    enum pl_motion_direction me_dir;
    enum pl_motion_feature me_features = 0;
    switch (params->algo) {
    case PL_FRAME_INTERP_GLOBAL_MOTION:
        me_dir = PL_MOTION_FORWARD;
        me_features = PL_MOTION_GLOBAL;
        // The global motion is estimated from the blocks, refining the
        // fields would be wasted
        me_params.block_size_fine = me_params.block_size_coarse;
        break;
    case PL_FRAME_INTERP_VISUALIZE:
    case PL_FRAME_INTERP_QUALITY:
        me_dir = PL_MOTION_FORWARD;
        me_features = PL_MOTION_QUALITY;
        break;
    case PL_FRAME_INTERP_ALGO_COUNT:
    default:
        PL_ERR(sh, "pl_shader_frame_interp: invalid algorithm %d!",
               (int) params->algo);
        return false;
    }

    // A no-op if the motion between this pair of frames is already known.
    bool ok = pl_motion_estimate(pl_motion_estimate_input(
        .gpu       = input->gpu,
        .dispatch  = input->dispatch,
        .a         = input->a,
        .b         = input->b,
        .direction = me_dir,
        .features  = me_features,
        .state     = input->state,
    ), &me_params);
    if (!ok)
        return false;

    struct pl_color_space out_color = input->out_color;
    pl_color_space_infer(&out_color);

    // Each algorithm must fail before modifying `sh` where possible, see
    // `pl_shader_frame_interp` documentation
    switch (params->algo) {
    case PL_FRAME_INTERP_GLOBAL_MOTION:
        return frame_interp_global(sh, input, &out_color);
    case PL_FRAME_INTERP_VISUALIZE:
        return frame_interp_visualize(sh, input, &out_color);
    case PL_FRAME_INTERP_QUALITY:
        return frame_interp_quality(sh, input, &out_color);
    case PL_FRAME_INTERP_ALGO_COUNT:
        break;
    }

    pl_unreachable(); // rejected above
}
