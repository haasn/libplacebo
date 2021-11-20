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

#ifndef LIBPLACEBO_DUMMY_H_
#define LIBPLACEBO_DUMMY_H_

#include <libplacebo/gpu.h>

PL_API_BEGIN

// The functions in this file allow creating and manipulating "dummy" contexts.
// A dummy context isn't actually mapped by the GPU, all data exists purely on
// the CPU. It also isn't capable of compiling or executing any shaders, any
// attempts to do so will simply fail.
//
// The main use case for this dummy context is for users who want to generate
// advanced shaders that depend on specific GLSL features or support for
// certain types of GPU resources (e.g. LUTs). This dummy context allows such
// shaders to be generated, with all of the referenced shader objects and
// textures simply containing their data in a host-accessible way.

struct pl_gpu_dummy_params {
    // These GPU parameters correspond to their equivalents in `pl_gpu`, and
    // must obey the same rules as documented there. The values from
    // `pl_gpu_dummy_default_params` are set to support pretty much everything
    // and are set for GLSL version 450.
    //
    // Individual fields such as `glsl.compute` or `glsl.version` description
    // can and should be overridden by the user based on their requirements.
    // Individual limits should ideally be set based on the corresponding
    // `glGet` queries etc.
    struct pl_glsl_version glsl;
    struct pl_gpu_limits limits;
};

#define PL_GPU_DUMMY_DEFAULTS                                           \
    .glsl = {                                                           \
        .version            = 450,                                      \
        .gles               = false,                                    \
        .vulkan             = false,                                    \
        .compute            = true,                                     \
        .max_shmem_size     = SIZE_MAX,                                 \
        .max_group_threads  = 1024,                                     \
        .max_group_size     = { 1024, 1024, 1024 },                     \
        .subgroup_size      = 32,                                       \
        .min_gather_offset  = INT16_MIN,                                \
        .max_gather_offset  = INT16_MAX,                                \
    },                                                                  \
    .limits = {                                                         \
        /* pl_gpu */                                                    \
        .callbacks          = false,                                    \
        .thread_safe        = true,                                     \
        /* pl_buf */                                                    \
        .max_buf_size       = SIZE_MAX,                                 \
        .max_ubo_size       = SIZE_MAX,                                 \
        .max_ssbo_size      = SIZE_MAX,                                 \
        .max_vbo_size       = SIZE_MAX,                                 \
        .max_mapped_size    = SIZE_MAX,                                 \
        .max_buffer_texels  = UINT64_MAX,                               \
        /* pl_tex */                                                    \
        .max_tex_1d_dim     = UINT32_MAX,                               \
        .max_tex_2d_dim     = UINT32_MAX,                               \
        .max_tex_3d_dim     = UINT32_MAX,                               \
        .buf_transfer       = true,                                     \
        .align_tex_xfer_pitch = 1,                                      \
        .align_tex_xfer_offset = 1,                                     \
        /* pl_pass */                                                   \
        .max_variable_comps = SIZE_MAX,                                 \
        .max_constants      = SIZE_MAX,                                 \
        .max_pushc_size     = SIZE_MAX,                                 \
        .max_dispatch       = { UINT32_MAX, UINT32_MAX, UINT32_MAX },   \
        .fragment_queues    = 0,                                        \
        .compute_queues     = 0,                                        \
    },

#define pl_gpu_dummy_params(...) (&(struct pl_gpu_dummy_params) { PL_GPU_DUMMY_DEFAULTS __VA_ARGS__ })
extern const struct pl_gpu_dummy_params pl_gpu_dummy_default_params;

// Create a dummy GPU context based on the given parameters. This GPU will have
// a format for each host-representable type (i.e. intN_t, floats and doubles),
// in the canonical channel order RGBA. These formats will have every possible
// capability activated, respectively.
//
// If `params` is left as NULL, it defaults to `&pl_gpu_dummy_params`.
pl_gpu pl_gpu_dummy_create(pl_log log, const struct pl_gpu_dummy_params *params);
void pl_gpu_dummy_destroy(pl_gpu *gpu);

// Back-doors into the `pl_tex` and `pl_buf` representations. These allow you
// to access the raw data backing this object. Textures are always laid out in
// a tightly packed manner.
//
// For "placeholder" dummy textures, this always returns NULL.
uint8_t *pl_buf_dummy_data(pl_buf buf);
uint8_t *pl_tex_dummy_data(pl_tex tex);

// Skeleton of `pl_tex_params` containing only the fields relevant to
// `pl_tex_dummy_create`, plus the extra `sampler_type` field.
struct pl_tex_dummy_params {
    int w, h, d;
    pl_fmt format;
    enum pl_sampler_type sampler_type;
    void *user_data;
};

#define pl_tex_dummy_params(...) (&(struct pl_tex_dummy_params) { __VA_ARGS__ })

// Allows creating a "placeholder" dummy texture. This is basically a texture
// that isn't even backed by anything. All `pl_tex_*` operations (other than
// `pl_tex_destroy`) performed on it will simply fail.
//
// All of the permissions will be set to `false`, except `sampleable`, which is
// set to `true`. (So you can use it as an input to shader sampling functions)
pl_tex pl_tex_dummy_create(pl_gpu gpu, const struct pl_tex_dummy_params *params);

PL_API_END

#endif // LIBPLACEBO_DUMMY_H_
