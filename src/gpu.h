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

#pragma once

#include "common.h"
#include <libplacebo/gpu.h>
#include <libplacebo/dispatch.h>

#define GPU_PFN(name) __typeof__(pl_##name) *name
struct pl_gpu_fns {
    // Destructors: These also free the corresponding objects, but they
    // must not be called on NULL. (The NULL checks are done by the pl_*_destroy
    // wrappers)
    void (*destroy)(const struct pl_gpu *gpu);
    void (*tex_destroy)(const struct pl_gpu *, const struct pl_tex *);
    void (*buf_destroy)(const struct pl_gpu *, const struct pl_buf *);
    void (*pass_destroy)(const struct pl_gpu *, const struct pl_pass *);

    GPU_PFN(tex_create);
    GPU_PFN(tex_invalidate);
    GPU_PFN(tex_clear);
    GPU_PFN(tex_blit);
    GPU_PFN(tex_upload);
    GPU_PFN(tex_download);
    GPU_PFN(buf_create);
    GPU_PFN(buf_write);
    GPU_PFN(buf_read);
    GPU_PFN(buf_poll); // optional: if NULL buffers are always free to use
    GPU_PFN(desc_namespace);
    GPU_PFN(pass_create);
    GPU_PFN(pass_run);
    GPU_PFN(gpu_flush); // optional
    GPU_PFN(gpu_finish);

    // The following functions are optional if the corresponding pl_limit
    // size restriction is 0
    GPU_PFN(buf_uniform_layout);
    GPU_PFN(buf_storage_layout);
    GPU_PFN(push_constant_layout);
};
#undef GPU_PFN

// All resources such as textures and buffers allocated from the GPU must be
// destroyed before calling pl_destroy.
void pl_gpu_destroy(const struct pl_gpu *gpu);

// Recreates a texture with new parameters, no-op if nothing changed
bool pl_tex_recreate(const struct pl_gpu *gpu, const struct pl_tex **tex,
                     const struct pl_tex_params *params);

// Incrementally build up a buffer by adding new variable elements to the
// buffer, resizing buf.buffer_vars if necessary. Returns whether or not the
// variable could be successfully added (which may fail if you try exceeding
// the size limits of the buffer type). If successful, the layout is stored
// in *out_layout
bool pl_buf_desc_append(void *tactx, const struct pl_gpu *gpu,
                        struct pl_desc *buf_desc,
                        struct pl_var_layout *out_layout,
                        const struct pl_var new_var);

size_t pl_buf_desc_size(const struct pl_desc *buf_desc);

// GPU-internal helpers: these should not be used outside of GPU implementations

// Log some metadata about the created GPU
void pl_gpu_print_info(const struct pl_gpu *gpu, enum pl_log_level lev);

// Sort the pl_format list into an optimal order. This tries to prefer formats
// supporting more capabilities, while also trying to maintain a sane order
// in terms of bit depth / component index.
void pl_gpu_sort_formats(struct pl_gpu *gpu);

// Pretty-print the format list
void pl_gpu_print_formats(const struct pl_gpu *gpu, enum pl_log_level lev);

// Look up the right GLSL image format qualifier from a partially filled-in
// pl_fmt, or NULL if the format does not have a legal matching GLSL name.
//
// Warning: If `fmt->emulated` is true, this function makes the hard assumption
// that 3-channel formats are being emulated as equivalent 4-channel formats!
const char *pl_fmt_glsl_format(const struct pl_fmt *fmt);

// Compute the total size (in bytes) of a texture transfer operation
size_t pl_tex_transfer_size(const struct pl_tex_transfer_params *par);

// Layout rules for GLSL's packing modes
struct pl_var_layout std140_layout(const struct pl_gpu *gpu, size_t offset,
                                   const struct pl_var *var);
struct pl_var_layout std430_layout(const struct pl_gpu *gpu, size_t offset,
                                   const struct pl_var *var);

// A hard-coded upper limit on a pl_buf_pool's size, to prevent OOM loops
#define PL_BUF_POOL_MAX_BUFFERS 8

// A pool of buffers, which can grow as needed
struct pl_buf_pool {
    struct pl_buf_params current_params;
    const struct pl_buf **buffers;
    int num_buffers;
    int index;
};

void pl_buf_pool_uninit(const struct pl_gpu *gpu, struct pl_buf_pool *pool);

// Note: params->initial_data is *not* supported
const struct pl_buf *pl_buf_pool_get(const struct pl_gpu *gpu,
                                     struct pl_buf_pool *pool,
                                     const struct pl_buf_params *params);

// Helper that wraps pl_tex_upload/download using texture upload buffers to
// ensure that params->buf is always set.
bool pl_tex_upload_pbo(const struct pl_gpu *gpu, struct pl_buf_pool *pbo,
                       const struct pl_tex_transfer_params *params);
bool pl_tex_download_pbo(const struct pl_gpu *gpu, struct pl_buf_pool *pbo,
                         const struct pl_tex_transfer_params *params);

// This requires that params.buf has been set and is of type PL_BUF_TEXEL_*
bool pl_tex_upload_texel(const struct pl_gpu *gpu, struct pl_dispatch *dp,
                         const struct pl_tex_transfer_params *params);
bool pl_tex_download_texel(const struct pl_gpu *gpu, struct pl_dispatch *dp,
                           const struct pl_tex_transfer_params *params);

// Make a deep-copy of the pass params. Note: cached_program etc. are not
// copied, but cleared explicitly.
struct pl_pass_params pl_pass_params_copy(void *tactx,
                                          const struct pl_pass_params *params);
