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

#define RA_PFN(name) __typeof__(ra_##name) *name
struct ra_fns {
    // Destructors: These also free the corresponding objects, but they
    // must not be called on NULL. (The NULL checks are done by the ra_*_destroy
    // wrappers)
    void (*destroy)(const struct ra *ra);
    void (*tex_destroy)(const struct ra *, const struct ra_tex *);
    void (*buf_destroy)(const struct ra *, const struct ra_buf *);
    void (*pass_destroy)(const struct ra *, const struct ra_pass *);

    RA_PFN(tex_create);
    RA_PFN(tex_invalidate);
    RA_PFN(tex_clear);
    RA_PFN(tex_blit);
    RA_PFN(tex_upload);
    RA_PFN(tex_download);
    RA_PFN(buf_create);
    RA_PFN(buf_write);
    RA_PFN(buf_read);
    RA_PFN(buf_poll); // optional: if NULL buffers are always free to use
    RA_PFN(desc_namespace);
    RA_PFN(pass_create);
    RA_PFN(pass_run);
    RA_PFN(flush); // optional

    // The following functions are optional if the corresponding ra_limit
    // size restriction is 0
    RA_PFN(buf_uniform_layout);
    RA_PFN(buf_storage_layout);
    RA_PFN(push_constant_layout);
};
#undef RA_PFN

// All resources such as textures and buffers allocated from the RA must be
// destroyed before calling ra_destroy.
void ra_destroy(const struct ra *ra);

// Recreates a texture with new parameters, no-op if nothing changed
bool ra_tex_recreate(const struct ra *ra, const struct ra_tex **tex,
                     const struct ra_tex_params *params);

// RA-internal helpers: these should not be used outside of RA implementations

// Compute the total size (in bytes) of a texture transfer operation
size_t ra_tex_transfer_size(const struct ra_tex_transfer_params *par);

// Layout rules for GLSL's packing modes
struct ra_var_layout std140_layout(const struct ra *ra, size_t offset,
                                   const struct ra_var *var);
struct ra_var_layout std430_layout(const struct ra *ra, size_t offset,
                                   const struct ra_var *var);

// A pool of buffers, which can grow as needed
struct ra_buf_pool {
    struct ra_buf_params current_params;
    const struct ra_buf **buffers;
    int num_buffers;
    int index;
};

void ra_buf_pool_uninit(const struct ra *ra, struct ra_buf_pool *pool);

// Note: params->initial_data is *not* supported
const struct ra_buf *ra_buf_pool_get(const struct ra *ra,
                                     struct ra_buf_pool *pool,
                                     const struct ra_buf_params *params);

// Helper that wraps ra_tex_upload/download using texture upload buffers to
// ensure that params->buf is always set.
bool ra_tex_upload_pbo(const struct ra *ra, struct ra_buf_pool *pbo,
                       const struct ra_tex_transfer_params *params);
bool ra_tex_download_pbo(const struct ra *ra, struct ra_buf_pool *pbo,
                         const struct ra_tex_transfer_params *params);

struct ra_pass_params ra_pass_params_copy(void *tactx,
                                          const struct ra_pass_params *params);
