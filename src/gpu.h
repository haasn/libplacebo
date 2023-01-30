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
#include "log.h"

// To avoid having to include drm_fourcc.h
#ifndef DRM_FORMAT_MOD_LINEAR
#define DRM_FORMAT_MOD_LINEAR   UINT64_C(0x0)
#define DRM_FORMAT_MOD_INVALID  ((UINT64_C(1) << 56) - 1)
#endif

// This struct must be the first member of the gpu's priv struct. The `pl_gpu`
// helpers will cast the priv struct to this struct!

#define GPU_PFN(name) __typeof__(pl_##name) *name
struct pl_gpu_fns {
    // This is a pl_dispatch used (on the pl_gpu itself!) for the purposes of
    // dispatching compute shaders for performing various emulation tasks (e.g.
    // partial clears, blits or emulated texture transfers, see below).
    //
    // Warning: Care must be taken to avoid recursive calls.
    pl_dispatch dp;

    // Destructors: These also free the corresponding objects, but they
    // must not be called on NULL. (The NULL checks are done by the pl_*_destroy
    // wrappers)
    void (*destroy)(pl_gpu gpu);
    void (*tex_destroy)(pl_gpu, pl_tex);
    void (*buf_destroy)(pl_gpu, pl_buf);
    void (*pass_destroy)(pl_gpu, pl_pass);
    void (*sync_destroy)(pl_gpu, pl_sync);
    void (*timer_destroy)(pl_gpu, pl_timer);

    GPU_PFN(tex_create);
    GPU_PFN(tex_invalidate); // optional
    GPU_PFN(tex_clear_ex); // optional if no blittable formats
    GPU_PFN(tex_blit); // optional if no blittable formats
    GPU_PFN(tex_upload);
    GPU_PFN(tex_download);
    GPU_PFN(tex_poll); // optional: if NULL, textures are always free to use
    GPU_PFN(buf_create);
    GPU_PFN(buf_write);
    GPU_PFN(buf_read);
    GPU_PFN(buf_copy);
    GPU_PFN(buf_export); // optional if !gpu->export_caps.buf
    GPU_PFN(buf_poll); // optional: if NULL, buffers are always free to use
    GPU_PFN(desc_namespace);
    GPU_PFN(pass_create);
    GPU_PFN(pass_run);
    GPU_PFN(sync_create); // optional if !gpu->export_caps.sync
    GPU_PFN(tex_export); // optional if !gpu->export_caps.sync
    GPU_PFN(timer_create); // optional
    GPU_PFN(timer_query); // optional
    GPU_PFN(gpu_flush); // optional
    GPU_PFN(gpu_finish);
    GPU_PFN(gpu_is_failed); // optional
};
#undef GPU_PFN

// All resources such as textures and buffers allocated from the GPU must be
// destroyed before calling pl_destroy.
void pl_gpu_destroy(pl_gpu gpu);

// Returns true if the device supports interop. This is considered to be
// the case if at least one of `gpu->export/import_caps` is nonzero.
static inline bool pl_gpu_supports_interop(pl_gpu gpu)
{
    return gpu->export_caps.tex ||
           gpu->import_caps.tex ||
           gpu->export_caps.buf ||
           gpu->import_caps.buf ||
           gpu->export_caps.sync ||
           gpu->import_caps.sync;
}

// Returns the GPU-internal `pl_dispatch` object.
pl_dispatch pl_gpu_dispatch(pl_gpu gpu);

// GPU-internal helpers: these should not be used outside of GPU implementations

// This performs several tasks. It sorts the format list, logs GPU metadata,
// performs verification and fixes up backwards compatibility fields. This
// should be returned as the last step when creating a `pl_gpu`.
pl_gpu pl_gpu_finalize(struct pl_gpu_t *gpu);

// Look up the right GLSL image format qualifier from a partially filled-in
// pl_fmt, or NULL if the format does not have a legal matching GLSL name.
//
// `components` may differ from fmt->num_components (for emulated formats)
const char *pl_fmt_glsl_format(pl_fmt fmt, int components);

// Look up the right fourcc from a partially filled-in pl_fmt, or 0 if the
// format does not have a legal matching fourcc format.
uint32_t pl_fmt_fourcc(pl_fmt fmt);

// Compute the total size (in bytes) of a texture transfer operation
size_t pl_tex_transfer_size(const struct pl_tex_transfer_params *par);

// Helper that wraps pl_tex_upload/download using texture upload buffers to
// ensure that params->buf is always set.
bool pl_tex_upload_pbo(pl_gpu gpu, const struct pl_tex_transfer_params *params);
bool pl_tex_download_pbo(pl_gpu gpu, const struct pl_tex_transfer_params *params);

// This requires that params.buf has been set and is of type PL_BUF_TEXEL_*
bool pl_tex_upload_texel(pl_gpu gpu, const struct pl_tex_transfer_params *params);
bool pl_tex_download_texel(pl_gpu gpu, const struct pl_tex_transfer_params *params);

// Both `src` and `dst must be storable. `src` must also be sampleable, if the
// blit requires linear sampling. Returns false if these conditions are unmet.
bool pl_tex_blit_compute(pl_gpu gpu, const struct pl_tex_blit_params *params);

// Helper to do a 2D blit with stretch and scale using a raster pass
void pl_tex_blit_raster(pl_gpu gpu, const struct pl_tex_blit_params *params);

// Helper for GPU-accelerated endian swapping
//
// Note: `src` and `dst` can be the same buffer, for an in-place operation. In
// this case, `src_offset` and `dst_offset` must be the same.
struct pl_buf_copy_swap_params {
    // Source of the copy operation. Must be `storable`.
    pl_buf src;
    size_t src_offset;

    // Destination of the copy operation. Must be `storable`.
    pl_buf dst;
    size_t dst_offset;

    // Number of bytes to copy. Must be a multiple of 4.
    size_t size;

    // Underlying word size. Must be 2 (for 16-bit swap) or 4 (for 32-bit swap)
    int wordsize;
};

bool pl_buf_copy_swap(pl_gpu gpu, const struct pl_buf_copy_swap_params *params);

void pl_pass_run_vbo(pl_gpu gpu, const struct pl_pass_run_params *params);

// Make a deep-copy of the pass params. Note: cached_program etc. are not
// copied, but cleared explicitly.
struct pl_pass_params pl_pass_params_copy(void *alloc, const struct pl_pass_params *params);

// Helper to compute the size of an index buffer
static inline size_t pl_index_buf_size(const struct pl_pass_run_params *params)
{
    switch (params->index_fmt) {
    case PL_INDEX_UINT16: return params->vertex_count * sizeof(uint16_t);
    case PL_INDEX_UINT32: return params->vertex_count * sizeof(uint32_t);
    case PL_INDEX_FORMAT_COUNT: break;
    }

    pl_unreachable();
}

// Helper to compute the size of a vertex buffer required to fit all indices
size_t pl_vertex_buf_size(const struct pl_pass_run_params *params);

// Utility function for pretty-printing UUIDs
#define UUID_SIZE 16
#define PRINT_UUID(uuid) (print_uuid((char[3 * UUID_SIZE]){0}, (uuid)))
const char *print_uuid(char buf[3 * UUID_SIZE], const uint8_t uuid[UUID_SIZE]);

// Helper to pretty-print fourcc codes
#define PRINT_FOURCC(fcc)       \
    (!(fcc) ? "" : (char[5]) {  \
        (fcc) & 0xFF,           \
        ((fcc) >> 8) & 0xFF,    \
        ((fcc) >> 16) & 0xFF,   \
        ((fcc) >> 24) & 0xFF    \
    })

#define DRM_MOD_SIZE 26
#define PRINT_DRM_MOD(mod) (print_drm_mod((char[DRM_MOD_SIZE]){0}, (mod)))
const char *print_drm_mod(char buf[DRM_MOD_SIZE], uint64_t mod);
