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

#pragma once

#include "common.h"
#include "command.h"
#include "formats.h"
#include "malloc.h"
#include "utils.h"

#include "../gpu.h"
#include "../pl_thread.h"

pl_gpu pl_gpu_create_vk(struct vk_ctx *vk);

// This function takes the current graphics command and steals it from the
// GPU, so the caller can do custom vk_cmd_ calls on it. The caller should
// submit it as well.
struct vk_cmd *pl_vk_steal_cmd(pl_gpu gpu);

// Print memory usage statistics
void pl_vk_print_heap(pl_gpu, enum pl_log_level);

// --- pl_gpu internal structs and helpers

struct pl_fmt_vk {
    const struct vk_format *vk_fmt;
    bool blit_emulated;
};

enum queue_type {
    GRAPHICS,
    COMPUTE,
    TRANSFER,
    ANY,
};

struct pl_vk {
    struct pl_gpu_fns impl;
    struct vk_ctx *vk;
    struct spirv_compiler *spirv;

    // Some additional cached device limits and features checks
    uint32_t max_push_descriptors;
    size_t min_texel_alignment;
    bool host_query_reset;

    // This is a pl_dispatch used (on ourselves!) for the purposes of
    // dispatching compute shaders for performing various emulation tasks
    // (e.g. partial clears, blits or emulated texture transfers).
    // Warning: Care must be taken to avoid recursive calls.
    pl_dispatch dp;

    // The "currently recording" command. This will be queued and replaced by
    // a new command every time we need to "switch" between queue families.
    pl_mutex recording;
    struct vk_cmd *cmd;
    pl_timer cmd_timer;

    // Array of VkSamplers for every combination of sample/address modes
    VkSampler samplers[PL_TEX_SAMPLE_MODE_COUNT][PL_TEX_ADDRESS_MODE_COUNT];

    // To avoid spamming warnings
    bool warned_modless;
};

struct vk_cmd *_begin_cmd(pl_gpu, enum queue_type, const char *label, pl_timer);
void _end_cmd(pl_gpu, struct vk_cmd **, bool submit);

#define CMD_BEGIN(type)              _begin_cmd(gpu, type, __func__, NULL)
#define CMD_BEGIN_TIMED(type, timer) _begin_cmd(gpu, type, __func__, timer)
#define CMD_FINISH(cmd) _end_cmd(gpu, cmd, false)
#define CMD_SUBMIT(cmd) _end_cmd(gpu, cmd, true)

struct pl_tex_vk {
    pl_rc_t rc;
    bool external_img;
    enum queue_type transfer_queue;
    VkImageType type;
    VkImage img;
    struct vk_memslice mem;
    // cached properties
    VkFormat img_fmt;
    VkImageUsageFlags usage_flags;
    // for sampling
    VkImageView view;
    // for rendering
    VkFramebuffer framebuffer;
    // for vk_tex_upload/download fallback code
    pl_fmt texel_fmt;

    // synchronization and current state
    struct vk_sem sem;
    VkImageLayout layout;
    PL_ARRAY(pl_vulkan_sem) ext_deps; // external semaphore, not owned by the pl_tex
    pl_sync ext_sync; // indicates an exported image
    bool may_invalidate;
    bool held;
};

pl_tex vk_tex_create(pl_gpu, const struct pl_tex_params *);
void vk_tex_deref(pl_gpu, pl_tex);
void vk_tex_invalidate(pl_gpu, pl_tex);
void vk_tex_clear_ex(pl_gpu, pl_tex, const union pl_clear_color);
void vk_tex_blit(pl_gpu, const struct pl_tex_blit_params *);
bool vk_tex_upload(pl_gpu, const struct pl_tex_transfer_params *);
bool vk_tex_download(pl_gpu, const struct pl_tex_transfer_params *);
bool vk_tex_poll(pl_gpu, pl_tex, uint64_t timeout);
bool vk_tex_export(pl_gpu, pl_tex, pl_sync);
void vk_tex_barrier(pl_gpu, struct vk_cmd *, pl_tex, VkPipelineStageFlags,
                    VkAccessFlags, VkImageLayout, bool export);

struct pl_buf_vk {
    pl_rc_t rc;
    struct vk_memslice mem;
    enum queue_type update_queue;
    VkBufferView view; // for texel buffers

    // synchronization and current state
    struct vk_sem sem;
    bool exported;
    bool needs_flush;
};

pl_buf vk_buf_create(pl_gpu, const struct pl_buf_params *);
void vk_buf_deref(pl_gpu, pl_buf);
void vk_buf_write(pl_gpu, pl_buf, size_t offset, const void *src, size_t size);
bool vk_buf_read(pl_gpu, pl_buf, size_t offset, void *dst, size_t size);
void vk_buf_copy(pl_gpu, pl_buf dst, size_t dst_offset,
                 pl_buf src, size_t src_offset, size_t size);
bool vk_buf_export(pl_gpu, pl_buf);
bool vk_buf_poll(pl_gpu, pl_buf, uint64_t timeout);

// Helper to ease buffer barrier creation. (`offset` is relative to pl_buf)
void vk_buf_barrier(pl_gpu, struct vk_cmd *, pl_buf, VkPipelineStageFlags,
                    VkAccessFlags, size_t offset, size_t size, bool export);

// Flush visible writes to a buffer made by the API
void vk_buf_flush(pl_gpu, struct vk_cmd *, pl_buf, size_t offset, size_t size);

struct pl_pass_vk;

int vk_desc_namespace(pl_gpu, enum pl_desc_type);
pl_pass vk_pass_create(pl_gpu, const struct pl_pass_params *);
void vk_pass_destroy(pl_gpu, pl_pass);
void vk_pass_run(pl_gpu, const struct pl_pass_run_params *);

struct pl_sync_vk {
    pl_rc_t rc;
    VkSemaphore wait;
    VkSemaphore signal;
};

void vk_sync_deref(pl_gpu, pl_sync);
