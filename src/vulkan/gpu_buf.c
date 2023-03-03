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

#include "gpu.h"

void vk_buf_barrier(pl_gpu gpu, struct vk_cmd *cmd, pl_buf buf,
                    VkPipelineStageFlags stage, VkAccessFlags access,
                    size_t offset, size_t size, bool export)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    pl_assert(!export || !buf_vk->exported); // can't re-export exported buffers
    pl_rc_ref(&buf_vk->rc);

    bool needs_flush = buf_vk->needs_flush || buf->params.host_mapped ||
                       buf->params.import_handle == PL_HANDLE_HOST_PTR;
    bool noncoherent = buf_vk->mem.data && !buf_vk->mem.coherent;
    if (needs_flush && noncoherent) {
        VK(vk->FlushMappedMemoryRanges(vk->dev, 1, &(struct VkMappedMemoryRange) {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = buf_vk->mem.vkmem,
            .offset = buf_vk->mem.map_offset,
            .size = buf_vk->mem.map_size,
        }));

        // Just ignore errors, not much we can do about them other than
        // logging them and moving on...
    error: ;
    }

    struct vk_sync_scope last;
    last = vk_sem_barrier(vk, cmd, &buf_vk->sem, stage, access, export);

    if (needs_flush) {
        last.access |= VK_ACCESS_HOST_WRITE_BIT;
        last.stage  |= VK_PIPELINE_STAGE_HOST_BIT;
    }

    if (needs_flush || buf_vk->mem.data) {
        last.access |= VK_ACCESS_HOST_READ_BIT;
        last.stage  |= VK_PIPELINE_STAGE_HOST_BIT;
    }

    // CONCURRENT buffers require transitioning to/from IGNORED, EXCLUSIVE
    // buffers require transitioning to/from the concrete QF index
    uint32_t qf = vk->pools.num > 1 ? VK_QUEUE_FAMILY_IGNORED : cmd->pool->qf;
    VkBufferMemoryBarrier barr = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcQueueFamilyIndex = buf_vk->exported ? VK_QUEUE_FAMILY_EXTERNAL_KHR : qf,
        .dstQueueFamilyIndex = export ? VK_QUEUE_FAMILY_EXTERNAL_KHR : qf,
        .srcAccessMask = last.access,
        .dstAccessMask = access,
        .buffer = buf_vk->mem.buf,
        .offset = buf_vk->mem.offset + offset,
        .size = size,
    };

    if (last.access || barr.srcQueueFamilyIndex != barr.dstQueueFamilyIndex) {
        vk->CmdPipelineBarrier(cmd->buf, last.stage, stage, 0, 0, NULL,
                               1, &barr, 0, NULL);
    }

    buf_vk->needs_flush = false;
    buf_vk->exported = export;
    vk_cmd_callback(cmd, (vk_cb) vk_buf_deref, gpu, buf);
}

void vk_buf_deref(pl_gpu gpu, pl_buf buf)
{
    if (!buf)
        return;

    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);

    if (pl_rc_deref(&buf_vk->rc)) {
        vk->DestroyBufferView(vk->dev, buf_vk->view, PL_VK_ALLOC);
        vk_malloc_free(vk->ma, &buf_vk->mem);
        pl_free((void *) buf);
    }
}

pl_buf vk_buf_create(pl_gpu gpu, const struct pl_buf_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    struct pl_buf_t *buf = pl_zalloc_obj(NULL, buf, struct pl_buf_vk);
    buf->params = *params;
    buf->params.initial_data = NULL;

    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    pl_rc_init(&buf_vk->rc);
    vk_sem_init(&buf_vk->sem);

    struct vk_malloc_params mparams = {
        .reqs = {
            .size = PL_ALIGN2(params->size, 4), // for vk_buf_write
            .memoryTypeBits = UINT32_MAX,
            .alignment = 1,
        },
        // these are always set, because `vk_buf_copy` can always be used
        .buf_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .export_handle = params->export_handle,
        .import_handle = params->import_handle,
        .shared_mem = params->shared_mem,
        .debug_tag = params->debug_tag,
    };

    // Mandatory/optimal buffer offset alignment
    VkDeviceSize *align = &mparams.reqs.alignment;
    VkDeviceSize extra_align = vk->limits.optimalBufferCopyOffsetAlignment;

    // Try and align all buffers to the minimum texel alignment, to make sure
    // tex_upload/tex_download always gets aligned buffer copies if possible
    extra_align = pl_lcm(extra_align, p->min_texel_alignment);

    enum pl_buf_mem_type mem_type = params->memory_type;
    bool is_texel = false;

    if (params->uniform) {
        mparams.buf_usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        *align = pl_lcm(*align, vk->limits.minUniformBufferOffsetAlignment);
        mem_type = PL_BUF_MEM_DEVICE;
        if (params->format) {
            mparams.buf_usage |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
            is_texel = true;
        }
    }

    if (params->storable) {
        mparams.buf_usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        *align = pl_lcm(*align, vk->limits.minStorageBufferOffsetAlignment);
        buf_vk->update_queue = COMPUTE;
        mem_type = PL_BUF_MEM_DEVICE;
        if (params->format) {
            mparams.buf_usage |= VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
            is_texel = true;
        }
    }

    if (is_texel) {
        *align = pl_lcm(*align, vk->limits.minTexelBufferOffsetAlignment);
        *align = pl_lcm(*align, params->format->texel_size);
    }

    if (params->drawable) {
        mparams.buf_usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                             VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        mem_type = PL_BUF_MEM_DEVICE;
    }

    if (params->host_writable || params->initial_data) {
        // Buffers should be written using mapped memory if possible
        mparams.optimal = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        // Use the transfer queue for updates on very large buffers (1 MB)
        if (params->size > 1024*1024)
            buf_vk->update_queue = TRANSFER;
    }

    if (params->host_mapped || params->host_readable) {
        mparams.required |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

        if (params->size > 1024) {
            // Prefer cached memory for large buffers (1 kB) which may be read
            // from, because uncached reads are extremely slow
            mparams.optimal |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        }
    }

    switch (mem_type) {
    case PL_BUF_MEM_AUTO:
        // We generally prefer VRAM since it's faster than RAM, but any number
        // of other requirements could potentially exclude it, so just mark it
        // as optimal by default.
        if (!(mparams.optimal & VK_MEMORY_PROPERTY_HOST_CACHED_BIT))
            mparams.optimal |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case PL_BUF_MEM_DEVICE:
        // Force device local memory.
        mparams.required |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case PL_BUF_MEM_HOST:
        // This isn't a true guarantee, but actually trying to restrict the
        // device-local bit locks out all memory heaps on iGPUs. Requiring
        // the memory be host-mapped is the easiest compromise.
        mparams.required |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        mparams.optimal  |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        break;
    case PL_BUF_MEM_TYPE_COUNT:
        pl_unreachable();
    }

    if (params->import_handle) {
        size_t offset = params->shared_mem.offset;
        if (PL_ALIGN(offset, *align) != offset) {
            PL_ERR(gpu, "Imported memory offset %zu violates minimum alignment "
                   "requirement of enabled usage flags (%zu)!",
                   offset, (size_t) *align);
            goto error;
        }
    } else {
        *align = pl_lcm(*align, extra_align);
    }

    if (!vk_malloc_slice(vk->ma, &buf_vk->mem, &mparams))
        goto error;

    if (params->host_mapped)
        buf->data = buf_vk->mem.data;

    if (params->export_handle) {
        buf->shared_mem = buf_vk->mem.shared_mem;
        buf->shared_mem.drm_format_mod = DRM_FORMAT_MOD_LINEAR;
        buf_vk->exported = true;
    }

    if (is_texel) {
        struct pl_fmt_vk *fmtp = PL_PRIV(params->format);
        VkBufferViewCreateInfo vinfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
            .buffer = buf_vk->mem.buf,
            .format = PL_DEF(fmtp->vk_fmt->bfmt, fmtp->vk_fmt->tfmt),
            .offset = buf_vk->mem.offset,
            .range = buf_vk->mem.size,
        };

        VK(vk->CreateBufferView(vk->dev, &vinfo, PL_VK_ALLOC, &buf_vk->view));
        PL_VK_NAME(BUFFER_VIEW, buf_vk->view, PL_DEF(params->debug_tag, "texel"));
    }

    if (params->initial_data)
        vk_buf_write(gpu, buf, 0, params->initial_data, params->size);

    return buf;

error:
    vk_buf_deref(gpu, buf);
    return NULL;
}

static void invalidate_buf(pl_gpu gpu, pl_buf buf)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);

    if (buf_vk->mem.data && !buf_vk->mem.coherent) {
        VK(vk->InvalidateMappedMemoryRanges(vk->dev, 1, &(VkMappedMemoryRange) {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = buf_vk->mem.vkmem,
            .offset = buf_vk->mem.map_offset,
            .size = buf_vk->mem.map_size,
        }));
    }

    // Ignore errors (after logging), nothing useful we can do anyway
error: ;
    vk_buf_deref(gpu, buf);
}

void vk_buf_flush(pl_gpu gpu, struct vk_cmd *cmd, pl_buf buf,
                  size_t offset, size_t size)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);

    // We need to perform a flush if the host is capable of reading back from
    // the buffer, or if we intend to overwrite it using mapped memory
    bool can_read = buf->params.host_readable;
    bool can_write = buf_vk->mem.data && buf->params.host_writable;
    if (buf->params.host_mapped || buf->params.import_handle == PL_HANDLE_HOST_PTR)
        can_read = can_write = true;

    if (!can_read && !can_write)
        return;

    VkBufferMemoryBarrier buffBarrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .srcAccessMask = buf_vk->sem.write.access,
        .dstAccessMask = (can_read ? VK_ACCESS_HOST_READ_BIT : 0)
                       | (can_write ? VK_ACCESS_HOST_WRITE_BIT : 0),
        .buffer = buf_vk->mem.buf,
        .offset = buf_vk->mem.offset + offset,
        .size = size,
    };

    vk->CmdPipelineBarrier(cmd->buf, buf_vk->sem.write.stage,
                           VK_PIPELINE_STAGE_HOST_BIT, 0,
                           0, NULL, 1, &buffBarrier, 0, NULL);

    // We need to hold on to the buffer until this barrier completes
    vk_cmd_callback(cmd, (vk_cb) invalidate_buf, gpu, buf);
    pl_rc_ref(&buf_vk->rc);
}

bool vk_buf_poll(pl_gpu gpu, pl_buf buf, uint64_t timeout)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);

    // Opportunistically check if we can re-use this buffer without flush
    vk_poll_commands(vk, 0);
    if (pl_rc_count(&buf_vk->rc) == 1)
        return false;

    // Otherwise, we're force to submit any queued command so that the
    // user is guaranteed to see progress eventually, even if they call
    // this in a tight loop
    CMD_SUBMIT(NULL);
    vk_poll_commands(vk, timeout);

    return pl_rc_count(&buf_vk->rc) > 1;
}

void vk_buf_write(pl_gpu gpu, pl_buf buf, size_t offset,
                  const void *data, size_t size)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);

    // For host-mapped buffers, we can just directly memcpy the buffer contents.
    // Otherwise, we can update the buffer from the GPU using a command buffer.
    if (buf_vk->mem.data) {
        // ensure no queued operations
        while (vk_buf_poll(gpu, buf, UINT64_MAX))
            ; // do nothing

        uintptr_t addr = (uintptr_t) buf_vk->mem.data + offset;
        memcpy((void *) addr, data, size);
        buf_vk->needs_flush = true;
    } else {
        struct vk_cmd *cmd = CMD_BEGIN(buf_vk->update_queue);
        if (!cmd) {
            PL_ERR(gpu, "Failed updating buffer!");
            return;
        }

        vk_buf_barrier(gpu, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_WRITE_BIT, offset, size, false);

        // Vulkan requires `size` to be a multiple of 4, so we need to make
        // sure to handle the end separately if the original data is not
        const size_t max_transfer = 64 * 1024;
        size_t size_rem = size % 4;
        size_t size_base = size - size_rem;
        VkDeviceSize buf_offset = buf_vk->mem.offset + offset;

        if (size_base > max_transfer) {
            PL_TRACE(gpu, "Using multiple vkCmdUpdateBuffer calls to upload "
                     "large buffer. Consider using buffer-buffer transfers "
                     "instead!");
        }

        for (size_t xfer = 0; xfer < size_base; xfer += max_transfer) {
            vk->CmdUpdateBuffer(cmd->buf, buf_vk->mem.buf,
                                buf_offset + xfer,
                                PL_MIN(size_base, max_transfer),
                                (void *) ((uint8_t *) data + xfer));
        }

        if (size_rem) {
            uint8_t tail[4] = {0};
            memcpy(tail, data, size_rem);
            vk->CmdUpdateBuffer(cmd->buf, buf_vk->mem.buf, buf_offset + size_base,
                                sizeof(tail), tail);
        }

        pl_assert(!buf->params.host_readable); // no flush needed due to this
        CMD_FINISH(&cmd);
    }
}

bool vk_buf_read(pl_gpu gpu, pl_buf buf, size_t offset, void *dest, size_t size)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    pl_assert(buf_vk->mem.data);

    if (vk_buf_poll(gpu, buf, 0) && buf_vk->sem.write.sync.sem) {
        // ensure no more queued writes
        VK(vk->WaitSemaphoresKHR(vk->dev, &(VkSemaphoreWaitInfo) {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .semaphoreCount = 1,
            .pSemaphores = &buf_vk->sem.write.sync.sem,
            .pValues = &buf_vk->sem.write.sync.value,
        }, UINT64_MAX));

        // process callbacks
        vk_poll_commands(vk, 0);
    }

    uintptr_t addr = (uintptr_t) buf_vk->mem.data + (size_t) offset;
    memcpy(dest, (void *) addr, size);
    return true;

error:
    return false;
}

void vk_buf_copy(pl_gpu gpu, pl_buf dst, size_t dst_offset,
                 pl_buf src, size_t src_offset, size_t size)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_buf_vk *dst_vk = PL_PRIV(dst);
    struct pl_buf_vk *src_vk = PL_PRIV(src);

    struct vk_cmd *cmd = CMD_BEGIN(dst_vk->update_queue);
    if (!cmd) {
        PL_ERR(gpu, "Failed copying buffer!");
        return;
    }

    vk_buf_barrier(gpu, cmd, dst, VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_ACCESS_TRANSFER_WRITE_BIT, dst_offset, size, false);
    vk_buf_barrier(gpu, cmd, src, VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_ACCESS_TRANSFER_READ_BIT, src_offset, size, false);

    VkBufferCopy region = {
        .srcOffset = src_vk->mem.offset + src_offset,
        .dstOffset = dst_vk->mem.offset + dst_offset,
        .size = size,
    };

    vk->CmdCopyBuffer(cmd->buf, src_vk->mem.buf, dst_vk->mem.buf,
                      1, &region);

    vk_buf_flush(gpu, cmd, dst, dst_offset, size);
    CMD_FINISH(&cmd);
}

bool vk_buf_export(pl_gpu gpu, pl_buf buf)
{
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    if (buf_vk->exported)
        return true;

    struct vk_cmd *cmd = CMD_BEGIN(ANY);
    if (!cmd) {
        PL_ERR(gpu, "Failed exporting buffer!");
        return false;
    }

    // For the queue family ownership transfer, we can ignore all pipeline
    // stages since the synchronization via fences/semaphores is required
    vk_buf_barrier(gpu, cmd, buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                   0, buf->params.size, true);


    return CMD_SUBMIT(&cmd);
}
