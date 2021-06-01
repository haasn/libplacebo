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

static void vk_buf_finish_write(pl_gpu gpu, pl_buf buf)
{
    if (!buf)
        return;

    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    buf_vk->writes--;
}

void vk_buf_barrier(pl_gpu gpu, struct vk_cmd *cmd, pl_buf buf,
                    VkPipelineStageFlags stage, VkAccessFlags newAccess,
                    size_t offset, size_t size, enum buffer_op op)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    pl_rc_ref(&buf_vk->rc);

    // CONCURRENT buffers require transitioning to/from IGNORED, EXCLUSIVE
    // buffers require transitioning to/from the concrete QF index
    uint32_t qf = vk->pools.num > 1 ? VK_QUEUE_FAMILY_IGNORED : cmd->pool->qf;

    VkBufferMemoryBarrier buffBarrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcQueueFamilyIndex = buf_vk->exported ? VK_QUEUE_FAMILY_EXTERNAL_KHR : qf,
        .dstQueueFamilyIndex = (op & BUF_EXPORT) ? VK_QUEUE_FAMILY_EXTERNAL_KHR : qf,
        .srcAccessMask = buf_vk->current_access,
        .dstAccessMask = newAccess,
        .buffer = buf_vk->mem.buf,
        .offset = buf_vk->mem.offset + offset,
        .size = size,
    };

    // Can't re-export exported buffers
    pl_assert(!(op & BUF_EXPORT) || !buf_vk->exported);

    VkEvent event = VK_NULL_HANDLE;
    enum vk_wait_type type = vk_cmd_wait(vk, cmd, &buf_vk->sig, stage, &event);
    VkPipelineStageFlags src_stages = 0;

    if (buf_vk->needs_flush || buf->params.host_mapped ||
        buf->params.import_handle == PL_HANDLE_HOST_PTR)
    {
        if (!buf_vk->exported) {
            buffBarrier.srcAccessMask |= VK_ACCESS_HOST_WRITE_BIT;
            src_stages |= VK_PIPELINE_STAGE_HOST_BIT;
        }

        if (buf_vk->mem.data && !buf_vk->mem.coherent) {
            if (buf_vk->exported) {
                // TODO: figure out and clean up the semantics?
                PL_WARN(vk, "Mixing host-mapped or user-writable buffers with "
                        "external APIs is risky and untested. If you run into "
                        "any issues, please try using a non-mapped buffer and "
                        "avoid pl_buf_write.");
            }

            VK(vk->FlushMappedMemoryRanges(vk->dev, 1, &(struct VkMappedMemoryRange) {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = buf_vk->mem.vkmem,
                .offset = buf_vk->mem.offset,
                .size = buf_vk->mem.size,
            }));

            // Just ignore errors, not much we can do about them other than
            // logging them and moving on...
        error: ;
        }

        // Forcibly degrade to non-event based pipeline barrier, because
        // mixing events with host writes is nonsensical
        if (type == VK_WAIT_EVENT)
            type = VK_WAIT_BARRIER;

        buf_vk->needs_flush = false;
    }

    if (buffBarrier.srcAccessMask != buffBarrier.dstAccessMask ||
        buffBarrier.srcQueueFamilyIndex != buffBarrier.dstQueueFamilyIndex)
    {
        switch (type) {
        case VK_WAIT_NONE:
            // No synchronization required, so we can safely transition out of
            // VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            buffBarrier.srcAccessMask = 0;
            src_stages |= VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            vk->CmdPipelineBarrier(cmd->buf, src_stages, stage, 0, 0, NULL,
                                   1, &buffBarrier, 0, NULL);
            break;
        case VK_WAIT_BARRIER:
            // Regular pipeline barrier is required
            vk->CmdPipelineBarrier(cmd->buf, buf_vk->sig_stage | src_stages,
                                   stage, 0, 0, NULL, 1, &buffBarrier, 0, NULL);
            break;
        case VK_WAIT_EVENT:
            // We can/should use the VkEvent for synchronization
            pl_assert(!src_stages);
            vk->CmdWaitEvents(cmd->buf, 1, &event, buf_vk->sig_stage,
                              stage, 0, NULL, 1, &buffBarrier, 0, NULL);
            break;
        }
    }

    if (op & BUF_WRITE) {
        buf_vk->writes++;
        vk_cmd_callback(cmd, (vk_cb) vk_buf_finish_write, gpu, buf);
    }

    buf_vk->current_access = newAccess;
    buf_vk->exported = (op & BUF_EXPORT);
    vk_cmd_callback(cmd, (vk_cb) vk_buf_deref, gpu, buf);
    vk_cmd_obj(cmd, buf);
}

void vk_buf_signal(pl_gpu gpu, struct vk_cmd *cmd, pl_buf buf,
                   VkPipelineStageFlags stage)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    pl_assert(!buf_vk->sig);

    buf_vk->sig = vk_cmd_signal(p->vk, cmd, stage);
    buf_vk->sig_stage = stage;
}

void vk_buf_deref(pl_gpu gpu, pl_buf buf)
{
    if (!buf)
        return;

    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);

    if (pl_rc_deref(&buf_vk->rc)) {
        vk_signal_destroy(vk, &buf_vk->sig);
        vk->DestroyBufferView(vk->dev, buf_vk->view, PL_VK_ALLOC);
        vk_malloc_free(p->alloc, &buf_vk->mem);
        pl_free((void *) buf);
    }
}

pl_buf vk_buf_create(pl_gpu gpu, const struct pl_buf_params *params)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;

    struct pl_buf *buf = pl_zalloc_obj(NULL, buf, struct pl_buf_vk);
    buf->params = *params;
    buf->params.initial_data = NULL;

    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    buf_vk->current_access = 0;
    pl_rc_init(&buf_vk->rc);

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
            *align = pl_lcm(*align, vk->limits.minTexelBufferOffsetAlignment);
            is_texel = true;
        }
    }

    if (params->storable) {
        mparams.buf_usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        *align = pl_lcm(*align, vk->limits.minStorageBufferOffsetAlignment);
        buf_vk->update_queue = vk->pool_compute ? COMPUTE : GRAPHICS;
        mem_type = PL_BUF_MEM_DEVICE;
        if (params->format) {
            mparams.buf_usage |= VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
            *align = pl_lcm(*align, vk->limits.minTexelBufferOffsetAlignment);
            is_texel = true;
        }
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
            // Require cached memory for large buffers (1 kB) which may be read
            // from, because uncached reads are extremely slow
            mparams.required |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        }
    }

    if (params->host_writable || params->host_readable) {
        // Prefer buffers requiring frequent host operations in host mem
        mem_type = PL_DEF(mem_type, PL_BUF_MEM_HOST);
    }

    switch (mem_type) {
    case PL_BUF_MEM_AUTO:
        // We generally prefer VRAM since it's faster than RAM, but any number
        // of other requirements could potentially exclude it, so just mark it
        // as optimal by default.
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

    if (!vk_malloc_slice(p->alloc, &buf_vk->mem, &mparams))
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
        PL_VK_NAME(BUFFER_VIEW, buf_vk->view, "texel");
    }

    if (params->initial_data)
        vk_buf_write(gpu, buf, 0, params->initial_data, params->size);

    return buf;

error:
    vk_buf_deref(gpu, buf);
    return NULL;
}

static void invalidate_memslice(struct vk_ctx *vk, const struct vk_memslice *mem)
{
    VK(vk->InvalidateMappedMemoryRanges(vk->dev, 1, &(VkMappedMemoryRange) {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = mem->vkmem,
        .offset = mem->offset,
        .size = mem->size,
    }));

    // Ignore errors (after logging), nothing useful we can do anyway
error: ;
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
        .srcAccessMask = buf_vk->current_access,
        .dstAccessMask = (can_read ? VK_ACCESS_HOST_READ_BIT : 0)
                       | (can_write ? VK_ACCESS_HOST_WRITE_BIT : 0),
        .buffer = buf_vk->mem.buf,
        .offset = buf_vk->mem.offset + offset,
        .size = size,
    };

    vk->CmdPipelineBarrier(cmd->buf, buf_vk->sig_stage,
                           VK_PIPELINE_STAGE_HOST_BIT, 0,
                           0, NULL, 1, &buffBarrier, 0, NULL);

    // Invalidate the mapped memory as soon as this barrier completes
    if (buf_vk->mem.data && !buf_vk->mem.coherent)
        vk_cmd_callback(cmd, (vk_cb) invalidate_memslice, vk, &buf_vk->mem);
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

    // Otherwise, we're force to submit all queued commands so that the
    // user is guaranteed to see progress eventually, even if they call
    // this in a tight loop
    CMD_SUBMIT(NULL);
    vk_flush_obj(vk, buf);
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
                       VK_ACCESS_TRANSFER_WRITE_BIT, offset, size, BUF_WRITE);

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
        vk_buf_signal(gpu, cmd, buf, VK_PIPELINE_STAGE_TRANSFER_BIT);
        CMD_FINISH(&cmd);
    }
}

bool vk_buf_read(pl_gpu gpu, pl_buf buf, size_t offset, void *dest, size_t size)
{
    struct pl_buf_vk *buf_vk = PL_PRIV(buf);
    pl_assert(buf_vk->mem.data);

    // ensure no more queued writes
    while (buf_vk->writes)
        vk_buf_poll(gpu, buf, UINT64_MAX);

    uintptr_t addr = (uintptr_t) buf_vk->mem.data + (size_t) offset;
    memcpy(dest, (void *) addr, size);
    return true;
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
                   VK_ACCESS_TRANSFER_WRITE_BIT, dst_offset, size, BUF_WRITE);
    vk_buf_barrier(gpu, cmd, src, VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_ACCESS_TRANSFER_READ_BIT, src_offset, size, BUF_READ);

    VkBufferCopy region = {
        .srcOffset = src_vk->mem.offset + src_offset,
        .dstOffset = dst_vk->mem.offset + dst_offset,
        .size = size,
    };

    vk->CmdCopyBuffer(cmd->buf, src_vk->mem.buf, dst_vk->mem.buf,
                      1, &region);

    vk_buf_signal(gpu, cmd, src, VK_PIPELINE_STAGE_TRANSFER_BIT);
    vk_buf_signal(gpu, cmd, dst, VK_PIPELINE_STAGE_TRANSFER_BIT);
    vk_buf_flush(gpu, cmd, dst, dst_offset, size);
    CMD_FINISH(&cmd);
}

bool vk_buf_export(pl_gpu gpu, pl_buf buf)
{
    struct pl_vk *p = PL_PRIV(gpu);
    struct vk_ctx *vk = p->vk;
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
                   0, buf->params.size, BUF_EXPORT);


    CMD_SUBMIT(&cmd);
    return vk_flush_commands(vk);
}
