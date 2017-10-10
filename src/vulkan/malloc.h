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

// All memory allocated from a vk_malloc MUST be explicitly released by
// the caller before vk_malloc_destroy is called.
struct vk_malloc *vk_malloc_create(struct vk_ctx *vk);
void vk_malloc_destroy(struct vk_malloc **ma);

// Represents a single "slice" of generic (non-buffer) memory, plus some
// metadata for accounting. This struct is essentially read-only.
struct vk_memslice {
    VkDeviceMemory vkmem;
    size_t offset;
    size_t size;
    void *priv;
};

void vk_free_memslice(struct vk_malloc *ma, struct vk_memslice slice);
bool vk_malloc_generic(struct vk_malloc *ma, VkMemoryRequirements reqs,
                       VkMemoryPropertyFlags flags, struct vk_memslice *out);

// Represents a single "slice" of a larger buffer
struct vk_bufslice {
    struct vk_memslice mem; // must be freed by the user when done
    VkBuffer buf;           // the buffer this memory was sliced from
    // For persistently mapped buffers, this points to the first usable byte of
    // this slice.
    void *data;
};

// Allocate a buffer slice. This is more efficient than vk_malloc_generic for
// when the user needs lots of buffers, since it doesn't require
// creating/destroying lots of (little) VkBuffers. `alignment` must be a power
// of two.
bool vk_malloc_buffer(struct vk_malloc *ma, VkBufferUsageFlags bufFlags,
                      VkMemoryPropertyFlags memFlags, VkDeviceSize size,
                      VkDeviceSize alignment, struct vk_bufslice *out);
