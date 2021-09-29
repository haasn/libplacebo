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

// Get the supported handle types for this malloc instance
pl_handle_caps vk_malloc_handle_caps(const struct vk_malloc *ma, bool import);

// Represents a single "slice" of generic (non-buffer) memory, plus some
// metadata for accounting. This struct is essentially read-only.
struct vk_memslice {
    VkDeviceMemory vkmem;
    VkDeviceSize offset;
    VkDeviceSize size;
    void *priv;
    // depending on the type/flags:
    struct pl_shared_mem shared_mem;
    VkBuffer buf;   // associated buffer (when `buf_usage` is nonzero)
    void *data;     // pointer to slice (for persistently mapped slices)
    bool coherent;  // whether `data` is coherent
};

struct vk_malloc_params {
    VkMemoryRequirements reqs;
    VkMemoryPropertyFlags required;
    VkMemoryPropertyFlags optimal;
    VkBufferUsageFlags buf_usage;
    VkImage ded_image; // for dedicated image allocations
    enum pl_handle_type export_handle;
    enum pl_handle_type import_handle;
    struct pl_shared_mem shared_mem; // for `import_handle`
};

bool vk_malloc_slice(struct vk_malloc *ma, struct vk_memslice *out,
                     const struct vk_malloc_params *params);

void vk_malloc_free(struct vk_malloc *ma, struct vk_memslice *slice);

// Clean up unused slabs. Call this roughly once per frame to reduce
// memory pressure / memory leaks.
void vk_malloc_garbage_collect(struct vk_malloc *ma);

// For debugging purposes. Doesn't include dedicated slab allocations!
void vk_malloc_print_stats(struct vk_malloc *ma, enum pl_log_level);
