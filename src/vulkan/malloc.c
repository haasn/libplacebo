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

#include "malloc.h"
#include "command.h"
#include "utils.h"
#include "pl_thread.h"

#ifdef PL_HAVE_UNIX
#include <errno.h>
#include <strings.h>
#include <unistd.h>
#endif

// Controls the multiplication factor for new slab allocations. The new slab
// will always be allocated such that the size of the slab is this factor times
// the previous slab. Higher values make it grow faster.
#define PL_VK_HEAP_SLAB_GROWTH_RATE 4

// Controls the minimum slab size, to reduce the frequency at which very small
// slabs would need to get allocated when allocating the first few buffers.
// (Default: 1 MB)
#define PL_VK_HEAP_MINIMUM_SLAB_SIZE (1LLU << 20)

// Controls the maximum slab size, to reduce the effect of unbounded slab
// growth exhausting memory. If the application needs a single allocation
// that's bigger than this value, it will be allocated directly from the
// device. (Default: 256 MB)
#define PL_VK_HEAP_MAXIMUM_SLAB_SIZE (1LLU << 28)

// Controls the minimum free region size, to reduce thrashing the free space
// map with lots of small buffers during uninit. (Default: 1 KB)
#define PL_VK_HEAP_MINIMUM_REGION_SIZE (1LLU << 10)

// Represents a region of available memory
struct vk_region {
    size_t start; // first offset in region
    size_t end;   // first offset *not* in region
};

static inline size_t region_len(struct vk_region r)
{
    return r.end - r.start;
}

// A single slab represents a contiguous region of allocated memory. Actual
// allocations are served as slices of this. Slabs are organized into linked
// lists, which represent individual heaps.
struct vk_slab {
    pl_mutex lock;
    VkDeviceMemory mem;   // underlying device allocation
    size_t size;          // total size of `slab`
    size_t used;          // number of bytes actually in use (for GC accounting)
    bool dedicated;       // slab is allocated specifically for one object
    bool imported;        // slab represents an imported memory allocation
    // free space map: a sorted list of memory regions that are available
    PL_ARRAY(struct vk_region) regions;
    // optional, depends on the memory type:
    VkBuffer buffer;        // buffer spanning the entire slab
    void *data;             // mapped memory corresponding to `mem`
    bool coherent;          // mapped memory is coherent
    union pl_handle handle; // handle associated with this device memory
    enum pl_handle_type handle_type;
    // cached flags for convenience
    VkMemoryPropertyFlags flags;
};

// Represents a single memory heap. We keep track of a vk_heap for each
// combination of malloc parameters. This shouldn't actually be that many in
// practice, because some combinations simply never occur, and others will
// generally be the same for the same objects.
//
// Note: `vk_heap` addresses are not immutable, so we mustn't expose any dangling
// references to a `vk_heap` from e.g. `vk_memslice.priv = vk_slab`.
struct vk_heap {
    struct vk_malloc_params params;   // allocation params (with some fields nulled)
    PL_ARRAY(struct vk_slab *) slabs; // array of slabs sorted by size
};

// The overall state of the allocator, which keeps track of a vk_heap for each
// memory type.
struct vk_malloc {
    struct vk_ctx *vk;
    pl_mutex lock;
    VkPhysicalDeviceMemoryProperties props;
    PL_ARRAY(struct vk_heap) heaps;
};

void vk_malloc_print_heap(struct vk_malloc *ma, enum pl_log_level lev)
{
    struct vk_ctx *vk = ma->vk;
    size_t total_used = 0;
    size_t total_size = 0;

    pl_mutex_lock(&ma->lock);
    for (int i = 0; i < ma->heaps.num; i++) {
        struct vk_heap *heap = &ma->heaps.elem[i];
        const struct vk_malloc_params *par = &heap->params;

        PL_MSG(vk, lev, "Memory heap %d:", i);
        PL_MSG(vk, lev, "    Compatible types: 0x%"PRIx32, par->reqs.memoryTypeBits);
        if (par->required)
            PL_MSG(vk, lev, "    Required flags: 0x%"PRIx32, par->required);
        if (par->optimal)
            PL_MSG(vk, lev, "    Optimal flags: 0x%"PRIx32, par->optimal);
        if (par->buf_usage)
            PL_MSG(vk, lev, "    Buffer flags: 0x%"PRIx32, par->buf_usage);
        if (par->export_handle)
            PL_MSG(vk, lev, "    Export handle: 0x%x", par->export_handle);

        size_t heap_used = 0;
        size_t heap_size = 0;

        for (int j = 0; j < heap->slabs.num; j++) {
            struct vk_slab *slab = heap->slabs.elem[j];
            pl_mutex_lock(&slab->lock);
            PL_MSG(vk, lev, "    Slab %d:", j);
            PL_MSG(vk, lev, "      Used: %zu", slab->used);
            PL_MSG(vk, lev, "      Size: %zu", slab->size);
            PL_MSG(vk, lev, "      Regions: %d", slab->regions.num);

            if (slab->used > 0) {
                size_t largest_region = 0;
                for (int k = 0; k < slab->regions.num; k++) {
                    if (region_len(slab->regions.elem[k]) > largest_region)
                        largest_region = region_len(slab->regions.elem[k]);
                }

                float efficiency = slab->used / (slab->size - largest_region);
                PL_MSG(vk, lev, "      Efficiency: %.1f%%", 100 * efficiency);
            }

            heap_used += slab->used;
            heap_size += slab->size;
            pl_mutex_unlock(&slab->lock);
        }

        PL_MSG(vk, lev, "    Heap summary: %zu used / %zu total",
               heap_used, heap_size);

        total_used += heap_used;
        total_size += heap_size;
    }
    pl_mutex_unlock(&ma->lock);

    PL_MSG(vk, lev, "Memory summary: %zu used / %zu total",
           total_used, total_size);
}

static void slab_free(struct vk_ctx *vk, struct vk_slab *slab)
{
    if (!slab)
        return;

#ifndef NDEBUG
    if (!slab->dedicated && slab->used > 0) {
        fprintf(stderr, "!!! libplacebo: leaked %zu bytes of vulkan memory\n"
                "!!! slab total size: %zu bytes, flags: 0x%"PRIX64"\n",
                slab->used, slab->size, (uint64_t) slab->flags);
    }
#endif

    if (slab->imported) {
        switch (slab->handle_type) {
        case PL_HANDLE_FD:
        case PL_HANDLE_DMA_BUF:
            PL_TRACE(vk, "Unimporting slab of size %zu from fd: %d",
                     (size_t) slab->size, slab->handle.fd);
            break;
        case PL_HANDLE_WIN32:
        case PL_HANDLE_WIN32_KMT:
#ifdef PL_HAVE_WIN32
            PL_TRACE(vk, "Unimporting slab of size %zu from handle: %p",
                     (size_t) slab->size, (void *) slab->handle.handle);
#endif
            break;
        case PL_HANDLE_HOST_PTR:
            PL_TRACE(vk, "Unimporting slab of size %zu from ptr: %p",
                     (size_t) slab->size, (void *) slab->handle.ptr);
            break;
        }
    } else {
        switch (slab->handle_type) {
        case PL_HANDLE_FD:
        case PL_HANDLE_DMA_BUF:
#ifdef PL_HAVE_UNIX
            if (slab->handle.fd > -1)
                close(slab->handle.fd);
#endif
            break;
        case PL_HANDLE_WIN32:
#ifdef PL_HAVE_WIN32
            if (slab->handle.handle != NULL)
                CloseHandle(slab->handle.handle);
#endif
            break;
        case PL_HANDLE_WIN32_KMT:
            // PL_HANDLE_WIN32_KMT is just an identifier. It doesn't get closed.
            break;
        case PL_HANDLE_HOST_PTR:
            // Implicitly unmapped
            break;
        }

        PL_DEBUG(vk, "Freeing slab of size %zu", (size_t) slab->size);
    }

    vk->DestroyBuffer(vk->dev, slab->buffer, PL_VK_ALLOC);
    // also implicitly unmaps the memory if needed
    vk->FreeMemory(vk->dev, slab->mem, PL_VK_ALLOC);

    pl_mutex_destroy(&slab->lock);
    pl_free(slab);
}

// type_mask: optional
// thread-safety: safe
static bool find_best_memtype(const struct vk_malloc *ma, uint32_t type_mask,
                              const struct vk_malloc_params *params,
                              uint32_t *out_index)
{
    struct vk_ctx *vk = ma->vk;
    int best = -1;

    // The vulkan spec requires memory types to be sorted in the "optimal"
    // order, so the first matching type we find will be the best/fastest one.
    // That being said, we still want to prioritize memory types that have
    // better optional flags.

    type_mask &= params->reqs.memoryTypeBits;
    for (int i = 0; i < ma->props.memoryTypeCount; i++) {
        const VkMemoryType *mtype = &ma->props.memoryTypes[i];

        // The memory type flags must include our properties
        if ((mtype->propertyFlags & params->required) != params->required)
            continue;

        // The memory heap must be large enough for the allocation
        VkDeviceSize heapSize = ma->props.memoryHeaps[mtype->heapIndex].size;
        if (params->reqs.size > heapSize)
            continue;

        // The memory type must be supported by the type mask (bitfield)
        if (!(type_mask & (1LU << i)))
            continue;

        // Calculate the score as the number of optimal property flags matched
        int score = __builtin_popcountl(mtype->propertyFlags & params->optimal);
        if (score > best) {
            *out_index = i;
            best = score;
        }
    }

    if (best < 0) {
        PL_ERR(vk, "Found no memory type matching property flags 0x%x and type "
               "bits 0x%x!",
               (unsigned) params->required, (unsigned) type_mask);
        return false;
    }

    return true;
}

static bool buf_external_check(struct vk_ctx *vk, VkBufferUsageFlags usage,
                               enum pl_handle_type handle_type, bool import)
{
    if (!handle_type)
        return true;

    VkPhysicalDeviceExternalBufferInfo info = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO_KHR,
        .usage = usage,
        .handleType = vk_mem_handle_type(handle_type),
    };

    VkExternalBufferProperties props = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES_KHR,
    };

    pl_assert(info.handleType);
    vk->GetPhysicalDeviceExternalBufferProperties(vk->physd, &info, &props);
    return vk_external_mem_check(vk, &props.externalMemoryProperties,
                                 handle_type, import);
}

// thread-safety: safe
static struct vk_slab *slab_alloc(struct vk_malloc *ma,
                                  const struct vk_malloc_params *params)
{
    struct vk_ctx *vk = ma->vk;
    struct vk_slab *slab = pl_alloc_ptr(NULL, slab);
    *slab = (struct vk_slab) {
        .size = params->reqs.size,
        .handle_type = params->export_handle,
    };
    pl_mutex_init(&slab->lock);

    switch (slab->handle_type) {
    case PL_HANDLE_FD:
    case PL_HANDLE_DMA_BUF:
        slab->handle.fd = -1;
        break;
    case PL_HANDLE_WIN32:
    case PL_HANDLE_WIN32_KMT:
        slab->handle.handle = NULL;
        break;
    case PL_HANDLE_HOST_PTR:
        slab->handle.ptr = NULL;
        break;
    }

    VkExportMemoryAllocateInfoKHR ext_info = {
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
        .handleTypes = vk_mem_handle_type(slab->handle_type),
    };

    uint32_t type_mask = UINT32_MAX;
    if (params->buf_usage) {
        // Queue family sharing modes don't matter for buffers, so we just
        // set them as concurrent and stop worrying about it.
        uint32_t qfs[3] = {0};
        for (int i = 0; i < vk->pools.num; i++)
            qfs[i] = vk->pools.elem[i]->qf;

        VkExternalMemoryBufferCreateInfoKHR ext_buf_info = {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR,
            .handleTypes = ext_info.handleTypes,
        };

        VkBufferCreateInfo binfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = slab->handle_type ? &ext_buf_info : NULL,
            .size  = slab->size,
            .usage = params->buf_usage,
            .sharingMode = vk->pools.num > 1 ? VK_SHARING_MODE_CONCURRENT
                                             : VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = vk->pools.num,
            .pQueueFamilyIndices = qfs,
        };

        if (!buf_external_check(vk, binfo.usage, slab->handle_type, false)) {
            PL_ERR(vk, "Failed allocating shared memory buffer: possibly "
                   "the handle type is unsupported?");
            goto error;
        }

        VK(vk->CreateBuffer(vk->dev, &binfo, PL_VK_ALLOC, &slab->buffer));
        PL_VK_NAME(BUFFER, slab->buffer, "slab");

        VkMemoryRequirements reqs = {0};
        vk->GetBufferMemoryRequirements(vk->dev, slab->buffer, &reqs);
        slab->size = reqs.size; // this can be larger than `slab->size`
        type_mask = reqs.memoryTypeBits;

        // Note: we can ignore `reqs.align` because we always bind the buffer
        // memory to offset 0
    }

    VkMemoryAllocateInfo minfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = slab->size,
    };

    if (params->export_handle)
        vk_link_struct(&minfo, &ext_info);

    VkMemoryDedicatedAllocateInfoKHR dinfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR,
        .image = params->ded_image,
    };

    if (params->ded_image)
        vk_link_struct(&minfo, &dinfo);

    if (!find_best_memtype(ma, type_mask, params, &minfo.memoryTypeIndex))
        goto error;

    const VkMemoryType *mtype = &ma->props.memoryTypes[minfo.memoryTypeIndex];
    PL_DEBUG(vk, "Allocating %zu memory of type 0x%x (id %d) in heap %d",
             (size_t) slab->size, (unsigned) mtype->propertyFlags,
             (int) minfo.memoryTypeIndex, (int) mtype->heapIndex);

    VkResult res = vk->AllocateMemory(vk->dev, &minfo, PL_VK_ALLOC, &slab->mem);
    switch (res) {
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
    case VK_ERROR_OUT_OF_HOST_MEMORY:
        PL_ERR(vk, "Allocation of size %zu failed: %s!",
               (size_t) slab->size, vk_res_str(res));
        vk_malloc_print_heap(ma, PL_LOG_ERR);
        goto error;

    default:
        PL_VK_ASSERT(res, "vkAllocateMemory");
    }

    slab->flags = ma->props.memoryTypes[minfo.memoryTypeIndex].propertyFlags;
    if (slab->flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        VK(vk->MapMemory(vk->dev, slab->mem, 0, VK_WHOLE_SIZE, 0, &slab->data));
        slab->coherent = slab->flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }

    if (slab->buffer)
        VK(vk->BindBufferMemory(vk->dev, slab->buffer, slab->mem, 0));

#ifdef PL_HAVE_UNIX
    if (slab->handle_type == PL_HANDLE_FD ||
        slab->handle_type == PL_HANDLE_DMA_BUF)
    {
        VkMemoryGetFdInfoKHR fd_info = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
            .memory = slab->mem,
            .handleType = ext_info.handleTypes,
        };

        VK(vk->GetMemoryFdKHR(vk->dev, &fd_info, &slab->handle.fd));
    }
#endif

#ifdef PL_HAVE_WIN32
    if (slab->handle_type == PL_HANDLE_WIN32 ||
        slab->handle_type == PL_HANDLE_WIN32_KMT)
    {
        VkMemoryGetWin32HandleInfoKHR handle_info = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
            .memory = slab->mem,
            .handleType = ext_info.handleTypes,
        };

        VK(vk->GetMemoryWin32HandleKHR(vk->dev, &handle_info,
                                       &slab->handle.handle));
    }
#endif

    PL_ARRAY_APPEND(slab, slab->regions, (struct vk_region) {
        .start = 0,
        .end   = slab->size,
    });

    return slab;

error:
    slab_free(vk, slab);
    return NULL;
}

static void insert_region(struct vk_slab *slab, struct vk_region region)
{
    if (region.start == region.end)
        return;

    pl_assert(!slab->dedicated);
    bool big_enough = region_len(region) >= PL_VK_HEAP_MINIMUM_REGION_SIZE;

    // Find the index of the first region that comes after this
    for (int i = 0; i < slab->regions.num; i++) {
        struct vk_region *r = &slab->regions.elem[i];

        // Check for a few special cases which can be coalesced
        if (r->end == region.start) {
            // The new region is at the tail of this region. In addition to
            // modifying this region, we also need to coalesce all the following
            // regions for as long as possible
            r->end = region.end;

            struct vk_region *next = &slab->regions.elem[i+1];
            while (i+1 < slab->regions.num && r->end == next->start) {
                r->end = next->end;
                PL_ARRAY_REMOVE_AT(slab->regions, i+1);
            }
            return;
        }

        if (r->start == region.end) {
            // The new region is at the head of this region. We don't need to
            // do anything special here - because if this could be further
            // coalesced backwards, the previous loop iteration would already
            // have caught it.
            r->start = region.start;
            return;
        }

        if (r->start > region.start) {
            // The new region comes somewhere before this region, so insert
            // it into this index in the array.
            if (big_enough)
                PL_ARRAY_INSERT_AT(slab, slab->regions, i, region);
            return;
        }
    }

    // If we've reached the end of this loop, then all of the regions
    // come before the new region, and are disconnected - so append it
    if (big_enough)
        PL_ARRAY_APPEND(slab, slab->regions, region);
}

static void heap_uninit(struct vk_ctx *vk, struct vk_heap *heap)
{
    for (int i = 0; i < heap->slabs.num; i++)
        slab_free(vk, heap->slabs.elem[i]);

    pl_free(heap->slabs.elem);
    *heap = (struct vk_heap) {0};
}

struct vk_malloc *vk_malloc_create(struct vk_ctx *vk)
{
    struct vk_malloc *ma = pl_zalloc_ptr(NULL, ma);
    pl_mutex_init(&ma->lock);
    vk->GetPhysicalDeviceMemoryProperties(vk->physd, &ma->props);
    ma->vk = vk;

    PL_INFO(vk, "Memory heaps supported by device:");
    for (int i = 0; i < ma->props.memoryHeapCount; i++) {
        VkMemoryHeap heap = ma->props.memoryHeaps[i];
        PL_INFO(vk, "    %d: flags 0x%x size %zu",
                i, (unsigned) heap.flags, (size_t) heap.size);
    }

    PL_DEBUG(vk, "Memory types supported by device:");
    for (int i = 0; i < ma->props.memoryTypeCount; i++) {
        VkMemoryType type = ma->props.memoryTypes[i];
        PL_DEBUG(vk, "    %d: flags 0x%x heap %d",
                 i, (unsigned) type.propertyFlags, (int) type.heapIndex);
    }

    return ma;
}

void vk_malloc_destroy(struct vk_malloc **ma_ptr)
{
    struct vk_malloc *ma = *ma_ptr;
    if (!ma)
        return;

    for (int i = 0; i < ma->heaps.num; i++)
        heap_uninit(ma->vk, &ma->heaps.elem[i]);

    pl_mutex_destroy(&ma->lock);
    pl_free_ptr(ma_ptr);
}

pl_handle_caps vk_malloc_handle_caps(const struct vk_malloc *ma, bool import)
{
    struct vk_ctx *vk = ma->vk;
    pl_handle_caps caps = 0;

    for (int i = 0; vk_mem_handle_list[i]; i++) {
        // Try seeing if we could allocate a "basic" buffer using these
        // capabilities, with no fancy buffer usage. More specific checks will
        // happen down the line at VkBuffer creation time, but this should give
        // us a rough idea of what the driver supports.
        enum pl_handle_type type = vk_mem_handle_list[i];
        if (buf_external_check(vk, VK_BUFFER_USAGE_TRANSFER_DST_BIT, type, import))
            caps |= type;
    }

    return caps;
}

void vk_malloc_free(struct vk_malloc *ma, struct vk_memslice *slice)
{
    struct vk_ctx *vk = ma->vk;
    struct vk_slab *slab = slice->priv;
    if (!slab)
        goto done;

    pl_mutex_lock(&slab->lock);
    pl_assert(slab->used >= slice->size);
    slab->used -= slice->size;

    if (slab->dedicated) {
        pl_mutex_unlock(&slab->lock); // don't destroy locked mutex
        slab_free(vk, slab);
    } else {
        PL_TRACE(vk, "Freeing slice %zu + %zu from slab with size %zu",
                 (size_t) slice->offset, (size_t) slice->size,
                 (size_t) slab->size);

        // Return the allocation to the free space map
        insert_region(slab, (struct vk_region) {
            .start = slice->offset,
            .end   = slice->offset + slice->size,
        });
        pl_mutex_unlock(&slab->lock);
    }

done:
    *slice = (struct vk_memslice) {0};
}

static inline bool heap_params_eq(const struct vk_malloc_params *a,
                                  const struct vk_malloc_params *b)
{
    return a->reqs.size == b->reqs.size &&
           a->reqs.alignment == b->reqs.alignment &&
           a->reqs.memoryTypeBits == b->reqs.memoryTypeBits &&
           a->required == b->required &&
           a->optimal == b->optimal &&
           a->buf_usage == b->buf_usage &&
           a->export_handle == b->export_handle;
}

static struct vk_heap *find_heap(struct vk_malloc *ma,
                                 const struct vk_malloc_params *params)
{
    pl_assert(!params->import_handle);
    pl_assert(!params->ded_image);

    struct vk_malloc_params fixed = *params;
    fixed.reqs.alignment = 0;
    fixed.reqs.size = 0;
    fixed.shared_mem = (struct pl_shared_mem) {0};

    for (int i = 0; i < ma->heaps.num; i++) {
        if (heap_params_eq(&ma->heaps.elem[i].params, &fixed))
            return &ma->heaps.elem[i];
    }

    // Not found => add it
    PL_ARRAY_GROW(ma, ma->heaps);
    struct vk_heap *heap = &ma->heaps.elem[ma->heaps.num++];
    *heap = (struct vk_heap) {
        .params = fixed,
    };
    return heap;
}

static inline bool region_fits(struct vk_region r, size_t size, size_t align)
{
    return PL_ALIGN(r.start, align) + size <= r.end;
}

// Finds the best-fitting slab in a heap. If the heap is too small or too
// fragmented, a new slab will be allocated under the hood.
//
// Note: This locks the slab it returns
static struct vk_slab *heap_get_slab(struct vk_malloc *ma, struct vk_heap *heap,
                                     size_t size, size_t align,
                                     struct vk_region *out_region)
{
    struct vk_slab *slab = NULL;

    for (int i = 0; i < heap->slabs.num; i++) {
        slab = heap->slabs.elem[i];
        if (slab->size < size)
            continue;

        // Attempt a best fit search
        pl_mutex_lock(&slab->lock);
        size_t best_size = 0;
        int best_idx;
        for (int n = 0; n < slab->regions.num; n++) {
            struct vk_region r = slab->regions.elem[n];
            if (!region_fits(r, size, align))
                continue;
            if (best_size && region_len(r) > best_size)
                continue;
            best_idx = n;
            best_size = region_len(r);
        }

        if (best_size) {
            *out_region = slab->regions.elem[best_idx];
            PL_ARRAY_REMOVE_AT(slab->regions, best_idx);
            return slab;
        } else {
            pl_mutex_unlock(&slab->lock);
            continue;
        }
    }

    // Otherwise, allocate a new vk_slab and append it to the list.
    size_t cur_size = PL_MAX(size, slab ? slab->size : 0);
    size_t slab_size = PL_VK_HEAP_SLAB_GROWTH_RATE * cur_size;
    slab_size = PL_MAX(PL_VK_HEAP_MINIMUM_SLAB_SIZE, slab_size);
    slab_size = PL_MIN(PL_VK_HEAP_MAXIMUM_SLAB_SIZE, slab_size);
    pl_assert(slab_size >= size);

    struct vk_malloc_params params = heap->params;
    params.reqs.size = slab_size;

    // Don't hold the lock while allocating the slab, because it can be a
    // potentially very costly operation.
    pl_mutex_unlock(&ma->lock);
    slab = slab_alloc(ma, &params);
    pl_mutex_lock(&ma->lock);

    if (!slab)
        return NULL;
    pl_mutex_lock(&slab->lock);
    PL_ARRAY_APPEND(NULL, heap->slabs, slab);

    // Return the only region there is in a newly allocated slab
    assert(slab->regions.num == 1);
    *out_region = slab->regions.elem[0];
    slab->regions.num = 0;
    return slab;
}

static bool vk_malloc_import(struct vk_malloc *ma, struct vk_memslice *out,
                             const struct vk_malloc_params *params)
{
    struct vk_ctx *vk = ma->vk;
    VkExternalMemoryHandleTypeFlagBitsKHR vk_handle_type;
    vk_handle_type = vk_mem_handle_type(params->import_handle);

    struct vk_slab *slab = NULL;
    const struct pl_shared_mem *shmem = &params->shared_mem;

    VkMemoryDedicatedAllocateInfoKHR dinfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR,
        .image = params->ded_image,
    };

    VkImportMemoryFdInfoKHR fdinfo = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = vk_handle_type,
        .fd = -1,
    };

    VkImportMemoryHostPointerInfoEXT ptrinfo = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .handleType = vk_handle_type,
    };

    VkMemoryAllocateInfo ainfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = shmem->size,
    };

    if (params->ded_image)
        vk_link_struct(&ainfo, &dinfo);

    VkBuffer buffer = VK_NULL_HANDLE;
    VkMemoryRequirements reqs = params->reqs;

    if (params->buf_usage) {
        uint32_t qfs[3] = {0};
        for (int i = 0; i < vk->pools.num; i++)
            qfs[i] = vk->pools.elem[i]->qf;

        VkExternalMemoryBufferCreateInfoKHR ext_buf_info = {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR,
            .handleTypes = vk_handle_type,
        };

        VkBufferCreateInfo binfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = &ext_buf_info,
            .size = shmem->size,
            .usage = params->buf_usage,
            .sharingMode = vk->pools.num > 1 ? VK_SHARING_MODE_CONCURRENT
                                             : VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = vk->pools.num,
            .pQueueFamilyIndices = qfs,
        };

        VK(vk->CreateBuffer(vk->dev, &binfo, PL_VK_ALLOC, &buffer));
        PL_VK_NAME(BUFFER, buffer, "imported");

        vk->GetBufferMemoryRequirements(vk->dev, buffer, &reqs);
    }

    if (reqs.size > shmem->size) {
        PL_ERR(vk, "Imported object requires %zu bytes, larger than the "
               "provided size %zu!",
               (size_t) reqs.size, shmem->size);
        goto error;
    }

    if (shmem->offset % reqs.alignment || shmem->offset % params->reqs.alignment) {
        PL_ERR(vk, "Imported object offset %zu conflicts with alignment %zu!",
               shmem->offset, pl_lcm(reqs.alignment, params->reqs.alignment));
        goto error;
    }

    switch (params->import_handle) {
#ifdef PL_HAVE_UNIX
    case PL_HANDLE_DMA_BUF: {
        if (!vk->GetMemoryFdPropertiesKHR) {
            PL_ERR(vk, "Importing PL_HANDLE_DMA_BUF requires %s.",
                   VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
            goto error;
        }

        VkMemoryFdPropertiesKHR fdprops = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
        };

        VK(vk->GetMemoryFdPropertiesKHR(vk->dev,
                                        vk_handle_type,
                                        shmem->handle.fd,
                                        &fdprops));

        // We dup() the fd to make it safe to import the same original fd
        // multiple times.
        fdinfo.fd = dup(shmem->handle.fd);
        if (fdinfo.fd == -1) {
            PL_ERR(vk, "Failed to dup() fd (%d) when importing memory: %s",
                   fdinfo.fd, strerror(errno));
            goto error;
        }

        reqs.memoryTypeBits &= fdprops.memoryTypeBits;
        vk_link_struct(&ainfo, &fdinfo);
        break;
    }
#else // !PL_HAVE_UNIX
    case PL_HANDLE_DMA_BUF:
        PL_ERR(vk, "PL_HANDLE_DMA_BUF requires building with UNIX support!");
        goto error;
#endif

    case PL_HANDLE_HOST_PTR: {
        VkMemoryHostPointerPropertiesEXT ptrprops = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
        };

        VK(vk->GetMemoryHostPointerPropertiesEXT(vk->dev, vk_handle_type,
                                                 shmem->handle.ptr,
                                                 &ptrprops));

        ptrinfo.pHostPointer = (void *) shmem->handle.ptr;
        reqs.memoryTypeBits &= ptrprops.memoryTypeBits;
        vk_link_struct(&ainfo, &ptrinfo);
        break;
    }

    case PL_HANDLE_FD:
    case PL_HANDLE_WIN32:
    case PL_HANDLE_WIN32_KMT:
        PL_ERR(vk, "vk_malloc_import: unsupported handle type %d",
               params->import_handle);
        goto error;
    }

    pl_assert(ainfo.pNext);

    if (!find_best_memtype(ma, reqs.memoryTypeBits, params, &ainfo.memoryTypeIndex)) {
        PL_ERR(vk, "No compatible memory types offered for imported memory!");
        goto error;
    }

    VkDeviceMemory vkmem = NULL;
    VK(vk->AllocateMemory(vk->dev, &ainfo, PL_VK_ALLOC, &vkmem));

    slab = pl_alloc_ptr(NULL, slab);
    *slab = (struct vk_slab) {
        .mem = vkmem,
        .dedicated = true,
        .imported = true,
        .buffer = buffer,
        .size = shmem->size,
        .used = shmem->size,
        .handle_type = params->import_handle,
    };
    pl_mutex_init(&slab->lock);

    *out = (struct vk_memslice) {
        .vkmem = vkmem,
        .buf = buffer,
        .size = shmem->size - shmem->offset,
        .offset = shmem->offset,
        .shared_mem = *shmem,
        .priv = slab,
    };

    switch (params->import_handle) {
    case PL_HANDLE_DMA_BUF:
    case PL_HANDLE_FD:
        PL_TRACE(vk, "Imported %zu of memory from fd: %d%s",
                 (size_t) slab->size, shmem->handle.fd,
                 params->ded_image ? " (dedicated)" : "");
        // fd ownership is transferred at this point.
        slab->handle.fd = fdinfo.fd;
        fdinfo.fd = -1;
        break;
    case PL_HANDLE_HOST_PTR:
        PL_TRACE(vk, "Imported %zu of memory from ptr: %p%s",
                 (size_t) slab->size, shmem->handle.ptr,
                 params->ded_image ? " (dedicated" : "");
        slab->handle.ptr = ptrinfo.pHostPointer;
        break;
    case PL_HANDLE_WIN32:
    case PL_HANDLE_WIN32_KMT:
        break;
    }

    VkMemoryPropertyFlags flags = ma->props.memoryTypes[ainfo.memoryTypeIndex].propertyFlags;
    if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        VK(vk->MapMemory(vk->dev, slab->mem, 0, VK_WHOLE_SIZE, 0, &slab->data));
        slab->coherent = flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        out->data = (uint8_t *) slab->data + out->offset;
        out->coherent = slab->coherent;
        if (!slab->coherent) {
            // Mapping does not implicitly invalidate mapped memory
            VK(vk->InvalidateMappedMemoryRanges(vk->dev, 1, &(VkMappedMemoryRange) {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = slab->mem,
                .size = VK_WHOLE_SIZE,
            }));
        }
    }

    if (buffer)
        VK(vk->BindBufferMemory(vk->dev, buffer, vkmem, 0));

    return true;

error:
    vk->DestroyBuffer(vk->dev, buffer, PL_VK_ALLOC);
#ifdef PL_HAVE_UNIX
    if (fdinfo.fd > -1)
        close(fdinfo.fd);
#endif
    pl_free(slab);
    *out = (struct vk_memslice) {0};
    return false;
}

bool vk_malloc_slice(struct vk_malloc *ma, struct vk_memslice *out,
                     const struct vk_malloc_params *params)
{
    struct vk_ctx *vk = ma->vk;
    pl_assert(!params->import_handle || !params->export_handle);
    if (params->import_handle)
        return vk_malloc_import(ma, out, params);

    size_t size = params->reqs.size;
    size_t align = params->reqs.alignment;
    align = pl_lcm(align, vk->limits.bufferImageGranularity);

    struct vk_slab *slab;
    VkDeviceSize offset;

    if (params->ded_image || size >= PL_VK_HEAP_MAXIMUM_SLAB_SIZE) {
        slab = slab_alloc(ma, params);
        if (!slab)
            return false;
        slab->dedicated = true;
        slab->used = size;
        offset = 0;
    } else {
        pl_mutex_lock(&ma->lock);
        struct vk_heap *heap = find_heap(ma, params);
        struct vk_region region;
        slab = heap_get_slab(ma, heap, size, align, &region);
        pl_mutex_unlock(&ma->lock);
        if (!slab)
            return false;

        bool noncoherent = (slab->flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                          !(slab->flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (noncoherent) {
            size = PL_ALIGN(size, vk->limits.nonCoherentAtomSize);
            align = pl_lcm(align, vk->limits.nonCoherentAtomSize);
        }

        offset = PL_ALIGN(region.start, align);
        size_t out_end = offset + size;
        insert_region(slab, (struct vk_region) { region.start, offset });
        insert_region(slab, (struct vk_region) { out_end, region.end });
        slab->used += size;
        pl_mutex_unlock(&slab->lock);

        PL_TRACE(vk, "Sub-allocating slice %zu + %zu from slab with size %zu",
                 (size_t) offset,  size, (size_t) slab->size);
    }

    *out = (struct vk_memslice) {
        .vkmem = slab->mem,
        .offset = offset,
        .size = size,
        .buf = slab->buffer,
        .data = slab->data ? (uint8_t *) slab->data + offset : 0x0,
        .coherent = slab->coherent,
        .priv = slab,
        .shared_mem = {
            .handle = slab->handle,
            .offset = offset,
            .size = slab->size,
        },
    };
    return true;
}
