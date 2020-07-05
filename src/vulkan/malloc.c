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

#ifdef VK_HAVE_UNIX
#include <errno.h>
#include <strings.h>
#include <unistd.h>
#endif

// Controls the multiplication factor for new slab allocations. The new slab
// will always be allocated such that the size of the slab is this factor times
// the previous slab. Higher values make it grow faster.
#define PLVK_HEAP_SLAB_GROWTH_RATE 4

// Controls the minimum slab size, to reduce the frequency at which very small
// slabs would need to get allocated when allocating the first few buffers.
// (Default: 1 MB)
#define PLVK_HEAP_MINIMUM_SLAB_SIZE (1 << 20)

// Controls the maximum slab size, to reduce the effect of unbounded slab
// growth exhausting memory. If the application needs a single allocation
// that's bigger than this value, it will be allocated directly from the
// device. (Default: 256 MB)
#define PLVK_HEAP_MAXIMUM_SLAB_SIZE (1 << 28)

// Controls the minimum free region size, to reduce thrashing the free space
// map with lots of small buffers during uninit. (Default: 1 KB)
#define PLVK_HEAP_MINIMUM_REGION_SIZE (1 << 10)

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
    VkDeviceMemory mem;   // underlying device allocation
    size_t size;          // total size of `slab`
    size_t used;          // number of bytes actually in use (for GC accounting)
    bool dedicated;       // slab is allocated specifically for one object
    bool imported;        // slab represents an imported memory allocation
    // free space map: a sorted list of memory regions that are available
    struct vk_region *regions;
    int num_regions;
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
// Note: `vk_heap` addresses are not immutable, so we musn't expose any dangling
// references to a `vk_heap` from e.g. `vk_memslice.priv = vk_slab`.
struct vk_heap {
    struct vk_malloc_params params; // allocation params (with some fields nulled)
    struct vk_slab **slabs; // array of slabs sorted by size
    int num_slabs;
};

// The overall state of the allocator, which keeps track of a vk_heap for each
// memory type.
struct vk_malloc {
    struct vk_ctx *vk;
    VkPhysicalDeviceMemoryProperties props;
    VkDeviceSize host_ptr_align;
    struct vk_heap *heaps;
    int num_heaps;
};

static void slab_free(struct vk_ctx *vk, struct vk_slab *slab)
{
    if (!slab)
        return;

    pl_assert(slab->dedicated || slab->used == 0);
    if (slab->imported) {
        switch (slab->handle_type) {
        case PL_HANDLE_FD:
        case PL_HANDLE_DMA_BUF:
            PL_DEBUG(vk, "Unimporting slab of size %zu from fd: %d",
                     (size_t) slab->size, slab->handle.fd);
            break;
        case PL_HANDLE_WIN32:
        case PL_HANDLE_WIN32_KMT:
#ifdef VK_HAVE_WIN32
            PL_DEBUG(vk, "Unimporting slab of size %zu from handle: %p",
                     (size_t) slab->size, (void *) slab->handle.handle);
#endif
            break;
        case PL_HANDLE_HOST_PTR:
            PL_DEBUG(vk, "Unimporting slab of size %zu from ptr: %p",
                     (size_t) slab->size, (void *) slab->handle.ptr);
            break;
        }
    } else {
        switch (slab->handle_type) {
        case PL_HANDLE_FD:
        case PL_HANDLE_DMA_BUF:
#ifdef VK_HAVE_UNIX
            if (slab->handle.fd > -1)
                close(slab->handle.fd);
#endif
            break;
        case PL_HANDLE_WIN32:
#ifdef VK_HAVE_WIN32
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

        PL_INFO(vk, "Freeing slab of size %zu", (size_t) slab->size);
    }

    vk->DestroyBuffer(vk->dev, slab->buffer, VK_ALLOC);
    // also implicitly unmaps the memory if needed
    vk->FreeMemory(vk->dev, slab->mem, VK_ALLOC);

    talloc_free(slab);
}

// type_mask: optional
static bool find_best_memtype(struct vk_malloc *ma, uint32_t type_mask,
                              const struct vk_malloc_params *params,
                              uint32_t *out_index)
{
    struct vk_ctx *vk = ma->vk;
    int best = -1;

    // The vulkan spec requires memory types to be sorted in the "optimal"
    // order, so the first matching type we find will be the best/fastest one.
    // That being said, we still want to prioritize memory types that have
    // better optional flags.

    type_mask = PL_DEF(type_mask, UINT32_MAX);
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
        if (!(type_mask & (1 << i)))
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
               (unsigned) params->required,
               (unsigned) params->reqs.memoryTypeBits);
        return false;
    }

    return true;
}

static bool buf_external_check(struct vk_ctx *vk, VkBufferUsageFlags usage,
                               enum pl_handle_type handle_type, bool import)
{
    if (!handle_type)
        return true;

    if (!vk->GetPhysicalDeviceExternalBufferPropertiesKHR)
        return false;

    VkPhysicalDeviceExternalBufferInfoKHR info = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO_KHR,
        .usage = usage,
        .handleType = vk_mem_handle_type(handle_type),
    };

    VkExternalBufferPropertiesKHR props = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES_KHR,
    };

    pl_assert(info.handleType);
    vk->GetPhysicalDeviceExternalBufferPropertiesKHR(vk->physd, &info, &props);
    return vk_external_mem_check(&props.externalMemoryProperties, handle_type,
                                 import);
}

static struct vk_slab *slab_alloc(struct vk_malloc *ma,
                                  const struct vk_malloc_params *params)
{
    struct vk_ctx *vk = ma->vk;
    struct vk_slab *slab = talloc_ptrtype(NULL, slab);
    *slab = (struct vk_slab) {
        .size = params->reqs.size,
        .handle_type = params->export_handle,
    };

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

    uint32_t type_mask = 0;
    if (params->buf_usage) {
        // FIXME: Since we can't keep track of queue family ownership properly,
        // and we don't know in advance what types of queue families this buffer
        // will belong to, we're forced to share all of our buffers between all
        // command pools.
        uint32_t qfs[3] = {0};
        for (int i = 0; i < vk->num_pools; i++)
            qfs[i] = vk->pools[i]->qf;

        VkExternalMemoryBufferCreateInfoKHR ext_buf_info = {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR,
            .handleTypes = ext_info.handleTypes,
        };

        VkBufferCreateInfo binfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = slab->handle_type ? &ext_buf_info : NULL,
            .size  = slab->size,
            .usage = params->buf_usage,
            .sharingMode = vk->num_pools > 1 ? VK_SHARING_MODE_CONCURRENT
                                             : VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = vk->num_pools,
            .pQueueFamilyIndices = qfs,
        };

        if (!buf_external_check(vk, binfo.usage, slab->handle_type, false)) {
            PL_ERR(vk, "Failed allocating shared memory buffer: possibly "
                   "the handle type is unsupported?");
            goto error;
        }

        VK(vk->CreateBuffer(vk->dev, &binfo, VK_ALLOC, &slab->buffer));
        VK_NAME(BUFFER, slab->buffer, "slab");

        VkMemoryRequirements reqs = {0};
        vk->GetBufferMemoryRequirements(vk->dev, slab->buffer, &reqs);
        slab->size = reqs.size; // this can be larger than `slab->size`
        type_mask = reqs.memoryTypeBits;

        // Note: we can ignore `reqs.align` because we always bind the buffer
        // memory to offset 0
    }

    VkMemoryAllocateInfo minfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = params->export_handle ? &ext_info : NULL,
        .allocationSize = slab->size,
    };

    if (!find_best_memtype(ma, type_mask, params, &minfo.memoryTypeIndex))
        goto error;

    const VkMemoryType *mtype = &ma->props.memoryTypes[minfo.memoryTypeIndex];
    PL_INFO(vk, "Allocating %zu memory of type 0x%x (id %d) in heap %d",
            (size_t) slab->size, (unsigned) mtype->propertyFlags,
            (int) minfo.memoryTypeIndex, (int) mtype->heapIndex);

    VK(vk->AllocateMemory(vk->dev, &minfo, VK_ALLOC, &slab->mem));

    slab->flags = ma->props.memoryTypes[minfo.memoryTypeIndex].propertyFlags;
    if (slab->flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        VK(vk->MapMemory(vk->dev, slab->mem, 0, VK_WHOLE_SIZE, 0, &slab->data));
        slab->coherent = slab->flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }

    if (slab->buffer)
        VK(vk->BindBufferMemory(vk->dev, slab->buffer, slab->mem, 0));

#ifdef VK_HAVE_UNIX
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

#ifdef VK_HAVE_WIN32
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

    TARRAY_APPEND(slab, slab->regions, slab->num_regions, (struct vk_region) {
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

    bool big_enough = region_len(region) >= PLVK_HEAP_MINIMUM_REGION_SIZE;

    // Find the index of the first region that comes after this
    for (int i = 0; i < slab->num_regions; i++) {
        struct vk_region *r = &slab->regions[i];

        // Check for a few special cases which can be coalesced
        if (r->end == region.start) {
            // The new region is at the tail of this region. In addition to
            // modifying this region, we also need to coalesce all the following
            // regions for as long as possible
            r->end = region.end;

            struct vk_region *next = &slab->regions[i+1];
            while (i+1 < slab->num_regions && r->end == next->start) {
                r->end = next->end;
                TARRAY_REMOVE_AT(slab->regions, slab->num_regions, i+1);
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
            if (big_enough) {
                TARRAY_INSERT_AT(slab, slab->regions, slab->num_regions,
                                    i, region);
            }
            return;
        }
    }

    // If we've reached the end of this loop, then all of the regions
    // come before the new region, and are disconnected - so append it
    if (big_enough)
        TARRAY_APPEND(slab, slab->regions, slab->num_regions, region);
}

static void heap_uninit(struct vk_ctx *vk, struct vk_heap *heap)
{
    for (int i = 0; i < heap->num_slabs; i++)
        slab_free(vk, heap->slabs[i]);

    talloc_free(heap->slabs);
    *heap = (struct vk_heap){0};
}

struct vk_malloc *vk_malloc_create(struct vk_ctx *vk)
{
    struct vk_malloc *ma = talloc_zero(NULL, struct vk_malloc);
    ma->vk = vk;

    VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
    };

    VkPhysicalDeviceProperties2 dprops = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &host_props,
    };

    vk->GetPhysicalDeviceProperties2KHR(vk->physd, &dprops);
    vk->GetPhysicalDeviceMemoryProperties(vk->physd, &ma->props);
    ma->host_ptr_align = host_props.minImportedHostPointerAlignment;

    PL_INFO(vk, "Memory heaps supported by device:");
    for (int i = 0; i < ma->props.memoryHeapCount; i++) {
        VkMemoryHeap heap = ma->props.memoryHeaps[i];
        PL_INFO(vk, "    heap %d: flags 0x%x size %zu",
                i, (unsigned) heap.flags, (size_t) heap.size);
    }

    PL_INFO(vk, "Memory types supported by device:");
    for (int i = 0; i < ma->props.memoryTypeCount; i++) {
        VkMemoryType type = ma->props.memoryTypes[i];
        PL_INFO(vk, "    type %d: flags 0x%x heap %d",
                i, (unsigned) type.propertyFlags, (int) type.heapIndex);
    }

    return ma;
}

void vk_malloc_destroy(struct vk_malloc **ma_ptr)
{
    struct vk_malloc *ma = *ma_ptr;
    if (!ma)
        return;

    for (int i = 0; i < ma->num_heaps; i++)
        heap_uninit(ma->vk, &ma->heaps[i]);

    TA_FREEP(ma_ptr);
}

pl_handle_caps vk_malloc_handle_caps(struct vk_malloc *ma, bool import)
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

    pl_assert(slab->used >= slice->size);
    slab->used -= slice->size;

    if (slab->dedicated) {
        // If the slab was purpose-allocated for this memslice, we can just
        // free it here
        slab_free(vk, slab);
    } else {
        PL_DEBUG(vk, "Freeing slice %zu + %zu from slab with size %zu",
                 (size_t) slice->offset, (size_t) slice->size,
                 (size_t) slab->size);

        // Return the allocation to the free space map
        insert_region(slab, (struct vk_region) {
            .start = slice->offset,
            .end   = slice->offset + slice->size,
        });
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

    struct vk_malloc_params fixed = *params;
    fixed.reqs.alignment = 0;
    fixed.reqs.size = 0;
    fixed.shared_mem = (struct pl_shared_mem) {0};

    for (int i = 0; i < ma->num_heaps; i++) {
        if (heap_params_eq(&ma->heaps[i].params, &fixed))
            return &ma->heaps[i];
    }

    // Not found => add it
    TARRAY_GROW(ma, ma->heaps, ma->num_heaps + 1);
    struct vk_heap *heap = &ma->heaps[ma->num_heaps++];
    *heap = (struct vk_heap) {
        .params = fixed,
    };
    return heap;
}

static inline bool region_fits(struct vk_region r, size_t size, size_t align)
{
    return PL_ALIGN(r.start, align) + size <= r.end;
}

// Finds the best-fitting region in a heap. If the heap is too small or too
// fragmented, a new slab will be allocated under the hood.
static bool heap_get_region(struct vk_malloc *ma, struct vk_heap *heap,
                            size_t size, size_t align,
                            struct vk_slab **out_slab, int *out_index)
{
    struct vk_slab *slab = NULL;
    struct vk_malloc_params params = heap->params;

    // If the allocation is very big, serve it directly instead of bothering
    // with the heap
    if (size > PLVK_HEAP_MAXIMUM_SLAB_SIZE) {
        params.reqs.size = size;
        params.reqs.alignment = align;
        slab = slab_alloc(ma, &params);
        if (slab)
            slab->dedicated = true;
        *out_slab = slab;
        *out_index = 0;
        return !!slab;
    }

    for (int i = 0; i < heap->num_slabs; i++) {
        slab = heap->slabs[i];
        if (slab->size < size)
            continue;

        // Attempt a best fit search
        int best = -1;
        for (int n = 0; n < slab->num_regions; n++) {
            struct vk_region r = slab->regions[n];
            if (!region_fits(r, size, align))
                continue;
            if (best >= 0 && region_len(r) > region_len(slab->regions[best]))
                continue;
            best = n;
        }

        if (best >= 0) {
            *out_slab = slab;
            *out_index = best;
            return true;
        }
    }

    // Otherwise, allocate a new vk_slab and append it to the list.
    size_t cur_size = PL_MAX(size, slab ? slab->size : 0);
    size_t slab_size = PLVK_HEAP_SLAB_GROWTH_RATE * cur_size;
    slab_size = PL_MAX(PLVK_HEAP_MINIMUM_SLAB_SIZE, slab_size);
    slab_size = PL_MIN(PLVK_HEAP_MAXIMUM_SLAB_SIZE, slab_size);
    pl_assert(slab_size >= size);

    params.reqs.size = slab_size;
    slab = slab_alloc(ma, &params);
    if (!slab)
        return false;
    TARRAY_APPEND(NULL, heap->slabs, heap->num_slabs, slab);

    // Return the only region there is in a newly allocated slab
    pl_assert(slab->num_regions == 1);
    *out_slab = slab;
    *out_index = 0;
    return true;
}

static bool vk_malloc_import(struct vk_malloc *ma, struct vk_memslice *out,
                             const struct vk_malloc_params *params)
{
    struct vk_ctx *vk = ma->vk;
    VkExternalMemoryHandleTypeFlagBitsKHR vk_handle_type;
    vk_handle_type = vk_mem_handle_type(params->import_handle);

    struct vk_slab *slab = NULL;
    const struct pl_shared_mem *shmem = &params->shared_mem;

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

    VkBuffer buffer = VK_NULL_HANDLE;
    VkMemoryRequirements reqs = params->reqs;

    if (params->buf_usage) {
        // FIXME: Avoid use of CONCURRENT sharing
        uint32_t qfs[3] = {0};
        for (int i = 0; i < vk->num_pools; i++)
            qfs[i] = vk->pools[i]->qf;

        VkExternalMemoryBufferCreateInfoKHR ext_buf_info = {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR,
            .handleTypes = vk_handle_type,
        };

        VkBufferCreateInfo binfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = &ext_buf_info,
            .size = shmem->size,
            .usage = params->buf_usage,
            .sharingMode = vk->num_pools > 1 ? VK_SHARING_MODE_CONCURRENT
                                             : VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = vk->num_pools,
            .pQueueFamilyIndices = qfs,
        };

        VK(vk->CreateBuffer(vk->dev, &binfo, VK_ALLOC, &buffer));
        VK_NAME(BUFFER, buffer, "imported");

        vk->GetBufferMemoryRequirements(vk->dev, buffer, &reqs);
    }

    if (reqs.size > shmem->size) {
        PL_ERR(vk, "Imported object requires %zu bytes, larger than the "
               "provided size %zu!",
               reqs.size, shmem->size);
        goto error;
    }

    switch (params->import_handle) {
#ifdef VK_HAVE_UNIX
    case PL_HANDLE_DMA_BUF: {
        if (!vk->GetMemoryFdPropertiesKHR) {
            PL_ERR(vk, "Importing PL_HANDLE_DMA_BUF requires %s.",
                   VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
            return false;
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
            return false;
        }

        reqs.memoryTypeBits &= fdprops.memoryTypeBits;
        ainfo.pNext = &fdinfo;
        break;
    }
#else // !VK_HAVE_UNIX
    case PL_HANDLE_DMA_BUF:
        PL_ERR(vk, "PL_HANDLE_DMA_BUF requires building with UNIX support!");
        return false;
#endif

    case PL_HANDLE_HOST_PTR: {
        if (!vk->GetMemoryHostPointerPropertiesEXT) {
            PL_ERR(vk, "Importing PL_HANDLE_HOST_PTR requires %s.",
                   VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
            return false;
        }

        if ((uintptr_t) shmem->handle.ptr % ma->host_ptr_align) {
            PL_ERR(vk, "Imported host pointer %p does not adhere to the "
                   "alignment required to import pointers: %zu",
                   shmem->handle.ptr, (size_t) ma->host_ptr_align);
            return false;
        }

        if (shmem->size % ma->host_ptr_align) {
            PL_ERR(vk, "Imported host pointer %p of size %zu does not adhere "
                   "to the size alignment required to import pointers: %zu",
                   shmem->handle.ptr, shmem->size,
                   (size_t) ma->host_ptr_align);
            return false;
        }

        VkMemoryHostPointerPropertiesEXT ptrprops = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
        };

        VK(vk->GetMemoryHostPointerPropertiesEXT(vk->dev, vk_handle_type,
                                                 shmem->handle.ptr,
                                                 &ptrprops));

        ptrinfo.pHostPointer = shmem->handle.ptr;
        reqs.memoryTypeBits &= ptrprops.memoryTypeBits;
        ainfo.pNext = &ptrinfo;
        break;
    }

    case PL_HANDLE_FD:
    case PL_HANDLE_WIN32:
    case PL_HANDLE_WIN32_KMT:
        PL_ERR(vk, "vk_malloc_import: unsupported handle type %d",
               params->import_handle);
        return false;
    }

    pl_assert(ainfo.pNext);

    if (!find_best_memtype(ma, reqs.memoryTypeBits, params, &ainfo.memoryTypeIndex)) {
        PL_ERR(vk, "No compatible memory types offered for imported memory!");
        return false;
    }

    VkDeviceMemory vkmem = NULL;
    VK(vk->AllocateMemory(vk->dev, &ainfo, VK_ALLOC, &vkmem));

    slab = talloc_ptrtype(NULL, slab);
    *slab = (struct vk_slab) {
        .mem = vkmem,
        .dedicated = true,
        .imported = true,
        .buffer = buffer,
        .size = shmem->size,
        .used = shmem->size,
        .handle_type = params->import_handle,
    };

    *out = (struct vk_memslice) {
        .vkmem = vkmem,
        .buf = buffer,
        .size = shmem->size,
        .offset = shmem->offset,
        .shared_mem = *shmem,
        .priv = slab,
    };

    switch (params->import_handle) {
    case PL_HANDLE_DMA_BUF:
    case PL_HANDLE_FD:
        PL_DEBUG(vk, "Imported %zu of memory from fd: %d",
                 (size_t) slab->size, shmem->handle.fd);
        // fd ownership is transferred at this point.
        slab->handle.fd = fdinfo.fd;
        fdinfo.fd = -1;
        break;
    case PL_HANDLE_HOST_PTR:
        PL_DEBUG(vk, "Imported %zu of memory from ptr: %p",
                 (size_t) slab->size, shmem->handle.ptr);
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
    }

    if (buffer)
        VK(vk->BindBufferMemory(vk->dev, buffer, vkmem, 0));

    return true;

error:
    vk->DestroyBuffer(vk->dev, buffer, VK_ALLOC);
#ifdef VK_HAVE_UNIX
    if (fdinfo.fd > -1)
        close(fdinfo.fd);
#endif
    talloc_free(slab);
    return false;
}

bool vk_malloc_slice(struct vk_malloc *ma, struct vk_memslice *out,
                     const struct vk_malloc_params *params)
{
    pl_assert(!params->import_handle || !params->export_handle);
    if (params->import_handle)
        return vk_malloc_import(ma, out, params);

    struct vk_ctx *vk = ma->vk;
    struct vk_heap *heap = find_heap(ma, params);
    struct vk_slab *slab;

    size_t size = params->reqs.size;
    size_t align = params->reqs.alignment;
    align = pl_lcm(align, vk->limits.bufferImageGranularity);

    int index_region;
    if (!heap_get_region(ma, heap, size, align, &slab, &index_region))
        return false;

    bool noncoherent = (slab->flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                      !(slab->flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (noncoherent) {
        size = PL_ALIGN(size, vk->limits.nonCoherentAtomSize);
        align = pl_lcm(align, vk->limits.nonCoherentAtomSize);
    }

    struct vk_region region = slab->regions[index_region];
    TARRAY_REMOVE_AT(slab->regions, slab->num_regions, index_region);
    VkDeviceSize offset = PL_ALIGN(region.start, align);
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

    PL_DEBUG(vk, "Sub-allocating slice %zu + %zu from slab with size %zu",
             (size_t) out->offset, (size_t) out->size, (size_t) slab->size);

    size_t out_end = out->offset + out->size;
    insert_region(slab, (struct vk_region) { region.start, out->offset });
    insert_region(slab, (struct vk_region) { out_end, region.end });

    slab->used += size;
    return true;
}
