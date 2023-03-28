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
#include <unistd.h>
#endif

// Controls the page size alignment, to help coalesce allocations into the same
// slab. Pages are rounded up to multiples of this value. (Default: 4 KB)
#define PAGE_SIZE_ALIGN (1LLU << 12)

// Controls the minimum/maximum number of pages for new slabs. As slabs are
// exhausted of memory, the number of pages per new slab grows exponentially,
// starting with the minimum until the maximum is reached.
//
// Note: The maximum must never exceed the size of `vk_slab.spacemap`.
#define MINIMUM_PAGE_COUNT 4
#define MAXIMUM_PAGE_COUNT (sizeof(uint64_t) * 8)

// Controls the maximum page size. Any allocations above this threshold
// will be served by dedicated allocations. (Default: 64 MB)
#define MAXIMUM_PAGE_SIZE (1LLU << 26)

// Controls the minimum slab size, to avoid excessive re-allocation of very
// small slabs. (Default: 256 KB)
#define MINIMUM_SLAB_SIZE (1LLU << 18)

// Controls the maximum slab size, to avoid ballooning memory requirements
// due to overzealous allocation of extra pages. (Default: 256 MB)
#define MAXIMUM_SLAB_SIZE (1LLU << 28)

// How long to wait before garbage collecting empty slabs. Slabs older than
// this many invocations of `vk_malloc_garbage_collect` will be released.
#define MAXIMUM_SLAB_AGE 8

// A single slab represents a contiguous region of allocated memory. Actual
// allocations are served as pages of this. Slabs are organized into pools,
// each of which contains a list of slabs of differing page sizes.
struct vk_slab {
    pl_mutex lock;
    pl_debug_tag debug_tag; // debug tag of the triggering allocation
    VkDeviceMemory mem;     // underlying device allocation
    VkDeviceSize size;      // total allocated size of `mem`
    VkMemoryType mtype;     // underlying memory type
    bool dedicated;         // slab is allocated specifically for one object
    bool imported;          // slab represents an imported memory allocation

    // free space accounting (only for non-dedicated slabs)
    uint64_t spacemap;      // bitset of available pages
    size_t pagesize;        // size in bytes per page
    size_t used;            // number of bytes actually in use
    uint64_t age;           // timestamp of last use

    // optional, depends on the memory type:
    VkBuffer buffer;        // buffer spanning the entire slab
    void *data;             // mapped memory corresponding to `mem`
    bool coherent;          // mapped memory is coherent
    union pl_handle handle; // handle associated with this device memory
    enum pl_handle_type handle_type;
};

// Represents a single memory pool. We keep track of a vk_pool for each
// combination of malloc parameters. This shouldn't actually be that many in
// practice, because some combinations simply never occur, and others will
// generally be the same for the same objects.
//
// Note: `vk_pool` addresses are not immutable, so we mustn't expose any
// dangling references to a `vk_pool` from e.g. `vk_memslice.priv = vk_slab`.
struct vk_pool {
    struct vk_malloc_params params;   // allocation params (with some fields nulled)
    PL_ARRAY(struct vk_slab *) slabs; // array of slabs, unsorted
    int index;                        // running index in `vk_malloc.pools`
};

// The overall state of the allocator, which keeps track of a vk_pool for each
// memory type.
struct vk_malloc {
    struct vk_ctx *vk;
    pl_mutex lock;
    VkPhysicalDeviceMemoryProperties props;
    PL_ARRAY(struct vk_pool) pools;
    uint64_t age;
};

static inline float efficiency(size_t used, size_t total)
{
    if (!total)
        return 100.0;

    return 100.0f * used / total;
}

static const char *print_size(char buf[8], size_t size)
{
    const char *suffixes = "\0KMG";
    while (suffixes[1] && size > 9999) {
        size >>= 10;
        suffixes++;
    }

    int ret = *suffixes ? snprintf(buf, 8, "%4zu%c", size, *suffixes)
                        : snprintf(buf, 8, "%5zu", size);

    return ret >= 0 ? buf : "(error)";
}

#define PRINT_SIZE(x) (print_size((char[8]){0}, (size_t) (x)))

void vk_malloc_print_stats(struct vk_malloc *ma, enum pl_log_level lev)
{
    struct vk_ctx *vk = ma->vk;
    size_t total_size = 0;
    size_t total_used = 0;
    size_t total_res = 0;

    PL_MSG(vk, lev, "Memory heaps supported by device:");
    for (int i = 0; i < ma->props.memoryHeapCount; i++) {
        VkMemoryHeap heap = ma->props.memoryHeaps[i];
        PL_MSG(vk, lev, "    %d: flags 0x%x size %s",
                i, (unsigned) heap.flags, PRINT_SIZE(heap.size));
    }

    PL_DEBUG(vk, "Memory types supported by device:");
    for (int i = 0; i < ma->props.memoryTypeCount; i++) {
        VkMemoryType type = ma->props.memoryTypes[i];
        PL_DEBUG(vk, "    %d: flags 0x%x heap %d",
                 i, (unsigned) type.propertyFlags, (int) type.heapIndex);
    }

    pl_mutex_lock(&ma->lock);
    for (int i = 0; i < ma->pools.num; i++) {
        struct vk_pool *pool = &ma->pools.elem[i];
        const struct vk_malloc_params *par = &pool->params;

        PL_MSG(vk, lev, "Memory pool %d:", i);
        PL_MSG(vk, lev, "    Compatible types: 0x%"PRIx32, par->reqs.memoryTypeBits);
        if (par->required)
            PL_MSG(vk, lev, "    Required flags: 0x%"PRIx32, par->required);
        if (par->optimal)
            PL_MSG(vk, lev, "    Optimal flags: 0x%"PRIx32, par->optimal);
        if (par->buf_usage)
            PL_MSG(vk, lev, "    Buffer flags: 0x%"PRIx32, par->buf_usage);
        if (par->export_handle)
            PL_MSG(vk, lev, "    Export handle: 0x%x", par->export_handle);

        size_t pool_size = 0;
        size_t pool_used = 0;
        size_t pool_res = 0;

        for (int j = 0; j < pool->slabs.num; j++) {
            struct vk_slab *slab = pool->slabs.elem[j];
            pl_mutex_lock(&slab->lock);

            size_t avail = __builtin_popcountll(slab->spacemap) * slab->pagesize;
            size_t slab_res = slab->size - avail;

            PL_MSG(vk, lev, "    Slab %2d: %8"PRIx64" x %s: "
                   "%s used %s res %s alloc from heap %d, efficiency %.2f%%  [%s]",
                   j, slab->spacemap, PRINT_SIZE(slab->pagesize),
                   PRINT_SIZE(slab->used), PRINT_SIZE(slab_res),
                   PRINT_SIZE(slab->size), (int) slab->mtype.heapIndex,
                   efficiency(slab->used, slab_res),
                   PL_DEF(slab->debug_tag, "unknown"));

            pool_size += slab->size;
            pool_used += slab->used;
            pool_res += slab_res;
            pl_mutex_unlock(&slab->lock);
        }

        PL_MSG(vk, lev, "    Pool summary: %s used %s res %s alloc, "
               "efficiency %.2f%%, utilization %.2f%%",
               PRINT_SIZE(pool_used), PRINT_SIZE(pool_res),
               PRINT_SIZE(pool_size), efficiency(pool_used, pool_res),
               efficiency(pool_res, pool_size));

        total_size += pool_size;
        total_used += pool_used;
        total_res += pool_res;
    }
    pl_mutex_unlock(&ma->lock);

    PL_MSG(vk, lev, "Memory summary: %s used %s res %s alloc, "
           "efficiency %.2f%%, utilization %.2f%%",
           PRINT_SIZE(total_used), PRINT_SIZE(total_res),
           PRINT_SIZE(total_size), efficiency(total_used, total_res),
           efficiency(total_res, total_size));
}

static void slab_free(struct vk_ctx *vk, struct vk_slab *slab)
{
    if (!slab)
        return;

#ifndef NDEBUG
    if (!slab->dedicated && slab->used > 0) {
        PL_WARN(vk, "Leaked %zu bytes of vulkan memory!", slab->used);
        PL_WARN(vk, "slab total size: %zu bytes, heap: %d, flags: 0x%"PRIX64,
                (size_t) slab->size, (int) slab->mtype.heapIndex,
                (uint64_t) slab->mtype.propertyFlags);
        if (slab->debug_tag)
            PL_WARN(vk, "last used for: %s", slab->debug_tag);
        pl_log_stack_trace(vk->log, PL_LOG_WARN);
        pl_debug_abort();
    }
#endif

    if (slab->imported) {
        switch (slab->handle_type) {
        case PL_HANDLE_FD:
        case PL_HANDLE_DMA_BUF:
            PL_TRACE(vk, "Unimporting slab of size %s from fd: %d",
                     PRINT_SIZE(slab->size), slab->handle.fd);
            break;
        case PL_HANDLE_WIN32:
        case PL_HANDLE_WIN32_KMT:
#ifdef PL_HAVE_WIN32
            PL_TRACE(vk, "Unimporting slab of size %s from handle: %p",
                     PRINT_SIZE(slab->size), (void *) slab->handle.handle);
#endif
            break;
        case PL_HANDLE_HOST_PTR:
            PL_TRACE(vk, "Unimporting slab of size %s from ptr: %p",
                     PRINT_SIZE(slab->size), (void *) slab->handle.ptr);
            break;
        case PL_HANDLE_IOSURFACE:
        case PL_HANDLE_MTL_TEX:
            pl_unreachable();
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
        case PL_HANDLE_IOSURFACE:
        case PL_HANDLE_MTL_TEX:
            pl_unreachable();
        }

        PL_DEBUG(vk, "Freeing slab of size %s", PRINT_SIZE(slab->size));
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

    if (!info.handleType)
        return false;

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
        .age = ma->age,
        .size = params->reqs.size,
        .handle_type = params->export_handle,
        .debug_tag = params->debug_tag,
    };
    pl_mutex_init(&slab->lock);

    switch (slab->handle_type) {
    case PL_HANDLE_FD:
    case PL_HANDLE_DMA_BUF:
        slab->handle.fd = -1;
        break;
    case PL_HANDLE_WIN32:
    case PL_HANDLE_WIN32_KMT:
    case PL_HANDLE_MTL_TEX:
    case PL_HANDLE_IOSURFACE:
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
        pl_assert(vk->pools.num <= PL_ARRAY_SIZE(qfs));
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
        PL_ERR(vk, "Allocation of size %s failed: %s!",
               PRINT_SIZE(slab->size), vk_res_str(res));
        vk_malloc_print_stats(ma, PL_LOG_ERR);
        pl_log_stack_trace(vk->log, PL_LOG_ERR);
        goto error;

    default:
        PL_VK_ASSERT(res, "vkAllocateMemory");
    }

    slab->mtype = *mtype;
    if (mtype->propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        VK(vk->MapMemory(vk->dev, slab->mem, 0, VK_WHOLE_SIZE, 0, &slab->data));
        slab->coherent = mtype->propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
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

    // free space accounting is done by the caller
    return slab;

error:
    if (params->debug_tag)
        PL_ERR(vk, "  for malloc: %s", params->debug_tag);
    slab_free(vk, slab);
    return NULL;
}

static void pool_uninit(struct vk_ctx *vk, struct vk_pool *pool)
{
    for (int i = 0; i < pool->slabs.num; i++)
        slab_free(vk, pool->slabs.elem[i]);

    pl_free(pool->slabs.elem);
    *pool = (struct vk_pool) {0};
}

struct vk_malloc *vk_malloc_create(struct vk_ctx *vk)
{
    struct vk_malloc *ma = pl_zalloc_ptr(NULL, ma);
    pl_mutex_init(&ma->lock);
    vk->GetPhysicalDeviceMemoryProperties(vk->physd, &ma->props);
    ma->vk = vk;

    vk_malloc_print_stats(ma, PL_LOG_INFO);
    return ma;
}

void vk_malloc_destroy(struct vk_malloc **ma_ptr)
{
    struct vk_malloc *ma = *ma_ptr;
    if (!ma)
        return;

    vk_malloc_print_stats(ma, PL_LOG_DEBUG);
    for (int i = 0; i < ma->pools.num; i++)
        pool_uninit(ma->vk, &ma->pools.elem[i]);

    pl_mutex_destroy(&ma->lock);
    pl_free_ptr(ma_ptr);
}

void vk_malloc_garbage_collect(struct vk_malloc *ma)
{
    struct vk_ctx *vk = ma->vk;

    pl_mutex_lock(&ma->lock);
    ma->age++;

    for (int i = 0; i < ma->pools.num; i++) {
        struct vk_pool *pool = &ma->pools.elem[i];
        for (int n = 0; n < pool->slabs.num; n++) {
            struct vk_slab *slab = pool->slabs.elem[n];
            pl_mutex_lock(&slab->lock);
            if (slab->used || (ma->age - slab->age) <= MAXIMUM_SLAB_AGE) {
                pl_mutex_unlock(&slab->lock);
                continue;
            }

            PL_DEBUG(vk, "Garbage collected slab of size %s from pool %d",
                     PRINT_SIZE(slab->size), pool->index);

            pl_mutex_unlock(&slab->lock);
            slab_free(ma->vk, slab);
            PL_ARRAY_REMOVE_AT(pool->slabs, n--);
        }
    }

    pl_mutex_unlock(&ma->lock);
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
    if (!slab || slab->dedicated) {
        slab_free(vk, slab);
        goto done;
    }

    pl_mutex_lock(&slab->lock);

    int page_idx = slice->offset / slab->pagesize;
    slab->spacemap |= 0x1LLU << page_idx;
    slab->used -= slice->size;
    slab->age = ma->age;
    pl_assert(slab->used >= 0);

    pl_mutex_unlock(&slab->lock);

done:
    *slice = (struct vk_memslice) {0};
}

static inline bool pool_params_eq(const struct vk_malloc_params *a,
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

static struct vk_pool *find_pool(struct vk_malloc *ma,
                                 const struct vk_malloc_params *params)
{
    pl_assert(!params->import_handle);
    pl_assert(!params->ded_image);

    struct vk_malloc_params fixed = *params;
    fixed.reqs.alignment = 0;
    fixed.reqs.size = 0;
    fixed.shared_mem = (struct pl_shared_mem) {0};

    for (int i = 0; i < ma->pools.num; i++) {
        if (pool_params_eq(&ma->pools.elem[i].params, &fixed))
            return &ma->pools.elem[i];
    }

    // Not found => add it
    PL_ARRAY_GROW(ma, ma->pools);
    size_t idx = ma->pools.num++;
    ma->pools.elem[idx] = (struct vk_pool) {
        .params = fixed,
        .index = idx,
    };
    return &ma->pools.elem[idx];
}

// Returns a suitable memory page from the pool. A new slab will be allocated
// under the hood, if necessary.
//
// Note: This locks the slab it returns
static struct vk_slab *pool_get_page(struct vk_malloc *ma, struct vk_pool *pool,
                                     size_t size, size_t align,
                                     VkDeviceSize *offset)
{
    struct vk_slab *slab = NULL;
    int slab_pages = MINIMUM_PAGE_COUNT;
    size = PL_ALIGN2(size, PAGE_SIZE_ALIGN);
    const size_t pagesize = PL_ALIGN(size, align);

    for (int i = 0; i < pool->slabs.num; i++) {
        slab = pool->slabs.elem[i];
        if (slab->pagesize < size)
            continue;
        if (slab->pagesize > pagesize * MINIMUM_PAGE_COUNT) // rough heuristic
            continue;
        if (slab->pagesize % align)
            continue;

        pl_mutex_lock(&slab->lock);
        int page_idx = __builtin_ffsll(slab->spacemap);
        if (!page_idx--) {
            pl_mutex_unlock(&slab->lock);
            // Increase the number of slabs to allocate for new slabs the
            // more existing full slabs exist for this size range
            slab_pages = PL_MIN(slab_pages << 1, MAXIMUM_PAGE_COUNT);
            continue;
        }

        slab->spacemap ^= 0x1LLU << page_idx;
        *offset = page_idx * slab->pagesize;
        return slab;
    }

    // Otherwise, allocate a new vk_slab and append it to the list.
    VkDeviceSize slab_size = slab_pages * pagesize;
    pl_static_assert(MINIMUM_SLAB_SIZE <= PAGE_SIZE_ALIGN * MAXIMUM_PAGE_COUNT);
    slab_size = PL_CLAMP(slab_size, MINIMUM_SLAB_SIZE, MAXIMUM_SLAB_SIZE);
    slab_pages = slab_size / pagesize;

    struct vk_malloc_params params = pool->params;
    params.reqs.size = slab_size;

    // Don't hold the lock while allocating the slab, because it can be a
    // potentially very costly operation.
    pl_mutex_unlock(&ma->lock);
    slab = slab_alloc(ma, &params);
    pl_mutex_lock(&ma->lock);
    if (!slab)
        return NULL;
    pl_mutex_lock(&slab->lock);

    slab->spacemap = (slab_pages == sizeof(uint64_t) * 8) ? ~0LLU : ~(~0LLU << slab_pages);
    slab->pagesize = pagesize;
    PL_ARRAY_APPEND(NULL, pool->slabs, slab);

    // Return the first page in this newly allocated slab
    slab->spacemap ^= 0x1;
    *offset = 0;
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
        pl_assert(vk->pools.num <= PL_ARRAY_SIZE(qfs));
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
    case PL_HANDLE_IOSURFACE:
    case PL_HANDLE_MTL_TEX:
        PL_ERR(vk, "vk_malloc_import: unsupported handle type %d",
               params->import_handle);
        goto error;
    }

    if (!find_best_memtype(ma, reqs.memoryTypeBits, params, &ainfo.memoryTypeIndex)) {
        PL_ERR(vk, "No compatible memory types offered for imported memory!");
        goto error;
    }

    VkDeviceMemory vkmem = VK_NULL_HANDLE;
    VK(vk->AllocateMemory(vk->dev, &ainfo, PL_VK_ALLOC, &vkmem));

    slab = pl_alloc_ptr(NULL, slab);
    *slab = (struct vk_slab) {
        .mem = vkmem,
        .dedicated = true,
        .imported = true,
        .buffer = buffer,
        .size = shmem->size,
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
        PL_TRACE(vk, "Imported %s bytes from fd: %d%s",
                 PRINT_SIZE(slab->size), shmem->handle.fd,
                 params->ded_image ? " (dedicated)" : "");
        // fd ownership is transferred at this point.
        slab->handle.fd = fdinfo.fd;
        fdinfo.fd = -1;
        break;
    case PL_HANDLE_HOST_PTR:
        PL_TRACE(vk, "Imported %s bytes from ptr: %p%s",
                 PRINT_SIZE(slab->size), shmem->handle.ptr,
                 params->ded_image ? " (dedicated" : "");
        slab->handle.ptr = ptrinfo.pHostPointer;
        break;
    case PL_HANDLE_WIN32:
    case PL_HANDLE_WIN32_KMT:
    case PL_HANDLE_IOSURFACE:
    case PL_HANDLE_MTL_TEX:
        break;
    }

    VkMemoryPropertyFlags flags = ma->props.memoryTypes[ainfo.memoryTypeIndex].propertyFlags;
    if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        VK(vk->MapMemory(vk->dev, slab->mem, 0, VK_WHOLE_SIZE, 0, &slab->data));
        slab->coherent = flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        out->data = (uint8_t *) slab->data + out->offset;
        out->coherent = slab->coherent;
        if (!slab->coherent) {
            // Use entire buffer range, since this is a dedicated memory
            // allocation. This avoids issues with noncoherent atomicity
            out->map_offset = 0;
            out->map_size = VK_WHOLE_SIZE;

            // Mapping does not implicitly invalidate mapped memory
            VK(vk->InvalidateMappedMemoryRanges(vk->dev, 1, &(VkMappedMemoryRange) {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = slab->mem,
                .offset = out->map_offset,
                .size = out->map_size,
            }));
        }
    }

    if (buffer)
        VK(vk->BindBufferMemory(vk->dev, buffer, vkmem, 0));

    return true;

error:
    if (params->debug_tag)
        PL_ERR(vk, "  for malloc: %s", params->debug_tag);
    vk->DestroyBuffer(vk->dev, buffer, PL_VK_ALLOC);
#ifdef PL_HAVE_UNIX
    if (fdinfo.fd > -1)
        close(fdinfo.fd);
#endif
    pl_free(slab);
    *out = (struct vk_memslice) {0};
    return false;
}

size_t vk_malloc_avail(struct vk_malloc *ma, VkMemoryPropertyFlags flags)
{
    size_t avail = 0;
    for (int i = 0; i < ma->props.memoryTypeCount; i++) {
        const VkMemoryType *mtype = &ma->props.memoryTypes[i];
        if ((mtype->propertyFlags & flags) != flags)
            continue;
        avail = PL_MAX(avail, ma->props.memoryHeaps[mtype->heapIndex].size);
    }

    return avail;
}

bool vk_malloc_slice(struct vk_malloc *ma, struct vk_memslice *out,
                     const struct vk_malloc_params *params)
{
    struct vk_ctx *vk = ma->vk;
    pl_assert(!params->import_handle || !params->export_handle);
    if (params->import_handle)
        return vk_malloc_import(ma, out, params);

    pl_assert(params->reqs.size);
    size_t size = params->reqs.size;
    size_t align = params->reqs.alignment;
    align = pl_lcm(align, vk->props.limits.bufferImageGranularity);
    align = pl_lcm(align, vk->props.limits.nonCoherentAtomSize);

    struct vk_slab *slab;
    VkDeviceSize offset;

    if (params->ded_image || size > MAXIMUM_PAGE_SIZE) {
        slab = slab_alloc(ma, params);
        if (!slab)
            return false;
        slab->dedicated = true;
        offset = 0;
    } else {
        pl_mutex_lock(&ma->lock);
        struct vk_pool *pool = find_pool(ma, params);
        slab = pool_get_page(ma, pool, size, align, &offset);
        pl_mutex_unlock(&ma->lock);
        if (!slab) {
            PL_ERR(ma->vk, "No slab to serve request for %s bytes (with "
                   "alignment 0x%zx) in pool %d!",
                   PRINT_SIZE(size), align, pool->index);
            return false;
        }

        // For accounting, just treat the alignment as part of the used size.
        // Doing it this way makes sure that the sizes reported to vk_memslice
        // consumers are always aligned properly.
        size = PL_ALIGN(size, align);
        slab->used += size;
        slab->age = ma->age;
        if (params->debug_tag)
            slab->debug_tag = params->debug_tag;
        pl_mutex_unlock(&slab->lock);
    }

    pl_assert(offset % align == 0);
    *out = (struct vk_memslice) {
        .vkmem = slab->mem,
        .offset = offset,
        .size = size,
        .buf = slab->buffer,
        .data = slab->data ? (uint8_t *) slab->data + offset : 0x0,
        .coherent = slab->coherent,
        .map_offset = slab->data ? offset : 0,
        .map_size = slab->data ? size : 0,
        .priv = slab,
        .shared_mem = {
            .handle = slab->handle,
            .offset = offset,
            .size = slab->size,
        },
    };
    return true;
}
