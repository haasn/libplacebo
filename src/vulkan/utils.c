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

#include "utils.h"

VkExternalMemoryHandleTypeFlagBitsKHR
vk_mem_handle_type(enum pl_handle_type handle_type)
{
    if (!handle_type)
        return 0;

    switch (handle_type) {
    case PL_HANDLE_FD:
        return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    case PL_HANDLE_WIN32:
        return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
    case PL_HANDLE_WIN32_KMT:
        return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR;
    case PL_HANDLE_DMA_BUF:
        return VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    case PL_HANDLE_HOST_PTR:
        return VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    }

    abort();
}

VkExternalSemaphoreHandleTypeFlagBitsKHR
vk_sync_handle_type(enum pl_handle_type handle_type)
{
    if (!handle_type)
        return 0;

    switch (handle_type) {
    case PL_HANDLE_FD:
        return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    case PL_HANDLE_WIN32:
        return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
    case PL_HANDLE_WIN32_KMT:
        return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR;
    case PL_HANDLE_DMA_BUF:
    case PL_HANDLE_HOST_PTR:
        return 0;
    }

    abort();
}

bool vk_external_mem_check(const VkExternalMemoryPropertiesKHR *props,
                           enum pl_handle_type handle_type,
                           bool import)
{
    VkExternalMemoryFeatureFlagsKHR flags = props->externalMemoryFeatures;

    // No support for this handle type;
    if (!(props->compatibleHandleTypes & vk_mem_handle_type(handle_type)))
        return false;

    if (import) {
        if (!(flags & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT_KHR))
            return false;
    } else {
        if (!(flags & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT_KHR))
            return false;
    }

    // We can't handle VkMemoryDedicatedAllocateInfo currently. (Maybe soon?)
    if (flags & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT_KHR)
        return false;

    return true;
}

const enum pl_handle_type vk_mem_handle_list[] = {
        PL_HANDLE_HOST_PTR,
#ifdef VK_HAVE_UNIX
        PL_HANDLE_FD,
        PL_HANDLE_DMA_BUF,
#endif
#ifdef VK_HAVE_WIN32
        PL_HANDLE_WIN32,
        PL_HANDLE_WIN32_KMT,
#endif
        0
};

const enum pl_handle_type vk_sync_handle_list[] = {
#ifdef VK_HAVE_UNIX
        PL_HANDLE_FD,
#endif
#ifdef VK_HAVE_WIN32
        PL_HANDLE_WIN32,
        PL_HANDLE_WIN32_KMT,
#endif
        0
};

const void *vk_find_struct(const void *chain, VkStructureType stype)
{
    const VkBaseInStructure *in = chain;
    while (in) {
        if (in->sType == stype)
            return in;

        in = in->pNext;
    }

    return NULL;
}

void vk_link_struct(void *chain, void *in)
{
    if (!in)
        return;

    VkBaseOutStructure *out = chain;
    while (out->pNext)
        out = out->pNext;

    out->pNext = in;
}

void *vk_struct_memdup(void *tactx, const void *pin)
{
    if (!pin)
        return NULL;

    const VkBaseInStructure *in = pin;
    size_t size = vk_struct_size(in->sType);
    if (!size)
        return NULL;

    VkBaseOutStructure *out = talloc_memdup(tactx, in, size);
    out->pNext = NULL;
    return out;
}

void *vk_chain_memdup(void *tactx, const void *pin)
{
    const VkBaseInStructure *in = pin;
    VkBaseOutStructure *out = vk_struct_memdup(tactx, in);
    if (!out)
        return NULL;

    out->pNext = vk_chain_memdup(tactx, in->pNext);
    return out;
}
