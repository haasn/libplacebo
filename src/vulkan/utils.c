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
    case PL_HANDLE_DMA_BUF: abort();
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
