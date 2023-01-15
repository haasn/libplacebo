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

#include "../common.h"
#include "log.h"

const struct pl_vk_inst_params pl_vk_inst_default_params = {0};
const struct pl_vulkan_params pl_vulkan_default_params = { PL_VULKAN_DEFAULTS };

pl_vk_inst pl_vk_inst_create(pl_log log, const struct pl_vk_inst_params *params)
{
    pl_fatal(log, "libplacebo compiled without Vulkan support!");
    return NULL;
}

void pl_vk_inst_destroy(pl_vk_inst *pinst)
{
    pl_vk_inst inst = *pinst;
    pl_assert(!inst);
}

pl_vulkan pl_vulkan_create(pl_log log, const struct pl_vulkan_params *params)
{
    pl_fatal(log, "libplacebo compiled without Vulkan support!");
    return NULL;
}

void pl_vulkan_destroy(pl_vulkan *pvk)
{
    pl_vulkan vk = *pvk;
    pl_assert(!vk);
}

pl_vulkan pl_vulkan_get(pl_gpu gpu)
{
    return NULL;
}

VkPhysicalDevice pl_vulkan_choose_device(pl_log log,
                              const struct pl_vulkan_device_params *params)
{
    pl_err(log, "libplacebo compiled without Vulkan support!");
    return NULL;
}

pl_swapchain pl_vulkan_create_swapchain(pl_vulkan vk,
                              const struct pl_vulkan_swapchain_params *params)
{
    pl_unreachable();
}

bool pl_vulkan_swapchain_suboptimal(pl_swapchain sw)
{
    pl_unreachable();
}

pl_vulkan pl_vulkan_import(pl_log log, const struct pl_vulkan_import_params *params)
{
    pl_fatal(log, "libplacebo compiled without Vulkan support!");
    return NULL;
}

pl_tex pl_vulkan_wrap(pl_gpu gpu, const struct pl_vulkan_wrap_params *params)
{
    pl_unreachable();
}

VkImage pl_vulkan_unwrap(pl_gpu gpu, pl_tex tex,
                         VkFormat *out_format, VkImageUsageFlags *out_flags)
{
    pl_unreachable();
}

bool pl_vulkan_hold_ex(pl_gpu gpu, const struct pl_vulkan_hold_params *params)
{
    pl_unreachable();
}

void pl_vulkan_release_ex(pl_gpu gpu, const struct pl_vulkan_release_params *params)
{
    pl_unreachable();
}

VkSemaphore pl_vulkan_sem_create(pl_gpu gpu, const struct pl_vulkan_sem_params *params)
{
    pl_unreachable();
}

void pl_vulkan_sem_destroy(pl_gpu gpu, VkSemaphore *semaphore)
{
    pl_unreachable();
}
