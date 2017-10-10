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

const char *vk_res_str(VkResult res)
{
    switch (res) {
#define CASE(name) case name: return #name
    // success codes
    CASE(VK_SUCCESS);
    CASE(VK_NOT_READY);
    CASE(VK_TIMEOUT);
    CASE(VK_EVENT_SET);
    CASE(VK_EVENT_RESET);
    CASE(VK_INCOMPLETE);
    CASE(VK_SUBOPTIMAL_KHR);
    // error codes
    CASE(VK_ERROR_OUT_OF_HOST_MEMORY);
    CASE(VK_ERROR_OUT_OF_DEVICE_MEMORY);
    CASE(VK_ERROR_INITIALIZATION_FAILED);
    CASE(VK_ERROR_DEVICE_LOST);
    CASE(VK_ERROR_MEMORY_MAP_FAILED);
    CASE(VK_ERROR_LAYER_NOT_PRESENT);
    CASE(VK_ERROR_EXTENSION_NOT_PRESENT);
    CASE(VK_ERROR_FEATURE_NOT_PRESENT);
    CASE(VK_ERROR_INCOMPATIBLE_DRIVER);
    CASE(VK_ERROR_TOO_MANY_OBJECTS);
    CASE(VK_ERROR_FORMAT_NOT_SUPPORTED);
    CASE(VK_ERROR_FRAGMENTED_POOL);
    CASE(VK_ERROR_INVALID_SHADER_NV);
    CASE(VK_ERROR_OUT_OF_DATE_KHR);
    CASE(VK_ERROR_SURFACE_LOST_KHR);
    CASE(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR);
    CASE(VK_ERROR_INCOMPATIBLE_DISPLAY_KHR);
    CASE(VK_ERROR_VALIDATION_FAILED_EXT);
    CASE(VK_ERROR_OUT_OF_POOL_MEMORY_KHR);
    CASE(VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR);
    // misc (to satisfy the switch coverage check
    CASE(VK_RESULT_RANGE_SIZE);
    CASE(VK_RESULT_MAX_ENUM);
#undef CASE
    }

    return "Unknown error!";
}
