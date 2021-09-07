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

bool pl_needs_fg_av1(const struct pl_film_grain_params *);

bool pl_shader_fg_av1(pl_shader, pl_shader_obj *, const struct pl_film_grain_params *);

// Common helper function
static inline enum pl_channel channel_map(int i, const struct pl_film_grain_params *params)
{
    static const enum pl_channel map_rgb[3] = {
        [PL_CHANNEL_G] = PL_CHANNEL_Y,
        [PL_CHANNEL_B] = PL_CHANNEL_CB,
        [PL_CHANNEL_R] = PL_CHANNEL_CR,
    };

    static const enum pl_channel map_xyz[3] = {
        [1] = PL_CHANNEL_Y,  // Y
        [2] = PL_CHANNEL_CB, // Z
        [0] = PL_CHANNEL_CR, // X
    };

    if (i >= params->components)
        return PL_CHANNEL_NONE;

    int comp = params->component_mapping[i];
    if (comp < 0 || comp > 2)
        return PL_CHANNEL_NONE;

    switch (params->repr->sys) {
    case PL_COLOR_SYSTEM_UNKNOWN:
    case PL_COLOR_SYSTEM_RGB:
        return map_rgb[comp];

    case PL_COLOR_SYSTEM_XYZ:
        return map_xyz[comp];

    case PL_COLOR_SYSTEM_BT_601:
    case PL_COLOR_SYSTEM_BT_709:
    case PL_COLOR_SYSTEM_SMPTE_240M:
    case PL_COLOR_SYSTEM_BT_2020_NC:
    case PL_COLOR_SYSTEM_BT_2020_C:
    case PL_COLOR_SYSTEM_BT_2100_PQ:
    case PL_COLOR_SYSTEM_BT_2100_HLG:
    case PL_COLOR_SYSTEM_YCGCO:
        return comp;

    case PL_COLOR_SYSTEM_COUNT:
        break;
    }

    pl_unreachable();
}
