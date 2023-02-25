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

#include "shaders.h"
#include "shaders/film_grain.h"

bool pl_needs_film_grain(const struct pl_film_grain_params *params)
{
    switch (params->data.type) {
    case PL_FILM_GRAIN_NONE: return false;
    case PL_FILM_GRAIN_AV1:  return pl_needs_fg_av1(params);
    case PL_FILM_GRAIN_H274: return pl_needs_fg_h274(params);
    default: pl_unreachable();
    }
}

struct sh_grain_obj {
    pl_shader_obj av1;
    pl_shader_obj h274;
};

static void sh_grain_uninit(pl_gpu gpu, void *ptr)
{
    struct sh_grain_obj *obj = ptr;
    pl_shader_obj_destroy(&obj->av1);
    pl_shader_obj_destroy(&obj->h274);
}

bool pl_shader_film_grain(pl_shader sh, pl_shader_obj *grain_state,
                          const struct pl_film_grain_params *params)
{
    if (!pl_needs_film_grain(params)) {
        // FIXME: Instead of erroring, sample directly
        SH_FAIL(sh, "pl_shader_film_grain called but no film grain needs to be "
                    "applied, test with `pl_needs_film_grain` first!");
        return false;
    }

    struct sh_grain_obj *obj;
    obj = SH_OBJ(sh, grain_state, PL_SHADER_OBJ_FILM_GRAIN,
                 struct sh_grain_obj, sh_grain_uninit);
    if (!obj)
        return false;

    switch (params->data.type) {
    case PL_FILM_GRAIN_NONE: return false;
    case PL_FILM_GRAIN_AV1:  return pl_shader_fg_av1(sh, &obj->av1, params);
    case PL_FILM_GRAIN_H274: return pl_shader_fg_h274(sh, &obj->h274, params);
    default: pl_unreachable();
    }
}
