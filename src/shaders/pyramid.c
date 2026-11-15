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
#include "shaders/pyramid.h"

// Destroys every level of `pyramid`, along with the array holding them.
void pl_pyramid_destroy(pl_gpu gpu, pl_tex_pyramid *pyramid)
{
    for (int i = 0; i < pyramid->num; i++)
        pl_tex_destroy(gpu, &pyramid->elem[i]);
    pl_free_ptr(&pyramid->elem);
    pyramid->num = 0;
}

// Number of levels before degenerating into a single pixel.
int pl_pyramid_max_levels(int w, int h)
{
    int levels = 1;
    while (w > 1 || h > 1) {
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        levels++;
    }
    return levels;
}

bool pl_pyramid_recreate(pl_gpu gpu, void *alloc, pl_tex_pyramid *pyramid,
                         int levels, const struct pl_tex_params *params)
{
    for (int i = levels; i < pyramid->num; i++)
        pl_tex_destroy(gpu, &pyramid->elem[i]);

    PL_ARRAY_RESIZE(alloc, *pyramid, levels);
    for (int i = pyramid->num; i < levels; i++)
        pyramid->elem[i] = NULL;
    pyramid->num = levels;

    struct pl_tex_params level = *params;
    for (int i = 0; i < levels; i++) {
        pl_tex tex = pyramid->elem[i];
        if (!(tex && pl_tex_params_superset(tex->params, level)) &&
            !pl_tex_recreate(gpu, &pyramid->elem[i], &level))
        {
            PL_ERR(gpu, "Failed creating %dx%d image pyramid level!",
                   level.w, level.h);
            return false;
        }

        level.w = (level.w + 1) / 2;
        level.h = (level.h + 1) / 2;
    }

    return true;
}

// Downscale `src` by one pyramid level into `dst` using the Burt-Adelson REDUCE
// filter, i.e. convolution with the separable binomial kernel with a weighting
// of [1 4 6 4 1]/16. `dst` must measure exactly `src` halved in each dimension,
// rounding up.
static bool pyramid_reduce(pl_dispatch dispatch, pl_tex src, pl_tex dst)
{
    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "pyramid REDUCE %dx%d -> %dx%d", src->params.w,
                 src->params.h, dst->params.w, dst->params.h);
    sh->output = PL_SHADER_SIG_COLOR;

    // Align the output texel (i,j) on the center of source texel (2i,2j).
    const pl_rect2df rect = {
        .x0 = -0.5f,
        .y0 = -0.5f,
        .x1 = 2.0f * dst->params.w - 0.5f,
        .y1 = 2.0f * dst->params.h - 0.5f,
    };

    ident_t pos, pt;
    ident_t tex = sh_bind(sh, src, PL_TEX_ADDRESS_MIRROR, PL_TEX_SAMPLE_LINEAR,
                          "src", &rect, &pos, &pt);
    if (!tex) {
        pl_dispatch_abort(dispatch, &sh);
        return false;
    }

    // clang-format off
#pragma GLSL /* pyramid_reduce */                                               \
    vec4 color;                                                                 \
    {                                                                           \
        vec2 c = $pos;                                                          \
        vec2 d = 1.2 * $pt;                                                     \
        float sum  = 36.0 * textureLod($tex, c, 0.0).r;                         \
        sum += 30.0 * textureLod($tex, c - vec2(d.x, 0.0), 0.0).r;              \
        sum += 30.0 * textureLod($tex, c + vec2(d.x, 0.0), 0.0).r;              \
        sum += 30.0 * textureLod($tex, c - vec2(0.0, d.y), 0.0).r;              \
        sum += 30.0 * textureLod($tex, c + vec2(0.0, d.y), 0.0).r;              \
        sum += 25.0 * textureLod($tex, c - d, 0.0).r;                           \
        sum += 25.0 * textureLod($tex, c + d, 0.0).r;                           \
        sum += 25.0 * textureLod($tex, c - vec2(d.x, -d.y), 0.0).r;             \
        sum += 25.0 * textureLod($tex, c + vec2(d.x, -d.y), 0.0).r;             \
        color = vec4(sum * (1.0 / 256.0), 0.0, 0.0, 1.0);                       \
    }
    // clang-format on

    return pl_dispatch_finish(dispatch, pl_dispatch_params(
        .shader = &sh,
        .target = dst,
    ));
}

// Fills every level of `pyramid` but the first by reducing the level before it.
bool pl_pyramid_build(pl_gpu gpu, pl_dispatch dispatch,
                      const pl_tex_pyramid *pyramid)
{
    for (int i = 1; i < pyramid->num; i++) {
        if (!pyramid_reduce(dispatch, pyramid->elem[i-1], pyramid->elem[i])) {
            PL_ERR(gpu, "Failed reducing image pyramid level %d!", i);
            return false;
        }
    }

    return true;
}
