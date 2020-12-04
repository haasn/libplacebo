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
 * License along with libplacebo.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LIBPLACEBO_DUMMY_H_
#define LIBPLACEBO_DUMMY_H_

#include <libplacebo/gpu.h>

// The functions in this file allow creating and manipulating "dummy" contexts.
// A dummy context isn't actually mapped by the GPU, all data exists purely on
// the CPU. It also isn't capable of compiling or executing any shaders, any
// attempts to do so will simply fail.
//
// The main use case for this dummy context is for users who want to generate
// advanced shaders that depend on specific GLSL features or support for
// certain types of GPU resources (e.g. LUTs). This dummy context allows such
// shaders to be generated, with all of the referenced shader objects and
// textures simply containing their data in a host-accessible way.

struct pl_gpu_dummy_params {
    // These GPU parameters correspond to their equialents in struct `pl_gpu`,
    // and must obey the same rules as documented there. The values from
    // `pl_gpu_dummy_default_params` are set to support pretty much everything
    // and are set for GLSL version 450.
    //
    // Individual fields such as PL_GPU_CAP_COMPUTE or the glsl version
    // description can and should be overriden by the user based on their
    // requirements. Individual limits should ideally be set based on the
    // corresponding `glGet` queries etc.
    pl_gpu_caps caps;
    struct pl_glsl_desc glsl;
    struct pl_gpu_limits limits;
};

extern const struct pl_gpu_dummy_params pl_gpu_dummy_default_params;

// Create a dummy GPU context based on the given parameters. This GPU will have
// a format for each host-representable type (i.e. intN_t, floats and doubles),
// in the canonical channel order RGBA. These formats will have every possible
// capability activated, respectively.
//
// If `params` is left as NULL, it defaults to `&pl_gpu_dummy_params`.
const struct pl_gpu *pl_gpu_dummy_create(struct pl_context *ctx,
                                         const struct pl_gpu_dummy_params *params);

void pl_gpu_dummy_destroy(const struct pl_gpu **gpu);

// Back-doors into the `pl_tex` and `pl_buf` representations. These allow you
// to access the raw data backing this object. Textures are always laid out in
// a tightly packed manner.
//
// For "placeholder" dummy textures, this always returns NULL.
uint8_t *pl_buf_dummy_data(const struct pl_buf *buf);
uint8_t *pl_tex_dummy_data(const struct pl_tex *tex);

// Skeleton of `pl_tex_params` containing only the fields relevant to
// `pl_tex_dummy_create`, plus the extra `sampler_type` field.
struct pl_tex_dummy_params {
    int w, h, d;
    const struct pl_fmt *format;
    enum pl_sampler_type sampler_type;
    void *user_data;

    // Deprecated fields. Ignored. Will be deleted in the future.
    enum pl_tex_sample_mode sample_mode PL_DEPRECATED;
    enum pl_tex_address_mode address_mode PL_DEPRECATED;
};

// Allows creating a "placeholder" dummy texture. This is basically a texture
// that isn't even backed by anything. All `pl_tex_*` operations (other than
// `pl_tex_destroy`) performed on it will simply fail.
//
// All of the permissions will be set to `false`, except `sampleable`, which is
// set to `true`. (So you can use it as an input to shader sampling functions)
const struct pl_tex *pl_tex_dummy_create(const struct pl_gpu *gpu,
                                         const struct pl_tex_dummy_params *params);

#endif // LIBPLACEBO_DUMMY_H_
