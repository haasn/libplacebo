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

#include <stdio.h>
#include "bstr/bstr.h"

#include "common.h"
#include "context.h"
#include "ra.h"

// This represents an identifier (e.g. name of function, uniform etc.) for
// a shader resource. The generated identifiers are immutable, but only live
// until pl_shader_reset - so make copies when passing to external stuff.
typedef const char * ident_t;

enum pl_shader_buf {
    SH_BUF_PRELUDE, // extra #defines etc.
    SH_BUF_HEADER,  // previous passes, helper function definitions, etc.
    SH_BUF_BODY,    // partial contents of the "current" function
    SH_BUF_COUNT,
};

struct pl_shader {
    // Read-only fields
    struct pl_context *ctx;
    const struct ra *ra;

    // Internal state
    struct ta_ref *tmp;
    bool mutable;
    int output_w;
    int output_h;
    struct pl_shader_res res; // for accumulating vertex_attribs etc.
    struct bstr buffers[SH_BUF_COUNT];
    bool is_compute;
    bool flexible_work_groups;
    uint8_t ident;
    uint8_t index;
    int fresh;
};

// Attempt enabling compute shaders for this pass, if possible
bool sh_try_compute(struct pl_shader *sh, int bw, int bh, bool flex, size_t mem);

// Attempt merging a secondary shader into the current shader. Returns NULL if
// merging fails (e.g. incompatible signatures); otherwise returns an identifier
// corresponding to the generated subpass function.
ident_t sh_subpass(struct pl_shader *sh, const struct pl_shader *sub);

// Helpers for adding new variables/descriptors/etc. with fresh, unique
// identifier names. These will never conflcit with other identifiers, even
// if the shaders are merged together.
ident_t sh_fresh(struct pl_shader *sh, const char *name);

// Add a new shader var and return its identifier
ident_t sh_var(struct pl_shader *sh, struct pl_shader_var sv);

// Add a new shader desc and return its identifier. This function takes care of
// setting the binding to a fresh bind point according to the namespace
// requirements, so the caller may leave it blank.
ident_t sh_desc(struct pl_shader *sh, struct pl_shader_desc sd);

// Add a new vec2 vertex attribute from a pl_rect2df, or returns NULL on failure.
ident_t sh_attr_vec2(struct pl_shader *sh, const char *name,
                     const struct pl_rect2df *rc);

// Bind a texture under a given transformation and make its attributes
// available as well. If an output pointer for one of the attributes is left
// as NULL, that attribute will not be added. Returns NULL on failure. `rect`
// is optional, and defaults to the full texture if left as NULL.
//
// Note that for e.g. compute shaders, the vec2 out_pos might be a macro that
// expands to an expensive computation, and should be cached by the user.
ident_t sh_bind(struct pl_shader *sh, const struct ra_tex *tex,
                const char *name, const struct pl_rect2df *rect,
                ident_t *out_pos, ident_t *out_size, ident_t *out_pt);

// Underlying function for appending text to a shader
void pl_shader_append(struct pl_shader *sh, enum pl_shader_buf buf,
                      const char *fmt, ...)
    PRINTF_ATTRIBUTE(3, 4);

#define GLSLP(...) pl_shader_append(sh, SH_BUF_PRELUDE, __VA_ARGS__)
#define GLSLH(...) pl_shader_append(sh, SH_BUF_HEADER, __VA_ARGS__)
#define GLSL(...)  pl_shader_append(sh, SH_BUF_BODY, __VA_ARGS__)

// Requires that the share is mutable, has an output signature compatible
// with the given input signature, as well as an output size compatible with
// the given size requirements. Errors and returns false otherwise.
bool sh_require(struct pl_shader *sh, enum pl_shader_sig insig, int w, int h);

// Shader resources

enum pl_shader_obj_type {
    PL_SHADER_OBJ_INVALID = 0,
    PL_SHADER_OBJ_PEAK_DETECT,
    PL_SHADER_OBJ_SAMPLER,
    PL_SHADER_OBJ_DITHER,
    PL_SHADER_OBJ_LUT,
};

struct pl_shader_obj {
    enum pl_shader_obj_type type;
    const struct ra *ra;
    void (*uninit)(const struct ra *ra, void *priv);
    void *priv;
};

// Returns (*ptr)->priv, or NULL on failure
void *sh_require_obj(struct pl_shader *sh, struct pl_shader_obj **ptr,
                     enum pl_shader_obj_type type, size_t priv_size,
                     void (*uninit)(const struct ra *ra, void *priv));

#define SH_OBJ(sh, ptr, type, t, uninit) \
    ((t*) sh_require_obj(sh, ptr, type, sizeof(t), uninit))

// Initializes a PRNG. The resulting string will directly evaluate to a
// pseudorandom, uniformly distributed float from [0.0,1.0]. Since this
// algorithm works by mutating a state variable, if the user wants to use the
// resulting PRNG inside a subfunction, they must add an extra `inout float %s`
// with the name of `state` to the signature. (Optional)
//
// If `temporal` is set, the PRNG will vary across frames.
ident_t sh_prng(struct pl_shader *sh, bool temporal, ident_t *state);

enum sh_lut_method {
    SH_LUT_AUTO = 0, // pick whatever makes the most sense
    SH_LUT_TEXTURE,  // upload as texture
    SH_LUT_UNIFORM,  // uniform array
    SH_LUT_LITERAL,  // constant / literal array in shader source (fallback)

    // this is never picked by SH_DATA_AUTO
    SH_LUT_LINEAR,   // upload as linearly-sampleable texture
};

// Makes a table of float values available as a shader variable, using an a
// given method (falling back if needed). The resulting identifier can be
// sampled directly as %s(pos), where pos is a vector with the right number of
// dimensions. `pos` must be an integer vector within the bounds of the array,
// unless the method is `SH_LUT_LINEAR` or `SH_LUT_TEXTURE` in which case it's
// a float vector that gets interpolated and clamped as needed. Returns NULL on
// error.
//
// This function also acts as `sh_require_obj`, and uses the `buf`, `tex`
// and `text` fields of the resulting `obj`. (The other fields may be used by
// the caller)
//
// The `fill` function will be called with a zero-initialized buffer whenever
// the data needs to be computed, which happens whenever the size is changed,
// the shader object is invalidated, or `update` is set to true.
ident_t sh_lut(struct pl_shader *sh, struct pl_shader_obj **obj,
               enum sh_lut_method method, int width, int height, int depth,
               bool update, void *priv,
               void (*fill)(void *priv, float *data, int w, int h, int d));
