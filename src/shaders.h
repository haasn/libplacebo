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

#include "common.h"
#include "log.h"
#include "gpu.h"

#include <libplacebo/shaders.h>

// This represents an identifier (e.g. name of function, uniform etc.) for
// a shader resource. The generated identifiers are immutable, but only live
// until pl_shader_reset - so make copies when passing to external stuff.
typedef const char * ident_t;

enum pl_shader_buf {
    SH_BUF_PRELUDE, // extra #defines etc.
    SH_BUF_HEADER,  // previous passes, helper function definitions, etc.
    SH_BUF_BODY,    // partial contents of the "current" function
    SH_BUF_FOOTER,  // will be appended to the end of the current function
    SH_BUF_COUNT,
};

enum pl_shader_type {
    SH_AUTO,
    SH_COMPUTE,
    SH_FRAGMENT
};

struct pl_shader_t {
    pl_log log;
    struct pl_shader_res res; // for accumulating some of the fields
    PL_ARRAY(struct pl_ref *) tmp; // only used for var/va/desc names and data
    PL_ARRAY(pl_shader_obj) obj;
    bool failed;
    bool mutable;
    int output_w;
    int output_h;
    bool transpose;
    pl_str_builder buffers[SH_BUF_COUNT];
    enum pl_shader_type type;
    bool flexible_work_groups;
    enum pl_sampler_type sampler_type;
    char sampler_prefix;
    int fresh;

    // mutable versions of the fields from pl_shader_res
    PL_ARRAY(struct pl_shader_va) vas;
    PL_ARRAY(struct pl_shader_var) vars;
    PL_ARRAY(struct pl_shader_desc) descs;
    PL_ARRAY(struct pl_shader_const) consts;
    PL_ARRAY(const char *) steps;
};

// Same as `pl_shader_finalize` but doesn't template `sh->res.glsl`, instead
// returns the string builder to be used to finalize the shader.
pl_str_builder sh_finalize_internal(pl_shader sh);

// Helper functions for convenience
#define SH_PARAMS(sh) ((sh)->res.params)
#define SH_GPU(sh) (SH_PARAMS(sh).gpu)
#define SH_TMP(sh) ((sh)->tmp.elem[0])

// Returns the GLSL version, defaulting to desktop 130.
struct pl_glsl_version sh_glsl(const pl_shader sh);

#define SH_FAIL(sh, ...) do {    \
        sh->failed = true;       \
        PL_ERR(sh, __VA_ARGS__); \
    } while (0)

// Attempt enabling compute shaders for this pass, if possible
bool sh_try_compute(pl_shader sh, int bw, int bh, bool flex, size_t mem);

// Attempt merging a secondary shader into the current shader. Returns NULL if
// merging fails (e.g. incompatible signatures); otherwise returns an identifier
// corresponding to the generated subpass function.
ident_t sh_subpass(pl_shader sh, const pl_shader sub);

// Helpers for adding new variables/descriptors/etc. with fresh, unique
// identifier names. These will never conflict with other identifiers, even
// if the shaders are merged together.
ident_t sh_fresh(pl_shader sh, const char *name);

// Add a new shader var and return its identifier
ident_t sh_var(pl_shader sh, struct pl_shader_var sv);

// Add a new shader desc and return its identifier.
ident_t sh_desc(pl_shader sh, struct pl_shader_desc sd);

// Add a new shader constant and return its identifier.
ident_t sh_const(pl_shader sh, struct pl_shader_const sc);

// Helper functions for `sh_const`
ident_t sh_const_int(pl_shader sh, const char *name, int val);
ident_t sh_const_uint(pl_shader sh, const char *name, unsigned int val);
ident_t sh_const_float(pl_shader sh, const char *name, float val);
#define SH_INT(val)     sh_const_int(sh, "const", val)
#define SH_UINT(val)    sh_const_uint(sh, "const", val)
#define SH_FLOAT(val)   sh_const_float(sh, "const", val)

// Add a new vec2 vertex attribute from a pl_rect2df, or returns NULL on failure.
ident_t sh_attr_vec2(pl_shader sh, const char *name,
                     const struct pl_rect2df *rc);

// Bind a texture under a given transformation and make its attributes
// available as well. If an output pointer for one of the attributes is left
// as NULL, that attribute will not be added. Returns NULL on failure. `rect`
// is optional, and defaults to the full texture if left as NULL.
//
// Note that for e.g. compute shaders, the vec2 out_pos might be a macro that
// expands to an expensive computation, and should be cached by the user.
ident_t sh_bind(pl_shader sh, pl_tex tex,
                enum pl_tex_address_mode address_mode,
                enum pl_tex_sample_mode sample_mode,
                const char *name, const struct pl_rect2df *rect,
                ident_t *out_pos, ident_t *out_size, ident_t *out_pt);

// Incrementally build up a buffer by adding new variable elements to the
// buffer, resizing buf.buffer_vars if necessary. Returns whether or not the
// variable could be successfully added (which may fail if you try exceeding
// the size limits of the buffer type). If successful, the layout is stored
// in *out_layout (may be NULL).
bool sh_buf_desc_append(void *alloc, pl_gpu gpu,
                        struct pl_shader_desc *buf_desc,
                        struct pl_var_layout *out_layout,
                        const struct pl_var new_var);

size_t sh_buf_desc_size(const struct pl_shader_desc *buf_desc);


// Underlying function for appending text to a shader
#define sh_append(sh, buf, ...) \
    pl_str_builder_addf((sh)->buffers[buf], __VA_ARGS__)

#define sh_append_str(sh, buf, str) \
    pl_str_builder_str((sh)->buffers[buf], str)

#define GLSLP(...) sh_append(sh, SH_BUF_PRELUDE, __VA_ARGS__)
#define GLSLH(...) sh_append(sh, SH_BUF_HEADER, __VA_ARGS__)
#define GLSL(...)  sh_append(sh, SH_BUF_BODY, __VA_ARGS__)
#define GLSLF(...) sh_append(sh, SH_BUF_FOOTER, __VA_ARGS__)

// Attach a description to a shader
void sh_describef(pl_shader sh, const char *fmt, ...)
    PL_PRINTF(2, 3);

static inline void sh_describe(pl_shader sh, const char *desc)
{
    PL_ARRAY_APPEND(sh, sh->steps, desc);
};

// Requires that the share is mutable, has an output signature compatible
// with the given input signature, as well as an output size compatible with
// the given size requirements. Errors and returns false otherwise.
bool sh_require(pl_shader sh, enum pl_shader_sig insig, int w, int h);

// Shader resources

enum pl_shader_obj_type {
    PL_SHADER_OBJ_INVALID = 0,
    PL_SHADER_OBJ_TONE_MAP,
    PL_SHADER_OBJ_SAMPLER,
    PL_SHADER_OBJ_DITHER,
    PL_SHADER_OBJ_LUT,
    PL_SHADER_OBJ_AV1_GRAIN,
    PL_SHADER_OBJ_FILM_GRAIN,
    PL_SHADER_OBJ_RESHAPE,
};

struct pl_shader_obj_t {
    enum pl_shader_obj_type type;
    pl_rc_t rc;
    pl_gpu gpu;
    void (*uninit)(pl_gpu gpu, void *priv);
    void *priv;
};

// Returns (*ptr)->priv, or NULL on failure
void *sh_require_obj(pl_shader sh, pl_shader_obj *ptr,
                     enum pl_shader_obj_type type, size_t priv_size,
                     void (*uninit)(pl_gpu gpu, void *priv));

#define SH_OBJ(sh, ptr, type, t, uninit) \
    ((t*) sh_require_obj(sh, ptr, type, sizeof(t), uninit))

// Initializes a PRNG. The resulting string will directly evaluate to a
// pseudorandom, uniformly distributed vec3 from [0.0,1.0]. Since this
// algorithm works by mutating a state variable, if the user wants to use the
// resulting PRNG inside a subfunction, they must add an extra `inout prng_t %s`
// with the contents of `state` to the signature. (Optional)
//
// If `temporal` is set, the PRNG will vary across frames.
ident_t sh_prng(pl_shader sh, bool temporal, ident_t *state);

// Backing memory type
enum sh_lut_type {
    SH_LUT_AUTO = 0, // pick whatever makes the most sense
    SH_LUT_TEXTURE,  // upload as texture
    SH_LUT_UNIFORM,  // uniform array
    SH_LUT_LITERAL,  // constant / literal array in shader source (fallback)
};

// Interpolation method
enum sh_lut_method {
    SH_LUT_NONE = 0,    // no interpolation, integer indices
    SH_LUT_LINEAR,      // linear interpolation, vecN indices in range [0,1]
    SH_LUT_TETRAHEDRAL, // tetrahedral interpolation for vec3, equivalent to
                        // SH_LUT_LINEAR for lower dimensions
};

struct sh_lut_params {
    pl_shader_obj *object;

    // Type of the LUT we intend to generate.
    //
    // Note: If `var_type` is PL_VAR_*INT, `method` must be SH_LUT_NONE.
    enum pl_var_type var_type;
    enum sh_lut_type lut_type;
    enum sh_lut_method method;

    // For SH_LUT_TEXTURE, this can be used to override the texture's internal
    // format, in which case it takes precedence over the default for `type`.
    pl_fmt fmt;

    // LUT dimensions. Unused dimensions may be left as 0.
    int width;
    int height;
    int depth;
    int comps;

    // If true, the LUT will always be regenerated, even if the dimensions have
    // not changed.
    bool update;

    // Alternate way of triggering shader invalidations. If the signature
    // does not match the LUT's signature, it will be regenerated.
    uint64_t signature;

    // If set to true, shader objects will be preserved and updated in-place
    // rather than being treated as read-only.
    bool dynamic;

    // Will be called with a zero-initialized buffer whenever the data needs to
    // be computed, which happens whenever the size is changed, the shader
    // object is invalidated, or `update` is set to true.
    //
    // Note: Interpretation of `data` is according to `type` and `fmt`.
    void (*fill)(void *data, const struct sh_lut_params *params);
    void *priv;
};

#define sh_lut_params(...) (&(struct sh_lut_params) { __VA_ARGS__ })

// Makes a table of values available as a shader variable, using an a given
// method (falling back if needed). The resulting identifier can be sampled
// directly as %s(pos), where pos is a vector with the right number of
// dimensions. `pos` must be an integer vector within the bounds of the array,
// unless the method is `SH_LUT_LINEAR`, in which case it's a float vector that
// gets interpolated and clamped as needed. Returns NULL on error.
ident_t sh_lut(pl_shader sh, const struct sh_lut_params *params);

// Returns a GLSL-version appropriate "bvec"-like type. For GLSL 130+, this
// returns bvecN. For GLSL 120, this returns vecN instead. The intended use of
// this function is with mix(), which only accepts bvec in GLSL 130+.
const char *sh_bvec(const pl_shader sh, int dims);

// Returns the appropriate `texture`-equivalent function for the shader and
// given texture.
static inline const char *sh_tex_fn(const pl_shader sh,
                                    const struct pl_tex_params params)
{
    static const char *suffixed[] = {
        [1] = "texture1D",
        [2] = "texture2D",
        [3] = "texture3D",
    };

    int dims = pl_tex_params_dimension(params);
    return sh_glsl(sh).version >= 130 ? "texture" : suffixed[dims];
}

static inline const char *sh_float_type(uint8_t num_comps)
{
    switch (num_comps) {
    case 1: return "float";
    case 2: return "vec2";
    case 3: return "vec3";
    case 4: return "vec4";
    }

    pl_unreachable();
}

static inline uint8_t sh_tex_swiz(char swiz[5], uint8_t comp_mask)
{
    uint8_t num_comps = 0;
    for (uint8_t comps = comp_mask; comps;) {
        uint8_t c = __builtin_ctz(comps);
        assert(c < 4 && num_comps < 4);
        swiz[num_comps++] = "xyzw"[c];
        comps &= ~(1u << c);
    }

    swiz[num_comps] = '\0';
    return num_comps;
}
