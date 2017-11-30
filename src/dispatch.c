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

#include "common.h"
#include "context.h"
#include "shaders.h"
#include "ra.h"

enum {
    TMP_PRELUDE,   // GLSL version, global definitions, etc.
    TMP_MAIN,      // main GLSL shader body
    TMP_VERT_HEAD, // vertex shader inputs/outputs
    TMP_VERT_BODY, // vertex shader body
    TMP_COUNT,
};

struct pl_dispatch {
    struct pl_context *ctx;
    const struct ra *ra;
    uint8_t current_ident;
    uint8_t current_index;

    // pool of pl_shaders, in order to avoid frequent re-allocations
    struct pl_shader **shaders;
    int num_shaders;

    // cache of compiled passes
    struct pass **passes;
    int num_passes;

    // temporary buffers to help avoid re_allocations during pass creation
    struct bstr tmp[TMP_COUNT];
};

enum pass_var_type {
    PASS_VAR_GLOBAL, // regular/global uniforms (RA_CAP_INPUT_VARIABLES)
    PASS_VAR_UBO,    // uniform buffers
    PASS_VAR_PUSHC   // push constants
};

// Cached metadata about a variable's effective placement / update method
struct pass_var {
    int index; // for ra_var_update
    enum pass_var_type type;
    struct ra_var_layout layout;
    void *cached_data;
};

struct pass {
    uint64_t signature; // as returned by pl_shader_signature
    const struct ra_pass *pass;
    bool failed;

    // contains cached data and update metadata, same order as pl_shader
    struct pass_var *vars;

    // for uniform buffer updates
    const struct ra_buf *ubo;
    struct ra_desc ubo_desc; // temporary

    // Cached ra_pass_run_params. This will also contain mutable allocations
    // for the push constants, descriptor bindings (including the binding for
    // the UBO pre-filled), vertex array and variable updates
    struct ra_pass_run_params run_params;
};

static void pass_destroy(struct pl_dispatch *dp, struct pass *pass)
{
    if (!pass)
        return;

    ra_buf_destroy(dp->ra, &pass->ubo);
    ra_pass_destroy(dp->ra, &pass->pass);
    talloc_free(pass);
}

struct pl_dispatch *pl_dispatch_create(struct pl_context *ctx, const struct ra *ra)
{
    pl_assert(ctx);
    struct pl_dispatch *dp = talloc_zero(ctx, struct pl_dispatch);
    dp->ctx = ctx;
    dp->ra = ra;

    return dp;
}

void pl_dispatch_destroy(struct pl_dispatch **ptr)
{
    struct pl_dispatch *dp = *ptr;
    if (!dp)
        return;

    for (int i = 0; i < dp->num_passes; i++)
        pass_destroy(dp, dp->passes[i]);
    for (int i = 0; i < dp->num_shaders; i++)
        pl_shader_free(&dp->shaders[i]);

    talloc_free(dp);
    *ptr = NULL;
}

struct pl_shader *pl_dispatch_begin(struct pl_dispatch *dp)
{
    uint8_t ident = dp->current_ident++;

    struct pl_shader *sh;
    if (TARRAY_POP(dp->shaders, dp->num_shaders, &sh)) {
        pl_shader_reset(sh, ident, dp->current_index);
        return sh;
    }

    return pl_shader_alloc(dp->ctx, dp->ra, ident, dp->current_index);
}

void pl_dispatch_reset_frame(struct pl_dispatch *dp)
{
    dp->current_ident = 0;
    dp->current_index++;
}

static bool add_pass_var(struct pl_dispatch *dp, void *tmp, struct pass *pass,
                         struct ra_pass_params *params,
                         const struct pl_shader_var *sv, struct pass_var *pv)
{
    const struct ra *ra = dp->ra;

    // Try not to use push constants for "large" values like matrices, since
    // this is likely to exceed the VGPR/pushc size budgets
    bool try_pushc = (sv->var.dim_m == 1 && sv->var.dim_a == 1) || sv->dynamic;
    if (try_pushc && ra->glsl.vulkan && ra->limits.max_pushc_size) {
        pv->layout = ra_push_constant_layout(ra, params->push_constants_size, &sv->var);
        size_t new_size = pv->layout.offset + pv->layout.size;
        if (new_size <= ra->limits.max_pushc_size) {
            params->push_constants_size = new_size;
            pv->type = PASS_VAR_PUSHC;
            return true;
        }
    }

    // Attempt using uniform buffer next. The GLSL version 440 check is due
    // to explicit offsets on UBO entries. In theory we could leave away
    // the offsets and support UBOs for older GL as well, but this is a nice
    // safety net for driver bugs (and also rules out potentially buggy drivers)
    // Also avoid UBOs for highly dynamic stuff since that requires synchronizing
    // the UBO writes every frame
    bool try_ubo = !(ra->caps & RA_CAP_INPUT_VARIABLES) || !sv->dynamic;
    if (try_ubo && ra->glsl.version >= 440 && ra->limits.max_ubo_size) {
        if (ra_buf_desc_append(tmp, ra, &pass->ubo_desc, &pv->layout, sv->var)) {
            pv->type = PASS_VAR_UBO;
            return true;
        }
    }

    // Otherwise, use global uniforms
    if (ra->caps & RA_CAP_INPUT_VARIABLES) {
        pv->type = PASS_VAR_GLOBAL;
        pv->index = params->num_variables;
        pv->layout = ra_var_host_layout(0, &sv->var);
        TARRAY_APPEND(tmp, params->variables, params->num_variables, sv->var);
        return true;
    }

    // Ran out of variable binding methods. The most likely scenario in which
    // this can happen is if we're using a RA that does not support global
    // input vars and we've exhausted the UBO size limits.
    PL_ERR(dp, "Unable to add input variable '%s': possibly exhausted "
           "UBO size limits?", sv->var.name);
    return false;
}

#define ADD(x, ...) bstr_xappend_asprintf(dp, (x), __VA_ARGS__)
#define ADD_BSTR(x, s) bstr_xappend(dp, (x), (s))

static void add_var(struct pl_dispatch *dp, struct bstr *body,
                    const struct ra_var *var)
{
    ADD(body, "%s %s", ra_var_glsl_type_name(*var), var->name);

    if (var->dim_a > 1) {
        ADD(body, "[%d];\n", var->dim_a);
    } else {
        ADD(body, ";\n");
    }
}

static void add_buffer_vars(struct pl_dispatch *dp, struct bstr *body,
                            const struct ra_buffer_var *vars, int num)
{
    ADD(body, "{\n");
    for (int i = 0; i < num; i++) {
        ADD(body, "    layout(offset=%zu) ", vars[i].layout.offset);
        add_var(dp, body, &vars[i].var);
    }
    ADD(body, "};\n");
}

static ident_t sh_var_from_va(struct pl_shader *sh, const char *name,
                              const struct ra_vertex_attrib *va,
                              const void *data)
{
    return sh_var(sh, (struct pl_shader_var) {
        .var  = ra_var_from_fmt(va->fmt, name),
        .data = data,
    });
}

static void generate_shaders(struct pl_dispatch *dp, struct pass *pass,
                             struct ra_pass_params *params,
                             struct pl_shader *sh, ident_t vert_pos)
{
    const struct ra *ra = dp->ra;
    const struct pl_shader_res *res = pl_shader_finalize(sh);

    struct bstr *pre = &dp->tmp[TMP_PRELUDE];
    ADD(pre, "#version %d%s\n", ra->glsl.version, ra->glsl.gles ? " es" : "");
    if (params->type == RA_PASS_COMPUTE)
        ADD(pre, "#extension GL_ARB_compute_shader : enable\n");

    if (ra->glsl.gles) {
        ADD(pre, "precision mediump float;\n");
        ADD(pre, "precision mediump sampler2D;\n");
        if (ra->limits.max_tex_1d_dim)
            ADD(pre, "precision mediump sampler1D;\n");
        if (ra->limits.max_tex_3d_dim)
            ADD(pre, "precision mediump sampler3D;\n");
    }

    char *vert_in  = ra->glsl.version >= 130 ? "in" : "attribute";
    char *vert_out = ra->glsl.version >= 130 ? "out" : "varying";
    char *frag_in  = ra->glsl.version >= 130 ? "in" : "varying";

    struct bstr *glsl = &dp->tmp[TMP_MAIN];
    ADD_BSTR(glsl, *pre);

    const char *out_color = "gl_FragColor";
    switch(params->type) {
    case RA_PASS_RASTER: {
        pl_assert(vert_pos);
        struct bstr *vert_head = &dp->tmp[TMP_VERT_HEAD];
        struct bstr *vert_body = &dp->tmp[TMP_VERT_BODY];

        // Set up a trivial vertex shader
        ADD_BSTR(vert_head, *pre);
        ADD(vert_body, "void main() {\n");
        for (int i = 0; i < res->num_vertex_attribs; i++) {
            const struct ra_vertex_attrib *va = &params->vertex_attribs[i];
            const struct pl_shader_va *sva = &res->vertex_attribs[i];
            const char *type = va->fmt->glsl_type;

            // Use the pl_shader_va for the name in the fragment shader since
            // the ra_vertex_attrib is already mangled for the vertex shader
            const char *name = sva->attr.name;

            char loc[32];
            snprintf_c(loc, sizeof(loc), "layout(location=%d) ", va->location);
            ADD(vert_head, "%s%s %s %s;\n", loc, vert_in, type, va->name);

            if (strcmp(name, vert_pos) == 0) {
                pl_assert(va->fmt->num_components == 2);
                ADD(vert_body, "gl_Position = vec4(%s, 0.0, 1.0);\n", va->name);
            } else {
                // Everything else is just blindly passed through
                ADD(vert_head, "%s%s %s %s;\n", loc, vert_out, type, name);
                ADD(vert_body, "%s = %s;\n", name, va->name);
                ADD(glsl, "%s%s %s %s;\n", loc, frag_in, type, name);
            }
        }

        ADD(vert_body, "}");
        ADD_BSTR(vert_head, *vert_body);
        params->vertex_shader = vert_head->start;

        // GLSL 130+ doesn't use the magic gl_FragColor
        if (ra->glsl.version >= 130) {
            out_color = "out_color";
            ADD(glsl, "layout(location=0) out vec4 %s;\n", out_color);
        }
        break;
    }
    case RA_PASS_COMPUTE:
        ADD(glsl, "layout (local_size_x = %d, local_size_y = %d) in;\n",
            res->compute_group_size[0], res->compute_group_size[1]);
        break;
    default: abort();
    }

    // Add all of the push constants as their own element
    if (params->push_constants_size) {
        ADD(glsl, "layout(std430, push_constant) uniform PushC {\n");
        for (int i = 0; i < res->num_variables; i++) {
            struct ra_var *var = &res->variables[i].var;
            struct pass_var *pv = &pass->vars[i];
            if (pv->type != PASS_VAR_PUSHC)
                continue;
            ADD(glsl, "/*offset=%zu*/ ", pv->layout.offset);
            add_var(dp, glsl, var);
        }
        ADD(glsl, "};\n");
    }

    // Add all of the required descriptors
    for (int i = 0; i < res->num_descriptors; i++) {
        const struct pl_shader_desc *sd = &res->descriptors[i];
        const struct ra_desc *desc = &params->descriptors[i];

        switch (desc->type) {
        case RA_DESC_SAMPLED_TEX: {
            static const char *types[] = {
                [1] = "sampler1D",
                [2] = "sampler2D",
                [3] = "sampler3D",
            };

            // Vulkan requires explicit bindings; ra_gl always sets the
            // bindings manually to avoid relying on the user doing so
            if (ra->glsl.vulkan)
                ADD(glsl, "layout(binding=%d) ", desc->binding);

            const struct ra_tex *tex = sd->object;
            int dims = ra_tex_params_dimension(tex->params);
            ADD(glsl, "uniform %s %s;\n", types[dims], desc->name);
            break;
        }

        case RA_DESC_STORAGE_IMG: {
            static const char *types[] = {
                [1] = "image1D",
                [2] = "image2D",
                [3] = "image3D",
            };

            // For better compatibility, we have to explicitly label the
            // type of data we will be reading/writing to this image.
            const struct ra_tex *tex = sd->object;
            const char *format = tex->params.format->glsl_format;
            const char *access = ra_desc_access_glsl_name(desc->access);
            int dims = ra_tex_params_dimension(tex->params);
            pl_assert(format);

            if (ra->glsl.vulkan) {
                ADD(glsl, "layout(binding=%d, %s) ", desc->binding, format);
            } else {
                ADD(glsl, "layout(%s) ", format);
            }
            ADD(glsl, "%s uniform %s %s;\n", access, types[dims], desc->name);
            break;
        }

        case RA_DESC_BUF_UNIFORM:
            ADD(glsl, "layout(std140, binding=%d) uniform %s ", desc->binding,
                desc->name);
            add_buffer_vars(dp, glsl, desc->buffer_vars, desc->num_buffer_vars);
            break;
        case RA_DESC_BUF_STORAGE:
            ADD(glsl, "layout(std430, binding=%d) %s buffer %s ", desc->binding,
                ra_desc_access_glsl_name(desc->access), desc->name);
            add_buffer_vars(dp, glsl, desc->buffer_vars, desc->num_buffer_vars);
            break;
        default: abort();
        }
    }

    // Add all of the remaining variables
    for (int i = 0; i < res->num_variables; i++) {
        const struct ra_var *var = &res->variables[i].var;
        const struct pass_var *pv = &pass->vars[i];
        if (pv->type != PASS_VAR_GLOBAL)
            continue;
        ADD(glsl, "uniform ");
        add_var(dp, glsl, var);
    }

    // Set up the main shader body
    ADD(glsl, "%s", res->glsl);
    ADD(glsl, "void main() {\n");

    pl_assert(res->input == PL_SHADER_SIG_NONE);
    switch (params->type) {
    case RA_PASS_RASTER:
        pl_assert(res->output == PL_SHADER_SIG_COLOR);
        ADD(glsl, "%s = %s();\n", out_color, res->name);
        break;
    case RA_PASS_COMPUTE:
        pl_assert(res->output == PL_SHADER_SIG_NONE);
        ADD(glsl, "%s();\n", res->name);
        break;
    default: abort();
    }

    ADD(glsl, "}");
    params->glsl_shader = glsl->start;
}

#undef ADD
#undef ADD_BSTR

static struct pass *find_pass(struct pl_dispatch *dp, struct pl_shader *sh,
                              const struct ra_tex *target, ident_t vert_pos)
{
    uint64_t sig = pl_shader_signature(sh);

    for (int i = 0; i < dp->num_passes; i++) {
        if (dp->passes[i]->signature == sig)
            return dp->passes[i];
    }

    void *tmp = talloc_new(NULL); // for resources attached to `params`

    struct pass *pass = talloc_zero(dp, struct pass);
    pass->signature = sig;
    pass->failed = true; // will be set to false on success
    pass->ubo_desc = (struct ra_desc) {
        .name = "UBO",
        .type = RA_DESC_BUF_UNIFORM,
    };

    struct pl_shader_res *res = &sh->res;

    struct ra_pass_run_params *rparams = &pass->run_params;
    struct ra_pass_params params = {
        .type = pl_shader_is_compute(sh) ? RA_PASS_COMPUTE : RA_PASS_RASTER,
        .num_descriptors = res->num_descriptors,
    };

    switch (params.type) {
    case RA_PASS_RASTER: {
        params.target_dummy = *target;

        // Fill in the vertex attributes array
        params.num_vertex_attribs = res->num_vertex_attribs;
        params.vertex_attribs = talloc_zero_array(tmp, struct ra_vertex_attrib,
                                                  res->num_vertex_attribs);

        int va_loc = 0;
        for (int i = 0; i < res->num_vertex_attribs; i++) {
            struct ra_vertex_attrib *va = &params.vertex_attribs[i];
            *va = res->vertex_attribs[i].attr;

            // Mangle the name to make sure it doesn't conflict with the
            // fragment shader input
            va->name = talloc_asprintf(tmp, "vert%s", va->name);

            // Place the vertex attribute
            va->offset = params.vertex_stride;
            va->location = va_loc;
            params.vertex_stride += va->fmt->texel_size;

            // The number of vertex attribute locations consumed by a vertex
            // attribute is the number of vec4s it consumes, rounded up
            const size_t va_loc_size = sizeof(float[4]);
            va_loc += (va->fmt->texel_size + va_loc_size - 1) / va_loc_size;
        }

        // Generate the vertex array placeholder
        params.vertex_type = RA_PRIM_TRIANGLE_STRIP;
        rparams->vertex_count = 4; // single quad
        size_t vert_size = rparams->vertex_count * params.vertex_stride;
        rparams->vertex_data = talloc_zero_size(pass, vert_size);
        break;

    }
    case RA_PASS_COMPUTE: {
        // Round up to make sure we don-t leave off a part of the target
        int block_w = sh->res.compute_group_size[0],
            block_h = sh->res.compute_group_size[1],
            num_x   = (target->params.w + block_w - 1) / block_w,
            num_y   = (target->params.h + block_h - 1) / block_h;

        rparams->compute_groups[0] = num_x;
        rparams->compute_groups[1] = num_y;
        rparams->compute_groups[2] = 1;
        break;
    }
    default: abort();
    }

    // Place all the variables; these will dynamically end up in different
    // locations based on what the underlying RA supports (UBOs, pushc, etc.)
    pass->vars = talloc_zero_array(pass, struct pass_var, res->num_variables);
    for (int i = 0; i < res->num_variables; i++) {
        if (!add_pass_var(dp, tmp, pass, &params, &res->variables[i], &pass->vars[i]))
            goto error;
    }

    // Create and attach the UBO if necessary
    int ubo_index = -1;
    size_t ubo_size = ra_buf_desc_size(&pass->ubo_desc);
    if (ubo_size) {
        pass->ubo = ra_buf_create(dp->ra, &(struct ra_buf_params) {
            .type = RA_BUF_UNIFORM,
            .size = ubo_size,
            .host_writable = true,
        });

        if (!pass->ubo) {
            PL_ERR(dp, "Failed creating uniform buffer for dispatch");
            goto error;
        }

        ubo_index = res->num_descriptors;
        sh_desc(sh, (struct pl_shader_desc) {
            .desc = pass->ubo_desc,
            .object = pass->ubo,
        });
    }

    // Place and fill in the descriptors
    int num = res->num_descriptors;
    int binding[RA_DESC_TYPE_COUNT] = {0};
    params.num_descriptors = num;
    params.descriptors = talloc_zero_array(tmp, struct ra_desc, num);
    rparams->desc_bindings = talloc_zero_array(pass, struct ra_desc_binding, num);
    for (int i = 0; i < num; i++) {
        struct ra_desc *desc = &params.descriptors[i];
        *desc = res->descriptors[i].desc;
        desc->binding = binding[ra_desc_namespace(dp->ra, desc->type)]++;
    }

    // Pre-fill the desc_binding for the UBO
    if (pass->ubo) {
        pl_assert(ubo_index >= 0);
        rparams->desc_bindings[ubo_index].object = pass->ubo;
    }

    // Create the push constants region
    params.push_constants_size = PL_ALIGN2(params.push_constants_size, 4);
    rparams->push_constants = talloc_zero_size(pass, params.push_constants_size);

    // Finally, finalize the shaders and create the pass itself
    generate_shaders(dp, pass, &params, sh, vert_pos);
    pass->pass = rparams->pass = ra_pass_create(dp->ra, &params);
    if (!pass->pass) {
        PL_ERR(dp, "Failed creating render pass for dispatch");
        goto error;
    }

    pass->failed = false;

error:
    pass->ubo_desc = (struct ra_desc) {0}; // contains temporary pointers
    talloc_free(tmp);
    TARRAY_APPEND(dp, dp->passes, dp->num_passes, pass);
    return pass;
}

static void update_pass_var(struct pl_dispatch *dp, struct pass *pass,
                            const struct pl_shader_var *sv, struct pass_var *pv)
{
    struct ra_var_layout host_layout = ra_var_host_layout(0, &sv->var);
    pl_assert(host_layout.size);

    // Use the cache to skip updates if possible
    if (pv->cached_data && !memcmp(sv->data, pv->cached_data, host_layout.size))
        return;
    if (!pv->cached_data)
        pv->cached_data = talloc_size(pass, host_layout.size);
    memcpy(pv->cached_data, sv->data, host_layout.size);

    struct ra_pass_run_params *rparams = &pass->run_params;
    switch (pv->type) {
    case PASS_VAR_GLOBAL: {
        struct ra_var_update vu = {
            .index = pv->index,
            .data  = sv->data,
        };
        TARRAY_APPEND(pass, rparams->var_updates, rparams->num_var_updates, vu);
        break;
    }
    case PASS_VAR_UBO: {
        pl_assert(pass->ubo);
        uintptr_t src = (uintptr_t) sv->data;
        uintptr_t end = src + (ptrdiff_t) host_layout.size;
        size_t dst = pv->layout.offset;
        while (src < end) {
            ra_buf_write(dp->ra, pass->ubo, dst, (void *) src, host_layout.stride);
            src += host_layout.stride;
            dst += pv->layout.stride;
        }
        break;
    }
    case PASS_VAR_PUSHC:
        pl_assert(rparams->push_constants);
        memcpy_layout(rparams->push_constants, pv->layout, sv->data, host_layout);
        break;
    };
}

static void translate_compute_shader(struct pl_dispatch *dp,
                                     struct pl_shader *sh,
                                     const struct ra_tex *target)
{
    // Simulate a framebuffer using storage images
    pl_assert(target->params.storable);
    ident_t fbo = sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name    = "out_image",
            .type    = RA_DESC_STORAGE_IMG,
            .access  = RA_DESC_ACCESS_WRITEONLY,
        },
        .object = target,
    });

    pl_assert(sh->res.output == PL_SHADER_SIG_COLOR);
    GLSL("imageStore(%s, ivec2(gl_GlobalInvocationID), color);\n", fbo);
    sh->res.output = PL_SHADER_SIG_NONE;

    // Simulate vertex attributes using global definitions
    ident_t out_scale = sh_var(sh, (struct pl_shader_var) {
        .var     = ra_var_vec2("out_scale"),
        .data    = &(float[2]){ 1.0 / target->params.w, 1.0 / target->params.h },
        .dynamic = true,
    });

    GLSLP("#define frag_pos(id) (vec2(id) + vec2(0.5)) \n"
          "#define frag_map(id) (%s * frag_pos(id))    \n"
          "#define gl_FragCoord vec4(frag_pos(gl_GlobalInvocationID), 0.0, 1.0) \n",
          out_scale);

    for (int n = 0; n < sh->res.num_vertex_attribs; n++) {
        const struct pl_shader_va *sva = &sh->res.vertex_attribs[n];

        ident_t points[4];
        for (int i = 0; i < PL_ARRAY_SIZE(points); i++) {
            char name[4];
            snprintf_c(name, sizeof(name), "p%d", i);
            points[i] = sh_var_from_va(sh, name, &sva->attr, sva->data[i]);
        }

        GLSLP("#define %s_map(id) "
             "(mix(mix(%s, %s, frag_map(id).x), "
             "     mix(%s, %s, frag_map(id).x), "
             "frag_map(id).y))\n"
             "#define %s (%s_map(gl_GlobalInvocationID))\n",
             sva->attr.name,
             points[0], points[1], points[2], points[3],
             sva->attr.name, sva->attr.name);
    }
}

bool pl_dispatch_finish(struct pl_dispatch *dp, struct pl_shader *sh,
                        const struct ra_tex *target, const struct pl_rect2d *rc)
{
    const struct pl_shader_res *res = &sh->res;
    bool ret = false;

    if (!sh->mutable) {
        PL_ERR(dp, "Trying to dispatch non-mutable shader?");
        goto error;
    }

    if (res->input != PL_SHADER_SIG_NONE || res->output != PL_SHADER_SIG_COLOR) {
        PL_ERR(dp, "Trying to dispatch shader with incompatible signature!");
        goto error;
    }

    const struct ra_tex_params *tpars = &target->params;
    if (ra_tex_params_dimension(*tpars) != 2 || !tpars->renderable) {
        PL_ERR(dp, "Trying to dispatch using a shader using an invalid target "
               "texture. The target must be a renderable 2D texture.");
        goto error;
    }

    struct pl_rect2d full = {0, 0, tpars->w, tpars->h};
    rc = PL_DEF(rc, &full);

    int w, h, tw = abs(pl_rect_w(*rc)), th = abs(pl_rect_h(*rc));
    if (pl_shader_output_size(sh, &w, &h) && (w != tw || h != th))
    {
        PL_ERR(dp, "Trying to dispatch a shader with explicit output size "
               "requirements %dx%d using a target rect of size %dx%d.",
               w, h, tw, th);
        goto error;
    }

    ident_t vert_pos = NULL;

    if (pl_shader_is_compute(sh)) {
        // Translate the compute shader to simulate vertices etc.
        // FIXME: take into account *rc
        translate_compute_shader(dp, sh, target);
    } else {
        // Add the vertex information encoding the position
        vert_pos = sh_attr_vec2(sh, "position", &(const struct pl_rect2df) {
            .x0 = 2.0 * rc->x0 / tpars->w - 1.0,
            .y0 = 2.0 * rc->y0 / tpars->h - 1.0,
            .x1 = 2.0 * rc->x1 / tpars->w - 1.0,
            .y1 = 2.0 * rc->y1 / tpars->h - 1.0,
        });
        // TODO: also update the stencil for performance
    }

    struct pass *pass = find_pass(dp, sh, target, vert_pos);

    // Silently return on failed passes
    if (pass->failed)
        goto error;

    struct ra_pass_run_params *rparams = &pass->run_params;

    // Update the descriptor bindings
    for (int i = 0; i < sh->res.num_descriptors; i++)
        rparams->desc_bindings[i].object = sh->res.descriptors[i].object;

    // Update all of the variables (if needed)
    rparams->num_var_updates = 0;
    for (int i = 0; i < res->num_variables; i++)
        update_pass_var(dp, pass, &res->variables[i], &pass->vars[i]);

    // Update the vertex data
    if (rparams->vertex_data) {
        uintptr_t vert_base = (uintptr_t) rparams->vertex_data;
        size_t stride = rparams->pass->params.vertex_stride;
        for (int i = 0; i < res->num_vertex_attribs; i++) {
            struct pl_shader_va *sva = &res->vertex_attribs[i];
            struct ra_vertex_attrib *va = &rparams->pass->params.vertex_attribs[i];

            size_t size = sva->attr.fmt->texel_size;
            uintptr_t va_base = vert_base + va->offset; // use placed offset
            for (int n = 0; n < 4; n++)
                memcpy((void *) (va_base + n * stride), sva->data[n], size);
        }
    }

    // Dispatch the actual shader
    rparams->target = target;
    ra_pass_run(dp->ra, &pass->run_params);
    ret = true;

error:
    // Reset the temporary buffers which we use to build the shader
    for (int i = 0; i < PL_ARRAY_SIZE(dp->tmp); i++)
        dp->tmp[i].len = 0;

    pl_dispatch_abort(dp, sh);
    return ret;
}

void pl_dispatch_abort(struct pl_dispatch *dp, struct pl_shader *sh)
{
    // Re-add the shader to the internal pool of shaders
    TARRAY_APPEND(dp, dp->shaders, dp->num_shaders, sh);
}
