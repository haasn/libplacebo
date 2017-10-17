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

struct pl_dispatch {
    struct pl_context *ctx;
    const struct ra *ra;

    // pool of pl_shaders, in order to avoid frequent re-allocations
    struct pl_shader **shaders;
    int num_shaders;

    // cache of compiled passes
    struct pass **passes;
    int num_passes;

    // temporary buffers to help avoid re_allocations during pass creation
    struct bstr tmp_glsl[4];
};

enum pass_var_type {
    PASS_VAR_GLOBAL, // regular/global uniforms (RA_CAP_INPUT_VARIABLES)
    PASS_VAR_UBO,    // uniform buffers
    PASS_VAR_PUSHC   // push constants
};

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

    // for the attached storage image
    int img_index;    // index into run_params.var_updates
    ident_t img_name; // name as used in the shader (for `main`)

    // for uniform buffer updates
    int ubo_index;    // for desc_bindings
    size_t ubo_size;
    const struct ra_buf *ubo;
    struct ra_buffer_var *ubo_vars; // temporary
    int num_ubo_vars;

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
    assert(ctx);
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
    struct pl_shader *sh;
    if (TARRAY_POP(dp->shaders, dp->num_shaders, &sh))
        return sh;

    return pl_shader_alloc(dp->ctx, dp->ra);
}

static bool add_pass_var(struct pl_dispatch *dp, void *tmp, struct pass *pass,
                         struct ra_pass_params *params,
                         const struct pl_shader_var *sv, struct pass_var *pv)
{
    const struct ra *ra = dp->ra;

    // Try not to use push constants for "large" values like matrices, since
    // this is likely to exceed the VGPR/pushc size budgets
    bool try_pushc = sv->var.dim_m == 1 || sv->dynamic;
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
        pv->layout = ra_buf_uniform_layout(ra, pass->ubo_size, &sv->var);
        size_t new_size = pv->layout.offset + pv->layout.size;
        if (new_size <= ra->limits.max_ubo_size) {
            pv->type = PASS_VAR_UBO;
            pass->ubo_size = new_size;
            struct ra_buffer_var bv = {
                .var = sv->var,
                .layout = pv->layout,
            };
            TARRAY_APPEND(tmp, pass->ubo_vars, pass->num_ubo_vars, bv);
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

static void add_buffer_vars(struct pl_dispatch *dp, struct bstr *body,
                            struct ra_buffer_var *vars, int num)
{
    ADD(body, "{\n");
    for (int i = 0; i < num; i++) {
        ADD(body, "    layout(offset=%zu) %s %s;\n", vars[i].layout.offset,
            ra_var_glsl_type_name(vars[i].var), vars[i].var.name);
    }
    ADD(body, "};\n");
}

static void generate_shaders(struct pl_dispatch *dp, struct pass *pass,
                             struct ra_pass_params *params,
                             struct pl_shader *sh, ident_t vert_pos)
{
    const struct ra *ra = dp->ra;
    const struct pl_shader_res *res = pl_shader_finalize(sh);

    // Reset the temporary buffers which we use to build the shader
    for (int i = 0; i < PL_ARRAY_SIZE(dp->tmp_glsl); i++)
        dp->tmp_glsl[i].len = 0;

    struct bstr *header = &dp->tmp_glsl[0];
    ADD(header, "#version %d%s\n", ra->glsl.version, ra->glsl.gles ? " es" : "");
    if (params->type == RA_PASS_COMPUTE)
        ADD(header, "#extension GL_ARB_compute_shader : enable\n");

    if (ra->glsl.gles) {
        ADD(header, "precision mediump float;\n");
        ADD(header, "precision mediump sampler2D;\n");
        if (ra->limits.max_tex_1d_dim)
            ADD(header, "precision mediump sampler1D;\n");
        if (ra->limits.max_tex_3d_dim)
            ADD(header, "precision mediump sampler3D;\n");
    }

    if (ra->glsl.vulkan && params->type == RA_PASS_COMPUTE) {
        // For some reason this isn't defined in vulkan-flavored GLSL
        ADD(header, "#define gl_GlobalInvocationIndex "
                    "(gl_WorkGroupID * gl_WorkGroupSize + gl_LocalInvocationID)\n");
    }

    char *vert_in  = ra->glsl.version >= 130 ? "in" : "attribute";
    char *vert_out = ra->glsl.version >= 130 ? "out" : "varying";
    char *frag_in  = ra->glsl.version >= 130 ? "in" : "varying";

    struct bstr *body = &dp->tmp_glsl[1];
    ADD_BSTR(body, *header);

    const char *out_color = "gl_FragColor";
    if (params->type == RA_PASS_RASTER) {
        struct bstr *vert_head = &dp->tmp_glsl[2];
        struct bstr *vert_body = &dp->tmp_glsl[3];
        // Set up a trivial vertex shader
        ADD_BSTR(vert_head, *header);
        ADD(vert_body, "void main() {\n");
        for (int i = 0; i < res->num_vertex_attribs; i++) {
            const struct ra_vertex_attrib *va = &res->vertex_attribs[i].attr;
            const char *type = va->fmt->glsl_type;

            char loc[32];
            snprintf(loc, sizeof(loc), "layout(location=%d) ", va->location);
            ADD(vert_head, "%s%s %s vert%s;\n", loc, vert_in, type, va->name);

            if (strcmp(va->name, vert_pos) == 0) {
                assert(va->fmt->num_components == 2);
                ADD(vert_body, "gl_Position = vec4(vert%s, 0.0, 1.0);\n", va->name);
            } else {
                // Everything else is just blindly passed through
                ADD(vert_head, "%s%s %s %s;\n", loc, vert_out, type, va->name);
                ADD(vert_body, "%s = vert%s;\n", va->name, va->name);
                ADD(body, "%s%s %s %s;\n", loc, frag_in, type, va->name);
            }
        }
        ADD(vert_body, "}");
        ADD_BSTR(vert_head, *vert_body);
        params->vertex_shader = vert_head->start;

        // GLSL 130+ doesn't use the magic gl_FragColor
        if (ra->glsl.version >= 130) {
            out_color = "out_color";
            ADD(body, "layout(location=0) out vec4 %s;\n", out_color);
        }
    }

    if (params->type == RA_PASS_COMPUTE) {
        // Simulate fake vertices
        abort(); // TODO
    }

    // Add all of the push constants as their own element
    if (params->push_constants_size) {
        ADD(body, "layout(std430, push_constant) uniform PushC {\n");
        for (int i = 0; i < res->num_variables; i++) {
            struct ra_var *var = &res->variables[i].var;
            struct pass_var *pv = &pass->vars[i];
            if (pv->type != PASS_VAR_PUSHC)
                continue;
            ADD(body, "/*offset=%zu*/ %s %s;\n", pv->layout.offset,
                ra_var_glsl_type_name(*var), var->name);
        }
        ADD(body, "};\n");
    }

    // Add all of the required descriptors
    for (int i = 0; i < res->num_descriptors; i++) {
        const struct pl_shader_desc *sd = &res->descriptors[i];
        const struct ra_desc *desc = &sd->desc;

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
                ADD(body, "layout(binding=%d) ", desc->binding);

            const struct ra_tex *tex = sd->object;
            int dims = ra_tex_params_dimension(tex->params);
            ADD(body, "uniform %s %s;\n", types[dims], desc->name);
            break;
        }

        case RA_DESC_STORAGE_IMG: {
            // For better compatibility, we have to explicitly label the
            // type of data we will be reading/writing to this image. For
            // simplicity, just pick 32-bit float with however many components.
            static const char *fmts[] = {
                [1] = "r32f",
                [2] = "rg32f",
                [3] = "rgba32f", // rgb32f doesn't exist
                [4] = "rgba32f",
            };

            static const char *types[] = {
                [1] = "image1D",
                [2] = "image2D",
                [3] = "image3D",
            };

            const char *access = ra_desc_access_glsl_name(desc->access);
            const struct ra_tex *tex = sd->object;
            int dims = ra_tex_params_dimension(tex->params);
            int comps = tex->params.format->num_components;

            if (ra->glsl.vulkan) {
                ADD(body, "layout(binding=%d, %s) ", desc->binding, fmts[comps]);
            } else {
                ADD(body, "layout(%s) ", fmts[comps]);
            }

            ADD(body, "%s uniform %s %s;\n", access, types[dims], desc->name);
            break;
        }

        case RA_DESC_BUF_UNIFORM:
            ADD(body, "layout(std140, binding=%d) uniform %s ", desc->binding,
                desc->name);
            add_buffer_vars(dp, body, desc->buffer_vars, desc->num_buffer_vars);
            break;
        case RA_DESC_BUF_STORAGE:
            ADD(body, "layout(std430, binding=%d) %s buffer %s ", desc->binding,
                ra_desc_access_glsl_name(desc->access), desc->name);
            add_buffer_vars(dp, body, desc->buffer_vars, desc->num_buffer_vars);
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
        ADD(body, "uniform %s %s;\n", ra_var_glsl_type_name(*var), var->name);
    }

    // Set up the main shader body
    ADD(body, "%s", res->glsl);
    ADD(body, "void main() {\n");
    ADD(body, "vec4 res = %s();\n", res->name);

    switch (params->type) {
    case RA_PASS_RASTER:
        ADD(body, "%s = res;\n", out_color);
        break;
    case RA_PASS_COMPUTE:
        assert(pass->img_name);
        ADD(body, "imageStore(%s, ivec2(gl_GlobalInvocationID), res);\n",
            pass->img_name);
        break;
    default: abort();
    }

    ADD(body, "}");
    params->glsl_shader = body->start;
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

        for (int i = 0; i < res->num_vertex_attribs; i++) {
            struct ra_vertex_attrib *va = &params.vertex_attribs[i];
            *va = res->vertex_attribs[i].attr;
            // Mangle the name to make sure it doesn't conflict with the
            // fragment shader input
            va->name = talloc_asprintf(tmp, "vert%s", va->name);
        }

        // Generate the vertex array placeholder
        params.vertex_stride = sh->current_va_offset;
        params.vertex_type = RA_PRIM_TRIANGLE_STRIP;
        rparams->vertex_count = 4; // single quad
        size_t vert_size = rparams->vertex_count * params.vertex_stride;
        rparams->vertex_data = talloc_zero_size(pass, vert_size);
        break;
    }
    case RA_PASS_COMPUTE: {
        // Round up to make sure we don-t leave off a part of the target
        int block_w = sh->res.compute_work_groups[0],
            block_h = sh->res.compute_work_groups[1],
            num_x   = (target->params.w + block_w - 1) / block_w,
            num_y   = (target->params.h + block_h - 1) / block_h;

        rparams->compute_groups[0] = num_x;
        rparams->compute_groups[1] = num_y;
        rparams->compute_groups[2] = 1;
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

    // Attach the storage image if necessary
    if (params.type == RA_PASS_COMPUTE) {
        pass->img_index = res->num_descriptors;
        pass->img_name = sh_desc(sh, (struct pl_shader_desc) {
            .desc = {
                .name = "out_image",
                .type = RA_DESC_STORAGE_IMG,
                .access = RA_DESC_ACCESS_WRITEONLY,
            },
            .object = target,
        });
    }

    // Create and attach the UBO if necessary
    if (pass->ubo_size) {
        pass->ubo = ra_buf_create(dp->ra, &(struct ra_buf_params) {
            .type = RA_BUF_UNIFORM,
            .size = pass->ubo_size,
            .host_writable = true,
        });

        if (!pass->ubo) {
            PL_ERR(dp, "Failed creating uniform buffer for dispatch");
            goto error;
        }

        pass->ubo_index = res->num_descriptors;
        sh_desc(sh, (struct pl_shader_desc) {
            .desc = {
                .name = "UBO",
                .type = RA_DESC_BUF_UNIFORM,
                .buffer_vars = pass->ubo_vars,
                .num_buffer_vars = pass->num_ubo_vars,
            },
            .object = pass->ubo,
        });
    }

    // Fill in the descriptors
    int num = res->num_descriptors;
    params.num_descriptors = num;
    params.descriptors = talloc_zero_array(tmp, struct ra_desc, num);
    rparams->desc_bindings = talloc_zero_array(pass, struct ra_desc_binding, num);
    for (int i = 0; i < num; i++)
        params.descriptors[i] = res->descriptors[i].desc;

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
    pass->img_name = NULL; // allocated via sh_fresh
    pass->ubo_vars = NULL;
    talloc_free(tmp);
    TARRAY_APPEND(dp, dp->passes, dp->num_passes, pass);
    return pass;
}

static void update_pass_var(struct pl_dispatch *dp, struct pass *pass,
                            const struct pl_shader_var *sv, struct pass_var *pv)
{
    struct ra_var_layout host_layout = ra_var_host_layout(0, &sv->var);
    assert(host_layout.size);

    // Use the cache to skip updates if possible
    if (pv->cached_data && !memcmp(sv->data, pv->cached_data, host_layout.size))
        return;
    if (!pv->cached_data)
        pv->cached_data = talloc_size(pass, host_layout.size);
    memcpy(pv->cached_data, sv->data, host_layout.size);

    struct ra_pass_run_params *rparams = &pass->run_params;
    uintptr_t src = (uintptr_t) sv->data;

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
        assert(pass->ubo);
        size_t dst = pv->layout.offset;
        for (int i = 0; i < sv->var.dim_m; i++) {
            ra_buf_write(dp->ra, pass->ubo, dst, (void *) src, host_layout.stride);
            src += host_layout.stride;
            dst += pv->layout.stride;
        }
        break;
    }
    case PASS_VAR_PUSHC: {
        assert(rparams->push_constants);
        uintptr_t dst = (uintptr_t) rparams->push_constants +
                        (ptrdiff_t) pv->layout.offset;
        for (int i = 0; i < sv->var.dim_m; i++) {
            memcpy((void *) dst, (void *) src, host_layout.stride);
            src += host_layout.stride;
            dst += pv->layout.stride;
        }
        break;
    }
    };
}

bool pl_dispatch_finish(struct pl_dispatch *dp, struct pl_shader *sh,
                        const struct ra_tex *target)
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

    int w, h;
    if (pl_shader_output_size(sh, &w, &h) && (w != tpars->w || h != tpars->h)) {
        PL_ERR(dp, "Trying to dispatch a shader with explicit output size "
               "requirements %dx%d using a target of size %dx%d.",
               w, h, tpars->w, tpars->h);
        goto error;
    }

    // Add the vertex information encoding the position
    ident_t vert_pos = sh_attr_vec2(sh, "position", &(const struct pl_rect2df) {
        .x0 = -1.0,
        .y0 = -1.0,
        .x1 =  1.0,
        .y1 =  1.0,
    });

    struct pass *pass = find_pass(dp, sh, target, vert_pos);

    // Silently return on failed passes
    if (pass->failed)
        goto error;

    struct ra_pass_run_params *rparams = &pass->run_params;

    // Update the descriptor bindings
    for (int i = 0; i < sh->res.num_descriptors; i++)
        rparams->desc_bindings[i].object = sh->res.descriptors[i].object;
    if (pass->ubo)
        rparams->desc_bindings[pass->ubo_index].object = pass->ubo;
    if (pl_shader_is_compute(sh))
        rparams->desc_bindings[pass->img_index].object = target;

    // Update all of the variables (if needed)
    rparams->num_var_updates = 0;
    for (int i = 0; i < res->num_variables; i++)
        update_pass_var(dp, pass, &res->variables[i], &pass->vars[i]);

    // Update the vertex data
    uintptr_t vert_base = (uintptr_t) rparams->vertex_data;
    size_t stride = rparams->pass->params.vertex_stride;
    for (int i = 0; i < res->num_vertex_attribs; i++) {
        struct pl_shader_va sva = res->vertex_attribs[i];
        size_t size = sva.attr.fmt->texel_size;
        uintptr_t va_base = vert_base + sva.attr.offset;
        for (int n = 0; n < 4; n++)
            memcpy((void *) (va_base + n * stride), sva.data[n], size);
    }

    // Dispatch the actual shader
    rparams->target = target;
    ra_pass_run(dp->ra, &pass->run_params);
    ret = true;

error:
    // Re-add the shader to the internal pool of shaders
    pl_shader_reset(sh);
    TARRAY_APPEND(dp, dp->shaders, dp->num_shaders, sh);
    return ret;
}
