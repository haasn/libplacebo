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

#include "gpu.h"
#include "cache.h"
#include "formats.h"
#include "utils.h"

int gl_desc_namespace(pl_gpu gpu, enum pl_desc_type type)
{
    return (int) type;
}

struct gl_cache_header {
    GLenum format;
};

static GLuint load_cached_program(pl_gpu gpu, pl_cache cache, pl_cache_obj *obj)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!gl_test_ext(gpu, "GL_ARB_get_program_binary", 41, 30))
        return 0;

    if (!pl_cache_get(cache, obj))
        return 0;

    if (obj->size < sizeof(struct gl_cache_header))
        return 0;

    GLuint prog = gl->CreateProgram();
    if (!gl_check_err(gpu, "load_cached_program: glCreateProgram"))
        return 0;

    struct gl_cache_header *header = (struct gl_cache_header *) obj->data;
    pl_str rest = (pl_str) { obj->data, obj->size };
    rest = pl_str_drop(rest, sizeof(*header));
    gl->ProgramBinary(prog, header->format, rest.buf, rest.len);
    gl->GetError(); // discard potential useless error

    GLint status = 0;
    gl->GetProgramiv(prog, GL_LINK_STATUS, &status);
    if (status)
        return prog;

    gl->DeleteProgram(prog);
    gl_check_err(gpu, "load_cached_program: glProgramBinary");
    return 0;
}

static enum pl_log_level gl_log_level(GLint status, GLint log_length)
{
    if (!status) {
        return PL_LOG_ERR;
    } else if (log_length > 0) {
        return PL_LOG_INFO;
    } else {
        return PL_LOG_DEBUG;
    }
}

static bool gl_attach_shader(pl_gpu gpu, GLuint program, GLenum type, const char *src)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    GLuint shader = gl->CreateShader(type);
    gl->ShaderSource(shader, 1, &src, NULL);
    gl->CompileShader(shader);

    GLint status = 0;
    gl->GetShaderiv(shader, GL_COMPILE_STATUS, &status);
    GLint log_length = 0;
    gl->GetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);

    enum pl_log_level level = gl_log_level(status, log_length);
    if (pl_msg_test(gpu->log, level)) {
        GLchar *logstr = pl_zalloc(NULL, log_length + 1);
        gl->GetShaderInfoLog(shader, log_length, NULL, logstr);
        PL_MSG(gpu, level, "shader compile log (status=%d): %s", status, logstr);
        pl_free(logstr);
    }

    if (!status || !gl_check_err(gpu, "gl_attach_shader"))
        goto error;

    gl->AttachShader(program, shader);
    gl->DeleteShader(shader);
    return true;

error:
    gl->DeleteShader(shader);
    return false;
}

static GLuint gl_compile_program(pl_gpu gpu, const struct pl_pass_params *params)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    GLuint prog = gl->CreateProgram();
    bool ok = true;

    switch (params->type) {
    case PL_PASS_COMPUTE:
        ok &= gl_attach_shader(gpu, prog, GL_COMPUTE_SHADER, params->glsl_shader);
        break;
    case PL_PASS_RASTER:
        ok &= gl_attach_shader(gpu, prog, GL_VERTEX_SHADER, params->vertex_shader);
        ok &= gl_attach_shader(gpu, prog, GL_FRAGMENT_SHADER, params->glsl_shader);
        for (int i = 0; i < params->num_vertex_attribs; i++)
            gl->BindAttribLocation(prog, i, params->vertex_attribs[i].name);
        break;
    case PL_PASS_INVALID:
    case PL_PASS_TYPE_COUNT:
        pl_unreachable();
    }

    if (!ok || !gl_check_err(gpu, "gl_compile_program: attach shader"))
        goto error;

    gl->LinkProgram(prog);
    GLint status = 0;
    gl->GetProgramiv(prog, GL_LINK_STATUS, &status);
    GLint log_length = 0;
    gl->GetProgramiv(prog, GL_INFO_LOG_LENGTH, &log_length);

    enum pl_log_level level = gl_log_level(status, log_length);
    if (pl_msg_test(gpu->log, level)) {
        GLchar *logstr = pl_zalloc(NULL, log_length + 1);
        gl->GetProgramInfoLog(prog, log_length, NULL, logstr);
        PL_MSG(gpu, level, "shader link log (status=%d): %s", status, logstr);
        pl_free(logstr);
    }

    if (!gl_check_err(gpu, "gl_compile_program: link program"))
        goto error;

    return prog;

error:
    gl->DeleteProgram(prog);
    PL_ERR(gpu, "Failed compiling/linking GLSL program");
    return 0;
}

// For pl_pass.priv
struct pl_pass_gl {
    GLuint program;
    GLuint vao;         // the VAO object
    uint64_t vao_id;    // buf_gl.id of VAO
    size_t vao_offset;  // VBO offset of VAO
    GLuint buffer;      // VBO for raw vertex pointers
    GLuint index_buffer;
    GLint *var_locs;
};

void gl_pass_destroy(pl_gpu gpu, pl_pass pass)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT()) {
        PL_ERR(gpu, "Failed uninitializing pass, leaking resources!");
        return;
    }

    struct pl_pass_gl *pass_gl = PL_PRIV(pass);
    if (pass_gl->vao)
        gl->DeleteVertexArrays(1, &pass_gl->vao);
    gl->DeleteBuffers(1, &pass_gl->index_buffer);
    gl->DeleteBuffers(1, &pass_gl->buffer);
    gl->DeleteProgram(pass_gl->program);

    gl_check_err(gpu, "gl_pass_destroy");
    RELEASE_CURRENT();
    pl_free((void *) pass);
}

static void gl_update_va(pl_gpu gpu, pl_pass pass, size_t vbo_offset)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    for (int i = 0; i < pass->params.num_vertex_attribs; i++) {
        const struct pl_vertex_attrib *va = &pass->params.vertex_attribs[i];
        const struct gl_format **glfmtp = PL_PRIV(va->fmt);
        const struct gl_format *glfmt = *glfmtp;

        bool norm = false;
        switch (va->fmt->type) {
        case PL_FMT_UNORM:
        case PL_FMT_SNORM:
            norm = true;
            break;

        case PL_FMT_UNKNOWN:
        case PL_FMT_FLOAT:
        case PL_FMT_UINT:
        case PL_FMT_SINT:
            break;
        case PL_FMT_TYPE_COUNT:
            pl_unreachable();
        }

        gl->EnableVertexAttribArray(i);
        gl->VertexAttribPointer(i, va->fmt->num_components, glfmt->type, norm,
                                pass->params.vertex_stride,
                                (void *) (va->offset + vbo_offset));
    }
}

pl_pass gl_pass_create(pl_gpu gpu, const struct pl_pass_params *params)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT())
        return NULL;

    struct pl_gl *p = PL_PRIV(gpu);
    struct pl_pass_t *pass = pl_zalloc_obj(NULL, pass, struct pl_pass_gl);
    struct pl_pass_gl *pass_gl = PL_PRIV(pass);
    pl_cache cache = pl_gpu_cache(gpu);
    pass->params = pl_pass_params_copy(pass, params);

    pl_cache_obj obj = { .key = CACHE_KEY_GL_PROG };
    if (cache) {
        pl_hash_merge(&obj.key, p->sig);
        pl_hash_merge(&obj.key, pl_str0_hash(params->glsl_shader));
        if (params->type == PL_PASS_RASTER)
            pl_hash_merge(&obj.key, pl_str0_hash(params->vertex_shader));
    }

    // Load/Compile program
    if ((pass_gl->program = load_cached_program(gpu, cache, &obj))) {
        PL_DEBUG(gpu, "Using cached GL program");
    } else {
        pl_clock_t start = pl_clock_now();
        pass_gl->program = gl_compile_program(gpu, params);
        pl_log_cpu_time(gpu->log, start, pl_clock_now(), "compiling shader");
    }

    if (!pass_gl->program)
        goto error;

    // Update program cache if possible
    if (cache && gl_test_ext(gpu, "GL_ARB_get_program_binary", 41, 30)) {
        GLint buf_size = 0;
        gl->GetProgramiv(pass_gl->program, GL_PROGRAM_BINARY_LENGTH, &buf_size);
        if (buf_size > 0) {
            buf_size += sizeof(struct gl_cache_header);
            pl_cache_obj_resize(NULL, &obj, buf_size);
            struct gl_cache_header *header = obj.data;
            void *buffer = &header[1];
            GLsizei binary_size = 0;
            gl->GetProgramBinary(pass_gl->program, buf_size, &binary_size,
                                 &header->format, buffer);
            bool ok = gl_check_err(gpu, "gl_pass_create: get program binary");
            if (ok) {
                obj.size = sizeof(*header) + binary_size;
                pl_assert(obj.size <= buf_size);
                pl_cache_set(cache, &obj);
            }
        }
    }

    gl->UseProgram(pass_gl->program);
    pass_gl->var_locs = pl_calloc(pass, params->num_variables, sizeof(GLint));

    for (int i = 0; i < params->num_variables; i++) {
        pass_gl->var_locs[i] = gl->GetUniformLocation(pass_gl->program,
                                                      params->variables[i].name);

        // Due to OpenGL API restrictions, we need to ensure that this is a
        // variable type we can actually *update*. Fortunately, this is easily
        // checked by virtue of the fact that all legal combinations of
        // parameters will have a valid GLSL type name
        if (!pl_var_glsl_type_name(params->variables[i])) {
            gl->UseProgram(0);
            PL_ERR(gpu, "Input variable '%s' does not match any known type!",
                   params->variables[i].name);
            goto error;
        }
    }

    for (int i = 0; i < params->num_descriptors; i++) {
        const struct pl_desc *desc = &params->descriptors[i];
        switch (desc->type) {
        case PL_DESC_SAMPLED_TEX:
        case PL_DESC_STORAGE_IMG: {
            // For compatibility with older OpenGL, we need to explicitly
            // update the texture/image unit bindings after creating the shader
            // program, since specifying it directly requires GLSL 4.20+
            GLint loc = gl->GetUniformLocation(pass_gl->program, desc->name);
            gl->Uniform1i(loc, desc->binding);
            break;
        }
        case PL_DESC_BUF_UNIFORM: {
            GLuint idx = gl->GetUniformBlockIndex(pass_gl->program, desc->name);
            gl->UniformBlockBinding(pass_gl->program, idx, desc->binding);
            break;
        }
        case PL_DESC_BUF_STORAGE: {
            GLuint idx = gl->GetProgramResourceIndex(pass_gl->program,
                                                     GL_SHADER_STORAGE_BLOCK,
                                                     desc->name);
            gl->ShaderStorageBlockBinding(pass_gl->program, idx, desc->binding);
            break;
        }
        case PL_DESC_BUF_TEXEL_UNIFORM:
        case PL_DESC_BUF_TEXEL_STORAGE:
            assert(!"unimplemented"); // TODO
        case PL_DESC_INVALID:
        case PL_DESC_TYPE_COUNT:
            pl_unreachable();
        }
    }

    gl->UseProgram(0);

    // Initialize the VAO and single vertex buffer
    gl->GenBuffers(1, &pass_gl->buffer);
    if (p->has_vao) {
        gl->GenVertexArrays(1, &pass_gl->vao);
        gl->BindBuffer(GL_ARRAY_BUFFER, pass_gl->buffer);
        gl->BindVertexArray(pass_gl->vao);
        gl_update_va(gpu, pass, 0);
        gl->BindVertexArray(0);
        gl->BindBuffer(GL_ARRAY_BUFFER, 0);
    }

    if (!gl_check_err(gpu, "gl_pass_create"))
        goto error;

    pl_cache_obj_free(&obj);
    RELEASE_CURRENT();
    return pass;

error:
    PL_ERR(gpu, "Failed creating pass");
    pl_cache_obj_free(&obj);
    gl_pass_destroy(gpu, pass);
    RELEASE_CURRENT();
    return NULL;
}

static void update_var(pl_gpu gpu, pl_pass pass,
                       const struct pl_var_update *vu)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    struct pl_pass_gl *pass_gl = PL_PRIV(pass);
    const struct pl_var *var = &pass->params.variables[vu->index];
    GLint loc = pass_gl->var_locs[vu->index];

    switch (var->type) {
    case PL_VAR_SINT: {
        const int *i = vu->data;
        pl_assert(var->dim_m == 1);
        switch (var->dim_v) {
        case 1: gl->Uniform1iv(loc, var->dim_a, i); break;
        case 2: gl->Uniform2iv(loc, var->dim_a, i); break;
        case 3: gl->Uniform3iv(loc, var->dim_a, i); break;
        case 4: gl->Uniform4iv(loc, var->dim_a, i); break;
        default: pl_unreachable();
        }
        return;
    }
    case PL_VAR_UINT: {
        const unsigned int *u = vu->data;
        pl_assert(var->dim_m == 1);
        switch (var->dim_v) {
        case 1: gl->Uniform1uiv(loc, var->dim_a, u); break;
        case 2: gl->Uniform2uiv(loc, var->dim_a, u); break;
        case 3: gl->Uniform3uiv(loc, var->dim_a, u); break;
        case 4: gl->Uniform4uiv(loc, var->dim_a, u); break;
        default: pl_unreachable();
        }
        return;
    }
    case PL_VAR_FLOAT: {
        const float *f = vu->data;
        if (var->dim_m == 1) {
            switch (var->dim_v) {
            case 1: gl->Uniform1fv(loc, var->dim_a, f); break;
            case 2: gl->Uniform2fv(loc, var->dim_a, f); break;
            case 3: gl->Uniform3fv(loc, var->dim_a, f); break;
            case 4: gl->Uniform4fv(loc, var->dim_a, f); break;
            default: pl_unreachable();
            }
        } else if (var->dim_m == 2 && var->dim_v == 2) {
            gl->UniformMatrix2fv(loc, var->dim_a, GL_FALSE, f);
        } else if (var->dim_m == 3 && var->dim_v == 3) {
            gl->UniformMatrix3fv(loc, var->dim_a, GL_FALSE, f);
        } else if (var->dim_m == 4 && var->dim_v == 4) {
            gl->UniformMatrix4fv(loc, var->dim_a, GL_FALSE, f);
        } else if (var->dim_m == 2 && var->dim_v == 3) {
            gl->UniformMatrix2x3fv(loc, var->dim_a, GL_FALSE, f);
        } else if (var->dim_m == 3 && var->dim_v == 2) {
            gl->UniformMatrix3x2fv(loc, var->dim_a, GL_FALSE, f);
        } else if (var->dim_m == 2 && var->dim_v == 4) {
            gl->UniformMatrix2x4fv(loc, var->dim_a, GL_FALSE, f);
        } else if (var->dim_m == 4 && var->dim_v == 2) {
            gl->UniformMatrix4x2fv(loc, var->dim_a, GL_FALSE, f);
        } else if (var->dim_m == 3 && var->dim_v == 4) {
            gl->UniformMatrix3x4fv(loc, var->dim_a, GL_FALSE, f);
        } else if (var->dim_m == 4 && var->dim_v == 3) {
            gl->UniformMatrix4x3fv(loc, var->dim_a, GL_FALSE, f);
        } else {
            pl_unreachable();
        }
        return;
    }

    case PL_VAR_INVALID:
    case PL_VAR_TYPE_COUNT:
        break;
    }

    pl_unreachable();
}

static void update_desc(pl_gpu gpu, pl_pass pass, int index,
                        const struct pl_desc_binding *db)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    const struct pl_desc *desc = &pass->params.descriptors[index];

    static const GLenum access[] = {
        [PL_DESC_ACCESS_READWRITE] = GL_READ_WRITE,
        [PL_DESC_ACCESS_READONLY]  = GL_READ_ONLY,
        [PL_DESC_ACCESS_WRITEONLY] = GL_WRITE_ONLY,
    };

    static const GLint wraps[PL_TEX_ADDRESS_MODE_COUNT] = {
        [PL_TEX_ADDRESS_CLAMP]  = GL_CLAMP_TO_EDGE,
        [PL_TEX_ADDRESS_REPEAT] = GL_REPEAT,
        [PL_TEX_ADDRESS_MIRROR] = GL_MIRRORED_REPEAT,
    };

    static const GLint filters[PL_TEX_SAMPLE_MODE_COUNT] = {
        [PL_TEX_SAMPLE_NEAREST] = GL_NEAREST,
        [PL_TEX_SAMPLE_LINEAR]  = GL_LINEAR,
    };

    switch (desc->type) {
    case PL_DESC_SAMPLED_TEX: {
        pl_tex tex = db->object;
        struct pl_tex_gl *tex_gl = PL_PRIV(tex);
        gl->ActiveTexture(GL_TEXTURE0 + desc->binding);
        gl->BindTexture(tex_gl->target, tex_gl->texture);

        GLint filter = filters[db->sample_mode];
        GLint wrap = wraps[db->address_mode];
        gl->TexParameteri(tex_gl->target, GL_TEXTURE_MIN_FILTER, filter);
        gl->TexParameteri(tex_gl->target, GL_TEXTURE_MAG_FILTER, filter);
        switch (pl_tex_params_dimension(tex->params)) {
        case 3: gl->TexParameteri(tex_gl->target, GL_TEXTURE_WRAP_R, wrap); // fall through
        case 2: gl->TexParameteri(tex_gl->target, GL_TEXTURE_WRAP_T, wrap); // fall through
        case 1: gl->TexParameteri(tex_gl->target, GL_TEXTURE_WRAP_S, wrap); break;
        }
        return;
    }
    case PL_DESC_STORAGE_IMG: {
        pl_tex tex = db->object;
        struct pl_tex_gl *tex_gl = PL_PRIV(tex);
        gl->BindImageTexture(desc->binding, tex_gl->texture, 0, GL_FALSE, 0,
                             access[desc->access], tex_gl->iformat);
        return;
    }
    case PL_DESC_BUF_UNIFORM: {
        pl_buf buf = db->object;
        struct pl_buf_gl *buf_gl = PL_PRIV(buf);
        gl->BindBufferRange(GL_UNIFORM_BUFFER, desc->binding, buf_gl->buffer,
                            buf_gl->offset, buf->params.size);
        return;
    }
    case PL_DESC_BUF_STORAGE: {
        pl_buf buf = db->object;
        struct pl_buf_gl *buf_gl = PL_PRIV(buf);
        gl->BindBufferRange(GL_SHADER_STORAGE_BUFFER, desc->binding, buf_gl->buffer,
                            buf_gl->offset, buf->params.size);
        return;
    }
    case PL_DESC_BUF_TEXEL_UNIFORM:
    case PL_DESC_BUF_TEXEL_STORAGE:
        assert(!"unimplemented"); // TODO

    case PL_DESC_INVALID:
    case PL_DESC_TYPE_COUNT:
        break;
    }

    pl_unreachable();
}

static void unbind_desc(pl_gpu gpu, pl_pass pass, int index,
                        const struct pl_desc_binding *db)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    const struct pl_desc *desc = &pass->params.descriptors[index];

    switch (desc->type) {
    case PL_DESC_SAMPLED_TEX: {
        pl_tex tex = db->object;
        struct pl_tex_gl *tex_gl = PL_PRIV(tex);
        gl->ActiveTexture(GL_TEXTURE0 + desc->binding);
        gl->BindTexture(tex_gl->target, 0);
        return;
    }
    case PL_DESC_STORAGE_IMG: {
        pl_tex tex = db->object;
        struct pl_tex_gl *tex_gl = PL_PRIV(tex);
        gl->BindImageTexture(desc->binding, 0, 0, GL_FALSE, 0,
                             GL_WRITE_ONLY, GL_R32F);
        if (desc->access != PL_DESC_ACCESS_READONLY)
            gl->MemoryBarrier(tex_gl->barrier);
        return;
    }
    case PL_DESC_BUF_UNIFORM:
        gl->BindBufferBase(GL_UNIFORM_BUFFER, desc->binding, 0);
        return;
    case PL_DESC_BUF_STORAGE: {
        pl_buf buf = db->object;
        struct pl_buf_gl *buf_gl = PL_PRIV(buf);
        gl->BindBufferBase(GL_SHADER_STORAGE_BUFFER, desc->binding, 0);
        if (desc->access != PL_DESC_ACCESS_READONLY)
            gl->MemoryBarrier(buf_gl->barrier);
        return;
    }
    case PL_DESC_BUF_TEXEL_UNIFORM:
    case PL_DESC_BUF_TEXEL_STORAGE:
        assert(!"unimplemented"); // TODO
    case PL_DESC_INVALID:
    case PL_DESC_TYPE_COUNT:
        break;
    }

    pl_unreachable();
}

void gl_pass_run(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT())
        return;

    pl_pass pass = params->pass;
    struct pl_pass_gl *pass_gl = PL_PRIV(pass);
    struct pl_gl *p = PL_PRIV(gpu);

    gl->UseProgram(pass_gl->program);

    for (int i = 0; i < params->num_var_updates; i++)
        update_var(gpu, pass, &params->var_updates[i]);
    for (int i = 0; i < pass->params.num_descriptors; i++)
        update_desc(gpu, pass, i, &params->desc_bindings[i]);
    gl->ActiveTexture(GL_TEXTURE0);

    if (!gl_check_err(gpu, "gl_pass_run: updating uniforms")) {
        RELEASE_CURRENT();
        return;
    }

    switch (pass->params.type) {
    case PL_PASS_RASTER: {
        struct pl_tex_gl *target_gl = PL_PRIV(params->target);
        gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, target_gl->fbo);
        if (!pass->params.load_target && p->has_invalidate_fb) {
            GLenum fb = target_gl->fbo ? GL_COLOR_ATTACHMENT0 : GL_COLOR;
            gl->InvalidateFramebuffer(GL_DRAW_FRAMEBUFFER, 1, &fb);
        }

        gl->Viewport(params->viewport.x0, params->viewport.y0,
                     pl_rect_w(params->viewport), pl_rect_h(params->viewport));
        gl->Scissor(params->scissors.x0, params->scissors.y0,
                    pl_rect_w(params->scissors), pl_rect_h(params->scissors));
        gl->Enable(GL_SCISSOR_TEST);
        gl->Disable(GL_DEPTH_TEST);
        gl->Disable(GL_CULL_FACE);
        gl_check_err(gpu, "gl_pass_run: enabling viewport/scissor");

        const struct pl_blend_params *blend = pass->params.blend_params;
        if (blend) {
            static const GLenum map_blend[] = {
                [PL_BLEND_ZERO]                 = GL_ZERO,
                [PL_BLEND_ONE]                  = GL_ONE,
                [PL_BLEND_SRC_ALPHA]            = GL_SRC_ALPHA,
                [PL_BLEND_ONE_MINUS_SRC_ALPHA]  = GL_ONE_MINUS_SRC_ALPHA,
            };

            gl->BlendFuncSeparate(map_blend[blend->src_rgb],
                                  map_blend[blend->dst_rgb],
                                  map_blend[blend->src_alpha],
                                  map_blend[blend->dst_alpha]);
            gl->Enable(GL_BLEND);
            gl_check_err(gpu, "gl_pass_run: enabling blend");
        }

        // Update VBO and VAO
        pl_buf vert = params->vertex_buf;
        struct pl_buf_gl *vert_gl = vert ? PL_PRIV(vert) : NULL;
        gl->BindBuffer(GL_ARRAY_BUFFER, vert ? vert_gl->buffer : pass_gl->buffer);

        if (!vert) {
            // Update the buffer directly. In theory we could also do a memcmp
            // cache here to avoid unnecessary updates.
            gl->BufferData(GL_ARRAY_BUFFER, pl_vertex_buf_size(params),
                           params->vertex_data, GL_STREAM_DRAW);
        }

        if (pass_gl->vao)
            gl->BindVertexArray(pass_gl->vao);

        uint64_t vert_id = vert ? vert_gl->id : 0;
        size_t vert_offset = vert ? params->buf_offset : 0;
        if (!pass_gl->vao || pass_gl->vao_id != vert_id ||
             pass_gl->vao_offset != vert_offset)
        {
            // We need to update the VAO when the buffer ID or offset changes
            gl_update_va(gpu, pass, vert_offset);
            pass_gl->vao_id = vert_id;
            pass_gl->vao_offset = vert_offset;
        }

        gl_check_err(gpu, "gl_pass_run: update/bind vertex buffer");

        static const GLenum map_prim[PL_PRIM_TYPE_COUNT] = {
            [PL_PRIM_TRIANGLE_LIST]     = GL_TRIANGLES,
            [PL_PRIM_TRIANGLE_STRIP]    = GL_TRIANGLE_STRIP,
        };
        GLenum mode = map_prim[pass->params.vertex_type];

        gl_timer_begin(gpu, params->timer);

        if (params->index_data) {

            static const GLenum index_fmts[PL_INDEX_FORMAT_COUNT] = {
                [PL_INDEX_UINT16] = GL_UNSIGNED_SHORT,
                [PL_INDEX_UINT32] = GL_UNSIGNED_INT,
            };

            // Upload indices to temporary buffer object
            if (!pass_gl->index_buffer)
                gl->GenBuffers(1, &pass_gl->index_buffer); // lazily allocated
            gl->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, pass_gl->index_buffer);
            gl->BufferData(GL_ELEMENT_ARRAY_BUFFER, pl_index_buf_size(params),
                           params->index_data, GL_STREAM_DRAW);
            gl->DrawElements(mode, params->vertex_count,
                             index_fmts[params->index_fmt], 0);
            gl->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        } else if (params->index_buf) {

            // The pointer argument becomes the index buffer offset
            struct pl_buf_gl *index_gl = PL_PRIV(params->index_buf);
            gl->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_gl->buffer);
            gl->DrawElements(mode, params->vertex_count, GL_UNSIGNED_SHORT,
                             (void *) params->index_offset);
            gl->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        } else {

            // Note: the VBO offset is handled in the VAO
            gl->DrawArrays(mode, 0, params->vertex_count);
        }

        gl_timer_end(gpu, params->timer);
        gl_check_err(gpu, "gl_pass_run: drawing");

        if (pass_gl->vao) {
            gl->BindVertexArray(0);
        } else {
            for (int i = 0; i < pass->params.num_vertex_attribs; i++)
                gl->DisableVertexAttribArray(i);
        }

        gl->BindBuffer(GL_ARRAY_BUFFER, 0);
        gl->Disable(GL_SCISSOR_TEST);
        gl->Disable(GL_BLEND);
        gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        break;
    }

    case PL_PASS_COMPUTE:
        gl_timer_begin(gpu, params->timer);
        gl->DispatchCompute(params->compute_groups[0],
                            params->compute_groups[1],
                            params->compute_groups[2]);
        gl_timer_end(gpu, params->timer);
        break;

    case PL_PASS_INVALID:
    case PL_PASS_TYPE_COUNT:
        pl_unreachable();
    }

    for (int i = 0; i < pass->params.num_descriptors; i++)
        unbind_desc(gpu, pass, i, &params->desc_bindings[i]);
    gl->ActiveTexture(GL_TEXTURE0);

    gl->UseProgram(0);
    gl_check_err(gpu, "gl_pass_run");
    RELEASE_CURRENT();
}
