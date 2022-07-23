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
#include "formats.h"
#include "utils.h"

int gl_desc_namespace(pl_gpu gpu, enum pl_desc_type type)
{
    return (int) type;
}

#define CACHE_MAGIC {'P','L','G','L'}
#define CACHE_VERSION 1
static const char gl_cache_magic[4] = CACHE_MAGIC;

struct gl_cache_header {
    char magic[sizeof(gl_cache_magic)];
    int cache_version;
    GLenum format;
};

static GLuint load_cached_program(pl_gpu gpu, const struct pl_pass_params *params)
{
    if (!gl_test_ext(gpu, "GL_ARB_get_program_binary", 41, 30))
        return 0;

    pl_str cache = {
        .buf = (void *) params->cached_program,
        .len = params->cached_program_len,
    };

    if (cache.len < sizeof(struct gl_cache_header))
        return false;

    struct gl_cache_header *header = (struct gl_cache_header *) cache.buf;
    cache = pl_str_drop(cache, sizeof(*header));

    if (strncmp(header->magic, gl_cache_magic, sizeof(gl_cache_magic)) != 0)
        return 0;
    if (header->cache_version != CACHE_VERSION)
        return 0;

    GLuint prog = glCreateProgram();
    if (!gl_check_err(gpu, "load_cached_program: glCreateProgram"))
        return 0;

    glProgramBinary(prog, header->format, cache.buf, cache.len);
    glGetError(); // discard potential useless error

    GLint status = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &status);
    if (status)
        return prog;

    glDeleteProgram(prog);
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
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);

    GLint status = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    GLint log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);

    enum pl_log_level level = gl_log_level(status, log_length);
    if (pl_msg_test(gpu->log, level)) {
        GLchar *logstr = pl_zalloc(NULL, log_length + 1);
        glGetShaderInfoLog(shader, log_length, NULL, logstr);
        PL_MSG(gpu, level, "shader compile log (status=%d): %s", status, logstr);
        pl_free(logstr);
    }

    if (!status || !gl_check_err(gpu, "gl_attach_shader"))
        goto error;

    glAttachShader(program, shader);
    glDeleteShader(shader);
    return true;

error:
    glDeleteShader(shader);
    return false;
}

static GLuint gl_compile_program(pl_gpu gpu, const struct pl_pass_params *params)
{
    GLuint prog = glCreateProgram();
    bool ok = true;

    switch (params->type) {
    case PL_PASS_COMPUTE:
        ok &= gl_attach_shader(gpu, prog, GL_COMPUTE_SHADER, params->glsl_shader);
        break;
    case PL_PASS_RASTER:
        ok &= gl_attach_shader(gpu, prog, GL_VERTEX_SHADER, params->vertex_shader);
        ok &= gl_attach_shader(gpu, prog, GL_FRAGMENT_SHADER, params->glsl_shader);
        for (int i = 0; i < params->num_vertex_attribs; i++)
            glBindAttribLocation(prog, i, params->vertex_attribs[i].name);
        break;
    case PL_PASS_INVALID:
    case PL_PASS_TYPE_COUNT:
        pl_unreachable();
    }

    if (!ok || !gl_check_err(gpu, "gl_compile_program: attach shader"))
        goto error;

    glLinkProgram(prog);
    GLint status = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &status);
    GLint log_length = 0;
    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &log_length);

    enum pl_log_level level = gl_log_level(status, log_length);
    if (pl_msg_test(gpu->log, level)) {
        GLchar *logstr = pl_zalloc(NULL, log_length + 1);
        glGetProgramInfoLog(prog, log_length, NULL, logstr);
        PL_MSG(gpu, level, "shader link log (status=%d): %s", status, logstr);
        pl_free(logstr);
    }

    if (!gl_check_err(gpu, "gl_compile_program: link program"))
        goto error;

    return prog;

error:
    glDeleteProgram(prog);
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
    if (!MAKE_CURRENT()) {
        PL_ERR(gpu, "Failed uninitializing pass, leaking resources!");
        return;
    }

    struct pl_pass_gl *pass_gl = PL_PRIV(pass);
    if (pass_gl->vao)
        glDeleteVertexArrays(1, &pass_gl->vao);
    glDeleteBuffers(1, &pass_gl->index_buffer);
    glDeleteBuffers(1, &pass_gl->buffer);
    glDeleteProgram(pass_gl->program);

    gl_check_err(gpu, "gl_pass_destroy");
    RELEASE_CURRENT();
    pl_free((void *) pass);
}

static void gl_update_va(pl_pass pass, size_t vbo_offset)
{
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

        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, va->fmt->num_components, glfmt->type, norm,
                              pass->params.vertex_stride,
                              (void *) (va->offset + vbo_offset));
    }
}

pl_pass gl_pass_create(pl_gpu gpu, const struct pl_pass_params *params)
{
    if (!MAKE_CURRENT())
        return NULL;

    struct pl_gl *p = PL_PRIV(gpu);
    struct pl_pass *pass = pl_zalloc_obj(NULL, pass, struct pl_pass_gl);
    struct pl_pass_gl *pass_gl = PL_PRIV(pass);
    pass->params = pl_pass_params_copy(pass, params);

    // Load/Compile program
    if ((pass_gl->program = load_cached_program(gpu, params))) {
        PL_DEBUG(gpu, "Using cached GL program");
    } else {
        clock_t start = clock();
        pass_gl->program = gl_compile_program(gpu, params);
        pl_log_cpu_time(gpu->log, start, clock(), "compiling shader");
    }

    if (!pass_gl->program)
        goto error;

    // Update program cache if possible
    if (gl_test_ext(gpu, "GL_ARB_get_program_binary", 41, 30)) {
        GLint size = 0;
        glGetProgramiv(pass_gl->program, GL_PROGRAM_BINARY_LENGTH, &size);

        if (size > 0) {
            uint8_t *buffer = pl_alloc(NULL, size);
            GLsizei actual_size = 0;
            struct gl_cache_header header = {
                .magic = CACHE_MAGIC,
                .cache_version = CACHE_VERSION,
            };

            glGetProgramBinary(pass_gl->program, size, &actual_size,
                               &header.format, buffer);
            if (actual_size > 0) {
                pl_str cache = {0};
                pl_str_append(pass, &cache, (pl_str) { (void *) &header, sizeof(header) });
                pl_str_append(pass, &cache, (pl_str) { buffer, actual_size });
                pass->params.cached_program = cache.buf;
                pass->params.cached_program_len = cache.len;
            }

            pl_free(buffer);
        }

        if (!gl_check_err(gpu, "gl_pass_create: get program binary")) {
            PL_WARN(gpu, "Failed generating program binary.. ignoring");
            pl_free((void *) pass->params.cached_program);
            pass->params.cached_program = NULL;
            pass->params.cached_program_len = 0;
        }
    }

    glUseProgram(pass_gl->program);
    pass_gl->var_locs = pl_calloc(pass, params->num_variables, sizeof(GLint));

    for (int i = 0; i < params->num_variables; i++) {
        pass_gl->var_locs[i] = glGetUniformLocation(pass_gl->program,
                                                    params->variables[i].name);

        // Due to OpenGL API restrictions, we need to ensure that this is a
        // variable type we can actually *update*. Fortunately, this is easily
        // checked by virtue of the fact that all legal combinations of
        // parameters will have a valid GLSL type name
        if (!pl_var_glsl_type_name(params->variables[i])) {
            glUseProgram(0);
            PL_ERR(gpu, "Input variable '%s' does not match any known type!",
                   params->variables[i].name);
            goto error;
        }
    }

    for (int i = 0; i < params->num_descriptors; i++) {
        // For compatibility with older OpenGL, we need to explicitly update
        // the texture/image unit bindings after creating the shader program,
        // since specifying it directly requires GLSL 4.20+
        GLint loc = glGetUniformLocation(pass_gl->program, params->descriptors[i].name);
        glUniform1i(loc, params->descriptors[i].binding);
    }

    glUseProgram(0);

    // Initialize the VAO and single vertex buffer
    glGenBuffers(1, &pass_gl->buffer);
    if (p->has_vao) {
        glGenVertexArrays(1, &pass_gl->vao);
        glBindBuffer(GL_ARRAY_BUFFER, pass_gl->buffer);
        glBindVertexArray(pass_gl->vao);
        gl_update_va(pass, 0);
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    if (!gl_check_err(gpu, "gl_pass_create"))
        goto error;

    RELEASE_CURRENT();
    return pass;

error:
    PL_ERR(gpu, "Failed creating pass");
    gl_pass_destroy(gpu, pass);
    RELEASE_CURRENT();
    return NULL;
}

static void update_var(pl_pass pass, const struct pl_var_update *vu)
{
    struct pl_pass_gl *pass_gl = PL_PRIV(pass);
    const struct pl_var *var = &pass->params.variables[vu->index];
    GLint loc = pass_gl->var_locs[vu->index];

    switch (var->type) {
    case PL_VAR_SINT: {
        const int *i = vu->data;
        pl_assert(var->dim_m == 1);
        switch (var->dim_v) {
        case 1: glUniform1iv(loc, var->dim_a, i); break;
        case 2: glUniform2iv(loc, var->dim_a, i); break;
        case 3: glUniform3iv(loc, var->dim_a, i); break;
        case 4: glUniform4iv(loc, var->dim_a, i); break;
        default: pl_unreachable();
        }
        return;
    }
    case PL_VAR_UINT: {
        const unsigned int *u = vu->data;
        pl_assert(var->dim_m == 1);
        switch (var->dim_v) {
        case 1: glUniform1uiv(loc, var->dim_a, u); break;
        case 2: glUniform2uiv(loc, var->dim_a, u); break;
        case 3: glUniform3uiv(loc, var->dim_a, u); break;
        case 4: glUniform4uiv(loc, var->dim_a, u); break;
        default: pl_unreachable();
        }
        return;
    }
    case PL_VAR_FLOAT: {
        const float *f = vu->data;
        if (var->dim_m == 1) {
            switch (var->dim_v) {
            case 1: glUniform1fv(loc, var->dim_a, f); break;
            case 2: glUniform2fv(loc, var->dim_a, f); break;
            case 3: glUniform3fv(loc, var->dim_a, f); break;
            case 4: glUniform4fv(loc, var->dim_a, f); break;
            default: pl_unreachable();
            }
        } else if (var->dim_m == 2 && var->dim_v == 2) {
            glUniformMatrix2fv(loc, var->dim_a, GL_FALSE, f);
        } else if (var->dim_m == 3 && var->dim_v == 3) {
            glUniformMatrix3fv(loc, var->dim_a, GL_FALSE, f);
        } else if (var->dim_m == 4 && var->dim_v == 4) {
            glUniformMatrix4fv(loc, var->dim_a, GL_FALSE, f);
        } else if (var->dim_m == 2 && var->dim_v == 3) {
            glUniformMatrix2x3fv(loc, var->dim_a, GL_FALSE, f);
        } else if (var->dim_m == 3 && var->dim_v == 2) {
            glUniformMatrix3x2fv(loc, var->dim_a, GL_FALSE, f);
        } else if (var->dim_m == 2 && var->dim_v == 4) {
            glUniformMatrix2x4fv(loc, var->dim_a, GL_FALSE, f);
        } else if (var->dim_m == 4 && var->dim_v == 2) {
            glUniformMatrix4x2fv(loc, var->dim_a, GL_FALSE, f);
        } else if (var->dim_m == 3 && var->dim_v == 4) {
            glUniformMatrix3x4fv(loc, var->dim_a, GL_FALSE, f);
        } else if (var->dim_m == 4 && var->dim_v == 3) {
            glUniformMatrix4x3fv(loc, var->dim_a, GL_FALSE, f);
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

static void update_desc(pl_pass pass, int index, const struct pl_desc_binding *db)
{
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
        glActiveTexture(GL_TEXTURE0 + desc->binding);
        glBindTexture(tex_gl->target, tex_gl->texture);

        GLint filter = filters[db->sample_mode];
        GLint wrap = wraps[db->address_mode];
        glTexParameteri(tex_gl->target, GL_TEXTURE_MIN_FILTER, filter);
        glTexParameteri(tex_gl->target, GL_TEXTURE_MAG_FILTER, filter);
        switch (pl_tex_params_dimension(tex->params)) {
        case 3: glTexParameteri(tex_gl->target, GL_TEXTURE_WRAP_R, wrap);
            // fall through
        case 2:
            glTexParameteri(tex_gl->target, GL_TEXTURE_WRAP_T, wrap);
            // fall through
        case 1:
            glTexParameteri(tex_gl->target, GL_TEXTURE_WRAP_S, wrap);
            break;
        }
        return;
    }
    case PL_DESC_STORAGE_IMG: {
        pl_tex tex = db->object;
        struct pl_tex_gl *tex_gl = PL_PRIV(tex);
        glBindImageTexture(desc->binding, tex_gl->texture, 0, GL_FALSE, 0,
                           access[desc->access], tex_gl->iformat);
        return;
    }
    case PL_DESC_BUF_UNIFORM: {
        pl_buf buf = db->object;
        struct pl_buf_gl *buf_gl = PL_PRIV(buf);
        glBindBufferRange(GL_UNIFORM_BUFFER, desc->binding, buf_gl->buffer,
                          buf_gl->offset, buf->params.size);
        return;
    }
    case PL_DESC_BUF_STORAGE: {
        pl_buf buf = db->object;
        struct pl_buf_gl *buf_gl = PL_PRIV(buf);
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, desc->binding, buf_gl->buffer,
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

static void unbind_desc(pl_pass pass, int index, const struct pl_desc_binding *db)
{
    const struct pl_desc *desc = &pass->params.descriptors[index];

    switch (desc->type) {
    case PL_DESC_SAMPLED_TEX: {
        pl_tex tex = db->object;
        struct pl_tex_gl *tex_gl = PL_PRIV(tex);
        glActiveTexture(GL_TEXTURE0 + desc->binding);
        glBindTexture(tex_gl->target, 0);
        return;
    }
    case PL_DESC_STORAGE_IMG: {
        pl_tex tex = db->object;
        struct pl_tex_gl *tex_gl = PL_PRIV(tex);
        glBindImageTexture(desc->binding, 0, 0, GL_FALSE, 0,
                           GL_WRITE_ONLY, GL_R32F);
        if (desc->access != PL_DESC_ACCESS_READONLY)
            glMemoryBarrier(tex_gl->barrier);
        return;
    }
    case PL_DESC_BUF_UNIFORM:
        glBindBufferBase(GL_UNIFORM_BUFFER, desc->binding, 0);
        return;
    case PL_DESC_BUF_STORAGE: {
        pl_buf buf = db->object;
        struct pl_buf_gl *buf_gl = PL_PRIV(buf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, desc->binding, 0);
        if (desc->access != PL_DESC_ACCESS_READONLY)
            glMemoryBarrier(buf_gl->barrier);
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
    if (!MAKE_CURRENT())
        return;

    pl_pass pass = params->pass;
    struct pl_pass_gl *pass_gl = PL_PRIV(pass);
    struct pl_gl *p = PL_PRIV(gpu);

    glUseProgram(pass_gl->program);

    for (int i = 0; i < params->num_var_updates; i++)
        update_var(pass, &params->var_updates[i]);
    for (int i = 0; i < pass->params.num_descriptors; i++)
        update_desc(pass, i, &params->desc_bindings[i]);
    glActiveTexture(GL_TEXTURE0);

    if (!gl_check_err(gpu, "gl_pass_run: updating uniforms")) {
        RELEASE_CURRENT();
        return;
    }

    switch (pass->params.type) {
    case PL_PASS_RASTER: {
        struct pl_tex_gl *target_gl = PL_PRIV(params->target);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target_gl->fbo);
        if (!pass->params.load_target && p->has_invalidate_fb) {
            GLenum fb = target_gl->fbo ? GL_COLOR_ATTACHMENT0 : GL_COLOR;
            glInvalidateFramebuffer(GL_DRAW_FRAMEBUFFER, 1, &fb);
        }

        glViewport(params->viewport.x0, params->viewport.y0,
                   pl_rect_w(params->viewport), pl_rect_h(params->viewport));
        glScissor(params->scissors.x0, params->scissors.y0,
                  pl_rect_w(params->scissors), pl_rect_h(params->scissors));
        glEnable(GL_SCISSOR_TEST);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        gl_check_err(gpu, "gl_pass_run: enabling viewport/scissor");

        const struct pl_blend_params *blend = pass->params.blend_params;
        if (blend) {
            static const GLenum map_blend[] = {
                [PL_BLEND_ZERO]                 = GL_ZERO,
                [PL_BLEND_ONE]                  = GL_ONE,
                [PL_BLEND_SRC_ALPHA]            = GL_SRC_ALPHA,
                [PL_BLEND_ONE_MINUS_SRC_ALPHA]  = GL_ONE_MINUS_SRC_ALPHA,
            };

            glBlendFuncSeparate(map_blend[blend->src_rgb],
                                map_blend[blend->dst_rgb],
                                map_blend[blend->src_alpha],
                                map_blend[blend->dst_alpha]);
            glEnable(GL_BLEND);
            gl_check_err(gpu, "gl_pass_run: enabling blend");
        }

        // Update VBO and VAO
        pl_buf vert = params->vertex_buf;
        struct pl_buf_gl *vert_gl = vert ? PL_PRIV(vert) : NULL;
        glBindBuffer(GL_ARRAY_BUFFER, vert ? vert_gl->buffer : pass_gl->buffer);

        if (!vert) {
            // Update the buffer directly. In theory we could also do a memcmp
            // cache here to avoid unnecessary updates.
            glBufferData(GL_ARRAY_BUFFER, pl_vertex_buf_size(params),
                         params->vertex_data, GL_STREAM_DRAW);
        }

        if (pass_gl->vao)
            glBindVertexArray(pass_gl->vao);

        uint64_t vert_id = vert ? vert_gl->id : 0;
        size_t vert_offset = vert ? params->buf_offset : 0;
        if (!pass_gl->vao || pass_gl->vao_id != vert_id ||
             pass_gl->vao_offset != vert_offset)
        {
            // We need to update the VAO when the buffer ID or offset changes
            gl_update_va(pass, vert_offset);
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
                glGenBuffers(1, &pass_gl->index_buffer); // lazily allocated
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pass_gl->index_buffer);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, pl_index_buf_size(params),
                         params->index_data, GL_STREAM_DRAW);
            glDrawElements(mode, params->vertex_count,
                           index_fmts[params->index_fmt], 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        } else if (params->index_buf) {

            // The pointer argument becomes the index buffer offset
            struct pl_buf_gl *index_gl = PL_PRIV(params->index_buf);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_gl->buffer);
            glDrawElements(mode, params->vertex_count, GL_UNSIGNED_SHORT,
                           (void *) params->index_offset);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        } else {

            // Note: the VBO offset is handled in the VAO
            glDrawArrays(mode, 0, params->vertex_count);
        }

        gl_timer_end(gpu, params->timer);
        gl_check_err(gpu, "gl_pass_run: drawing");

        if (pass_gl->vao) {
            glBindVertexArray(0);
        } else {
            for (int i = 0; i < pass->params.num_vertex_attribs; i++)
                glDisableVertexAttribArray(i);
        }

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_BLEND);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        break;
    }

    case PL_PASS_COMPUTE:
        gl_timer_begin(gpu, params->timer);
        glDispatchCompute(params->compute_groups[0],
                          params->compute_groups[1],
                          params->compute_groups[2]);
        gl_timer_end(gpu, params->timer);
        break;

    case PL_PASS_INVALID:
    case PL_PASS_TYPE_COUNT:
        pl_unreachable();
    }

    for (int i = 0; i < pass->params.num_descriptors; i++)
        unbind_desc(pass, i, &params->desc_bindings[i]);
    glActiveTexture(GL_TEXTURE0);

    glUseProgram(0);
    gl_check_err(gpu, "gl_pass_run");
    RELEASE_CURRENT();
}
