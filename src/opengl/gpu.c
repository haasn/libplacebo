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
#include "common.h"
#include "formats.h"
#include "utils.h"

#ifdef PL_HAVE_UNIX
#include <unistd.h>
#endif

#ifdef PL_HAVE_WIN32
#include <windows.h>
#include <sysinfoapi.h>
#endif

static const struct pl_gpu_fns pl_fns_gl;

static void gl_gpu_destroy(pl_gpu gpu)
{
    struct pl_gl *p = PL_PRIV(gpu);

    pl_gpu_finish(gpu);
    while (p->callbacks.num > 0)
        gl_poll_callbacks(gpu);

    pl_free((void *) gpu);
}

pl_opengl pl_opengl_get(pl_gpu gpu)
{
    const struct pl_gpu_fns *impl = PL_PRIV(gpu);
    if (impl->destroy == gl_gpu_destroy) {
        struct pl_gl *p = (struct pl_gl *) impl;
        return p->gl;
    }

    return NULL;
}

static pl_handle_caps tex_handle_caps(pl_gpu gpu, bool import)
{
    pl_handle_caps caps = 0;
    struct pl_gl *p = PL_PRIV(gpu);

    if (!p->egl_dpy || (!p->has_egl_storage && !p->has_egl_import))
        return 0;

    if (import) {
        if (pl_opengl_has_ext(p->gl, "EGL_EXT_image_dma_buf_import"))
            caps |= PL_HANDLE_DMA_BUF;
    } else if (!import && p->egl_ctx) {
        if (pl_opengl_has_ext(p->gl, "EGL_MESA_image_dma_buf_export"))
            caps |= PL_HANDLE_DMA_BUF;
    }

    return caps;
}

static inline size_t get_page_size(void)
{

#ifdef PL_HAVE_UNIX
    return sysconf(_SC_PAGESIZE);
#endif

#ifdef PL_HAVE_WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return sysInfo.dwAllocationGranularity;
#endif

    pl_assert(!"Unsupported platform!");
}

#define get(pname, field)                   \
    do {                                    \
        GLint tmp = 0;                      \
        gl->GetIntegerv((pname), &tmp);     \
        *(field) = tmp;                     \
    } while (0)

#define geti(pname, i, field)               \
    do {                                    \
        GLint tmp = 0;                      \
        gl->GetIntegeri_v((pname), i, &tmp);\
        *(field) = tmp;                     \
    } while (0)

pl_gpu pl_gpu_create_gl(pl_log log, pl_opengl pl_gl, const struct pl_opengl_params *params)
{
    struct pl_gpu_t *gpu = pl_zalloc_obj(NULL, gpu, struct pl_gl);
    gpu->log = log;

    struct pl_gl *p = PL_PRIV(gpu);
    p->impl = pl_fns_gl;
    p->gl = pl_gl;

    const gl_funcs *gl = gl_funcs_get(gpu);
    struct pl_glsl_version *glsl = &gpu->glsl;
    glsl->gles = gl_is_gles(pl_gl);
    int ver = pl_gl->major * 10 + pl_gl->minor;
    p->gl_ver = glsl->gles ? 0 : ver;
    p->gles_ver = glsl->gles ? ver : 0;

    // If possible, query the GLSL version from the implementation
    const char *glslver_p = (char *) gl->GetString(GL_SHADING_LANGUAGE_VERSION);
    pl_str glslver = pl_str0(glslver_p);
    if (glslver.len) {
        PL_INFO(gpu, "    GL_SHADING_LANGUAGE_VERSION: %.*s", PL_STR_FMT(glslver));
        pl_str_eatstart0(&glslver, "OpenGL ES GLSL ES ");
        int major = 0, minor = 0;
        if (pl_str_sscanf(glslver, "%d.%d", &major, &minor) == 2)
            glsl->version = major * 100 + minor;
    }

    if (!glsl->version) {
        // Otherwise, use the fixed magic versions 100 and 300 for GLES.
        if (p->gles_ver >= 30) {
            glsl->version = 300;
        } else if (p->gles_ver >= 20) {
            glsl->version = 100;
        } else {
            goto error;
        }
    }

    static const int glsl_ver_req = 130;
    if (glsl->version < glsl_ver_req) {
        PL_FATAL(gpu, "GLSL version too old (%d < %d), please use a newer "
                 "OpenGL implementation or downgrade libplacebo!",
                 glsl->version, glsl_ver_req);
        goto error;
    }

    if (params->max_glsl_version && params->max_glsl_version >= glsl_ver_req) {
        glsl->version = PL_MIN(glsl->version, params->max_glsl_version);
        PL_INFO(gpu, "Restricting GLSL version to %d... new version is %d",
                params->max_glsl_version, glsl->version);
    }

    if (gl_test_ext(gpu, "GL_ARB_compute_shader", 43, 0) && glsl->version >= 420) {
        glsl->compute = true;
        get(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &glsl->max_shmem_size);
        get(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &glsl->max_group_threads);
        for (int i = 0; i < 3; i++)
            geti(GL_MAX_COMPUTE_WORK_GROUP_SIZE, i, &glsl->max_group_size[i]);
    }

    if (gl_test_ext(gpu, "GL_ARB_texture_gather", 40, 31) &&
        glsl->version >= (p->gles_ver ? 310 : 400)) {
        if (p->gles_ver)
            p->gather_comps = 4;
        else
            get(GL_MAX_PROGRAM_TEXTURE_GATHER_COMPONENTS_ARB, &p->gather_comps);
        get(GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET_ARB, &glsl->min_gather_offset);
        get(GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET_ARB, &glsl->max_gather_offset);
    }

    // Query all device limits
    struct pl_gpu_limits *limits = &gpu->limits;
    limits->thread_safe = params->make_current;
    limits->callbacks = gl_test_ext(gpu, "GL_ARB_sync", 32, 30);
    limits->align_vertex_stride = 1;
    if (gl_test_ext(gpu, "GL_ARB_pixel_buffer_object", 31, 0)) {
        limits->max_buf_size = SIZE_MAX; // no restriction imposed by GL
        if (gl_test_ext(gpu, "GL_ARB_uniform_buffer_object", 31, 0))
            get(GL_MAX_UNIFORM_BLOCK_SIZE, &limits->max_ubo_size);
        if (gl_test_ext(gpu, "GL_ARB_shader_storage_buffer_object", 43, 0) &&
            gpu->glsl.version >= 140)
        {
            get(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &limits->max_ssbo_size);
        }
        limits->max_vbo_size = limits->max_buf_size; // No additional restrictions
        if (gl_test_ext(gpu, "GL_ARB_buffer_storage", 44, 0)) {
            const char *vendor = (char *) gl->GetString(GL_VENDOR);
            limits->max_mapped_size = limits->max_buf_size;
            limits->max_mapped_vram = limits->max_buf_size;
            limits->host_cached = strcmp(vendor, "AMD") == 0 ||
                                  strcmp(vendor, "NVIDIA Corporation") == 0;
        }
    }

    get(GL_MAX_TEXTURE_SIZE, &limits->max_tex_2d_dim);
    if (gl_test_ext(gpu, "GL_EXT_texture3D", 21, 30))
        get(GL_MAX_3D_TEXTURE_SIZE, &limits->max_tex_3d_dim);
    // There's no equivalent limit for 1D textures for whatever reason, so
    // just set it to the same as the 2D limit
    if (p->gl_ver >= 21)
        limits->max_tex_1d_dim = limits->max_tex_2d_dim;
    limits->buf_transfer = true;

    if (p->gl_ver || p->gles_ver >= 30) {
        get(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS, &limits->max_variable_comps);
    } else {
        // fallback for GLES 2.0, which doesn't have max_comps
        get(GL_MAX_FRAGMENT_UNIFORM_VECTORS, &limits->max_variable_comps);
        limits->max_variable_comps *= 4;
    }

    if (glsl->compute) {
        for (int i = 0; i < 3; i++)
            geti(GL_MAX_COMPUTE_WORK_GROUP_COUNT, i, &limits->max_dispatch[i]);
    }

    // Query import/export support
    p->egl_dpy = params->egl_display;
    p->egl_ctx = params->egl_context;
    p->has_egl_storage = pl_opengl_has_ext(p->gl, "GL_EXT_EGL_image_storage");
    p->has_egl_import = pl_opengl_has_ext(p->gl, "GL_OES_EGL_image_external");
    gpu->export_caps.tex = tex_handle_caps(gpu, false);
    gpu->import_caps.tex = tex_handle_caps(gpu, true);

    if (p->egl_dpy) {
        p->has_modifiers = pl_opengl_has_ext(p->gl,
                                        "EGL_EXT_image_dma_buf_import_modifiers");
    }

    if (pl_opengl_has_ext(pl_gl, "GL_AMD_pinned_memory")) {
        gpu->import_caps.buf |= PL_HANDLE_HOST_PTR;
        gpu->limits.align_host_ptr = get_page_size();
    }

    // Cache some internal capability checks
    p->has_vao = gl_test_ext(gpu, "GL_ARB_vertex_array_object", 30, 30);
    p->has_invalidate_fb = gl_test_ext(gpu, "GL_ARB_invalidate_subdata", 43, 30);
    p->has_invalidate_tex = gl_test_ext(gpu, "GL_ARB_invalidate_subdata", 43, 0);
    p->has_queries = gl_test_ext(gpu, "GL_ARB_timer_query", 30, 0);
    p->has_storage = gl_test_ext(gpu, "GL_ARB_shader_image_load_store", 42, 31);
    p->has_readback = true;

    if (p->has_readback && p->gles_ver) {
        GLuint fbo = 0, tex = 0;
        GLint read_type = 0, read_fmt = 0;
        const GLenum target = p->gles_ver >= 30 ? GL_READ_FRAMEBUFFER : GL_FRAMEBUFFER;
        gl->GenTextures(1, &tex);
        gl->BindTexture(GL_TEXTURE_2D, tex);
        gl->GenFramebuffers(1, &fbo);
        gl->TexImage2D(GL_TEXTURE_2D, 0, GL_R8, 64, 64, 0, GL_RED,
                       GL_UNSIGNED_BYTE, NULL);
        gl->BindFramebuffer(target, fbo);
        gl->FramebufferTexture2D(target, GL_COLOR_ATTACHMENT0,
                                 GL_TEXTURE_2D, tex, 0);
        gl->GetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &read_type);
        gl->GetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &read_fmt);
        if (read_type != GL_UNSIGNED_BYTE || read_fmt != GL_RED) {
            PL_INFO(gpu, "GPU does not seem to support lossless texture "
                    "readback, restricting readback capabilities! This is a "
                    "GLES/driver limitation, there is little we can do to "
                    "work around it.");
            p->has_readback = false;
        }
        gl->BindFramebuffer(target, 0);
        gl->BindTexture(GL_TEXTURE_2D, 0);
        gl->DeleteFramebuffers(1, &fbo);
        gl->DeleteTextures(1, &tex);
    }

    // We simply don't know, so make up some values
    limits->align_tex_xfer_offset = 32;
    limits->align_tex_xfer_pitch = 4;
    limits->fragment_queues = 1;
    limits->compute_queues = glsl->compute ? 1 : 0;

    if (!gl_check_err(gpu, "pl_gpu_create_gl")) {
        PL_WARN(gpu, "Encountered errors while detecting GPU capabilities... "
                "ignoring, but expect limitations/issues");
        p->failed = false;
    }

    // Filter out error messages during format probing
    pl_log_level_cap(gpu->log, PL_LOG_INFO);
    bool formats_ok = gl_setup_formats(gpu);
    pl_log_level_cap(gpu->log, PL_LOG_NONE);
    if (!formats_ok)
        goto error;

    return pl_gpu_finalize(gpu);

error:
    gl_gpu_destroy(gpu);
    return NULL;
}

void gl_buf_destroy(pl_gpu gpu, pl_buf buf)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT()) {
        PL_ERR(gpu, "Failed uninitializing buffer, leaking resources!");
        return;
    }

    struct pl_buf_gl *buf_gl = PL_PRIV(buf);
    if (buf_gl->fence)
        gl->DeleteSync(buf_gl->fence);

    if (buf_gl->mapped) {
        gl->BindBuffer(GL_COPY_WRITE_BUFFER, buf_gl->buffer);
        gl->UnmapBuffer(GL_COPY_WRITE_BUFFER);
        gl->BindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }

    gl->DeleteBuffers(1, &buf_gl->buffer);
    gl_check_err(gpu, "gl_buf_destroy");
    RELEASE_CURRENT();
    pl_free((void *) buf);
}

pl_buf gl_buf_create(pl_gpu gpu, const struct pl_buf_params *params)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT())
        return NULL;

    struct pl_buf_t *buf = pl_zalloc_obj(NULL, buf, struct pl_buf_gl);
    buf->params = *params;
    buf->params.initial_data = NULL;

    struct pl_gl *p = PL_PRIV(gpu);
    struct pl_buf_gl *buf_gl = PL_PRIV(buf);
    buf_gl->id = ++p->buf_id;

    // Just use this since the generic GL_BUFFER doesn't work
    GLenum target = GL_ARRAY_BUFFER;
    const void *data = params->initial_data;
    size_t total_size = params->size;
    bool import = false;

    if (params->import_handle == PL_HANDLE_HOST_PTR) {
        const struct pl_shared_mem *shmem = &params->shared_mem;
        target = GL_EXTERNAL_VIRTUAL_MEMORY_BUFFER_AMD;

        data = shmem->handle.ptr;
        buf_gl->offset = shmem->offset;
        total_size = shmem->size;
        import = true;

        if (params->host_mapped)
            buf->data = (uint8_t *) data + buf_gl->offset;

        if (buf_gl->offset > 0 && params->drawable) {
            PL_ERR(gpu, "Cannot combine non-aligned host pointer imports with "
                   "drawable (vertex) buffers! This is a design limitation, "
                   "open an issue if you absolutely need this.");
            goto error;
        }
    }

    gl->GenBuffers(1, &buf_gl->buffer);
    gl->BindBuffer(target, buf_gl->buffer);

    if (gl_test_ext(gpu, "GL_ARB_buffer_storage", 44, 0) && !import) {

        GLbitfield mapflags = 0, storflags = 0;
        if (params->host_writable)
            storflags |= GL_DYNAMIC_STORAGE_BIT;
        if (params->host_mapped) {
            mapflags |= GL_MAP_READ_BIT | GL_MAP_WRITE_BIT |
                        GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
        }
        if (params->memory_type == PL_BUF_MEM_HOST)
            storflags |= GL_CLIENT_STORAGE_BIT; // hopefully this works

        gl->BufferStorage(target, total_size, data, storflags | mapflags);

        if (params->host_mapped) {
            buf_gl->mapped = true;
            buf->data = gl->MapBufferRange(target, buf_gl->offset, params->size,
                                           mapflags);
            if (!buf->data) {
                gl->BindBuffer(target, 0);
                if (!gl_check_err(gpu, "gl_buf_create: map"))
                    PL_ERR(gpu, "Failed mapping buffer: unknown reason");
                goto error;
            }
        }

    } else {

        // Make a random guess based on arbitrary criteria we can't know
        GLenum hint = GL_STREAM_DRAW;
        if (params->initial_data && !params->host_writable && !params->host_mapped)
            hint = GL_STATIC_DRAW;
        if (params->host_readable && !params->host_writable && !params->host_mapped)
            hint = GL_STREAM_READ;
        if (params->storable)
            hint = GL_DYNAMIC_COPY;

        gl->BufferData(target, total_size, data, hint);

        if (import && gl->GetError() == GL_INVALID_OPERATION) {
            PL_ERR(gpu, "Failed importing host pointer!");
            goto error;
        }

    }

    gl->BindBuffer(target, 0);
    if (!gl_check_err(gpu, "gl_buf_create"))
        goto error;

    if (params->storable) {
        buf_gl->barrier = GL_BUFFER_UPDATE_BARRIER_BIT | // for buf_copy etc.
                          GL_PIXEL_BUFFER_BARRIER_BIT | // for tex_upload
                          GL_SHADER_STORAGE_BARRIER_BIT;

        if (params->host_mapped)
            buf_gl->barrier |= GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT;
        if (params->uniform)
            buf_gl->barrier |= GL_UNIFORM_BARRIER_BIT;
        if (params->drawable)
            buf_gl->barrier |= GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT;
    }

    RELEASE_CURRENT();
    return buf;

error:
    gl_buf_destroy(gpu, buf);
    RELEASE_CURRENT();
    return NULL;
}

bool gl_buf_poll(pl_gpu gpu, pl_buf buf, uint64_t timeout)
{
    const gl_funcs *gl = gl_funcs_get(gpu);

    // Non-persistently mapped buffers are always implicitly reusable in OpenGL,
    // the implementation will create more buffers under the hood if needed.
    if (!buf->data)
        return false;

    if (!MAKE_CURRENT())
        return true; // conservative guess

    struct pl_buf_gl *buf_gl = PL_PRIV(buf);
    if (buf_gl->fence) {
        GLenum res = gl->ClientWaitSync(buf_gl->fence,
                                        timeout ? GL_SYNC_FLUSH_COMMANDS_BIT : 0,
                                        timeout);
        if (res == GL_ALREADY_SIGNALED || res == GL_CONDITION_SATISFIED) {
            gl->DeleteSync(buf_gl->fence);
            buf_gl->fence = NULL;
        }
    }

    gl_poll_callbacks(gpu);
    RELEASE_CURRENT();
    return !!buf_gl->fence;
}

void gl_buf_write(pl_gpu gpu, pl_buf buf, size_t offset,
                  const void *data, size_t size)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT())
        return;

    struct pl_buf_gl *buf_gl = PL_PRIV(buf);
    gl->BindBuffer(GL_ARRAY_BUFFER, buf_gl->buffer);
    gl->BufferSubData(GL_ARRAY_BUFFER, buf_gl->offset + offset, size, data);
    gl->BindBuffer(GL_ARRAY_BUFFER, 0);
    gl_check_err(gpu, "gl_buf_write");
    RELEASE_CURRENT();
}

bool gl_buf_read(pl_gpu gpu, pl_buf buf, size_t offset,
                 void *dest, size_t size)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT())
        return false;

    struct pl_buf_gl *buf_gl = PL_PRIV(buf);
    gl->BindBuffer(GL_ARRAY_BUFFER, buf_gl->buffer);
    gl->GetBufferSubData(GL_ARRAY_BUFFER, buf_gl->offset + offset, size, dest);
    gl->BindBuffer(GL_ARRAY_BUFFER, 0);
    bool ok = gl_check_err(gpu, "gl_buf_read");
    RELEASE_CURRENT();
    return ok;
}

void gl_buf_copy(pl_gpu gpu, pl_buf dst, size_t dst_offset,
                 pl_buf src, size_t src_offset, size_t size)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT())
        return;

    struct pl_buf_gl *src_gl = PL_PRIV(src);
    struct pl_buf_gl *dst_gl = PL_PRIV(dst);
    gl->BindBuffer(GL_COPY_READ_BUFFER, src_gl->buffer);
    gl->BindBuffer(GL_COPY_WRITE_BUFFER, dst_gl->buffer);
    gl->CopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                          src_gl->offset + src_offset,
                          dst_gl->offset + dst_offset, size);
    gl_check_err(gpu, "gl_buf_copy");
    RELEASE_CURRENT();
}

#define QUERY_OBJECT_NUM 8

struct pl_timer_t {
    GLuint query[QUERY_OBJECT_NUM];
    int index_write; // next index to write to
    int index_read; // next index to read from
};

static pl_timer gl_timer_create(pl_gpu gpu)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    struct pl_gl *p = PL_PRIV(gpu);
    if (!p->has_queries || !MAKE_CURRENT())
        return NULL;

    pl_timer timer = pl_zalloc_ptr(NULL, timer);
    gl->GenQueries(QUERY_OBJECT_NUM, timer->query);
    RELEASE_CURRENT();
    return timer;
}

static void gl_timer_destroy(pl_gpu gpu, pl_timer timer)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT()) {
        PL_ERR(gpu, "Failed uninitializing timer, leaking resources!");
        return;
    }

    gl->DeleteQueries(QUERY_OBJECT_NUM, timer->query);
    gl_check_err(gpu, "gl_timer_destroy");
    RELEASE_CURRENT();
    pl_free(timer);
}

static uint64_t gl_timer_query(pl_gpu gpu, pl_timer timer)
{
    if (timer->index_read == timer->index_write)
        return 0; // no more unprocessed results

    struct pl_gl *p = PL_PRIV(gpu);

    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT())
        return 0;

    uint64_t res = 0;
    GLuint query = timer->query[timer->index_read];
    GLuint avail = 0;
    gl->GetQueryObjectuiv(query, GL_QUERY_RESULT_AVAILABLE, &avail);
    if (!avail)
        goto done;
    if (p->gles_ver || p->gl_ver < 33) {
        GLuint tmp = 0;
        gl->GetQueryObjectuiv(query, GL_QUERY_RESULT, &tmp);
        res = tmp;
    } else {
        gl->GetQueryObjectui64v(query, GL_QUERY_RESULT, &res);
    }

    timer->index_read = (timer->index_read + 1) % QUERY_OBJECT_NUM;
    // fall through

done:
    RELEASE_CURRENT();
    return res;
}

void gl_timer_begin(pl_gpu gpu, pl_timer timer)
{
    if (!timer)
        return;

    const gl_funcs *gl = gl_funcs_get(gpu);
    gl->BeginQuery(GL_TIME_ELAPSED, timer->query[timer->index_write]);
}

void gl_timer_end(pl_gpu gpu, pl_timer timer)
{
    if (!timer)
        return;

    const gl_funcs *gl = gl_funcs_get(gpu);
    gl->EndQuery(GL_TIME_ELAPSED);

    timer->index_write = (timer->index_write + 1) % QUERY_OBJECT_NUM;
    if (timer->index_write == timer->index_read) {
        // forcibly drop the least recent result to make space
        timer->index_read = (timer->index_read + 1) % QUERY_OBJECT_NUM;
    }
}

static void gl_gpu_flush(pl_gpu gpu)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT())
        return;

    gl->Flush();
    gl_check_err(gpu, "gl_gpu_flush");
    RELEASE_CURRENT();
}

static void gl_gpu_finish(pl_gpu gpu)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT())
        return;

    gl->Finish();
    gl_check_err(gpu, "gl_gpu_finish");
    RELEASE_CURRENT();
}

static bool gl_gpu_is_failed(pl_gpu gpu)
{
    struct pl_gl *gl = PL_PRIV(gpu);
    return gl->failed;
}

static const struct pl_gpu_fns pl_fns_gl = {
    .destroy                = gl_gpu_destroy,
    .tex_create             = gl_tex_create,
    .tex_destroy            = gl_tex_destroy,
    .tex_invalidate         = gl_tex_invalidate,
    .tex_clear_ex           = gl_tex_clear_ex,
    .tex_blit               = gl_tex_blit,
    .tex_upload             = gl_tex_upload,
    .tex_download           = gl_tex_download,
    .buf_create             = gl_buf_create,
    .buf_destroy            = gl_buf_destroy,
    .buf_write              = gl_buf_write,
    .buf_read               = gl_buf_read,
    .buf_copy               = gl_buf_copy,
    .buf_poll               = gl_buf_poll,
    .desc_namespace         = gl_desc_namespace,
    .pass_create            = gl_pass_create,
    .pass_destroy           = gl_pass_destroy,
    .pass_run               = gl_pass_run,
    .timer_create           = gl_timer_create,
    .timer_destroy          = gl_timer_destroy,
    .timer_query            = gl_timer_query,
    .gpu_flush              = gl_gpu_flush,
    .gpu_finish             = gl_gpu_finish,
    .gpu_is_failed          = gl_gpu_is_failed,
};
