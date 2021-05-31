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
#include <errno.h>
#endif

#ifdef PL_HAVE_WIN32
#include <windows.h>
#include <sysinfoapi.h>
#endif

static const struct pl_gpu_fns pl_fns_gl;

static bool test_ext(pl_gpu gpu, const char *ext, int gl_ver, int gles_ver)
{
    struct pl_gl *p = PL_PRIV(gpu);
    if (gl_ver && p->gl_ver >= gl_ver)
        return true;
    if (gles_ver && p->gles_ver >= gles_ver)
        return true;

    return ext ? epoxy_has_gl_extension(ext) : false;
}

static inline bool make_current(pl_gpu gpu)
{
    struct pl_gl *p = PL_PRIV(gpu);
    if (!gl_make_current(p->gl)) {
        p->failed = true;
        return false;
    }

    return true;
}

static inline void release_current(pl_gpu gpu)
{
    struct pl_gl *p = PL_PRIV(gpu);
    gl_release_current(p->gl);
}

static void gl_destroy_gpu(pl_gpu gpu)
{
    struct pl_gl *p = PL_PRIV(gpu);

    pl_gpu_finish(gpu);
    while (p->callbacks.num > 0)
        gl_poll_callbacks(gpu);

    pl_free((void *) gpu);
}

#define get(pname, field)               \
    do {                                \
        GLint tmp = 0;                  \
        glGetIntegerv((pname), &tmp);   \
        *(field) = tmp;                 \
    } while (0)

#define geti(pname, i, field)               \
    do {                                    \
        GLint tmp = 0;                      \
        glGetIntegeri_v((pname), i, &tmp);  \
        *(field) = tmp;                     \
    } while (0)

static void add_format(pl_gpu pgpu, const struct gl_format *gl_fmt)
{
    struct pl_gpu *gpu = (struct pl_gpu *) pgpu;
    struct pl_gl *p = PL_PRIV(gpu);

    struct pl_fmt *fmt = pl_alloc_obj(gpu, fmt, gl_fmt);
    const struct gl_format **fmtp = PL_PRIV(fmt);
    *fmt = gl_fmt->tmpl;
    *fmtp = gl_fmt;

    // Calculate the host size and number of components
    switch (gl_fmt->fmt) {
    case GL_RED:
    case GL_RED_INTEGER:
        fmt->num_components = 1;
        break;
    case GL_RG:
    case GL_RG_INTEGER:
        fmt->num_components = 2;
        break;
    case GL_RGB:
    case GL_RGB_INTEGER:
        fmt->num_components = 3;
        break;
    case GL_RGBA:
    case GL_RGBA_INTEGER:
        fmt->num_components = 4;
        break;
    default:
        pl_unreachable();
    }

    int size;
    switch (gl_fmt->type) {
    case GL_BYTE:
    case GL_UNSIGNED_BYTE:
        size = 1;
        break;
    case GL_SHORT:
    case GL_UNSIGNED_SHORT:
        size = 2;
        break;
    case GL_INT:
    case GL_UNSIGNED_INT:
    case GL_FLOAT:
        size = 4;
        break;
    default:
        pl_unreachable();
    }

    // Host visible representation
    fmt->texel_size = fmt->num_components * size;
    for (int i = 0; i < fmt->num_components; i++)
        fmt->host_bits[i] = size * 8;

    // Compute internal size by summing up the depth
    int ibits = 0;
    for (int i = 0; i < fmt->num_components; i++)
        ibits += fmt->component_depth[i];
    fmt->internal_size = (ibits + 7) / 8;

    // We're not the ones actually emulating these texture format - the
    // driver is - but we might as well set the hint.
    fmt->emulated = fmt->texel_size != fmt->internal_size;

    // 3-component formats are almost surely also emulated
    if (fmt->num_components == 3)
        fmt->emulated = true;

    // Older OpenGL most likely emulates 32-bit float formats as well
    if (p->gl_ver < 30 && fmt->component_depth[0] >= 32)
        fmt->emulated = true;

    // For sanity, clear the superfluous fields
    for (int i = fmt->num_components; i < 4; i++) {
        fmt->component_depth[i] = 0;
        fmt->sample_order[i] = 0;
        fmt->host_bits[i] = 0;
    }

    fmt->glsl_type = pl_var_glsl_type_name(pl_var_from_fmt(fmt, ""));
    fmt->glsl_format = pl_fmt_glsl_format(fmt, fmt->num_components);
    fmt->fourcc = pl_fmt_fourcc(fmt);
    pl_assert(fmt->glsl_type);

#ifdef PL_HAVE_UNIX
    if (p->has_modifiers) {
        int num_mods = 0;
        bool ok = eglQueryDmaBufModifiersEXT(p->egl_dpy, fmt->fourcc,
                                             0, NULL, NULL, &num_mods);
        if (ok && num_mods) {
            // On my system eglQueryDmaBufModifiersEXT seems to never return
            // MOD_INVALID even though eglExportDMABUFImageQueryMESA happily
            // returns such modifiers. Since we handle INVALID by not
            // requiring modifiers at all, always add this value to the
            // list of supported modifiers. May result in duplicates, but
            // whatever.
            uint64_t *mods = pl_calloc(fmt, num_mods + 1, sizeof(uint64_t));
            mods[0] = DRM_FORMAT_MOD_INVALID;
            ok = eglQueryDmaBufModifiersEXT(p->egl_dpy, fmt->fourcc, num_mods,
                                            &mods[1], NULL, &num_mods);

            if (ok) {
                fmt->modifiers = mods;
                fmt->num_modifiers = num_mods + 1;
            }
        }

        eglGetError(); // ignore probing errors
    }

    if (!fmt->num_modifiers) {
        // Hacky fallback for older drivers that don't support properly
        // querying modifiers
        static const uint64_t static_mods[] = {
            DRM_FORMAT_MOD_INVALID,
            DRM_FORMAT_MOD_LINEAR,
        };

        fmt->num_modifiers = PL_ARRAY_SIZE(static_mods);
        fmt->modifiers = static_mods;
    }
#endif

    // Gathering requires checking the format type (and extension presence)
    if (fmt->caps & PL_FMT_CAP_SAMPLEABLE)
        fmt->gatherable = p->gather_comps >= fmt->num_components;

    // Mask renderable/blittable if no FBOs available
    if (!p->has_fbos)
        fmt->caps &= ~(PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLITTABLE);

    // Reading from textures on GLES requires FBO support for this fmt
    if (p->gl_ver || (fmt->caps & PL_FMT_CAP_RENDERABLE))
        fmt->caps |= PL_FMT_CAP_HOST_READABLE;

    if (gpu->glsl.compute && fmt->glsl_format && p->has_storage)
        fmt->caps |= PL_FMT_CAP_STORABLE;

    // Only float-type formats are considered blendable in OpenGL
    switch (fmt->type) {
    case PL_FMT_UNKNOWN:
    case PL_FMT_UINT:
    case PL_FMT_SINT:
        break;
    case PL_FMT_FLOAT:
    case PL_FMT_UNORM:
    case PL_FMT_SNORM:
        if (fmt->caps & PL_FMT_CAP_RENDERABLE)
            fmt->caps |= PL_FMT_CAP_BLENDABLE;
        break;
    case PL_FMT_TYPE_COUNT:
        pl_unreachable();
    }

    // TODO: Texel buffers

    PL_ARRAY_APPEND_RAW(gpu, gpu->formats, gpu->num_formats, fmt);
}

static bool gl_setup_formats(struct pl_gpu *gpu)
{
    pl_gl_enumerate_formats(gpu, add_format);
    return gl_check_err(gpu, "gl_setup_formats");
}

#ifdef EPOXY_HAS_EGL

static pl_handle_caps tex_handle_caps(pl_gpu gpu, bool import)
{
    pl_handle_caps caps = 0;
    struct pl_gl *p = PL_PRIV(gpu);

    if (!p->egl_dpy)
        return 0;

    if (import) {
        if (epoxy_has_egl_extension(p->egl_dpy, "EXT_image_dma_buf_import"))
            caps |= PL_HANDLE_DMA_BUF;
    } else if (!import && p->egl_ctx) {
        if (epoxy_has_egl_extension(p->egl_dpy, "EGL_MESA_image_dma_buf_export"))
            caps |= PL_HANDLE_DMA_BUF;
    }

    return caps;
}

#endif // EPOXY_HAS_EGL

static inline size_t get_page_size()
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

pl_gpu pl_gpu_create_gl(pl_log log, pl_opengl gl, const struct pl_opengl_params *params)
{
    struct pl_gpu *gpu = pl_zalloc_obj(NULL, gpu, struct pl_gl);
    gpu->log = log;
    gpu->ctx = gpu->log;

    struct pl_gl *p = PL_PRIV(gpu);
    p->impl = pl_fns_gl;
    p->gl = gl;

    struct pl_glsl_version *glsl = &gpu->glsl;
    int ver = epoxy_gl_version();
    glsl->gles = !epoxy_is_desktop_gl();
    p->gl_ver = glsl->gles ? 0 : ver;
    p->gles_ver = glsl->gles ? ver : 0;

    // If possible, query the GLSL version from the implementation
    const char *glslver = glGetString(GL_SHADING_LANGUAGE_VERSION);
    if (glslver) {
        PL_INFO(gpu, "    GL_SHADING_LANGUAGE_VERSION: %s", glslver);
        int major = 0, minor = 0;
        if (sscanf(glslver, "%d.%d", &major, &minor) == 2)
            glsl->version = major * 100 + minor;
    }

    if (!glsl->version) {
        // Otherwise, use the fixed magic versions 200 and 300 for early GLES,
        // and otherwise fall back to 110 if all else fails.
        if (p->gles_ver >= 30) {
            glsl->version = 300;
        } else if (p->gles_ver >= 20) {
            glsl->version = 200;
        } else {
            glsl->version = 110;
        }
    }

    if (test_ext(gpu, "GL_ARB_compute_shader", 43, 0)) {
        glsl->compute = true;
        get(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &glsl->max_shmem_size);
        get(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &glsl->max_group_threads);
        for (int i = 0; i < 3; i++)
            geti(GL_MAX_COMPUTE_WORK_GROUP_SIZE, i, &glsl->max_group_size[i]);
    }

    if (test_ext(gpu, "GL_ARB_texture_gather", 40, 0)) {
        get(GL_MAX_PROGRAM_TEXTURE_GATHER_COMPONENTS_ARB, &p->gather_comps);
        get(GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET_ARB, &glsl->min_gather_offset);
        get(GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET_ARB, &glsl->max_gather_offset);
    }

    // Query all device limits
    struct pl_gpu_limits *limits = &gpu->limits;
    limits->thread_safe = params->make_current;
    limits->callbacks = test_ext(gpu, "GL_ARB_sync", 32, 30);
    if (test_ext(gpu, "GL_ARB_pixel_buffer_object", 31, 0))
        limits->max_buf_size = SIZE_MAX; // no restriction imposed by GL
    if (test_ext(gpu, "GL_ARB_uniform_buffer_object", 31, 0))
        get(GL_MAX_UNIFORM_BLOCK_SIZE, &limits->max_ubo_size);
    if (test_ext(gpu, "GL_ARB_shader_storage_buffer_object", 43, 0))
        get(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &limits->max_ssbo_size);
    limits->max_vbo_size = limits->max_buf_size; // No additional restrictions
    if (test_ext(gpu, "GL_ARB_buffer_storage", 44, 0))
        limits->max_mapped_size = limits->max_buf_size;

    get(GL_MAX_TEXTURE_SIZE, &limits->max_tex_2d_dim);
    if (test_ext(gpu, "GL_EXT_texture3D", 21, 30))
        get(GL_MAX_3D_TEXTURE_SIZE, &limits->max_tex_3d_dim);
    // There's no equivalent limit for 1D textures for whatever reason, so
    // just set it to the same as the 2D limit
    if (p->gl_ver >= 21)
        limits->max_tex_1d_dim = limits->max_tex_2d_dim;
    limits->buf_transfer = true;
    get(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS, &limits->max_variables);
    if (glsl->compute) {
        for (int i = 0; i < 3; i++)
            geti(GL_MAX_COMPUTE_WORK_GROUP_COUNT, i, &limits->max_dispatch[i]);
    }

    // Query import/export support
#ifdef EPOXY_HAS_EGL
    p->egl_dpy = params->egl_display;
    p->egl_ctx = params->egl_context;
    gpu->export_caps.tex = tex_handle_caps(gpu, false);
    gpu->import_caps.tex = tex_handle_caps(gpu, true);

    if (p->egl_dpy) {
        p->has_modifiers = epoxy_has_egl_extension(p->egl_dpy,
                                        "EXT_image_dma_buf_import_modifiers");
    }
#endif

    if (epoxy_has_gl_extension("GL_AMD_pinned_memory")) {
        gpu->import_caps.buf |= PL_HANDLE_HOST_PTR;
        gpu->limits.align_host_ptr = get_page_size();
    }

    // Cache some internal capability checks
    p->has_stride = test_ext(gpu, "GL_EXT_unpack_subimage", 11, 30);
    p->has_vao = test_ext(gpu, "GL_ARB_vertex_array_object", 30, 0);
    p->has_invalidate_fb = test_ext(gpu, "GL_ARB_invalidate_subdata", 43, 30);
    p->has_invalidate_tex = test_ext(gpu, "GL_ARB_invalidate_subdata", 43, 0);
    p->has_queries = test_ext(gpu, "GL_ARB_timer_query", 33, 0);
    p->has_fbos = test_ext(gpu, "GL_ARB_framebuffer_object", 30, 20);
    p->has_storage = test_ext(gpu, "GL_ARB_shader_image_load_store", 42, 0);

    // We simply don't know, so make up some values
    limits->align_tex_xfer_offset = 32;
    limits->align_tex_xfer_stride = 1;
    limits->fragment_queues = 1;
    limits->compute_queues = 1;
    if (test_ext(gpu, "GL_EXT_unpack_subimage", 11, 30))
        limits->align_tex_xfer_stride = 4;

    if (!gl_check_err(gpu, "pl_gpu_create_gl"))
        goto error;

    // Filter out error messages during format probing
    pl_log_level_cap(gpu->log, PL_LOG_INFO);
    bool formats_ok = gl_setup_formats(gpu);
    pl_log_level_cap(gpu->log, PL_LOG_NONE);
    if (!formats_ok)
        goto error;

    return pl_gpu_finalize(gpu);

error:
    gl_destroy_gpu(gpu);
    return NULL;
}

// For pl_tex.priv
struct pl_tex_gl {
    GLenum target;
    GLuint texture;
    bool wrapped_tex;
    GLuint fbo; // or 0
    bool wrapped_fb;
    GLbitfield barrier;

    // GL format fields
    GLenum format;
    GLint iformat;
    GLenum type;

    // For imported/exported textures
#ifdef EPOXY_HAS_EGL
    EGLImageKHR image;
#endif
    int fd;
};

static void gl_tex_destroy(pl_gpu gpu, pl_tex tex)
{
    if (!make_current(gpu)) {
        PL_ERR(gpu, "Failed uninitializing texture, leaking resources!");
        return;
    }

    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    if (tex_gl->fbo && !tex_gl->wrapped_fb)
        glDeleteFramebuffers(1, &tex_gl->fbo);
#ifdef EPOXY_HAS_EGL
    if (tex_gl->image) {
        struct pl_gl *p = PL_PRIV(gpu);
        eglDestroyImageKHR(p->egl_dpy, tex_gl->image);
    }
#endif
    if (!tex_gl->wrapped_tex)
        glDeleteTextures(1, &tex_gl->texture);

#ifdef PL_HAVE_UNIX
    if (tex_gl->fd != -1)
        close(tex_gl->fd);
#endif

    gl_check_err(gpu, "gl_tex_destroy");
    release_current(gpu);
    pl_free((void *) tex);
}

static GLbitfield tex_barrier(pl_tex tex)
{
    GLbitfield barrier = 0;
    const struct pl_tex_params *params = &tex->params;

    if (params->sampleable)
        barrier |= GL_TEXTURE_FETCH_BARRIER_BIT;
    if (params->renderable || params->blit_src || params->blit_dst)
        barrier |= GL_FRAMEBUFFER_BARRIER_BIT;
    if (params->storable)
        barrier |= GL_SHADER_IMAGE_ACCESS_BARRIER_BIT;
    if (params->host_writable || params->host_readable)
        barrier |= GL_TEXTURE_UPDATE_BARRIER_BIT;

    return barrier;
}

#ifdef EPOXY_HAS_EGL

#define ADD_ATTRIB(name, value)                                     \
    do {                                                            \
        assert(num_attribs + 3 < PL_ARRAY_SIZE(attribs));           \
        attribs[num_attribs++] = (name);                            \
        attribs[num_attribs++] = (value);                           \
    } while (0)

#define ADD_DMABUF_PLANE_ATTRIBS(plane, fd, offset, stride)         \
    do {                                                            \
        ADD_ATTRIB(EGL_DMA_BUF_PLANE ## plane ## _FD_EXT,           \
                   fd);                                             \
        ADD_ATTRIB(EGL_DMA_BUF_PLANE ## plane ## _OFFSET_EXT,       \
                   offset);                                         \
        ADD_ATTRIB(EGL_DMA_BUF_PLANE ## plane ## _PITCH_EXT,        \
                   stride);                                         \
    } while (0)

#define ADD_DMABUF_PLANE_MODIFIERS(plane, mod)                      \
    do {                                                            \
        ADD_ATTRIB(EGL_DMA_BUF_PLANE ## plane ## _MODIFIER_LO_EXT,  \
                   (uint32_t) ((mod) & 0xFFFFFFFFlu));              \
        ADD_ATTRIB(EGL_DMA_BUF_PLANE ## plane ## _MODIFIER_HI_EXT,  \
                   (uint32_t) (((mod) >> 32u) & 0xFFFFFFFFlu));     \
    } while (0)

static bool gl_tex_import(pl_gpu gpu,
                          enum pl_handle_type handle_type,
                          const struct pl_shared_mem *shared_mem,
                          struct pl_tex *tex)
{
    if (!make_current(gpu))
        return false;

    struct pl_gl *p = PL_PRIV(gpu);
    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    const struct pl_tex_params *params = &tex->params;

    int attribs[20] = {};
    int num_attribs = 0;
    ADD_ATTRIB(EGL_WIDTH,  params->w);
    ADD_ATTRIB(EGL_HEIGHT, params->h);

    switch (handle_type) {

#ifdef PL_HAVE_UNIX
    case PL_HANDLE_DMA_BUF:
        if (shared_mem->handle.fd == -1) {
            PL_ERR(gpu, "%s: invalid fd", __func__);
            goto error;
        }

        tex_gl->fd = dup(shared_mem->handle.fd);
        if (tex_gl->fd == -1) {
            PL_ERR(gpu, "%s: cannot duplicate fd for importing", __func__);
            goto error;
        }

        if (shared_mem->drm_format_mod != DRM_FORMAT_MOD_INVALID)
            ADD_DMABUF_PLANE_MODIFIERS(0, shared_mem->drm_format_mod);

        ADD_ATTRIB(EGL_LINUX_DRM_FOURCC_EXT, params->format->fourcc);
        ADD_DMABUF_PLANE_ATTRIBS(0, tex_gl->fd, shared_mem->offset,
                                 PL_DEF(shared_mem->stride_w, params->w));
        attribs[num_attribs] = EGL_NONE;

        // EGL_LINUX_DMA_BUF_EXT requires EGL_NO_CONTEXT
        tex_gl->image = eglCreateImageKHR(p->egl_dpy,
                                          EGL_NO_CONTEXT,
                                          EGL_LINUX_DMA_BUF_EXT,
                                          (EGLClientBuffer) NULL,
                                          attribs);

        break;
#endif // PL_HAVE_UNIX

    case PL_HANDLE_WIN32:
    case PL_HANDLE_WIN32_KMT:
    case PL_HANDLE_HOST_PTR:
    case PL_HANDLE_FD:
        pl_unreachable();

    }

    if (!egl_check_err(gpu, "eglCreateImageKHR") || !tex_gl->image)
        goto error;

    // tex_gl->image should be already bound
    glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, tex_gl->image);
    if (!egl_check_err(gpu, "EGLImageTargetTexture2DOES"))
        goto error;

    release_current(gpu);
    return true;

error:
    PL_ERR(gpu, "Failed importing GL texture!");
    release_current(gpu);
    return false;
}

static EGLenum egl_from_gl_target(pl_gpu gpu, int target)
{
    switch(target) {
    case GL_TEXTURE_2D: return EGL_GL_TEXTURE_2D;
    case GL_TEXTURE_3D: return EGL_GL_TEXTURE_3D;
    default:
        PL_ERR(gpu, "%s: unsupported texture target 0x%x", __func__, target);
        return 0;
    }
}

static bool gl_tex_export(pl_gpu gpu, enum pl_handle_type handle_type,
                          bool preserved, struct pl_tex *tex)
{
    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    struct pl_gl *p = PL_PRIV(gpu);
    struct pl_shared_mem *shared_mem = &tex->shared_mem;
    bool ok;

    EGLenum egltarget = egl_from_gl_target(gpu, tex_gl->target);
    if (!egltarget)
        goto error;

    int attribs[] = {
        EGL_IMAGE_PRESERVED, preserved,
        EGL_NONE,
    };

    // We assume that tex_gl->texture is already bound
    tex_gl->image = eglCreateImageKHR(p->egl_dpy,
                                      p->egl_ctx,
                                      egltarget,
                                      (EGLClientBuffer) (uintptr_t) tex_gl->texture,
                                      attribs);
    if (!egl_check_err(gpu, "eglCreateImageKHR") || !tex_gl->image)
        goto error;

    switch (handle_type) {

#ifdef PL_HAVE_UNIX
    case PL_HANDLE_DMA_BUF: {
        int fourcc = 0;
        int num_planes = 0;
        EGLuint64KHR modifier = 0;
        ok = eglExportDMABUFImageQueryMESA(p->egl_dpy,
                                           tex_gl->image,
                                           &fourcc,
                                           &num_planes,
                                           &modifier);
        if (!egl_check_err(gpu, "eglExportDMABUFImageQueryMESA") || !ok)
            goto error;

        if (fourcc != tex->params.format->fourcc) {
            PL_ERR(gpu, "Exported DRM format %s does not match fourcc of "
                   "specified pl_fmt %s? Please open a bug.",
                   PRINT_FOURCC(fourcc), PRINT_FOURCC(tex->params.format->fourcc));
            goto error;
        }

        if (num_planes != 1) {
            PL_ERR(gpu, "Unsupported number of planes: %d", num_planes);
            goto error;
        }

        int offset = 0, stride = 0;
        ok = eglExportDMABUFImageMESA(p->egl_dpy,
                                      tex_gl->image,
                                      &tex_gl->fd,
                                      &stride,
                                      &offset);
        if (!egl_check_err(gpu, "eglExportDMABUFImageMesa") || !ok)
            goto error;

        off_t fdsize = lseek(tex_gl->fd, 0, SEEK_END);
        off_t err = fdsize > 0 && lseek(tex_gl->fd, 0, SEEK_SET);
        if (fdsize <= 0 || err < 0) {
            PL_ERR(gpu, "Failed querying FD size: %s", strerror(errno));
            goto error;
        }

        *shared_mem = (struct pl_shared_mem) {
            .handle.fd = tex_gl->fd,
            .size = fdsize,
            .offset = offset,
            .drm_format_mod = modifier,
            .stride_w = stride,
        };
        break;
    }
#endif // PL_HAVE_UNIX

    case PL_HANDLE_WIN32:
    case PL_HANDLE_WIN32_KMT:
    case PL_HANDLE_HOST_PTR:
    case PL_HANDLE_FD:
        pl_unreachable();

    }

    return true;

error:
    PL_ERR(gpu, "Failed exporting GL texture!");
    return false;
}

#else // !EPOXY_HAS_EGL

static bool gl_tex_import(pl_gpu gpu, enum pl_handle_type handle_type,
                          const struct pl_shared_mem *shared_mem,
                          struct pl_tex *tex)
{
    abort(); // no implementations
}

static bool gl_tex_export(pl_gpu gpu, enum pl_handle_type handle_type,
                          bool preserved, struct pl_tex *tex)
{
    abort(); // no implementations
}

#endif // EPOXY_HAS_EGL

static const char *fb_err_str(GLenum err)
{
    switch (err) {
#define CASE(name) case name: return #name
    CASE(GL_FRAMEBUFFER_COMPLETE);
    CASE(GL_FRAMEBUFFER_UNDEFINED);
    CASE(GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);
    CASE(GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT);
    CASE(GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS);
    CASE(GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER);
    CASE(GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER);
    CASE(GL_FRAMEBUFFER_UNSUPPORTED);
    CASE(GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE);
    CASE(GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS);
#undef CASE

    default: return "unknown error";
    }
}

static pl_tex gl_tex_create(pl_gpu gpu, const struct pl_tex_params *params)
{
    if (!make_current(gpu))
        return NULL;

    struct pl_gl *p = PL_PRIV(gpu);
    struct pl_tex *tex = pl_zalloc_obj(NULL, tex, struct pl_tex_gl);
    tex->params = *params;
    tex->params.initial_data = NULL;
    tex->sampler_type = PL_SAMPLER_NORMAL;

    struct pl_tex_gl *tex_gl = PL_PRIV(tex);

    const struct gl_format **fmtp = PL_PRIV(params->format);
    const struct gl_format *fmt = *fmtp;
    *tex_gl = (struct pl_tex_gl) {
        .format = fmt->fmt,
        .iformat = fmt->ifmt,
        .type = fmt->type,
        .barrier = tex_barrier(tex),
        .fd = -1,
    };

    static const GLint targets[] = {
        [1] = GL_TEXTURE_1D,
        [2] = GL_TEXTURE_2D,
        [3] = GL_TEXTURE_3D,
    };

    int dims = pl_tex_params_dimension(*params);
    pl_assert(dims >= 1 && dims <= 3);
    tex_gl->target = targets[dims];

    glGenTextures(1, &tex_gl->texture);
    glBindTexture(tex_gl->target, tex_gl->texture);

    if (params->import_handle) {
        if (!gl_tex_import(gpu, params->import_handle, &params->shared_mem, tex))
            goto error;
    } else {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        switch (dims) {
        case 1:
            glTexImage1D(tex_gl->target, 0, tex_gl->iformat, params->w, 0,
                         tex_gl->format, tex_gl->type, params->initial_data);
            break;
        case 2:
            glTexImage2D(tex_gl->target, 0, tex_gl->iformat, params->w, params->h,
                         0, tex_gl->format, tex_gl->type, params->initial_data);
            break;
        case 3:
            glTexImage3D(tex_gl->target, 0, tex_gl->iformat, params->w, params->h,
                         params->d, 0, tex_gl->format, tex_gl->type,
                         params->initial_data);
            break;
        }

        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    }

    if (params->export_handle) {
        if (!gl_tex_export(gpu, params->export_handle, params->initial_data, tex))
            goto error;
    }

    glBindTexture(tex_gl->target, 0);

    if (!gl_check_err(gpu, "gl_tex_create: texture"))
        goto error;

    bool need_fbo = tex->params.renderable;
    if (tex->params.blit_src || tex->params.blit_dst) {
        if (dims != 2) {
            PL_ERR(gpu, "Blittable textures may only be 2D!");
            goto error;
        }

        need_fbo = true;
    }

    bool can_fbo = tex->params.format->caps & PL_FMT_CAP_RENDERABLE &&
                   tex->params.d == 0;

    // Try creating an FBO for host-readable textures, since this allows
    // reading back with glReadPixels instead of glGetTexImage. (Additionally,
    // GLES does not support glGetTexImage)
    if (tex->params.host_readable && (can_fbo || p->gles_ver))
        need_fbo = true;

    if (need_fbo) {
        if (!can_fbo) {
            PL_ERR(gpu, "Trying to create a renderable/blittable/readable "
                   "texture with an incompatible (non-renderable) format!");
            goto error;
        }

        glGenFramebuffers(1, &tex_gl->fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, tex_gl->fbo);
        switch (dims) {
        case 1:
            glFramebufferTexture1D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_1D, tex_gl->texture, 0);
            break;
        case 2:
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, tex_gl->texture, 0);
            break;
        case 3: pl_unreachable();
        }

        GLenum err = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
        if (err != GL_FRAMEBUFFER_COMPLETE) {
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            PL_ERR(gpu, "Failed creating framebuffer: %s", fb_err_str(err));
            goto error;
        }

        if (params->host_readable && p->gles_ver) {
            GLint read_type = 0, read_fmt = 0;
            glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &read_type);
            glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &read_fmt);
            if (read_type != tex_gl->type || read_fmt != tex_gl->format) {
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
                PL_ERR(gpu, "Trying to create host_readable texture whose "
                       "implementation-defined pixel read format "
                       "(type=0x%X, fmt=0x%X) does not match the texture's "
                       "internal format (type=0x%X, fmt=0x%X)! This is a "
                       "GLES/driver limitation, there's little we can do "
                       "about it.",
                       read_type, read_fmt, tex_gl->type, tex_gl->format);
                goto error;
            }
        }

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        if (!gl_check_err(gpu, "gl_tex_create: fbo"))
            goto error;
    }

    release_current(gpu);
    return tex;

error:
    gl_tex_destroy(gpu, tex);
    release_current(gpu);
    return NULL;
}

static bool gl_fb_query(pl_gpu gpu, int fbo, struct pl_fmt *fmt,
                        struct gl_format *glfmt)
{
    struct pl_gl *p = PL_PRIV(gpu);
    *fmt = (struct pl_fmt) {
        .name = "fbo",
        .type = PL_FMT_UNKNOWN,
        .caps = PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLITTABLE | PL_FMT_CAP_BLENDABLE,
        .num_components = 4,
        .component_depth = {8, 8, 8, 8}, // default to rgba8
        .sample_order = {0, 1, 2, 3},
    };

    *glfmt = (struct gl_format) {
        .fmt = GL_RGBA,
    };

    bool can_query = !test_ext(gpu, "GL_ARB_framebuffer_object", 30, 20);
    if (!fbo && p->gles_ver && p->gles_ver < 30)
        can_query = false; // can't query default framebuffer on GLES 2.0

    if (can_query) {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);

        GLenum obj = p->gles_ver ? GL_BACK : GL_BACK_LEFT;
        if (fbo != 0)
            obj = GL_COLOR_ATTACHMENT0;

        GLint type = 0;
        glGetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, obj,
                GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE, &type);
        switch (type) {
        case GL_FLOAT:                  fmt->type = PL_FMT_FLOAT; break;
        case GL_INT:                    fmt->type = PL_FMT_SINT; break;
        case GL_UNSIGNED_INT:           fmt->type = PL_FMT_UINT; break;
        case GL_SIGNED_NORMALIZED:      fmt->type = PL_FMT_SNORM; break;
        case GL_UNSIGNED_NORMALIZED:    fmt->type = PL_FMT_UNORM; break;
        default:                        fmt->type = PL_FMT_UNKNOWN; break;
        }

        glGetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, obj,
                GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE, &fmt->component_depth[0]);
        glGetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, obj,
                GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE, &fmt->component_depth[1]);
        glGetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, obj,
                GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE, &fmt->component_depth[2]);
        glGetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, obj,
                GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE, &fmt->component_depth[3]);

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        gl_check_err(gpu, "gl_fb_query");
    }

    int gpu_bits = 0;
    for (int i = 0; i < 4; i++)
        gpu_bits += fmt->component_depth[i];
    fmt->internal_size = (gpu_bits + 7) / 8;

    size_t host_size = 0;
    switch (fmt->type) {
    case PL_FMT_UNKNOWN:
        fmt->opaque = true;
        return true;
    case PL_FMT_FLOAT:
        glfmt->type = GL_FLOAT;
        host_size = sizeof(float);
        break;
    case PL_FMT_UNORM:
    case PL_FMT_UINT:
        if (gpu_bits > 32) {
            glfmt->type = GL_UNSIGNED_SHORT;
            host_size = sizeof(uint16_t);
        } else {
            glfmt->type = GL_UNSIGNED_BYTE;
            host_size = sizeof(uint8_t);
        }
        break;
    case PL_FMT_SNORM:
    case PL_FMT_SINT:
        if (gpu_bits > 32) {
            glfmt->type = GL_SHORT;
            host_size = sizeof(int16_t);
        } else {
            glfmt->type = GL_BYTE;
            host_size = sizeof(int8_t);
        }
        break;
    case PL_FMT_TYPE_COUNT:
        pl_unreachable();
    }

    fmt->texel_size = fmt->num_components * host_size;
    for (int i = 0; i < fmt->num_components; i++)
        fmt->host_bits[i] = 8 * host_size;
    fmt->caps |= PL_FMT_CAP_HOST_READABLE;

    return true;
}

pl_tex pl_opengl_wrap(pl_gpu gpu, const struct pl_opengl_wrap_params *params)
{
    if (!make_current(gpu))
        return NULL;

    struct pl_gl *p = PL_PRIV(gpu);
    struct pl_tex *tex = pl_alloc_obj(NULL, tex, struct pl_tex_gl);
    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    *tex = (struct pl_tex) {
        .params = {
            .w = params->width,
            .h = params->height,
            .d = params->depth,
        },
    };

    pl_fmt fmt = NULL;
    const struct gl_format *glfmt = NULL;

    if (params->texture) {
        // Wrapping texture: Require matching iformat
        pl_assert(params->iformat);
        for (int i = 0; i < gpu->num_formats; i++) {
            const struct gl_format **glfmtp = PL_PRIV(gpu->formats[i]);
            if ((*glfmtp)->ifmt == params->iformat) {
                fmt = gpu->formats[i];
                glfmt = *glfmtp;
                break;
            }
        }

        if (!fmt) {
            PL_ERR(gpu, "Failed mapping iformat %d to any equivalent `pl_fmt`",
                   params->iformat);
            goto error;
        }
    } else {
        // Wrapping framebuffer: Allocate/infer generic FBO format
        struct pl_fmt new_fmt;
        struct gl_format new_glfmt;
        memset(&new_fmt, 0, sizeof(new_fmt)); // to enable memcmp
        memset(&new_glfmt, 0, sizeof(new_glfmt));
        if (!gl_fb_query(gpu, params->framebuffer, &new_fmt, &new_glfmt)) {
            PL_ERR(gpu, "Failed querying framebuffer specifics!");
            goto error;
        }

        // Look up this fmt/glfmt in the existing list of FBO formats
        for (int i = 0; i < p->fbo_formats.num; i++) {
            const struct fbo_format *fbofmt = &p->fbo_formats.elem[i];
            if (memcmp(&new_fmt, fbofmt->fmt, sizeof(new_fmt)) == 0 &&
                memcmp(&new_glfmt, fbofmt->glfmt, sizeof(new_glfmt)) == 0)
            {
                fmt = fbofmt->fmt;
                glfmt = fbofmt->glfmt;
                break;
            }
        }

        if (!fmt) {
            fmt = pl_alloc_obj((void *) gpu, fmt, const struct gl_format *);
            memcpy((struct pl_fmt *) fmt, &new_fmt, sizeof(new_fmt));
            glfmt = pl_memdup((void *) gpu, &new_glfmt, sizeof(new_glfmt));
            const struct gl_format **glfmtp = PL_PRIV(fmt);
            *glfmtp = glfmt;

            PL_ARRAY_APPEND(gpu, p->fbo_formats, (struct fbo_format) {
                .fmt = fmt,
                .glfmt = glfmt,
            });
        }
    }

    *tex_gl = (struct pl_tex_gl) {
        .target = params->target,
        .texture = params->texture,
        .fbo = params->framebuffer,
        .wrapped_tex = !!params->texture,
        .wrapped_fb = params->framebuffer || !params->texture,
        .iformat = glfmt->ifmt,
        .format = glfmt->fmt,
        .type = glfmt->type,
    };

    int dims = pl_tex_params_dimension(tex->params);
    if (!tex_gl->target) {
        switch (dims) {
        case 1: tex_gl->target = GL_TEXTURE_1D; break;
        case 2: tex_gl->target = GL_TEXTURE_2D; break;
        case 3: tex_gl->target = GL_TEXTURE_3D; break;
        }
    }

    // Map texture-specific sampling metadata
    if (params->texture) {
        switch (params->target) {
        case GL_TEXTURE_1D:
            if (params->width || params->depth) {
                PL_ERR(gpu, "Invalid texture dimensions for GL_TEXTURE_1D");
                goto error;
            }
            // fall through
        case GL_TEXTURE_2D:
            if (params->depth) {
                PL_ERR(gpu, "Invalid texture dimensions for GL_TEXTURE_2D");
                goto error;
            }
            // fall through
        case 0:
        case GL_TEXTURE_3D:
            tex->sampler_type = PL_SAMPLER_NORMAL;
            break;

        case GL_TEXTURE_RECTANGLE: tex->sampler_type = PL_SAMPLER_RECT; break;
        case GL_TEXTURE_EXTERNAL_OES: tex->sampler_type = PL_SAMPLER_EXTERNAL; break;

        default:
            PL_ERR(gpu, "Failed mapping texture target %u to any equivalent "
                   "`pl_sampler_type`", params->target);
            goto error;
        }
    }

    // Create optional extra fbo if needed/possible
    bool can_fbo = tex_gl->texture &&
                   (fmt->caps & PL_FMT_CAP_RENDERABLE) &&
                   tex->sampler_type != PL_SAMPLER_EXTERNAL &&
                   dims < 3;

    if (can_fbo && !tex_gl->fbo) {
        glGenFramebuffers(1, &tex_gl->fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, tex_gl->fbo);
        switch (dims) {
        case 1:
            glFramebufferTexture1D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   tex_gl->target, tex_gl->texture, 0);
            break;
        case 2:
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   tex_gl->target, tex_gl->texture, 0);
            break;
        }

        GLenum err = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
        if (err != GL_FRAMEBUFFER_COMPLETE) {
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            PL_ERR(gpu, "Failed creating framebuffer: error code %d", err);
            goto error;
        }

        if (p->gles_ver) {
            GLint read_type = 0, read_fmt = 0;
            glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &read_type);
            glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &read_fmt);
            tex->params.host_readable = read_type == tex_gl->type &&
                                        read_fmt == tex_gl->format;
        } else {
            tex->params.host_readable = true;
        }

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        if (!gl_check_err(gpu, "pl_opengl_wrap: fbo"))
            goto error;
    }

    // Complete the process of inferring the texture capabilities
    tex->params.format = fmt;
    if (tex_gl->texture) {
        tex->params.sampleable = fmt->caps & PL_FMT_CAP_SAMPLEABLE;
        tex->params.storable = fmt->caps & PL_FMT_CAP_STORABLE;
        tex->params.host_writable = !fmt->opaque;
        tex->params.host_readable |= fmt->caps & PL_FMT_CAP_HOST_READABLE;
    }
    if (tex_gl->fbo || tex_gl->wrapped_fb) {
        tex->params.renderable = fmt->caps & PL_FMT_CAP_RENDERABLE;
        tex->params.host_readable |= fmt->caps & PL_FMT_CAP_HOST_READABLE;
        if (dims == 2 && (fmt->caps & PL_FMT_CAP_BLITTABLE)) {
            tex->params.blit_src = true;
            tex->params.blit_dst = true;
        }
    }

    tex_gl->barrier = tex_barrier(tex);
    release_current(gpu);
    return tex;

error:
    gl_tex_destroy(gpu, tex);
    release_current(gpu);
    return NULL;
}

unsigned int pl_opengl_unwrap(pl_gpu gpu, pl_tex tex,
                              unsigned int *out_target, int *out_iformat,
                              unsigned int *out_fbo)
{
    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    if (!tex_gl->texture) {
        PL_ERR(gpu, "Trying to call `pl_opengl_unwrap` on a pseudo-texture "
               "(perhaps obtained by `pl_swapchain_start_frame`?)");
        return 0;
    }

    if (out_target)
        *out_target = tex_gl->target;
    if (out_iformat)
        *out_iformat = tex_gl->iformat;
    if (out_fbo)
        *out_fbo = tex_gl->fbo;

    return tex_gl->texture;
}

static void gl_tex_invalidate(pl_gpu gpu, pl_tex tex)
{
    struct pl_gl *p = PL_PRIV(gpu);
    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    if (!p->has_invalidate_fb || !make_current(gpu))
        return;

    if (tex_gl->wrapped_fb) {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, tex_gl->fbo);
        glInvalidateFramebuffer(GL_DRAW_FRAMEBUFFER, 1, (GLenum[]){GL_COLOR});
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    } else if (p->has_invalidate_tex) {
        glInvalidateTexImage(tex_gl->texture, 0);
    } else {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, tex_gl->fbo);
        glInvalidateFramebuffer(GL_DRAW_FRAMEBUFFER,
                                1, (GLenum[]){GL_COLOR_ATTACHMENT0});
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    }

    gl_check_err(gpu, "gl_tex_invalidate");
    release_current(gpu);
}

static void gl_tex_clear_ex(pl_gpu gpu, pl_tex tex, const union pl_clear_color color)
{
    if (!make_current(gpu))
        return;

    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    pl_assert(tex_gl->fbo || tex_gl->wrapped_fb);

    switch (tex->params.format->type) {
    case PL_FMT_UNKNOWN:
    case PL_FMT_FLOAT:
    case PL_FMT_UNORM:
    case PL_FMT_SNORM:
        glClearColor(color.f[0], color.f[1], color.f[2], color.f[3]);
        break;

    case PL_FMT_UINT:
        glClearColorIuiEXT(color.u[0], color.u[1], color.u[2], color.u[3]);
        break;

    case PL_FMT_SINT:
        glClearColorIiEXT(color.i[0], color.i[1], color.i[2], color.i[3]);
        break;

    case PL_FMT_TYPE_COUNT:
        pl_unreachable();
    }

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, tex_gl->fbo);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    gl_check_err(gpu, "gl_tex_clear");
    release_current(gpu);
}

static const GLint filters[PL_TEX_SAMPLE_MODE_COUNT] = {
    [PL_TEX_SAMPLE_NEAREST] = GL_NEAREST,
    [PL_TEX_SAMPLE_LINEAR]  = GL_LINEAR,
};

static void gl_tex_blit(pl_gpu gpu, const struct pl_tex_blit_params *params)
{
    if (!make_current(gpu))
        return;

    struct pl_tex_gl *src_gl = PL_PRIV(params->src);
    struct pl_tex_gl *dst_gl = PL_PRIV(params->dst);

    pl_assert(src_gl->fbo || src_gl->wrapped_fb);
    pl_assert(dst_gl->fbo || dst_gl->wrapped_fb);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, src_gl->fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst_gl->fbo);

    struct pl_rect3d src_rc = params->src_rc, dst_rc = params->dst_rc;
    glBlitFramebuffer(src_rc.x0, src_rc.y0, src_rc.x1, src_rc.y1,
                      dst_rc.x0, dst_rc.y0, dst_rc.x1, dst_rc.y1,
                      GL_COLOR_BUFFER_BIT, filters[params->sample_mode]);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    gl_check_err(gpu, "gl_tex_blit");
    release_current(gpu);
}

// For pl_buf.priv
struct pl_buf_gl {
    uint64_t id; // unique per buffer
    GLuint buffer;
    size_t offset;
    GLsync fence;
    GLbitfield barrier;
    bool mapped;
};

static void gl_buf_destroy(pl_gpu gpu, pl_buf buf)
{
    if (!make_current(gpu)) {
        PL_ERR(gpu, "Failed uninitializing buffer, leaking resources!");
        return;
    }

    struct pl_buf_gl *buf_gl = PL_PRIV(buf);
    if (buf_gl->fence)
        glDeleteSync(buf_gl->fence);

    if (buf_gl->mapped) {
        glBindBuffer(GL_COPY_WRITE_BUFFER, buf_gl->buffer);
        glUnmapBuffer(GL_COPY_WRITE_BUFFER);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }

    glDeleteBuffers(1, &buf_gl->buffer);
    gl_check_err(gpu, "gl_buf_destroy");
    release_current(gpu);
    pl_free((void *) buf);
}

static pl_buf gl_buf_create(pl_gpu gpu, const struct pl_buf_params *params)
{
    if (!make_current(gpu))
        return NULL;

    struct pl_buf *buf = pl_zalloc_obj(NULL, buf, struct pl_buf_gl);
    buf->params = *params;
    buf->params.initial_data = NULL;

    struct pl_gl *p = PL_PRIV(gpu);
    struct pl_buf_gl *buf_gl = PL_PRIV(buf);
    buf_gl->id = ++p->buf_id;

    // Just use this since the generic GL_BUFFER doesn't work
    GLenum target = GL_COPY_WRITE_BUFFER;
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

    glGenBuffers(1, &buf_gl->buffer);
    glBindBuffer(target, buf_gl->buffer);

    if (test_ext(gpu, "GL_ARB_buffer_storage", 44, 0) && !import) {

        GLbitfield mapflags = 0, storflags = 0;
        if (params->host_writable)
            storflags |= GL_DYNAMIC_STORAGE_BIT;
        if (params->host_mapped) {
            mapflags |= GL_MAP_READ_BIT | GL_MAP_WRITE_BIT |
                        GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
        }
        if (params->memory_type == PL_BUF_MEM_HOST)
            storflags |= GL_CLIENT_STORAGE_BIT; // hopefully this works

        glBufferStorage(target, total_size, data, storflags | mapflags);

        if (params->host_mapped) {
            buf_gl->mapped = true;
            buf->data = glMapBufferRange(target, buf_gl->offset, params->size,
                                         mapflags);
            if (!buf->data) {
                glBindBuffer(target, 0);
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

        glBufferData(target, total_size, data, hint);

        if (import && glGetError() == GL_INVALID_OPERATION) {
            PL_ERR(gpu, "Failed importing host pointer!");
            goto error;
        }

    }

    glBindBuffer(target, 0);
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

    release_current(gpu);
    return buf;

error:
    gl_buf_destroy(gpu, buf);
    release_current(gpu);
    return NULL;
}

static bool gl_buf_poll(pl_gpu gpu, pl_buf buf, uint64_t timeout)
{
    // Non-persistently mapped buffers are always implicitly reusable in OpenGL,
    // the implementation will create more buffers under the hood if needed.
    if (!buf->data)
        return false;

    if (!make_current(gpu))
        return true; // conservative guess

    struct pl_buf_gl *buf_gl = PL_PRIV(buf);
    if (buf_gl->fence) {
        GLenum res = glClientWaitSync(buf_gl->fence,
                                      timeout ? GL_SYNC_FLUSH_COMMANDS_BIT : 0,
                                      timeout);
        if (res == GL_ALREADY_SIGNALED || res == GL_CONDITION_SATISFIED) {
            glDeleteSync(buf_gl->fence);
            buf_gl->fence = NULL;
        }
    }

    gl_poll_callbacks(gpu);
    release_current(gpu);
    return !!buf_gl->fence;
}

static void gl_buf_write(pl_gpu gpu, pl_buf buf, size_t offset,
                         const void *data, size_t size)
{
    if (!make_current(gpu))
        return;

    struct pl_buf_gl *buf_gl = PL_PRIV(buf);
    glBindBuffer(GL_COPY_WRITE_BUFFER, buf_gl->buffer);
    glBufferSubData(GL_COPY_WRITE_BUFFER, buf_gl->offset + offset, size, data);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    gl_check_err(gpu, "gl_buf_write");
    release_current(gpu);
}

static bool gl_buf_read(pl_gpu gpu, pl_buf buf, size_t offset,
                        void *dest, size_t size)
{
    if (!make_current(gpu))
        return false;

    struct pl_buf_gl *buf_gl = PL_PRIV(buf);
    glBindBuffer(GL_COPY_READ_BUFFER, buf_gl->buffer);
    glGetBufferSubData(GL_COPY_READ_BUFFER, buf_gl->offset + offset, size, dest);
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    bool ok = gl_check_err(gpu, "gl_buf_read");
    release_current(gpu);
    return ok;
}

static void gl_buf_copy(pl_gpu gpu, pl_buf dst, size_t dst_offset,
                        pl_buf src, size_t src_offset, size_t size)
{
    if (!make_current(gpu))
        return;

    struct pl_buf_gl *src_gl = PL_PRIV(src);
    struct pl_buf_gl *dst_gl = PL_PRIV(dst);
    glBindBuffer(GL_COPY_READ_BUFFER, src_gl->buffer);
    glBindBuffer(GL_COPY_WRITE_BUFFER, dst_gl->buffer);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                        src_gl->offset + src_offset,
                        dst_gl->offset + dst_offset, size);
    gl_check_err(gpu, "gl_buf_copy");
    release_current(gpu);
}

static int get_alignment(int stride)
{
    if (stride % 8 == 0)
        return 8;
    if (stride % 4 == 0)
        return 4;
    if (stride % 2 == 0)
        return 2;
    return 1;
}

static void gl_timer_begin(pl_timer timer);
static void gl_timer_end(pl_timer timer);

static bool gl_tex_upload(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    struct pl_gl *p = PL_PRIV(gpu);
    pl_tex tex = params->tex;
    pl_buf buf = params->buf;
    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    struct pl_buf_gl *buf_gl = buf ? PL_PRIV(buf) : NULL;

    // If the user requests asynchronous uploads, it's more efficient to do
    // them via a PBO - this allows us to skip blocking the caller, especially
    // when the host pointer can be imported directly.
    if (params->callback && !buf) {
        size_t buf_size = pl_tex_transfer_size(params);
        const size_t min_size = 32*1024; // 32 KiB
        if (buf_size >= min_size && buf_size <= gpu->limits.max_buf_size)
            return pl_tex_upload_pbo(gpu, params);
    }

    if (!make_current(gpu))
        return false;

    const void *src = params->ptr;
    if (buf) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf_gl->buffer);
        src = (void *) (buf_gl->offset + params->buf_offset);
    }

    int dims = pl_tex_params_dimension(tex->params);
    if (dims > 1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, get_alignment(params->stride_w));

    int rows = pl_rect_h(params->rc);
    if (params->stride_w != tex->params.w) {
        if (p->has_stride) {
            glPixelStorei(GL_UNPACK_ROW_LENGTH, params->stride_w);
        } else {
            rows = tex->params.w == params->stride_w ? rows : 1;
        }
    }

    int imgs = pl_rect_d(params->rc);
    if (params->stride_h != tex->params.h) {
        if (p->has_stride) {
            glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, params->stride_h);
        } else {
            imgs = tex->params.h == params->stride_h ? imgs : 1;
        }
    }

    glBindTexture(tex_gl->target, tex_gl->texture);
    gl_timer_begin(params->timer);

    switch (dims) {
    case 1:
        glTexSubImage1D(tex_gl->target, 0, params->rc.x0, pl_rect_w(params->rc),
                        tex_gl->format, tex_gl->type, src);
        break;
    case 2:
        for (int y = params->rc.y0; y < params->rc.y1; y += rows) {
            glTexSubImage2D(tex_gl->target, 0, params->rc.x0, y,
                            pl_rect_w(params->rc), rows, tex_gl->format,
                            tex_gl->type, src);
        }
        break;
    case 3:
        for (int z = params->rc.z0; z < params->rc.z1; z += imgs) {
            for (int y = params->rc.y0; y < params->rc.y1; y += rows) {
                glTexSubImage3D(tex_gl->target, 0, params->rc.x0, y, z,
                                pl_rect_w(params->rc), rows, imgs,
                                tex_gl->format, tex_gl->type, src);
            }
        }
        break;
    }

    gl_timer_end(params->timer);
    glBindTexture(tex_gl->target, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    if (p->has_stride) {
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glPixelStorei(GL_UNPACK_IMAGE_HEIGHT,0);
    }

    if (buf) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        if (buf->params.host_mapped) {
            // Make sure the PBO is not reused until GL is done with it. If a
            // previous operation is pending, "update" it by creating a new
            // fence that will cover the previous operation as well.
            glDeleteSync(buf_gl->fence);
            buf_gl->fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        }
    }

    if (params->callback) {
        PL_ARRAY_APPEND(gpu, p->callbacks, (struct gl_cb) {
            .sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0),
            .callback = params->callback,
            .priv = params->priv,
        });
    }

    bool ok = gl_check_err(gpu, "gl_tex_upload");
    release_current(gpu);
    return ok;
}

static bool gl_tex_download(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    struct pl_gl *p = PL_PRIV(gpu);
    pl_tex tex = params->tex;
    pl_buf buf = params->buf;
    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    struct pl_buf_gl *buf_gl = buf ? PL_PRIV(buf) : NULL;
    bool ok = true;

    if (params->callback && !buf) {
        size_t buf_size = pl_tex_transfer_size(params);
        const size_t min_size = 32*1024; // 32 KiB
        if (buf_size >= min_size && buf_size <= gpu->limits.max_buf_size)
            return pl_tex_download_pbo(gpu, params);
    }

    if (!make_current(gpu))
        return false;

    void *dst = params->ptr;
    if (buf) {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, buf_gl->buffer);
        dst = (void *) (buf_gl->offset + params->buf_offset);
    }

    struct pl_rect3d full = {
        0, 0, 0,
        tex->params.w,
        PL_DEF(tex->params.h, 1),
        PL_DEF(tex->params.d, 1),
    };

    int dims = pl_tex_params_dimension(tex->params);
    bool is_copy = pl_rect3d_eq(params->rc, full) &&
                   params->stride_w == tex->params.w &&
                   params->stride_h == PL_DEF(tex->params.h, 1);

    gl_timer_begin(params->timer);

    if (tex_gl->fbo || tex_gl->wrapped_fb) {
        // We can use a more efficient path when we have an FBO available
        if (dims > 1) {
            size_t real_stride = params->stride_w * tex->params.format->texel_size;
            glPixelStorei(GL_PACK_ALIGNMENT, get_alignment(real_stride));
        }

        int rows = pl_rect_h(params->rc);
        if (params->stride_w != tex->params.w) {
            if (p->has_stride) {
                glPixelStorei(GL_PACK_ROW_LENGTH, params->stride_w);
            } else {
                rows = tex->params.w == params->stride_w ? rows : 1;
            }
        }

        // No 3D framebuffers
        pl_assert(pl_rect_d(params->rc) == 1);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, tex_gl->fbo);
        for (int y = params->rc.y0; y < params->rc.y1; y += rows) {
            glReadPixels(params->rc.x0, y, pl_rect_w(params->rc), rows,
                         tex_gl->format, tex_gl->type, dst);
        }
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        if (p->has_stride)
            glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    } else if (is_copy) {
        // We're downloading the entire texture
        glBindTexture(tex_gl->target, tex_gl->texture);
        glGetTexImage(tex_gl->target, 0, tex_gl->format, tex_gl->type, dst);
        glBindTexture(tex_gl->target, 0);
    } else {
        PL_ERR(gpu, "Partial downloads of 3D textures not implemented!");
        ok = false;
    }

    gl_timer_end(params->timer);

    if (buf) {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        if (ok && buf->params.host_mapped) {
            glDeleteSync(buf_gl->fence);
            buf_gl->fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        }
    }

    if (params->callback) {
        PL_ARRAY_APPEND(gpu, p->callbacks, (struct gl_cb) {
            .sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0),
            .callback = params->callback,
            .priv = params->priv,
        });
    }

    ok &= gl_check_err(gpu, "gl_tex_download");
    release_current(gpu);
    return ok;
}

static int gl_desc_namespace(pl_gpu gpu, enum pl_desc_type type)
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
    if (!test_ext(gpu, "GL_ARB_get_program_binary", 41, 30))
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
        static const char *shader_name;
        switch (type) {
        case GL_VERTEX_SHADER:   shader_name = "vertex"; break;
        case GL_FRAGMENT_SHADER: shader_name = "fragment"; break;
        case GL_COMPUTE_SHADER:  shader_name = "compute"; break;
        default: pl_unreachable();
        };

        PL_MSG(gpu, level, "%s shader source:", shader_name);
        pl_msg_source(gpu->log, level, src);

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
    GLint *var_locs;
};

static void gl_pass_destroy(pl_gpu gpu, pl_pass pass)
{
    if (!make_current(gpu)) {
        PL_ERR(gpu, "Failed uninitializing pass, leaking resources!");
        return;
    }

    struct pl_pass_gl *pass_gl = PL_PRIV(pass);
    if (pass_gl->vao)
        glDeleteVertexArrays(1, &pass_gl->vao);
    glDeleteBuffers(1, &pass_gl->buffer);
    glDeleteProgram(pass_gl->program);

    gl_check_err(gpu, "gl_pass_destroy");
    release_current(gpu);
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

static pl_pass gl_pass_create(pl_gpu gpu, const struct pl_pass_params *params)
{
    if (!make_current(gpu))
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
    if (test_ext(gpu, "GL_ARB_get_program_binary", 41, 30)) {
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

    release_current(gpu);
    return pass;

error:
    PL_ERR(gpu, "Failed creating pass");
    gl_pass_destroy(gpu, pass);
    release_current(gpu);
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

static void gl_pass_run(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    if (!make_current(gpu))
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
        release_current(gpu);
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
        }
        gl_check_err(gpu, "gl_pass_run: enabling blend");

        // Update VBO and VAO
        pl_buf vert = params->vertex_buf;
        struct pl_buf_gl *vert_gl = vert ? PL_PRIV(vert) : NULL;
        glBindBuffer(GL_ARRAY_BUFFER, vert ? vert_gl->buffer : pass_gl->buffer);

        if (!vert) {
            // Update the buffer directly. In theory we could also do a memcmp
            // cache here to avoid unnecessary updates.
            int num_vertices = 0;
            if (params->index_data) {
                // Indexed draw, so we need to store all indexed vertices
                for (int i = 0; i < params->vertex_count; i++)
                    num_vertices = PL_MAX(num_vertices, params->index_data[i]);
                num_vertices += 1;
            } else {
                num_vertices = params->vertex_count;
            }
            size_t vert_size = num_vertices * pass->params.vertex_stride;
            glBufferData(GL_ARRAY_BUFFER, vert_size, params->vertex_data, GL_STREAM_DRAW);
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

        gl_timer_begin(params->timer);

        if (params->index_data) {

            // GL allows taking indices directly from a pointer
            glDrawElements(mode, params->vertex_count, GL_UNSIGNED_SHORT,
                           params->index_data);

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

        gl_timer_end(params->timer);
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
        gl_timer_begin(params->timer);
        glDispatchCompute(params->compute_groups[0],
                          params->compute_groups[1],
                          params->compute_groups[2]);
        gl_timer_end(params->timer);
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
    release_current(gpu);
}

#define QUERY_OBJECT_NUM 8

struct pl_timer {
    GLuint query[QUERY_OBJECT_NUM];
    int index_write; // next index to write to
    int index_read; // next index to read from
};

static pl_timer gl_timer_create(pl_gpu gpu)
{
    struct pl_gl *p = PL_PRIV(gpu);
    if (!p->has_queries || !make_current(gpu))
        return NULL;

    pl_timer timer = pl_zalloc_ptr(NULL, timer);
    glGenQueries(QUERY_OBJECT_NUM, timer->query);
    release_current(gpu);
    return timer;
}

static void gl_timer_destroy(pl_gpu gpu, pl_timer timer)
{
    if (!make_current(gpu)) {
        PL_ERR(gpu, "Failed uninitializing timer, leaking resources!");
        return;
    }

    glDeleteQueries(QUERY_OBJECT_NUM, timer->query);
    gl_check_err(gpu, "gl_timer_destroy");
    release_current(gpu);
    pl_free(timer);
}

static uint64_t gl_timer_query(pl_gpu gpu, pl_timer timer)
{
    if (timer->index_read == timer->index_write)
        return 0; // no more unprocessed results

    if (!make_current(gpu))
        return 0;

    uint64_t res = 0;
    GLuint query = timer->query[timer->index_read];
    int avail = 0;
    glGetQueryObjectiv(query, GL_QUERY_RESULT_AVAILABLE, &avail);
    if (!avail)
        goto done;
    glGetQueryObjectui64v(query, GL_QUERY_RESULT, &res);

    timer->index_read = (timer->index_read + 1) % QUERY_OBJECT_NUM;
    // fall through

done:
    release_current(gpu);
    return res;
}

static void gl_timer_begin(pl_timer timer)
{
    if (!timer)
        return;

    glBeginQuery(GL_TIME_ELAPSED, timer->query[timer->index_write]);
}

static void gl_timer_end(pl_timer timer)
{
    if (!timer)
        return;

    glEndQuery(GL_TIME_ELAPSED);

    timer->index_write = (timer->index_write + 1) % QUERY_OBJECT_NUM;
    if (timer->index_write == timer->index_read) {
        // forcibly drop the least recent result to make space
        timer->index_read = (timer->index_read + 1) % QUERY_OBJECT_NUM;
    }
}

static void gl_gpu_flush(pl_gpu gpu)
{
    if (!make_current(gpu))
        return;

    glFlush();
    gl_check_err(gpu, "gl_gpu_flush");
    release_current(gpu);
}

static void gl_gpu_finish(pl_gpu gpu)
{
    if (!make_current(gpu))
        return;

    glFinish();
    gl_check_err(gpu, "gl_gpu_finish");
    release_current(gpu);
}

static bool gl_gpu_is_failed(pl_gpu gpu)
{
    struct pl_gl *gl = PL_PRIV(gpu);
    return gl->failed;
}

static const struct pl_gpu_fns pl_fns_gl = {
    .destroy                = gl_destroy_gpu,
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
