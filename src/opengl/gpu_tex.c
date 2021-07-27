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

#ifdef PL_HAVE_UNIX
#include <unistd.h>
#include <errno.h>
#endif

void gl_tex_destroy(pl_gpu gpu, pl_tex tex)
{
    if (!MAKE_CURRENT()) {
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
    RELEASE_CURRENT();
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
    if (!MAKE_CURRENT())
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
            PL_ERR(gpu, "%s: cannot duplicate fd %d for importing: %s",
                   __func__, shared_mem->handle.fd, strerror(errno));
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

    RELEASE_CURRENT();
    return true;

error:
    PL_ERR(gpu, "Failed importing GL texture!");
    RELEASE_CURRENT();
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

pl_tex gl_tex_create(pl_gpu gpu, const struct pl_tex_params *params)
{
    if (!MAKE_CURRENT())
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

    RELEASE_CURRENT();
    return tex;

error:
    gl_tex_destroy(gpu, tex);
    RELEASE_CURRENT();
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

    bool can_query = !gl_test_ext(gpu, "GL_ARB_framebuffer_object", 30, 20);
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
    if (!MAKE_CURRENT())
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
    RELEASE_CURRENT();
    return tex;

error:
    gl_tex_destroy(gpu, tex);
    RELEASE_CURRENT();
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

void gl_tex_invalidate(pl_gpu gpu, pl_tex tex)
{
    struct pl_gl *p = PL_PRIV(gpu);
    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    if (!p->has_invalidate_fb || !MAKE_CURRENT())
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
    RELEASE_CURRENT();
}

void gl_tex_clear_ex(pl_gpu gpu, pl_tex tex, const union pl_clear_color color)
{
    if (!MAKE_CURRENT())
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
    RELEASE_CURRENT();
}

void gl_tex_blit(pl_gpu gpu, const struct pl_tex_blit_params *params)
{
    if (!MAKE_CURRENT())
        return;

    struct pl_tex_gl *src_gl = PL_PRIV(params->src);
    struct pl_tex_gl *dst_gl = PL_PRIV(params->dst);

    pl_assert(src_gl->fbo || src_gl->wrapped_fb);
    pl_assert(dst_gl->fbo || dst_gl->wrapped_fb);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, src_gl->fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst_gl->fbo);

    static const GLint filters[PL_TEX_SAMPLE_MODE_COUNT] = {
        [PL_TEX_SAMPLE_NEAREST] = GL_NEAREST,
        [PL_TEX_SAMPLE_LINEAR]  = GL_LINEAR,
    };

    struct pl_rect3d src_rc = params->src_rc, dst_rc = params->dst_rc;
    glBlitFramebuffer(src_rc.x0, src_rc.y0, src_rc.x1, src_rc.y1,
                      dst_rc.x0, dst_rc.y0, dst_rc.x1, dst_rc.y1,
                      GL_COLOR_BUFFER_BIT, filters[params->sample_mode]);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    gl_check_err(gpu, "gl_tex_blit");
    RELEASE_CURRENT();
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

bool gl_tex_upload(pl_gpu gpu, const struct pl_tex_transfer_params *params)
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

    if (!MAKE_CURRENT())
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
        if (p->has_unpack_image_height) {
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
    if (p->has_stride && params->stride_w != tex->params.w)
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    if (p->has_unpack_image_height && params->stride_h != tex->params.h)
        glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, 0);

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
    RELEASE_CURRENT();
    return ok;
}

bool gl_tex_download(pl_gpu gpu, const struct pl_tex_transfer_params *params)
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

    if (!MAKE_CURRENT())
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
    RELEASE_CURRENT();
    return ok;
}
