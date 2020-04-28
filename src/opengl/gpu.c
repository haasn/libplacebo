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

static const struct pl_gpu_fns pl_fns_gl;

// For gpu.priv
struct pl_gl {
    struct pl_gpu_fns impl;

    // Cached capabilities
    int gl_ver;
    int gles_ver;
    bool has_stride;
    bool has_invalidate;
    bool has_vao;
};

static bool test_ext(const struct pl_gpu *gpu, const char *ext,
                     int gl_ver, int gles_ver)
{
    struct pl_gl *p = TA_PRIV(gpu);
    if (gl_ver && p->gl_ver >= gl_ver)
        return true;
    if (gles_ver && p->gles_ver >= gles_ver)
        return true;

    return ext ? epoxy_has_gl_extension(ext) : false;
}

static void gl_destroy_gpu(const struct pl_gpu *gpu)
{
    talloc_free((void *) gpu);
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


static bool gl_setup_formats(struct pl_gpu *gpu)
{
    struct pl_gl *p = TA_PRIV(gpu);
    int features = gl_format_feature_flags(gpu);
    bool has_fbos = test_ext(gpu, "GL_ARB_framebuffer_object", 30, 20);
    bool has_storage = test_ext(gpu, "GL_ARB_shader_image_load_store", 42, 0);

    for (const struct gl_format *gl_fmt = gl_formats; gl_fmt->ifmt; gl_fmt++) {
        if (gl_fmt->ver && !(gl_fmt->ver & features))
            continue;

        // Eliminate duplicate formats
        for (int i = 0; i < gpu->num_formats; i++) {
            const struct gl_format **fmtp = TA_PRIV(gpu->formats[i]);
            if ((*fmtp)->ifmt == gl_fmt->ifmt)
                goto next_gl_fmt;
        }

        struct pl_fmt *fmt = talloc_ptrtype_priv(gpu, fmt, gl_fmt);
        const struct gl_format **fmtp = TA_PRIV(fmt);
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
        default: abort();
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
        default: abort();
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
        // NOTE: 3-component formats (e.g. rgb8) are probably also emulated,
        // but we have no way of knowing this for sure, so let's just ignore it.
        fmt->emulated = fmt->texel_size != fmt->internal_size;

        // For sanity, clear the superfluous fields
        for (int i = fmt->num_components; i < 4; i++) {
            fmt->component_depth[i] = 0;
            fmt->sample_order[i] = 0;
            fmt->host_bits[i] = 0;
        }

        fmt->glsl_type = pl_var_glsl_type_name(pl_var_from_fmt(fmt, ""));
        fmt->glsl_format = pl_fmt_glsl_format(fmt, fmt->num_components);
        pl_assert(fmt->glsl_type);

        // Add format capabilities based on the flags
        fmt->caps = gl_fmt->caps;

        // Mask renderable/blittable if no FBOs available
        if (!has_fbos)
            fmt->caps &= ~(PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLITTABLE);

        // Reading from FBOs on GLES requires FBO support for this fmt
        if (p->gl_ver || (fmt->caps & PL_FMT_CAP_RENDERABLE))
            fmt->caps |= PL_FMT_CAP_HOST_READABLE;

        if ((gpu->caps & PL_GPU_CAP_COMPUTE) && fmt->glsl_format && has_storage)
            fmt->caps |= PL_FMT_CAP_STORABLE;

        // Only float-type formats are considered blendable in OpenGL
        switch (fmt->type) {
        case PL_FMT_FLOAT:
        case PL_FMT_UNORM:
        case PL_FMT_SNORM:
            if (fmt->caps & PL_FMT_CAP_RENDERABLE)
                fmt->caps |= PL_FMT_CAP_BLENDABLE;
        default: break;
        }

        // TODO: Texel buffers

        TARRAY_APPEND(gpu, gpu->formats, gpu->num_formats, fmt);

next_gl_fmt: ;
    }

    pl_gpu_sort_formats(gpu);
    pl_gpu_verify_formats(gpu);
    return gl_check_err(gpu, "gl_setup_formats");
}

const struct pl_gpu *pl_gpu_create_gl(struct pl_context *ctx)
{
    struct pl_gpu *gpu = talloc_zero_priv(NULL, struct pl_gpu, struct pl_gl);
    gpu->ctx = ctx;
    gpu->glsl.gles = !epoxy_is_desktop_gl();

    struct pl_gl *p = TA_PRIV(gpu);
    p->impl = pl_fns_gl;
    int ver = epoxy_gl_version();
    p->gl_ver = gpu->glsl.gles ? 0 : ver;
    p->gles_ver = gpu->glsl.gles ? ver : 0;

    // Query support for the capabilities
    gpu->caps |= PL_GPU_CAP_INPUT_VARIABLES;
    if (test_ext(gpu, "GL_ARB_compute_shader", 43, 0))
        gpu->caps |= PL_GPU_CAP_COMPUTE;
    if (test_ext(gpu, "GL_ARB_buffer_storage", 44, 0))
        gpu->caps |= PL_GPU_CAP_MAPPED_BUFFERS;

    // If possible, query the GLSL version from the implementation
    const char *glslver = glGetString(GL_SHADING_LANGUAGE_VERSION);
    if (glslver) {
        PL_INFO(gpu, "    GL_SHADING_LANGUAGE_VERSION: %s", glslver);
        int major = 0, minor = 0;
        if (sscanf(glslver, "%d.%d", &major, &minor) == 2)
            gpu->glsl.version = major * 100 + minor;
    }

    if (!gpu->glsl.version) {
        // Otherwise, use the fixed magic versions 200 and 300 for early GLES,
        // and otherwise fall back to 110 if all else fails.
        if (p->gles_ver >= 30) {
            gpu->glsl.version = 300;
        } else if (p->gles_ver >= 20) {
            gpu->glsl.version = 200;
        } else {
            gpu->glsl.version = 110;
        }
    }

    // Query all device limits
    struct pl_gpu_limits *l = &gpu->limits;
    get(GL_MAX_TEXTURE_SIZE, &l->max_tex_2d_dim);
    if (test_ext(gpu, NULL, 21, 30)) // FIXME: is there an ext for this?
        get(GL_MAX_3D_TEXTURE_SIZE, &l->max_tex_3d_dim);

    // There's no equivalent limit for 1D textures for whatever reason, so
    // just set it to the same as the 2D limit
    if (p->gl_ver >= 21)
        l->max_tex_1d_dim = l->max_tex_2d_dim;

    if (test_ext(gpu, "GL_ARB_pixel_buffer_object", 0, 0)) // FIXME: when is this core?
        l->max_xfer_size = SIZE_MAX; // no limit imposed by GL
    if (test_ext(gpu, "GL_ARB_uniform_buffer_object", 31, 0))
        get(GL_MAX_UNIFORM_BLOCK_SIZE, &l->max_ubo_size);
    if (test_ext(gpu, "GL_ARB_shader_storage_buffer_object", 43, 0) &&
        test_ext(gpu, "GL_ARB_shader_image_load_store", 42, 0))
    {
        // storage image support is needed for glMemoryBarrier!
        get(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &l->max_ssbo_size);
    }

    if (test_ext(gpu, "GL_ARB_texture_gather", 0, 0)) { // FIXME: when is this core?
        get(GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET_ARB, &l->min_gather_offset);
        get(GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET_ARB, &l->max_gather_offset);
    }

    if (gpu->caps & PL_GPU_CAP_COMPUTE) {
        get(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &l->max_shmem_size);
        get(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &l->max_group_threads);
        for (int i = 0; i < 3; i++) {
            geti(GL_MAX_COMPUTE_WORK_GROUP_COUNT, i, &l->max_dispatch[i]);
            geti(GL_MAX_COMPUTE_WORK_GROUP_SIZE, i, &l->max_group_size[i]);
        }
    }

    // Cached some existing capability checks
    p->has_stride = test_ext(gpu, "GL_EXT_unpack_subimage", 11, 30);
    p->has_vao = test_ext(gpu, "GL_ARB_vertex_array_object", 30, 0);
    p->has_invalidate = test_ext(gpu, "GL_ARB_invalidate_subdata", 43, 30);

    // We simply don't know, so make up some values
    gpu->limits.align_tex_xfer_offset = 32;
    gpu->limits.align_tex_xfer_stride = 1;
    if (test_ext(gpu, "GL_EXT_unpack_subimage", 11, 30))
        gpu->limits.align_tex_xfer_stride = 4;

    if (!gl_check_err(gpu, "pl_gpu_create_gl"))
        goto error;

    if (!gl_setup_formats(gpu))
        goto error;

    pl_gpu_print_info(gpu, PL_LOG_INFO);
    pl_gpu_print_formats(gpu, PL_LOG_DEBUG);
    return gpu;

error:
    gl_destroy_gpu(gpu);
    return NULL;
}

// For pl_tex.priv
struct pl_tex_gl {
    GLenum target;
    GLuint texture;
    GLint filter;
    GLuint fbo; // or 0
    bool wrapped_fb;

    // GL format fields
    GLenum format;
    GLint iformat;
    GLenum type;
};

static void gl_tex_destroy(const struct pl_gpu *gpu, const struct pl_tex *tex)
{
    struct pl_tex_gl *tex_gl = TA_PRIV(tex);
    if (tex_gl->fbo)
        glDeleteFramebuffers(1, &tex_gl->fbo);
    glDeleteTextures(1, &tex_gl->texture);

    talloc_free((void *) tex);
    gl_check_err(gpu, "gl_tex_destroy");
}

static const struct pl_tex *gl_tex_create(const struct pl_gpu *gpu,
                                          const struct pl_tex_params *params)
{
    struct pl_gl *p = TA_PRIV(gpu);

    struct pl_tex *tex = talloc_zero_priv(NULL, struct pl_tex, struct pl_tex_gl);
    tex->params = *params;
    tex->params.initial_data = NULL;

    struct pl_tex_gl *tex_gl = TA_PRIV(tex);

    const struct gl_format **fmtp = TA_PRIV(params->format);
    const struct gl_format *fmt = *fmtp;
    tex_gl->format = fmt->fmt;
    tex_gl->iformat = fmt->ifmt;
    tex_gl->type = fmt->type;

    static const GLint targets[] = {
        [1] = GL_TEXTURE_1D,
        [2] = GL_TEXTURE_2D,
        [3] = GL_TEXTURE_3D,
    };

    static const GLint filters[] = {
        [PL_TEX_SAMPLE_NEAREST] = GL_NEAREST,
        [PL_TEX_SAMPLE_LINEAR]  = GL_LINEAR,
    };

    static const GLint wraps[] = {
        [PL_TEX_ADDRESS_CLAMP]  = GL_CLAMP_TO_EDGE,
        [PL_TEX_ADDRESS_REPEAT] = GL_REPEAT,
        [PL_TEX_ADDRESS_MIRROR] = GL_MIRRORED_REPEAT,
    };

    int dims = pl_tex_params_dimension(*params);
    pl_assert(dims >= 1 && dims <= 3);
    tex_gl->target = targets[dims];
    tex_gl->filter = filters[params->sample_mode];
    GLint wrap = wraps[params->address_mode];

    glGenTextures(1, &tex_gl->texture);
    glBindTexture(tex_gl->target, tex_gl->texture);
    glTexParameteri(tex_gl->target, GL_TEXTURE_MIN_FILTER, tex_gl->filter);
    glTexParameteri(tex_gl->target, GL_TEXTURE_MAG_FILTER, tex_gl->filter);

    switch (dims) {
    case 3: glTexParameteri(tex_gl->target, GL_TEXTURE_WRAP_R, wrap);
        // fall through
    case 2:
        glTexParameteri(tex_gl->target, GL_TEXTURE_WRAP_T, wrap);
        // fall through
    case 1:
        glTexParameteri(tex_gl->target, GL_TEXTURE_WRAP_S, wrap);
        break;
    }

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
        glBindFramebuffer(GL_FRAMEBUFFER, tex_gl->fbo);
        switch (dims) {
        case 1:
            glFramebufferTexture1D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_1D, tex_gl->texture, 0);
            break;
        case 2:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, tex_gl->texture, 0);
            break;
        case 3: abort();
        }

        GLenum err = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (err != GL_FRAMEBUFFER_COMPLETE) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            PL_ERR(gpu, "Failed creating framebuffer: error code %d", err);
            goto error;
        }

        if (params->host_readable && p->gles_ver) {
            GLint read_type = 0, read_fmt = 0;
            glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &read_type);
            glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &read_fmt);
            if (read_type != tex_gl->type || read_fmt != tex_gl->format) {
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
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

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        if (!gl_check_err(gpu, "gl_tex_create: fbo"))
            goto error;
    }

    return tex;

error:
    gl_tex_destroy(gpu, tex);
    return NULL;
}

static bool gl_fb_query(const struct pl_gpu *gpu, int fbo, struct pl_fmt *fmt)
{
    if (!test_ext(gpu, "GL_ARB_framebuffer_object", 30, 20))
        return true;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    GLenum obj = gpu->glsl.gles ? GL_BACK : GL_BACK_LEFT;
    if (fbo != 0)
        obj = GL_COLOR_ATTACHMENT0;

    GLint type = 0;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, obj,
            GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE, &type);
    switch (type) {
    case GL_FLOAT:                  fmt->type = PL_FMT_FLOAT; break;
    case GL_INT:                    fmt->type = PL_FMT_SINT; break;
    case GL_UNSIGNED_INT:           fmt->type = PL_FMT_UINT; break;
    case GL_SIGNED_NORMALIZED:      fmt->type = PL_FMT_SNORM; break;
    case GL_UNSIGNED_NORMALIZED:    fmt->type = PL_FMT_UNORM; break;
    default:                        fmt->type = PL_FMT_UNKNOWN; break;
    }

    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, obj,
            GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE, &fmt->component_depth[0]);
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, obj,
            GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE, &fmt->component_depth[1]);
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, obj,
            GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE, &fmt->component_depth[2]);
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, obj,
            GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE, &fmt->component_depth[3]);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    int bits = 0;
    for (int i = 0; i < 4; i++)
        bits += fmt->component_depth[i];
    fmt->internal_size = (bits + 7) / 8;

    return gl_check_err(gpu, "gl_fb_query");
}

static const struct gl_format fbo_dummy_format = {
    .fmt = GL_RGBA,
    .caps = PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLITTABLE,
};

const struct pl_tex *pl_opengl_wrap_fb(const struct pl_gpu *gpu, GLuint fbo,
                                       int w, int h)
{
    struct pl_tex *tex = talloc_priv(NULL, struct pl_tex, struct pl_tex_gl);
    struct pl_tex_gl *tex_gl = TA_PRIV(tex);

    struct pl_fmt *fmt = talloc_priv(tex, struct pl_fmt, const struct gl_format *);
    const struct gl_format **glfmtp = TA_PRIV(fmt);
    *glfmtp = &fbo_dummy_format;

    *fmt = (struct pl_fmt) {
        .name = "fbo",
        .type = PL_FMT_UNKNOWN,
        .caps = PL_FMT_CAP_RENDERABLE,
        .num_components = 4,
        .opaque = true,
    };

    if (!gl_fb_query(gpu, fbo, fmt)) {
        PL_ERR(gpu, "Failed querying framebuffer specifics!");
        goto error;
    }

    *tex = (struct pl_tex) {
        .params = {
            .w = w,
            .h = h,
            .format = fmt,
            .renderable = true,
            .blit_dst = true,
        },
    };

    *tex_gl = (struct pl_tex_gl) {
        .fbo = fbo,
        .format = GL_RGBA,
        .iformat = 0,
        .type = 0,
        .wrapped_fb = true,
    };

    return tex;

error:
    talloc_free(tex);
    return NULL;
}

static void gl_tex_invalidate(const struct pl_gpu *gpu, const struct pl_tex *tex)
{
    struct pl_gl *p = TA_PRIV(gpu);
    struct pl_tex_gl *tex_gl = TA_PRIV(tex);

    if (!p->has_invalidate)
        return;

    if (tex_gl->wrapped_fb) {
        glBindFramebuffer(GL_FRAMEBUFFER, tex_gl->fbo);
        glInvalidateFramebuffer(GL_FRAMEBUFFER, 1, (GLenum[]){GL_COLOR});
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    } else {
        glInvalidateTexImage(tex_gl->texture, 0);
    }

    gl_check_err(gpu, "gl_tex_invalidate");
}

static void gl_tex_clear(const struct pl_gpu *gpu, const struct pl_tex *tex,
                         const float color[4])
{
    struct pl_tex_gl *tex_gl = TA_PRIV(tex);
    pl_assert(tex_gl->fbo || tex_gl->wrapped_fb);

    glBindFramebuffer(GL_FRAMEBUFFER, tex_gl->fbo);
    glClearColor(color[0], color[1], color[2], color[3]);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    gl_check_err(gpu, "gl_tex_clear");
}

static void gl_tex_blit(const struct pl_gpu *gpu,
                        const struct pl_tex *dst, const struct pl_tex *src,
                        struct pl_rect3d dst_rc, struct pl_rect3d src_rc)
{
    struct pl_tex_gl *src_gl = TA_PRIV(src);
    struct pl_tex_gl *dst_gl = TA_PRIV(dst);

    pl_assert(src_gl->fbo || src_gl->wrapped_fb);
    pl_assert(dst_gl->fbo || dst_gl->wrapped_fb);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, src_gl->fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst_gl->fbo);

    glBlitFramebuffer(src_rc.x0, src_rc.y0, src_rc.x1, src_rc.y1,
                      dst_rc.x0, dst_rc.y0, dst_rc.x1, dst_rc.y1,
                      GL_COLOR_BUFFER_BIT, src_gl->filter);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    gl_check_err(gpu, "gl_tex_blit");
}

// For pl_buf.priv
struct pl_buf_gl {
    GLenum target;
    GLuint buffer;
    GLsync fence;
};

static void gl_buf_destroy(const struct pl_gpu *gpu, const struct pl_buf *buf)
{
    struct pl_buf_gl *buf_gl = TA_PRIV(buf);
    if (buf_gl->fence)
        glDeleteSync(buf_gl->fence);

    if (buf->data) {
        glBindBuffer(buf_gl->target, buf_gl->buffer);
        glUnmapBuffer(buf_gl->target);
        glBindBuffer(buf_gl->target, 0);
    }

    glDeleteBuffers(1, &buf_gl->buffer);
    talloc_free((void *) buf);
    gl_check_err(gpu, "gl_buf_destroy");
}

static const struct pl_buf *gl_buf_create(const struct pl_gpu *gpu,
                                          const struct pl_buf_params *params)
{
    struct pl_buf *buf = talloc_zero_priv(NULL, struct pl_buf, struct pl_buf_gl);
    buf->params = *params;
    buf->params.initial_data = NULL;

    struct pl_buf_gl *buf_gl = TA_PRIV(buf);
    glGenBuffers(1, &buf_gl->buffer);

    static const GLenum targets[PL_BUF_TYPE_COUNT] = {
        [PL_BUF_TEX_TRANSFER]   = GL_PIXEL_UNPACK_BUFFER,
        [PL_BUF_UNIFORM]        = GL_UNIFORM_BUFFER,
        [PL_BUF_STORAGE]        = GL_SHADER_STORAGE_BUFFER,
        // TODO: texel buffers
    };

    pl_assert(params->type < PL_BUF_PRIVATE);
    buf_gl->target = targets[params->type];

    glBindBuffer(buf_gl->target, buf_gl->buffer);

    if (test_ext(gpu, "GL_ARB_buffer_storage", 44, 0)) {
        GLbitfield mapflags = 0, storflags = 0;
        if (params->host_writable)
            storflags |= GL_DYNAMIC_STORAGE_BIT;
        if (params->host_mapped) {
            mapflags |= GL_MAP_READ_BIT | GL_MAP_WRITE_BIT |
                        GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
        }
        if (params->memory_type == PL_BUF_MEM_HOST)
            storflags |= GL_CLIENT_STORAGE_BIT; // hopefully this works

        glBufferStorage(buf_gl->target, params->size, params->initial_data,
                        storflags | mapflags);

        if (params->host_mapped) {
            buf->data = glMapBufferRange(buf_gl->target, 0, params->size,
                                         mapflags);
            if (!buf->data) {
                glBindBuffer(buf_gl->target, 0);
                if (!gl_check_err(gpu, "gl_buf_create: map"))
                    PL_ERR(gpu, "Failed mapping buffer: unknown reason");
                goto error;
            }
        }
    } else {
        // Make a random guess based on arbitrary criteria we can't know
        static const GLenum hints[PL_BUF_TYPE_COUNT] = {
            [PL_BUF_TEX_TRANSFER] = GL_STREAM_DRAW,
            [PL_BUF_UNIFORM]      = GL_STATIC_DRAW,
            [PL_BUF_STORAGE]      = GL_DYNAMIC_COPY,
            // TODO: texel buffers
        };

        GLenum hint = hints[params->type];
        if (params->type == PL_BUF_TEX_TRANSFER &&
            params->memory_type == PL_BUF_MEM_DEVICE)
        {
            // This might be a texture download buffer?
            hint = GL_STREAM_READ;
        }

        glBufferData(buf_gl->target, params->size, params->initial_data, hint);
    }

    glBindBuffer(buf_gl->target, 0);
    if (!gl_check_err(gpu, "gl_buf_create"))
        goto error;

    return buf;

error:
    gl_buf_destroy(gpu, buf);
    return NULL;
}

static void gl_buf_write(const struct pl_gpu *gpu, const struct pl_buf *buf,
                         size_t offset, const void *data, size_t size)
{
    if (buf->data) {
        memcpy(buf->data + offset, data, size);
    } else {
        struct pl_buf_gl *buf_gl = TA_PRIV(buf);
        glBindBuffer(buf_gl->target, buf_gl->buffer);
        glBufferSubData(buf_gl->target, offset, size, data);
        glBindBuffer(buf_gl->target, 0);
        gl_check_err(gpu, "gl_buf_write");
    }
}

static bool gl_buf_read(const struct pl_gpu *gpu, const struct pl_buf *buf,
                        size_t offset, void *dest, size_t size)
{
    if (buf->data) {
        memcpy(dest, buf->data + offset, size);
        return true;
    } else {
        struct pl_buf_gl *buf_gl = TA_PRIV(buf);
        glBindBuffer(buf_gl->target, buf_gl->buffer);
        glGetBufferSubData(buf_gl->target, offset, size, dest);
        glBindBuffer(buf_gl->target, 0);
        return gl_check_err(gpu, "gl_buf_read");
    }
}

static bool gl_buf_poll(const struct pl_gpu *gpu, const struct pl_buf *buf,
                        uint64_t timeout)
{
    // Non-persistently mapped buffers are always implicitly reusable in OpenGL,
    // the implementation will create more buffers under the hood if needed.
    if (!buf->data)
        return false;

    struct pl_buf_gl *buf_gl = TA_PRIV(buf);
    if (buf_gl->fence) {
        GLenum res = glClientWaitSync(buf_gl->fence,
                                      timeout ? GL_SYNC_FLUSH_COMMANDS_BIT : 0,
                                      timeout);
        if (res == GL_ALREADY_SIGNALED || res == GL_CONDITION_SATISFIED) {
            glDeleteSync(buf_gl->fence);
            buf_gl->fence = NULL;
        }
    }

    return !!buf_gl->fence;
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

static bool gl_tex_upload(const struct pl_gpu *gpu,
                          const struct pl_tex_transfer_params *params)
{
    struct pl_gl *p = TA_PRIV(gpu);
    const struct pl_tex *tex = params->tex;
    const struct pl_buf *buf = params->buf;
    struct pl_tex_gl *tex_gl = TA_PRIV(tex);
    struct pl_buf_gl *buf_gl = buf ? TA_PRIV(buf) : NULL;

    const void *src = params->ptr;
    if (buf) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf_gl->buffer);
        src = (void *) params->buf_offset;
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

    return gl_check_err(gpu, "gl_tex_upload");
}

static bool gl_tex_download(const struct pl_gpu *gpu,
                            const struct pl_tex_transfer_params *params)
{
    struct pl_gl *p = TA_PRIV(gpu);
    const struct pl_tex *tex = params->tex;
    const struct pl_buf *buf = params->buf;
    struct pl_tex_gl *tex_gl = TA_PRIV(tex);
    struct pl_buf_gl *buf_gl = buf ? TA_PRIV(buf) : NULL;
    bool ok = true;

    void *dst = params->ptr;
    if (buf) {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, buf_gl->buffer);
        dst = (void *) params->buf_offset;
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

        glBindFramebuffer(GL_FRAMEBUFFER, tex_gl->fbo);
        for (int y = params->rc.y0; y < params->rc.y1; y += rows) {
            glReadPixels(params->rc.x0, y, pl_rect_w(params->rc), rows,
                         tex_gl->format, tex_gl->type, dst);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
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

    if (buf) {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        if (ok && buf->params.host_mapped) {
            glDeleteSync(buf_gl->fence);
            buf_gl->fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        }
    }

    return gl_check_err(gpu, "gl_tex_download") && ok;
}

static int gl_desc_namespace(const struct pl_gpu *gpu, enum pl_desc_type type)
{
    return (int) type;
}

#define GL_CACHE_MAGIC {'P','L','G','L'}
#define GL_CACHE_VERSION 1
static const char gl_cache_magic[4] = GL_CACHE_MAGIC;

struct gl_cache_header {
    char magic[sizeof(gl_cache_magic)];
    int cache_version;
    GLenum format;
};

static GLuint load_cached_program(const struct pl_gpu *gpu,
                                  const struct pl_pass_params *params)
{
    if (!test_ext(gpu, "GL_ARB_get_program_binary", 41, 30))
        return 0;

    struct bstr cache = {
        .start = (void *) params->cached_program,
        .len = params->cached_program_len,
    };

    if (cache.len < sizeof(struct gl_cache_header))
        return false;

    struct gl_cache_header *header = (struct gl_cache_header *) cache.start;
    cache = bstr_cut(cache, sizeof(*header));

    if (strncmp(header->magic, gl_cache_magic, sizeof(gl_cache_magic)) != 0)
        return 0;
    if (header->cache_version != GL_CACHE_VERSION)
        return 0;

    GLuint prog = glCreateProgram();
    if (!gl_check_err(gpu, "load_cached_program: glCreateProgram"))
        return 0;

    glProgramBinary(prog, header->format, cache.start, cache.len);
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

static bool gl_attach_shader(const struct pl_gpu *gpu, GLuint program, GLenum type,
                             const char *src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);

    GLint status = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    GLint log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);

    enum pl_log_level level = gl_log_level(status, log_length);
    if (pl_msg_test(gpu->ctx, level)) {
        static const char *shader_name;
        switch (type) {
        case GL_VERTEX_SHADER:   shader_name = "vertex"; break;
        case GL_FRAGMENT_SHADER: shader_name = "fragment"; break;
        case GL_COMPUTE_SHADER:  shader_name = "compute"; break;
        default: abort();
        };

        PL_MSG(gpu, level, "%s shader source:", shader_name);
        pl_msg_source(gpu->ctx, level, src);

        GLchar *logstr = talloc_zero_size(NULL, log_length + 1);
        glGetShaderInfoLog(shader, log_length, NULL, logstr);
        PL_MSG(gpu, level, "shader compile log (status=%d): %s", status, logstr);
        talloc_free(logstr);
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

static GLuint gl_compile_program(const struct pl_gpu *gpu,
                                 const struct pl_pass_params *params)
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
    default: abort();
    }

    if (!ok || !gl_check_err(gpu, "gl_compile_program: attach shader"))
        goto error;

    glLinkProgram(prog);
    GLint status = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &status);
    GLint log_length = 0;
    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &log_length);

    enum pl_log_level level = gl_log_level(status, log_length);
    if (pl_msg_test(gpu->ctx, level)) {
        GLchar *logstr = talloc_zero_size(NULL, log_length + 1);
        glGetProgramInfoLog(prog, log_length, NULL, logstr);
        PL_MSG(gpu, level, "shader link log (status=%d): %s", status, logstr);
        talloc_free(logstr);
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
    GLuint vao;     // the VAO object
    GLuint buffer;  // GL_ARRAY_BUFFER for vao
    GLint *var_locs;
};

static void gl_pass_destroy(const struct pl_gpu *gpu, const struct pl_pass *pass)
{
    struct pl_pass_gl *pass_gl = TA_PRIV(pass);

    if (pass_gl->vao)
        glDeleteVertexArrays(1, &pass_gl->vao);
    glDeleteBuffers(1, &pass_gl->buffer);
    glDeleteProgram(pass_gl->program);

    talloc_free((void *) pass);
}

static void gl_enable_vao_attribs(const struct pl_pass *pass)
{
    for (int i = 0; i < pass->params.num_vertex_attribs; i++) {
        const struct pl_vertex_attrib *va = &pass->params.vertex_attribs[i];
        const struct gl_format **glfmtp = TA_PRIV(va->fmt);
        const struct gl_format *glfmt = *glfmtp;

        bool norm = false;
        switch (va->fmt->type) {
        case PL_FMT_UNORM:
        case PL_FMT_SNORM:
            norm = true;
        default: break;
        }

        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, va->fmt->num_components, glfmt->type, norm,
                              pass->params.vertex_stride, (void *) va->offset);
    }
}

static const struct pl_pass *gl_pass_create(const struct pl_gpu *gpu,
                                            const struct pl_pass_params *params)
{
    struct pl_gl *p = TA_PRIV(gpu);
    struct pl_pass *pass = talloc_zero_priv(NULL, struct pl_pass, struct pl_pass_gl);
    struct pl_pass_gl *pass_gl = TA_PRIV(pass);
    pass->params = pl_pass_params_copy(pass, params);

    // Load/Compile program
    if ((pass_gl->program = load_cached_program(gpu, params))) {
        PL_DEBUG(gpu, "Using cached GL program");
    } else {
        pass_gl->program = gl_compile_program(gpu, params);
    }

    if (!pass_gl->program)
        goto error;

    // Update program cache if possible
    if (test_ext(gpu, "GL_ARB_get_program_binary", 41, 30)) {
        GLint size = 0;
        glGetProgramiv(pass_gl->program, GL_PROGRAM_BINARY_LENGTH, &size);

        if (size > 0) {
            uint8_t *buffer = talloc_size(NULL, size);
            GLsizei actual_size = 0;
            struct gl_cache_header header = {
                .magic = GL_CACHE_MAGIC,
                .cache_version = GL_CACHE_VERSION,
            };

            glGetProgramBinary(pass_gl->program, size, &actual_size,
                               &header.format, buffer);
            if (actual_size > 0) {
                struct bstr cache = {0};
                bstr_xappend(pass, &cache, (struct bstr) { (char *) &header,
                                                           sizeof(header) });
                bstr_xappend(pass, &cache, (struct bstr) { buffer, actual_size });
                pass->params.cached_program = cache.start;
                pass->params.cached_program_len = cache.len;
            }

            talloc_free(buffer);
        }
    }

    glUseProgram(pass_gl->program);
    pass_gl->var_locs = talloc_zero_array(pass, GLint, params->num_variables);

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

    // Initialize VAO / vertex buffer
    glGenBuffers(1, &pass_gl->buffer);
    if (p->has_vao) {
        glBindBuffer(GL_ARRAY_BUFFER, pass_gl->buffer);
        glGenVertexArrays(1, &pass_gl->vao);
        glBindVertexArray(pass_gl->vao);
        gl_enable_vao_attribs(pass);
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    return pass;

error:
    PL_ERR(gpu, "Failed creating pass");
    gl_pass_destroy(gpu, pass);
    return NULL;
}

static void update_var(const struct pl_pass *pass,
                       const struct pl_var_update *vu)
{
    struct pl_pass_gl *pass_gl = TA_PRIV(pass);
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
        default: abort();
        }
        break;
    }
    case PL_VAR_UINT: {
        const unsigned int *u = vu->data;
        pl_assert(var->dim_m == 1);
        switch (var->dim_v) {
        case 1: glUniform1uiv(loc, var->dim_a, u); break;
        case 2: glUniform2uiv(loc, var->dim_a, u); break;
        case 3: glUniform3uiv(loc, var->dim_a, u); break;
        case 4: glUniform4uiv(loc, var->dim_a, u); break;
        default: abort();
        }
        break;
    }
    case PL_VAR_FLOAT: {
        const float *f = vu->data;
        if (var->dim_m == 1) {
            switch (var->dim_v) {
            case 1: glUniform1fv(loc, var->dim_a, f); break;
            case 2: glUniform2fv(loc, var->dim_a, f); break;
            case 3: glUniform3fv(loc, var->dim_a, f); break;
            case 4: glUniform4fv(loc, var->dim_a, f); break;
            default: abort();
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
            abort(); // unreachable
        }
        break;
    }
    default: abort();
    }
}

static void update_desc(const struct pl_pass *pass, int index,
                        const struct pl_desc_binding *db)
{
    const struct pl_desc *desc = &pass->params.descriptors[index];

    static const GLenum access[] = {
        [PL_DESC_ACCESS_READWRITE] = GL_READ_WRITE,
        [PL_DESC_ACCESS_READONLY]  = GL_READ_ONLY,
        [PL_DESC_ACCESS_WRITEONLY] = GL_WRITE_ONLY,
    };

    switch (desc->type) {
    case PL_DESC_SAMPLED_TEX: {
        const struct pl_tex *tex = db->object;
        struct pl_tex_gl *tex_gl = TA_PRIV(tex);
        glActiveTexture(GL_TEXTURE0 + desc->binding);
        glBindTexture(tex_gl->target, tex_gl->texture);
        break;
    }
    case PL_DESC_STORAGE_IMG: {
        const struct pl_tex *tex = db->object;
        struct pl_tex_gl *tex_gl = TA_PRIV(tex);
        glBindImageTexture(desc->binding, tex_gl->texture, 0, GL_FALSE, 0,
                           access[desc->access], tex_gl->iformat);
        break;
    }
    case PL_DESC_BUF_UNIFORM: {
        const struct pl_buf *buf = db->object;
        struct pl_buf_gl *buf_gl = TA_PRIV(buf);
        glBindBufferBase(buf_gl->target, desc->binding, buf_gl->buffer);
        break;
    }
    case PL_DESC_BUF_STORAGE: {
        const struct pl_buf *buf = db->object;
        struct pl_buf_gl *buf_gl = TA_PRIV(buf);
        glBindBufferBase(buf_gl->target, desc->binding, buf_gl->buffer);
        // SSBOs are not implicitly coherent in OpenGL
        glMemoryBarrier(buf_gl->target);
        break;
    }
    case PL_DESC_BUF_TEXEL_UNIFORM:
    case PL_DESC_BUF_TEXEL_STORAGE:
        abort(); // TODO
    default: abort();
    }
}

static void unbind_desc(const struct pl_pass *pass, int index,
                        const struct pl_desc_binding *db)
{
    const struct pl_desc *desc = &pass->params.descriptors[index];

    switch (desc->type) {
    case PL_DESC_SAMPLED_TEX: {
        const struct pl_tex *tex = db->object;
        struct pl_tex_gl *tex_gl = TA_PRIV(tex);
        glActiveTexture(GL_TEXTURE0 + desc->binding);
        glBindTexture(tex_gl->target, 0);
        break;
    }
    case PL_DESC_STORAGE_IMG:
        glBindImageTexture(desc->binding, 0, 0, GL_FALSE, 0,
                           GL_WRITE_ONLY, GL_R32F);
        break;
    case PL_DESC_BUF_UNIFORM:
    case PL_DESC_BUF_STORAGE: {
        const struct pl_buf *buf = db->object;
        struct pl_buf_gl *buf_gl = TA_PRIV(buf);
        glBindBufferBase(buf_gl->target, desc->binding, 0);
        break;
    }
    case PL_DESC_BUF_TEXEL_UNIFORM:
    case PL_DESC_BUF_TEXEL_STORAGE:
        abort(); // TODO
    default: abort();
    }
}

static void gl_pass_run(const struct pl_gpu *gpu,
                        const struct pl_pass_run_params *params)
{
    const struct pl_pass *pass = params->pass;
    struct pl_pass_gl *pass_gl = TA_PRIV(pass);
    struct pl_gl *p = TA_PRIV(gpu);

    glUseProgram(pass_gl->program);

    for (int i = 0; i < params->num_var_updates; i++)
        update_var(pass, &params->var_updates[i]);
    for (int i = 0; i < pass->params.num_descriptors; i++)
        update_desc(pass, i, &params->desc_bindings[i]);
    glActiveTexture(GL_TEXTURE0);

    if (!gl_check_err(gpu, "gl_pass_run: updating uniforms"))
        return;

    switch (pass->params.type) {
    case PL_PASS_RASTER: {
        struct pl_tex_gl *target_gl = TA_PRIV(params->target);
        glBindFramebuffer(GL_FRAMEBUFFER, target_gl->fbo);
        if (!pass->params.load_target && p->has_invalidate) {
            GLenum fb = target_gl->fbo ? GL_COLOR_ATTACHMENT0 : GL_COLOR;
            glInvalidateFramebuffer(GL_FRAMEBUFFER, 1, &fb);
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

        // Update vertex buffer and bind VAO
        size_t vertex_size = params->vertex_count * pass->params.vertex_stride;
        glBindBuffer(GL_ARRAY_BUFFER, pass_gl->buffer);
        glBufferData(GL_ARRAY_BUFFER, vertex_size, params->vertex_data, GL_STREAM_DRAW);

        if (pass_gl->vao) {
            glBindVertexArray(pass_gl->vao);
        } else {
            gl_enable_vao_attribs(pass);
        }

        gl_check_err(gpu, "gl_pass_run: update/bind vertex buffer");

        static const GLenum map_prim[] = {
            [PL_PRIM_TRIANGLE_LIST]     = GL_TRIANGLES,
            [PL_PRIM_TRIANGLE_STRIP]    = GL_TRIANGLE_STRIP,
            [PL_PRIM_TRIANGLE_FAN]      = GL_TRIANGLE_FAN,
        };

        glDrawArrays(map_prim[pass->params.vertex_type], 0, params->vertex_count);
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
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        break;
    }

    case PL_PASS_COMPUTE:
        glDispatchCompute(params->compute_groups[0],
                          params->compute_groups[1],
                          params->compute_groups[2]);

        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
        break;

    default: abort();
    }

    for (int i = 0; i < pass->params.num_descriptors; i++)
        unbind_desc(pass, i, &params->desc_bindings[i]);
    glActiveTexture(GL_TEXTURE0);

    glUseProgram(0);
    gl_check_err(gpu, "gl_pass_run");
}

static void gl_gpu_flush(const struct pl_gpu *gpu)
{
    glFlush();
    gl_check_err(gpu, "gl_gpu_flush");
}

static void gl_gpu_finish(const struct pl_gpu *gpu)
{
    glFinish();
    gl_check_err(gpu, "gl_gpu_finish");
}

static const struct pl_gpu_fns pl_fns_gl = {
    .destroy                = gl_destroy_gpu,
    .tex_create             = gl_tex_create,
    .tex_destroy            = gl_tex_destroy,
    .tex_invalidate         = gl_tex_invalidate,
    .tex_clear              = gl_tex_clear,
    .tex_blit               = gl_tex_blit,
    .tex_upload             = gl_tex_upload,
    .tex_download           = gl_tex_download,
    .buf_create             = gl_buf_create,
    .buf_destroy            = gl_buf_destroy,
    .buf_write              = gl_buf_write,
    .buf_read               = gl_buf_read,
    .buf_poll               = gl_buf_poll,
    .desc_namespace         = gl_desc_namespace,
    .pass_create            = gl_pass_create,
    .pass_destroy           = gl_pass_destroy,
    .pass_run               = gl_pass_run,
    .gpu_flush              = gl_gpu_flush,
    .gpu_finish             = gl_gpu_finish,
};
