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
 * License along with libplacebo.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "gpu.h"
#include "common.h"
#include "formats.h"
#include "utils.h"

#ifdef PL_HAVE_UNIX
static bool supported_fourcc(struct pl_gl *p, EGLint fourcc)
{
    for (int i = 0; i < p->egl_formats.num; ++i)
        if (fourcc == p->egl_formats.elem[i])
            return true;
    return false;
}
#endif

#define FMT(_name, bits, ftype, _caps)               \
    (struct pl_fmt_t) {                              \
        .name = _name,                               \
        .type = PL_FMT_##ftype,                      \
        .caps = (enum pl_fmt_caps) (_caps),          \
        .sample_order = {0, 1, 2, 3},                \
        .component_depth = {bits, bits, bits, bits}, \
    }

// Convenience to make the names simpler
enum {
    // Type aliases
    U8    = GL_UNSIGNED_BYTE,
    U16   = GL_UNSIGNED_SHORT,
    U32   = GL_UNSIGNED_INT,
    I8    = GL_BYTE,
    I16   = GL_SHORT,
    I32   = GL_INT,
    FLT   = GL_FLOAT,
    HALF  = GL_HALF_FLOAT,

    // Component aliases
    R     = GL_RED,
    RG    = GL_RG,
    RGB   = GL_RGB,
    RGBA  = GL_RGBA,
    BGRA  = GL_BGRA,
    RI    = GL_RED_INTEGER,
    RGI   = GL_RG_INTEGER,
    RGBI  = GL_RGB_INTEGER,
    RGBAI = GL_RGBA_INTEGER,

    // Capability aliases
    S     = PL_FMT_CAP_SAMPLEABLE,
    L     = PL_FMT_CAP_LINEAR,
    F     = PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLITTABLE, // FBO support
    V     = PL_FMT_CAP_VERTEX,
};

// Basic 8-bit formats
const struct gl_format formats_norm8[] = {
    {GL_R8,             R,     U8,  FMT("r8",       8, UNORM, S|L|F|V)},
    {GL_RG8,            RG,    U8,  FMT("rg8",      8, UNORM, S|L|F|V)},
    {GL_RGB8,           RGB,   U8,  FMT("rgb8",     8, UNORM, S|L|F|V)},
    {GL_RGBA8,          RGBA,  U8,  FMT("rgba8",    8, UNORM, S|L|F|V)},
};

// Signed variants
/* TODO: these are broken in mesa
const struct gl_format formats_snorm8[] = {
    {GL_R8_SNORM,       R,     I8,  FMT("r8s",      8, SNORM, S|L|F|V)},
    {GL_RG8_SNORM,      RG,    I8,  FMT("rg8s",     8, SNORM, S|L|F|V)},
    {GL_RGB8_SNORM,     RGB,   I8,  FMT("rgb8s",    8, SNORM, S|L|F|V)},
    {GL_RGBA8_SNORM,    RGBA,  I8,  FMT("rgba8s",   8, SNORM, S|L|F|V)},
};
*/

// BGRA 8-bit
const struct gl_format formats_bgra8[] = {
    {GL_RGBA8,          BGRA,  U8,  {
        .name               = "bgra8",
        .type               = PL_FMT_UNORM,
        .caps               = S|L|F|V,
        .sample_order       = {2, 1, 0, 3},
        .component_depth    = {8, 8, 8, 8},
    }},
};

// Basic 16-bit formats, excluding rgb16 (special cased below)
const struct gl_format formats_norm16[] = {
    {GL_R16,            R,     U16, FMT("r16",     16, UNORM, S|L|F|V)},
    {GL_RG16,           RG,    U16, FMT("rg16",    16, UNORM, S|L|F|V)},
    {GL_RGBA16,         RGBA,  U16, FMT("rgba16",  16, UNORM, S|L|F|V)},
};

// Renderable version of rgb16
const struct gl_format formats_rgb16_fbo[] = {
    {GL_RGB16,          RGB,   U16, FMT("rgb16",   16, UNORM, S|L|F|V)},
};

// Non-renderable version of rgb16
const struct gl_format formats_rgb16_fallback[] = {
    {GL_RGB16,          RGB,   U16, FMT("rgb16",   16, UNORM, S|L|V)},
};

// Signed 16-bit variants
/* TODO: these are broken in mesa and nvidia
const struct gl_format formats_snorm16[] = {
    {GL_R16_SNORM,      R,     I16, FMT("r16s",    16, SNORM, S|L|F|V)},
    {GL_RG16_SNORM,     RG,    I16, FMT("rg16s",   16, SNORM, S|L|F|V)},
    {GL_RGB16_SNORM,    RGB,   I16, FMT("rgb16s",  16, SNORM, S|L|F|V)},
    {GL_RGBA16_SNORM,   RGBA,  I16, FMT("rgba16s", 16, SNORM, S|L|F|V)},
};
*/

// 32-bit floating point texture formats
const struct gl_format formats_float32[] = {
    {GL_R32F,           R,     FLT, FMT("r32f",    32, FLOAT, S|L|F|V)},
    {GL_RG32F,          RG,    FLT, FMT("rg32f",   32, FLOAT, S|L|F|V)},
    {GL_RGB32F,         RGB,   FLT, FMT("rgb32f",  32, FLOAT, S|L|F|V)},
    {GL_RGBA32F,        RGBA,  FLT, FMT("rgba32f", 32, FLOAT, S|L|F|V)},
};

// 16-bit floating point texture formats
const struct gl_format formats_float16[] = {
    {GL_R16F,           R,     FLT,  FMT("r16f",    16, FLOAT, S|L|F)},
    {GL_RG16F,          RG,    FLT,  FMT("rg16f",   16, FLOAT, S|L|F)},
    {GL_RGB16F,         RGB,   FLT,  FMT("rgb16f",  16, FLOAT, S|L|F)},
    {GL_RGBA16F,        RGBA,  FLT,  FMT("rgba16f", 16, FLOAT, S|L|F)},
};

// 16-bit half float texture formats
const struct gl_format formats_half16[] = {
    {GL_R16F,           R,     HALF, FMT("r16hf",   16, FLOAT, S|L|F)},
    {GL_RG16F,          RG,    HALF, FMT("rg16hf",  16, FLOAT, S|L|F)},
    {GL_RGB16F,         RGB,   HALF, FMT("rgb16hf", 16, FLOAT, S|L|F)},
    {GL_RGBA16F,        RGBA,  HALF, FMT("rgba16hf",16, FLOAT, S|L|F)},
};

// Renderable 16-bit float formats (excluding rgb16f)
const struct gl_format formats_float16_fbo[] = {
    {GL_R16F,           R,     HALF, FMT("r16hf",   16, FLOAT, S|L|F)},
    {GL_RG16F,          RG,    HALF, FMT("rg16hf",  16, FLOAT, S|L|F)},
    {GL_RGB16F,         RGB,   HALF, FMT("rgb16hf", 16, FLOAT, S|L)},
    {GL_RGBA16F,        RGBA,  HALF, FMT("rgba16hf",16, FLOAT, S|L|F)},
    {GL_R16F,           R,     FLT,  FMT("r16f",    16, FLOAT, S|L|F)},
    {GL_RG16F,          RG,    FLT,  FMT("rg16f",   16, FLOAT, S|L|F)},
    {GL_RGB16F,         RGB,   FLT,  FMT("rgb16f",  16, FLOAT, S|L)},
    {GL_RGBA16F,        RGBA,  FLT,  FMT("rgba16f", 16, FLOAT, S|L|F)},
};

// Non-renderable 16-bit float formats
const struct gl_format formats_float16_fallback[] = {
    {GL_R16F,           R,     HALF, FMT("r16hf",   16, FLOAT, S|L)},
    {GL_RG16F,          RG,    HALF, FMT("rg16hf",  16, FLOAT, S|L)},
    {GL_RGB16F,         RGB,   HALF, FMT("rgb16hf", 16, FLOAT, S|L)},
    {GL_RGBA16F,        RGBA,  HALF, FMT("rgba16hf",16, FLOAT, S|L)},
    {GL_R16F,           R,     FLT,  FMT("r16f",    16, FLOAT, S|L)},
    {GL_RG16F,          RG,    FLT,  FMT("rg16f",   16, FLOAT, S|L)},
    {GL_RGB16F,         RGB,   FLT,  FMT("rgb16f",  16, FLOAT, S|L)},
    {GL_RGBA16F,        RGBA,  FLT,  FMT("rgba16f", 16, FLOAT, S|L)},
};

// (Unsigned) integer formats
const struct gl_format formats_uint[] = {
    {GL_R8UI,           RI,    U8,  FMT("r8u",      8, UINT, S|F|V)},
    {GL_RG8UI,          RGI,   U8,  FMT("rg8u",     8, UINT, S|F|V)},
    {GL_RGB8UI,         RGBI,  U8,  FMT("rgb8u",    8, UINT, S|V)},
    {GL_RGBA8UI,        RGBAI, U8,  FMT("rgba8u",   8, UINT, S|F|V)},
    {GL_R16UI,          RI,    U16, FMT("r16u",    16, UINT, S|F|V)},
    {GL_RG16UI,         RGI,   U16, FMT("rg16u",   16, UINT, S|F|V)},
    {GL_RGB16UI,        RGBI,  U16, FMT("rgb16u",  16, UINT, S|V)},
    {GL_RGBA16UI,       RGBAI, U16, FMT("rgba16u", 16, UINT, S|F|V)},
};

/* TODO
    {GL_R32UI,          RI,    U32, FMT("r32u",    32, UINT)},
    {GL_RG32UI,         RGI,   U32, FMT("rg32u",   32, UINT)},
    {GL_RGB32UI,        RGBI,  U32, FMT("rgb32u",  32, UINT)},
    {GL_RGBA32UI,       RGBAI, U32, FMT("rgba32u", 32, UINT)},

    {GL_R8I,            RI,    I8,  FMT("r8i",      8, SINT)},
    {GL_RG8I,           RGI,   I8,  FMT("rg8i",     8, SINT)},
    {GL_RGB8I,          RGBI,  I8,  FMT("rgb8i",    8, SINT)},
    {GL_RGBA8I,         RGBAI, I8,  FMT("rgba8i",   8, SINT)},
    {GL_R16I,           RI,    I16, FMT("r16i",    16, SINT)},
    {GL_RG16I,          RGI,   I16, FMT("rg16i",   16, SINT)},
    {GL_RGB16I,         RGBI,  I16, FMT("rgb16i",  16, SINT)},
    {GL_RGBA16I,        RGBAI, I16, FMT("rgba16i", 16, SINT)},
    {GL_R32I,           RI,    I32, FMT("r32i",    32, SINT)},
    {GL_RG32I,          RGI,   I32, FMT("rg32i",   32, SINT)},
    {GL_RGB32I,         RGBI,  I32, FMT("rgb32i",  32, SINT)},
    {GL_RGBA32I,        RGBAI, I32, FMT("rgba32i", 32, SINT)},
*/

// GL2 legacy formats
const struct gl_format formats_legacy_gl2[] = {
    {GL_RGB8,           RGB,   U8,  FMT("rgb8",     8, UNORM, S|L|V)},
    {GL_RGBA8,          RGBA,  U8,  FMT("rgba8",    8, UNORM, S|L|V)},
    {GL_RGB16,          RGB,   U16, FMT("rgb16",   16, UNORM, S|L|V)},
    {GL_RGBA16,         RGBA,  U16, FMT("rgba16",  16, UNORM, S|L|V)},
};

// GLES2 legacy formats
const struct gl_format formats_legacy_gles2[] = {
    {GL_RGB,            RGB,   U8,  FMT("rgb",      8, UNORM, S|L)},
    {GL_RGBA,           RGBA,  U8,  FMT("rgba",     8, UNORM, S|L)},
};

// GLES BGRA
const struct gl_format formats_bgra_gles[] = {
    {GL_BGRA,           BGRA,  U8,  {
        .name               = "bgra8",
        .type               = PL_FMT_UNORM,
        .caps               = S|L|F|V,
        .sample_order       = {2, 1, 0, 3},
        .component_depth    = {8, 8, 8, 8},
    }},
};

// Fallback for vertex-only formats, as a last resort
const struct gl_format formats_basic_vertex[] = {
    {GL_R32F,           R,     FLT, FMT("r32f",    32, FLOAT, V)},
    {GL_RG32F,          RG,    FLT, FMT("rg32f",   32, FLOAT, V)},
    {GL_RGB32F,         RGB,   FLT, FMT("rgb32f",  32, FLOAT, V)},
    {GL_RGBA32F,        RGBA,  FLT, FMT("rgba32f", 32, FLOAT, V)},
};

static void add_format(pl_gpu pgpu, const struct gl_format *gl_fmt)
{
    struct pl_gpu_t *gpu = (struct pl_gpu_t *) pgpu;
    struct pl_gl *p = PL_PRIV(gpu);

    struct pl_fmt_t *fmt = pl_alloc_obj(gpu, fmt, gl_fmt);
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
    case GL_BGRA:
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
    case GL_HALF_FLOAT:
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
    fmt->texel_align = 1;
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
    if (p->has_modifiers && fmt->fourcc && supported_fourcc(p, fmt->fourcc)) {
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
            } else {
                pl_free(mods);
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

    bool host_readable = false;
    if (p->gl_ver && p->has_readback)
        host_readable = true;
    // Reading from textures on GLES requires FBO support for this fmt
    if (fmt->caps & PL_FMT_CAP_RENDERABLE) {
        // this combination always works in glReadPixels
        if ((gl_fmt->fmt == GL_RGBA && gl_fmt->type == GL_UNSIGNED_BYTE) ||
            p->has_readback)
            host_readable = true;
    }
    if (host_readable)
        fmt->caps |= PL_FMT_CAP_HOST_READABLE;

    if (gpu->glsl.compute && fmt->glsl_format && p->has_storage)
        fmt->caps |= PL_FMT_CAP_STORABLE | PL_FMT_CAP_READWRITE;

    // GLES 2 can't do blitting
    if (p->gles_ver && p->gles_ver < 30)
        fmt->caps &= ~PL_FMT_CAP_BLITTABLE;

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

#define DO_FORMATS(formats)                                 \
    do {                                                    \
        for (int i = 0; i < PL_ARRAY_SIZE(formats); i++)    \
            add_format(gpu, &formats[i]);                   \
    } while (0)

bool gl_setup_formats(struct pl_gpu_t *gpu)
{
    struct pl_gl *p = PL_PRIV(gpu);

#ifdef PL_HAVE_UNIX
    if (p->has_modifiers) {
        EGLint num_formats = 0;
        bool ok = eglQueryDmaBufFormatsEXT(p->egl_dpy, 0, NULL,
                                           &num_formats);
        if (ok && num_formats) {
            p->egl_formats.elem = pl_calloc(gpu, num_formats, sizeof(EGLint));
            p->egl_formats.num = num_formats;
            ok = eglQueryDmaBufFormatsEXT(p->egl_dpy, num_formats,
                                          p->egl_formats.elem, &num_formats);
            pl_assert(ok);

            PL_DEBUG(gpu, "EGL formats supported:");
            for (int i = 0; i < num_formats; ++i) {
                PL_DEBUG(gpu, "    0x%08x(%.4s)", p->egl_formats.elem[i],
                         PRINT_FOURCC(p->egl_formats.elem[i]));
            }
        }
    }
#endif

    if (p->gl_ver >= 30) {
        // Desktop GL3+ has everything
        DO_FORMATS(formats_norm8);
        DO_FORMATS(formats_bgra8);
        DO_FORMATS(formats_norm16);
        DO_FORMATS(formats_rgb16_fbo);
        DO_FORMATS(formats_float32);
        DO_FORMATS(formats_float16);
        DO_FORMATS(formats_half16);
        DO_FORMATS(formats_uint);
        goto done;
    }

    if (p->gl_ver >= 21) {
        // If we have a reasonable set of extensions, we can enable most
        // things. Otherwise, pick simple fallback formats
        if (pl_opengl_has_ext(p->gl, "GL_ARB_texture_float") &&
            pl_opengl_has_ext(p->gl, "GL_ARB_texture_rg") &&
            pl_opengl_has_ext(p->gl, "GL_ARB_framebuffer_object"))
        {
            DO_FORMATS(formats_norm8);
            DO_FORMATS(formats_bgra8);
            DO_FORMATS(formats_norm16);
            DO_FORMATS(formats_rgb16_fbo);
            DO_FORMATS(formats_float32);
            DO_FORMATS(formats_float16);
            if (pl_opengl_has_ext(p->gl, "GL_ARB_half_float_pixel"))
            {
                DO_FORMATS(formats_half16);
            }
        } else {
            // Fallback for GL2
            DO_FORMATS(formats_legacy_gl2);
            DO_FORMATS(formats_basic_vertex);
        }
        goto done;
    }

    if (p->gles_ver >= 30) {
        // GLES 3.0 has some basic formats, with framebuffers for float16
        // depending on GL_EXT_color_buffer_(half_)float support
        DO_FORMATS(formats_norm8);
        if (pl_opengl_has_ext(p->gl, "GL_EXT_texture_norm16")) {
            DO_FORMATS(formats_norm16);
            DO_FORMATS(formats_rgb16_fallback);
        }
        if (pl_opengl_has_ext(p->gl, "GL_EXT_texture_format_BGRA8888"))
            DO_FORMATS(formats_bgra_gles);
        DO_FORMATS(formats_uint);
        DO_FORMATS(formats_basic_vertex);
        if (p->gles_ver >= 32 || pl_opengl_has_ext(p->gl, "GL_EXT_color_buffer_float")) {
            DO_FORMATS(formats_float16_fbo);
        } else {
            DO_FORMATS(formats_float16_fallback);
        }
        goto done;
    }

    if (p->gles_ver >= 20) {
        // GLES 2.0 only has some legacy fallback formats, with support for
        // float16 depending on GL_EXT_texture_norm16 being present
        DO_FORMATS(formats_legacy_gles2);
        DO_FORMATS(formats_basic_vertex);
        if (pl_opengl_has_ext(p->gl, "GL_EXT_texture_rg")) {
            DO_FORMATS(formats_norm8);
        }
        if (pl_opengl_has_ext(p->gl, "GL_EXT_texture_format_BGRA8888")) {
            DO_FORMATS(formats_bgra_gles);
        }
        goto done;
    }

    // Last resort fallback. Probably not very useful
    DO_FORMATS(formats_basic_vertex);
    goto done;

done:
    return gl_check_err(gpu, "gl_setup_formats");
}
