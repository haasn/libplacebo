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

#include "formats.h"
#include "common.h"

#define FMT(_name, bits, ftype, _caps)               \
    (struct pl_fmt) {                                \
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

    // Component aliases
    R     = GL_RED,
    RG    = GL_RG,
    RGB   = GL_RGB,
    RGBA  = GL_RGBA,
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

// Floating point texture formats
const struct gl_format formats_float[] = {
    {GL_R16F,           R,     FLT, FMT("r16f",    16, FLOAT, S|L|F)},
    {GL_RG16F,          RG,    FLT, FMT("rg16f",   16, FLOAT, S|L|F)},
    {GL_RGB16F,         RGB,   FLT, FMT("rgb16f",  16, FLOAT, S|L|F)},
    {GL_RGBA16F,        RGBA,  FLT, FMT("rgba16f", 16, FLOAT, S|L|F)},
    {GL_R32F,           R,     FLT, FMT("r32f",    32, FLOAT, S|L|F|V)},
    {GL_RG32F,          RG,    FLT, FMT("rg32f",   32, FLOAT, S|L|F|V)},
    {GL_RGB32F,         RGB,   FLT, FMT("rgb32f",  32, FLOAT, S|L|F|V)},
    {GL_RGBA32F,        RGBA,  FLT, FMT("rgba32f", 32, FLOAT, S|L|F|V)},
};

// Renderable 16-bit float formats (excluding rgb16f)
const struct gl_format formats_float16_fbo[] = {
    {GL_R16F,           R,     FLT, FMT("r16f",    16, FLOAT, S|L|F)},
    {GL_RG16F,          RG,    FLT, FMT("rg16f",   16, FLOAT, S|L|F)},
    {GL_RGB16F,         RGB,   FLT, FMT("rgb16f",  16, FLOAT, S|L)},
    {GL_RGBA16F,        RGBA,  FLT, FMT("rgba16f", 16, FLOAT, S|L|F)},
};

// Non-renderable 16-bit float formats
const struct gl_format formats_float16_fallback[] = {
    {GL_R16F,           R,     FLT, FMT("r16f",    16, FLOAT, S|L)},
    {GL_RG16F,          RG,    FLT, FMT("rg16f",   16, FLOAT, S|L)},
    {GL_RGB16F,         RGB,   FLT, FMT("rgb16f",  16, FLOAT, S|L)},
    {GL_RGBA16F,        RGBA,  FLT, FMT("rgba16f", 16, FLOAT, S|L)},
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

    {GL_R8_SNORM,       R,     I8,  FMT("r8s",      8, SNORM)},
    {GL_RG8_SNORM,      RG,    I8,  FMT("rg8s",     8, SNORM)},
    {GL_RGB8_SNORM,     RGB,   I8,  FMT("rgb8s",    8, SNORM)},
    {GL_RGBA8_SNORM,    RGBA,  I8,  FMT("rgba8s",   8, SNORM)},
    {GL_R16_SNORM,      R,     I16, FMT("r16s",    16, SNORM)},
    {GL_RG16_SNORM,     RG,    I16, FMT("rg16s",   16, SNORM)},
    {GL_RGB16_SNORM,    RGB,   I16, FMT("rgb16s",  16, SNORM)},
    {GL_RGBA16_SNORM,   RGBA,  I16, FMT("rgba16s", 16, SNORM)},

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

// Fallback for vertex-only formats, as a last resort
const struct gl_format formats_basic_vertex[] = {
    {GL_R32F,           R,     FLT, FMT("r32f",    32, FLOAT, V)},
    {GL_RG32F,          RG,    FLT, FMT("rg32f",   32, FLOAT, V)},
    {GL_RGB32F,         RGB,   FLT, FMT("rgb32f",  32, FLOAT, V)},
    {GL_RGBA32F,        RGBA,  FLT, FMT("rgba32f", 32, FLOAT, V)},
};

#define DO_FORMATS(formats)                                 \
    do {                                                    \
        for (int i = 0; i < PL_ARRAY_SIZE(formats); i++)    \
            do_format(gpu, &formats[i]);                    \
    } while (0)

void pl_gl_enumerate_formats(const struct pl_gpu *gpu, gl_format_cb do_format)
{
    struct pl_gl *p = PL_PRIV(gpu);

    if (p->gl_ver >= 30) {
        // Desktop GL3+ has everything
        DO_FORMATS(formats_norm8);
        DO_FORMATS(formats_norm16);
        DO_FORMATS(formats_rgb16_fbo);
        DO_FORMATS(formats_float);
        DO_FORMATS(formats_uint);
        return;
    }

    if (p->gl_ver >= 21) {
        // If we have a reasonable set of extensions, we can enable most
        // things. Otherwise, pick simple fallback formats
        if (epoxy_has_gl_extension("GL_ARB_texture_float") &&
            epoxy_has_gl_extension("GL_ARB_texture_rg") &&
            epoxy_has_gl_extension("GL_ARB_framebuffer_object"))
        {
            DO_FORMATS(formats_norm8);
            DO_FORMATS(formats_norm16);
            DO_FORMATS(formats_rgb16_fbo);
            DO_FORMATS(formats_float);
        } else {
            // Fallback for GL2
            DO_FORMATS(formats_legacy_gl2);
            DO_FORMATS(formats_basic_vertex);
        }
        return;
    }

    if (p->gles_ver >= 30) {
        // GLES 3.0 has some basic formats, with framebuffers for float16
        // depending on GL_EXT_color_buffer_half_float support
        DO_FORMATS(formats_norm8);
        DO_FORMATS(formats_uint);
        DO_FORMATS(formats_basic_vertex);
        if (p->gles_ver >= 32 || epoxy_has_gl_extension("GL_EXT_color_buffer_half_float")) {
            DO_FORMATS(formats_float16_fbo);
        } else {
            DO_FORMATS(formats_float16_fallback);
        }
        return;
    }

    if (p->gles_ver >= 20) {
        // GLES 2.0 only has some legacy fallback formats, with support for
        // float16 depending on GL_EXT_texture_norm16 being present
        DO_FORMATS(formats_legacy_gles2);
        DO_FORMATS(formats_basic_vertex);
        if (epoxy_has_gl_extension("GL_EXT_texture_norm16")) {
            DO_FORMATS(formats_norm16);
            DO_FORMATS(formats_rgb16_fallback);
        }
        return;
    }

    // Last resort fallback. Probably not very useful
    DO_FORMATS(formats_basic_vertex);
}
