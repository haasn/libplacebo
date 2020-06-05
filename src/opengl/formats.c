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

#define FMT(_name, bits, ftype)                      \
    (struct pl_fmt) {                                \
        .name = _name,                               \
        .type = PL_FMT_##ftype,                      \
        .sample_order = {0, 1, 2, 3},                \
        .component_depth = {bits, bits, bits, bits}, \
    }

enum {
    // Type aliases
    T_U8    = GL_UNSIGNED_BYTE,
    T_U16   = GL_UNSIGNED_SHORT,
    T_U32   = GL_UNSIGNED_INT,
    T_I8    = GL_BYTE,
    T_I16   = GL_SHORT,
    T_I32   = GL_INT,
    T_FLT   = GL_FLOAT,

    // Component aliases
    C_R     = GL_RED,
    C_RG    = GL_RG,
    C_RGB   = GL_RGB,
    C_RGBA  = GL_RGBA,
    C_RI    = GL_RED_INTEGER,
    C_RGI   = GL_RG_INTEGER,
    C_RGBI  = GL_RGB_INTEGER,
    C_RGBAI = GL_RGBA_INTEGER,
};

// Capability aliases
#define F_S     PL_FMT_CAP_SAMPLEABLE
#define F_VF    PL_FMT_CAP_VERTEX
#define F_CR    (PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLITTABLE)
#define F_TF    PL_FMT_CAP_LINEAR
#define F_CFS   (F_CR | F_TF | F_S)

const struct gl_format gl_formats[] = {
    // Copyright note: This list and the feature flags are largely copied 1:1
    // from mpv.
    //
    // Regular, byte-aligned integer formats. These are used for desktop GL 3+,
    // and GLES 3+ with GL_EXT_texture_norm16.
    {GL_R8,             C_R,     T_U8,  FMT("r8",       8, UNORM), F_VF | F_CFS, V_GL3 | V_GL2F | V_ES3},
    {GL_RG8,            C_RG,    T_U8,  FMT("rg8",      8, UNORM), F_VF | F_CFS, V_GL3 | V_GL2F | V_ES3},
    {GL_RGB8,           C_RGB,   T_U8,  FMT("rgb8",     8, UNORM), F_VF | F_CFS, V_GL3 | V_GL2F | V_ES3},
    {GL_RGBA8,          C_RGBA,  T_U8,  FMT("rgba8",    8, UNORM), F_VF | F_CFS, V_GL3 | V_GL2F | V_ES3},
    {GL_R16,            C_R,     T_U16, FMT("r16",     16, UNORM), F_VF | F_CFS, V_GL3 | V_GL2F | V_EXT16},
    {GL_RG16,           C_RG,    T_U16, FMT("rg16",    16, UNORM), F_VF | F_CFS, V_GL3 | V_GL2F | V_EXT16},
    {GL_RGB16,          C_RGB,   T_U16, FMT("rgb16",   16, UNORM), F_VF | F_CFS, V_GL3 | V_GL2F},
    {GL_RGBA16,         C_RGBA,  T_U16, FMT("rgba16",  16, UNORM), F_VF | F_CFS, V_GL3 | V_GL2F | V_EXT16},

    // Specifically not color-renderable.
    {GL_RGB16,          C_RGB,   T_U16, FMT("rgb16",   16, UNORM), F_VF | F_TF | F_S, V_EXT16},

    // GL2 legacy formats. Ignores possibly present FBO extensions (no CF flag set)
    {GL_RGB8,           C_RGB,   T_U8,  FMT("rgb8",     8, UNORM), F_VF | F_TF | F_S, V_GL2},
    {GL_RGBA8,          C_RGBA,  T_U8,  FMT("rgba8",    8, UNORM), F_VF | F_TF | F_S, V_GL2},
    {GL_RGB16,          C_RGB,   T_U16, FMT("rgb16",   16, UNORM), F_VF | F_TF | F_S, V_GL2},
    {GL_RGBA16,         C_RGBA,  T_U16, FMT("rgba16",  16, UNORM), F_VF | F_TF | F_S, V_GL2},

    // ES2 legacy
    {GL_RGB,            C_RGB,   T_U8,  FMT("rgb",      8, UNORM), F_TF | F_S, V_ES2},
    {GL_RGBA,           C_RGBA,  T_U8,  FMT("rgba",     8, UNORM), F_TF | F_S, V_ES2},

    // Non-normalized integer formats.
    // Follows ES 3.0 as to which are color-renderable.
    {GL_R8UI,           C_RI,    T_U8,  FMT("r8u",      8, UINT),  F_VF | F_S | F_CR, V_GL3 | V_ES3},
    {GL_RG8UI,          C_RGI,   T_U8,  FMT("rg8u",     8, UINT),  F_VF | F_S | F_CR, V_GL3 | V_ES3},
    {GL_RGB8UI,         C_RGBI,  T_U8,  FMT("rgb8u",    8, UINT),  F_VF | F_S,        V_GL3 | V_ES3},
    {GL_RGBA8UI,        C_RGBAI, T_U8,  FMT("rgba8u",   8, UINT),  F_VF | F_S | F_CR, V_GL3 | V_ES3},
    {GL_R16UI,          C_RI,    T_U16, FMT("r16u",    16, UINT),  F_VF | F_S | F_CR, V_GL3 | V_ES3},
    {GL_RG16UI,         C_RGI,   T_U16, FMT("rg16u",   16, UINT),  F_VF | F_S | F_CR, V_GL3 | V_ES3},
    {GL_RGB16UI,        C_RGBI,  T_U16, FMT("rgb16u",  16, UINT),  F_VF | F_S,        V_GL3 | V_ES3},
    {GL_RGBA16UI,       C_RGBAI, T_U16, FMT("rgba16u", 16, UINT),  F_VF | F_S | F_CR, V_GL3 | V_ES3},

    /* TODO
    {GL_R32UI,          C_RI,    T_U32, FMT("r32u",    32, UINT)},
    {GL_RG32UI,         C_RGI,   T_U32, FMT("rg32u",   32, UINT)},
    {GL_RGB32UI,        C_RGBI,  T_U32, FMT("rgb32u",  32, UINT)},
    {GL_RGBA32UI,       C_RGBAI, T_U32, FMT("rgba32u", 32, UINT)},
    */

    // On GL3+ or GL2.1 with GL_ARB_texture_float, floats work fully.
    {GL_R16F,           C_R,     T_FLT, FMT("r16f",    16, FLOAT), F_CFS, V_GL3 | V_GL2F},
    {GL_RG16F,          C_RG,    T_FLT, FMT("rg16f",   16, FLOAT), F_CFS, V_GL3 | V_GL2F},
    {GL_RGB16F,         C_RGB,   T_FLT, FMT("rgb16f",  16, FLOAT), F_CFS, V_GL3 | V_GL2F},
    {GL_RGBA16F,        C_RGBA,  T_FLT, FMT("rgba16f", 16, FLOAT), F_CFS, V_GL3 | V_GL2F},
    {GL_R32F,           C_R,     T_FLT, FMT("r32f",    32, FLOAT), F_VF | F_CFS, V_GL3 | V_GL2F},
    {GL_RG32F,          C_RG,    T_FLT, FMT("rg32f",   32, FLOAT), F_VF | F_CFS, V_GL3 | V_GL2F},
    {GL_RGB32F,         C_RGB,   T_FLT, FMT("rgb32f",  32, FLOAT), F_VF | F_CFS, V_GL3 | V_GL2F},
    {GL_RGBA32F,        C_RGBA,  T_FLT, FMT("rgba32f", 32, FLOAT), F_VF | F_CFS, V_GL3 | V_GL2F},

    // Note: we simply don't support float anything on ES2, despite extensions.
    // We also don't bother with non-filterable float formats, and we ignore
    // 32 bit float formats that are not blendable when rendering to them.

    // On ES3.2+, both 16 bit floats work fully (except 3-component formats).
    // F_EXTF16 implies extensions that also enable 16 bit floats fully.
    {GL_R16F,           C_R,     T_FLT, FMT("r16f",    16, FLOAT), F_CFS,      V_ES32 | V_EXTF16},
    {GL_RG16F,          C_RG,    T_FLT, FMT("rg16f",   16, FLOAT), F_CFS,      V_ES32 | V_EXTF16},
    {GL_RGB16F,         C_RGB,   T_FLT, FMT("rgb16f",  16, FLOAT), F_TF | F_S, V_ES32 | V_EXTF16},
    {GL_RGBA16F,        C_RGBA,  T_FLT, FMT("rgba16f", 16, FLOAT), F_CFS,      V_ES32 | V_EXTF16},

    // On ES3.0+, 16 bit floats are texture-filterable.
    // Don't bother with 32 bit floats; they exist but are neither CR nor TF.
    {GL_R16F,           C_R,     T_FLT, FMT("r16f",    16, FLOAT), F_TF | F_S, V_ES3},
    {GL_RG16F,          C_RG,    T_FLT, FMT("rg16f",   16, FLOAT), F_TF | F_S, V_ES3},
    {GL_RGB16F,         C_RGB,   T_FLT, FMT("rgb16f",  16, FLOAT), F_TF | F_S, V_ES3},
    {GL_RGBA16F,        C_RGBA,  T_FLT, FMT("rgba16f", 16, FLOAT), F_TF | F_S, V_ES3},

    // Fallback for vertex formats that should always exist
    {GL_R32F,           C_R,     T_FLT, FMT("r32f",    32, FLOAT), F_VF},
    {GL_RG32F,          C_RG,    T_FLT, FMT("rg32f",   32, FLOAT), F_VF},
    {GL_RGB32F,         C_RGB,   T_FLT, FMT("rgb32f",  32, FLOAT), F_VF},
    {GL_RGBA32F,        C_RGBA,  T_FLT, FMT("rgba32f", 32, FLOAT), F_VF},

    /* TODO
    {GL_R8_SNORM,       C_R,     T_I8,  FMT("r8s",      8, SNORM)},
    {GL_RG8_SNORM,      C_RG,    T_I8,  FMT("rg8s",     8, SNORM)},
    {GL_RGB8_SNORM,     C_RGB,   T_I8,  FMT("rgb8s",    8, SNORM)},
    {GL_RGBA8_SNORM,    C_RGBA,  T_I8,  FMT("rgba8s",   8, SNORM)},
    {GL_R16_SNORM,      C_R,     T_I16, FMT("r16s",    16, SNORM)},
    {GL_RG16_SNORM,     C_RG,    T_I16, FMT("rg16s",   16, SNORM)},
    {GL_RGB16_SNORM,    C_RGB,   T_I16, FMT("rgb16s",  16, SNORM)},
    {GL_RGBA16_SNORM,   C_RGBA,  T_I16, FMT("rgba16s", 16, SNORM)},
    */

    /* TODO
    {GL_R8I,            C_RI,    T_I8,  FMT("r8i",      8, SINT)},
    {GL_RG8I,           C_RGI,   T_I8,  FMT("rg8i",     8, SINT)},
    {GL_RGB8I,          C_RGBI,  T_I8,  FMT("rgb8i",    8, SINT)},
    {GL_RGBA8I,         C_RGBAI, T_I8,  FMT("rgba8i",   8, SINT)},
    {GL_R16I,           C_RI,    T_I16, FMT("r16i",    16, SINT)},
    {GL_RG16I,          C_RGI,   T_I16, FMT("rg16i",   16, SINT)},
    {GL_RGB16I,         C_RGBI,  T_I16, FMT("rgb16i",  16, SINT)},
    {GL_RGBA16I,        C_RGBAI, T_I16, FMT("rgba16i", 16, SINT)},
    {GL_R32I,           C_RI,    T_I32, FMT("r32i",    32, SINT)},
    {GL_RG32I,          C_RGI,   T_I32, FMT("rg32i",   32, SINT)},
    {GL_RGB32I,         C_RGBI,  T_I32, FMT("rgb32i",  32, SINT)},
    {GL_RGBA32I,        C_RGBAI, T_I32, FMT("rgba32i", 32, SINT)},
    */
    {0},
};

// Return an or-ed combination of all F_ flags that apply.
int gl_format_feature_flags(const struct pl_gpu *gpu)
{
    int gl_ver =  epoxy_is_desktop_gl() ? epoxy_gl_version() : 0;
    int es_ver = !epoxy_is_desktop_gl() ? epoxy_gl_version() : 0;

    int flags = (gl_ver == 21 ? V_GL2 : 0)
              | (gl_ver >= 30 ? V_GL3 : 0)
              | (es_ver == 20 ? V_ES2 : 0)
              | (es_ver >= 30 ? V_ES3 : 0)
              | (es_ver >= 32 ? V_ES32 : 0);

    if (epoxy_has_gl_extension("GL_EXT_texture_norm16"))
        flags |= V_EXT16;

    if (es_ver >= 30 && epoxy_has_gl_extension("GL_EXT_color_buffer_half_float"))
        flags |= V_EXTF16;

    if (gl_ver == 21 &&
        epoxy_has_gl_extension("GL_ARB_texture_float") &&
        epoxy_has_gl_extension("GL_ARB_texture_rg") &&
        epoxy_has_gl_extension("GL_ARB_framebuffer_object"))
    {
        flags |= V_GL2F;
    }

    return flags;
}
