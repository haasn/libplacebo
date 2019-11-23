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

#pragma once

#include "common.h"

struct gl_format {
    GLint ifmt;         // sized internal format (e.g. GL_RGBA16F)
    GLenum fmt;         // base internal format (e.g. GL_RGBA)
    GLenum type;        // host-visible type (e.g. GL_FLOAT)
    struct pl_fmt tmpl; // pl_fmt template
    int flags;          // F_* flags
};

enum {
    // --- gl_format.flags

    // Version flags. If at least 1 flag matches, the format entry is considered
    // supported on the current GL context.
    F_GL2       = 1 << 0, // GL2.1-only
    F_GL3       = 1 << 1, // GL3.0 or later
    F_ES2       = 1 << 2, // ES2-only
    F_ES3       = 1 << 3, // ES3.0 or later
    F_ES32      = 1 << 4, // ES3.2 or later
    F_EXT16     = 1 << 5, // ES with GL_EXT_texture_norm16
    F_EXTF16    = 1 << 6, // GL_EXT_color_buffer_half_float
    F_GL2F      = 1 << 7, // GL2.1-only with texture_rg + texture_float + FBOs

    // Feature flags. They are additional and signal presence of features.
    F_CR        = 1 << 16, // color-renderable
    F_TF        = 1 << 17, // texture-filterable with GL_LINEAR
    F_CF        = F_CR | F_TF,
    F_VF        = 1 << 18, // vertex format
};

extern const struct gl_format gl_formats[];

int gl_format_feature_flags(const struct pl_gpu *gpu);
