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

#pragma once

#include <stdio.h>
#include "bstr/bstr.h"

#include "common.h"
#include "context.h"

struct pl_shader {
    // Read-only fields
    struct pl_context *ctx;
    const struct ra *ra;

    // Internal state
    bool mutable;
    struct pl_shader_res res; // for accumulating vertex_attribs etc.
    struct bstr buffer_head;
    struct bstr buffer_body;
    bool flexible_work_groups;
    int fresh;
    int namespace;
    void *tmp;

    // For vertex attributes, since we need to keep track of their location
    int current_va_location;
    size_t current_va_offset;

    // For bindings, since we need to keep the namespaces unique
    int *current_binding;
};

typedef const char * ident_t;

// Helpers for adding new variables/descriptors/etc. with fresh, unique
// identifier names. These will never conflcit with other identifiers, even
// if the shaders are merged together.
ident_t sh_fresh(struct pl_shader *sh, const char *name);

// Add a new shader var and return its identifier
ident_t sh_var(struct pl_shader *sh, struct pl_shader_var sv);

// Add a new shader desc and return its identifier. This function takes care of
// setting the binding to a fresh bind point according to the namespace
// requirements, so the caller may leave it blank.
ident_t sh_desc(struct pl_shader *sh, struct pl_shader_desc sd);

// Add a new vec2 vertex attribute from a pl_rect2df, or returns NULL on failure.
ident_t sh_attr_vec2(struct pl_shader *sh, const char *name,
                     const struct pl_rect2df *rc);

// Bind a texture under a given transformation and make its attributes
// available as well. If an output pointer for one of the attributes is left
// as NULL, that attribute will not be added. Returns NULL on failure.
ident_t sh_bind(struct pl_shader *sh, const struct ra_tex *tex,
                const char *name, const struct pl_transform2x2 *tf,
                ident_t *out_pos, ident_t *out_size, ident_t *out_pt);

// Replace all of the free variables in the glsl and input list by literally
// string replacing it with an encoded representation of the given namespace
void sh_rename_vars(struct pl_shader *sh, int namespace);

// Underlying function for appending text to a shader
void pl_shader_append(struct pl_shader *sh, struct bstr *buf,
                      const char *fmt, ...)
    PRINTF_ATTRIBUTE(3, 4);

#define GLSLH(...) pl_shader_append(sh, &sh->buffer_head, __VA_ARGS__)
#define GLSL(...)  pl_shader_append(sh, &sh->buffer_body, __VA_ARGS__)

// Requires that the share is mutable and has an output signature compatible
// with the given input signature. Errors and returns false otherwise.
bool sh_require_input(struct pl_shader *sh, enum pl_shader_sig insig);
