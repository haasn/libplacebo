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

#include <libplacebo/options.h>

#include "pl_string.h"

extern const struct pl_opt_t pl_color_repr_opts[];
extern const struct pl_opt_t pl_color_space_opts[];
extern const struct pl_opt_t pl_input_opts[];
extern const struct pl_opt_t pl_compositor_opts[];

// Generalization of pl_options_set_str to arbitrary structs
bool pl_options_set_str_ex(pl_log log, void *base, const struct pl_opt_t *list,
                           pl_str key, pl_str value);

// Returns a pointer to the field backing `opt`, inside the struct `base`
void *pl_opt_field(pl_opt opt, void *base);

// Returns the size of the field backing `opt`
size_t pl_opt_size(pl_opt opt);

// Parses `value` into `field`, the field backing a (non-struct) `opt`. `base`
// is the top-level struct the option list belongs to, which may differ from
// the struct containing `field` for options inside of a nested struct.
bool pl_opt_parse(pl_log log, void *base, pl_opt opt, void *field, pl_str value);
