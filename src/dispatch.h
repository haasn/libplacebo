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

#include "common.h"

// Like `pl_dispatch_begin`, but has an extra `unique` parameter. If this is
// true, the generated shader will be uniquely namespaced `unique` and may be
// freely merged with other shaders (`sh_subpass`). Otherwise, all shaders have
// the same namespace and merging them is an error.
pl_shader pl_dispatch_begin_ex(pl_dispatch dp, bool unique);

// Set the `dynamic_constants` field for newly created `pl_shader` objects.
//
// This is a private API because it's sort of clunky/stateful.
void pl_dispatch_mark_dynamic(pl_dispatch dp, bool dynamic);
