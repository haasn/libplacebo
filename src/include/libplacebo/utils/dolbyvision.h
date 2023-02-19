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

#ifndef LIBPLACEBO_DOLBYVISION_H_
#define LIBPLACEBO_DOLBYVISION_H_

#include <libplacebo/colorspace.h>

PL_API_BEGIN

// Parses the Dolby Vision RPU, and sets the `pl_hdr_metadata` dynamic
// brightness metadata fields accordingly.
//
// Note: requires `PL_HAVE_LIBDOVI` to be defined, no-op otherwise.
void pl_hdr_metadata_from_dovi_rpu(struct pl_hdr_metadata *out,
                                   const uint8_t *buf, size_t size);

PL_API_END

#endif // LIBPLACEBO_DOLBYVISION_H_
