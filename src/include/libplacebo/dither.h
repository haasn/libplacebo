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

#ifndef LIBPLACEBO_DITHER_H_
#define LIBPLACEBO_DITHER_H_

// Generates a deterministic NxN bayer (ordered) dither matrix, storing the
// result in `data`. `size` must be a power of two. The resulting matrix will
// be roughly uniformly distributed within the range [0,1).
void pl_generate_bayer_matrix(float *data, int size);

// Generates a random NxN blue noise texture. storing the result in `data`.
// `size` must be a positive power of two no larger than 256. The resulting
// texture will be roughly uniformly distributed within the range [0,1).
//
// Note: This function is very, *very* slow for large sizes. Generating a
// dither matrix with size 256 can take several seconds on a modern processor.
void pl_generate_blue_noise(float *data, int size);

#endif // LIBPLACEBO_DITHER_H_
