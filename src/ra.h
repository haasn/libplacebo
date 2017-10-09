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

// Note: The `destroy` functions also free the corresponding objects, but
// must not be called on NULL. (The NULL checks are done by the ra_*_destroy
// wrappers)
struct ra_fns {
    void (*destroy)(const struct ra *ra);

    const struct ra_tex *(*tex_create)(const struct ra *ra,
                                       const struct ra_tex_params *params);
    void (*tex_destroy)(const struct ra *ra, const struct ra_tex *tex);
    void (*tex_clear)(const struct ra *ra, const struct ra_tex *dst,
                      struct pl_rect3d rect, const float color[4]);
    // Optional if RA_CAP_TEX_BLIT is not present.
    void (*tex_blit)(const struct ra *ra,
                 const struct ra_tex *dst, const struct ra_tex *src,
                 struct pl_rect3d dst_rc, struct pl_rect3d src_rc);
    bool (*tex_upload)(const struct ra *ra,
                       const struct ra_tex_upload_params *params);

    const struct ra_buf *(*buf_create)(const struct ra *ra,
                                       const struct ra_buf_params *params);
    void (*buf_destroy)(const struct ra *ra, const struct ra_buf *buf);
    void (*buf_update)(const struct ra *ra, const struct ra_buf *buf,
                       size_t buf_offset, const void *data, size_t size);
    // Optional, if NULL it's assumed that ra_buf_poll always returns true.
    bool (*buf_poll)(const struct ra *ra, const struct ra_buf *buf);

    // The following functions are optional if the corresponding ra_limit
    // size restriction is 0
    struct ra_var_layout (*buf_uniform_layout)(const struct ra *, size_t offset,
                                               const struct ra_var *var);
    struct ra_var_layout (*buf_storage_layout)(const struct ra *ra, size_t offset,
                                               const struct ra_var *var);
    struct ra_var_layout (*push_constant_layout)(const struct ra *ra, size_t offset,
                                                 const struct ra_var *var);

    const struct ra_renderpass *(*renderpass_create)(const struct ra *ra,
                                    const struct ra_renderpass_params *params);
    void (*renderpass_destroy)(const struct ra *ra,
                               const struct ra_renderpass *pass);
    void (*renderpass_run)(const struct ra *ra,
                           const struct ra_renderpass_run_params *params);

    // The following functions are all optional, but they must either all be
    // supported or all be absent. They will never be called on NULL timers
    // (the ra_timer_* wrappers check for this).
    struct ra_timer *(*timer_create)(const struct ra *ra);
    void (*timer_destroy)(const struct ra *ra, struct ra_timer *timer);
    void (*timer_start)(const struct ra *ra, struct ra_timer *timer);
    uint64_t (*timer_stop)(const struct ra *ra, struct ra_timer *timer);
};
