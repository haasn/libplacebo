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

#define RA_PFN(name) __typeof__(ra_##name) *name

struct ra_fns {
    // Destructors: These also free the corresponding objects, but they
    // must not be called on NULL. (The NULL checks are done by the ra_*_destroy
    // wrappers)
    void (*destroy)(const struct ra *ra);
    void (*tex_destroy)(const struct ra *, const struct ra_tex *);
    void (*buf_destroy)(const struct ra *, const struct ra_buf *);
    void (*renderpass_destroy)(const struct ra *, const struct ra_renderpass *);

    RA_PFN(tex_create);
    RA_PFN(tex_clear);
    RA_PFN(tex_blit); // optional if RA_CAP_TEX_BLIT is not present
    RA_PFN(tex_upload);
    RA_PFN(tex_download);
    RA_PFN(buf_create);
    RA_PFN(buf_update);
    RA_PFN(buf_poll); // optional: if NULL buffers are always free to use
    RA_PFN(desc_namespace);
    RA_PFN(renderpass_create);
    RA_PFN(renderpass_run);

    // The following functions are all optional, but they must either all be
    // supported or all be absent. They will never be called on NULL timers
    // (the ra_timer_* wrappers check for this).
    RA_PFN(timer_create);
    RA_PFN(timer_start);
    RA_PFN(timer_stop);
    void (*timer_destroy)(const struct ra *, struct ra_timer *);

    // The following functions are optional if the corresponding ra_limit
    // size restriction is 0
    RA_PFN(buf_uniform_layout);
    RA_PFN(buf_storage_layout);
    RA_PFN(push_constant_layout);
};

#undef RA_PFN
