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

#include "common.h"
#include "formats.h"
#include "gpu.h"
#include "swapchain.h"
#include "utils.h"

struct priv {
    struct pl_opengl_swapchain_params params;
    bool has_sync;

    // current parameters
    const struct pl_tex *fb;
    bool frame_started;

    // vsync fences
    int swapchain_depth;
    GLsync *vsync_fences;
    int num_vsync_fences;
};

static struct pl_sw_fns opengl_swapchain;

const struct pl_swapchain *pl_opengl_create_swapchain(const struct pl_opengl *gl,
                              const struct pl_opengl_swapchain_params *params)
{
    const struct pl_gpu *gpu = gl->gpu;

    if (params->max_swapchain_depth < 0) {
        PL_ERR(gpu, "Tried specifying negative swapchain depth?");
        return NULL;
    }

    struct pl_swapchain *sw = talloc_zero_priv(NULL, struct pl_swapchain, struct priv);
    sw->impl = &opengl_swapchain;
    sw->ctx = gpu->ctx;
    sw->gpu = gpu;

    struct priv *p = TA_PRIV(sw);
    p->params = *params;
    p->has_sync = epoxy_has_gl_extension("GL_ARB_sync");

    return sw;
}

static void gl_sw_destroy(const struct pl_swapchain *sw)
{
    const struct pl_gpu *gpu = sw->gpu;
    struct priv *p = TA_PRIV(sw);

    pl_gpu_flush(gpu);
    pl_tex_destroy(gpu, &p->fb);
    talloc_free((void *) sw);
}

static int gl_sw_latency(const struct pl_swapchain *sw)
{
    struct priv *p = TA_PRIV(sw);
    return p->params.max_swapchain_depth;
}

static bool gl_sw_resize(const struct pl_swapchain *sw, int *width, int *height)
{
    struct priv *p = TA_PRIV(sw);
    const int w = *width, h = *height;
    if (p->fb && w == p->fb->params.w && h == p->fb->params.h)
        return true;

    if (p->frame_started && (w || h)) {
        PL_ERR(sw, "Tried resizing the swapchain while a frame was in progress! "
               "Please submit the current frame first.");
        return false;
    }

    if (w && h) {
        pl_tex_destroy(sw->gpu, &p->fb);
        p->fb = pl_opengl_wrap(sw->gpu, &(struct pl_opengl_wrap_params) {
            .framebuffer = p->params.framebuffer.id,
            .width = w,
            .height = h,
        });
        if (!p->fb) {
            PL_ERR(sw, "Failed wrapping OpenGL framebuffer!");
            return false;
        }
    }

    if (!p->fb) {
        PL_ERR(sw, "Tried calling `pl_swapchain_resize` with unknown size! "
               "This is forbidden for OpenGL. The first call to "
               "`pl_swapchain_resize` must include the width and height of the "
               "swapchain, because there's no way to figure this out from "
               "within the API.");
        return false;
    }

    *width = p->fb->params.w;
    *height = p->fb->params.h;
    return true;
}

void pl_opengl_swapchain_update_fb(const struct pl_swapchain *sw,
                                   const struct pl_opengl_framebuffer *fb)
{
    struct priv *p = TA_PRIV(sw);
    if (p->frame_started) {
        PL_ERR(sw,"Tried calling `pl_opengl_swapchain_update_fb` while a frame "
               "was in progress! Please submit the current frame first.");
        return;
    }

    if (p->params.framebuffer.id != fb->id)
        pl_tex_destroy(sw->gpu, &p->fb);

    p->params.framebuffer = *fb;
}

static bool gl_sw_start_frame(const struct pl_swapchain *sw,
                              struct pl_swapchain_frame *out_frame)
{
    struct priv *p = TA_PRIV(sw);
    if (!p->fb) {
        PL_ERR(sw, "Unknown framebuffer size. Please call `pl_swapchain_resize` "
               "before `pl_swapchain_start_frame` for OpenGL swapchains!");
        return false;
    }

    if (p->frame_started) {
        PL_ERR(sw, "Attempted calling `pl_swapchain_start` while a frame was "
               "already in progress! Call `pl_swapchain_submit_frame` first.");
        return false;
    }

    *out_frame = (struct pl_swapchain_frame) {
        .fbo = p->fb,
        .flipped = !p->params.framebuffer.flipped,
        .color_repr = {
            .sys = PL_COLOR_SYSTEM_RGB,
            .levels = PL_COLOR_LEVELS_PC,
            .alpha = PL_ALPHA_UNKNOWN,
            .bits = {
                // Just use the red channel in the absence of anything more
                // sane to do, because the red channel is both guaranteed to
                // exist and also typically has the minimum number of bits
                // (which is arguably what matters for dithering)
                .sample_depth = p->fb->params.format->component_depth[0],
                .color_depth = p->fb->params.format->component_depth[0],
            },
        },
        .color_space = {
            .primaries = PL_COLOR_PRIM_UNKNOWN,
            .transfer = PL_COLOR_TRC_UNKNOWN,
            .light = PL_COLOR_LIGHT_DISPLAY,
        },
    };

    return p->frame_started = gl_check_err(sw->gpu, "gl_sw_start_frame");
}

static bool gl_sw_submit_frame(const struct pl_swapchain *sw)
{
    struct priv *p = TA_PRIV(sw);
    if (!p->frame_started) {
        PL_ERR(sw, "Attempted calling `pl_swapchain_submit_frame` with no "
               "frame in progress. Call `pl_swapchain_start_frame` first!");
        return false;
    }

    if (p->has_sync && p->params.max_swapchain_depth) {
        GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        if (fence)
            TARRAY_APPEND(sw, p->vsync_fences, p->num_vsync_fences, fence);
    }

    p->frame_started = false;
    return gl_check_err(sw->gpu, "gl_sw_submit_frame");
}

static void gl_sw_swap_buffers(const struct pl_swapchain *sw)
{
    struct priv *p = TA_PRIV(sw);

    if (!p->params.swap_buffers) {
        PL_ERR(sw, "`pl_swapchain_swap_buffers` called but no "
               "`params.swap_buffers` callback set!");
        return;
    }

    p->params.swap_buffers(p->params.priv);

    const int max_depth = p->params.max_swapchain_depth;
    while (max_depth && p->num_vsync_fences >= max_depth) {
        glClientWaitSync(p->vsync_fences[0], GL_SYNC_FLUSH_COMMANDS_BIT, 1e9);
        glDeleteSync(p->vsync_fences[0]);
        TARRAY_REMOVE_AT(p->vsync_fences, p->num_vsync_fences, 0);
    }

    gl_check_err(sw->gpu, "gl_sw_swap_buffers");
}

static struct pl_sw_fns opengl_swapchain = {
    .destroy      = gl_sw_destroy,
    .latency      = gl_sw_latency,
    .resize       = gl_sw_resize,
    .start_frame  = gl_sw_start_frame,
    .submit_frame = gl_sw_submit_frame,
    .swap_buffers = gl_sw_swap_buffers,
};
