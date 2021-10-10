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
#include "pl_thread.h"

struct priv {
    struct pl_opengl_swapchain_params params;
    pl_opengl gl;
    pl_mutex lock;
    bool has_sync;

    // current parameters
    pl_tex fb;
    bool frame_started;

    // vsync fences
    int swapchain_depth;
    PL_ARRAY(GLsync) vsync_fences;
};

static struct pl_sw_fns opengl_swapchain;

pl_swapchain pl_opengl_create_swapchain(pl_opengl gl,
                              const struct pl_opengl_swapchain_params *params)
{
    pl_gpu gpu = gl->gpu;

    if (params->max_swapchain_depth < 0) {
        PL_ERR(gpu, "Tried specifying negative swapchain depth?");
        return NULL;
    }

    if (!gl_make_current(gl))
        return NULL;

    struct pl_swapchain *sw = pl_zalloc_obj(NULL, sw, struct priv);
    sw->impl = &opengl_swapchain;
    sw->log = gpu->log;
    sw->ctx = sw->log;
    sw->gpu = gpu;

    struct priv *p = PL_PRIV(sw);
    pl_mutex_init(&p->lock);
    p->params = *params;
    p->has_sync = epoxy_has_gl_extension("GL_ARB_sync");
    p->gl = gl;

    gl_release_current(gl);
    return sw;
}

static void gl_sw_destroy(pl_swapchain sw)
{
    pl_gpu gpu = sw->gpu;
    struct priv *p = PL_PRIV(sw);

    pl_gpu_flush(gpu);
    pl_tex_destroy(gpu, &p->fb);
    pl_mutex_destroy(&p->lock);
    pl_free((void *) sw);
}

static int gl_sw_latency(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);
    return p->params.max_swapchain_depth;
}

static bool gl_sw_resize(pl_swapchain sw, int *width, int *height)
{
    struct priv *p = PL_PRIV(sw);
    const int w = *width, h = *height;

    pl_mutex_lock(&p->lock);
    if (p->fb && w == p->fb->params.w && h == p->fb->params.h) {
        pl_mutex_unlock(&p->lock);
        return true;
    }

    if (p->frame_started && (w || h)) {
        PL_ERR(sw, "Tried resizing the swapchain while a frame was in progress! "
               "Please submit the current frame first.");
        pl_mutex_unlock(&p->lock);
        return false;
    }

    if (w && h) {
        pl_tex_destroy(sw->gpu, &p->fb);
        p->fb = pl_opengl_wrap(sw->gpu, pl_opengl_wrap_params(
            .framebuffer = p->params.framebuffer.id,
            .width = w,
            .height = h,
        ));
        if (!p->fb) {
            PL_ERR(sw, "Failed wrapping OpenGL framebuffer!");
            pl_mutex_unlock(&p->lock);
            return false;
        }
    }

    if (!p->fb) {
        PL_ERR(sw, "Tried calling `pl_swapchain_resize` with unknown size! "
               "This is forbidden for OpenGL. The first call to "
               "`pl_swapchain_resize` must include the width and height of the "
               "swapchain, because there's no way to figure this out from "
               "within the API.");
        pl_mutex_unlock(&p->lock);
        return false;
    }

    *width = p->fb->params.w;
    *height = p->fb->params.h;
    pl_mutex_unlock(&p->lock);
    return true;
}

void pl_opengl_swapchain_update_fb(pl_swapchain sw,
                                   const struct pl_opengl_framebuffer *fb)
{
    struct priv *p = PL_PRIV(sw);
    pl_mutex_lock(&p->lock);
    if (p->frame_started) {
        PL_ERR(sw,"Tried calling `pl_opengl_swapchain_update_fb` while a frame "
               "was in progress! Please submit the current frame first.");
        pl_mutex_unlock(&p->lock);
        return;
    }

    if (p->params.framebuffer.id != fb->id)
        pl_tex_destroy(sw->gpu, &p->fb);

    p->params.framebuffer = *fb;
    pl_mutex_unlock(&p->lock);
}

static bool gl_sw_start_frame(pl_swapchain sw,
                              struct pl_swapchain_frame *out_frame)
{
    struct priv *p = PL_PRIV(sw);
    pl_mutex_lock(&p->lock);
    bool ok = false;

    if (!p->fb) {
        PL_ERR(sw, "Unknown framebuffer size. Please call `pl_swapchain_resize` "
               "before `pl_swapchain_start_frame` for OpenGL swapchains!");
        goto error;
    }

    if (p->frame_started) {
        PL_ERR(sw, "Attempted calling `pl_swapchain_start` while a frame was "
               "already in progress! Call `pl_swapchain_submit_frame` first.");
        goto error;
    }

    if (!gl_make_current(p->gl))
        goto error;

    *out_frame = (struct pl_swapchain_frame) {
        .fbo = p->fb,
        .flipped = !p->params.framebuffer.flipped,
        .color_repr = {
            .sys = PL_COLOR_SYSTEM_RGB,
            .levels = PL_COLOR_LEVELS_FULL,
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
        .color_space = pl_color_space_monitor,
    };

    ok = p->frame_started = gl_check_err(sw->gpu, "gl_sw_start_frame");
    gl_release_current(p->gl);
    // fall through

error:
    pl_mutex_unlock(&p->lock);
    return ok;
}

static bool gl_sw_submit_frame(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);
    pl_mutex_lock(&p->lock);
    if (!p->frame_started) {
        PL_ERR(sw, "Attempted calling `pl_swapchain_submit_frame` with no "
               "frame in progress. Call `pl_swapchain_start_frame` first!");
        pl_mutex_unlock(&p->lock);
        return false;
    }

    if (!gl_make_current(p->gl)) {
        pl_mutex_unlock(&p->lock);
        return false;
    }

    if (p->has_sync && p->params.max_swapchain_depth) {
        GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        if (fence)
            PL_ARRAY_APPEND(sw, p->vsync_fences, fence);
    }

    p->frame_started = false;
    bool ok = gl_check_err(sw->gpu, "gl_sw_submit_frame");
    gl_release_current(p->gl);
    pl_mutex_unlock(&p->lock);

    return ok;
}

static void gl_sw_swap_buffers(pl_swapchain sw)
{
    struct priv *p = PL_PRIV(sw);
    if (!p->params.swap_buffers) {
        PL_ERR(sw, "`pl_swapchain_swap_buffers` called but no "
               "`params.swap_buffers` callback set!");
        return;
    }

    pl_mutex_lock(&p->lock);
    if (!gl_make_current(p->gl)) {
        pl_mutex_unlock(&p->lock);
        return;
    }

    p->params.swap_buffers(p->params.priv);

    const int max_depth = p->params.max_swapchain_depth;
    while (max_depth && p->vsync_fences.num >= max_depth) {
        glClientWaitSync(p->vsync_fences.elem[0], GL_SYNC_FLUSH_COMMANDS_BIT, 1e9);
        glDeleteSync(p->vsync_fences.elem[0]);
        PL_ARRAY_REMOVE_AT(p->vsync_fences, 0);
    }

    gl_check_err(sw->gpu, "gl_sw_swap_buffers");
    gl_release_current(p->gl);
    pl_mutex_unlock(&p->lock);
}

static struct pl_sw_fns opengl_swapchain = {
    .destroy      = gl_sw_destroy,
    .latency      = gl_sw_latency,
    .resize       = gl_sw_resize,
    .start_frame  = gl_sw_start_frame,
    .submit_frame = gl_sw_submit_frame,
    .swap_buffers = gl_sw_swap_buffers,
};
