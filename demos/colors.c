/* Simplistic demo that just makes the window colorful, including alpha
 * transparency if supported by the windowing system.
 *
 * License: CC0 / Public Domain
 */

#include <time.h>
#include <math.h>

#include "common.h"
#include "window.h"

static pl_log logger;
static struct window *win;

static void uninit(int ret)
{
    window_destroy(&win);
    pl_log_destroy(&logger);
    exit(ret);
}

static void evolve_rgba(float rgba[4], int *pos)
{
    const int scale = 512;
    const float circle = 2.0 * M_PI;
    const float piece  = (float)(*pos % scale) / (scale - 1);

    float alpha = (cosf(circle * (*pos) / scale * 0.5) + 1.0) / 2.0;
    rgba[0] = alpha * (sinf(circle * piece + 0.0) + 1.0) / 2.0;
    rgba[1] = alpha * (sinf(circle * piece + 2.0) + 1.0) / 2.0;
    rgba[2] = alpha * (sinf(circle * piece + 4.0) + 1.0) / 2.0;
    rgba[3] = alpha;

    *pos += 1;
}

int main(int argc, char **argv)
{
    logger = pl_log_create(PL_API_VER, &(struct pl_log_params) {
        .log_cb = pl_log_color,
        .log_level = PL_LOG_DEBUG,
    });

    win = window_create(logger, "colors demo", 640, 480, WIN_ALPHA);
    if (!win)
        uninit(1);

    float rgba[4] = {0.0, 0.0, 0.0, 1.0};
    int rainbow_pos = 0;

    while (!win->window_lost) {
        struct pl_swapchain_frame frame;
        bool ok = pl_swapchain_start_frame(win->swapchain, &frame);
        if (!ok) {
            // Something unexpected happened, perhaps the window is not
            // visible? Wait for events and try again.
            window_poll(win, true);
            continue;
        }

        assert(frame.fbo->params.blit_dst);
        evolve_rgba(rgba, &rainbow_pos);
        pl_tex_clear(win->gpu, frame.fbo, rgba);

        ok = pl_swapchain_submit_frame(win->swapchain);
        if (!ok) {
            fprintf(stderr, "libplacebo: failed submitting frame!\n");
            uninit(3);
        }

        pl_swapchain_swap_buffers(win->swapchain);
        window_poll(win, false);
    }

    uninit(0);
}
