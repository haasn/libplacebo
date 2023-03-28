/* Simplistic demo that just makes the window colorful, including alpha
 * transparency if supported by the windowing system.
 *
 * License: CC0 / Public Domain
 */

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "common.h"
#include "utils.h"
#include "window.h"

static pl_log logger;
static struct window *win;

static void uninit(int ret)
{
    window_destroy(&win);
    pl_log_destroy(&logger);
    exit(ret);
}

int main(int argc, char **argv)
{
    logger = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb = pl_log_color,
        .log_level = PL_LOG_DEBUG,
    ));

    win = window_create(logger, &(struct window_params) {
        .title = "colors demo",
        .width = 640,
        .height = 480,
        .alpha = true,
    });
    if (!win)
        uninit(1);

    double ts_start, ts;
    if (!utils_gettime(&ts_start)) {
        uninit(1);
    }

    while (!win->window_lost) {
        if (window_get_key(win, KEY_ESC))
            break;

        struct pl_swapchain_frame frame;
        bool ok = pl_swapchain_start_frame(win->swapchain, &frame);
        if (!ok) {
            // Something unexpected happened, perhaps the window is not
            // visible? Wait for events and try again.
            window_poll(win, true);
            continue;
        }

        if (!utils_gettime(&ts))
            uninit(1);

        const double period = 10.; // in seconds
        float secs = fmod(ts - ts_start, period);

        float pos = 2 * M_PI * secs / period;
        float alpha = (cosf(pos) + 1.0) / 2.0;

        assert(frame.fbo->params.blit_dst);
        pl_tex_clear(win->gpu, frame.fbo, (float[4]) {
            alpha * (sinf(2 * pos + 0.0) + 1.0) / 2.0,
            alpha * (sinf(2 * pos + 2.0) + 1.0) / 2.0,
            alpha * (sinf(2 * pos + 4.0) + 1.0) / 2.0,
            alpha,
        });

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
