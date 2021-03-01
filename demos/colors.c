/* Compiling:
 *
 *   gcc colors.c glfw.c -o ./colors -O2 -lm -DUSE_VK \
 *       $(pkg-config --cflags --libs glfw3 vulkan libplacebo)
 *
 *  or:
 *
 *   gcc colors.c glfw.c -o ./colors -O2 -lm -DUSE_GL \
 *       $(pkg-config --cflags --libs glfw3 libplacebo)
 *
 * License: CC0 / Public Domain
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <libplacebo/renderer.h>

#include "glfw.h"

#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480

struct pl_context *ctx;
static struct winstate win;

static void uninit(int ret)
{
    glfw_uninit(&win);
    pl_context_destroy(&ctx);
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
    ctx = pl_context_create(PL_API_VER, &(struct pl_context_params) {
        .log_cb    = pl_log_color,
        .log_level = PL_LOG_DEBUG,
    });
    assert(ctx);

    if (!glfw_init(ctx, &win, WINDOW_WIDTH, WINDOW_HEIGHT, WIN_ALPHA))
        uninit(1);

    float rgba[4] = {0.0, 0.0, 0.0, 1.0};
    int rainbow_pos = 0;

    while (!win.window_lost) {
        struct pl_swapchain_frame frame;
        bool ok = pl_swapchain_start_frame(win.swapchain, &frame);
        if (!ok) {
            // Something unexpected happened, perhaps the window is not
            // visible? Wait for events and try again.
            glfwWaitEvents();
            continue;
        }

        assert(frame.fbo->params.blit_dst);
        evolve_rgba(rgba, &rainbow_pos);
        pl_tex_clear(win.gpu, frame.fbo, rgba);

        ok = pl_swapchain_submit_frame(win.swapchain);
        if (!ok) {
            fprintf(stderr, "libplacebo: failed submitting frame!\n");
            uninit(3);
        }

        pl_swapchain_swap_buffers(win.swapchain);
        glfwPollEvents();
    }

    uninit(0);
}
