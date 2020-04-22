/* Compiling:
 *
 *   gcc glfw.c -o ./glfw -O2 -lm -DUSE_VK \
 *       $(pkg-config --cflags --libs glfw3 vulkan libplacebo)
 *
 *  or:
 *
 *   gcc glfw.c -o ./glfw -O2 -lm -DUSE_GL \
 *       $(pkg-config --cflags --libs glfw3 libplacebo)
 *
 * Notes:
 *
 * - This example currently does nothing except making the window purple.
 *
 * - This proof-of-concept is extremely naive. It uses global state, and
 *   ignores uninitialization on errors (just exit()s). This is probably not
 *   what you should be doing for a real program, but I wanted to avoid the
 *   example becoming too complicated.
 *
 * License: CC0 / Public Domain
 */

#if !defined(USE_GL) && !defined(USE_VK) || defined(USE_GL) && defined(USE_VK)
#error Specify exactly one of -DUSE_GL or -DUSE_VULKAN when compiling!
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifdef USE_VK
#define GLFW_INCLUDE_VULKAN
#endif

#include <GLFW/glfw3.h>
#include <libplacebo/renderer.h>

#ifdef USE_VK
#include <libplacebo/vulkan.h>
#endif

#ifdef USE_GL
#include <libplacebo/opengl.h>
#endif

#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480

struct pl_context *ctx;
static GLFWwindow *win;
const struct pl_gpu *gpu;
const struct pl_swapchain *swapchain;

#ifdef USE_VK
VkSurfaceKHR surf;
const struct pl_vulkan *vk;
const struct pl_vk_inst *vk_inst;
#endif

#ifdef USE_GL
const struct pl_opengl *gl;
#endif

static void uninit(int ret)
{
    // Destroy all libplacebo state
    pl_swapchain_destroy(&swapchain);

#ifdef USE_VK
    pl_vulkan_destroy(&vk);
    if (surf)
        vkDestroySurfaceKHR(vk_inst->instance, surf, NULL);
    pl_vk_inst_destroy(&vk_inst);
#endif

#ifdef USE_GL
    pl_opengl_destroy(&gl);
#endif

    pl_context_destroy(&ctx);
    glfwTerminate();
    exit(ret);
}

static void init_glfw()
{
    if (!glfwInit()) {
        fprintf(stderr, "GLFW: failed initializing\n");
        uninit(1);
    }

#ifdef USE_VK
    if (!glfwVulkanSupported()) {
        fprintf(stderr, "GLFW: no vulkan support\n");
        uninit(1);
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#endif

#ifdef USE_GL
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);

    /* Request OpenGL 3.2 (or higher) core profile */
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

    // Attempt creating a transparent window
    glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);

    win = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "libplacebo GLFW demo",
                            NULL, NULL);
    if (!win) {
        fprintf(stderr, "GLFW: failed creating window\n");
        uninit(1);
    }
}

static void init_api();

#ifdef USE_VK
static void init_api()
{
    struct pl_vk_inst_params iparams = pl_vk_inst_default_params;
#ifndef NDEBUG
    iparams.debug = true;
#endif

    // Load all extensions required for WSI
    uint32_t num;
    iparams.extensions = glfwGetRequiredInstanceExtensions(&num);
    iparams.num_extensions = num;

    vk_inst = pl_vk_inst_create(ctx, &iparams);
    if (!vk_inst) {
        fprintf(stderr, "libplacebo: failed creating vulkan instance\n");
        uninit(2);
    }

    VkResult err = glfwCreateWindowSurface(vk_inst->instance, win, NULL, &surf);
    if (err != VK_SUCCESS) {
        fprintf(stderr, "GLFW: failed creating vulkan surface\n");
        uninit(1);
    }

    struct pl_vulkan_params params = pl_vulkan_default_params;
    params.instance = vk_inst->instance;
    params.surface = surf;
    params.allow_software = true;
    vk = pl_vulkan_create(ctx, &params);
    if (!vk) {
        fprintf(stderr, "libplacebo: failed creating vulkan device\n");
        uninit(2);
    }

    swapchain = pl_vulkan_create_swapchain(vk, &(struct pl_vulkan_swapchain_params) {
        .surface = surf,
        .present_mode = VK_PRESENT_MODE_FIFO_KHR,
    });

    if (!swapchain) {
        fprintf(stderr, "libplacebo: failed creating vulkan swapchain\n");
        uninit(2);
    }

    gpu = vk->gpu;
}
#endif // USE_VK

#ifdef USE_GL
static void init_api()
{
    struct pl_opengl_params params = pl_opengl_default_params;
#ifndef NDEBUG
    params.debug = true;
#endif

    glfwMakeContextCurrent(win);

    gl = pl_opengl_create(ctx, &params);
    if (!gl) {
        fprintf(stderr, "libplacebo: failed creating opengl device\n");
        uninit(2);
    }

    swapchain = pl_opengl_create_swapchain(gl, &(struct pl_opengl_swapchain_params) {
        .swap_buffers = (void (*)(void *)) glfwSwapBuffers,
        .priv = win,
    });

    if (!swapchain) {
        fprintf(stderr, "libplacebo: failed creating opengl swapchain\n");
        uninit(2);
    }

    gpu = gl->gpu;


    int w, h;
    glfwGetFramebufferSize(win, &w, &h);
    if (!pl_swapchain_resize(swapchain, &w, &h)) {
        fprintf(stderr, "libplacebo: failed initializing swapchain\n");
        uninit(2);
    }
}
#endif // USE_GL

static void err_cb(int code, const char *desc)
{
    fprintf(stderr, "GLFW err %d: %s\n", code, desc);
}

static void resize_cb(GLFWwindow *win, int w, int h)
{
    if (!pl_swapchain_resize(swapchain, &w, &h)) {
        fprintf(stderr, "libplacebo: failed resizing swapchain\n");
        uninit(2);
    }
}

static void exit_cb(GLFWwindow *win)
{
    uninit(0);
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
    glfwSetErrorCallback(&err_cb);

    ctx = pl_context_create(PL_API_VER, &(struct pl_context_params) {
        .log_cb    = pl_log_color,
        .log_level = PL_LOG_DEBUG,
    });
    assert(ctx);

    init_glfw();
    init_api();

    // Set up GLFW event callbacks
    glfwSetFramebufferSizeCallback(win, resize_cb);
    glfwSetWindowCloseCallback(win, exit_cb);

    float rgba[4] = {0.0, 0.0, 0.0, 1.0};
    int rainbow_pos = 0;

    while (true) {
        struct pl_swapchain_frame frame;
        bool ok = pl_swapchain_start_frame(swapchain, &frame);
        if (!ok) {
            // Something unexpected happened, perhaps the window is not
            // visible? Wait up to 10ms for events and try again
            glfwWaitEventsTimeout(10e-3);
            continue;
        }

        assert(frame.fbo->params.blit_dst);
        evolve_rgba(rgba, &rainbow_pos);
        pl_tex_clear(gpu, frame.fbo, rgba);

        ok = pl_swapchain_submit_frame(swapchain);
        if (!ok) {
            fprintf(stderr, "libplacebo: failed submitting frame!\n");
            uninit(3);
        }

        pl_swapchain_swap_buffers(swapchain);
        glfwPollEvents();
    }
}
