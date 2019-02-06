/* Compiling:
 *
 *   gcc glfw.c -o ./glfw -O2 \
 *       $(pkg-config --cflags --libs glfw3 vulkan libplacebo)
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <libplacebo/renderer.h>
#include <libplacebo/vulkan.h>

#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480

struct pl_context *ctx;
static GLFWwindow *win;
VkSurfaceKHR surf;

const struct pl_vulkan *vk;
const struct pl_vk_inst *vk_inst;
const struct pl_swapchain *swapchain;

static void uninit(int ret)
{
    // Destroy all libplacebo state
    pl_swapchain_destroy(&swapchain);
    pl_vulkan_destroy(&vk);
    if (surf)
        vkDestroySurfaceKHR(vk_inst->instance, surf, NULL);
    pl_vk_inst_destroy(&vk_inst);
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

    if (!glfwVulkanSupported()) {
        fprintf(stderr, "GLFW: no vulkan support\n");
        uninit(1);
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    win = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "libplacebo GLFW demo",
                            NULL, NULL);
    if (!win) {
        fprintf(stderr, "GLFW: failed creating window\n");
        uninit(1);
    }
}

static void init_vulkan()
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
}

static void err_cb(int code, const char *desc)
{
    fprintf(stderr, "GLFW err %d: %s\n", code, desc);
}

static void resize_cb(GLFWwindow *win, int w, int h)
{
    if (!pl_swapchain_resize(swapchain, &w, &h)) {
        fprintf(stderr, "libplacebo: failed resizing vulkan swapchain\n");
        uninit(2);
    }
}

static void exit_cb(GLFWwindow *win)
{
    uninit(0);
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
    init_vulkan();

    // Set up GLFW event callbacks
    glfwSetFramebufferSizeCallback(win, resize_cb);
    glfwSetWindowCloseCallback(win, exit_cb);

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
        pl_tex_clear(vk->gpu, frame.fbo, (float[4]){0.5, 0.0, 1.0, 1.0});

        ok = pl_swapchain_submit_frame(swapchain);
        if (!ok) {
            fprintf(stderr, "libplacebo: failed submitting frame!\n");
            uninit(3);
        }

        pl_swapchain_swap_buffers(swapchain);
        glfwPollEvents();
    }
}
