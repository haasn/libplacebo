// License: CC0 / Public Domain
#include "glfw.h"

#include <stdio.h>

#ifdef NDEBUG
#define DEBUG false
#else
#define DEBUG true
#endif

static void err_cb(int code, const char *desc)
{
    fprintf(stderr, "GLFW err %d: %s\n", code, desc);
}

static void close_cb(GLFWwindow *win)
{
    struct winstate *w = glfwGetWindowUserPointer(win);
    w->window_lost = true;
}

static void resize_cb(GLFWwindow *win, int width, int height)
{
    struct winstate *w = glfwGetWindowUserPointer(win);
    if (!pl_swapchain_resize(w->swapchain, &width, &height)) {
        fprintf(stderr, "libplacebo: Failed resizing swapchain? Exiting...\n");
        w->window_lost = true;
    }
}

bool glfw_init(struct pl_context *ctx, struct winstate *w,
               int width, int height, enum winflags flags)
{
    *w = (struct winstate) {
        .ctx = ctx,
    };

    if (!glfwInit()) {
        fprintf(stderr, "GLFW: Failed initializing?\n");
        return false;
    }

    glfwSetErrorCallback(&err_cb);

#ifdef USE_VK
    if (!glfwVulkanSupported()) {
        fprintf(stderr, "GLFW: No vulkan support! Perhaps recompile with -DUSE_GL\n");
        return false;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#endif // USE_VK

#ifdef USE_GL
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);

    /* Request OpenGL 3.2 (or higher) core profile */
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif // USE_GL

    bool alpha = flags & WIN_ALPHA;
    if (alpha)
        glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);

    printf("Creating %dx%d window%s...\n", width, height,
           alpha ? " (with alpha)" : "");

    w->win = glfwCreateWindow(width, height, "libplacebo demo", NULL, NULL);
    if (!w->win) {
        fprintf(stderr, "GLFW: Failed creating window\n");
        return false;
    }

    // Set up GLFW event callbacks
    glfwSetWindowUserPointer(w->win, w);
    glfwSetFramebufferSizeCallback(w->win, resize_cb);
    glfwSetWindowCloseCallback(w->win, close_cb);

#ifdef USE_VK
    VkResult err;

    struct pl_vk_inst_params iparams = pl_vk_inst_default_params;
    iparams.debug = DEBUG;

    // Load all extensions required for WSI
    uint32_t num;
    iparams.extensions = glfwGetRequiredInstanceExtensions(&num);
    iparams.num_extensions = num;

    w->vk_inst = pl_vk_inst_create(w->ctx, &iparams);
    if (!w->vk_inst) {
        fprintf(stderr, "libplacebo: Failed creating vulkan instance\n");
        return false;
    }

    err = glfwCreateWindowSurface(w->vk_inst->instance, w->win, NULL, &w->surf);
    if (err != VK_SUCCESS) {
        fprintf(stderr, "GLFW: Failed creating vulkan surface\n");
        return false;
    }

    struct pl_vulkan_params params = pl_vulkan_default_params;
    params.instance = w->vk_inst->instance;
    params.surface = w->surf;
    params.allow_software = true;
    w->vk = pl_vulkan_create(w->ctx, &params);
    if (!w->vk) {
        fprintf(stderr, "libplacebo: Failed creating vulkan device\n");
        return false;
    }

    w->swapchain = pl_vulkan_create_swapchain(w->vk, &(struct pl_vulkan_swapchain_params) {
        .surface = w->surf,
        .present_mode = VK_PRESENT_MODE_FIFO_KHR,
        .prefer_hdr = (flags & WIN_HDR),
    });

    if (!w->swapchain) {
        fprintf(stderr, "libplacebo: Failed creating vulkan swapchain\n");
        return false;
    }

    w->gpu = w->vk->gpu;
#endif // USE_VK

#ifdef USE_GL
    struct pl_opengl_params params = pl_opengl_default_params;
    params.allow_software = true;
    params.debug = DEBUG;

    glfwMakeContextCurrent(w->win);

    w->gl = pl_opengl_create(w->ctx, &params);
    if (!w->gl) {
        fprintf(stderr, "libplacebo: Failed creating opengl device\n");
        return false;
    }

    w->swapchain = pl_opengl_create_swapchain(w->gl, &(struct pl_opengl_swapchain_params) {
        .swap_buffers = (void (*)(void *)) glfwSwapBuffers,
        .priv = w->win,
    });

    if (!w->swapchain) {
        fprintf(stderr, "libplacebo: Failed creating opengl swapchain\n");
        return false;
    }

    if (!pl_swapchain_resize(w->swapchain, &width, &height)) {
        fprintf(stderr, "libplacebo: Failed initializing swapchain\n");
        return false;
    }

    w->gpu = w->gl->gpu;
#endif // USE_GL

    return true;
}

void glfw_uninit(struct winstate *w)
{
    pl_swapchain_destroy(&w->swapchain);

#ifdef USE_VK
    pl_vulkan_destroy(&w->vk);
    if (w->surf)
        vkDestroySurfaceKHR(w->vk_inst->instance, w->surf, NULL);
    pl_vk_inst_destroy(&w->vk_inst);
#endif

#ifdef USE_GL
    pl_opengl_destroy(&w->gl);
#endif

    glfwTerminate();
    *w = (struct winstate) {0};
}
