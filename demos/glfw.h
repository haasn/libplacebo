// License: CC0 / Public Domain
#pragma once

#if !defined(USE_GL) && !defined(USE_VK) || defined(USE_GL) && defined(USE_VK)
#error Specify exactly one of -DUSE_GL or -DUSE_VK when compiling!
#endif

#ifdef USE_VK
#define GLFW_INCLUDE_VULKAN
#endif

#include <GLFW/glfw3.h>
#ifdef USE_VK
#include <libplacebo/vulkan.h>
#endif

#ifdef USE_GL
#include <libplacebo/opengl.h>
#endif

struct winstate {
    struct pl_context *ctx;
    const struct pl_swapchain *swapchain;
    const struct pl_gpu *gpu;
    bool window_lost;
    GLFWwindow *win;

#ifdef USE_VK
    VkSurfaceKHR surf;
    const struct pl_vulkan *vk;
    const struct pl_vk_inst *vk_inst;
#endif

#ifdef USE_GL
    const struct pl_opengl *gl;
#endif
};

enum winflags {
    WIN_ALPHA,
    WIN_HDR,
};

bool glfw_init(struct pl_context *ctx, struct winstate *w,
               int width, int height, enum winflags flags);
void glfw_uninit(struct winstate *w);
