// License: CC0 / Public Domain

#if defined(USE_GL) + defined(USE_VK) + defined(USE_D3D11) != 1
#error Specify exactly one of -DUSE_GL, -DUSE_VK or -DUSE_D3D11 when compiling!
#endif

#include <string.h>

#include "common.h"
#include "window.h"

#ifdef USE_VK
#define VK_NO_PROTOTYPES
#include <libplacebo/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#define IMPL win_impl_glfw_vk
#define IMPL_NAME "GLFW (vulkan)"
#endif

#ifdef USE_GL
#include <libplacebo/opengl.h>
#define IMPL win_impl_glfw_gl
#define IMPL_NAME "GLFW (opengl)"
#endif

#ifdef USE_D3D11
#include <libplacebo/d3d11.h>
#define IMPL win_impl_glfw_d3d11
#define IMPL_NAME "GLFW (D3D11)"
#endif

#include <GLFW/glfw3.h>

#ifdef USE_D3D11
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

#ifdef NDEBUG
#define DEBUG false
#else
#define DEBUG true
#endif

#define PL_ARRAY_SIZE(s) (sizeof(s) / sizeof((s)[0]))

const struct window_impl IMPL;

struct priv {
    struct window w;
    GLFWwindow *win;

#ifdef USE_VK
    VkSurfaceKHR surf;
    pl_vulkan vk;
    pl_vk_inst vk_inst;
#endif

#ifdef USE_GL
    pl_opengl gl;
#endif

#ifdef USE_D3D11
    pl_d3d11 d3d11;
#endif

    float scroll_dx, scroll_dy;
    char **files;
    size_t files_num;
    size_t files_size;
    bool file_seen;
};

static void err_cb(int code, const char *desc)
{
    fprintf(stderr, "GLFW err %d: %s\n", code, desc);
}

static void close_cb(GLFWwindow *win)
{
    struct priv *p = glfwGetWindowUserPointer(win);
    p->w.window_lost = true;
}

static void resize_cb(GLFWwindow *win, int width, int height)
{
    struct priv *p = glfwGetWindowUserPointer(win);
    if (!pl_swapchain_resize(p->w.swapchain, &width, &height)) {
        fprintf(stderr, "libplacebo: Failed resizing swapchain? Exiting...\n");
        p->w.window_lost = true;
    }
}

static void scroll_cb(GLFWwindow *win, double dx, double dy)
{
    struct priv *p = glfwGetWindowUserPointer(win);
    p->scroll_dx += dx;
    p->scroll_dy += dy;
}

static void drop_cb(GLFWwindow *win, int num, const char *files[])
{
    struct priv *p = glfwGetWindowUserPointer(win);

    for (int i = 0; i < num; i++) {
        if (p->files_num == p->files_size) {
            size_t new_size = p->files_size ? p->files_size * 2 : 16;
            char **new_files = realloc(p->files, new_size * sizeof(char *));
            if (!new_files)
                return;
            p->files = new_files;
            p->files_size = new_size;
        }

        char *file = strdup(files[i]);
        if (!file)
            return;

        p->files[p->files_num++] = file;
    }
}

#ifdef USE_GL
static bool make_current(void *priv)
{
    GLFWwindow *win = priv;
    glfwMakeContextCurrent(win);
    return true;
}

static void release_current(void *priv)
{
    glfwMakeContextCurrent(NULL);
}
#endif

#ifdef USE_VK
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_vk_proc_addr(VkInstance instance, const char* pName)
{
    return (PFN_vkVoidFunction) glfwGetInstanceProcAddress(instance, pName);
}
#endif

static struct window *glfw_create(pl_log log, const struct window_params *params)
{
    struct priv *p = calloc(1, sizeof(struct priv));
    if (!p)
        return NULL;

    p->w.impl = &IMPL;
    if (!glfwInit()) {
        fprintf(stderr, "GLFW: Failed initializing?\n");
        goto error;
    }

    glfwSetErrorCallback(&err_cb);

#ifdef USE_VK
    if (!glfwVulkanSupported()) {
        fprintf(stderr, "GLFW: No vulkan support! Perhaps recompile with -DUSE_GL\n");
        goto error;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#endif // USE_VK

#ifdef USE_D3D11
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#endif // USE_D3D11

#ifdef USE_GL
    struct {
        int api;
        int major, minor;
        int glsl_ver;
        int profile;
    } gl_vers[] = {
        { GLFW_OPENGL_API,    4, 6, 460, GLFW_OPENGL_CORE_PROFILE },
        { GLFW_OPENGL_API,    4, 5, 450, GLFW_OPENGL_CORE_PROFILE },
        { GLFW_OPENGL_API,    4, 4, 440, GLFW_OPENGL_CORE_PROFILE },
        { GLFW_OPENGL_API,    4, 0, 400, GLFW_OPENGL_CORE_PROFILE },
        { GLFW_OPENGL_API,    3, 3, 330, GLFW_OPENGL_CORE_PROFILE },
        { GLFW_OPENGL_API,    3, 2, 150, GLFW_OPENGL_CORE_PROFILE },
        { GLFW_OPENGL_ES_API, 3, 2, 320, },
        { GLFW_OPENGL_API,    3, 1, 140, },
        { GLFW_OPENGL_ES_API, 3, 1, 310, },
        { GLFW_OPENGL_API,    3, 0, 130, },
        { GLFW_OPENGL_ES_API, 3, 0, 300, },
        { GLFW_OPENGL_ES_API, 2, 0, 100, },
        { GLFW_OPENGL_API,    2, 1, 120, },
        { GLFW_OPENGL_API,    2, 0, 110, },
    };

    for (int i = 0; i < PL_ARRAY_SIZE(gl_vers); i++) {
        glfwWindowHint(GLFW_CLIENT_API, gl_vers[i].api);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, gl_vers[i].major);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, gl_vers[i].minor);
        glfwWindowHint(GLFW_OPENGL_PROFILE, gl_vers[i].profile);

#endif // USE_GL

        if (params->alpha)
            glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);

        printf("Creating %dx%d window%s...\n", params->width, params->height,
               params->alpha ? " (with alpha)" : "");

        p->win = glfwCreateWindow(params->width, params->height, params->title, NULL, NULL);

#ifdef USE_GL
        if (p->win)
            break;
    }
#endif // USE_GL

    if (!p->win) {
        fprintf(stderr, "GLFW: Failed creating window\n");
        goto error;
    }

    // Set up GLFW event callbacks
    glfwSetWindowUserPointer(p->win, p);
    glfwSetFramebufferSizeCallback(p->win, resize_cb);
    glfwSetWindowCloseCallback(p->win, close_cb);
    glfwSetScrollCallback(p->win, scroll_cb);
    glfwSetDropCallback(p->win, drop_cb);

#ifdef USE_VK
    VkResult err;

    struct pl_vk_inst_params iparams = pl_vk_inst_default_params;
    iparams.get_proc_addr = get_vk_proc_addr,
    iparams.debug = DEBUG;

    // Load all extensions required for WSI
    uint32_t num;
    iparams.extensions = glfwGetRequiredInstanceExtensions(&num);
    iparams.num_extensions = num;

    p->vk_inst = pl_vk_inst_create(log, &iparams);
    if (!p->vk_inst) {
        fprintf(stderr, "libplacebo: Failed creating vulkan instance\n");
        goto error;
    }

    err = glfwCreateWindowSurface(p->vk_inst->instance, p->win, NULL, &p->surf);
    if (err != VK_SUCCESS) {
        fprintf(stderr, "GLFW: Failed creating vulkan surface\n");
        goto error;
    }

    struct pl_vulkan_params vkparams = pl_vulkan_default_params;
    vkparams.instance = p->vk_inst->instance;
    vkparams.get_proc_addr = p->vk_inst->get_proc_addr;
    vkparams.surface = p->surf;
    vkparams.allow_software = true;
    p->vk = pl_vulkan_create(log, &vkparams);
    if (!p->vk) {
        fprintf(stderr, "libplacebo: Failed creating vulkan device\n");
        goto error;
    }

    p->w.swapchain = pl_vulkan_create_swapchain(p->vk, &(struct pl_vulkan_swapchain_params) {
        .surface = p->surf,
        .present_mode = VK_PRESENT_MODE_FIFO_KHR,
    });

    if (!p->w.swapchain) {
        fprintf(stderr, "libplacebo: Failed creating vulkan swapchain\n");
        goto error;
    }

    p->w.gpu = p->vk->gpu;
#endif // USE_VK

#ifdef USE_GL
    struct pl_opengl_params glparams = pl_opengl_default_params;
    glparams.allow_software = true;
    glparams.debug = DEBUG;
    glparams.make_current = make_current;
    glparams.release_current = release_current;
    glparams.priv = p->win;

    p->gl = pl_opengl_create(log, &glparams);
    if (!p->gl) {
        fprintf(stderr, "libplacebo: Failed creating opengl device\n");
        goto error;
    }

    p->w.swapchain = pl_opengl_create_swapchain(p->gl, &(struct pl_opengl_swapchain_params) {
        .swap_buffers = (void (*)(void *)) glfwSwapBuffers,
        .priv = p->win,
    });

    if (!p->w.swapchain) {
        fprintf(stderr, "libplacebo: Failed creating opengl swapchain\n");
        goto error;
    }

    p->w.gpu = p->gl->gpu;
#endif // USE_GL

#ifdef USE_D3D11
    struct pl_d3d11_params d3dparams = pl_d3d11_default_params;
    d3dparams.debug = DEBUG;

    p->d3d11 = pl_d3d11_create(log, &d3dparams);
    if (!p->d3d11) {
        fprintf(stderr, "libplacebo: Failed creating D3D11 device\n");
        goto error;
    }

    p->w.swapchain = pl_d3d11_create_swapchain(p->d3d11,
                                               &(struct pl_d3d11_swapchain_params) {
        .window = glfwGetWin32Window(p->win),
    });
    if (!p->w.swapchain) {
        fprintf(stderr, "libplacebo: Failed creating D3D11 swapchain\n");
        goto error;
    }

    p->w.gpu = p->d3d11->gpu;
#endif // USE_D3D11

    int w = params->width, h = params->height;
    pl_swapchain_colorspace_hint(p->w.swapchain, &params->colors);
    if (!pl_swapchain_resize(p->w.swapchain, &w, &h)) {
        fprintf(stderr, "libplacebo: Failed initializing swapchain\n");
        goto error;
    }

    return &p->w;

error:
    window_destroy((struct window **) &p);
    return NULL;
}

static void glfw_destroy(struct window **window)
{
    struct priv *p = (struct priv *) *window;
    if (!p)
        return;

    pl_swapchain_destroy(&p->w.swapchain);

#ifdef USE_VK
    pl_vulkan_destroy(&p->vk);
    if (p->surf) {
        PFN_vkDestroySurfaceKHR vkDestroySurfaceKHR = (PFN_vkDestroySurfaceKHR)
            p->vk_inst->get_proc_addr(p->vk_inst->instance, "vkDestroySurfaceKHR");
        vkDestroySurfaceKHR(p->vk_inst->instance, p->surf, NULL);
    }
    pl_vk_inst_destroy(&p->vk_inst);
#endif

#ifdef USE_GL
    pl_opengl_destroy(&p->gl);
#endif

#ifdef USE_D3D11
    pl_d3d11_destroy(&p->d3d11);
#endif

    for (int i = 0; i < p->files_num; i++)
        free(p->files[i]);
    free(p->files);

    glfwTerminate();
    free(p);
    *window = NULL;
}

static void glfw_poll(struct window *window, bool block)
{
    if (block) {
        glfwWaitEvents();
    } else {
        glfwPollEvents();
    }
}

static void glfw_get_cursor(const struct window *window, int *x, int *y)
{
    struct priv *p = (struct priv *) window;
    double dx, dy;
    glfwGetCursorPos(p->win, &dx, &dy);
    *x = dx;
    *y = dy;
}

static bool glfw_get_button(const struct window *window, enum button btn)
{
    static const int button_map[] = {
        [BTN_LEFT] = GLFW_MOUSE_BUTTON_LEFT,
        [BTN_RIGHT] = GLFW_MOUSE_BUTTON_RIGHT,
        [BTN_MIDDLE] = GLFW_MOUSE_BUTTON_MIDDLE,
    };

    struct priv *p = (struct priv *) window;
    return glfwGetMouseButton(p->win, button_map[btn]) == GLFW_PRESS;
}

static bool glfw_get_key(const struct window *window, enum key key)
{
    static const int key_map[] = {
        [KEY_ESC] = GLFW_KEY_ESCAPE,
    };

    struct priv *p = (struct priv *) window;
    return glfwGetKey(p->win, key_map[key]) == GLFW_PRESS;
}

static void glfw_get_scroll(const struct window *window, float *dx, float *dy)
{
    struct priv *p = (struct priv *) window;
    *dx = p->scroll_dx;
    *dy = p->scroll_dy;
    p->scroll_dx = p->scroll_dy = 0.0;
}

static char *glfw_get_file(const struct window *window)
{
    struct priv *p = (struct priv *) window;
    if (p->file_seen) {
        assert(p->files_num);
        free(p->files[0]);
        memmove(&p->files[0], &p->files[1], --p->files_num * sizeof(char *));
        p->file_seen = false;
    }

    if (!p->files_num)
        return NULL;

    p->file_seen = true;
    return p->files[0];
}

const struct window_impl IMPL = {
    .name = IMPL_NAME,
    .create = glfw_create,
    .destroy = glfw_destroy,
    .poll = glfw_poll,
    .get_cursor = glfw_get_cursor,
    .get_button = glfw_get_button,
    .get_key = glfw_get_key,
    .get_scroll = glfw_get_scroll,
    .get_file = glfw_get_file,
};
