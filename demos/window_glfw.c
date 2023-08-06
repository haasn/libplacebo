// License: CC0 / Public Domain

#if defined(USE_GL) + defined(USE_VK) + defined(USE_D3D11) != 1
#error Specify exactly one of -DUSE_GL, -DUSE_VK or -DUSE_D3D11 when compiling!
#endif

#include <string.h>
#include <math.h>

#include "common.h"
#include "window.h"

#ifdef USE_VK
#define VK_NO_PROTOTYPES
#include <libplacebo/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#define IMPL win_impl_glfw_vk
#define IMPL_NAME "GLFW (vulkan)"
#define IMPL_TAG "glfw-vk"
#endif

#ifdef USE_GL
#include <libplacebo/opengl.h>
#define IMPL win_impl_glfw_gl
#define IMPL_NAME "GLFW (opengl)"
#define IMPL_TAG "glfw-gl"
#endif

#ifdef USE_D3D11
#include <libplacebo/d3d11.h>
#define IMPL win_impl_glfw_d3d11
#define IMPL_NAME "GLFW (D3D11)"
#define IMPL_TAG "glfw-d3d11"
#endif

#include <GLFW/glfw3.h>

#if defined(USE_GL) && defined(HAVE_EGL)
#define GLFW_EXPOSE_NATIVE_EGL
#include <GLFW/glfw3native.h>
#endif

#ifdef USE_D3D11
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

#ifdef _WIN32
#define strdup _strdup
#endif

#ifdef NDEBUG
#define DEBUG false
#else
#define DEBUG true
#endif

#define PL_ARRAY_SIZE(s) (sizeof(s) / sizeof((s)[0]))

const struct window_impl IMPL;

struct window_pos {
    int x;
    int y;
    int w;
    int h;
};

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

    struct window_pos windowed_pos;
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
#ifdef HAVE_EGL
        glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
#endif
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, gl_vers[i].major);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, gl_vers[i].minor);
        glfwWindowHint(GLFW_OPENGL_PROFILE, gl_vers[i].profile);
#ifdef __APPLE__
        if (gl_vers[i].profile == GLFW_OPENGL_CORE_PROFILE)
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

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

    uint32_t num;
    p->vk_inst = pl_vk_inst_create(log, pl_vk_inst_params(
        .get_proc_addr = get_vk_proc_addr,
        .debug = DEBUG,
        .extensions = glfwGetRequiredInstanceExtensions(&num),
        .num_extensions = num,
    ));

    if (!p->vk_inst) {
        fprintf(stderr, "libplacebo: Failed creating vulkan instance\n");
        goto error;
    }

    err = glfwCreateWindowSurface(p->vk_inst->instance, p->win, NULL, &p->surf);
    if (err != VK_SUCCESS) {
        fprintf(stderr, "GLFW: Failed creating vulkan surface\n");
        goto error;
    }

    p->vk = pl_vulkan_create(log, pl_vulkan_params(
        .instance = p->vk_inst->instance,
        .get_proc_addr = p->vk_inst->get_proc_addr,
        .surface = p->surf,
        .allow_software = true,
    ));
    if (!p->vk) {
        fprintf(stderr, "libplacebo: Failed creating vulkan device\n");
        goto error;
    }

    p->w.swapchain = pl_vulkan_create_swapchain(p->vk, pl_vulkan_swapchain_params(
        .surface = p->surf,
        .present_mode = VK_PRESENT_MODE_FIFO_KHR,
    ));

    if (!p->w.swapchain) {
        fprintf(stderr, "libplacebo: Failed creating vulkan swapchain\n");
        goto error;
    }

    p->w.gpu = p->vk->gpu;
#endif // USE_VK

#ifdef USE_GL
    p->gl = pl_opengl_create(log, pl_opengl_params(
        .allow_software = true,
        .debug = DEBUG,
#ifdef HAVE_EGL
        .egl_display = glfwGetEGLDisplay(),
        .egl_context = glfwGetEGLContext(p->win),
#endif
        .make_current = make_current,
        .release_current = release_current,
        .get_proc_addr = glfwGetProcAddress,
        .priv = p->win,
    ));
    if (!p->gl) {
        fprintf(stderr, "libplacebo: Failed creating opengl device\n");
        goto error;
    }

    p->w.swapchain = pl_opengl_create_swapchain(p->gl, pl_opengl_swapchain_params(
        .swap_buffers = (void (*)(void *)) glfwSwapBuffers,
        .priv = p->win,
    ));

    if (!p->w.swapchain) {
        fprintf(stderr, "libplacebo: Failed creating opengl swapchain\n");
        goto error;
    }

    p->w.gpu = p->gl->gpu;
#endif // USE_GL

#ifdef USE_D3D11
    p->d3d11 = pl_d3d11_create(log, pl_d3d11_params( .debug = DEBUG ));
    if (!p->d3d11) {
        fprintf(stderr, "libplacebo: Failed creating D3D11 device\n");
        goto error;
    }

    p->w.swapchain = pl_d3d11_create_swapchain(p->d3d11, pl_d3d11_swapchain_params(
        .window = glfwGetWin32Window(p->win),
    ));
    if (!p->w.swapchain) {
        fprintf(stderr, "libplacebo: Failed creating D3D11 swapchain\n");
        goto error;
    }

    p->w.gpu = p->d3d11->gpu;
#endif // USE_D3D11

    glfwGetWindowSize(p->win, &p->windowed_pos.w, &p->windowed_pos.h);
    glfwGetWindowPos(p->win, &p->windowed_pos.x, &p->windowed_pos.y);

    int w, h;
    glfwGetFramebufferSize(p->win, &w, &h);
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
    int fw, fh, ww, wh;
    glfwGetCursorPos(p->win, &dx, &dy);
    glfwGetFramebufferSize(p->win, &fw, &fh);
    glfwGetWindowSize(p->win, &ww, &wh);
    *x = floor(dx * fw / ww);
    *y = floor(dy * fh / wh);
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

static bool glfw_is_fullscreen(const struct window *window) {
    const struct priv *p = (const struct priv *) window;
    return glfwGetWindowMonitor(p->win);
}

static bool glfw_toggle_fullscreen(const struct window *window, bool fullscreen)
{
    struct priv *p = (struct priv *) window;
    bool window_fullscreen = glfw_is_fullscreen(window);

    if (window_fullscreen == fullscreen)
        return true;

    if (window_fullscreen) {
        glfwSetWindowMonitor(p->win, NULL, p->windowed_pos.x, p->windowed_pos.y,
                             p->windowed_pos.w, p->windowed_pos.h, GLFW_DONT_CARE);
        return true;
    }

    // For simplicity sake use primary monitor
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    if (!monitor)
        return false;

    const GLFWvidmode *mode = glfwGetVideoMode(monitor);
    if (!mode)
        return false;

    glfwGetWindowPos(p->win, &p->windowed_pos.x, &p->windowed_pos.y);
    glfwGetWindowSize(p->win, &p->windowed_pos.w, &p->windowed_pos.h);
    glfwSetWindowMonitor(p->win, monitor, 0, 0, mode->width, mode->height,
                         mode->refreshRate);

    return true;
}

static const char *glfw_get_clipboard(const struct window *window)
{
    struct priv *p = (struct priv *) window;
    return glfwGetClipboardString(p->win);
}

static void glfw_set_clipboard(const struct window *window, const char *text)
{
    struct priv *p = (struct priv *) window;
    glfwSetClipboardString(p->win, text);
}

const struct window_impl IMPL = {
    .name = IMPL_NAME,
    .tag = IMPL_TAG,
    .create = glfw_create,
    .destroy = glfw_destroy,
    .poll = glfw_poll,
    .get_cursor = glfw_get_cursor,
    .get_button = glfw_get_button,
    .get_key = glfw_get_key,
    .get_scroll = glfw_get_scroll,
    .get_file = glfw_get_file,
    .toggle_fullscreen = glfw_toggle_fullscreen,
    .is_fullscreen = glfw_is_fullscreen,
    .get_clipboard = glfw_get_clipboard,
    .set_clipboard = glfw_set_clipboard,
};
