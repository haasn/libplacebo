// License: CC0 / Public Domain

#if !defined(USE_GL) && !defined(USE_VK) || defined(USE_GL) && defined(USE_VK)
#error Specify exactly one of -DUSE_GL or -DUSE_VK when compiling!
#endif

#include "common.h"
#include "window.h"

#include <SDL2/SDL.h>

#ifdef USE_VK
#include <libplacebo/vulkan.h>
#include <SDL2/SDL_vulkan.h>
#define WINFLAG_API SDL_WINDOW_VULKAN
#endif

#ifdef USE_GL
#include <libplacebo/opengl.h>
#define WINFLAG_API SDL_WINDOW_OPENGL
#endif

#ifdef NDEBUG
#define DEBUG false
#else
#define DEBUG true
#endif

struct priv {
    struct window w;
    SDL_Window *win;

#ifdef USE_VK
    VkSurfaceKHR surf;
    const struct pl_vulkan *vk;
    const struct pl_vk_inst *vk_inst;
#endif

#ifdef USE_GL
    SDL_GLContext gl_ctx;
    const struct pl_opengl *gl;
#endif
};

struct window *window_create(struct pl_context *ctx, const char *title,
                             int width, int height, enum winflags flags)
{
    struct priv *p = calloc(1, sizeof(struct priv));
    if (!p)
        return NULL;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL2: Failed initializing: %s\n", SDL_GetError());
        goto error;
    }

    uint32_t sdl_flags = SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | WINFLAG_API;
    p->win = SDL_CreateWindow(title, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                              width, height, sdl_flags);
    if (!p->win) {
        fprintf(stderr, "SDL2: Failed creating window: %s\n", SDL_GetError());
        goto error;
    }

#ifdef USE_VK
    struct pl_vk_inst_params iparams = pl_vk_inst_default_params;
    iparams.debug = DEBUG;

    unsigned int num = 0;
    if (!SDL_Vulkan_GetInstanceExtensions(p->win, &num, NULL)) {
        fprintf(stderr, "SDL2: Failed enumerating vulkan extensions: %s\n", SDL_GetError());
        goto error;
    }

    const char **exts = malloc(num * sizeof(const char *));
    SDL_Vulkan_GetInstanceExtensions(p->win, &num, exts);
    iparams.extensions = exts;
    iparams.num_extensions = num;

    p->vk_inst = pl_vk_inst_create(ctx, &iparams);
    free(exts);
    if (!p->vk_inst) {
        fprintf(stderr, "libplacebo: Failed creating vulkan instance!\n");
        goto error;
    }

    if (!SDL_Vulkan_CreateSurface(p->win, p->vk_inst->instance, &p->surf)) {
        fprintf(stderr, "SDL2: Failed creating surface: %s\n", SDL_GetError());
        goto error;
    }

    struct pl_vulkan_params params = pl_vulkan_default_params;
    params.instance = p->vk_inst->instance;
    params.surface = p->surf;
    params.allow_software = true;
    p->vk = pl_vulkan_create(ctx, &params);
    if (!p->vk) {
        fprintf(stderr, "libplacebo: Failed creating vulkan device\n");
        goto error;
    }

    p->w.swapchain = pl_vulkan_create_swapchain(p->vk, &(struct pl_vulkan_swapchain_params) {
        .surface = p->surf,
        .present_mode = VK_PRESENT_MODE_FIFO_KHR,
        .prefer_hdr = (flags & WIN_HDR),
    });

    if (!p->w.swapchain) {
        fprintf(stderr, "libplacebo: Failed creating vulkan swapchain\n");
        goto error;
    }

    p->w.gpu = p->vk->gpu;
#endif // USE_VK

#ifdef USE_GL
    p->gl_ctx = SDL_GL_CreateContext(p->win);
    if (!p->gl_ctx) {
        fprintf(stderr, "SDL2: Failed creating GL context: %s\n", SDL_GetError());
        goto error;
    }

    SDL_GL_MakeCurrent(p->win, p->gl_ctx);

    struct pl_opengl_params params = pl_opengl_default_params;
    params.allow_software = true;
    params.debug = DEBUG;
    p->gl = pl_opengl_create(ctx, &params);
    if (!p->gl) {
        fprintf(stderr, "libplacebo: Failed creating opengl device\n");
        goto error;
    }

    p->w.swapchain = pl_opengl_create_swapchain(p->gl, &(struct pl_opengl_swapchain_params) {
        .swap_buffers = (void (*)(void *)) SDL_GL_SwapWindow,
        .priv = p->win,
    });

    if (!p->w.swapchain) {
        fprintf(stderr, "libplacebo: Failed creating opengl swapchain\n");
        goto error;
    }

    if (!pl_swapchain_resize(p->w.swapchain, &width, &height)) {
        fprintf(stderr, "libplacebo: Failed initializing swapchain\n");
        goto error;
    }

    p->w.gpu = p->gl->gpu;
#endif // USE_GL

    return &p->w;

error:
    window_destroy((struct window **) &p);
    return NULL;
}

void window_destroy(struct window **window)
{
    struct priv *p = (struct priv *) *window;
    if (!p)
        return;

    pl_swapchain_destroy(&p->w.swapchain);

#ifdef USE_VK
    pl_vulkan_destroy(&p->vk);
    if (p->surf)
        vkDestroySurfaceKHR(p->vk_inst->instance, p->surf, NULL);
    pl_vk_inst_destroy(&p->vk_inst);
#endif

#ifdef USE_GL
    pl_opengl_destroy(&p->gl);
    SDL_GL_DeleteContext(p->gl_ctx);
#endif

    SDL_DestroyWindow(p->win);
    SDL_Quit();
    free(p);
    *window = NULL;
}

static inline void handle_event(struct priv *p, SDL_Event *event)
{
    switch (event->type) {
    case SDL_QUIT:
        p->w.window_lost = true;
        return;

    case SDL_WINDOWEVENT:
        if (event->window.windowID != SDL_GetWindowID(p->win))
            return;

        if (event->window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
            int width = event->window.data1, height = event->window.data2;
            if (!pl_swapchain_resize(p->w.swapchain, &width, &height)) {
                fprintf(stderr, "libplacebo: Failed resizing swapchain? Exiting...\n");
                p->w.window_lost = true;
            }
        }
        return;
    }
}

void window_poll(struct window *window, bool block)
{
    struct priv *p = (struct priv *) window;
    SDL_Event event;
    int ret;

    do {
        ret = block ? SDL_WaitEvent(&event) : SDL_PollEvent(&event);
        if (ret)
            handle_event(p, &event);

        // Only block on the first iteration
        block = false;
    } while (ret);
}

void window_get_cursor(const struct window *window, int *x, int *y)
{
    SDL_GetMouseState(x, y);
}

bool window_get_button(const struct window *window, enum button btn)
{
    static const uint32_t button_mask[] = {
        [BTN_LEFT] = SDL_BUTTON_LMASK,
        [BTN_RIGHT] = SDL_BUTTON_RMASK,
        [BTN_MIDDLE] = SDL_BUTTON_MMASK,
    };

    return SDL_GetMouseState(NULL, NULL) & button_mask[btn];
}
