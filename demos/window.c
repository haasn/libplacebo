// License: CC0 / Public Domain

#include "common.h"
#include "window.h"

extern const struct window_impl win_impl_glfw_vk;
extern const struct window_impl win_impl_glfw_gl;
extern const struct window_impl win_impl_glfw_d3d11;
extern const struct window_impl win_impl_sdl_vk;
extern const struct window_impl win_impl_sdl_gl;

static const struct window_impl *win_impls[] = {
#ifdef HAVE_GLFW_VULKAN
    &win_impl_glfw_vk,
#endif
#ifdef HAVE_GLFW_OPENGL
    &win_impl_glfw_gl,
#endif
#ifdef HAVE_GLFW_D3D11
    &win_impl_glfw_d3d11,
#endif
#ifdef HAVE_SDL_VULKAN
    &win_impl_sdl_vk,
#endif
#ifdef HAVE_SDL_OPENGL
    &win_impl_sdl_gl,
#endif
    NULL
};

struct window *window_create(pl_log log, const struct window_params *params)
{
    for (const struct window_impl **impl = win_impls; *impl; impl++) {
        printf("Attempting to initialize API: %s\n", (*impl)->name);
        struct window *win = (*impl)->create(log, params);
        if (win)
            return win;
    }

    fprintf(stderr, "No windowing system / graphical API compiled or supported!\n");
    exit(1);
}

void window_destroy(struct window **win)
{
    if (*win)
        (*win)->impl->destroy(win);
}

void window_poll(struct window *win, bool block)
{
    return win->impl->poll(win, block);
}

void window_get_cursor(const struct window *win, int *x, int *y)
{
    return win->impl->get_cursor(win, x, y);
}

void window_get_scroll(const struct window *win, float *dx, float *dy)
{
    return win->impl->get_scroll(win, dx, dy);
}

bool window_get_button(const struct window *win, enum button btn)
{
    return win->impl->get_button(win, btn);
}

bool window_get_key(const struct window *win, enum key key)
{
    return win->impl->get_key(win, key);
}

char *window_get_file(const struct window *win)
{
    return win->impl->get_file(win);
}

bool window_toggle_fullscreen(const struct window *win, bool fullscreen)
{
    return win->impl->toggle_fullscreen(win, fullscreen);
}

bool window_is_fullscreen(const struct window *win)
{
    return win->impl->is_fullscreen(win);
}
