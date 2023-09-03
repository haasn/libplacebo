// License: CC0 / Public Domain

#include <string.h>

#include "common.h"
#include "window.h"

#ifdef _WIN32
#include <windows.h>
#include <timeapi.h>
#endif

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
        if (params->forced_impl && strcmp((*impl)->tag, params->forced_impl) != 0)
            continue;

        printf("Attempting to initialize API: %s\n", (*impl)->name);
        struct window *win = (*impl)->create(log, params);
        if (win) {
#ifdef _WIN32
            if (timeBeginPeriod(1) != TIMERR_NOERROR)
                fprintf(stderr, "timeBeginPeriod failed!\n");
#endif
            return win;
        }
    }

    if (params->forced_impl)
        fprintf(stderr, "'%s' windowing system not compiled or supported!\n", params->forced_impl);
    else
        fprintf(stderr, "No windowing system / graphical API compiled or supported!\n");

    exit(1);
}

void window_destroy(struct window **win)
{
    if (!*win)
        return;

    (*win)->impl->destroy(win);

#ifdef _WIN32
    timeEndPeriod(1);
#endif
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

const char *window_get_clipboard(const struct window *win)
{
    return win->impl->get_clipboard(win);
}

void window_set_clipboard(const struct window *win, const char *text)
{
    win->impl->set_clipboard(win, text);
}
