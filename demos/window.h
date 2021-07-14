// License: CC0 / Public Domain
#pragma once

#include <libplacebo/swapchain.h>

struct window {
    const struct window_impl *impl;
    pl_swapchain swapchain;
    pl_gpu gpu;
    bool window_lost;
};

struct window_params {
    const char *title;
    int width;
    int height;

    // initial color space
    struct pl_swapchain_colors colors;
    bool alpha;
};

struct window *window_create(pl_log log, const struct window_params *params);
void window_destroy(struct window **win);

// Poll/wait for window events
void window_poll(struct window *win, bool block);

// Input handling
enum button {
    BTN_LEFT,
    BTN_RIGHT,
    BTN_MIDDLE,
};

void window_get_cursor(const struct window *win, int *x, int *y);
void window_get_scroll(const struct window *win, float *dx, float *dy);
bool window_get_button(const struct window *win, enum button);
char *window_get_file(const struct window *win);

// For implementations
struct window_impl {
    const char *name;
    __typeof__(window_create) *create;
    __typeof__(window_destroy) *destroy;
    __typeof__(window_poll) *poll;
    __typeof__(window_get_cursor) *get_cursor;
    __typeof__(window_get_scroll) *get_scroll;
    __typeof__(window_get_button) *get_button;
    __typeof__(window_get_file) *get_file;
};
