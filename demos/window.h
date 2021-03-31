// License: CC0 / Public Domain
#pragma once

#include <libplacebo/swapchain.h>

struct window {
    const struct pl_swapchain *swapchain;
    const struct pl_gpu *gpu;
    bool window_lost;
};

enum winflags {
    WIN_ALPHA,
    WIN_HDR,
};

struct window *window_create(struct pl_context *ctx, const char *title,
                             int width, int height, enum winflags flags);

void window_destroy(struct window **window);

// Poll/wait for window events
void window_poll(struct window *window, bool block);

// Input handling
enum button {
    BTN_LEFT,
    BTN_RIGHT,
    BTN_MIDDLE,
};

void window_get_cursor(const struct window *window, int *x, int *y);
void window_get_scroll(const struct window *window, float *dx, float *dy);
bool window_get_button(const struct window *window, enum button);
char *window_get_file(const struct window *window);
