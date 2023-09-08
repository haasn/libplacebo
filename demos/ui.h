// License: CC0 / Public Domain
#pragma once

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_BOOL
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_BUTTON_TRIGGER_ON_RELEASE
#define NK_UINT_DRAW_INDEX
#include <nuklear.h>

#include "common.h"
#include "window.h"

struct ui;

struct ui *ui_create(pl_gpu gpu);
void ui_destroy(struct ui **ui);

// Update/Logic/Draw cycle
void ui_update_input(struct ui *ui, const struct window *window);
struct nk_context *ui_get_context(struct ui *ui);
bool ui_draw(struct ui *ui, const struct pl_swapchain_frame *frame);

// Helper function to draw a custom widget for drag&drop operations, returns
// true if the widget is hovered
static inline bool ui_widget_hover(struct nk_context *nk, const char *label)
{
    struct nk_rect bounds;
    if (!nk_widget(&bounds, nk))
        return false;

    struct nk_command_buffer *canvas = nk_window_get_canvas(nk);
    bool hover = nk_input_is_mouse_hovering_rect(&nk->input, bounds);

    float h, s, v;
    nk_color_hsv_f(&h, &s, &v, nk->style.window.background);
    struct nk_color background = nk_hsv_f(h, s, v + (hover ? 0.1f : -0.02f));
    struct nk_color border = nk_hsv_f(h, s, v + 0.20f);
    nk_fill_rect(canvas, bounds, 0.0f, background);
    nk_stroke_rect(canvas, bounds, 0.0f, 2.0f, border);

    const float pad = 10.0f;
    struct nk_rect text = {
        .x = bounds.x + pad,
        .y = bounds.y + pad,
        .w = bounds.w - 2 * pad,
        .h = bounds.h - 2 * pad,
    };

    nk_draw_text(canvas, text, label, nk_strlen(label), nk->style.font,
                 background, nk->style.text.color);

    return hover;
}
