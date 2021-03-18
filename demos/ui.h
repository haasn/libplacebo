#pragma once

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_BUTTON_TRIGGER_ON_RELEASE
#include <nuklear.h>

#include "common.h"
#include "window.h"

struct ui;

struct ui *ui_create(const struct pl_gpu *gpu);
void ui_destroy(struct ui **ui);

// Update/Logic/Draw cycle
void ui_update_input(struct ui *ui, const struct window *window);
struct nk_context *ui_get_context(struct ui *ui);
bool ui_draw(struct ui *ui, const struct pl_swapchain_frame *frame);
