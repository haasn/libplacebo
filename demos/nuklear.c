/* Trivial UI demo, very basic. Currently just lets you change the background
 * color of the window and nothing else. See `plplay.c` for a more complete UI.
 *
 * License: CC0 / Public Domain
 */

#include "common.h"
#include "window.h"
#include "ui.h"

static struct pl_context *ctx;
static struct window *win;
static struct ui *ui;

static bool render(const struct pl_swapchain_frame *frame)
{
    ui_update_input(ui, win);

    enum nk_panel_flags win_flags = NK_WINDOW_BORDER | NK_WINDOW_MOVABLE |
        NK_WINDOW_SCALABLE | NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE;

    static struct nk_colorf background = { 0.0f, 0.0f, 0.0f, 1.0f };

    struct nk_context *nk = ui_get_context(ui);
    if (nk_begin(nk, "Settings", nk_rect(100, 100, 500, 200), win_flags)) {
        nk_layout_row_dynamic(nk, 20, 1);
        nk_label(nk, "Window background:", NK_TEXT_LEFT);
        nk_layout_row_dynamic(nk, 25, 1);
        if (nk_combo_begin_color(nk, nk_rgb_cf(background), nk_vec2(nk_widget_width(nk), 400))) {
            nk_layout_row_dynamic(nk, 120, 1);
            nk_color_pick(nk, &background, NK_RGB);
            nk_combo_end(nk);
        }
    }
    nk_end(nk);

    assert(frame->fbo->params.blit_dst);
    pl_tex_clear(win->gpu, frame->fbo, (const float *) &background.r);

    return ui_draw(ui, frame);
}

static void uninit(int ret)
{
    ui_destroy(&ui);
    window_destroy(&win);
    pl_context_destroy(&ctx);
    exit(ret);
}

int main(int argc, char **argv)
{
    ctx = pl_context_create(PL_API_VER, &(struct pl_context_params) {
        .log_cb = pl_log_color,
        .log_level = PL_LOG_DEBUG,
    });

    win = window_create(ctx, "nuklear demo", 640, 480, 0);
    ui = win ? ui_create(win->gpu) : NULL;
    if (!win || !ui)
        uninit(1);

    while (!win->window_lost) {
        struct pl_swapchain_frame frame;
        bool ok = pl_swapchain_start_frame(win->swapchain, &frame);
        if (!ok) {
            window_poll(win, true);
            continue;
        }

        if (!render(&frame))
            uninit(1);

        ok = pl_swapchain_submit_frame(win->swapchain);
        if (!ok) {
            fprintf(stderr, "libplacebo: failed submitting frame!\n");
            uninit(3);
        }

        pl_swapchain_swap_buffers(win->swapchain);
        window_poll(win, false);
    }

    uninit(0);
}
