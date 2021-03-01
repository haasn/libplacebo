/* Compiling:
 *
 *   gcc nuklear.c glfw.c -o ./nuklear -O2 -DUSE_VK \
 *       $(pkg-config --cflags --libs glfw3 vulkan libplacebo)
 *
 *  or:
 *
 *   gcc nuklear.c glfw.c -o ./nuklear -O2 -DUSE_GL \
 *       $(pkg-config --cflags --libs glfw3 libplacebo)
 *
 * License: CC0 / Public Domain
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <libplacebo/dispatch.h>
#include <libplacebo/shaders/custom.h>

#define NK_IMPLEMENTATION
#define NK_PRIVATE
#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_BUTTON_TRIGGER_ON_RELEASE
#include <nuklear.h>

#include "glfw.h"

#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480

static struct pl_context *ctx;
static struct pl_dispatch *dp;
static struct winstate win;

// UI state
static struct nk_context nk;
static struct nk_font_atlas atlas;
static struct nk_buffer cmds, verts, idx;
static const struct pl_tex *font_tex;

struct ui_vertex {
    float pos[2];
    float coord[2];
    uint8_t color[4];
};

static struct pl_vertex_attrib vertex_attribs_pl[3] = {
    { .name = "pos",    .offset = offsetof(struct ui_vertex, pos), },
    { .name = "coord",  .offset = offsetof(struct ui_vertex, coord), },
    { .name = "vcolor", .offset = offsetof(struct ui_vertex, color), },
};

static const struct nk_draw_vertex_layout_element vertex_layout_nk[] = {
    {NK_VERTEX_POSITION, NK_FORMAT_FLOAT, offsetof(struct ui_vertex, pos)},
    {NK_VERTEX_TEXCOORD, NK_FORMAT_FLOAT, offsetof(struct ui_vertex, coord)},
    {NK_VERTEX_COLOR, NK_FORMAT_R8G8B8A8, offsetof(struct ui_vertex, color)},
    {NK_VERTEX_LAYOUT_END}
};

static struct nk_convert_config convert_cfg = {
    .vertex_layout = vertex_layout_nk,
    .vertex_size = sizeof(struct ui_vertex),
    .vertex_alignment = NK_ALIGNOF(struct ui_vertex),
    .shape_AA = NK_ANTI_ALIASING_ON,
    .line_AA = NK_ANTI_ALIASING_ON,
    .circle_segment_count = 22,
    .curve_segment_count = 22,
    .arc_segment_count = 22,
    .global_alpha = 1.0f,
};

static bool ui_init()
{
    const struct pl_gpu *gpu = win.gpu;
    dp = pl_dispatch_create(ctx, gpu);

    // Initialize font
    nk_font_atlas_init_default(&atlas);
    nk_font_atlas_begin(&atlas);
    struct nk_font *font = nk_font_atlas_add_default(&atlas, 25, NULL);
    struct pl_tex_params tparams = {
        .format = pl_find_named_fmt(gpu, "r8"),
        .sampleable = true,
        .initial_data = nk_font_atlas_bake(&atlas, &tparams.w, &tparams.h,
                                           NK_FONT_ATLAS_ALPHA8),
    };
    font_tex = pl_tex_create(gpu, &tparams);
    nk_font_atlas_end(&atlas, nk_handle_ptr((void *) font_tex), &convert_cfg.null);
    nk_font_atlas_cleanup(&atlas);

    if (!font_tex)
        return false;

    // Initialize UI state
    if (!nk_init_default(&nk, &font->handle)) {
        fprintf(stderr, "NK: failed initializing UI!\n");
        return false;
    }

    nk_buffer_init_default(&cmds);
    nk_buffer_init_default(&verts);
    nk_buffer_init_default(&idx);

    // Pick vertex formats
    vertex_attribs_pl[0].fmt = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 2);
    vertex_attribs_pl[1].fmt = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 2);
    vertex_attribs_pl[2].fmt = pl_find_named_fmt(gpu, "rgba8");
    return true;
}

static bool render(const struct pl_swapchain_frame *frame)
{
    const struct pl_gpu *gpu = win.gpu;

    // update input
    nk_input_begin(&nk);
    double x, y;
    glfwGetCursorPos(win.win, &x, &y);
    nk_input_motion(&nk, (int) x, (int) y);
    nk_input_button(&nk, NK_BUTTON_LEFT, (int) x, (int) y,
                    glfwGetMouseButton(win.win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    nk_input_end(&nk);

    // update UI
    enum nk_panel_flags win_flags = NK_WINDOW_BORDER | NK_WINDOW_MOVABLE |
        NK_WINDOW_SCALABLE | NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE;

    static struct nk_colorf background = { 0.0f, 0.0f, 0.0f, 1.0f };

    if (nk_begin(&nk, "Settings", nk_rect(100, 100, 500, 200), win_flags)) {
        nk_layout_row_dynamic(&nk, 20, 1);
        nk_label(&nk, "Window background:", NK_TEXT_LEFT);
        nk_layout_row_dynamic(&nk, 25, 1);
        if (nk_combo_begin_color(&nk, nk_rgb_cf(background), nk_vec2(nk_widget_width(&nk), 400))) {
            nk_layout_row_dynamic(&nk, 120, 1);
            nk_color_pick(&nk, &background, NK_RGB);
            nk_combo_end(&nk);
        }
    }
    nk_end(&nk);

    assert(frame->fbo->params.blit_dst);
    pl_tex_clear(gpu, frame->fbo, (const float *) &background.r);

    // draw UI
    if (nk_convert(&nk, &cmds, &verts, &idx, &convert_cfg) != NK_CONVERT_SUCCESS) {
        fprintf(stderr, "NK: failed converting draw commands!\n");
        return false;
    }

    const struct nk_draw_command *cmd = NULL;
    const uint8_t *vertices = nk_buffer_memory(&verts);
    const nk_draw_index *indices = nk_buffer_memory(&idx);
    nk_draw_foreach(cmd, &nk, &cmds) {
        if (!cmd->elem_count)
            continue;

        struct pl_shader *sh = pl_dispatch_begin(dp);
        pl_shader_custom(sh, &(struct pl_custom_shader) {
            .body = "color = texture(ui_tex, coord).r * vcolor;",
            .output = PL_SHADER_SIG_COLOR,
            .num_descriptors = 1,
            .descriptors = &(struct pl_shader_desc) {
                .desc = {
                    .name = "ui_tex",
                    .type = PL_DESC_SAMPLED_TEX,
                },
                .binding = {
                    .object = cmd->texture.ptr,
                    .sample_mode = PL_TEX_SAMPLE_LINEAR,
                },
            },
        });

        bool ok = pl_dispatch_vertex(dp, &(struct pl_dispatch_vertex_params) {
            .shader = &sh,
            .target = frame->fbo,
            .blend_params = &pl_alpha_overlay,
            .scissors = {
                .x0 = cmd->clip_rect.x,
                .y0 = cmd->clip_rect.y,
                .x1 = cmd->clip_rect.x + cmd->clip_rect.w,
                .y1 = cmd->clip_rect.y + cmd->clip_rect.h,
            },
            .vertex_attribs = vertex_attribs_pl,
            .num_vertex_attribs = sizeof(vertex_attribs_pl) / sizeof(vertex_attribs_pl[0]),
            .vertex_stride = sizeof(struct ui_vertex),
            .vertex_position_idx = 0,
            .vertex_coords = PL_COORDS_ABSOLUTE,
            .vertex_flipped = frame->flipped,
            .vertex_type = PL_PRIM_TRIANGLE_LIST,
            .vertex_count = cmd->elem_count,
            .vertex_data = vertices,
            .index_data = indices,
        });

        if (!ok) {
            fprintf(stderr, "placebo: failed rendering UI!\n");
            return false;
        }

        indices += cmd->elem_count;
    }

    nk_clear(&nk);
    nk_buffer_clear(&cmds);
    nk_buffer_clear(&verts);
    nk_buffer_clear(&idx);
    return true;
}

static void ui_uninit()
{
    nk_buffer_free(&cmds);
    nk_buffer_free(&verts);
    nk_buffer_free(&idx);
    nk_free(&nk);
    nk_font_atlas_clear(&atlas);
    pl_tex_destroy(win.gpu, &font_tex);
    pl_dispatch_destroy(&dp);
}

static void uninit(int ret)
{
    ui_uninit();
    glfw_uninit(&win);
    pl_context_destroy(&ctx);
    exit(ret);
}

int main(int argc, char **argv)
{
    ctx = pl_context_create(PL_API_VER, &(struct pl_context_params) {
        .log_cb    = pl_log_color,
#ifdef NDEBUG
        .log_level = PL_LOG_INFO,
#else
        .log_level = PL_LOG_DEBUG,
#endif
    });
    assert(ctx);

    if (!glfw_init(ctx, &win, WINDOW_WIDTH, WINDOW_HEIGHT, 0))
        uninit(1);

    if (!ui_init())
        uninit(1);

    while (!win.window_lost) {
        struct pl_swapchain_frame frame;
        bool ok = pl_swapchain_start_frame(win.swapchain, &frame);
        if (!ok) {
            glfwWaitEvents();
            continue;
        }

        if (!render(&frame))
            uninit(1);

        ok = pl_swapchain_submit_frame(win.swapchain);
        if (!ok) {
            fprintf(stderr, "libplacebo: failed submitting frame!\n");
            uninit(3);
        }

        pl_swapchain_swap_buffers(win.swapchain);
        glfwPollEvents();
    }

    uninit(0);
}
