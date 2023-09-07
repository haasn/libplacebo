// License: CC0 / Public Domain

#define NK_IMPLEMENTATION
#include "ui.h"

#include <libplacebo/dispatch.h>
#include <libplacebo/shaders/custom.h>

struct ui_vertex {
    float pos[2];
    float coord[2];
    uint8_t color[4];
};

#define NUM_VERTEX_ATTRIBS 3

struct ui {
    pl_gpu gpu;
    pl_dispatch dp;
    struct nk_context nk;
    struct nk_font_atlas atlas;
    struct nk_buffer cmds, verts, idx;
    pl_tex font_tex;
    struct pl_vertex_attrib attribs_pl[NUM_VERTEX_ATTRIBS];
    struct nk_draw_vertex_layout_element attribs_nk[NUM_VERTEX_ATTRIBS+1];
    struct nk_convert_config convert_cfg;
};

struct ui *ui_create(pl_gpu gpu)
{
    struct ui *ui = malloc(sizeof(struct ui));
    if (!ui)
        return NULL;

    *ui = (struct ui) {
        .gpu = gpu,
        .dp = pl_dispatch_create(gpu->log, gpu),
        .attribs_pl = {
            {
                .name = "pos",
                .offset = offsetof(struct ui_vertex, pos),
                .fmt = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 2),
            }, {
                .name = "coord",
                .offset = offsetof(struct ui_vertex, coord),
                .fmt = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 2),
            }, {
                .name = "vcolor",
                .offset = offsetof(struct ui_vertex, color),
                .fmt = pl_find_named_fmt(gpu, "rgba8"),
            }
        },
        .attribs_nk = {
            {NK_VERTEX_POSITION, NK_FORMAT_FLOAT, offsetof(struct ui_vertex, pos)},
            {NK_VERTEX_TEXCOORD, NK_FORMAT_FLOAT, offsetof(struct ui_vertex, coord)},
            {NK_VERTEX_COLOR, NK_FORMAT_R8G8B8A8, offsetof(struct ui_vertex, color)},
            {NK_VERTEX_LAYOUT_END}
        },
        .convert_cfg = {
            .vertex_layout = ui->attribs_nk,
            .vertex_size = sizeof(struct ui_vertex),
            .vertex_alignment = NK_ALIGNOF(struct ui_vertex),
            .shape_AA = NK_ANTI_ALIASING_ON,
            .line_AA = NK_ANTI_ALIASING_ON,
            .circle_segment_count = 22,
            .curve_segment_count = 22,
            .arc_segment_count = 22,
            .global_alpha = 1.0f,
        },
    };

    // Initialize font atlas using built-in font
    nk_font_atlas_init_default(&ui->atlas);
    nk_font_atlas_begin(&ui->atlas);
    struct nk_font *font = nk_font_atlas_add_default(&ui->atlas, 20, NULL);
    struct pl_tex_params tparams = {
        .format = pl_find_named_fmt(gpu, "r8"),
        .sampleable = true,
        .initial_data = nk_font_atlas_bake(&ui->atlas, &tparams.w, &tparams.h,
                                           NK_FONT_ATLAS_ALPHA8),
        .debug_tag = PL_DEBUG_TAG,
    };
    ui->font_tex = pl_tex_create(gpu, &tparams);
    nk_font_atlas_end(&ui->atlas, nk_handle_ptr((void *) ui->font_tex),
                      &ui->convert_cfg.tex_null);
    nk_font_atlas_cleanup(&ui->atlas);

    if (!ui->font_tex)
        goto error;

    // Initialize nuklear state
    if (!nk_init_default(&ui->nk, &font->handle)) {
        fprintf(stderr, "NK: failed initializing UI!\n");
        goto error;
    }

    nk_buffer_init_default(&ui->cmds);
    nk_buffer_init_default(&ui->verts);
    nk_buffer_init_default(&ui->idx);

    return ui;

error:
    ui_destroy(&ui);
    return NULL;
}

void ui_destroy(struct ui **ptr)
{
    struct ui *ui = *ptr;
    if (!ui)
        return;

    nk_buffer_free(&ui->cmds);
    nk_buffer_free(&ui->verts);
    nk_buffer_free(&ui->idx);
    nk_free(&ui->nk);
    nk_font_atlas_clear(&ui->atlas);
    pl_tex_destroy(ui->gpu, &ui->font_tex);
    pl_dispatch_destroy(&ui->dp);

    free(ui);
    *ptr = NULL;
}

void ui_update_input(struct ui *ui, const struct window *win)
{
    int x, y;
    window_get_cursor(win, &x, &y);
    nk_input_begin(&ui->nk);
    nk_input_motion(&ui->nk, x, y);
    nk_input_button(&ui->nk, NK_BUTTON_LEFT, x, y, window_get_button(win, BTN_LEFT));
    nk_input_button(&ui->nk, NK_BUTTON_RIGHT, x, y, window_get_button(win, BTN_RIGHT));
    nk_input_button(&ui->nk, NK_BUTTON_MIDDLE, x, y, window_get_button(win, BTN_MIDDLE));
    struct nk_vec2 scroll;
    window_get_scroll(win, &scroll.x, &scroll.y);
    nk_input_scroll(&ui->nk, scroll);
    nk_input_end(&ui->nk);
}

struct nk_context *ui_get_context(struct ui *ui)
{
    return &ui->nk;
}

bool ui_draw(struct ui *ui, const struct pl_swapchain_frame *frame)
{
    if (nk_convert(&ui->nk, &ui->cmds, &ui->verts, &ui->idx, &ui->convert_cfg) != NK_CONVERT_SUCCESS) {
        fprintf(stderr, "NK: failed converting draw commands!\n");
        return false;
    }

    const struct nk_draw_command *cmd = NULL;
    const uint8_t *vertices = nk_buffer_memory(&ui->verts);
    const nk_draw_index *indices = nk_buffer_memory(&ui->idx);
    nk_draw_foreach(cmd, &ui->nk, &ui->cmds) {
        if (!cmd->elem_count)
            continue;

        pl_shader sh = pl_dispatch_begin(ui->dp);
        pl_shader_custom(sh, &(struct pl_custom_shader) {
            .description = "nuklear UI",
            .body = "color = textureLod(ui_tex, coord, 0.0).r * vcolor;",
            .output = PL_SHADER_SIG_COLOR,
            .num_descriptors = 1,
            .descriptors = &(struct pl_shader_desc) {
                .desc = {
                    .name = "ui_tex",
                    .type = PL_DESC_SAMPLED_TEX,
                },
                .binding = {
                    .object = cmd->texture.ptr,
                    .sample_mode = PL_TEX_SAMPLE_NEAREST,
                },
            },
        });

        struct pl_color_repr repr = frame->color_repr;
        pl_shader_color_map_ex(sh, NULL, pl_color_map_args(
            .src = pl_color_space_srgb,
            .dst = frame->color_space,
        ));
        pl_shader_encode_color(sh, &repr);

        bool ok = pl_dispatch_vertex(ui->dp, pl_dispatch_vertex_params(
            .shader = &sh,
            .target = frame->fbo,
            .blend_params = &pl_alpha_overlay,
            .scissors = {
                .x0 = cmd->clip_rect.x,
                .y0 = cmd->clip_rect.y,
                .x1 = cmd->clip_rect.x + cmd->clip_rect.w,
                .y1 = cmd->clip_rect.y + cmd->clip_rect.h,
            },
            .vertex_attribs = ui->attribs_pl,
            .num_vertex_attribs = NUM_VERTEX_ATTRIBS,
            .vertex_stride = sizeof(struct ui_vertex),
            .vertex_position_idx = 0,
            .vertex_coords = PL_COORDS_ABSOLUTE,
            .vertex_flipped = frame->flipped,
            .vertex_type = PL_PRIM_TRIANGLE_LIST,
            .vertex_count = cmd->elem_count,
            .vertex_data = vertices,
            .index_data = indices,
            .index_fmt = PL_INDEX_UINT32,
        ));

        if (!ok) {
            fprintf(stderr, "placebo: failed rendering UI!\n");
            return false;
        }

        indices += cmd->elem_count;
    }

    nk_clear(&ui->nk);
    nk_buffer_clear(&ui->cmds);
    nk_buffer_clear(&ui->verts);
    nk_buffer_clear(&ui->idx);
    return true;
}
