#include <stdatomic.h>
#include <getopt.h>

#include <libavutil/file.h>

#include "plplay.h"

#ifdef PL_HAVE_WIN32
#include <shlwapi.h>
#define PL_BASENAME PathFindFileNameA
#define strdup _strdup
#else
#include <libgen.h>
#define PL_BASENAME basename
#endif

#ifdef HAVE_NUKLEAR
#include "ui.h"

bool parse_args(struct plplay_args *args, int argc, char *argv[])
{
    static struct option long_options[] = {
        {"verbose", no_argument,        NULL, 'v'},
        {"quiet",   no_argument,        NULL, 'q'},
        {"preset",  required_argument,  NULL, 'p'},
        {"hwdec",   no_argument,        NULL, 'H'},
        {"window",  required_argument,  NULL, 'w'},
        {0}
    };

    int option;
    while ((option = getopt_long(argc, argv, "vqp:Hw:", long_options, NULL)) != -1) {
        switch (option) {
            case 'v':
                if (args->verbosity < PL_LOG_TRACE)
                    args->verbosity++;
                break;
            case 'q':
                if (args->verbosity > PL_LOG_NONE)
                    args->verbosity--;
                break;
            case 'p':
                if (!strcmp(optarg, "default")) {
                    args->preset = &pl_render_default_params;
                } else if (!strcmp(optarg, "fast")) {
                    args->preset = &pl_render_fast_params;
                } else if (!strcmp(optarg, "highquality") || !strcmp(optarg, "hq")) {
                    args->preset = &pl_render_high_quality_params;
                } else {
                    fprintf(stderr, "Invalid value for -p/--preset: '%s'\n", optarg);
                    goto error;
                }
                break;
            case 'H':
                args->hwdec = true;
                break;
            case 'w':
                args->window_impl = optarg;
                break;
            case '?':
            default:
                goto error;
        }
    }

    // Check for the required filename argument
    if (optind < argc) {
        args->filename = argv[optind++];
    } else {
        fprintf(stderr, "Missing filename!\n");
        goto error;
    }

    if (optind != argc) {
        fprintf(stderr, "Superfluous argument: %s\n", argv[optind]);
        goto error;
    }

    return true;

error:
    fprintf(stderr, "Usage: %s [-v/--verbose] [-q/--quiet] [-p/--preset <default|fast|hq|highquality>] [--hwdec] [-w/--window <api>] <filename>\n", argv[0]);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -v, --verbose   Increase verbosity\n");
    fprintf(stderr, "  -q, --quiet     Decrease verbosity\n");
    fprintf(stderr, "  -p, --preset    Set the rendering preset (default|fast|hq|highquality)\n");
    fprintf(stderr, "  -H, --hwdec     Enable hardware decoding\n");
    fprintf(stderr, "  -w, --window    Specify the windowing API\n");
    return false;
}

static void add_hook(struct plplay *p, const struct pl_hook *hook, const char *path)
{
    if (!hook)
        return;

    if (p->shader_num == p->shader_size) {
        // Grow array if needed
        size_t new_size = p->shader_size ? p->shader_size * 2 : 16;
        void *new_hooks = realloc(p->shader_hooks, new_size * sizeof(void *));
        if (!new_hooks)
            goto error;
        p->shader_hooks = new_hooks;
        char **new_paths = realloc(p->shader_paths, new_size * sizeof(char *));
        if (!new_paths)
            goto error;
        p->shader_paths = new_paths;
        p->shader_size = new_size;
    }

    // strip leading path
    while (true) {
        const char *fname = strchr(path, '/');
        if (!fname)
            break;
        path = fname + 1;
    }

    char *path_copy = strdup(path);
    if (!path_copy)
        goto error;

    p->shader_hooks[p->shader_num] = hook;
    p->shader_paths[p->shader_num] = path_copy;
    p->shader_num++;
    return;

error:
    pl_mpv_user_shader_destroy(&hook);
}

static void auto_property_int(struct nk_context *nk, int auto_val, int min, int *val,
                         int max, int step, float inc_per_pixel)
{
    int value = *val;
    if (!value)
        value = auto_val;

    // Auto label will be delayed 1 frame
    nk_property_int(nk, *val ? "" : "Auto", min, &value, max, step, inc_per_pixel);

    if (*val || value != auto_val)
        *val = value;
}

static void draw_shader_pass(struct nk_context *nk,
                             const struct pl_dispatch_info *info)
{
    pl_shader_info shader = info->shader;

    char label[128];
    int count = snprintf(label, sizeof(label), "%.3f/%.3f/%.3f ms: %s",
             info->last / 1e6,
             info->average / 1e6,
             info->peak / 1e6,
             shader->description);

    if (count >= sizeof(label)) {
        label[sizeof(label) - 4] = '.';
        label[sizeof(label) - 3] = '.';
        label[sizeof(label) - 2] = '.';
    }

    int id = (unsigned int) (uintptr_t) info; // pointer into `struct plplay`
    if (nk_tree_push_id(nk, NK_TREE_NODE, label, NK_MINIMIZED, id)) {
        nk_layout_row_dynamic(nk, 32, 1);
        if (nk_chart_begin(nk, NK_CHART_LINES,
                           info->num_samples,
                           0.0f, info->peak))
        {
            for (int k = 0; k < info->num_samples; k++)
                nk_chart_push(nk, info->samples[k]);
            nk_chart_end(nk);
        }

        nk_layout_row_dynamic(nk, 24, 1);
        for (int n = 0; n < shader->num_steps; n++)
            nk_labelf(nk, NK_TEXT_LEFT, "%d. %s", n + 1, shader->steps[n]);
        nk_tree_pop(nk);
    }
}

static void draw_timing(struct nk_context *nk, const char *label,
                        const struct timing *t)
{
    const double avg = t->count ? t->sum / t->count : 0.0;
    const double stddev = t->count ? sqrt(t->sum2 / t->count - avg * avg) : 0.0;
    nk_label(nk, label, NK_TEXT_LEFT);
    nk_labelf(nk, NK_TEXT_LEFT, "%.4f ± %.4f ms (%.3f ms)",
              avg * 1e3, stddev * 1e3, t->peak * 1e3);
}

static void draw_opt_data(void *priv, pl_opt_data data)
{
    struct nk_context *nk = priv;
    pl_opt opt = data->opt;
    if (opt->type == PL_OPT_FLOAT) {
        // Print floats less verbosely than the libplacebo built-in printf
        nk_labelf(nk, NK_TEXT_LEFT, "%s = %f", opt->key, *(const float *) data->value);
    } else {
        nk_labelf(nk, NK_TEXT_LEFT, "%s = %s", opt->key, data->text);
    }
}

static void draw_cache_line(void *priv, pl_cache_obj obj)
{
    struct nk_context *nk = priv;
    nk_labelf(nk, NK_TEXT_LEFT, " - 0x%016"PRIx64": %zu bytes", obj.key, obj.size);
}

void update_settings(struct plplay *p, const struct pl_frame *target)
{
    struct nk_context *nk = ui_get_context(p->ui);
    enum nk_panel_flags win_flags = NK_WINDOW_BORDER | NK_WINDOW_MOVABLE |
                                    NK_WINDOW_SCALABLE | NK_WINDOW_MINIMIZABLE |
                                    NK_WINDOW_TITLE;

    ui_update_input(p->ui, p->win);
    const char *dropped_file = window_get_file(p->win);

    pl_options opts = p->opts;
    struct pl_render_params *par = &opts->params;

    if (nk_begin(nk, "Settings", nk_rect(100, 100, 600, 600), win_flags)) {

        if (nk_tree_push(nk, NK_TREE_NODE, "Window settings", NK_MAXIMIZED)) {
            nk_layout_row_dynamic(nk, 24, 2);

            bool fullscreen = window_is_fullscreen(p->win);
            p->toggle_fullscreen = nk_checkbox_label(nk, "Fullscreen", &fullscreen);
            nk_property_float(nk, "Corner rounding", 0.0, &par->corner_rounding, 1.0, 0.1, 0.01);

            struct nk_colorf bg = {
                par->background_color[0],
                par->background_color[1],
                par->background_color[2],
                1.0 - par->background_transparency,
            };

            nk_layout_row_dynamic(nk, 24, 2);
            nk_label(nk, "Background color:", NK_TEXT_LEFT);
            if (nk_combo_begin_color(nk, nk_rgb_cf(bg), nk_vec2(nk_widget_width(nk), 300))) {
                nk_layout_row_dynamic(nk, 200, 1);
                nk_color_pick(nk, &bg, NK_RGBA);
                nk_combo_end(nk);

                par->background_color[0] = bg.r;
                par->background_color[1] = bg.g;
                par->background_color[2] = bg.b;
                par->background_transparency = 1.0 - bg.a;
            }

            nk_layout_row_dynamic(nk, 24, 2);
            par->blend_against_tiles = nk_check_label(nk, "Blend against tiles", par->blend_against_tiles);
            nk_property_int(nk, "Tile size", 2, &par->tile_size, 256, 1, 1);

            nk_layout_row(nk, NK_DYNAMIC, 24, 3, (float[]){ 0.4, 0.3, 0.3 });
            nk_label(nk, "Tile colors:", NK_TEXT_LEFT);
            for (int i = 0; i < 2; i++) {
                bg = (struct nk_colorf) {
                    par->tile_colors[i][0],
                    par->tile_colors[i][1],
                    par->tile_colors[i][2],
                };

                if (nk_combo_begin_color(nk, nk_rgb_cf(bg), nk_vec2(nk_widget_width(nk), 300))) {
                    nk_layout_row_dynamic(nk, 200, 1);
                    nk_color_pick(nk, &bg, NK_RGB);
                    nk_combo_end(nk);

                    par->tile_colors[i][0] = bg.r;
                    par->tile_colors[i][1] = bg.g;
                    par->tile_colors[i][2] = bg.b;
                }
            }

            static const char *rotations[4] = {
                [PL_ROTATION_0]   = "0°",
                [PL_ROTATION_90]  = "90°",
                [PL_ROTATION_180] = "180°",
                [PL_ROTATION_270] = "270°",
            };

            nk_layout_row_dynamic(nk, 24, 2);
            nk_label(nk, "Display orientation:", NK_TEXT_LEFT);
            p->target_rot = nk_combo(nk, rotations, 4, p->target_rot,
                                     16, nk_vec2(nk_widget_width(nk), 100));
            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Image scaling", NK_MAXIMIZED)) {
            const struct pl_filter_config *f;
            static const char *scale_none = "None (Built-in sampling)";
            static const char *pscale_none = "None (Use regular upscaler)";
            static const char *tscale_none = "None (No frame mixing)";
            #define SCALE_DESC(scaler, fallback) (par->scaler ? par->scaler->description : fallback)

            static const char *zoom_modes[ZOOM_COUNT] = {
                [ZOOM_PAD]        = "Pad to window",
                [ZOOM_CROP]       = "Crop to window",
                [ZOOM_STRETCH]    = "Stretch to window",
                [ZOOM_FIT]        = "Fit inside window",
                [ZOOM_RAW]        = "Unscaled (raw)",
                [ZOOM_400]        = "400% zoom",
                [ZOOM_200]        = "200% zoom",
                [ZOOM_100]        = "100% zoom",
                [ZOOM_50]         = " 50% zoom",
                [ZOOM_25]         = " 25% zoom",
            };

            nk_layout_row(nk, NK_DYNAMIC, 24, 2, (float[]){ 0.3, 0.7 });
            nk_label(nk, "Zoom mode:", NK_TEXT_LEFT);
            int zoom = nk_combo(nk, zoom_modes, ZOOM_COUNT, p->target_zoom, 16, nk_vec2(nk_widget_width(nk), 500));
            if (zoom != p->target_zoom) {
                // Image crop may change
                pl_renderer_flush_cache(p->renderer);
                p->target_zoom = zoom;
            }

            nk_label(nk, "Upscaler:", NK_TEXT_LEFT);
            if (nk_combo_begin_label(nk, SCALE_DESC(upscaler, scale_none), nk_vec2(nk_widget_width(nk), 500))) {
                nk_layout_row_dynamic(nk, 16, 1);
                if (nk_combo_item_label(nk, scale_none, NK_TEXT_LEFT))
                    par->upscaler = NULL;
                for (int i = 0; i < pl_num_filter_configs; i++) {
                    f = pl_filter_configs[i];
                    if (!f->description)
                        continue;
                    if (!(f->allowed & PL_FILTER_UPSCALING))
                        continue;
                    if (!p->advanced_scalers && !(f->recommended & PL_FILTER_UPSCALING))
                        continue;
                    if (nk_combo_item_label(nk, f->description, NK_TEXT_LEFT))
                        par->upscaler = f;
                }
                nk_combo_end(nk);
            }

            nk_label(nk, "Downscaler:", NK_TEXT_LEFT);
            if (nk_combo_begin_label(nk, SCALE_DESC(downscaler, scale_none), nk_vec2(nk_widget_width(nk), 500))) {
                nk_layout_row_dynamic(nk, 16, 1);
                if (nk_combo_item_label(nk, scale_none, NK_TEXT_LEFT))
                    par->downscaler = NULL;
                for (int i = 0; i < pl_num_filter_configs; i++) {
                    f = pl_filter_configs[i];
                    if (!f->description)
                        continue;
                    if (!(f->allowed & PL_FILTER_DOWNSCALING))
                        continue;
                    if (!p->advanced_scalers && !(f->recommended & PL_FILTER_DOWNSCALING))
                        continue;
                    if (nk_combo_item_label(nk, f->description, NK_TEXT_LEFT))
                        par->downscaler = f;
                }
                nk_combo_end(nk);
            }

            nk_label(nk, "Plane scaler:", NK_TEXT_LEFT);
            if (nk_combo_begin_label(nk, SCALE_DESC(plane_upscaler, pscale_none), nk_vec2(nk_widget_width(nk), 500))) {
                nk_layout_row_dynamic(nk, 16, 1);
                if (nk_combo_item_label(nk, pscale_none, NK_TEXT_LEFT))
                    par->downscaler = NULL;
                for (int i = 0; i < pl_num_filter_configs; i++) {
                    f = pl_filter_configs[i];
                    if (!f->description)
                        continue;
                    if (!(f->allowed & PL_FILTER_UPSCALING))
                        continue;
                    if (!p->advanced_scalers && !(f->recommended & PL_FILTER_UPSCALING))
                        continue;
                    if (nk_combo_item_label(nk, f->description, NK_TEXT_LEFT))
                        par->plane_upscaler = f;
                }
                nk_combo_end(nk);
            }

            nk_label(nk, "Frame mixing:", NK_TEXT_LEFT);
            if (nk_combo_begin_label(nk, SCALE_DESC(frame_mixer, tscale_none), nk_vec2(nk_widget_width(nk), 300))) {
                nk_layout_row_dynamic(nk, 16, 1);
                if (nk_combo_item_label(nk, tscale_none, NK_TEXT_LEFT))
                    par->frame_mixer = NULL;
                for (int i = 0; i < pl_num_filter_configs; i++) {
                    f = pl_filter_configs[i];
                    if (!f->description)
                        continue;
                    if (!(f->allowed & PL_FILTER_FRAME_MIXING))
                        continue;
                    if (!p->advanced_scalers && !(f->recommended & PL_FILTER_FRAME_MIXING))
                        continue;
                    if (nk_combo_item_label(nk, f->description, NK_TEXT_LEFT))
                        par->frame_mixer = f;
                }
                nk_combo_end(nk);
            }

            nk_layout_row_dynamic(nk, 24, 2);
            par->skip_anti_aliasing = !nk_check_label(nk, "Anti-aliasing", !par->skip_anti_aliasing);
            nk_property_float(nk, "Antiringing", 0, &par->antiringing_strength, 1.0, 0.05, 0.001);

            struct pl_sigmoid_params *spar = &opts->sigmoid_params;
            nk_layout_row_dynamic(nk, 24, 2);
            par->sigmoid_params = nk_check_label(nk, "Sigmoidization", par->sigmoid_params) ? spar : NULL;
            if (nk_button_label(nk, "Default values"))
                *spar = pl_sigmoid_default_params;
            nk_property_float(nk, "Sigmoid center", 0, &spar->center, 1, 0.1, 0.01);
            nk_property_float(nk, "Sigmoid slope", 0, &spar->slope, 100, 1, 0.1);
            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Deinterlacing", NK_MINIMIZED)) {
            struct pl_deinterlace_params *dpar = &opts->deinterlace_params;
            nk_layout_row_dynamic(nk, 24, 2);
            par->deinterlace_params = nk_check_label(nk, "Enable", par->deinterlace_params) ? dpar : NULL;
            if (nk_button_label(nk, "Reset settings"))
                *dpar = pl_deinterlace_default_params;

            static const char *deint_algos[PL_DEINTERLACE_ALGORITHM_COUNT] = {
                [PL_DEINTERLACE_WEAVE]  = "Field weaving (no-op)",
                [PL_DEINTERLACE_BOB]    = "Naive bob (line doubling)",
                [PL_DEINTERLACE_YADIF]  = "Yadif (\"yet another deinterlacing filter\")",
            };

            nk_label(nk, "Deinterlacing algorithm", NK_TEXT_LEFT);
            dpar->algo = nk_combo(nk, deint_algos, PL_DEINTERLACE_ALGORITHM_COUNT,
                                  dpar->algo, 16, nk_vec2(nk_widget_width(nk), 300));

            switch (dpar->algo) {
            case PL_DEINTERLACE_WEAVE:
            case PL_DEINTERLACE_BOB:
                break;
            case PL_DEINTERLACE_YADIF:
                nk_checkbox_label(nk, "Skip spatial check", &dpar->skip_spatial_check);
                break;
            default: abort();
            }
            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Debanding", NK_MINIMIZED)) {
            struct pl_deband_params *dpar = &opts->deband_params;
            nk_layout_row_dynamic(nk, 24, 2);
            par->deband_params = nk_check_label(nk, "Enable", par->deband_params) ? dpar : NULL;
            if (nk_button_label(nk, "Reset settings"))
                *dpar = pl_deband_default_params;
            nk_property_int(nk, "Iterations", 0, &dpar->iterations, 8, 1, 0);
            nk_property_float(nk, "Threshold", 0, &dpar->threshold, 256, 1, 0.5);
            nk_property_float(nk, "Radius", 0, &dpar->radius, 256, 1, 0.2);
            nk_property_float(nk, "Grain", 0, &dpar->grain, 512, 1, 0.5);
            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Distortion", NK_MINIMIZED)) {
            struct pl_distort_params *dpar = &opts->distort_params;
            nk_layout_row_dynamic(nk, 24, 2);
            par->distort_params = nk_check_label(nk, "Enable", par->distort_params) ? dpar : NULL;
            if (nk_button_label(nk, "Reset settings"))
                *dpar = pl_distort_default_params;

            static const char *address_modes[PL_TEX_ADDRESS_MODE_COUNT] = {
                [PL_TEX_ADDRESS_CLAMP]  = "Clamp edges",
                [PL_TEX_ADDRESS_REPEAT] = "Repeat edges",
                [PL_TEX_ADDRESS_MIRROR] = "Mirror edges",
            };

            nk_checkbox_label(nk, "Constrain bounds", &dpar->constrain);
            dpar->address_mode = nk_combo(nk, address_modes, PL_TEX_ADDRESS_MODE_COUNT,
                                          dpar->address_mode, 16, nk_vec2(nk_widget_width(nk), 100));
            bool alpha = nk_check_label(nk, "Transparent background", dpar->alpha_mode);
            dpar->alpha_mode = alpha ? PL_ALPHA_INDEPENDENT : PL_ALPHA_UNKNOWN;
            nk_checkbox_label(nk, "Bicubic interpolation", &dpar->bicubic);

            struct pl_transform2x2 *tf = &dpar->transform;
            nk_property_float(nk, "Scale X", -10.0, &tf->mat.m[0][0], 10.0, 0.1, 0.005);
            nk_property_float(nk, "Shear X", -10.0, &tf->mat.m[0][1], 10.0, 0.1, 0.005);
            nk_property_float(nk, "Shear Y", -10.0, &tf->mat.m[1][0], 10.0, 0.1, 0.005);
            nk_property_float(nk, "Scale Y", -10.0, &tf->mat.m[1][1], 10.0, 0.1, 0.005);
            nk_property_float(nk, "Offset X", -10.0, &tf->c[0], 10.0, 0.1, 0.005);
            nk_property_float(nk, "Offset Y", -10.0, &tf->c[1], 10.0, 0.1, 0.005);

            float zoom_ref = fabsf(tf->mat.m[0][0] * tf->mat.m[1][1] -
                                   tf->mat.m[0][1] * tf->mat.m[1][0]);
            zoom_ref = logf(fmaxf(zoom_ref, 1e-4));
            float zoom = zoom_ref;
            nk_property_float(nk, "log(Zoom)", -10.0, &zoom, 10.0, 0.1, 0.005);
            pl_transform2x2_scale(tf, expf(zoom - zoom_ref));

            float angle_ref = (atan2f(tf->mat.m[1][0], tf->mat.m[1][1]) -
                               atan2f(tf->mat.m[0][1], tf->mat.m[0][0])) / 2;
            angle_ref = fmodf(angle_ref * 180/M_PI + 540, 360) - 180;
            float angle = angle_ref;
            nk_property_float(nk, "Rotate (°)", -200, &angle, 200, -5, -0.2);
            float angle_delta = (angle - angle_ref) * M_PI / 180;
            const pl_matrix2x2 rot = pl_matrix2x2_rotation(angle_delta);
            pl_matrix2x2_rmul(&rot, &tf->mat);

            bool flip_ox = nk_button_label(nk, "Flip output X");
            bool flip_oy = nk_button_label(nk, "Flip output Y");
            bool flip_ix = nk_button_label(nk, "Flip input X");
            bool flip_iy = nk_button_label(nk, "Flip input Y");
            if (flip_ox ^ flip_ix)
                tf->mat.m[0][0] = -tf->mat.m[0][0];
            if (flip_ox ^ flip_iy)
                tf->mat.m[0][1] = -tf->mat.m[0][1];
            if (flip_oy ^ flip_ix)
                tf->mat.m[1][0] = -tf->mat.m[1][0];
            if (flip_oy ^ flip_iy)
                tf->mat.m[1][1] = -tf->mat.m[1][1];
            if (flip_ox)
                tf->c[0] = -tf->c[0];
            if (flip_oy)
                tf->c[1] = -tf->c[1];

            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Color adjustment", NK_MINIMIZED)) {
            struct pl_color_adjustment *adj = &opts->color_adjustment;
            nk_layout_row_dynamic(nk, 24, 2);
            par->color_adjustment = nk_check_label(nk, "Enable", par->color_adjustment) ? adj : NULL;
            if (nk_button_label(nk, "Default values"))
                *adj = pl_color_adjustment_neutral;
            nk_property_float(nk, "Brightness", -1, &adj->brightness, 1, 0.1, 0.005);
            nk_property_float(nk, "Contrast", 0, &adj->contrast, 10, 0.1, 0.005);

            // Convert to (cyclical) degrees for display
            int deg = roundf(adj->hue * 180.0 / M_PI);
            nk_property_int(nk, "Hue (°)", -50, &deg, 400, 1, 1);
            adj->hue = ((deg + 360) % 360) * M_PI / 180.0;

            nk_property_float(nk, "Saturation", 0, &adj->saturation, 10, 0.1, 0.005);
            nk_property_float(nk, "Gamma", 0, &adj->gamma, 10, 0.1, 0.005);

            // Convert to human-friendly temperature values for display
            int temp = (int) roundf(adj->temperature * 3500) + 6500;
            nk_property_int(nk, "Temperature (K)", 3000, &temp, 10000, 10, 5);
            adj->temperature = (temp - 6500) / 3500.0;

            struct pl_cone_params *cpar = &opts->cone_params;
            nk_layout_row_dynamic(nk, 24, 2);
            par->cone_params = nk_check_label(nk, "Color blindness", par->cone_params) ? cpar : NULL;
            if (nk_button_label(nk, "Default values"))
                *cpar = pl_vision_normal;
            nk_layout_row(nk, NK_DYNAMIC, 24, 5, (float[]){ 0.25, 0.25/3, 0.25/3, 0.25/3, 0.5 });
            nk_label(nk, "Cone model:", NK_TEXT_LEFT);
            unsigned int cones = cpar->cones;
            nk_checkbox_flags_label(nk, "L", &cones, PL_CONE_L);
            nk_checkbox_flags_label(nk, "M", &cones, PL_CONE_M);
            nk_checkbox_flags_label(nk, "S", &cones, PL_CONE_S);
            cpar->cones = cones;
            nk_property_float(nk, "Sensitivity", 0.0, &cpar->strength, 5.0, 0.1, 0.01);
            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "HDR peak detection", NK_MINIMIZED)) {
            struct pl_peak_detect_params *ppar = &opts->peak_detect_params;
            nk_layout_row_dynamic(nk, 24, 2);
            par->peak_detect_params = nk_check_label(nk, "Enable", par->peak_detect_params) ? ppar : NULL;
            if (nk_button_label(nk, "Reset settings"))
                *ppar = pl_peak_detect_default_params;
            nk_property_float(nk, "Threshold low", 0.0, &ppar->scene_threshold_low, 20.0, 0.5, 0.005);
            nk_property_float(nk, "Threshold high", 0.0, &ppar->scene_threshold_high, 20.0, 0.5, 0.005);
            nk_property_float(nk, "Smoothing period", 0.0, &ppar->smoothing_period, 1000.0, 5.0, 1.0);
            nk_property_float(nk, "Peak percentile", 95.0, &ppar->percentile, 100.0, 0.01, 0.001);
            nk_property_float(nk, "Black cutoff", 0.0, &ppar->black_cutoff, 100.0, 0.01, 0.001);
            nk_checkbox_label(nk, "Allow 1-frame delay", &ppar->allow_delayed);

            struct pl_hdr_metadata metadata;
            if (pl_renderer_get_hdr_metadata(p->renderer, &metadata)) {
                nk_layout_row_dynamic(nk, 24, 2);
                nk_label(nk, "Detected max luminance:", NK_TEXT_LEFT);
                nk_labelf(nk, NK_TEXT_LEFT, "%.2f cd/m² (%.2f%% PQ)",
                          pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NITS, metadata.max_pq_y),
                          100.0f * metadata.max_pq_y);
                nk_label(nk, "Detected avg luminance:", NK_TEXT_LEFT);
                nk_labelf(nk, NK_TEXT_LEFT, "%.2f cd/m² (%.2f%% PQ)",
                          pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NITS, metadata.avg_pq_y),
                          100.0f * metadata.avg_pq_y);
            }

            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Tone mapping", NK_MINIMIZED)) {
            struct pl_color_map_params *cpar = &opts->color_map_params;
            static const struct pl_color_map_params null_settings = {0};
            nk_layout_row_dynamic(nk, 24, 2);
            par->color_map_params = nk_check_label(nk, "Enable",
                par->color_map_params == cpar) ? cpar : &null_settings;
            if (nk_button_label(nk, "Reset settings"))
                *cpar = pl_color_map_default_params;

            nk_label(nk, "Gamut mapping function:", NK_TEXT_LEFT);
            if (nk_combo_begin_label(nk, cpar->gamut_mapping->description,
                                     nk_vec2(nk_widget_width(nk), 500)))
            {
                nk_layout_row_dynamic(nk, 16, 1);
                for (int i = 0; i < pl_num_gamut_map_functions; i++) {
                    const struct pl_gamut_map_function *f = pl_gamut_map_functions[i];
                    if (nk_combo_item_label(nk, f->description, NK_TEXT_LEFT))
                        cpar->gamut_mapping = f;
                }
                nk_combo_end(nk);
            }

            nk_label(nk, "Tone mapping function:", NK_TEXT_LEFT);
            if (nk_combo_begin_label(nk, cpar->tone_mapping_function->description,
                                     nk_vec2(nk_widget_width(nk), 500)))
            {
                nk_layout_row_dynamic(nk, 16, 1);
                for (int i = 0; i < pl_num_tone_map_functions; i++) {
                    const struct pl_tone_map_function *f = pl_tone_map_functions[i];
                    if (nk_combo_item_label(nk, f->description, NK_TEXT_LEFT))
                        cpar->tone_mapping_function = f;
                }
                nk_combo_end(nk);
            }

            static const char *metadata_types[PL_HDR_METADATA_TYPE_COUNT] = {
                [PL_HDR_METADATA_ANY]               = "Automatic selection",
                [PL_HDR_METADATA_NONE]              = "None (disabled)",
                [PL_HDR_METADATA_HDR10]             = "HDR10 (static)",
                [PL_HDR_METADATA_HDR10PLUS]         = "HDR10+ (MaxRGB)",
                [PL_HDR_METADATA_CIE_Y]             = "Luminance (CIE Y)",
            };

            nk_label(nk, "HDR metadata source:", NK_TEXT_LEFT);
            cpar->metadata = nk_combo(nk, metadata_types,
                                      PL_HDR_METADATA_TYPE_COUNT,
                                      cpar->metadata,
                                      16, nk_vec2(nk_widget_width(nk), 300));

            nk_property_float(nk, "Contrast recovery", 0.0, &cpar->contrast_recovery, 2.0, 0.05, 0.005);
            nk_property_float(nk, "Contrast smoothness", 1.0, &cpar->contrast_smoothness, 32.0, 0.1, 0.005);

            nk_property_int(nk, "LUT size", 16, &cpar->lut_size, 1024, 1, 1);
            nk_property_int(nk, "3DLUT size I", 7, &cpar->lut3d_size[0], 65, 1, 1);
            nk_property_int(nk, "3DLUT size C", 7, &cpar->lut3d_size[1], 256, 1, 1);
            nk_property_int(nk, "3DLUT size h", 7, &cpar->lut3d_size[2], 1024, 1, 1);

            nk_checkbox_label(nk, "Tricubic interpolation", &cpar->lut3d_tricubic);
            nk_checkbox_label(nk, "Force full LUT", &cpar->force_tone_mapping_lut);
            nk_checkbox_label(nk, "Inverse tone mapping", &cpar->inverse_tone_mapping);
            nk_checkbox_label(nk, "Gamut expansion", &cpar->gamut_expansion);
            nk_checkbox_label(nk, "Show clipping", &cpar->show_clipping);
            nk_checkbox_label(nk, "Visualize LUT", &cpar->visualize_lut);

            if (cpar->visualize_lut) {
                nk_layout_row_dynamic(nk, 24, 2);
                const float huerange = 2 * M_PI;
                nk_property_float(nk, "Hue",   -1, &cpar->visualize_hue, huerange + 1.0, 0.1, 0.01);
                nk_property_float(nk, "Theta", 0.0, &cpar->visualize_theta, M_PI_2, 0.1, 0.01);
                cpar->visualize_hue = fmodf(cpar->visualize_hue + huerange, huerange);
            }

            if (nk_tree_push(nk, NK_TREE_NODE, "Fine-tune constants (advanced)", NK_MINIMIZED)) {
                struct pl_tone_map_constants  *tc = &cpar->tone_constants;
                struct pl_gamut_map_constants *gc = &cpar->gamut_constants;
                nk_layout_row_dynamic(nk, 20, 2);
                nk_property_float(nk, "Perceptual deadzone", 0.0, &gc->perceptual_deadzone, 1.0, 0.05, 0.001);
                nk_property_float(nk, "Perceptual strength", 0.0, &gc->perceptual_strength, 1.0, 0.05, 0.001);
                nk_property_float(nk, "Colorimetric gamma", 0.0, &gc->colorimetric_gamma, 10.0, 0.05, 0.001);
                nk_property_float(nk, "Softclip knee", 0.0, &gc->softclip_knee, 1.0, 0.05, 0.001);
                nk_property_float(nk, "Softclip desaturation", 0.0, &gc->softclip_desat, 1.0, 0.05, 0.001);
                nk_property_float(nk, "Knee adaptation", 0.0, &tc->knee_adaptation, 1.0, 0.05, 0.001);
                nk_property_float(nk, "Knee minimum", 0.0, &tc->knee_minimum, 0.5, 0.05, 0.001);
                nk_property_float(nk, "Knee maximum", 0.5, &tc->knee_maximum, 1.0, 0.05, 0.001);
                nk_property_float(nk, "Knee default", tc->knee_minimum, &tc->knee_default, tc->knee_maximum, 0.05, 0.001);
                nk_property_float(nk, "BT.2390 offset", 0.5, &tc->knee_offset, 2.0, 0.05, 0.001);
                nk_property_float(nk, "Spline slope tuning", 0.0, &tc->slope_tuning, 10.0, 0.05, 0.001);
                nk_property_float(nk, "Spline slope offset", 0.0, &tc->slope_offset, 1.0, 0.05, 0.001);
                nk_property_float(nk, "Spline contrast", 0.0, &tc->spline_contrast, 1.5, 0.05, 0.001);
                nk_property_float(nk, "Reinhard contrast", 0.0, &tc->reinhard_contrast, 1.0, 0.05, 0.001);
                nk_property_float(nk, "Linear knee point", 0.0, &tc->linear_knee, 1.0, 0.05, 0.001);
                nk_property_float(nk, "Linear exposure", 0.0, &tc->exposure, 10.0, 0.05, 0.001);
                nk_tree_pop(nk);
            }

            nk_layout_row_dynamic(nk, 50, 1);
            if (ui_widget_hover(nk, "Drop .cube file here...") && dropped_file) {
                uint8_t *buf;
                size_t size;
                int ret = av_file_map(dropped_file, &buf, &size, 0, NULL);
                if (ret < 0) {
                    fprintf(stderr, "Failed opening '%s': %s\n", dropped_file,
                            av_err2str(ret));
                } else {
                    pl_lut_free((struct pl_custom_lut **) &par->lut);
                    par->lut = pl_lut_parse_cube(p->log, (char *) buf, size);
                    av_file_unmap(buf, size);
                }
            }

            static const char *lut_types[] = {
                [PL_LUT_UNKNOWN]    = "Auto (unknown)",
                [PL_LUT_NATIVE]     = "Raw RGB (native)",
                [PL_LUT_NORMALIZED] = "Linear RGB (normalized)",
                [PL_LUT_CONVERSION] = "Gamut conversion (native)",
            };

            nk_layout_row(nk, NK_DYNAMIC, 24, 3, (float[]){ 0.2, 0.3, 0.5 });
            if (nk_button_label(nk, "Reset LUT")) {
                pl_lut_free((struct pl_custom_lut **) &par->lut);
                par->lut_type = PL_LUT_UNKNOWN;
            }

            nk_label(nk, "LUT type:", NK_TEXT_CENTERED);
            par->lut_type = nk_combo(nk, lut_types, 4, par->lut_type,
                                     16, nk_vec2(nk_widget_width(nk), 100));

            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Dithering", NK_MINIMIZED)) {
            struct pl_dither_params *dpar = &opts->dither_params;
            nk_layout_row_dynamic(nk, 24, 2);
            par->dither_params = nk_check_label(nk, "Enable", par->dither_params) ? dpar : NULL;
            if (nk_button_label(nk, "Reset settings"))
                *dpar = pl_dither_default_params;

            static const char *dither_methods[PL_DITHER_METHOD_COUNT] = {
                [PL_DITHER_BLUE_NOISE]      = "Blue noise",
                [PL_DITHER_ORDERED_LUT]     = "Ordered (LUT)",
                [PL_DITHER_ORDERED_FIXED]   = "Ordered (fixed size)",
                [PL_DITHER_WHITE_NOISE]     = "White noise",
            };

            nk_label(nk, "Dither method:", NK_TEXT_LEFT);
            dpar->method = nk_combo(nk, dither_methods, PL_DITHER_METHOD_COUNT, dpar->method,
                                    16, nk_vec2(nk_widget_width(nk), 100));

            static const char *lut_sizes[8] = {
                "2x2", "4x4", "8x8", "16x16", "32x32", "64x64", "128x128", "256x256",
            };

            nk_label(nk, "LUT size:", NK_TEXT_LEFT);
            switch (dpar->method) {
            case PL_DITHER_BLUE_NOISE:
            case PL_DITHER_ORDERED_LUT: {
                int size = dpar->lut_size - 1;
                nk_combobox(nk, lut_sizes, 8, &size, 16, nk_vec2(nk_widget_width(nk), 200));
                dpar->lut_size = size + 1;
                break;
            }
            case PL_DITHER_ORDERED_FIXED:
                nk_label(nk, "64x64", NK_TEXT_LEFT);
                break;
            default:
                nk_label(nk, "(N/A)", NK_TEXT_LEFT);
                break;
            }

            nk_checkbox_label(nk, "Temporal dithering", &dpar->temporal);

            nk_layout_row_dynamic(nk, 24, 2);
            nk_label(nk, "Error diffusion:", NK_TEXT_LEFT);
            const char *name = par->error_diffusion ? par->error_diffusion->description : "(None)";
            if (nk_combo_begin_label(nk, name, nk_vec2(nk_widget_width(nk), 500))) {
                nk_layout_row_dynamic(nk, 16, 1);
                if (nk_combo_item_label(nk, "(None)", NK_TEXT_LEFT))
                    par->error_diffusion = NULL;
                for (int i = 0; i < pl_num_error_diffusion_kernels; i++) {
                    const struct pl_error_diffusion_kernel *k = pl_error_diffusion_kernels[i];
                    if (nk_combo_item_label(nk, k->description, NK_TEXT_LEFT))
                        par->error_diffusion = k;
                }
                nk_combo_end(nk);
            }

            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Output color space", NK_MINIMIZED)) {
            nk_layout_row_dynamic(nk, 24, 2);
            nk_checkbox_label(nk, "Enable", &p->target_override);
            bool reset = nk_button_label(nk, "Reset settings");
            bool reset_icc = reset;
            char buf[64] = {0};

            nk_layout_row(nk, NK_DYNAMIC, 24, 2, (float[]){ 0.3, 0.7 });

            const char *primaries[PL_COLOR_PRIM_COUNT] = {
                [PL_COLOR_PRIM_UNKNOWN]     = "Auto (unknown)",
                [PL_COLOR_PRIM_BT_601_525]  = "ITU-R Rec. BT.601 (525-line = NTSC, SMPTE-C)",
                [PL_COLOR_PRIM_BT_601_625]  = "ITU-R Rec. BT.601 (625-line = PAL, SECAM)",
                [PL_COLOR_PRIM_BT_709]      = "ITU-R Rec. BT.709 (HD), also sRGB",
                [PL_COLOR_PRIM_BT_470M]     = "ITU-R Rec. BT.470 M",
                [PL_COLOR_PRIM_EBU_3213]    = "EBU Tech. 3213-E / JEDEC P22 phosphors",
                [PL_COLOR_PRIM_BT_2020]     = "ITU-R Rec. BT.2020 (UltraHD)",
                [PL_COLOR_PRIM_APPLE]       = "Apple RGB",
                [PL_COLOR_PRIM_ADOBE]       = "Adobe RGB (1998)",
                [PL_COLOR_PRIM_PRO_PHOTO]   = "ProPhoto RGB (ROMM)",
                [PL_COLOR_PRIM_CIE_1931]    = "CIE 1931 RGB primaries",
                [PL_COLOR_PRIM_DCI_P3]      = "DCI-P3 (Digital Cinema)",
                [PL_COLOR_PRIM_DISPLAY_P3]  = "DCI-P3 (Digital Cinema) with D65 white point",
                [PL_COLOR_PRIM_V_GAMUT]     = "Panasonic V-Gamut (VARICAM)",
                [PL_COLOR_PRIM_S_GAMUT]     = "Sony S-Gamut",
                [PL_COLOR_PRIM_FILM_C]      = "Traditional film primaries with Illuminant C",
                [PL_COLOR_PRIM_ACES_AP0]    = "ACES Primaries #0",
                [PL_COLOR_PRIM_ACES_AP1]    = "ACES Primaries #1",
            };

            if (target->color.primaries) {
                snprintf(buf, sizeof(buf), "Auto (%s)", primaries[target->color.primaries]);
                primaries[PL_COLOR_PRIM_UNKNOWN] = buf;
            }

            nk_label(nk, "Primaries:", NK_TEXT_LEFT);
            p->force_prim = nk_combo(nk, primaries, PL_COLOR_PRIM_COUNT, p->force_prim,
                                       16, nk_vec2(nk_widget_width(nk), 200));

            const char *transfers[PL_COLOR_TRC_COUNT] = {
                [PL_COLOR_TRC_UNKNOWN]      = "Auto (unknown SDR)",
                [PL_COLOR_TRC_BT_1886]      = "ITU-R Rec. BT.1886 (CRT emulation + OOTF)",
                [PL_COLOR_TRC_SRGB]         = "IEC 61966-2-4 sRGB (CRT emulation)",
                [PL_COLOR_TRC_LINEAR]       = "Linear light content",
                [PL_COLOR_TRC_GAMMA18]      = "Pure power gamma 1.8",
                [PL_COLOR_TRC_GAMMA20]      = "Pure power gamma 2.0",
                [PL_COLOR_TRC_GAMMA22]      = "Pure power gamma 2.2",
                [PL_COLOR_TRC_GAMMA24]      = "Pure power gamma 2.4",
                [PL_COLOR_TRC_GAMMA26]      = "Pure power gamma 2.6",
                [PL_COLOR_TRC_GAMMA28]      = "Pure power gamma 2.8",
                [PL_COLOR_TRC_PRO_PHOTO]    = "ProPhoto RGB (ROMM)",
                [PL_COLOR_TRC_ST428]        = "Digital Cinema Distribution Master (XYZ)",
                [PL_COLOR_TRC_PQ]           = "ITU-R BT.2100 PQ (perceptual quantizer), aka SMPTE ST2048",
                [PL_COLOR_TRC_HLG]          = "ITU-R BT.2100 HLG (hybrid log-gamma), aka ARIB STD-B67",
                [PL_COLOR_TRC_V_LOG]        = "Panasonic V-Log (VARICAM)",
                [PL_COLOR_TRC_S_LOG1]       = "Sony S-Log1",
                [PL_COLOR_TRC_S_LOG2]       = "Sony S-Log2",
            };

            if (target->color.transfer) {
                snprintf(buf, sizeof(buf), "Auto (%s)", transfers[target->color.transfer]);
                transfers[PL_COLOR_TRC_UNKNOWN] = buf;
            }

            nk_label(nk, "Transfer:", NK_TEXT_LEFT);
            p->force_trc = nk_combo(nk, transfers, PL_COLOR_TRC_COUNT, p->force_trc,
                                      16, nk_vec2(nk_widget_width(nk), 200));

            nk_layout_row_dynamic(nk, 24, 2);
            nk_checkbox_label(nk, "Override HDR levels", &p->force_hdr_enable);

            // Ensure these values are always legal by going through
            // pl_color_space_infer
            nk_layout_row_dynamic(nk, 24, 2);
            struct pl_color_space fix = target->color;
            apply_csp_overrides(p, &fix);
            pl_color_space_infer(&fix);

            fix.hdr.min_luma *= 1000; // better value range
            nk_property_float(nk, "White point (cd/m²)",
                                10.0, &fix.hdr.max_luma, 10000.0,
                                fix.hdr.max_luma / 100, fix.hdr.max_luma / 1000);
            nk_property_float(nk, "Black point (mcd/m²)",
                                PL_COLOR_HDR_BLACK * 1000, &fix.hdr.min_luma,
                                100.0 * 1000, 5, 2);
            fix.hdr.min_luma /= 1000;
            pl_color_space_infer(&fix);
            p->force_hdr = fix.hdr;

            struct pl_color_repr *trepr = &p->force_repr;
            nk_layout_row(nk, NK_DYNAMIC, 24, 2, (float[]){ 0.3, 0.7 });

            const char *systems[PL_COLOR_SYSTEM_COUNT] = {
                [PL_COLOR_SYSTEM_UNKNOWN]       = "Auto (unknown)",
                [PL_COLOR_SYSTEM_BT_601]        = "ITU-R Rec. BT.601 (SD)",
                [PL_COLOR_SYSTEM_BT_709]        = "ITU-R Rec. BT.709 (HD)",
                [PL_COLOR_SYSTEM_SMPTE_240M]    = "SMPTE-240M",
                [PL_COLOR_SYSTEM_BT_2020_NC]    = "ITU-R Rec. BT.2020 (non-constant luminance)",
                [PL_COLOR_SYSTEM_BT_2020_C]     = "ITU-R Rec. BT.2020 (constant luminance)",
                [PL_COLOR_SYSTEM_BT_2100_PQ]    = "ITU-R Rec. BT.2100 ICtCp PQ variant",
                [PL_COLOR_SYSTEM_BT_2100_HLG]   = "ITU-R Rec. BT.2100 ICtCp HLG variant",
                [PL_COLOR_SYSTEM_DOLBYVISION]   = "Dolby Vision (invalid for output)",
                [PL_COLOR_SYSTEM_YCGCO]         = "YCgCo (derived from RGB)",
                [PL_COLOR_SYSTEM_RGB]           = "Red, Green and Blue",
                [PL_COLOR_SYSTEM_XYZ]           = "Digital Cinema Distribution Master (XYZ)",
            };

            if (target->repr.sys) {
                snprintf(buf, sizeof(buf), "Auto (%s)", systems[target->repr.sys]);
                systems[PL_COLOR_SYSTEM_UNKNOWN] = buf;
            }

            nk_label(nk, "System:", NK_TEXT_LEFT);
            trepr->sys = nk_combo(nk, systems, PL_COLOR_SYSTEM_COUNT, trepr->sys,
                                  16, nk_vec2(nk_widget_width(nk), 200));
            if (trepr->sys == PL_COLOR_SYSTEM_DOLBYVISION)
                trepr->sys = PL_COLOR_SYSTEM_UNKNOWN;

            const char *levels[PL_COLOR_LEVELS_COUNT] = {
                [PL_COLOR_LEVELS_UNKNOWN]   = "Auto (unknown)",
                [PL_COLOR_LEVELS_LIMITED]   = "Limited/TV range, e.g. 16-235",
                [PL_COLOR_LEVELS_FULL]      = "Full/PC range, e.g. 0-255",
            };

            if (target->repr.levels) {
                snprintf(buf, sizeof(buf), "Auto (%s)", levels[target->repr.levels]);
                levels[PL_COLOR_LEVELS_UNKNOWN] = buf;
            }

            nk_label(nk, "Levels:", NK_TEXT_LEFT);
            trepr->levels = nk_combo(nk, levels, PL_COLOR_LEVELS_COUNT, trepr->levels,
                                     16, nk_vec2(nk_widget_width(nk), 200));

            const char *alphas[PL_ALPHA_MODE_COUNT] = {
                [PL_ALPHA_UNKNOWN]          = "Auto (unknown, or no alpha)",
                [PL_ALPHA_INDEPENDENT]      = "Independent alpha channel",
                [PL_ALPHA_PREMULTIPLIED]    = "Premultiplied alpha channel",
            };

            if (target->repr.alpha) {
                snprintf(buf, sizeof(buf), "Auto (%s)", alphas[target->repr.alpha]);
                alphas[PL_ALPHA_UNKNOWN] = buf;
            }

            nk_label(nk, "Alpha:", NK_TEXT_LEFT);
            trepr->alpha = nk_combo(nk, alphas, PL_ALPHA_MODE_COUNT, trepr->alpha,
                                    16, nk_vec2(nk_widget_width(nk), 200));

            const struct pl_bit_encoding *bits = &target->repr.bits;
            nk_label(nk, "Bit depth:", NK_TEXT_LEFT);
            auto_property_int(nk, bits->color_depth, 0,
                              &trepr->bits.color_depth, 16, 1, 0);

            if (bits->color_depth != bits->sample_depth) {
                nk_label(nk, "Sample bit depth:", NK_TEXT_LEFT);
                auto_property_int(nk, bits->sample_depth, 0,
                                  &trepr->bits.sample_depth, 16, 1, 0);
            } else {
                // Adjust these two fields in unison
                trepr->bits.sample_depth = trepr->bits.color_depth;
            }

            if (bits->bit_shift) {
                nk_label(nk, "Bit shift:", NK_TEXT_LEFT);
                auto_property_int(nk, bits->bit_shift, 0,
                                  &trepr->bits.bit_shift, 16, 1, 0);
            } else {
                trepr->bits.bit_shift = 0;
            }

            nk_layout_row_dynamic(nk, 24, 1);
            nk_checkbox_label(nk, "Forward input color space to display", &p->colorspace_hint);

            if (p->colorspace_hint && !p->force_hdr_enable) {
                nk_checkbox_label(nk, "Forward dynamic brightness changes to display",
                                  &p->colorspace_hint_dynamic);
            }

            nk_layout_row_dynamic(nk, 50, 1);
            if (ui_widget_hover(nk, "Drop ICC profile here...") && dropped_file) {
                struct pl_icc_profile profile;
                int ret = av_file_map(dropped_file, (uint8_t **) &profile.data,
                                      &profile.len, 0, NULL);
                if (ret < 0) {
                    fprintf(stderr, "Failed opening '%s': %s\n", dropped_file,
                            av_err2str(ret));
                } else {
                    free(p->icc_name);
                    pl_icc_profile_compute_signature(&profile);
                    pl_icc_update(p->log, &p->icc, &profile, pl_icc_params(
                        .force_bpc = p->force_bpc,
                        .max_luma  = p->use_icc_luma ? 0 : PL_COLOR_SDR_WHITE,
                    ));
                    av_file_unmap((void *) profile.data, profile.len);
                    if (p->icc)
                        p->icc_name = strdup(PL_BASENAME((char *) dropped_file));
                }
            }

            if (p->icc) {
                nk_layout_row_dynamic(nk, 24, 2);
                nk_labelf(nk, NK_TEXT_LEFT, "Loaded: %s",
                          p->icc_name ? p->icc_name : "(unknown)");
                reset_icc |= nk_button_label(nk, "Reset ICC");
                nk_checkbox_label(nk, "Force BPC", &p->force_bpc);
                nk_checkbox_label(nk, "Use detected luminance", &p->use_icc_luma);
            }

            // Apply the reset last to prevent the UI from flashing for a frame
            if (reset) {
                p->force_repr = (struct pl_color_repr) {0};
                p->force_prim = PL_COLOR_PRIM_UNKNOWN;
                p->force_trc = PL_COLOR_TRC_UNKNOWN;
                p->force_hdr = (struct pl_hdr_metadata) {0};
                p->force_hdr_enable = false;
            }

            if (reset_icc && p->icc) {
                pl_icc_close(&p->icc);
                free(p->icc_name);
                p->icc_name = NULL;
            }

            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Custom shaders", NK_MINIMIZED)) {

            nk_layout_row_dynamic(nk, 50, 1);
            if (ui_widget_hover(nk, "Drop .hook/.glsl files here...") && dropped_file) {
                uint8_t *buf;
                size_t size;
                int ret = av_file_map(dropped_file, &buf, &size, 0, NULL);
                if (ret < 0) {
                    fprintf(stderr, "Failed opening '%s': %s\n", dropped_file,
                            av_err2str(ret));
                } else {
                    const struct pl_hook *hook;
                    hook = pl_mpv_user_shader_parse(p->win->gpu, (char *) buf, size);
                    av_file_unmap(buf, size);
                    add_hook(p, hook, dropped_file);
                }
            }

            const float px = 24.0;
            nk_layout_row_template_begin(nk, px);
            nk_layout_row_template_push_static(nk, px);
            nk_layout_row_template_push_static(nk, px);
            nk_layout_row_template_push_static(nk, px);
            nk_layout_row_template_push_dynamic(nk);
            nk_layout_row_template_end(nk);
            for (int i = 0; i < p->shader_num; i++) {

                if (i == 0) {
                    nk_label(nk, "·", NK_TEXT_CENTERED);
                } else if (nk_button_symbol(nk, NK_SYMBOL_TRIANGLE_UP)) {
                    const struct pl_hook *prev_hook = p->shader_hooks[i - 1];
                    char *prev_path = p->shader_paths[i - 1];
                    p->shader_hooks[i - 1] = p->shader_hooks[i];
                    p->shader_paths[i - 1] = p->shader_paths[i];
                    p->shader_hooks[i] = prev_hook;
                    p->shader_paths[i] = prev_path;
                }

                if (i == p->shader_num - 1) {
                    nk_label(nk, "·", NK_TEXT_CENTERED);
                } else if (nk_button_symbol(nk, NK_SYMBOL_TRIANGLE_DOWN)) {
                    const struct pl_hook *next_hook = p->shader_hooks[i + 1];
                    char *next_path = p->shader_paths[i + 1];
                    p->shader_hooks[i + 1] = p->shader_hooks[i];
                    p->shader_paths[i + 1] = p->shader_paths[i];
                    p->shader_hooks[i] = next_hook;
                    p->shader_paths[i] = next_path;
                }

                if (nk_button_symbol(nk, NK_SYMBOL_X)) {
                    pl_mpv_user_shader_destroy(&p->shader_hooks[i]);
                    free(p->shader_paths[i]);
                    p->shader_num--;
                    memmove(&p->shader_hooks[i], &p->shader_hooks[i+1],
                            (p->shader_num - i) * sizeof(void *));
                    memmove(&p->shader_paths[i], &p->shader_paths[i+1],
                            (p->shader_num - i) * sizeof(char *));
                    if (i == p->shader_num)
                        break;
                }

                if (p->shader_hooks[i]->num_parameters == 0) {
                    nk_label(nk, p->shader_paths[i], NK_TEXT_LEFT);
                    continue;
                }

                if (nk_combo_begin_label(nk, p->shader_paths[i], nk_vec2(nk_widget_width(nk), 500))) {
                    nk_layout_row_dynamic(nk, 32, 1);
                    for (int j = 0; j < p->shader_hooks[i]->num_parameters; j++) {
                        const struct pl_hook_par *hp = &p->shader_hooks[i]->parameters[j];
                        const char *name = hp->description ? hp->description : hp->name;
                        switch (hp->type) {
                        case PL_VAR_FLOAT:
                            nk_property_float(nk, name, hp->minimum.f,
                                              &hp->data->f, hp->maximum.f,
                                              hp->data->f / 100.0f,
                                              hp->data->f / 1000.0f);
                            break;
                        case PL_VAR_SINT:
                            nk_property_int(nk, name, hp->minimum.i,
                                            &hp->data->i, hp->maximum.i,
                                            1, 1.0f);
                            break;
                        case PL_VAR_UINT: {
                            int min = FFMIN(hp->minimum.u, INT_MAX);
                            int max = FFMIN(hp->maximum.u, INT_MAX);
                            int val = FFMIN(hp->data->u, INT_MAX);
                            nk_property_int(nk, name, min, &val, max, 1, 1);
                            hp->data->u = val;
                            break;
                        }
                        default: abort();
                        }
                    }
                    nk_combo_end(nk);
                }
            }

            par->hooks = p->shader_hooks;
            par->num_hooks = p->shader_num;
            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Debug", NK_MINIMIZED)) {
            nk_layout_row_dynamic(nk, 24, 1);
            nk_checkbox_label(nk, "Preserve mixing cache", &par->preserve_mixing_cache);
            nk_checkbox_label(nk, "Bypass mixing cache", &par->skip_caching_single_frame);
            nk_checkbox_label(nk, "Show all scaler presets", &p->advanced_scalers);
            nk_checkbox_label(nk, "Disable linear scaling", &par->disable_linear_scaling);
            nk_checkbox_label(nk, "Disable built-in scalers", &par->disable_builtin_scalers);
            nk_checkbox_label(nk, "Correct subpixel offsets", &par->correct_subpixel_offsets);
            nk_checkbox_label(nk, "Force-enable dither", &par->force_dither);
            nk_checkbox_label(nk, "Disable gamma-aware dither", &par->disable_dither_gamma_correction);
            nk_checkbox_label(nk, "Disable FBOs / advanced rendering", &par->disable_fbos);
            nk_checkbox_label(nk, "Force low-bit depth FBOs", &par->force_low_bit_depth_fbos);
            nk_checkbox_label(nk, "Disable constant hard-coding", &par->dynamic_constants);

            if (nk_check_label(nk, "Ignore Dolby Vision metadata", p->ignore_dovi) != p->ignore_dovi) {
                // Flush the renderer cache on changes, since this can
                // drastically alter the subjective appearance of the stream
                pl_renderer_flush_cache(p->renderer);
                p->ignore_dovi = !p->ignore_dovi;
            }

            nk_layout_row_dynamic(nk, 24, 2);

            double prev_fps = p->fps;
            bool fps_changed = nk_checkbox_label(nk, "Override display FPS", &p->fps_override);
            nk_property_float(nk, "FPS", 10.0, &p->fps, 240.0, 5, 0.1);
            if (fps_changed || p->fps != prev_fps)
                p->stats.pts_interval = p->stats.vsync_interval = (struct timing) {0};

            if (nk_button_label(nk, "Flush renderer cache"))
                pl_renderer_flush_cache(p->renderer);
            if (nk_button_label(nk, "Recreate renderer")) {
                pl_renderer_destroy(&p->renderer);
                p->renderer = pl_renderer_create(p->log, p->win->gpu);
            }

            if (nk_tree_push(nk, NK_TREE_NODE, "Shader passes / GPU timing", NK_MINIMIZED)) {
                nk_layout_row_dynamic(nk, 26, 1);
                nk_label(nk, "Full frames:", NK_TEXT_LEFT);
                for (int i = 0; i < p->num_frame_passes; i++)
                    draw_shader_pass(nk, &p->frame_info[i]);

                nk_layout_row_dynamic(nk, 26, 1);
                nk_label(nk, "Output blending:", NK_TEXT_LEFT);
                for (int j = 0; j < MAX_BLEND_FRAMES; j++) {
                    for (int i = 0; i < p->num_blend_passes[j]; i++)
                        draw_shader_pass(nk, &p->blend_info[j][i]);
                }

                nk_tree_pop(nk);
            }

            if (nk_tree_push(nk, NK_TREE_NODE, "Frame statistics / CPU timing", NK_MINIMIZED)) {
                nk_layout_row_dynamic(nk, 24, 2);
                nk_label(nk, "Current PTS:", NK_TEXT_LEFT);
                nk_labelf(nk, NK_TEXT_LEFT, "%.3f", p->stats.current_pts);
                nk_label(nk, "Estimated FPS:", NK_TEXT_LEFT);
                nk_labelf(nk, NK_TEXT_LEFT, "%.3f", pl_queue_estimate_fps(p->queue));
                nk_label(nk, "Estimated vsync rate:", NK_TEXT_LEFT);
                nk_labelf(nk, NK_TEXT_LEFT, "%.3f", pl_queue_estimate_vps(p->queue));
                nk_label(nk, "PTS drift offset:", NK_TEXT_LEFT);
                nk_labelf(nk, NK_TEXT_LEFT, "%.3f ms", 1e3 * pl_queue_pts_offset(p->queue));
                nk_label(nk, "Frames rendered:", NK_TEXT_LEFT);
                nk_labelf(nk, NK_TEXT_LEFT, "%"PRIu32, p->stats.rendered);
                nk_label(nk, "Decoded frames", NK_TEXT_LEFT);
                nk_labelf(nk, NK_TEXT_LEFT, "%"PRIu32, atomic_load(&p->stats.decoded));
                nk_label(nk, "Dropped frames:", NK_TEXT_LEFT);
                nk_labelf(nk, NK_TEXT_LEFT, "%"PRIu32, p->stats.dropped);
                nk_label(nk, "Missed timestamps:", NK_TEXT_LEFT);
                nk_labelf(nk, NK_TEXT_LEFT, "%"PRIu32" (%.3f ms)",
                          p->stats.missed, p->stats.missed_ms);
                nk_label(nk, "Times stalled:", NK_TEXT_LEFT);
                nk_labelf(nk, NK_TEXT_LEFT, "%"PRIu32" (%.3f ms)",
                          p->stats.stalled, p->stats.stalled_ms);
                draw_timing(nk, "Acquire FBO:", &p->stats.acquire);
                draw_timing(nk, "Update queue:", &p->stats.update);
                draw_timing(nk, "Render frame:", &p->stats.render);
                draw_timing(nk, "Draw interface:", &p->stats.draw_ui);
                draw_timing(nk, "Voluntary sleep:", &p->stats.sleep);
                draw_timing(nk, "Submit frame:", &p->stats.submit);
                draw_timing(nk, "Swap buffers:", &p->stats.swap);
                draw_timing(nk, "Vsync interval:", &p->stats.vsync_interval);
                draw_timing(nk, "PTS interval:", &p->stats.pts_interval);

                if (nk_button_label(nk, "Reset statistics"))
                    memset(&p->stats, 0, sizeof(p->stats));
                nk_tree_pop(nk);
            }

            if (nk_tree_push(nk, NK_TREE_NODE, "Settings dump", NK_MINIMIZED)) {

                nk_layout_row_dynamic(nk, 24, 2);
                if (nk_button_label(nk, "Copy to clipboard"))
                    window_set_clipboard(p->win, pl_options_save(opts));
                if (nk_button_label(nk, "Load from clipboard"))
                    pl_options_load(opts, window_get_clipboard(p->win));

                nk_layout_row_dynamic(nk, 24, 1);
                pl_options_iterate(opts, draw_opt_data, nk);
                nk_tree_pop(nk);
            }

            if (nk_tree_push(nk, NK_TREE_NODE, "Cache statistics", NK_MINIMIZED)) {
                nk_layout_row_dynamic(nk, 24, 2);
                nk_label(nk, "Cached objects:", NK_TEXT_LEFT);
                nk_labelf(nk, NK_TEXT_LEFT, "%d", pl_cache_objects(p->cache));
                nk_label(nk, "Total size:", NK_TEXT_LEFT);
                nk_labelf(nk, NK_TEXT_LEFT, "%zu", pl_cache_size(p->cache));
                nk_label(nk, "Maximum total size:", NK_TEXT_LEFT);
                nk_labelf(nk, NK_TEXT_LEFT, "%zu", p->cache->params.max_total_size);
                nk_label(nk, "Maximum object size:", NK_TEXT_LEFT);
                nk_labelf(nk, NK_TEXT_LEFT, "%zu", p->cache->params.max_object_size);

                if (nk_button_label(nk, "Clear cache"))
                    pl_cache_reset(p->cache);
                if (nk_button_label(nk, "Save cache")) {
                    FILE *file = fopen(p->cache_file, "wb");
                    if (file) {
                        pl_cache_save_file(p->cache, file);
                        fclose(file);
                    }
                }

                if (nk_tree_push(nk, NK_TREE_NODE, "Object list", NK_MINIMIZED)) {
                    nk_layout_row_dynamic(nk, 24, 1);
                    pl_cache_iterate(p->cache, draw_cache_line, nk);
                    nk_tree_pop(nk);
                }

                nk_tree_pop(nk);
            }

            nk_tree_pop(nk);
        }
    }
    nk_end(nk);
}

#else
void update_settings(struct plplay *p, const struct pl_frame *target) { }
#endif // HAVE_NUKLEAR
