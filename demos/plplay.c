/* Very basic video player based on ffmpeg. All it does is render a single
 * video stream to completion, and then exits. It exits on most errors, rather
 * than gracefully trying to recreate the context.
 *
 * The timing code is also rather naive, due to the current lack of
 * presentation feedback. That being said, an effort is made to time the video
 * stream to the system clock, using frame mixing for mismatches.
 *
 * License: CC0 / Public Domain
 */

#include <pthread.h>
#include <time.h>

#include <libavutil/file.h>
#include <libavutil/pixdesc.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>

#include "common.h"
#include "window.h"

#ifdef HAVE_NUKLEAR
#include "ui.h"
#else
struct ui;
static void ui_destroy(struct ui **ui) {}
static bool ui_draw(struct ui *ui, const struct pl_swapchain_frame *frame) { return true; };
#endif

#include <libplacebo/renderer.h>
#include <libplacebo/shaders/lut.h>
#include <libplacebo/utils/libav.h>
#include <libplacebo/utils/frame_queue.h>

struct plplay {
    struct window *win;
    struct ui *ui;

    // libplacebo
    struct pl_context *ctx;
    struct pl_renderer *renderer;
    struct pl_queue *queue;

    // libav*
    AVFormatContext *format;
    AVCodecContext *codec;
    const AVStream *stream; // points to first video stream of `format`
    pthread_t decoder_thread;

    // settings / ui state
    const struct pl_filter_preset *upscaler, *downscaler, *frame_mixer;
    struct pl_render_params params;
    struct pl_deband_params deband_params;
    struct pl_sigmoid_params sigmoid_params;
    struct pl_color_adjustment color_adjustment;
    struct pl_peak_detect_params peak_detect_params;
    struct pl_color_map_params color_map_params;
    struct pl_dither_params dither_params;
    struct pl_cone_params cone_params;
    int force_depth;

    // custom shaders
    const struct pl_hook **shader_hooks;
    char **shader_paths;
    size_t shader_num;
    size_t shader_size;
};

static void uninit(struct plplay *p)
{
    if (p->decoder_thread) {
        pthread_cancel(p->decoder_thread);
        pthread_join(p->decoder_thread, NULL);
    }

    for (int i = 0; i < p->shader_num; i++) {
        pl_mpv_user_shader_destroy(&p->shader_hooks[i]);
        free(p->shader_paths[i]);
    }

    pl_queue_destroy(&p->queue);
    pl_renderer_destroy(&p->renderer);
    ui_destroy(&p->ui);
    window_destroy(&p->win);

    free(p->shader_hooks);
    free(p->shader_paths);

    avcodec_free_context(&p->codec);
    avformat_free_context(p->format);

    pl_context_destroy(&p->ctx);
    *p = (struct plplay) {0};
}

static bool open_file(struct plplay *p, const char *filename)
{
    printf("Opening file: '%s'\n", filename);
    if (avformat_open_input(&p->format, filename, NULL, NULL) != 0) {
        fprintf(stderr, "libavformat: Failed opening file!\n");
        return false;
    }

    printf("Format: %s\n", p->format->iformat->name);
    printf("Duration: %.3f s\n", p->format->duration / 1e6);

    if (avformat_find_stream_info(p->format,  NULL) < 0) {
        fprintf(stderr, "libavformat: Failed finding stream info!\n");
        return false;
    }

    // Find "best" video stream
    int stream_idx =
        av_find_best_stream(p->format, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);

    if (stream_idx < 0) {
        fprintf(stderr, "plplay: File contains no video streams?\n");
        return false;
    }

    const AVStream *stream = p->format->streams[stream_idx];
    const AVCodecParameters *par = stream->codecpar;
    printf("Found video track (stream %d)\n", stream_idx);
    printf("Resolution: %d x %d\n", par->width, par->height);
    printf("FPS: %f\n", av_q2d(stream->avg_frame_rate));
    printf("Bitrate: %"PRIi64" kbps\n", par->bit_rate / 1000);

    p->stream = stream;
    return true;
}

static inline bool is_file_hdr(struct plplay *p)
{
    assert(p->stream);
    enum AVColorTransferCharacteristic trc = p->stream->codecpar->color_trc;
    return pl_color_transfer_is_hdr(pl_transfer_from_av(trc));
}

static bool init_codec(struct plplay *p)
{
    assert(p->stream);

    const AVCodec *codec = avcodec_find_decoder(p->stream->codecpar->codec_id);
    if (!codec) {
        fprintf(stderr, "libavcodec: Failed finding matching codec\n");
        return false;
    }

    p->codec = avcodec_alloc_context3(codec);
    if (!p->codec) {
        fprintf(stderr, "libavcodec: Failed allocating codec\n");
        return false;
    }

    if (avcodec_parameters_to_context(p->codec, p->stream->codecpar) < 0) {
        fprintf(stderr, "libavcodec: Failed copying codec parameters to codec\n");
        return false;
    }

    p->codec->thread_count = av_cpu_count();

    if (avcodec_open2(p->codec, codec, NULL) < 0) {
        fprintf(stderr, "libavcodec: Failed opening codec\n");
        return false;
    }

    return true;
}

static bool map_frame(const struct pl_gpu *gpu, const struct pl_tex **tex,
                      const struct pl_source_frame *src,
                      struct pl_frame *out_frame)
{
    if (!pl_upload_avframe(gpu, out_frame, tex, src->frame_data))
        return false;

    out_frame->user_data = src->frame_data;
    return true;
}

static void unmap_frame(const struct pl_gpu *gpu, struct pl_frame *frame,
                        const struct pl_source_frame *src)
{
    av_frame_free((AVFrame **) &src->frame_data);
}

static void discard_frame(const struct pl_source_frame *src)
{
    av_frame_free((AVFrame **) &src->frame_data);
    printf("Dropped frame with PTS %.3f\n", src->pts);
}

static void *decode_loop(void *arg)
{
    int ret;
    struct plplay *p = arg;
    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    if (!packet || !frame)
        goto done;

    double start_pts;
    bool first_frame = true;

    while (true) {
        switch ((ret = av_read_frame(p->format, packet))) {
        case 0:
            if (packet->stream_index != p->stream->index) {
                // Ignore unrelated packets
                av_packet_unref(packet);
                continue;
            }
            ret = avcodec_send_packet(p->codec, packet);
            av_packet_unref(packet);
            break;
        case AVERROR_EOF:
            // Send empty input to flush decoder
            ret = avcodec_send_packet(p->codec, NULL);
            break;
        default:
            fprintf(stderr, "libavformat: Failed reading packet: %s\n",
                    av_err2str(ret));
            goto done;
        }

        if (ret < 0) {
            fprintf(stderr, "libavcodec: Failed sending packet to decoder: %s\n",
                    av_err2str(ret));
            goto done;
        }

        // Decode all frames from this packet
        while ((ret = avcodec_receive_frame(p->codec, frame)) == 0) {
            double pts = frame->pts * av_q2d(p->stream->time_base);
            if (first_frame) {
                start_pts = pts;
                first_frame = false;
            }

            pl_queue_push_block(p->queue, UINT64_MAX, &(struct pl_source_frame) {
                .pts = pts - start_pts,
                .map = map_frame,
                .unmap = unmap_frame,
                .discard = discard_frame,
                .frame_data = frame,
            });
            frame = av_frame_alloc();
        }

        switch (ret) {
        case AVERROR(EAGAIN):
            continue;
        case AVERROR_EOF:
            goto done;
        default:
            fprintf(stderr, "libavcodec: Failed decoding frame: %s\n",
                    av_err2str(ret));
            goto done;
        }
    }

done:
    pl_queue_push(p->queue, NULL); // Signal EOF to flush queue
    av_frame_free(&frame);
    av_packet_free(&packet);
    return NULL;
}

static void update_settings(struct plplay *p);

static bool render_frame(struct plplay *p, const struct pl_swapchain_frame *frame,
                         const struct pl_frame_mix *mix)
{
    struct pl_frame target;
    pl_frame_from_swapchain(&target, frame);
    update_settings(p);

    assert(mix->num_frames);
    const AVFrame *avframe = mix->frames[0]->user_data;
    double dar = pl_rect2df_aspect(&mix->frames[0]->crop);
    if (avframe->sample_aspect_ratio.num)
        dar *= av_q2d(avframe->sample_aspect_ratio);
    pl_rect2df_aspect_set(&target.crop, dar, 0.0);

    if (p->force_depth) {
        target.repr.bits.color_depth = p->force_depth;
        target.repr.bits.sample_depth = p->force_depth;
    }

    if (!pl_render_image_mix(p->renderer, mix, &target, &p->params))
        return false;

    if (!ui_draw(p->ui, frame))
        return false;

    return true;
}

static bool render_loop(struct plplay *p)
{
    struct pl_queue_params qparams = {
        .radius = pl_frame_mix_radius(&p->params),
        .frame_duration = av_q2d(av_inv_q(p->stream->avg_frame_rate)),
        .timeout = UINT64_MAX,
    };

    // Initialize the frame queue, blocking indefinitely until done
    struct pl_frame_mix mix;
    switch (pl_queue_update(p->queue, &mix, &qparams)) {
    case PL_QUEUE_OK:  break;
    case PL_QUEUE_EOF: return true;
    case PL_QUEUE_ERR: goto error;
    default: abort();
    }

    struct pl_swapchain_frame frame;
    if (!pl_swapchain_start_frame(p->win->swapchain, &frame))
        goto error;
    if (!render_frame(p, &frame, &mix))
        goto error;
    if (!pl_swapchain_submit_frame(p->win->swapchain))
        goto error;

    // Wait until rendering is complete. Do this before measuring the time
    // start, to ensure we don't count initialization overhead as part of the
    // first vsync.
    pl_gpu_finish(p->win->gpu);

    struct timespec ts_prev, ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts_prev) < 0) {
        fprintf(stderr, "%s\n", strerror(errno));
        goto error;
    }

    pl_swapchain_swap_buffers(p->win->swapchain);
    window_poll(p->win, false);

    double pts = 0.0;
    bool stuck = false;

    while (!p->win->window_lost) {
        if (!pl_swapchain_start_frame(p->win->swapchain, &frame)) {
            // Window stuck/invisible? Block for events and try again.
            window_poll(p->win, true);
            continue;
        }

retry:
        if (clock_gettime(CLOCK_MONOTONIC, &ts) < 0)
            goto error;

        if (!stuck) {
            pts += (ts.tv_sec - ts_prev.tv_sec) +
                   (ts.tv_nsec - ts_prev.tv_nsec) * 1e-9;
        }
        ts_prev = ts;

        qparams.timeout = 50000000; // 50 ms
        qparams.pts = pts;

        switch (pl_queue_update(p->queue, &mix, &qparams)) {
        case PL_QUEUE_ERR: goto error;
        case PL_QUEUE_EOF: return true;
        case PL_QUEUE_OK:
            if (!render_frame(p, &frame, &mix))
                goto error;
            stuck = false;
            break;
        case PL_QUEUE_MORE:
            stuck = true;
            goto retry;
        }

        if (!pl_swapchain_submit_frame(p->win->swapchain)) {
            fprintf(stderr, "libplacebo: failed presenting frame!\n");
            goto error;
        }

        pl_swapchain_swap_buffers(p->win->swapchain);
        window_poll(p->win, false);
    }

    return true;

error:
    fprintf(stderr, "Render loop failed, exiting early...\n");
    return false;
}

int main(int argc, char **argv)
{
    const char *filename;
    if (argc == 2) {
        filename = argv[1];
    } else {
        fprintf(stderr, "Usage: ./%s <filename>\n", argv[0]);
        return -1;
    }

    struct plplay state = {
        .params = pl_render_default_params,
        .deband_params = pl_deband_default_params,
        .sigmoid_params = pl_sigmoid_default_params,
        .color_adjustment = pl_color_adjustment_neutral,
        .peak_detect_params = pl_peak_detect_default_params,
        .color_map_params = pl_color_map_default_params,
        .dither_params = pl_dither_default_params,
        .cone_params = pl_vision_normal,
    };

    // Redirect all of the pointers in `params.default` to instead point to the
    // structs inside `struct plplay`, so we can adjust them using the UI
#define DEFAULT_PARAMS(field) \
        state.params.field = state.params.field ? &state.field : NULL
    DEFAULT_PARAMS(deband_params);
    DEFAULT_PARAMS(sigmoid_params);
    DEFAULT_PARAMS(peak_detect_params);
    DEFAULT_PARAMS(dither_params);
    state.params.color_adjustment = &state.color_adjustment;
    state.params.color_map_params = &state.color_map_params;
    state.params.cone_params = &state.cone_params;

    struct plplay *p = &state;
    if (!open_file(p, filename))
        goto error;

    const AVCodecParameters *par = p->stream->codecpar;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(par->format);
    if (!desc)
        goto error;

    enum winflags flags = 0;
    if (desc->flags & AV_PIX_FMT_FLAG_ALPHA)
        flags |= WIN_ALPHA;
    if (is_file_hdr(p))
        flags |= WIN_HDR;

    p->ctx = pl_context_create(PL_API_VER, &(struct pl_context_params) {
        .log_cb = pl_log_color,
        .log_level = PL_LOG_INFO,
    });

    p->win = window_create(p->ctx, "plplay", par->width, par->height, flags);
    if (!p->win)
        goto error;

#ifdef HAVE_NUKLEAR
    p->ui = ui_create(p->win->gpu);
    if (!p->ui)
        goto error;

    // Find the right named filter entries for the defaults
    const struct pl_filter_preset *f;
    for (f = pl_filter_presets; f->name; f++) {
        if (p->params.upscaler == f->filter)
            p->upscaler = f;
        if (p->params.downscaler == f->filter)
            p->downscaler = f;
    }

    for (f = pl_frame_mixers; f->name; f++) {
        if (p->params.frame_mixer == f->filter)
            p->frame_mixer = f;
    }

    assert(p->upscaler && p->downscaler && p->frame_mixer);
#endif

    // TODO: Use direct rendering buffers
    if (!init_codec(p))
        goto error;

    p->queue = pl_queue_create(p->win->gpu);
    int ret = pthread_create(&p->decoder_thread, NULL, decode_loop, p);
    if (ret != 0) {
        fprintf(stderr, "Failed creating decode thread: %s\n", strerror(errno));
        goto error;
    }

    p->renderer = pl_renderer_create(p->ctx, p->win->gpu);
    if (!render_loop(p))
        goto error;

    printf("Exiting...\n");
    uninit(p);
    return 0;

error:
    uninit(p);
    return 1;
}

#ifdef HAVE_NUKLEAR

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

static void update_settings(struct plplay *p)
{
    struct nk_context *nk = ui_get_context(p->ui);
    enum nk_panel_flags win_flags = NK_WINDOW_BORDER | NK_WINDOW_MOVABLE |
                                    NK_WINDOW_SCALABLE | NK_WINDOW_MINIMIZABLE |
                                    NK_WINDOW_TITLE;

    ui_update_input(p->ui, p->win);
    const char *dropped_file = window_get_file(p->win);

    const struct pl_filter_preset *f;
    struct pl_render_params *par = &p->params;

    if (nk_begin(nk, "Settings", nk_rect(100, 100, 600, 600), win_flags)) {

        struct nk_colorf bg = {
            par->background_color[0],
            par->background_color[1],
            par->background_color[2],
            1.0,
        };

        nk_layout_row_dynamic(nk, 24, 2);
        nk_label(nk, "Background color:", NK_TEXT_LEFT);
        if (nk_combo_begin_color(nk, nk_rgb_cf(bg), nk_vec2(nk_widget_width(nk), 300))) {
            nk_layout_row_dynamic(nk, 200, 1);
            nk_color_pick(nk, &bg, NK_RGB);
            nk_combo_end(nk);

            par->background_color[0] = bg.r;
            par->background_color[1] = bg.g;
            par->background_color[2] = bg.b;
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Image scaling", NK_MAXIMIZED)) {
            nk_layout_row(nk, NK_DYNAMIC, 24, 2, (float[]){ 0.3, 0.7 });
            nk_label(nk, "Upscaler:", NK_TEXT_LEFT);
            if (nk_combo_begin_label(nk, p->upscaler->description, nk_vec2(nk_widget_width(nk), 500))) {
                nk_layout_row_dynamic(nk, 16, 1);
                for (f = pl_filter_presets; f->name; f++) {
                    if (!f->description)
                        continue;
                    if (nk_combo_item_label(nk, f->description, NK_TEXT_LEFT))
                        p->upscaler = f;
                }
                par->upscaler = p->upscaler->filter;
                nk_combo_end(nk);
            }

            nk_label(nk, "Downscaler:", NK_TEXT_LEFT);
            if (nk_combo_begin_label(nk, p->downscaler->description, nk_vec2(nk_widget_width(nk), 500))) {
                nk_layout_row_dynamic(nk, 16, 1);
                for (f = pl_filter_presets; f->name; f++) {
                    if (!f->description)
                        continue;
                    if (nk_combo_item_label(nk, f->description, NK_TEXT_LEFT))
                        p->downscaler = f;
                }
                par->downscaler = p->downscaler->filter;
                nk_combo_end(nk);
            }

            nk_label(nk, "Frame mixing:", NK_TEXT_LEFT);
            if (nk_combo_begin_label(nk, p->frame_mixer->description, nk_vec2(nk_widget_width(nk), 300))) {
                nk_layout_row_dynamic(nk, 16, 1);
                for (f = pl_frame_mixers; f->name; f++) {
                    if (!f->description)
                        continue;
                    if (nk_combo_item_label(nk, f->description, NK_TEXT_LEFT))
                        p->frame_mixer = f;
                }
                par->frame_mixer = p->frame_mixer->filter;
                nk_combo_end(nk);
            }

            nk_layout_row_dynamic(nk, 24, 2);
            par->skip_anti_aliasing = !nk_check_label(nk, "Anti-aliasing", !par->skip_anti_aliasing);
            nk_property_float(nk, "Antiringing", 0, &par->antiringing_strength, 1.0, 0.1, 0.01);
            nk_property_int(nk, "LUT precision", 0, &par->lut_entries, 256, 1, 1);

            float cutoff = par->polar_cutoff * 100.0;
            nk_property_float(nk, "Polar cutoff (%)", 0.0, &cutoff, 100.0, 0.1, 0.01);
            par->polar_cutoff = cutoff / 100.0;

            struct pl_sigmoid_params *spar = &p->sigmoid_params;
            nk_layout_row_dynamic(nk, 24, 2);
            par->sigmoid_params = nk_check_label(nk, "Sigmoidization", par->sigmoid_params) ? spar : NULL;
            if (nk_button_label(nk, "Default values"))
                *spar = pl_sigmoid_default_params;
            nk_property_float(nk, "Sigmoid center", 0, &spar->center, 1, 0.1, 0.01);
            nk_property_float(nk, "Sigmoid slope", 0, &spar->slope, 100, 1, 0.1);
            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Debanding", NK_MINIMIZED)) {
            struct pl_deband_params *dpar = &p->deband_params;
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

        if (nk_tree_push(nk, NK_TREE_NODE, "Color adjustment", NK_MINIMIZED)) {
            struct pl_color_adjustment *adj = &p->color_adjustment;
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

            struct pl_cone_params *cpar = &p->cone_params;
            nk_layout_row_dynamic(nk, 24, 2);
            par->cone_params = nk_check_label(nk, "Color blindness", par->cone_params) ? cpar : NULL;
            if (nk_button_label(nk, "Default values"))
                *cpar = pl_vision_normal;
            nk_layout_row(nk, NK_DYNAMIC, 24, 5, (float[]){ 0.25, 0.25/3, 0.25/3, 0.25/3, 0.5 });
            nk_label(nk, "Cone model:", NK_TEXT_LEFT);
            int cones = cpar->cones;
            nk_checkbox_flags_label(nk, "L", &cones, PL_CONE_L);
            nk_checkbox_flags_label(nk, "M", &cones, PL_CONE_M);
            nk_checkbox_flags_label(nk, "S", &cones, PL_CONE_S);
            cpar->cones = cones;
            nk_property_float(nk, "Sensitivity", 0.0, &cpar->strength, 5.0, 0.1, 0.01);
            nk_tree_pop(nk);
        }

        if (is_file_hdr(p)) {
            if (nk_tree_push(nk, NK_TREE_NODE, "HDR peak detection", NK_MINIMIZED)) {
                struct pl_peak_detect_params *ppar = &p->peak_detect_params;
                nk_layout_row_dynamic(nk, 24, 2);
                par->peak_detect_params = nk_check_label(nk, "Enable", par->peak_detect_params) ? ppar : NULL;
                if (nk_button_label(nk, "Reset settings"))
                    *ppar = pl_peak_detect_default_params;
                nk_property_float(nk, "Threshold low", 0.0, &ppar->scene_threshold_low, 20.0, 0.5, 0.005);
                nk_property_float(nk, "Threshold high", 0.0, &ppar->scene_threshold_high, 20.0, 0.5, 0.005);
                nk_property_float(nk, "Smoothing period", 1.0, &ppar->smoothing_period, 1000.0, 5.0, 1.0);

                int overshoot = roundf(ppar->overshoot_margin * 100.0);
                nk_property_int(nk, "Overshoot (%)", 0, &overshoot, 200, 1, 1);
                ppar->overshoot_margin = overshoot / 100.0;
                nk_tree_pop(nk);
            }
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Tone mapping", NK_MINIMIZED)) {
            struct pl_color_map_params *cpar = &p->color_map_params;
            nk_layout_row_dynamic(nk, 24, 2);
            par->color_map_params = nk_check_label(nk, "Enable", par->color_map_params) ? cpar : NULL;
            if (nk_button_label(nk, "Reset settings"))
                *cpar = pl_color_map_default_params;

            static const char *rendering_intents[4] = {
                [PL_INTENT_PERCEPTUAL]              = "Perceptual",
                [PL_INTENT_RELATIVE_COLORIMETRIC]   = "Relative colorimetric",
                [PL_INTENT_SATURATION]              = "Saturation",
                [PL_INTENT_ABSOLUTE_COLORIMETRIC]   = "Absolute colorimetric",
            };

            nk_label(nk, "Rendering intent:", NK_TEXT_LEFT);
            cpar->intent = nk_combo(nk, rendering_intents, 4, cpar->intent,
                                    16, nk_vec2(nk_widget_width(nk), 100));

            static const char *tone_mapping_algos[PL_TONE_MAPPING_ALGORITHM_COUNT] = {
                [PL_TONE_MAPPING_CLIP]              = "Clip",
                [PL_TONE_MAPPING_MOBIUS]            = "Mobius",
                [PL_TONE_MAPPING_REINHARD]          = "Reinhard",
                [PL_TONE_MAPPING_HABLE]             = "Hable",
                [PL_TONE_MAPPING_GAMMA]             = "Gamma",
                [PL_TONE_MAPPING_LINEAR]            = "Linear",
                [PL_TONE_MAPPING_BT_2390]           = "BT.2390",
            };

            nk_label(nk, "Tone mapping algorithm:", NK_TEXT_LEFT);
            enum pl_tone_mapping_algorithm new_algo;
            new_algo = nk_combo(nk, tone_mapping_algos, PL_TONE_MAPPING_ALGORITHM_COUNT,
                                cpar->tone_mapping_algo, 16, nk_vec2(nk_widget_width(nk), 300));

            const char *param = NULL;
            float param_min, param_max, param_def = 0.0;
            switch (new_algo) {
            case PL_TONE_MAPPING_MOBIUS:
                param = "Knee point";
                param_min = 0.00;
                param_max = 1.00;
                param_def = 0.5;
                break;
            case PL_TONE_MAPPING_REINHARD:
                param = "Contrast";
                param_min = 0.00;
                param_max = 1.00;
                param_def = 0.5;
                break;
            case PL_TONE_MAPPING_GAMMA:
                param = "Exponent";
                param_min = 0.5;
                param_max = 4.0;
                param_def = 1.8;
                break;
            case PL_TONE_MAPPING_LINEAR:
                param = "Exposure";
                param_min = 0.1;
                param_max = 100.0;
                param_def = 1.0;
                break;
            default: break;
            }

            // Explicitly reset the tone mapping parameter when changing this
            // function, since the interpretation depends on the algorithm
            if (new_algo != cpar->tone_mapping_algo)
                cpar->tone_mapping_param = param_def;
            cpar->tone_mapping_algo = new_algo;

            nk_label(nk, "Algorithm parameter:", NK_TEXT_LEFT);
            if (param) {
                nk_property_float(nk, param, param_min, &cpar->tone_mapping_param,
                                  param_max, 0.1, 0.01);
            } else {
                nk_label(nk, "(N/A)", NK_TEXT_LEFT);
            }

            nk_property_float(nk, "Maximum boost", 1.0, &cpar->max_boost, 10.0, 0.1, 0.01);
            nk_property_float(nk, "Desaturation", 0.0, &cpar->desaturation_strength, 1.0, 0.1, 0.01);
            nk_property_float(nk, "Desat exponent", 0.0, &cpar->desaturation_exponent, 10.0, 0.1, 0.01);
            nk_property_float(nk, "Desat base", 0.0, &cpar->desaturation_base, 10.0, 0.1, 0.01);
            nk_checkbox_label(nk, "Gamut warning", &cpar->gamut_warning);
            nk_checkbox_label(nk, "Colorimetric clipping", &cpar->gamut_clipping);

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
                    par->lut = pl_lut_parse_cube(p->ctx, buf, size);
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
            struct pl_dither_params *dpar = &p->dither_params;
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
            nk_property_int(nk, "Bit depth override", 0, &p->force_depth, 16, 1, 0);
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
                    hook = pl_mpv_user_shader_parse(p->win->gpu, buf, size);
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
                }

                if (i < p->shader_num)
                    nk_label(nk, p->shader_paths[i], NK_TEXT_LEFT);
            }

            par->hooks = p->shader_hooks;
            par->num_hooks = p->shader_num;
            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Debug", NK_MINIMIZED)) {
            nk_layout_row_dynamic(nk, 24, 1);
            nk_checkbox_label(nk, "Allow delayed peak-detect", &par->allow_delayed_peak_detect);
            nk_checkbox_label(nk, "Preserve mixing cache", &par->preserve_mixing_cache);
            nk_checkbox_label(nk, "Disable linear scaling", &par->disable_linear_scaling);
            nk_checkbox_label(nk, "Disable built-in scalers", &par->disable_builtin_scalers);
            nk_checkbox_label(nk, "Force-enable 3DLUT", &par->force_icc_lut);
            nk_checkbox_label(nk, "Force-enable dither", &par->force_dither);
            nk_checkbox_label(nk, "Disable FBOs / advanced rendering", &par->disable_fbos);

            nk_layout_row_dynamic(nk, 24, 2);
            if (nk_button_label(nk, "Flush renderer cache"))
                pl_renderer_flush_cache(p->renderer);
            if (nk_button_label(nk, "Recreate renderer")) {
                pl_renderer_destroy(&p->renderer);
                p->renderer = pl_renderer_create(p->ctx, p->win->gpu);
            }
            nk_tree_pop(nk);
        }
    }
    nk_end(nk);
}

#else
static void update_settings(struct plplay *p) { }
#endif
