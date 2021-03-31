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

#ifdef HAVE_UI
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

    // decoder thread
    pthread_t thread;
    pthread_mutex_t lock;
    pthread_cond_t wakeup;
    AVFrame *frame;
    bool failed;
    bool eof;

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

    if (p->thread)
        pthread_cancel(p->thread);
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

static inline bool decode_frame(struct plplay *p, AVFrame *frame)
{
    int ret = avcodec_receive_frame(p->codec, frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        return false;
    } else if (ret < 0) {
        fprintf(stderr, "libavcodec: Failed decoding frame: %s\n",
                av_err2str(ret));
        p->failed = true;
        return false;
    }

    return true;
}

static inline void send_frame(struct plplay *p, AVFrame *frame)
{
    pthread_mutex_lock(&p->lock);
    while (p->frame) {
        if (p->failed) {
            // Discard frame
            pthread_mutex_unlock(&p->lock);
            av_frame_free(&frame);
            return;
        }

        pthread_cond_wait(&p->wakeup, &p->lock);
    }
    p->frame = frame;
    pthread_cond_broadcast(&p->wakeup);
    pthread_mutex_unlock(&p->lock);
}

static void *decode_loop(void *arg)
{
    struct plplay *p = arg;
    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    if (!packet || !frame) {
        p->failed = true;
        goto done;
    }

    int ret;
    bool eof = false;

    while (!eof) {
        ret = av_read_frame(p->format, packet);
        if (!ret) {
            if (packet->stream_index != p->stream->index) {
                // Ignore unrelated packets
                av_packet_unref(packet);
                continue;
            }

            ret = avcodec_send_packet(p->codec, packet);
        } else if (ret == AVERROR_EOF) {
            // Send empty input to flush decoder
            ret = avcodec_send_packet(p->codec, NULL);
            eof = true;
        } else {
            fprintf(stderr, "libavformat: Failed reading packet: %s\n",
                    av_err2str(ret));
            p->failed = true;
            goto done;
        }

        if (ret < 0) {
            fprintf(stderr, "libavcodec: Failed sending packet to decoder: %s\n",
                    av_err2str(ret));
            p->failed = true;
            goto done;
        }

        // Decode all frames from this packet
        while (decode_frame(p, frame)) {
            send_frame(p, frame);
            frame = av_frame_alloc();
        }

        if (!eof)
            av_packet_unref(packet);
    }

    p->eof = true;

done:
    pthread_cond_broadcast(&p->wakeup);
    av_frame_free(&frame);
    av_packet_free(&packet);
    return NULL;
}

static bool upload_frame(const struct pl_gpu *gpu, const struct pl_tex **tex,
                         const struct pl_source_frame *src,
                         struct pl_frame *out_frame)
{
    AVFrame *frame = src->frame_data;
    bool ok = pl_upload_avframe(gpu, out_frame, tex, frame);
    av_frame_free(&frame);
    return ok;
}

static void discard_frame(const struct pl_source_frame *src)
{
    AVFrame *frame = src->frame_data;
    av_frame_free(&frame);
}

static enum pl_queue_status get_frame(struct pl_source_frame *out_frame,
                                      const struct pl_queue_params *params)
{
    struct plplay *p = params->priv;
    if (p->failed)
        return QUEUE_ERR;

    pthread_mutex_lock(&p->lock);
    while (!p->frame) {
        if (p->eof) {
            pthread_mutex_unlock(&p->lock);
            return QUEUE_EOF;
        }

        pthread_cond_wait(&p->wakeup, &p->lock);
    }

    *out_frame = (struct pl_source_frame) {
        .pts = p->frame->pts * av_q2d(p->stream->time_base),
        .map = upload_frame,
        .discard = discard_frame,
        .frame_data = p->frame,
    };
    p->frame = NULL;
    pthread_cond_broadcast(&p->wakeup);
    pthread_mutex_unlock(&p->lock);
    return QUEUE_OK;
}

static void update_settings(struct plplay *p);

static bool render_frame(struct plplay *p, const struct pl_swapchain_frame *frame,
                         const struct pl_frame_mix *mix)
{
    const struct pl_gpu *gpu = p->win->gpu;
    struct pl_frame target;
    pl_frame_from_swapchain(&target, frame);
    update_settings(p);

    assert(mix->num_frames);
    pl_rect2df_aspect_copy(&target.crop, &mix->frames[0]->crop, 0.0);
    if (pl_frame_is_cropped(&target))
        pl_frame_clear(gpu, &target, (float[3]) {0});

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
        .get_frame = get_frame,
        .priv = p,
    };

    // Initialize the frame queue, blocking indefinitely until done
    struct pl_frame_mix mix;
    switch (pl_queue_update(p->queue, &mix, &qparams)) {
    case QUEUE_OK:  break;
    case QUEUE_EOF: return true;
    case QUEUE_ERR: goto error;
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

    struct timespec ts_base, ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts_base) < 0) {
        fprintf(stderr, "%s\n", strerror(errno));
        goto error;
    }

    pl_swapchain_swap_buffers(p->win->swapchain);
    window_poll(p->win, false);

    while (!p->win->window_lost) {
        if (!pl_swapchain_start_frame(p->win->swapchain, &frame)) {
            // Window stuck/invisible? Block for events and try again.
            window_poll(p->win, true);
            continue;
        }

        if (clock_gettime(CLOCK_MONOTONIC, &ts) < 0)
            goto error;

        qparams.pts = (ts.tv_sec - ts_base.tv_sec) +
                      (ts.tv_nsec - ts_base.tv_nsec) * 1e-9;

        switch (pl_queue_update(p->queue, &mix, &qparams)) {
        case QUEUE_ERR: goto error;
        case QUEUE_EOF: return true;
        case QUEUE_OK:
            if (!render_frame(p, &frame, &mix))
                goto error;
            break;
        default: abort();
        }

        if (!pl_swapchain_submit_frame(p->win->swapchain)) {
            fprintf(stderr, "libplacebo: failed presenting frame!\n");
            goto error;
        }

        pl_swapchain_swap_buffers(p->win->swapchain);
        window_poll(p->win, false);
    }

    return !p->failed;

error:
    p->failed = true;
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
        .lock = PTHREAD_MUTEX_INITIALIZER,
        .wakeup = PTHREAD_COND_INITIALIZER,
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
    DEFAULT_PARAMS(cone_params);
    state.params.color_adjustment = &state.color_adjustment;
    state.params.color_map_params = &state.color_map_params;

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

    p->ctx = demo_context();
    p->win = window_create(p->ctx, "plplay", par->width, par->height, flags);
    if (!p->win)
        goto error;

#ifdef HAVE_UI
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

    int ret = pthread_create(&p->thread, NULL, decode_loop, p);
    if (ret != 0) {
        fprintf(stderr, "Failed creating decode thread: %s\n", strerror(errno));
        goto error;
    }

    p->renderer = pl_renderer_create(p->ctx, p->win->gpu);
    p->queue = pl_queue_create(p->win->gpu);
    if (!render_loop(p))
        goto error;

    printf("Exiting normally...\n");
    uninit(p);
    return 0;

error:
    uninit(p);
    return 1;
}

#ifdef HAVE_UI

static void add_hook(struct plplay *p, const struct pl_hook *hook, const char *path)
{
    if (!hook)
        return;

    if (p->shader_num == p->shader_size) {
        // Grow array if needed
        size_t new_size = p->shader_size ? p->shader_size * 2 : 16;
        void *new_hooks = reallocarray(p->shader_hooks, new_size, sizeof(void *));
        if (!new_hooks)
            goto error;
        p->shader_hooks = new_hooks;
        char **new_paths = reallocarray(p->shader_paths, new_size, sizeof(char *));
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

        nk_layout_row_dynamic(nk, 24, 2);
        if (par->lut_entries) {
            nk_labelf(nk, NK_TEXT_LEFT, "LUT precision: (%d)", par->lut_entries);
        } else {
            nk_label(nk, "LUT precision: (default)", NK_TEXT_LEFT);
        }

        nk_slider_int(nk, 0, &par->lut_entries, 256, 4);

        nk_label(nk, "Antiringing:", NK_TEXT_LEFT);
        nk_slider_float(nk, 0.0, &par->antiringing_strength, 1.0, 0.01f);

        nk_layout_row(nk, NK_DYNAMIC, 24, 2, (float[]){ 0.3, 0.7 });
        nk_label(nk, "Frame mixer:", NK_TEXT_LEFT);
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

        if (nk_tree_push(nk, NK_TREE_NODE, "Debanding", NK_MINIMIZED)) {
            struct pl_deband_params *dpar = &p->deband_params;
            nk_layout_row_dynamic(nk, 24, 2);
            par->deband_params = nk_check_label(nk, "Enable", par->deband_params) ? dpar : NULL;
            if (nk_button_label(nk, "Reset settings"))
                *dpar = pl_deband_default_params;
            nk_labelf(nk, NK_TEXT_LEFT, "Iterations: (%d)", dpar->iterations);
            nk_slider_int(nk, 0, &dpar->iterations, 8, 1);
            nk_labelf(nk, NK_TEXT_LEFT, "Threshold: (%.1f)", dpar->threshold);
            nk_slider_float(nk, 0.0, &dpar->threshold, 32.0, 0.1);
            nk_labelf(nk, NK_TEXT_LEFT, "Radius: (%.1f)", dpar->radius);
            nk_slider_float(nk, 0.0, &dpar->radius, 32.0, 0.1);
            nk_labelf(nk, NK_TEXT_LEFT, "Grain: (%.1f)", dpar->grain);
            nk_slider_float(nk, 0.0, &dpar->grain, 32.0, 0.1);
            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Sigmoidization", NK_MINIMIZED)) {
            struct pl_sigmoid_params *spar = &p->sigmoid_params;
            nk_layout_row_dynamic(nk, 24, 2);
            par->sigmoid_params = nk_check_label(nk, "Enable", par->sigmoid_params) ? spar : NULL;
            if (nk_button_label(nk, "Reset settings"))
                *spar = pl_sigmoid_default_params;
            nk_labelf(nk, NK_TEXT_LEFT, "Center: (%.2f)", spar->center);
            nk_slider_float(nk, 0.0, &spar->center, 1.0, 0.01);
            nk_labelf(nk, NK_TEXT_LEFT, "Slope: (%.1f)", spar->slope);
            nk_slider_float(nk, 0.0, &spar->slope, 20.0, 0.1);
            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Color adjustment", NK_MINIMIZED)) {
            struct pl_color_adjustment *adj = &p->color_adjustment;
            nk_layout_row_dynamic(nk, 24, 2);
            par->color_adjustment = nk_check_label(nk, "Enable", par->color_adjustment) ? adj : NULL;
            if (nk_button_label(nk, "Reset settings"))
                *adj = pl_color_adjustment_neutral;
            nk_label(nk, "Brightness:", NK_TEXT_LEFT);
            nk_slider_float(nk, -1.0, &adj->brightness, 1.0, 0.01);
            nk_label(nk, "Contrast:", NK_TEXT_LEFT);
            nk_slider_float(nk, 0.0, &adj->contrast, 2.0, 0.01);
            nk_label(nk, "Saturation:", NK_TEXT_LEFT);
            nk_slider_float(nk, 0.0, &adj->saturation, 2.0, 0.01);
            nk_label(nk, "Hue:", NK_TEXT_LEFT);
            nk_slider_float(nk, -M_PI, &adj->hue, M_PI, 0.01);
            nk_label(nk, "Gamma:", NK_TEXT_LEFT);
            nk_slider_float(nk, 0.0, &adj->gamma, 2.0, 0.01);
            nk_label(nk, "Temperature:", NK_TEXT_LEFT);
            nk_slider_float(nk, -1.0, &adj->temperature, 1.0, 0.01);
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
                if (nk_button_symbol(nk, NK_SYMBOL_TRIANGLE_UP) && i > 0) {
                    const struct pl_hook *prev_hook = p->shader_hooks[i - 1];
                    char *prev_path = p->shader_paths[i - 1];
                    p->shader_hooks[i - 1] = p->shader_hooks[i];
                    p->shader_paths[i - 1] = p->shader_paths[i];
                    p->shader_hooks[i] = prev_hook;
                    p->shader_paths[i] = prev_path;
                }

                if (nk_button_symbol(nk, NK_SYMBOL_TRIANGLE_DOWN) && i < p->shader_num - 1) {
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

                nk_label(nk, p->shader_paths[i], NK_TEXT_LEFT);
            }

            par->hooks = p->shader_hooks;
            par->num_hooks = p->shader_num;
            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Custom color LUT", NK_MINIMIZED)) {

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
            if (nk_button_label(nk, "Clear")) {
                pl_lut_free((struct pl_custom_lut **) &par->lut);
                par->lut_type = PL_LUT_UNKNOWN;
            }

            nk_labelf(nk, NK_TEXT_CENTERED, "LUT type:");
            par->lut_type = nk_combo(nk, lut_types, 4, par->lut_type,
                                     16, nk_vec2(nk_widget_width(nk), 100));


            nk_tree_pop(nk);
        }

        if (is_file_hdr(p)) {
            if (nk_tree_push(nk, NK_TREE_NODE, "HDR peak detection", NK_MINIMIZED)) {
                struct pl_peak_detect_params *ppar = &p->peak_detect_params;
                nk_layout_row_dynamic(nk, 24, 2);
                par->peak_detect_params = nk_check_label(nk, "Enable", par->peak_detect_params) ? ppar : NULL;
                if (nk_button_label(nk, "Reset settings"))
                    *ppar = pl_peak_detect_default_params;
                nk_labelf(nk, NK_TEXT_LEFT, "Smoothing period: (%d)", (int) ppar->smoothing_period);
                nk_slider_float(nk, 1.0, &ppar->smoothing_period, 1000.0, 1.0);
                nk_labelf(nk, NK_TEXT_LEFT, "Threshold low: (%.2f)", ppar->scene_threshold_low);
                nk_slider_float(nk, 0.0, &ppar->scene_threshold_low, 20.0, 0.01);
                nk_labelf(nk, NK_TEXT_LEFT, "Threshold high: (%.2f)", ppar->scene_threshold_high);
                nk_slider_float(nk, 0.0, &ppar->scene_threshold_high, 20.0, 0.01);
                nk_labelf(nk, NK_TEXT_LEFT, "Overshoot margin: (%.2f%%)", ppar->overshoot_margin);
                nk_slider_float(nk, 0.0, &ppar->overshoot_margin, 1.0, 0.01);
                nk_tree_pop(nk);
            }
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Gamut adaptation / tone mapping", NK_MINIMIZED)) {
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

            nk_labelf(nk, NK_TEXT_LEFT, "Rendering intent:");
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

            nk_labelf(nk, NK_TEXT_LEFT, "Tone mapping algorithm:");
            enum pl_tone_mapping_algorithm new_algo;
            new_algo = nk_combo(nk, tone_mapping_algos, PL_TONE_MAPPING_ALGORITHM_COUNT,
                                cpar->tone_mapping_algo, 16, nk_vec2(nk_widget_width(nk), 300));

            float param_min, param_max, param_def = 0.0;
            switch (new_algo) {
            case PL_TONE_MAPPING_MOBIUS:
            case PL_TONE_MAPPING_REINHARD:
                param_min = 0.01;
                param_max = 0.99;
                param_def = 0.5;
                break;
            case PL_TONE_MAPPING_GAMMA:
                param_min = 0.5;
                param_max = 4.0;
                param_def = 1.8;
                break;
            case PL_TONE_MAPPING_LINEAR:
                param_min = 0.1;
                param_max = 10.0;
                param_def = 1.0;
                break;
            default: break;
            }

            // Explicitly reset the tone mapping parameter when changing this
            // function, since the interpretation depends on the algorithm
            if (new_algo != cpar->tone_mapping_algo)
                cpar->tone_mapping_param = param_def;
            cpar->tone_mapping_algo = new_algo;

            if (param_def) {
                nk_labelf(nk, NK_TEXT_LEFT, "Algorithm parameter: %.2f", cpar->tone_mapping_param);
                nk_slider_float(nk, param_min, &cpar->tone_mapping_param, param_max, param_max / 100.0);
            }

            nk_labelf(nk, NK_TEXT_LEFT, "Desaturation strength:");
            nk_slider_float(nk, 0.0, &cpar->desaturation_strength, 1.0, 0.01);
            nk_labelf(nk, NK_TEXT_LEFT, "Exponent: (%.2f)", cpar->desaturation_exponent);
            nk_slider_float(nk, 0.0, &cpar->desaturation_exponent, 4.0, 0.01);
            nk_labelf(nk, NK_TEXT_LEFT, "Base: (%.2f)", cpar->desaturation_base);
            nk_slider_float(nk, 0.0, &cpar->desaturation_base, 1.0, 0.01);
            nk_labelf(nk, NK_TEXT_LEFT, "Maximum boost: (%.2f)", cpar->max_boost);
            nk_slider_float(nk, 1.0, &cpar->max_boost, 4.0, 0.01);
            nk_layout_row_dynamic(nk, 24, 1);
            nk_checkbox_label(nk, "Enable gamut warning", &cpar->gamut_warning);
            nk_checkbox_label(nk, "Enable colorimetric clipping", &cpar->gamut_clipping);

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

            nk_labelf(nk, NK_TEXT_LEFT, "Dither algorithm:");
            dpar->method = nk_combo(nk, dither_methods, PL_DITHER_METHOD_COUNT, dpar->method,
                                    16, nk_vec2(nk_widget_width(nk), 100));

            switch (dpar->method) {
            case PL_DITHER_BLUE_NOISE:
            case PL_DITHER_ORDERED_LUT:
                nk_labelf(nk, NK_TEXT_LEFT, "LUT size (%d):", 1 << dpar->lut_size);
                nk_slider_int(nk, 1, &dpar->lut_size, 8, 1);
                break;
            default: break;
            }

            nk_layout_row_dynamic(nk, 24, 1);
            nk_checkbox_label(nk, "Enable temporal dithering", &dpar->temporal);
            nk_layout_row_dynamic(nk, 24, 2);
            nk_labelf(nk, NK_TEXT_LEFT, "Simulate bit depth: (%d)", p->force_depth);
            nk_slider_int(nk, 0, &p->force_depth, 16, 1);
            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Color blindness adaptation", NK_MINIMIZED)) {
            struct pl_cone_params *cpar = &p->cone_params;
            nk_layout_row_dynamic(nk, 24, 2);
            par->cone_params = nk_check_label(nk, "Enable", par->cone_params) ? cpar : NULL;
            if (nk_button_label(nk, "Reset settings"))
                *cpar = pl_vision_normal;
            nk_layout_row_dynamic(nk, 24, 1);
            int cones = cpar->cones;
            nk_checkbox_flags_label(nk, "Red cone (L)", &cones, PL_CONE_L);
            nk_checkbox_flags_label(nk, "Green cone (M)", &cones, PL_CONE_M);
            nk_checkbox_flags_label(nk, "Blue cone (S)", &cones, PL_CONE_S);
            cpar->cones = cones;
            nk_layout_row_dynamic(nk, 24, 2);
            nk_labelf(nk, NK_TEXT_LEFT, "Strength:");
            nk_slider_float(nk, 0.0, &cpar->strength, 2.0, 0.01);
            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Performance / quality trade-off", NK_MINIMIZED)) {
            nk_layout_row_dynamic(nk, 24, 1);
            nk_checkbox_label(nk, "Disable anti-aliasing", &par->skip_anti_aliasing);
            nk_layout_row_dynamic(nk, 24, 2);
            nk_labelf(nk, NK_TEXT_LEFT, "Polar cut-off value: (%.2f)", par->polar_cutoff);
            nk_slider_float(nk, 0.0, &par->polar_cutoff, 1.0, 0.01f);
            nk_layout_row_dynamic(nk, 24, 1);
            nk_checkbox_label(nk, "Disable overlay sampling", &par->disable_overlay_sampling);
            nk_checkbox_label(nk, "Allow delayed peak-detect", &par->allow_delayed_peak_detect);
            nk_checkbox_label(nk, "Preserve mixing cache", &par->preserve_mixing_cache);
            nk_tree_pop(nk);
        }

        if (nk_tree_push(nk, NK_TREE_NODE, "Performance tuning / debugging", NK_MINIMIZED)) {
            nk_layout_row_dynamic(nk, 24, 1);
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
