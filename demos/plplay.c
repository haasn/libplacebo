/* Example video player based on ffmpeg. Designed to expose every libplacebo
 * option for testing purposes. Not a serious video player, no real error
 * handling. Simply infinitely loops its input.
 *
 * License: CC0 / Public Domain
 */

#include <stdatomic.h>

#include <libavutil/cpu.h>

#include "common.h"
#include "window.h"
#include "utils.h"
#include "plplay.h"
#include "pl_clock.h"
#include "pl_thread.h"

#ifdef HAVE_NUKLEAR
#include "ui.h"
#else
struct ui;
static void ui_destroy(struct ui **ui) {}
static bool ui_draw(struct ui *ui, const struct pl_swapchain_frame *frame) { return true; };
#endif

#include <libplacebo/utils/libav.h>

static inline void log_time(struct timing *t, double ts)
{
    t->sum += ts;
    t->sum2 += ts * ts;
    t->peak = fmax(t->peak, ts);
    t->count++;
}

static void uninit(struct plplay *p)
{
    if (p->decoder_thread_created) {
        p->exit_thread = true;
        pl_queue_push(p->queue, NULL); // Signal EOF to wake up thread
        pl_thread_join(p->decoder_thread);
    }

    pl_queue_destroy(&p->queue);
    pl_renderer_destroy(&p->renderer);
    pl_options_free(&p->opts);

    for (int i = 0; i < p->shader_num; i++) {
        pl_mpv_user_shader_destroy(&p->shader_hooks[i]);
        free(p->shader_paths[i]);
    }

    for (int i = 0; i < MAX_FRAME_PASSES; i++)
        pl_shader_info_deref(&p->frame_info[i].shader);
    for (int j = 0; j < MAX_BLEND_FRAMES; j++) {
        for (int i = 0; i < MAX_BLEND_PASSES; i++)
            pl_shader_info_deref(&p->blend_info[j][i].shader);
    }

    free(p->shader_hooks);
    free(p->shader_paths);
    free(p->icc_name);
    pl_icc_close(&p->icc);

    if (p->cache) {
        if (pl_cache_signature(p->cache) != p->cache_sig) {
            FILE *file = fopen(p->cache_file, "wb");
            if (file) {
                pl_cache_save_file(p->cache, file);
                fclose(file);
            }
        }
        pl_cache_destroy(&p->cache);
    }

    // Free this before destroying the window to release associated GPU buffers
    avcodec_free_context(&p->codec);
    avformat_free_context(p->format);

    ui_destroy(&p->ui);
    window_destroy(&p->win);

    pl_log_destroy(&p->log);
    memset(p, 0, sizeof(*p));
}

static bool open_file(struct plplay *p, const char *filename)
{
    static const int av_log_level[] = {
        [PL_LOG_NONE]  = AV_LOG_QUIET,
        [PL_LOG_FATAL] = AV_LOG_PANIC,
        [PL_LOG_ERR]   = AV_LOG_ERROR,
        [PL_LOG_WARN]  = AV_LOG_WARNING,
        [PL_LOG_INFO]  = AV_LOG_INFO,
        [PL_LOG_DEBUG] = AV_LOG_VERBOSE,
        [PL_LOG_TRACE] = AV_LOG_DEBUG,
    };

    av_log_set_level(av_log_level[p->args.verbosity]);

    printf("Opening file: '%s'\n", filename);
    if (avformat_open_input(&p->format, filename, NULL, NULL) != 0) {
        fprintf(stderr, "libavformat: Failed opening file!\n");
        return false;
    }

    printf("Format: %s\n", p->format->iformat->name);

    if (p->format->duration != AV_NOPTS_VALUE)
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

    if (stream->avg_frame_rate.den && stream->avg_frame_rate.num)
        printf("FPS: %f\n", av_q2d(stream->avg_frame_rate));

    if (stream->r_frame_rate.den && stream->r_frame_rate.num)
        printf("TBR: %f\n", av_q2d(stream->r_frame_rate));

    if (stream->time_base.den && stream->time_base.num)
        printf("TBN: %f\n", av_q2d(stream->time_base));

    if (par->bit_rate)
        printf("Bitrate: %"PRIi64" kbps\n", par->bit_rate / 1000);

    printf("Format: %s\n", av_get_pix_fmt_name(par->format));

    p->stream = stream;
    return true;
}

static bool init_codec(struct plplay *p)
{
    assert(p->stream);
    assert(p->win->gpu);

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

    printf("Codec: %s (%s)\n", codec->name, codec->long_name);

    const AVCodecHWConfig *hwcfg = 0;
    if (p->args.hwdec) {
        for (int i = 0; (hwcfg = avcodec_get_hw_config(codec, i)); i++) {
            if (!pl_test_pixfmt(p->win->gpu, hwcfg->pix_fmt))
                continue;
            if (!(hwcfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX))
                continue;

            int ret = av_hwdevice_ctx_create(&p->codec->hw_device_ctx,
                                            hwcfg->device_type,
                                            NULL, NULL, 0);
            if (ret < 0) {
                fprintf(stderr, "libavcodec: Failed opening HW device context, skipping\n");
                continue;
            }

            const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(hwcfg->pix_fmt);
            printf("Using hardware frame format: %s\n", desc->name);
            p->codec->extra_hw_frames = 4;
            break;
        }
    }

    if (!hwcfg)
        printf("Using software decoding\n");

    p->codec->thread_count = FFMIN(av_cpu_count() + 1, 16);
    p->codec->get_buffer2 = pl_get_buffer2;
    p->codec->opaque = &p->win->gpu;
#if LIBAVCODEC_VERSION_MAJOR < 60
    AV_NOWARN_DEPRECATED({
        p->codec->thread_safe_callbacks = 1;
    });
#endif
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(58, 113, 100)
    p->codec->export_side_data |= AV_CODEC_EXPORT_DATA_FILM_GRAIN;
#endif

    if (avcodec_open2(p->codec, codec, NULL) < 0) {
        fprintf(stderr, "libavcodec: Failed opening codec\n");
        return false;
    }

    return true;
}

static bool map_frame(pl_gpu gpu, pl_tex *tex,
                      const struct pl_source_frame *src,
                      struct pl_frame *out_frame)
{
    AVFrame *frame = src->frame_data;
    struct plplay *p = frame->opaque;
    bool ok = pl_map_avframe_ex(gpu, out_frame, pl_avframe_params(
        .frame      = frame,
        .tex        = tex,
        .map_dovi   = !p->ignore_dovi,
    ));

    av_frame_free(&frame); // references are preserved by `out_frame`
    if (!ok) {
        fprintf(stderr, "Failed mapping AVFrame!\n");
        return false;
    }

    p->stats.mapped++;
    pl_frame_copy_stream_props(out_frame, p->stream);
    return true;
}

static void unmap_frame(pl_gpu gpu, struct pl_frame *frame,
                        const struct pl_source_frame *src)
{
    pl_unmap_avframe(gpu, frame);
}

static void discard_frame(const struct pl_source_frame *src)
{
    AVFrame *frame = src->frame_data;
    struct plplay *p = frame->opaque;
    p->stats.dropped++;
    av_frame_free(&frame);
    printf("Dropped frame with PTS %.3f\n", src->pts);
}

static PL_THREAD_VOID decode_loop(void *arg)
{
    int ret;
    struct plplay *p = arg;
    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    if (!frame || !packet)
        goto done;

    float frame_duration = av_q2d(av_inv_q(p->stream->avg_frame_rate));
    double first_pts = 0.0, base_pts = 0.0, last_pts = 0.0;
    uint64_t num_frames = 0;

    while (!p->exit_thread) {
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
            last_pts = frame->pts * av_q2d(p->stream->time_base);
            if (num_frames++ == 0)
                first_pts = last_pts;
            frame->opaque = p;
            (void) atomic_fetch_add(&p->stats.decoded, 1);
            pl_queue_push_block(p->queue, UINT64_MAX, &(struct pl_source_frame) {
                .pts = last_pts - first_pts + base_pts,
                .duration = frame_duration,
                .map = map_frame,
                .unmap = unmap_frame,
                .discard = discard_frame,
                .frame_data = frame,

                // allow soft-disabling deinterlacing at the source frame level
                .first_field = p->opts->params.deinterlace_params
                                    ? pl_field_from_avframe(frame)
                                    : PL_FIELD_NONE,
            });
            frame = av_frame_alloc();
        }

        switch (ret) {
        case AVERROR(EAGAIN):
            continue;
        case AVERROR_EOF:
            if (num_frames <= 1)
                goto done; // still image or empty file
            // loop infinitely
            ret = av_seek_frame(p->format, p->stream->index, 0, AVSEEK_FLAG_BACKWARD);
            if (ret < 0) {
                fprintf(stderr, "libavformat: Failed seeking in stream: %s\n",
                        av_err2str(ret));
                goto done;
            }
            avcodec_flush_buffers(p->codec);
            base_pts += last_pts;
            num_frames = 0;
            continue;
        default:
            fprintf(stderr, "libavcodec: Failed decoding frame: %s\n",
                    av_err2str(ret));
            goto done;
        }
    }

done:
    pl_queue_push(p->queue, NULL); // Signal EOF to flush queue
    av_packet_free(&packet);
    av_frame_free(&frame);
    PL_THREAD_RETURN();
}

static void update_colorspace_hint(struct plplay *p, const struct pl_frame_mix *mix)
{
    const struct pl_frame *frame = NULL;

    for (int i = 0; i < mix->num_frames; i++) {
        if (mix->timestamps[i] > 0.0)
            break;
        frame = mix->frames[i];
    }

    if (!frame)
        return;

    struct pl_color_space hint = {0};
    if (p->colorspace_hint)
        hint = frame->color;
    if (p->target_override)
        apply_csp_overrides(p, &hint);
    pl_swapchain_colorspace_hint(p->win->swapchain, &hint);
}

static bool render_frame(struct plplay *p, const struct pl_swapchain_frame *frame,
                         const struct pl_frame_mix *mix)
{
    struct pl_frame target;
    pl_options opts = p->opts;
    pl_frame_from_swapchain(&target, frame);
    update_settings(p, &target);

    if (p->target_override) {
        target.repr = p->force_repr;
        pl_color_repr_merge(&target.repr, &frame->color_repr);
        apply_csp_overrides(p, &target.color);

        // Update ICC profile parameters dynamically
        float target_luma = 0.0f;
        if (!p->use_icc_luma) {
            pl_color_space_nominal_luma_ex(pl_nominal_luma_params(
                .metadata = PL_HDR_METADATA_HDR10, // use only static HDR nits
                .scaling  = PL_HDR_NITS,
                .color    = &target.color,
                .out_max  = &target_luma,
            ));
        }
        pl_icc_update(p->log, &p->icc, NULL, pl_icc_params(
            .max_luma  = target_luma,
            .force_bpc = p->force_bpc,
        ));
        target.icc = p->icc;
    }

    assert(mix->num_frames);
    pl_rect2df crop = mix->frames[0]->crop;
    if (p->stream->sample_aspect_ratio.num && p->target_zoom != ZOOM_RAW) {
        float sar = av_q2d(p->stream->sample_aspect_ratio);
        pl_rect2df_stretch(&crop, fmaxf(1.0f, sar), fmaxf(1.0f, 1.0 / sar));
    }

    // Apply target rotation and un-rotate crop relative to target
    target.rotation = p->target_rot;
    pl_rect2df_rotate(&crop, mix->frames[0]->rotation - target.rotation);

    switch (p->target_zoom) {
    case ZOOM_PAD:
        pl_rect2df_aspect_copy(&target.crop, &crop, 0.0);
        break;
    case ZOOM_CROP:
        pl_rect2df_aspect_copy(&target.crop, &crop, 1.0);
        break;
    case ZOOM_STRETCH:
        break; // target.crop already covers full image
    case ZOOM_FIT:
        pl_rect2df_aspect_fit(&target.crop, &crop, 0.0);
        break;
    case ZOOM_RAW: ;
        // Ensure pixels are exactly aligned, to avoid fractional scaling
        int w = roundf(fabsf(pl_rect_w(crop)));
        int h = roundf(fabsf(pl_rect_h(crop)));
        target.crop.x0 = roundf((pl_rect_w(target.crop) - w) / 2.0f);
        target.crop.y0 = roundf((pl_rect_h(target.crop) - h) / 2.0f);
        target.crop.x1 = target.crop.x0 + w;
        target.crop.y1 = target.crop.y0 + h;
        break;
    case ZOOM_400:
    case ZOOM_200:
    case ZOOM_100:
    case ZOOM_50:
    case ZOOM_25: ;
        const float z = powf(2.0f, (int) ZOOM_100 - p->target_zoom);
        const float sx = z * fabsf(pl_rect_w(crop)) / pl_rect_w(target.crop);
        const float sy = z * fabsf(pl_rect_h(crop)) / pl_rect_h(target.crop);
        pl_rect2df_stretch(&target.crop, sx, sy);
        break;
    }

    struct pl_color_map_params *cpars = &opts->color_map_params;
    if (cpars->visualize_lut) {
        cpars->visualize_rect = (pl_rect2df) {0, 0, 1, 1};
        float tar = pl_rect2df_aspect(&target.crop);
        pl_rect2df_aspect_set(&cpars->visualize_rect, 1.0f / tar, 0.0f);
    }

    pl_clock_t ts_pre = pl_clock_now();
    if (!pl_render_image_mix(p->renderer, mix, &target, &opts->params))
        return false;
    pl_clock_t ts_rendered = pl_clock_now();
    if (!ui_draw(p->ui, frame))
        return false;
    pl_clock_t ts_ui_drawn = pl_clock_now();

    log_time(&p->stats.render, pl_clock_diff(ts_rendered, ts_pre));
    log_time(&p->stats.draw_ui, pl_clock_diff(ts_ui_drawn, ts_rendered));

    p->stats.rendered++;
    return true;
}

static bool render_loop(struct plplay *p)
{
    pl_options opts = p->opts;

    struct pl_queue_params qparams = *pl_queue_params(
        .interpolation_threshold = 0.01,
        .timeout = UINT64_MAX,
    );

    // Initialize the frame queue, blocking indefinitely until done
    struct pl_frame_mix mix;
    switch (pl_queue_update(p->queue, &mix, &qparams)) {
    case PL_QUEUE_OK:  break;
    case PL_QUEUE_EOF: return true;
    case PL_QUEUE_ERR: goto error;
    default: abort();
    }

    struct pl_swapchain_frame frame;
    update_colorspace_hint(p, &mix);
    if (!pl_swapchain_start_frame(p->win->swapchain, &frame))
        goto error;

    // Disable background transparency by default if the swapchain does not
    // appear to support alpha transaprency
    if (frame.color_repr.alpha == PL_ALPHA_UNKNOWN)
        opts->params.background_transparency = 0.0;

    if (!render_frame(p, &frame, &mix))
        goto error;
    if (!pl_swapchain_submit_frame(p->win->swapchain))
        goto error;

    // Wait until rendering is complete. Do this before measuring the time
    // start, to ensure we don't count initialization overhead as part of the
    // first vsync.
    pl_gpu_finish(p->win->gpu);
    p->stats.render = p->stats.draw_ui = (struct timing) {0};

    pl_clock_t ts_start = 0, ts_prev = 0;
    pl_swapchain_swap_buffers(p->win->swapchain);
    window_poll(p->win, false);

    double pts_target = 0.0, prev_pts = 0.0;

    while (!p->win->window_lost) {
        if (window_get_key(p->win, KEY_ESC))
            break;

        if (p->toggle_fullscreen)
            window_toggle_fullscreen(p->win, !window_is_fullscreen(p->win));

        update_colorspace_hint(p, &mix);
        pl_clock_t ts_acquire = pl_clock_now();
        if (!pl_swapchain_start_frame(p->win->swapchain, &frame)) {
            // Window stuck/invisible? Block for events and try again.
            window_poll(p->win, true);
            continue;
        }

        pl_clock_t ts_pre_update = pl_clock_now();
        log_time(&p->stats.acquire, pl_clock_diff(ts_pre_update, ts_acquire));
        if (!ts_start)
            ts_start = ts_pre_update;

        qparams.timeout = 0; // non-blocking update
        qparams.radius = pl_frame_mix_radius(&p->opts->params);
        qparams.pts = fmax(pts_target, pl_clock_diff(ts_pre_update, ts_start));
        p->stats.current_pts = qparams.pts;
        if (qparams.pts != prev_pts)
            log_time(&p->stats.pts_interval, qparams.pts - prev_pts);
        prev_pts = qparams.pts;

retry:
        switch (pl_queue_update(p->queue, &mix, &qparams)) {
        case PL_QUEUE_ERR: goto error;
        case PL_QUEUE_EOF:
            printf("End of file reached\n");
            return true;
        case PL_QUEUE_OK:
            break;
        case PL_QUEUE_MORE:
            qparams.timeout = UINT64_MAX; // retry in blocking mode
            goto retry;
        }

        pl_clock_t ts_post_update = pl_clock_now();
        log_time(&p->stats.update, pl_clock_diff(ts_post_update, ts_pre_update));

        if (qparams.timeout) {
            double stuck_ms = 1e3 * pl_clock_diff(ts_post_update, ts_pre_update);
            fprintf(stderr, "Stalled for %.4f ms due to frame queue underrun!\n", stuck_ms);
            ts_start += ts_post_update - ts_pre_update; // subtract time spent waiting
            p->stats.stalled++;
            p->stats.stalled_ms += stuck_ms;
        }

        if (!render_frame(p, &frame, &mix))
            goto error;

        if (pts_target) {
            pl_gpu_flush(p->win->gpu);
            pl_clock_t ts_wait = pl_clock_now();
            double pts_now = pl_clock_diff(ts_wait, ts_start);
            if (pts_target >= pts_now) {
                log_time(&p->stats.sleep, pts_target - pts_now);
                pl_thread_sleep(pts_target - pts_now);
            } else {
                double missed_ms = 1e3 * (pts_now - pts_target);
                fprintf(stderr, "Missed PTS target %.3f (%.3f ms in the past)\n",
                        pts_target, missed_ms);
                p->stats.missed++;
                p->stats.missed_ms += missed_ms;
            }

            pts_target = 0.0;
        }

        pl_clock_t ts_pre_submit = pl_clock_now();
        if (!pl_swapchain_submit_frame(p->win->swapchain)) {
            fprintf(stderr, "libplacebo: failed presenting frame!\n");
            goto error;
        }
        pl_clock_t ts_post_submit = pl_clock_now();
        log_time(&p->stats.submit, pl_clock_diff(ts_post_submit, ts_pre_submit));

        if (ts_prev)
            log_time(&p->stats.vsync_interval, pl_clock_diff(ts_post_submit, ts_prev));
        ts_prev = ts_post_submit;

        pl_swapchain_swap_buffers(p->win->swapchain);
        pl_clock_t ts_post_swap = pl_clock_now();
        log_time(&p->stats.swap, pl_clock_diff(ts_post_swap, ts_post_submit));

        window_poll(p->win, false);

        // In content-timed mode (frame mixing disabled), delay rendering
        // until the next frame should become visible
        if (!opts->params.frame_mixer) {
            struct pl_source_frame next;
            for (int i = 0;; i++) {
                if (!pl_queue_peek(p->queue, i, &next))
                    break;
                if (next.pts > qparams.pts) {
                    pts_target = next.pts;
                    break;
                }
            }
        }

        if (p->fps_override)
            pts_target = fmax(pts_target, qparams.pts + 1.0 / p->fps);
    }

    return true;

error:
    fprintf(stderr, "Render loop failed, exiting early...\n");
    return false;
}

static void info_callback(void *priv, const struct pl_render_info *info)
{
    struct plplay *p = priv;
    switch (info->stage) {
    case PL_RENDER_STAGE_FRAME:
        if (info->index >= MAX_FRAME_PASSES)
            return;
        p->num_frame_passes = info->index + 1;
        pl_dispatch_info_move(&p->frame_info[info->index], info->pass);
        return;

    case PL_RENDER_STAGE_BLEND:
        if (info->index >= MAX_BLEND_PASSES || info->count >= MAX_BLEND_FRAMES)
            return;
        p->num_blend_passes[info->count] = info->index + 1;
        pl_dispatch_info_move(&p->blend_info[info->count][info->index], info->pass);
        return;

    case PL_RENDER_STAGE_COUNT:
        break;
    }

    abort();
}

static struct plplay state;

int main(int argc, char *argv[])
{
    state = (struct plplay) {
        .target_override = true,
        .use_icc_luma = true,
        .fps = 60.0,
        .args = {
            .preset = &pl_render_default_params,
            .verbosity = PL_LOG_INFO,
        },
    };

    if (!parse_args(&state.args, argc, argv))
        return -1;

    state.log = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb    = pl_log_color,
        .log_level = state.args.verbosity,
    ));

    pl_options opts = state.opts = pl_options_alloc(state.log);
    pl_options_reset(opts, state.args.preset);

    // Enable this by default to save one click
    opts->params.cone_params = &opts->cone_params;

    // Enable dynamic parameters by default, due to plplay's heavy reliance on
    // GUI controls for dynamically adjusting render parameters.
    opts->params.dynamic_constants = true;

    // Hook up our pass info callback
    opts->params.info_callback = info_callback;
    opts->params.info_priv = &state;

    struct plplay *p = &state;
    if (!open_file(p, state.args.filename))
        goto error;

    const AVCodecParameters *par = p->stream->codecpar;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(par->format);
    if (!desc)
        goto error;

    struct window_params params = {
        .title = "plplay",
        .width = par->width,
        .height = par->height,
        .forced_impl = state.args.window_impl,
    };

    if (desc->flags & AV_PIX_FMT_FLAG_ALPHA) {
        params.alpha = true;
        opts->params.background_transparency = 1.0;
    }

    p->win = window_create(p->log, &params);
    if (!p->win)
        goto error;

    // Test the AVPixelFormat against the GPU capabilities
    if (!pl_test_pixfmt(p->win->gpu, par->format)) {
        fprintf(stderr, "Unsupported AVPixelFormat: %s\n", desc->name);
        goto error;
    }

#ifdef HAVE_NUKLEAR
    p->ui = ui_create(p->win->gpu);
    if (!p->ui)
        goto error;
#endif

    if (!init_codec(p))
        goto error;

    const char *cache_dir = get_cache_dir(&(char[512]) {0});
    if (cache_dir) {
        int ret = snprintf(p->cache_file, sizeof(p->cache_file), "%s/plplay.cache", cache_dir);
        if (ret > 0 && ret < sizeof(p->cache_file)) {
            p->cache = pl_cache_create(pl_cache_params(
                .log             = p->log,
                .max_total_size  = 50 << 20, // 50 MB
            ));
            pl_gpu_set_cache(p->win->gpu, p->cache);
            FILE *file = fopen(p->cache_file, "rb");
            if (file) {
                pl_cache_load_file(p->cache, file);
                p->cache_sig = pl_cache_signature(p->cache);
                fclose(file);
            }
        }
    }

    p->queue = pl_queue_create(p->win->gpu);
    int ret = pl_thread_create(&p->decoder_thread, decode_loop, p);
    if (ret != 0) {
        fprintf(stderr, "Failed creating decode thread: %s\n", strerror(errno));
        goto error;
    }

    p->decoder_thread_created = true;

    p->renderer = pl_renderer_create(p->log, p->win->gpu);
    if (!render_loop(p))
        goto error;

    printf("Exiting...\n");
    uninit(p);
    return 0;

error:
    uninit(p);
    return 1;
}
