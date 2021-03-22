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

#include <libavutil/pixdesc.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>

#include "common.h"
#include "window.h"

#include <libplacebo/renderer.h>
#include <libplacebo/utils/libav.h>
#include <libplacebo/utils/frame_queue.h>

struct plplay {
    struct window *win;

    // libplacebo
    struct pl_context *ctx;
    const struct pl_tex *plane_tex[4];
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

    // settings
    struct pl_render_params params;
};

static void uninit(struct plplay *p)
{
    const struct pl_gpu *gpu = p->win->gpu;
    if (gpu) {
        for (int i = 0; i < 4; i++)
            pl_tex_destroy(gpu, &p->plane_tex[i]);
    }

    pl_queue_destroy(&p->queue);
    pl_renderer_destroy(&p->renderer);
    window_destroy(&p->win);

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
        fprintf(stderr, "libavformat: Failed opening file!");
        return false;
    }

    printf("Format: %s\n", p->format->iformat->name);
    printf("Duration: %.3f s\n", p->format->duration / 1e6);

    if (avformat_find_stream_info(p->format,  NULL) < 0) {
        fprintf(stderr, "libavformat: Failed finding stream info!");
        return false;
    }

    // Find "best" video stream
    int stream_idx =
        av_find_best_stream(p->format, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);

    if (stream_idx < 0) {
        fprintf(stderr, "plplay: File contains no video streams?");
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

static bool render_frame(struct plplay *p, const struct pl_swapchain_frame *frame,
                         const struct pl_frame_mix *mix)
{
    const struct pl_gpu *gpu = p->win->gpu;
    struct pl_frame target;
    pl_frame_from_swapchain(&target, frame);

    assert(mix->num_frames);
    pl_rect2df_aspect_copy(&target.crop, &mix->frames[0]->crop, 0.0);
    if (pl_frame_is_cropped(&target))
        pl_frame_clear(gpu, &target, (float[3]) {0});

    return pl_render_image_mix(p->renderer, mix, &target, &p->params);
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
    };

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
