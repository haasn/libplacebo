/* Very shitty proof-of-concept video player based on ffmpeg. All it does is
 * render a single video stream as fast as possible. It ignores timing
 * completely, and handles several failure paths by just exiting the entire
 * program (when it could, instead, try re-creating the context).
 *
 * License: CC0 / Public Domain
 */

#include <libavutil/pixdesc.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>

#include "common.h"
#include "window.h"

#include <libplacebo/renderer.h>
#include <libplacebo/utils/libav.h>

struct plplay {
    struct window *win;

    // libplacebo
    struct pl_context *ctx;
    const struct pl_tex *plane_tex[4];
    struct pl_renderer *renderer;

    // libav*
    AVFormatContext *format;
    AVCodecContext *codec;
    const AVStream *stream; // points to first video stream of `format`
};

static void uninit(struct plplay *p)
{
    const struct pl_gpu *gpu = p->win->gpu;
    if (gpu) {
        for (int i = 0; i < 4; i++)
            pl_tex_destroy(gpu, &p->plane_tex[i]);
    }

    pl_renderer_destroy(&p->renderer);
    window_destroy(&p->win);

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

static bool render_frame(struct plplay *p, AVFrame *in_frame)
{
    const struct pl_gpu *gpu = p->win->gpu;
    struct pl_swapchain_frame out_frame;
    int retry = 3;

    while (!pl_swapchain_start_frame(p->win->swapchain, &out_frame)) {
        if (retry-- == 0) {
            fprintf(stderr, "libplacebo: Swapchain appears stuck.. dropping frame\n");
            return true;
        }

        // Window possibly hidden/minimized/invisible? Block for window events
        window_poll(p->win, true);
    }

    bool ret = true;

    struct pl_frame image, target;
    struct pl_render_params params = pl_render_default_params;

    if (pl_upload_avframe(gpu, &image, p->plane_tex, in_frame)) {

        pl_frame_from_swapchain(&target, &out_frame);
        pl_rect2df_aspect_copy(&target.crop, &image.crop, 0.0);

        if (pl_frame_is_cropped(&target))
            pl_frame_clear(gpu, &target, (float[3]) {0});

        if (!pl_render_image(p->renderer, &image, &target, &params)) {
            fprintf(stderr, "libplacebo: Failed rendering... GPU lost?\n");
            pl_tex_clear(gpu, out_frame.fbo, (float[4]){ 1.0, 0.0, 0.0, 1.0 });
            ret = false;
        }

    } else {

        fprintf(stderr, "libplacebo: Failed uploading AVFrame... dropping\n");
        pl_tex_clear(gpu, out_frame.fbo, (float[4]){ 0.0, 0.0, 0.0, 1.0 });

    }

    if (!pl_swapchain_submit_frame(p->win->swapchain)) {
        fprintf(stderr, "libplacebo: Failed submitting frame, swapchain lost?\n");
        return false;
    }

    pl_swapchain_swap_buffers(p->win->swapchain);
    return ret;
}

static bool decode_packet(struct plplay *p, AVPacket *packet, AVFrame *frame)
{
    int ret;

    if ((ret = avcodec_send_packet(p->codec, packet)) < 0) {
        fprintf(stderr, "libavcodec: Failed sending packet to decoder: %s\n",
                av_err2str(ret));
        return false;
    }

    while (true) {
        ret = avcodec_receive_frame(p->codec, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            return true;
        } else if (ret < 0) {
            fprintf(stderr, "libavcodec: Failed receiving frame: %s\n",
                    av_err2str(ret));
            return false;
        }

        // TODO: Put this onto a separate thread and wait until the
        // corresponding correct PTS!
        if (!render_frame(p, frame)) {
            fprintf(stderr, "libplacebo: Failed rendering! Aborting...\n");
            return false;
        }
    }
}

static bool render_loop(struct plplay *p)
{
    int ret = true;

    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    if (!packet || !frame) {
        ret = false;
        goto error;
    }

    while (av_read_frame(p->format, packet) >= 0) {
        if (packet->stream_index != p->stream->index) {
            // Ignore all unrelated packets
            av_packet_unref(packet);
            continue;
        }

        if (!decode_packet(p, packet, frame))
            break;
        av_packet_unref(packet);

        window_poll(p->win, false);
        if (p->win->window_lost)
            break;
    }

    // fall through
error:
    av_frame_free(&frame);
    av_packet_free(&packet);
    return ret >= 0;
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

    struct plplay state = {0};
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

    p->renderer = pl_renderer_create(p->ctx, p->win->gpu);
    if (!render_loop(p))
        goto error;

    printf("Exiting normally...\n");
    uninit(p);
    return 0;

error:
    uninit(p);
    return 1;
}
