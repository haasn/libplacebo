/* Compiling:
 *
 *   gcc plplay.c -o ./plplay -O2 -DUSE_VK \
 *       $(pkg-config --cflags --libs glfw3 vulkan libplacebo libavcodec libavformat libavutil)
 *
 *  or:
 *
 *   gcc plplay.c -o ./plplay -O2 -DUSE_GL \
 *       $(pkg-config --cflags --libs glfw3 libplacebo libavcodec libavformat libavutil)
 *
 * Notes:
 *
 * - This is a very shitty proof-of-concept. All it does is render a single
 *   video stream as fast as possible. It ignores timing completely, and
 *   handles several failure paths by just exiting the entire frame (when it
 *   could, instead, try re-creating the context). It should also be split up
 *   into separate files and given a meson.build, but for now it'll suffice.
 *
 * License: CC0 / Public Domain
 */

#if !defined(USE_GL) == !defined(USE_VK)
#error Specify exactly one of -DUSE_GL or -DUSE_VK when compiling!
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <libavutil/pixdesc.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>

#ifdef USE_VK
#define GLFW_INCLUDE_VULKAN
#endif

#include <GLFW/glfw3.h>
#include <libplacebo/renderer.h>
#include <libplacebo/utils/libav.h>

#ifdef USE_VK
#include <libplacebo/vulkan.h>
#endif

#ifdef USE_GL
#include <libplacebo/opengl.h>
#endif

#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480

struct plplay {
    bool should_exit;

    // libplacebo
    struct pl_context *ctx;
    const struct pl_gpu *gpu; // points to either vk->gpu or gl->gpu
    const struct pl_swapchain *swapchain;
    const struct pl_tex *plane_tex[4];
    struct pl_renderer *renderer;

#ifdef USE_VK
    VkSurfaceKHR surf;
    const struct pl_vulkan *vk;
    const struct pl_vk_inst *vk_inst;
#endif

#ifdef USE_GL
    const struct pl_opengl *gl;
#endif

    // GLFW
    GLFWwindow *win;

    // libav*
    AVFormatContext *format;
    AVCodecContext *codec;
    const AVStream *stream; // points to first video stream of `format`
};

static void uninit(struct plplay *p)
{
    if (p->gpu) {
        for (int i = 0; i < 4; i++)
            pl_tex_destroy(p->gpu, &p->plane_tex[i]);
    }

    pl_renderer_destroy(&p->renderer);
    pl_swapchain_destroy(&p->swapchain);

#ifdef USE_VK
    pl_vulkan_destroy(&p->vk);
    if (p->surf)
        vkDestroySurfaceKHR(p->vk_inst->instance, p->surf, NULL);
    pl_vk_inst_destroy(&p->vk_inst);
#endif

#ifdef USE_GL
    pl_opengl_destroy(&p->gl);
#endif

    avcodec_free_context(&p->codec);
    avformat_free_context(p->format);

    pl_context_destroy(&p->ctx);
    glfwTerminate();

    *p = (struct plplay) {0};
}

static bool init_glfw(void)
{
    if (!glfwInit()) {
        fprintf(stderr, "GLFW: Failed initializing?\n");
        return false;
    }

#ifdef USE_VK
    if (!glfwVulkanSupported()) {
        fprintf(stderr, "GLFW: No vulkan support! Perhaps recompile with -DUSE_GL\n");
        return false;
    }
#endif

    return true;
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

    // Find first video stream
    for (int i = 0; i < p->format->nb_streams; i++) {
        const AVStream *stream = p->format->streams[i];
        const AVCodecParameters *par = stream->codecpar;
        if (par->codec_type != AVMEDIA_TYPE_VIDEO)
            continue;

        printf("Found video track (stream %d)\n", i);
        printf("Resolution: %d x %d\n", par->width, par->height);
        printf("FPS: %f\n", av_q2d(stream->avg_frame_rate));
        printf("Bitrate: %"PRIi64" kbps\n", par->bit_rate / 1000);
        p->stream = stream;
        break;
    }

    if (!p->stream) {
        fprintf(stderr, "plplay: File contains no video streams?");
        return false;
    }

    return true;
}

static void resize_cb(GLFWwindow *win, int w, int h)
{
    struct plplay *p = glfwGetWindowUserPointer(win);
    if (!pl_swapchain_resize(p->swapchain, &w, &h)) {
        fprintf(stderr, "libplacebo: Failed resizing swapchain? Exiting...\n");
        p->should_exit = true;
    }
}

static void exit_cb(GLFWwindow *win)
{
    struct plplay *p = glfwGetWindowUserPointer(win);
    p->should_exit = true;
}

static bool create_window(struct plplay *p, int width, int height, bool alpha)
{
    printf("Creating %dx%d window%s...\n", width, height,
           alpha ? " (with alpha)" : "");

#ifdef USE_VK
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#endif

#ifdef USE_GL
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);

    /* Request OpenGL 3.2 (or higher) core profile */
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

    if (alpha)
        glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);

    p->win = glfwCreateWindow(width, height, "plplay", NULL, NULL);
    if (!p->win) {
        fprintf(stderr, "GLFW: Failed creating window\n");
        return false;
    }

    // Set up GLFW event callbacks
    glfwSetWindowUserPointer(p->win, p);
    glfwSetFramebufferSizeCallback(p->win, resize_cb);
    glfwSetWindowCloseCallback(p->win, exit_cb);

    return true;
}

#ifdef USE_VK
static bool int_renderer(struct plplay *p)
{
    assert(p->win);
    VkResult err;

    struct pl_vk_inst_params iparams = pl_vk_inst_default_params;
#ifndef NDEBUG
    iparams.debug = true;
#endif

    // Load all extensions required for WSI
    uint32_t num;
    iparams.extensions = glfwGetRequiredInstanceExtensions(&num);
    iparams.num_extensions = num;

    p->vk_inst = pl_vk_inst_create(p->ctx, &iparams);
    if (!p->vk_inst) {
        fprintf(stderr, "libplacebo: Failed creating vulkan instance\n");
        return false;
    }

    err = glfwCreateWindowSurface(p->vk_inst->instance, p->win, NULL, &p->surf);
    if (err != VK_SUCCESS) {
        fprintf(stderr, "GLFW: Failed creating vulkan surface\n");
        return false;
    }

    struct pl_vulkan_params params = pl_vulkan_default_params;
    params.instance = p->vk_inst->instance;
    params.surface = p->surf;
    params.allow_software = true;
    p->vk = pl_vulkan_create(p->ctx, &params);
    if (!p->vk) {
        fprintf(stderr, "libplacebo: Failed creating vulkan device\n");
        return false;
    }

    p->swapchain = pl_vulkan_create_swapchain(p->vk, &(struct pl_vulkan_swapchain_params) {
        .surface = p->surf,
        .present_mode = VK_PRESENT_MODE_FIFO_KHR,
    });

    if (!p->swapchain) {
        fprintf(stderr, "libplacebo: Failed creating vulkan swapchain\n");
        return false;
    }

    p->gpu = p->vk->gpu;
    p->renderer = pl_renderer_create(p->ctx, p->gpu);
    return true;
}
#endif // USE_VK

#ifdef USE_GL
static bool int_renderer(struct plplay *p)
{
    assert(p->win);

    struct pl_opengl_params params = pl_opengl_default_params;
#ifndef NDEBUG
    params.debug = true;
#endif

    glfwMakeContextCurrent(p->win);

    p->gl = pl_opengl_create(p->ctx, &params);
    if (!p->gl) {
        fprintf(stderr, "libplacebo: Failed creating opengl device\n");
        return false;
    }

    p->swapchain = pl_opengl_create_swapchain(p->gl, &(struct pl_opengl_swapchain_params) {
        .swap_buffers = (void (*)(void *)) glfwSwapBuffers,
        .priv = p->win,
    });

    if (!p->swapchain) {
        fprintf(stderr, "libplacebo: Failed creating opengl swapchain\n");
        return false;
    }

    int w, h;
    glfwGetFramebufferSize(p->win, &w, &h);
    if (!pl_swapchain_resize(p->swapchain, &w, &h)) {
        fprintf(stderr, "libplacebo: Failed initializing swapchain\n");
        return false;
    }

    p->gpu = p->gl->gpu;
    p->renderer = pl_renderer_create(p->ctx, p->gpu);
    return true;
}
#endif // USE_GL

static bool init_codec(struct plplay *p)
{
    assert(p->gpu);
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
    struct pl_swapchain_frame out_frame;
    int retry = 3;

    while (!pl_swapchain_start_frame(p->swapchain, &out_frame)) {
        if (retry-- == 0) {
            fprintf(stderr, "libplacebo: Swapchain appears stuck.. dropping frame\n");
            return true;
        }

        // Window possibly hidden/minimized/invisible?
        glfwWaitEventsTimeout(5e-3);
    }

    bool ret = true;

    struct pl_image image;
    struct pl_render_target target;
    struct pl_render_params params = pl_render_default_params;

    if (pl_upload_avframe(p->gpu, &image, p->plane_tex, in_frame)) {

        pl_render_target_from_swapchain(&target, &out_frame);
        pl_rect2df_aspect_copy(&target.dst_rect, &image.src_rect, 0.0);

        if (pl_render_target_partial(&target))
            pl_tex_clear(p->gpu, out_frame.fbo, (float[4]){ 0.0, 0.0, 0.0, 1.0 });

        if (!pl_render_image(p->renderer, &image, &target, &params)) {
            fprintf(stderr, "libplacebo: Failed rendering... GPU lost?\n");
            pl_tex_clear(p->gpu, out_frame.fbo, (float[4]){ 1.0, 0.0, 0.0, 1.0 });
            ret = false;
        }

    } else {

        fprintf(stderr, "libplacebo: Failed uploading AVFrame... dropping\n");
        pl_tex_clear(p->gpu, out_frame.fbo, (float[4]){ 0.0, 0.0, 0.0, 1.0 });

    }

    if (!pl_swapchain_submit_frame(p->swapchain)) {
        fprintf(stderr, "libplacebo: Failed submitting frame, swapchain lost?\n");
        return false;
    }

    pl_swapchain_swap_buffers(p->swapchain);
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

        glfwPollEvents();
        if (p->should_exit)
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

    if (!init_glfw())
        return 2;

    struct plplay state = {0};
    struct plplay *p = &state;

    p->ctx = pl_context_create(PL_API_VER, &(struct pl_context_params) {
        .log_cb    = pl_log_color,
        .log_level = PL_LOG_INFO,
    });
    assert(p->ctx);

    if (!open_file(p, filename))
        goto error;

    const AVCodecParameters *par = p->stream->codecpar;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(par->format);
    bool has_alpha = desc->flags & AV_PIX_FMT_FLAG_ALPHA;
    if (!create_window(p, par->width, par->height, has_alpha))
        goto error;

    if (!int_renderer(p))
        goto error;

    // TODO: Use direct rendering buffers
    if (!init_codec(p))
        goto error;

    if (!render_loop(p))
        goto error;

    printf("Exiting normally...\n");
    uninit(p);
    return 0;

error:
    uninit(p);
    return 1;
}
