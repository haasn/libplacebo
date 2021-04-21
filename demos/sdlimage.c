/* Simple image viewer that opens an image using SDL2_image and presents it
 * to the screen.
 *
 * License: CC0 / Public Domain
 */

#include <SDL2/SDL_image.h>

#include "common.h"
#include "window.h"

#include <libplacebo/renderer.h>
#include <libplacebo/shaders/lut.h>
#include <libplacebo/utils/upload.h>

// Static configuration, done in the file to keep things simple
static const char *icc_profile = ""; // path to ICC profile
static const char *lut_file = ""; // path to .cube lut

// Program state
static pl_log logger;
static struct window *win;

// For rendering
static pl_tex img_tex;
static pl_tex osd_tex;
static struct pl_plane img_plane;
static struct pl_plane osd_plane;
static pl_renderer renderer;
static struct pl_custom_lut *lut;

struct file
{
    void *data;
    size_t size;
};

static struct file icc_file;

static bool open_file(const char *path, struct file *out)
{
    if (!path || !path[0]) {
        *out = (struct file) {0};
        return true;
    }

    FILE *fp = NULL;
    bool success = false;

    fp = fopen(path, "rb");
    if (!fp)
        goto done;

    if (fseeko(fp, 0, SEEK_END))
        goto done;
    off_t size = ftello(fp);
    if (size < 0)
        goto done;
    if (fseeko(fp, 0, SEEK_SET))
        goto done;

    void *data = malloc(size);
    if (!fread(data, size, 1, fp))
        goto done;

    *out = (struct file) {
        .data = data,
        .size = size,
    };

    success = true;
done:
    if (fp)
        fclose(fp);
    return success;
}

static void close_file(struct file *file)
{
    if (!file->data)
        return;

    free(file->data);
    *file = (struct file) {0};
}

static void uninit(int ret)
{
    pl_renderer_destroy(&renderer);
    pl_tex_destroy(win->gpu, &img_tex);
    pl_tex_destroy(win->gpu, &osd_tex);
    close_file(&icc_file);
    pl_lut_free(&lut);

    window_destroy(&win);
    pl_log_destroy(&logger);
    exit(ret);
}

static bool upload_plane(const SDL_Surface *img, pl_tex *tex,
                         struct pl_plane *plane)
{
    if (!img)
        return false;

    SDL_Surface *fixed = NULL;
    const SDL_PixelFormat *fmt = img->format;
    if (SDL_ISPIXELFORMAT_INDEXED(fmt->format)) {
        // libplacebo doesn't handle indexed formats yet
        fixed = SDL_CreateRGBSurfaceWithFormat(0, img->w, img->h, 32,
                                               SDL_PIXELFORMAT_ABGR8888);
        SDL_BlitSurface((SDL_Surface *) img, NULL, fixed, NULL);
        img = fixed;
        fmt = img->format;
    }

    struct pl_plane_data data = {
        .type           = PL_FMT_UNORM,
        .width          = img->w,
        .height         = img->h,
        .pixel_stride   = fmt->BytesPerPixel,
        .row_stride     = img->pitch,
        .pixels         = img->pixels,
    };

    uint64_t masks[4] = { fmt->Rmask, fmt->Gmask, fmt->Bmask, fmt->Amask };
    pl_plane_data_from_mask(&data, masks);

    bool ok = pl_upload_plane(win->gpu, plane, tex, &data);
    SDL_FreeSurface(fixed);
    return ok;
}

static bool render_frame(const struct pl_swapchain_frame *frame)
{
    pl_tex img = img_plane.texture;
    struct pl_frame image = {
        .num_planes = 1,
        .planes     = { img_plane },
        .repr       = pl_color_repr_unknown,
        .color      = pl_color_space_unknown,
        .crop       = {0, 0, img->params.w, img->params.h},
    };

    // This seems to be the case for SDL2_image
    image.repr.alpha = PL_ALPHA_INDEPENDENT;

    struct pl_frame target;
    pl_frame_from_swapchain(&target, frame);
    target.profile = (struct pl_icc_profile) {
        .data = icc_file.data,
        .len = icc_file.size,
    };

    pl_rect2df_aspect_copy(&target.crop, &image.crop, 0.0);

    struct pl_overlay osd;
    struct pl_overlay_part osd_part;
    if (osd_tex) {
        osd_part = (struct pl_overlay_part) {
            .src = { 0, 0, osd_tex->params.w, osd_tex->params.h },
            .dst = { 0, 0, osd_tex->params.w, osd_tex->params.h },
        };
        osd = (struct pl_overlay) {
            .tex        = osd_tex,
            .mode       = PL_OVERLAY_NORMAL,
            .repr       = image.repr,
            .color      = image.color,
            .parts      = &osd_part,
            .num_parts  = 1,
        };
        target.overlays = &osd;
        target.num_overlays = 1;
        if (frame->flipped) {
            osd_part.dst.y0 = frame->fbo->params.h - osd_part.dst.y0;
            osd_part.dst.y1 = frame->fbo->params.h - osd_part.dst.y1;
        }
    }

    // Use the heaviest preset purely for demonstration/testing purposes
    struct pl_render_params params = pl_render_high_quality_params;
    params.lut = lut;

    return pl_render_image(renderer, &image, &target, &params);
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <image> [<overlay>]\n", argv[0]);
        return 255;
    }

    const char *file = argv[1];
    const char *overlay = argc > 2 ? argv[2] : NULL;
    logger = pl_log_create(PL_API_VER, &(struct pl_log_params) {
        .log_cb = pl_log_color,
        .log_level = PL_LOG_INFO,
    });


    // Load image, do this first so we can use it for the window size
    SDL_Surface *img = IMG_Load(file);
    if (!img) {
        fprintf(stderr, "Failed loading '%s': %s\n", file, SDL_GetError());
        uninit(1);
    }

    // Create window
    unsigned int start = SDL_GetTicks();
    win = window_create(logger, "SDL2_image demo", img->w, img->h, 0);
    if (!win)
        uninit(1);

    // Initialize rendering state
    if (!upload_plane(img, &img_tex, &img_plane)) {
        fprintf(stderr, "Failed uploading image plane!\n");
        uninit(2);
    }
    SDL_FreeSurface(img);

    if (overlay) {
        SDL_Surface *osd = IMG_Load(overlay);
        if (!upload_plane(osd, &osd_tex, &osd_plane))
            fprintf(stderr, "Failed uploading OSD plane.. continuing anyway\n");
        SDL_FreeSurface(osd);
    }

    if (!open_file(icc_profile, &icc_file))
        fprintf(stderr, "Failed opening ICC profile.. continuing anyway\n");

    struct file lutf;
    if (open_file(lut_file, &lutf) && lutf.size) {
        if (!(lut = pl_lut_parse_cube(logger, lutf.data, lutf.size)))
            fprintf(stderr, "Failed parsing LUT.. continuing anyway\n");
        close_file(&lutf);
    }

    renderer = pl_renderer_create(logger, win->gpu);

    unsigned int last = SDL_GetTicks(), frames = 0;
    printf("Took %u ms for initialization\n", last - start);

    // Render loop
    while (!win->window_lost) {
        struct pl_swapchain_frame frame;
        bool ok = pl_swapchain_start_frame(win->swapchain, &frame);
        if (!ok) {
            window_poll(win, true);
            continue;
        }

        if (!render_frame(&frame)) {
            fprintf(stderr, "libplacebo: Failed rendering frame!\n");
            uninit(3);
        }

        ok = pl_swapchain_submit_frame(win->swapchain);
        if (!ok) {
            fprintf(stderr, "libplacebo: Failed submitting frame!\n");
            uninit(3);
        }

        pl_swapchain_swap_buffers(win->swapchain);
        frames++;

        unsigned int now = SDL_GetTicks();
        if (now - last > 5000) {
            printf("%u frames in %u ms = %f FPS\n", frames, now - last,
                   1000.0f * frames / (now - last));
            last = now;
            frames = 0;
        }

        window_poll(win, false);
    }

    uninit(0);
}
