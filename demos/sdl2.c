/* Compiling:
 *
 *   gcc sdl2.c -o ./sdl2 -O2 \
 *       $(pkg-config --cflags --libs sdl2 SDL2_image vulkan libplacebo)
 *
 * Notes:
 *
 * - This proof-of-concept is extremely naive. It uses global state, and
 *   ignores uninitialization on errors (just exit()s). This is probably not
 *   what you should be doing for a real program, but I wanted to avoid the
 *   example becoming too complicated.
 *
 * License: CC0 / Public Domain
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <SDL2/SDL_image.h>
#include <vulkan/vulkan.h>

#include <libplacebo/renderer.h>
#include <libplacebo/utils/upload.h>
#include <libplacebo/vulkan.h>

#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480

SDL_Window *window;
VkSurfaceKHR surf;
struct pl_context *ctx;

const struct pl_vulkan *vk;
const struct pl_vk_inst *vk_inst;
const struct pl_swapchain *swapchain;

// for rendering
const struct pl_tex *img_tex;
const struct pl_tex *osd_tex;
struct pl_plane img_plane;
struct pl_plane osd_plane;
struct pl_renderer *renderer;
struct file icc_profile;

struct file
{
    void *data;
    size_t size;
};

static bool open_file(const char *path, struct file *out)
{
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

static void uninit()
{
    pl_renderer_destroy(&renderer);
    pl_tex_destroy(vk->gpu, &img_tex);
    pl_tex_destroy(vk->gpu, &osd_tex);
    pl_swapchain_destroy(&swapchain);
    pl_vulkan_destroy(&vk);
    vkDestroySurfaceKHR(vk_inst->instance, surf, NULL);
    pl_vk_inst_destroy(&vk_inst);
    pl_context_destroy(&ctx);
    close_file(&icc_profile);

    SDL_DestroyWindow(window);
    SDL_Quit();
}

static void init_sdl() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "Failed to initialize SDL2: %s\n", SDL_GetError());
        exit(1);
    }

    uint32_t window_flags = SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE |
                            SDL_WINDOW_VULKAN;

    window = SDL_CreateWindow("libplacebo demo",
                              SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                              WINDOW_WIDTH, WINDOW_HEIGHT, window_flags);

    if (!window) {
        fprintf(stderr, "Failed creating window: %s\n", SDL_GetError());
        exit(1);
    }
}

static void init_placebo() {
    ctx = pl_context_create(PL_API_VER, &(struct pl_context_params) {
        .log_cb    = pl_log_color,
        .log_level = PL_LOG_DEBUG,
    });

    assert(ctx);
}

static void init_vulkan()
{
    struct pl_vk_inst_params iparams = pl_vk_inst_default_params;
#ifndef NDEBUG
    iparams.debug = true;
#endif

    unsigned int num = 0;
    if (!SDL_Vulkan_GetInstanceExtensions(window, &num, NULL)) {
        fprintf(stderr, "Failed enumerating vulkan extensions: %s\n", SDL_GetError());
        exit(1);
    }

    iparams.extensions = malloc(num * sizeof(const char *));
    iparams.num_extensions = num;
    assert(iparams.extensions);

    bool ok = SDL_Vulkan_GetInstanceExtensions(window, &num,(const char **) iparams.extensions);
    assert(ok);

    if (num > 0) {
        printf("Requesting %d additional vulkan extensions:\n", num);
        for (int i = 0; i < num; i++)
            printf("    %s\n", iparams.extensions[i]);
    }

    vk_inst = pl_vk_inst_create(ctx, &iparams);
    if (!vk_inst) {
        fprintf(stderr, "Failed creating vulkan instance!\n");
        exit(2);
    }

    free((const char **) iparams.extensions);
    if (!SDL_Vulkan_CreateSurface(window, vk_inst->instance, &surf)) {
        fprintf(stderr, "Failed creating vulkan surface: %s\n", SDL_GetError());
        exit(1);
    }

    struct pl_vulkan_params params = pl_vulkan_default_params;
    params.instance = vk_inst->instance;
    params.surface = surf;
    params.allow_software = true;
    vk = pl_vulkan_create(ctx, &params);
    if (!vk) {
        fprintf(stderr, "Failed creating vulkan device!\n");
        exit(2);
    }

    swapchain = pl_vulkan_create_swapchain(vk, &(struct pl_vulkan_swapchain_params) {
        .surface = surf,
        .present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR,
    });

    if (!swapchain) {
        fprintf(stderr, "Failed creating vulkan swapchain!\n");
        exit(2);
    }

    int w = WINDOW_WIDTH, h = WINDOW_HEIGHT;
    if (!pl_swapchain_resize(swapchain, &w, &h)) {
        fprintf(stderr, "Failed resizing vulkan swapchain!\n");
        exit(2);
    }

    if (w != WINDOW_WIDTH || h != WINDOW_HEIGHT)
        printf("Note: window dimensions differ (got %dx%d)\n", w, h);
}

static bool upload_plane(const char *filename, const struct pl_tex **tex,
                         struct pl_plane *plane)
{
    if (!filename)
        return true;

    SDL_Surface *img = IMG_Load(filename);
    if (!img) {
        fprintf(stderr, "Failed loading '%s': %s\n", filename, SDL_GetError());
        return false;
    }

    const SDL_PixelFormat *fmt = img->format;
    if (SDL_ISPIXELFORMAT_INDEXED(fmt->format)) {
        SDL_Surface *fixed;
        fixed = SDL_CreateRGBSurfaceWithFormat(0, img->w, img->h, 32,
                                               SDL_PIXELFORMAT_ABGR8888);
        SDL_BlitSurface(img, NULL, fixed, NULL);
        SDL_FreeSurface(img);
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

    bool ok = pl_upload_plane(vk->gpu, plane, tex, &data);
    SDL_FreeSurface(img);

    return ok;
}

static void init_rendering(const char *img, const char *osd, const char *icc)
{
    if (!upload_plane(img, &img_tex, &img_plane)) {
        fprintf(stderr, "Failed uploading image plane!\n");
        exit(2);
    }

    if (!upload_plane(osd, &osd_tex, &osd_plane))
        fprintf(stderr, "Failed uploading OSD plane.. continuing anyway\n");

    if (!open_file(icc, &icc_profile))
        fprintf(stderr, "Failed opening ICC profile.. continuing anyway\n");

    // Create a renderer instance
    renderer = pl_renderer_create(ctx, vk->gpu);
}

static void render_frame(const struct pl_swapchain_frame *frame)
{
    const struct pl_tex *img = img_plane.texture;
    struct pl_image image = {
        .num_planes = 1,
        .planes     = { img_plane },
        .repr       = pl_color_repr_unknown,
        .color      = pl_color_space_unknown,
        .src_rect   = {0, 0, img->params.w, img->params.h},
    };

    // This seems to be the case for SDL2_image
    image.repr.alpha = PL_ALPHA_INDEPENDENT;

    struct pl_render_target target;
    pl_render_target_from_swapchain(&target, frame);
    target.profile = (struct pl_icc_profile) {
        .data = icc_profile.data,
        .len = icc_profile.size,
    };

    pl_rect2d_aspect_copy(&target.dst_rect, &image.src_rect, 0.0);

    const struct pl_tex *osd = osd_plane.texture;
    struct pl_overlay target_ol;
    if (osd) {
        target_ol = (struct pl_overlay) {
            .plane      = osd_plane,
            .rect       = { 0, 0, osd->params.w, osd->params.h },
            .mode       = PL_OVERLAY_NORMAL,
            .repr       = image.repr,
            .color      = image.color,
        };
        target.overlays = &target_ol;
        target.num_overlays = 1;
    }

    if (pl_render_target_partial(&target))
        pl_tex_clear(vk->gpu, target.fbo, (float[4]) {0} );

    // Use the heaviest preset purely for demonstration/testing purposes
    if (!pl_render_image(renderer, &image, &target, &pl_render_high_quality_params)) {
        fprintf(stderr, "Failed rendering frame!\n");
        uninit();
        exit(2);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "Usage: ./sdl2 <image> [<overlay>] [<icc profile>]\n");
        return 255;
    }

    unsigned int start = SDL_GetTicks();

    int ret = 0;
    init_sdl();
    init_placebo();
    init_vulkan();
    init_rendering(argv[1], argc > 2 ? argv[2] : NULL, argc > 3 ? argv[3] : NULL);

    // Resize the window to match the content
    const struct pl_tex *img = img_plane.texture;
    SDL_SetWindowSize(window, img->params.w, img->params.h);

    unsigned int last = SDL_GetTicks(), frames = 0;
    printf("Took %u ms for initialization\n", last - start);

    while (true) {
        SDL_Event evt;
        while (SDL_PollEvent(&evt) == 1) {
            if (evt.type == SDL_QUIT)
                goto cleanup;
        }

        struct pl_swapchain_frame frame;
        bool ok = pl_swapchain_start_frame(swapchain, &frame);
        if (!ok) {
            SDL_Delay(10);
            continue;
        }

        render_frame(&frame);
        ok = pl_swapchain_submit_frame(swapchain);
        if (!ok) {
            fprintf(stderr, "Failed submitting frame!\n");
            ret = 3;
            goto cleanup;
        }

        pl_swapchain_swap_buffers(swapchain);
        frames++;

        unsigned int now = SDL_GetTicks();
        if (now - last > 5000) {
            printf("%u frames in %u ms = %f FPS\n", frames, now - last,
                   1000.0f * frames / (now - last));
            last = now;
            frames = 0;
        }
    }

cleanup:
    uninit();
    return ret;
}
