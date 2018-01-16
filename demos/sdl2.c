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
 * - The format handling is currently hard-coded to just use SDL conversion
 *   functions to force the format to be the one we want it to be, and we also
 *   ignore the possibility of the stride being different from the width. I
 *   wanted to avoid this example turning into an exercise in image format
 *   picking. (I have other helpers for that planned)
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

#include <libplacebo/colorspace.h>
#include <libplacebo/context.h>
#include <libplacebo/dispatch.h>
#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/sampling.h>
#include <libplacebo/swapchain.h>
#include <libplacebo/vulkan.h>

#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480

SDL_Window *window;
VkSurfaceKHR surf;
struct pl_context *ctx;

const struct pl_vulkan *vk;
const struct pl_vk_inst *vk_inst;
const struct ra_swapchain *swapchain;

// for rendering
const struct ra_tex *lena;
struct pl_dispatch *dispatch;
struct pl_shader_obj *dither_state;

static void init_sdl() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "Failed to initialize SDL2: %s\n", SDL_GetError());
        exit(1);
    }

    window = SDL_CreateWindow("libplacebo demo",
                              SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                              WINDOW_WIDTH, WINDOW_HEIGHT,
                              SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE |
                              SDL_WINDOW_VULKAN);

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
    iparams.debug = true;

    unsigned int num = 0;
    if (!SDL_Vulkan_GetInstanceExtensions(window, &num, NULL)) {
        fprintf(stderr, "Failed enumerating vulkan extensions: %s\n", SDL_GetError());
        exit(1);
    }

    iparams.extensions = malloc(num * sizeof(const char *));
    iparams.num_extensions = num;
    assert(iparams.extensions);

    bool ok = SDL_Vulkan_GetInstanceExtensions(window, &num, iparams.extensions);
    assert(ok);

    if (num > 0) {
        printf("Requesting %d additional vulkan extensions:\n", num);
        for (int i = 0; i < num; i++)
            printf("    %s\n", iparams.extensions[i]);
    }

    vk_inst = pl_vk_inst_create(ctx, &iparams);
    if (!vk_inst) {
        fprintf(stderr, "Failed creating vulkan instance!");
        exit(2);
    }

    free(iparams.extensions);
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
        fprintf(stderr, "Failed creating vulkan device!");
        exit(2);
    }

    swapchain = pl_vulkan_create_swapchain(vk, &(struct pl_vulkan_swapchain_params) {
        .surface = surf,
        .present_mode = VK_PRESENT_MODE_FIFO_KHR,
    });
    if (!swapchain) {
        fprintf(stderr, "Failed creating vulkan swapchain!");
        exit(2);
    }
}

static void init_rendering()
{
    // Pick a good texture format
    const struct ra_fmt *fmt;
    fmt = ra_find_fmt(vk->ra, RA_FMT_UNORM, 4, 8, 8, RA_FMT_CAP_LINEAR);
    if (!fmt) {
        fprintf(stderr, "Failed picking any suitable ra_fmt!\n");
        exit(2);
    }

    SDL_Surface *jpg = IMG_Load("lena.jpg");
    if (!jpg) {
        fprintf(stderr, "Failed loading 'lena.jpg': %s\n", SDL_GetError());
        exit(1);
    }

    // For simplicity, just always convert the loaded image to our chosen fmt
    // Note: SDL2 treats pixels as a single word, so on little-endian platforms
    // we need to specify ABGR in order to have the R bits be the first byte in
    // memory. For big endian this would have to be a different magic constant,
    // but this is a hack anyway so I won't worry about it.
    SDL_Surface *img = SDL_CreateRGBSurfaceWithFormat(0, jpg->w, jpg->h, 32,
                                                      SDL_PIXELFORMAT_ABGR8888);
    SDL_BlitSurface(jpg, NULL, img, NULL);

    // for simplicity, ignore the stride so we can use initial_data and not
    // have to worry about it
    assert(img->pitch == img->w * img->format->BytesPerPixel);
    lena = ra_tex_create(vk->ra, &(struct ra_tex_params) {
        .w = img->w,
        .h = img->h,
        .format = fmt,
        .sampleable = true,
        .sample_mode = RA_TEX_SAMPLE_LINEAR,
        .initial_data = img->pixels,
    });

    SDL_FreeSurface(img);
    SDL_FreeSurface(jpg);

    if (!lena) {
        fprintf(stderr, "Failed creating lena texture!\n");
        exit(2);
    }

    // create a shader dispatch object
    dispatch = pl_dispatch_create(ctx, vk->ra);
}

static void render_frame(const struct ra_swapchain_frame *frame)
{
    const struct ra_tex *fbo = frame->fbo;

    // Record some example rendering commands
    struct pl_shader *sh = pl_dispatch_begin(dispatch);
    pl_shader_sample_bicubic(sh, &(struct pl_sample_src) {
        .tex   = lena,
        .new_w = fbo->params.w,
        .new_h = fbo->params.h,
    });

    int depth = frame->color_repr.bits.color_depth;
    pl_shader_dither(sh, depth, &dither_state, NULL);

    ra_tex_clear(vk->ra, fbo, (float[4]){ 1.0, 0.5, 0.0, 1.0 });
    pl_dispatch_finish(dispatch, sh, fbo, NULL);
}

static void uninit()
{
    pl_shader_obj_destroy(&dither_state);
    pl_dispatch_destroy(&dispatch);
    ra_tex_destroy(vk->ra, &lena);
    ra_swapchain_destroy(&swapchain);
    pl_vulkan_destroy(&vk);
    vkDestroySurfaceKHR(vk_inst->instance, surf, NULL);
    pl_vk_inst_destroy(&vk_inst);
    pl_context_destroy(&ctx);

    SDL_DestroyWindow(window);
    SDL_Quit();
}

int main()
{
    int ret = 0;
    init_sdl();
    init_placebo();
    init_vulkan();
    init_rendering();

    while (true) {
        SDL_Event evt;
        while (SDL_PollEvent(&evt) == 1) {
            if (evt.type == SDL_QUIT)
                goto cleanup;
        }

        struct ra_swapchain_frame frame;
        pl_dispatch_reset_frame(dispatch);
        bool ok = ra_swapchain_start_frame(swapchain, &frame);
        if (!ok) {
            SDL_Delay(10);
            continue;
        }

        render_frame(&frame);
        ok = ra_swapchain_submit_frame(swapchain);
        if (!ok) {
            fprintf(stderr, "Failed submitting frame!");
            ret = 3;
            goto cleanup;
        }

        ra_swapchain_swap_buffers(swapchain);
    }

cleanup:
    uninit();
    return ret;
}
