#include "gpu_tests.h"
#include "vulkan/command.h"
#include "vulkan/gpu.h"

static void vulkan_tests(const struct pl_vulkan *pl_vk,
                         enum pl_handle_type handle_type)
{
    const struct pl_gpu *gpu = pl_vk->gpu;

    if (gpu->export_caps.buf & handle_type) {
        const struct pl_buf *buf = pl_buf_create(gpu, &(struct pl_buf_params) {
            .type = PL_BUF_TEX_TRANSFER,
            .size = 1024,
            .handle_type = handle_type,
        });

        REQUIRE(buf);
        REQUIRE(buf->shared_mem.handle.fd > -1);
        REQUIRE(buf->shared_mem.size >= buf->params.size);
        REQUIRE(pl_buf_export(gpu, buf));
        pl_buf_destroy(gpu, &buf);
    }

    const struct pl_fmt *fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 0, 0,
                                           PL_FMT_CAP_BLITTABLE);
    if (!fmt)
        return;

    if (gpu->export_caps.sync & handle_type) {
        const struct pl_sync *sync = pl_sync_create(gpu, handle_type);
        const struct pl_tex *tex = pl_tex_create(gpu, &(struct pl_tex_params) {
            .w = 32,
            .h = 32,
            .format = fmt,
            .blit_dst = true,
        });

        REQUIRE(sync);
        REQUIRE(tex);
        REQUIRE(pl_tex_export(gpu, tex, sync));

        // Re-use our internal helpers to signal this VkSemaphore
        struct vk_ctx *vk = pl_vk->priv;
        struct vk_cmd *cmd = vk_cmd_begin(vk, vk->pool_graphics);
        VkSemaphore signal;
        REQUIRE(cmd);
        pl_vk_sync_unwrap(sync, NULL, &signal);
        vk_cmd_sig(cmd, signal);
        vk_cmd_queue(vk, cmd);
        REQUIRE(vk_flush_commands(vk));

        // Do something with the image again to "import" it
        pl_tex_clear(gpu, tex, (float[4]){0});
        pl_gpu_finish(gpu);

        pl_sync_destroy(gpu, &sync);
        pl_tex_destroy(gpu, &tex);
    }
}

static void vulkan_test_export_import(const struct pl_vulkan *pl_vk,
                                      enum pl_handle_type handle_type)
{
    const struct pl_gpu *gpu = pl_vk->gpu;

    if (!(gpu->export_caps.tex & handle_type) ||
        !(gpu->import_caps.tex & handle_type))
        return;

    const struct pl_fmt *fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 0, 0,
                                           PL_FMT_CAP_BLITTABLE);
    if (!fmt)
        return;

    const struct pl_tex *export = pl_tex_create(gpu, &(struct pl_tex_params) {
        .w = 32,
        .h = 32,
        .format = fmt,
        .export_handle = handle_type,
    });
    REQUIRE(export);
    REQUIRE(export->shared_mem.handle.fd > -1);

    const struct pl_tex *import = pl_tex_create(gpu, &(struct pl_tex_params) {
        .w = 32,
        .h = 32,
        .format = fmt,
        .import_handle = handle_type,
        .shared_mem = export->shared_mem,
    });
    REQUIRE(import);

    pl_tex_destroy(gpu, &import);
    pl_tex_destroy(gpu, &export);
}

int main()
{
    struct pl_context *ctx = pl_test_context();

    struct pl_vulkan_params params = pl_vulkan_default_params;
    params.instance_params = &(struct pl_vk_inst_params) { .debug = true };
    const struct pl_vulkan *vk = pl_vulkan_create(ctx, &params);
    if (!vk)
        return SKIP;

    gpu_tests(vk->gpu);
#ifdef VK_HAVE_UNIX
    vulkan_tests(vk, PL_HANDLE_FD);
    vulkan_tests(vk, PL_HANDLE_DMA_BUF);
    vulkan_test_export_import(vk, PL_HANDLE_DMA_BUF);
#endif
#ifdef VK_HAVE_WIN32
    vulkan_tests(vk, PL_HANDLE_WIN32);
    vulkan_tests(vk, PL_HANDLE_WIN32_KMT);
#endif
    pl_vulkan_destroy(&vk);
    pl_context_destroy(&ctx);
}
