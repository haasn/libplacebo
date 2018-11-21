#include "gpu_tests.h"
#include "vulkan/command.h"
#include "vulkan/gpu.h"

static void vulkan_tests(const struct pl_vulkan *pl_vk)
{
    const struct pl_gpu *gpu = pl_vk->gpu;

    if (gpu->handle_caps.shared_mem & PL_HANDLE_FD) {
        const struct pl_buf *buf = pl_buf_create(gpu, &(struct pl_buf_params) {
            .type = PL_BUF_TEX_TRANSFER,
            .size = 1024,
            .handle_type = PL_HANDLE_FD,
        });

        REQUIRE(buf);
        REQUIRE(buf->shared_mem.handle.fd);
        REQUIRE(buf->shared_mem.size >= buf->params.size);
        REQUIRE(pl_buf_export(gpu, buf));
        pl_buf_destroy(gpu, &buf);
    }

    const struct pl_fmt *fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 0, 0,
                                           PL_FMT_CAP_BLITTABLE);
    if (!fmt)
        return;

    if (gpu->handle_caps.sync & PL_HANDLE_FD) {
        const struct pl_sync *sync = pl_sync_create(gpu, PL_HANDLE_FD);
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

int main()
{
    struct pl_context *ctx = pl_test_context();

    struct pl_vulkan_params params = pl_vulkan_default_params;
    params.instance_params = &(struct pl_vk_inst_params) { .debug = true };
    const struct pl_vulkan *vk = pl_vulkan_create(ctx, &params);
    if (!vk)
        return SKIP;

    gpu_tests(vk->gpu);
    vulkan_tests(vk);
    pl_vulkan_destroy(&vk);
    pl_context_destroy(&ctx);
}
