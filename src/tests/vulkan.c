#include "gpu_tests.h"

static void vulkan_tests(const struct pl_gpu *gpu)
{
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
}

int main()
{
    struct pl_context *ctx = pl_test_context();

    struct pl_vulkan_params params = pl_vulkan_default_params;
    params.instance_params = &(struct pl_vk_inst_params) { .debug = true };
    const struct pl_vulkan *vk = pl_vulkan_create(ctx, &params);
    if (!vk)
        return SKIP;

    vulkan_tests(vk->gpu);
    gpu_tests(vk->gpu);
    pl_vulkan_destroy(&vk);
    pl_context_destroy(&ctx);
}
