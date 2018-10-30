#include "gpu_tests.h"

static void vulkan_tests(const struct pl_gpu *gpu)
{
    if (gpu->handle_caps & PL_HANDLE_FD) {
        const struct pl_buf *buf = pl_buf_create(gpu, &(struct pl_buf_params) {
            .type = PL_BUF_TEX_TRANSFER,
            .size = 1024,
            .ext_handles = PL_HANDLE_FD,
        });

        REQUIRE(buf);
        REQUIRE(buf->handles.fd);
        REQUIRE(buf->handles.size >= buf->params.size);
        REQUIRE(pl_buf_export(gpu, buf));
        pl_buf_destroy(gpu, &buf);
    }
}

int main()
{
    struct pl_context *ctx = pl_test_context();
    struct pl_vk_inst_params iparams = pl_vk_inst_default_params;
    iparams.debug = true;
    const struct pl_vk_inst *vkinst = pl_vk_inst_create(ctx, &iparams);
    if (!vkinst)
        return SKIP;

    struct pl_vulkan_params params = pl_vulkan_default_params;
    params.instance = vkinst->instance;
    const struct pl_vulkan *vk = pl_vulkan_create(ctx, &params);
    if (!vk)
        return SKIP;

    vulkan_tests(vk->gpu);
    gpu_tests(vk->gpu);
    pl_vulkan_destroy(&vk);
    pl_vk_inst_destroy(&vkinst);
    pl_context_destroy(&ctx);
}
