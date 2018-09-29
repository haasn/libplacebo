#include "gpu_tests.h"
#include <libplacebo/vulkan.h>

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

    gpu_tests(vk->gpu);
    pl_vulkan_destroy(&vk);
    pl_vk_inst_destroy(&vkinst);
    pl_context_destroy(&ctx);
}
