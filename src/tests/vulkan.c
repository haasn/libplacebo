#include "ra_tests.h"

int main()
{
    struct pl_context *ctx = pl_test_context();
    struct pl_vulkan_params params = pl_vulkan_default_params;
    params.debug = true;

    const struct pl_vulkan *vk = pl_vulkan_create(ctx, &params);
    if (!vk)
        return SKIP;

    ra_tests(vk->ra);
    pl_vulkan_destroy(&vk);
    pl_context_destroy(&ctx);
}
