#include "gpu_tests.h"

int main()
{
    struct pl_context *ctx = pl_test_context();
    const struct pl_gpu *gpu = pl_gpu_dummy_create(ctx, NULL);
    pl_texture_tests(gpu);

    // Attempt creating a shader and accessing the resulting LUT
    const struct pl_tex *dummy = pl_tex_dummy_create(gpu, &(struct pl_tex_params) {
        .w = 100,
        .h = 100,
        .format = pl_find_named_fmt(gpu, "rgba8"),
        .sample_mode = PL_TEX_SAMPLE_LINEAR,
        .address_mode = PL_TEX_ADDRESS_CLAMP,
    });

    struct pl_sample_src src = {
        .tex = dummy,
        .new_w = 1000,
        .new_h = 1000,
    };

    struct pl_shader_obj *lut = NULL;
    struct pl_sample_filter_params filter_params = {
        .filter = pl_filter_ewa_lanczos,
        .lut = &lut,
    };

    struct pl_shader *sh;
    sh = pl_shader_alloc(ctx, &(struct pl_shader_params) { .gpu = gpu });
    REQUIRE(pl_shader_sample_polar(sh, &src, &filter_params));
    const struct pl_shader_res *res = pl_shader_finalize(sh);
    REQUIRE(res);

    for (int n = 0; n < res->num_descriptors; n++) {
        const struct pl_shader_desc *sd = &res->descriptors[n];
        if (sd->desc.type != PL_DESC_SAMPLED_TEX)
            continue;

        const struct pl_tex *tex = sd->object;
        const float *data = (float *) pl_tex_dummy_data(tex);
        if (!data)
            continue; // means this was the `dummy` texture

        for (int i = 0; i < tex->params.w; i++)
            printf("lut[%d] = %f\n", i, data[i]);
    }

    pl_shader_free(&sh);
    pl_shader_obj_destroy(&lut);
    pl_tex_destroy(gpu, &dummy);
    pl_gpu_dummy_destroy(&gpu);
    pl_context_destroy(&ctx);
}
