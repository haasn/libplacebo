#include "gpu_tests.h"

int main()
{
    pl_log log = pl_test_logger();
    const struct pl_gpu *gpu = pl_gpu_dummy_create(log, NULL);
    pl_buffer_tests(gpu);
    pl_texture_tests(gpu);

    // Attempt creating a shader and accessing the resulting LUT
    const struct pl_tex *dummy = pl_tex_dummy_create(gpu, &(struct pl_tex_dummy_params) {
        .w = 100,
        .h = 100,
        .format = pl_find_named_fmt(gpu, "rgba8"),
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
    sh = pl_shader_alloc(log, &(struct pl_shader_params) { .gpu = gpu });
    REQUIRE(pl_shader_sample_polar(sh, &src, &filter_params));
    const struct pl_shader_res *res = pl_shader_finalize(sh);
    REQUIRE(res);

    for (int n = 0; n < res->num_descriptors; n++) {
        const struct pl_shader_desc *sd = &res->descriptors[n];
        if (sd->desc.type != PL_DESC_SAMPLED_TEX)
            continue;

        const struct pl_tex *tex = sd->binding.object;
        const float *data = (float *) pl_tex_dummy_data(tex);
        if (!data)
            continue; // means this was the `dummy` texture

        for (int i = 0; i < tex->params.w; i++)
            printf("lut[%d] = %f\n", i, data[i]);
    }

    // Try out generation of the sampler2D interface
    src.tex = NULL;
    src.tex_w = 100;
    src.tex_h = 100;
    src.format = PL_FMT_UNORM;
    src.sampler = PL_SAMPLER_NORMAL;
    src.mode = PL_TEX_SAMPLE_LINEAR;

    pl_shader_reset(sh, &(struct pl_shader_params) { .gpu = gpu });
    REQUIRE(pl_shader_sample_polar(sh, &src, &filter_params));
    REQUIRE((res = pl_shader_finalize(sh)));
    REQUIRE(res->input == PL_SHADER_SIG_SAMPLER);
    printf("generated sampler2D shader:\n\n%s\n", res->glsl);

    pl_shader_free(&sh);
    pl_shader_obj_destroy(&lut);
    pl_tex_destroy(gpu, &dummy);
    pl_gpu_dummy_destroy(&gpu);
    pl_log_destroy(&log);
}
