#include "utils.h"

#include <libplacebo/dummy.h>
#include <libplacebo/shaders/lut.h>

static const char *luts[] = {

    "TITLE \"1D LUT example\"   \n"
    "LUT_1D_SIZE 11             \n"
    "# Random comment           \n"
    "0.0 0.0 0.0                \n"
    "0.1 0.1 0.1                \n"
    "0.2 0.2 0.2                \n"
    "0.3 0.3 0.3                \n"
    "0.4 0.4 0.4                \n"
    "0.5 0.5 0.5                \n"
    "0.6 0.6 0.6                \n"
    "0.7 0.7 0.7                \n"
    "0.8 0.8 0.8                \n"
    "0.9 0.9 0.9                \n"
    "0.10 0.10 0.10             \n",

    "LUT_3D_SIZE 3              \n"
    "TITLE \"3D LUT example\"   \n"
    "0.0 0.0 0.0                \n"
    "0.5 0.0 0.0                \n"
    "1.0 0.0 0.0                \n"
    "0.0 0.5 0.0                \n"
    "0.5 0.5 0.0                \n"
    "1.0 0.5 0.0                \n"
    "0.0 1.0 0.0                \n"
    "0.5 1.0 0.0                \n"
    "1.0 1.0 0.0                \n"
    "0.0 0.0 0.5                \n"
    "0.5 0.0 0.5                \n"
    "1.0 0.0 0.5                \n"
    "0.0 0.5 0.5                \n"
    "0.5 0.5 0.5                \n"
    "1.0 0.5 0.5                \n"
    "0.0 1.0 0.5                \n"
    "0.5 1.0 0.5                \n"
    "1.0 1.0 0.5                \n"
    "0.0 0.0 1.0                \n"
    "0.5 0.0 1.0                \n"
    "1.0 0.0 1.0                \n"
    "0.0 0.5 1.0                \n"
    "0.5 0.5 1.0                \n"
    "1.0 0.5 1.0                \n"
    "0.0 1.0 1.0                \n"
    "0.5 1.0 1.0                \n"
    "1.0 1.0 1.0                \n",

    "LUT_1D_SIZE 3              \n"
    "TITLE \"custom domain\"    \n"
    "DOMAIN_MAX 255 255 255     \n"
    "0 0 0                      \n"
    "128 128 128                \n"
    "255 255 255                \n"

};

int main()
{
    pl_log log = pl_test_logger();
    pl_gpu gpu = pl_gpu_dummy_create(log, NULL);
    pl_shader sh = pl_shader_alloc(log, NULL);
    pl_shader_obj obj = NULL;

    for (int i = 0; i < PL_ARRAY_SIZE(luts); i++) {
        struct pl_custom_lut *lut;
        lut = pl_lut_parse_cube(log, luts[i], strlen(luts[i]));
        REQUIRE(lut);

        pl_shader_reset(sh, pl_shader_params( .gpu = gpu ));
        pl_shader_custom_lut(sh, lut, &obj);
        const struct pl_shader_res *res = pl_shader_finalize(sh);
        REQUIRE(res);
        printf("Generated LUT shader:\n%s\n", res->glsl);
        pl_lut_free(&lut);
    }

    pl_shader_obj_destroy(&obj);
    pl_shader_free(&sh);
    pl_gpu_dummy_destroy(&gpu);
    pl_log_destroy(&log);
}
