#include "tests.h"

#define SHIFT 4
#define SIZE (1 << SHIFT)
float data[SIZE][SIZE];

int main()
{
    printf("Ordered dither matrix:\n");
    pl_generate_bayer_matrix(&data[0][0], SIZE);
    for (int y = 0; y < SIZE; y++) {
        for (int x = 0; x < SIZE; x++)
            printf(" %3d", (int)(data[y][x] * SIZE * SIZE));
        printf("\n");
    }

    printf("Blue noise dither matrix:\n");
    pl_generate_blue_noise(&data[0][0], SHIFT);
    for (int y = 0; y < SIZE; y++) {
        for (int x = 0; x < SIZE; x++)
            printf(" %3d", (int)(data[y][x] * SIZE * SIZE));
        printf("\n");
    }

    // Generate an example of a dither shader
    pl_log log = pl_test_logger();
    pl_shader sh = pl_shader_alloc(log, NULL);
    pl_shader_obj obj = NULL;

    pl_shader_dither(sh, 8, &obj, NULL);
    const struct pl_shader_res *res = pl_shader_finalize(sh);
    REQUIRE(res);
    printf("Generated dither shader:\n%s\n", res->glsl);

    pl_shader_obj_destroy(&obj);
    pl_shader_free(&sh);
    pl_log_destroy(&log);
}
