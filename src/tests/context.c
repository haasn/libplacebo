#include "tests.h"

static int irand()
{
    return rand() - RAND_MAX / 2;
}

int main()
{
    struct pl_context *ctx = pl_test_context();
    pl_context_destroy(&ctx);

    // Test some misc helper functions
    struct pl_rect2d rc2 = {
        irand(), irand(),
        irand(), irand(),
    };

    struct pl_rect3d rc3 = {
        irand(), irand(), irand(),
        irand(), irand(), irand(),
    };

    pl_rect2d_normalize(&rc2);
    REQUIRE(rc2.x1 >= rc2.x0);
    REQUIRE(rc2.y1 >= rc2.y0);

    pl_rect3d_normalize(&rc3);
    REQUIRE(rc3.x1 >= rc3.x0);
    REQUIRE(rc3.y1 >= rc3.y0);
    REQUIRE(rc3.z1 >= rc3.z0);

    struct pl_transform3x3 tr = {
        .mat = {{
            { RANDOM, RANDOM, RANDOM },
            { RANDOM, RANDOM, RANDOM },
            { RANDOM, RANDOM, RANDOM },
        }},
        .c = { RANDOM, RANDOM, RANDOM },
    };

    struct pl_transform3x3 tr2 = tr;
    float scale = 1.0 + RANDOM;
    pl_transform3x3_scale(&tr2, scale);
    pl_transform3x3_invert(&tr2);
    pl_transform3x3_invert(&tr2);
    pl_transform3x3_scale(&tr2, 1.0 / scale);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f %f\n", tr.mat.m[i][j], tr2.mat.m[i][j]);
            REQUIRE(fabs(tr.mat.m[i][j] - tr2.mat.m[i][j]) < 1e-4);
        }
        REQUIRE(fabs(tr.c[i] - tr2.c[i]) < 1e-4);
    }
}
