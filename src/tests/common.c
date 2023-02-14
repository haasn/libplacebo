#include "tests.h"

static int irand()
{
    return rand() - RAND_MAX / 2;
}

int main()
{
    pl_log log = pl_test_logger();
    pl_log_update(log, NULL);
    pl_log_destroy(&log);

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
    REQUIRE_CMP(rc2.x1, >=, rc2.x0, "d");
    REQUIRE_CMP(rc2.y1, >=, rc2.y0, "d");

    pl_rect3d_normalize(&rc3);
    REQUIRE_CMP(rc3.x1, >=, rc3.x0, "d");
    REQUIRE_CMP(rc3.y1, >=, rc3.y0, "d");
    REQUIRE_CMP(rc3.z1, >=, rc3.z0, "d");

    struct pl_rect2df rc2f = {
        RANDOM, RANDOM,
        RANDOM, RANDOM,
    };

    struct pl_rect3df rc3f = {
        RANDOM, RANDOM, RANDOM,
        RANDOM, RANDOM, RANDOM,
    };

    pl_rect2df_normalize(&rc2f);
    REQUIRE_CMP(rc2f.x1, >=, rc2f.x0, "f");
    REQUIRE_CMP(rc2f.y1, >=, rc2f.y0, "f");

    pl_rect3df_normalize(&rc3f);
    REQUIRE_CMP(rc3f.x1, >=, rc3f.x0, "f");
    REQUIRE_CMP(rc3f.y1, >=, rc3f.y0, "f");
    REQUIRE_CMP(rc3f.z1, >=, rc3f.z0, "f");

    struct pl_rect2d rc2r = pl_rect2df_round(&rc2f);
    struct pl_rect3d rc3r = pl_rect3df_round(&rc3f);

    REQUIRE_CMP(fabs(rc2r.x0 - rc2f.x0), <=, 0.5, "f");
    REQUIRE_CMP(fabs(rc2r.x1 - rc2f.x1), <=, 0.5, "f");
    REQUIRE_CMP(fabs(rc2r.y0 - rc2f.y0), <=, 0.5, "f");
    REQUIRE_CMP(fabs(rc2r.y1 - rc2f.y1), <=, 0.5, "f");

    REQUIRE_CMP(fabs(rc3r.x0 - rc3f.x0), <=, 0.5, "f");
    REQUIRE_CMP(fabs(rc3r.x1 - rc3f.x1), <=, 0.5, "f");
    REQUIRE_CMP(fabs(rc3r.y0 - rc3f.y0), <=, 0.5, "f");
    REQUIRE_CMP(fabs(rc3r.y1 - rc3f.y1), <=, 0.5, "f");
    REQUIRE_CMP(fabs(rc3r.z0 - rc3f.z0), <=, 0.5, "f");
    REQUIRE_CMP(fabs(rc3r.z1 - rc3f.z1), <=, 0.5, "f");

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
            REQUIRE_FEQ(tr.mat.m[i][j], tr2.mat.m[i][j], 1e-4);
        }
        REQUIRE_FEQ(tr.c[i], tr2.c[i], 1e-4);
    }

    // Test aspect ratio code
    const struct pl_rect2df rc1080p = {0, 0, 1920, 1080};
    const struct pl_rect2df rc43 = {0, 0, 1024, 768};
    struct pl_rect2df rc;

    REQUIRE_FEQ(pl_rect2df_aspect(&rc1080p), 16.0/9.0, 1e-8);
    REQUIRE_FEQ(pl_rect2df_aspect(&rc43), 4.0/3.0, 1e-8);

#define pl_rect2df_midx(rc) (((rc).x0 + (rc).x1) / 2.0)
#define pl_rect2df_midy(rc) (((rc).y0 + (rc).y1) / 2.0)

    for (float aspect = 0.2; aspect < 3.0; aspect += 0.4) {
        for (float scan = 0.0; scan <= 1.0; scan += 0.5) {
            rc = rc1080p;
            pl_rect2df_aspect_set(&rc, aspect, scan);
            printf("aspect %.2f, panscan %.1f: {%f %f} -> {%f %f}\n",
                   aspect, scan, rc.x0, rc.y0, rc.x1, rc.y1);
            REQUIRE_FEQ(pl_rect2df_aspect(&rc), aspect, 1e-6);
            REQUIRE_FEQ(pl_rect2df_midx(rc), pl_rect2df_midx(rc1080p), 1e-6);
            REQUIRE_FEQ(pl_rect2df_midy(rc), pl_rect2df_midy(rc1080p), 1e-6);
        }
    }

    rc = rc1080p;
    pl_rect2df_aspect_fit(&rc, &rc43, 0.0);
    REQUIRE_FEQ(pl_rect2df_aspect(&rc), pl_rect2df_aspect(&rc43), 1e-6);
    REQUIRE_FEQ(pl_rect2df_midx(rc), pl_rect2df_midx(rc1080p), 1e-6);
    REQUIRE_FEQ(pl_rect2df_midy(rc), pl_rect2df_midy(rc1080p), 1e-6);
    REQUIRE_FEQ(pl_rect_w(rc), pl_rect_w(rc43), 1e-6);
    REQUIRE_FEQ(pl_rect_h(rc), pl_rect_h(rc43), 1e-6);

    rc = rc43;
    pl_rect2df_aspect_fit(&rc, &rc1080p, 0.0);
    REQUIRE_FEQ(pl_rect2df_aspect(&rc), pl_rect2df_aspect(&rc1080p), 1e-6);
    REQUIRE_FEQ(pl_rect2df_midx(rc), pl_rect2df_midx(rc43), 1e-6);
    REQUIRE_FEQ(pl_rect2df_midy(rc), pl_rect2df_midy(rc43), 1e-6);
    REQUIRE_FEQ(pl_rect_w(rc), pl_rect_w(rc43), 1e-6);

    rc = (struct pl_rect2df) { 1920, 1080, 0, 0 };
    pl_rect2df_offset(&rc, 50, 100);
    REQUIRE_FEQ(rc.x0, 1870, 1e-6);
    REQUIRE_FEQ(rc.x1, -50, 1e-6);
    REQUIRE_FEQ(rc.y0, 980, 1e-6);
    REQUIRE_FEQ(rc.y1, -100, 1e-6);
}
