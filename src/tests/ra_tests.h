#include "tests.h"

static void ra_test_roundtrip(const struct ra *ra, const struct ra_tex *tex,
                              float *src, float *dst)
{
    REQUIRE(tex);

    int texels = tex->params.w;
    texels *= tex->params.h ? tex->params.h : 1;
    texels *= tex->params.d ? tex->params.d : 1;

    for (int i = 0; i < texels; i++)
        src[i] = RANDOM;

    REQUIRE(ra_tex_upload(ra, &(struct ra_tex_transfer_params){
        .tex = tex,
        .ptr = src,
    }));

    REQUIRE(ra_tex_download(ra, &(struct ra_tex_transfer_params){
        .tex = tex,
        .ptr = dst,
    }));

    for (int i = 0; i < texels; i++)
        REQUIRE(src[i] == dst[i]);
}

static void ra_texture_tests(const struct ra *ra)
{
    const struct ra_fmt *fmt;
    fmt = ra_find_fmt(ra, RA_FMT_FLOAT, 1, 32, true, 0);
    if (!fmt)
        return;

    struct ra_tex_params params = {
        .format        = fmt,
        .host_writable = true,
        .host_readable = true,
    };

    static float src[16*16*16] = {0};
    static float dst[16*16*16] = {0};

    const struct ra_tex *tex = NULL;
    if (ra->limits.max_tex_1d_dim >= 16) {
        params.w = 16;
        tex = ra_tex_create(ra, &params);
        ra_test_roundtrip(ra, tex, src, dst);
        ra_tex_destroy(ra, &tex);
    }

    if (ra->limits.max_tex_2d_dim >= 16) {
        params.w = params.h = 16;
        tex = ra_tex_create(ra, &params);
        ra_test_roundtrip(ra, tex, src, dst);
        ra_tex_destroy(ra, &tex);
    }

    if (ra->limits.max_tex_3d_dim >= 16) {
        params.w = params.h = params.d = 16;
        tex = ra_tex_create(ra, &params);
        ra_test_roundtrip(ra, tex, src, dst);
        ra_tex_destroy(ra, &tex);
    }
}
