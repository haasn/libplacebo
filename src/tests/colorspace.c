#include "tests.h"

int main()
{
    for (enum pl_color_system sys = 0; sys < PL_COLOR_SYSTEM_COUNT; sys++) {
        bool ycbcr = sys >= PL_COLOR_SYSTEM_BT_601 && sys <= PL_COLOR_SYSTEM_YCGCO;
        REQUIRE(ycbcr == pl_color_system_is_ycbcr_like(sys));
    }

    for (enum pl_color_transfer trc = 0; trc < PL_COLOR_TRC_COUNT; trc++) {
        bool hdr = trc >= PL_COLOR_TRC_PQ && trc <= PL_COLOR_TRC_S_LOG2;
        REQUIRE(hdr == pl_color_transfer_is_hdr(trc));
        REQUIRE(pl_color_transfer_nominal_peak(trc) >= 1.0);
    }

    float pq_peak = pl_color_transfer_nominal_peak(PL_COLOR_TRC_PQ);
    REQUIRE(feq(PL_COLOR_REF_WHITE * pq_peak, 10000));

    struct pl_color_repr tv_repr = {
        .sys       = PL_COLOR_SYSTEM_BT_709,
        .levels    = PL_COLOR_LEVELS_TV,
    };

    struct pl_color_repr pc_repr = {
        .sys       = PL_COLOR_SYSTEM_RGB,
        .levels    = PL_COLOR_LEVELS_PC,
    };

    // Ensure this is a no-op for bits == bits
    for (int bits = 1; bits <= 16; bits++) {
        tv_repr.bits.color_depth = tv_repr.bits.sample_depth = bits;
        pc_repr.bits.color_depth = pc_repr.bits.sample_depth = bits;
        REQUIRE(feq(pl_color_repr_normalize(&tv_repr), 1.0));
        REQUIRE(feq(pl_color_repr_normalize(&pc_repr), 1.0));
    }

    tv_repr.bits.color_depth  = 8;
    tv_repr.bits.sample_depth = 10;
    float tv8to10 = pl_color_repr_normalize(&tv_repr);

    tv_repr.bits.color_depth  = 8;
    tv_repr.bits.sample_depth = 12;
    float tv8to12 = pl_color_repr_normalize(&tv_repr);

    // Simulate the effect of GPU texture sampling on UNORM texture
    REQUIRE(feq(tv8to10 * 16 /1023.,  64/1023.)); // black
    REQUIRE(feq(tv8to10 * 235/1023., 940/1023.)); // nominal white
    REQUIRE(feq(tv8to10 * 128/1023., 512/1023.)); // achromatic
    REQUIRE(feq(tv8to10 * 240/1023., 960/1023.)); // nominal chroma peak

    REQUIRE(feq(tv8to12 * 16 /4095., 256 /4095.)); // black
    REQUIRE(feq(tv8to12 * 235/4095., 3760/4095.)); // nominal white
    REQUIRE(feq(tv8to12 * 128/4095., 2048/4095.)); // achromatic
    REQUIRE(feq(tv8to12 * 240/4095., 3840/4095.)); // nominal chroma peak

    // Ensure lavc's xyz12 is handled correctly
    struct pl_color_repr xyz12 = {
        .sys    = PL_COLOR_SYSTEM_XYZ,
        .levels = PL_COLOR_LEVELS_UNKNOWN,
        .bits   = {
            .sample_depth = 16,
            .color_depth  = 12,
            .bit_shift    = 4,
        },
    };

    float xyz = pl_color_repr_normalize(&xyz12);
    REQUIRE(feq(xyz * (4095 << 4), 65535));

    // Assume we uploaded a 10-bit source directly (unshifted) as a 16-bit
    // texture. This texture multiplication factor should make it behave as if
    // it was uploaded as a 10-bit texture instead.
    pc_repr.bits.color_depth = 10;
    pc_repr.bits.sample_depth = 16;
    float pc10to16 = pl_color_repr_normalize(&pc_repr);
    REQUIRE(feq(pc10to16 * 1000/65535., 1000/1023.));

    const struct pl_raw_primaries *bt709, *bt2020;
    bt709 = pl_raw_primaries_get(PL_COLOR_PRIM_BT_709);
    bt2020 = pl_raw_primaries_get(PL_COLOR_PRIM_BT_2020);

    struct pl_matrix3x3 rgb2xyz, rgb2xyz_;
    rgb2xyz = rgb2xyz_ = pl_get_rgb2xyz_matrix(bt709);
    pl_matrix3x3_invert(&rgb2xyz_);
    pl_matrix3x3_invert(&rgb2xyz_);

    // Make sure the double-inversion round trips
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++)
            REQUIRE(feq(rgb2xyz.m[y][x], rgb2xyz_.m[y][x]));
    }

    // Make sure mapping the spectral RGB colors (i.e. the matrix rows) matches
    // our original primaries
    float Y = rgb2xyz.m[1][0];
    REQUIRE(feq(rgb2xyz.m[0][0], pl_cie_X(bt709->red) * Y));
    REQUIRE(feq(rgb2xyz.m[2][0], pl_cie_Z(bt709->red) * Y));
    Y = rgb2xyz.m[1][1];
    REQUIRE(feq(rgb2xyz.m[0][1], pl_cie_X(bt709->green) * Y));
    REQUIRE(feq(rgb2xyz.m[2][1], pl_cie_Z(bt709->green) * Y));
    Y = rgb2xyz.m[1][2];
    REQUIRE(feq(rgb2xyz.m[0][2], pl_cie_X(bt709->blue) * Y));
    REQUIRE(feq(rgb2xyz.m[2][2], pl_cie_Z(bt709->blue) * Y));

    // Make sure the gamut mapping round-trips
    struct pl_matrix3x3 bt709_bt2020, bt2020_bt709;
    bt709_bt2020 = pl_get_color_mapping_matrix(bt709, bt2020, PL_INTENT_RELATIVE_COLORIMETRIC);
    bt2020_bt709 = pl_get_color_mapping_matrix(bt2020, bt709, PL_INTENT_RELATIVE_COLORIMETRIC);
    for (int n = 0; n < 10; n++) {
        float vec[3] = { RANDOM, RANDOM, RANDOM };
        float dst[3] = { vec[0],    vec[1],    vec[2]    };
        pl_matrix3x3_apply(&bt709_bt2020, dst);
        pl_matrix3x3_apply(&bt2020_bt709, dst);
        for (int i = 0; i < 3; i++)
            REQUIRE(feq(dst[i], vec[i]));
    }

    // Ensure the decoding matrix round-trips to white/black
    for (enum pl_color_system sys = 0; sys < PL_COLOR_SYSTEM_COUNT; sys++) {
        if (!pl_color_system_is_linear(sys))
            continue;

        printf("testing color system %u\n", (unsigned) sys);
        struct pl_color_repr repr = {
            .levels = PL_COLOR_LEVELS_TV,
            .sys = sys,
        };

        struct pl_transform3x3 yuv2rgb = pl_color_repr_decode(&repr, NULL);
        static const float white_ycbcr[3] = { 235/255., 128/255., 128/255. };
        static const float black_ycbcr[3] = {  16/255., 128/255., 128/255. };
        static const float white_other[3] = { 235/255., 235/255., 235/255. };
        static const float black_other[3] = {  16/255.,  16/255.,  16/255. };

        float white[3], black[3];
        for (int i = 0; i < 3; i++) {
            if (pl_color_system_is_ycbcr_like(sys)) {
                white[i] = white_ycbcr[i];
                black[i] = black_ycbcr[i];
            } else {
                white[i] = white_other[i];
                black[i] = black_other[i];
            }
        }

        pl_transform3x3_apply(&yuv2rgb, white);
        REQUIRE(feq(white[0], 1.0));
        REQUIRE(feq(white[1], 1.0));
        REQUIRE(feq(white[2], 1.0));

        pl_transform3x3_apply(&yuv2rgb, black);
        REQUIRE(feq(black[0], 0.0));
        REQUIRE(feq(black[1], 0.0));
        REQUIRE(feq(black[2], 0.0));
    }

    // Simulate a typical 10-bit YCbCr -> 16 bit texture conversion
    tv_repr.bits.color_depth  = 10;
    tv_repr.bits.sample_depth = 16;
    struct pl_transform3x3 yuv2rgb;
    yuv2rgb = pl_color_repr_decode(&tv_repr, NULL);
    float test[3] = { 575/65535., 336/65535., 640/65535. };
    pl_transform3x3_apply(&yuv2rgb, test);
    REQUIRE(feq(test[0], 0.808305));
    REQUIRE(feq(test[1], 0.553254));
    REQUIRE(feq(test[2], 0.218841));
}
