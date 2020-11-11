#include "tests.h"
#include "libplacebo/utils/libav.h"

int main()
{
    struct pl_plane_data data[4] = {0};
    struct pl_bit_encoding bits;

#define TEST(pixfmt, reference)                                         \
    do {                                                                \
        int planes = pl_plane_data_from_pixfmt(data, &bits, pixfmt);    \
        REQUIRE(planes == sizeof(reference) / sizeof(*reference));      \
        REQUIRE(memcmp(data, reference, sizeof(reference)) == 0);       \
    } while (0)

    // Planar and semiplanar formats
    static const struct pl_plane_data yuvp8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {0},
            .pixel_stride = 1,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {1},
            .pixel_stride = 1,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {2},
            .pixel_stride = 1,
        }
    };

    TEST(AV_PIX_FMT_YUV420P, yuvp8);
    TEST(AV_PIX_FMT_YUV422P, yuvp8);
    TEST(AV_PIX_FMT_YUV444P, yuvp8);
    TEST(AV_PIX_FMT_YUV410P, yuvp8);
    TEST(AV_PIX_FMT_YUV411P, yuvp8);
    TEST(AV_PIX_FMT_YUV440P, yuvp8);

    static const struct pl_plane_data yuvap8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {0},
            .pixel_stride = 1,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {1},
            .pixel_stride = 1,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {2},
            .pixel_stride = 1,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {3},
            .pixel_stride = 1,
        }
    };

    TEST(AV_PIX_FMT_YUVA420P, yuvap8);

    static const struct pl_plane_data yuvp16[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {16},
            .component_map = {0},
            .pixel_stride = 2,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {16},
            .component_map = {1},
            .pixel_stride = 2,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {16},
            .component_map = {2},
            .pixel_stride = 2,
        }
    };

    TEST(AV_PIX_FMT_YUV420P10, yuvp16);
    TEST(AV_PIX_FMT_YUV420P16, yuvp16);

    static const struct pl_plane_data nv12[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {0},
            .pixel_stride = 1,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8},
            .component_map = {1, 2},
            .pixel_stride = 2,
        }
    };

    TEST(AV_PIX_FMT_NV12, nv12);

    static const struct pl_plane_data nv21[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {0},
            .pixel_stride = 1,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8},
            .component_map = {2, 1},
            .pixel_stride = 2,
        }
    };

    TEST(AV_PIX_FMT_NV21, nv21);

    static const struct pl_plane_data p016[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {16},
            .component_map = {0},
            .pixel_stride = 2,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {16, 16},
            .component_map = {1, 2},
            .pixel_stride = 4,
        }
    };

    TEST(AV_PIX_FMT_P010, p016);
    TEST(AV_PIX_FMT_P016, p016);

    // Packed formats
    static const struct pl_plane_data r8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {0},
            .pixel_stride = 1,
        }
    };

    TEST(AV_PIX_FMT_GRAY8, r8);

    static const struct pl_plane_data rg8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8},
            .component_map = {0, 1},
            .pixel_stride = 2,
        }
    };

    TEST(AV_PIX_FMT_GRAY8A, rg8);

    static const struct pl_plane_data rgb8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8, 8},
            .component_map = {0, 1, 2},
            .pixel_stride = 3,
        }
    };

    TEST(AV_PIX_FMT_RGB24, rgb8);

    static const struct pl_plane_data bgr8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8, 8},
            .component_map = {2, 1, 0},
            .pixel_stride = 3,
        }
    };

    TEST(AV_PIX_FMT_BGR24, bgr8);

    static const struct pl_plane_data rgbx8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8, 8},
            .component_map = {0, 1, 2},
            .pixel_stride = 4,
        }
    };

    TEST(AV_PIX_FMT_RGB0, rgbx8);

    static const struct pl_plane_data xrgb8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8, 8},
            .component_map = {0, 1, 2},
            .component_pad = {8, 0, 0},
            .pixel_stride = 4,
        }
    };

    TEST(AV_PIX_FMT_0RGB, xrgb8);

    static const struct pl_plane_data rgba8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8, 8, 8},
            .component_map = {0, 1, 2, 3},
            .pixel_stride = 4,
        }
    };

    TEST(AV_PIX_FMT_RGBA, rgba8);

    static const struct pl_plane_data argb8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8, 8, 8},
            .component_map = {3, 0, 1, 2},
            .pixel_stride = 4,
        }
    };

    TEST(AV_PIX_FMT_ARGB, argb8);

    static const struct pl_plane_data bgra8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8, 8, 8},
            .component_map = {2, 1, 0, 3},
            .pixel_stride = 4,
        }
    };

    TEST(AV_PIX_FMT_BGRA, bgra8);

    static const struct pl_plane_data abgr8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8, 8, 8},
            .component_map = {3, 2, 1, 0},
            .pixel_stride = 4,
        }
    };

    TEST(AV_PIX_FMT_ABGR, abgr8);

    static const struct pl_plane_data r16[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {16},
            .component_map = {0},
            .pixel_stride = 2,
        }
    };

    TEST(AV_PIX_FMT_GRAY16, r16);

    static const struct pl_plane_data rgb16[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {16, 16, 16},
            .component_map = {0, 1, 2},
            .pixel_stride = 6,
        }
    };

    TEST(AV_PIX_FMT_RGB48, rgb16);

    static const struct pl_plane_data rgb565[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {5, 6, 5},
            .component_map = {2, 1, 0}, // LSB to MSB
            .pixel_stride = 2,
        }
    };

    TEST(AV_PIX_FMT_RGB565, rgb565);

    // Test pl_image <- AVFrame bridge
    struct pl_image image;
    struct pl_render_target target;
    AVFrame *frame = av_frame_alloc();
    frame->format = AV_PIX_FMT_RGBA;
    pl_image_from_avframe(&image, frame);
    pl_target_from_avframe(&target, frame);
    REQUIRE(image.num_planes == 1);
    REQUIRE(image.repr.sys == PL_COLOR_SYSTEM_RGB);
    REQUIRE(pl_color_space_equal(&image.color, &target.color));
    REQUIRE(pl_color_repr_equal(&image.repr, &target.repr));
    REQUIRE(pl_icc_profile_equal(&image.profile, &target.profile));

    // Test inverse mapping
    struct pl_color_space csp = image.color;
    pl_color_space_infer(&csp);
    pl_avframe_set_color(frame, csp);
    pl_avframe_set_repr(frame, image.repr);
    pl_avframe_set_profile(frame, image.profile);
    pl_image_from_avframe(&image, frame);
    pl_color_space_infer(&image.color);
    REQUIRE(pl_color_space_equal(&csp, &image.color));

    // Test enum functions
    for (enum pl_color_system sys = 0; sys < PL_COLOR_SYSTEM_COUNT; sys++) {
        enum AVColorSpace spc = pl_system_to_av(sys);
        enum pl_color_system sys2 = pl_system_from_av(spc);
        // Exception to the rule, due to different handling in libav*
        if (sys != PL_COLOR_SYSTEM_BT_2100_HLG)
            REQUIRE(!sys2 || sys2 == sys);
    }

    for (enum pl_color_levels lev = 0; lev < PL_COLOR_LEVELS_COUNT; lev++) {
        enum AVColorRange range = pl_levels_to_av(lev);
        enum pl_color_levels lev2 = pl_levels_from_av(range);
        REQUIRE(lev2 == lev);
    }

    for (enum pl_color_primaries prim = 0; prim < PL_COLOR_PRIM_COUNT; prim++) {
        enum AVColorPrimaries avpri = pl_primaries_to_av(prim);
        enum pl_color_primaries prim2 = pl_primaries_from_av(avpri);
        REQUIRE(!prim2 || prim2 == prim);
    }

    for (enum pl_color_transfer trc = 0; trc < PL_COLOR_TRC_COUNT; trc++) {
        enum AVColorTransferCharacteristic avtrc = pl_transfer_to_av(trc);
        enum pl_color_transfer trc2 = pl_transfer_from_av(avtrc);
        REQUIRE(!trc2 || trc2 == trc);
    }

    for (enum pl_chroma_location loc = 0; loc < PL_CHROMA_COUNT; loc++) {
        enum AVChromaLocation avloc = pl_chroma_to_av(loc);
        enum pl_chroma_location loc2 = pl_chroma_from_av(avloc);
        REQUIRE(loc2 == loc);
    }
}
