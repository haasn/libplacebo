#include "tests.h"
#include "gpu.h"

#include <libplacebo/utils/upload.h>

int main()
{
    struct pl_bit_encoding bits = {0};
    struct pl_plane_data data = {0};

    static const struct pl_bit_encoding bits0 = {0};
    static const struct pl_bit_encoding bits8 = {
        .sample_depth = 8,
        .color_depth = 8,
    };

    static const struct pl_bit_encoding bits16 = {
        .sample_depth = 16,
        .color_depth = 16,
    };

    static const struct pl_bit_encoding bits10_16 = {
        .sample_depth = 16,
        .color_depth = 10,
    };

    static const struct pl_bit_encoding bits10_16_6 = {
        .sample_depth = 16,
        .color_depth = 10,
        .bit_shift = 6,
    };

#define TEST_ALIGN(ref, ref_align, ref_bits, ...)                       \
    do {                                                                \
        pl_plane_data_from_mask(&data, (uint64_t[4]){ __VA_ARGS__ });   \
        REQUIRE_MEMEQ(&data, &ref, sizeof(ref));                        \
        pl_plane_data_align(&data, &bits);                              \
        REQUIRE_MEMEQ(&data, &ref_align, sizeof(ref_align));            \
        REQUIRE_MEMEQ(&bits, &ref_bits, sizeof(bits));                  \
    } while (0)

#define TEST(ref, bits, ...) TEST_ALIGN(ref, ref, bits, __VA_ARGS__)

    static const struct pl_plane_data rgb8 = {
        .component_size = {8, 8, 8},
        .component_map  = {0, 1, 2},
    };

    TEST(rgb8, bits8, 0xFF, 0xFF00, 0xFF0000);

    static const struct pl_plane_data bgra8 = {
        .component_size = {8, 8, 8, 8},
        .component_map  = {2, 1, 0, 3},
    };

    TEST(bgra8, bits8, 0xFF0000, 0xFF00, 0xFF, 0xFF000000);

    static const struct pl_plane_data gr16 = {
        .component_size = {16, 16},
        .component_map  = {1, 0},
    };

    TEST(gr16, bits16, 0xFFFF0000, 0xFFFF);

    static const struct pl_plane_data r10x6g10 = {
        .component_size = {10, 10},
        .component_map  = {1, 0}, // LSB -> MSB ordering
        .component_pad  = {0, 6},
    };

    TEST_ALIGN(r10x6g10, gr16, bits10_16, 0x03FF0000, 0x03FF);

    static const struct pl_plane_data rgb565 = {
        .component_size = {5, 6, 5},
        .component_map  = {2, 1, 0}, // LSB -> MSB ordering
    };

    TEST(rgb565, bits0, 0xF800, 0x07E0, 0x001F);

    static const struct pl_plane_data rgba16 = {
        .component_size = {16, 16, 16, 16},
        .component_map  = {0, 1, 2, 3},
    };

    TEST(rgba16, bits16, 0xFFFFllu, 0xFFFF0000llu, 0xFFFF00000000llu, 0xFFFF000000000000llu);

    static const struct pl_plane_data p010 = {
        .component_size = {10, 10, 10},
        .component_map  = {0, 1, 2},
        .component_pad  = {6, 6, 6},
    };

    static const struct pl_plane_data rgb16 = {
        .component_size = {16, 16, 16},
        .component_map  = {0, 1, 2},
    };

    TEST_ALIGN(p010, rgb16, bits10_16_6, 0xFFC0llu, 0xFFC00000llu, 0xFFC000000000llu);

    // Test GLSL structure packing
    struct pl_var vec1 = pl_var_float(""),
                  vec2 = pl_var_vec2(""),
                  vec3 = pl_var_vec3(""),
                  mat2 = pl_var_mat2(""),
                  mat3 = pl_var_mat3("");

    struct pl_var_layout layout;
    layout = pl_std140_layout(0, &vec2);
    REQUIRE_CMP(layout.offset, ==, 0 * sizeof(float), "zu");
    REQUIRE_CMP(layout.stride, ==, 2 * sizeof(float), "zu");
    REQUIRE_CMP(layout.size, ==, 2 * sizeof(float), "zu");

    layout = pl_std140_layout(3 * sizeof(float), &vec3);
    REQUIRE_CMP(layout.offset, ==, 4 * sizeof(float), "zu");
    REQUIRE_CMP(layout.stride, ==, 3 * sizeof(float), "zu");
    REQUIRE_CMP(layout.size, ==, 3 * sizeof(float), "zu");

    layout = pl_std140_layout(2 * sizeof(float), &mat3);
    REQUIRE_CMP(layout.offset, ==, 4 * sizeof(float), "zu");
    REQUIRE_CMP(layout.stride, ==, 4 * sizeof(float), "zu");
    REQUIRE_CMP(layout.size, ==, 3 * 4 * sizeof(float), "zu");

    layout = pl_std430_layout(2 * sizeof(float), &mat3);
    REQUIRE_CMP(layout.offset, ==, 4 * sizeof(float), "zu");
    REQUIRE_CMP(layout.stride, ==, 4 * sizeof(float), "zu");
    REQUIRE_CMP(layout.size, ==, 4 * 3 * sizeof(float), "zu");

    layout = pl_std140_layout(3 * sizeof(float), &vec1);
    REQUIRE_CMP(layout.offset, ==, 3 * sizeof(float), "zu");
    REQUIRE_CMP(layout.stride, ==, sizeof(float), "zu");
    REQUIRE_CMP(layout.size, ==, sizeof(float), "zu");

    struct pl_var vec2a = vec2;
    vec2a.dim_a = 50;

    layout = pl_std140_layout(sizeof(float), &vec2a);
    REQUIRE_CMP(layout.offset, ==, 4 * sizeof(float), "zu");
    REQUIRE_CMP(layout.stride, ==, 4 * sizeof(float), "zu");
    REQUIRE_CMP(layout.size, ==, 50 * 4 * sizeof(float), "zu");

    layout = pl_std430_layout(sizeof(float), &vec2a);
    REQUIRE_CMP(layout.offset, ==, 2 * sizeof(float), "zu");
    REQUIRE_CMP(layout.stride, ==, 2 * sizeof(float), "zu");
    REQUIRE_CMP(layout.size, ==, 50 * 2 * sizeof(float), "zu");

    struct pl_var mat2a = mat2;
    mat2a.dim_a = 20;

    layout = pl_std140_layout(5 * sizeof(float), &mat2a);
    REQUIRE_CMP(layout.offset, ==, 8 * sizeof(float), "zu");
    REQUIRE_CMP(layout.stride, ==, 4 * sizeof(float), "zu");
    REQUIRE_CMP(layout.size, ==, 20 * 2 * 4 * sizeof(float), "zu");

    layout = pl_std430_layout(5 * sizeof(float), &mat2a);
    REQUIRE_CMP(layout.offset, ==, 6 * sizeof(float), "zu");
    REQUIRE_CMP(layout.stride, ==, 2 * sizeof(float), "zu");
    REQUIRE_CMP(layout.size, ==, 20 * 2 * 2 * sizeof(float), "zu");

    for (const struct pl_named_var *nvar = pl_var_glsl_types; nvar->glsl_name; nvar++) {
        struct pl_var var = nvar->var;
        REQUIRE_CMP(nvar->glsl_name, ==, pl_var_glsl_type_name(var), "s");
        var.dim_a = 100;
        REQUIRE_CMP(nvar->glsl_name, ==, pl_var_glsl_type_name(var), "s");
    }
}
