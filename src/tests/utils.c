#include "tests.h"
#include "gpu.h"

int main()
{
    struct pl_plane_data data = {0};

#define TEST(ref, ...)                                                  \
    do {                                                                \
        pl_plane_data_from_mask(&data, (uint64_t[4]){ __VA_ARGS__ });   \
        REQUIRE(memcmp(&data, &ref, sizeof(ref)) == 0);                 \
    } while (0)

    static const struct pl_plane_data rgb8 = {
        .component_size = {8, 8, 8},
        .component_map  = {0, 1, 2},
    };

    TEST(rgb8, 0xFF, 0xFF00, 0xFF0000);

    static const struct pl_plane_data bgra8 = {
        .component_size = {8, 8, 8, 8},
        .component_map  = {2, 1, 0, 3},
    };

    TEST(bgra8, 0xFF0000, 0xFF00, 0xFF, 0xFF000000);

    static const struct pl_plane_data gr16 = {
        .component_size = {16, 16},
        .component_map  = {1, 0},
    };

    TEST(gr16, 0xFFFF0000, 0xFFFF);

    static const struct pl_plane_data r10x6g10 = {
        .component_size = {10, 10},
        .component_map  = {1, 0}, // LSB -> MSB ordering
        .component_pad  = {0, 6},
    };

    TEST(r10x6g10, 0x03FF0000, 0x03FF);

    static const struct pl_plane_data rgb565 = {
        .component_size = {5, 6, 5},
        .component_map  = {2, 1, 0}, // LSB -> MSB ordering
    };

    TEST(rgb565, 0xF800, 0x07E0, 0x001F);

    static const struct pl_plane_data rgba16 = {
        .component_size = {16, 16, 16, 16},
        .component_map  = {0, 1, 2, 3},
    };

    TEST(rgba16, 0xFFFFllu, 0xFFFF0000llu, 0xFFFF00000000llu, 0xFFFF000000000000llu);

    // Test GLSL structure packing
    struct pl_var vec1 = pl_var_float(""),
                  vec2 = pl_var_vec2(""),
                  vec3 = pl_var_vec3(""),
                  mat2 = pl_var_mat2(""),
                  mat3 = pl_var_mat3("");

    struct pl_var_layout layout;
    layout = pl_std140_layout(0, &vec2);
    REQUIRE(layout.offset == 0);
    REQUIRE(layout.stride == 2 * sizeof(float));
    REQUIRE(layout.size == 2 * sizeof(float));

    layout = pl_std140_layout(3 * sizeof(float), &vec3);
    REQUIRE(layout.offset == 4 * sizeof(float));
    REQUIRE(layout.stride == 3 * sizeof(float));
    REQUIRE(layout.size == 3 * sizeof(float));

    layout = pl_std140_layout(2 * sizeof(float), &mat3);
    REQUIRE(layout.offset == 4 * sizeof(float));
    REQUIRE(layout.stride == 4 * sizeof(float));
    REQUIRE(layout.size == 3 * 4 * sizeof(float));

    layout = pl_std430_layout(2 * sizeof(float), &mat3);
    REQUIRE(layout.offset == 4 * sizeof(float));
    REQUIRE(layout.stride == 4 * sizeof(float));
    REQUIRE(layout.size == 4 * 3 * sizeof(float));

    layout = pl_std140_layout(3 * sizeof(float), &vec1);
    REQUIRE(layout.offset == 3 * sizeof(float));
    REQUIRE(layout.stride == sizeof(float));
    REQUIRE(layout.size == sizeof(float));

    struct pl_var vec2a = vec2;
    vec2a.dim_a = 50;

    layout = pl_std140_layout(sizeof(float), &vec2a);
    REQUIRE(layout.offset == 4 * sizeof(float));
    REQUIRE(layout.stride == 4 * sizeof(float));
    REQUIRE(layout.size == 50 * 4 * sizeof(float));

    layout = pl_std430_layout(sizeof(float), &vec2a);
    REQUIRE(layout.offset == 2 * sizeof(float));
    REQUIRE(layout.stride == 2 * sizeof(float));
    REQUIRE(layout.size == 50 * 2 * sizeof(float));

    struct pl_var mat2a = mat2;
    mat2a.dim_a = 20;

    layout = pl_std140_layout(5 * sizeof(float), &mat2a);
    REQUIRE(layout.offset == 8 * sizeof(float));
    REQUIRE(layout.stride == 4 * sizeof(float));
    REQUIRE(layout.size == 20 * 2 * 4 * sizeof(float));

    layout = pl_std430_layout(5 * sizeof(float), &mat2a);
    REQUIRE(layout.offset == 6 * sizeof(float));
    REQUIRE(layout.stride == 2 * sizeof(float));
    REQUIRE(layout.size == 20 * 2 * 2 * sizeof(float));

    for (const struct pl_named_var *nvar = pl_var_glsl_types; nvar->glsl_name; nvar++) {
        struct pl_var var = nvar->var;
        REQUIRE(nvar->glsl_name == pl_var_glsl_type_name(var));
        var.dim_a = 100;
        REQUIRE(nvar->glsl_name == pl_var_glsl_type_name(var));
    }
}
