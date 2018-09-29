#include "tests.h"
#include "gpu.h"

#include <libplacebo/utils/upload.h>

int main()
{
    struct pl_plane_data data = {0};

    pl_plane_data_from_mask(&data, (uint64_t[4]){ 0xFF, 0xFF00, 0xFF0000 });
    for (int i = 0; i < 3; i++) {
        REQUIRE(data.component_size[i] == 8);
        REQUIRE(data.component_pad[i] == 0);
        REQUIRE(data.component_map[i] == i);
    }

    pl_plane_data_from_mask(&data, (uint64_t[4]){ 0xFFFF0000, 0xFFFF });
    for (int i = 0; i < 2; i++) {
        REQUIRE(data.component_size[i] == 16);
        REQUIRE(data.component_pad[i] == 0);
        REQUIRE(data.component_map[i] == 1 - i);
    }

    pl_plane_data_from_mask(&data, (uint64_t[4]){ 0x03FF, 0x03FF0000 });
    REQUIRE(data.component_pad[0] == 0);
    REQUIRE(data.component_pad[1] == 6);
    for (int i = 0; i < 2; i++) {
        REQUIRE(data.component_size[i] == 10);
        REQUIRE(data.component_map[i] == i);
    }

    // Test GLSL structure packing
    struct pl_var vec1 = pl_var_float(""),
                  vec2 = pl_var_vec2(""),
                  vec3 = pl_var_vec3(""),
                  mat2 = pl_var_mat2(""),
                  mat3 = pl_var_mat3("");

    struct pl_var_layout layout;
    layout = std140_layout(NULL, 0, &vec2);
    REQUIRE(layout.offset == 0);
    REQUIRE(layout.stride == 2 * sizeof(float));
    REQUIRE(layout.size == 2 * sizeof(float));

    layout = std140_layout(NULL, 3 * sizeof(float), &vec3);
    REQUIRE(layout.offset == 4 * sizeof(float));
    REQUIRE(layout.stride == 3 * sizeof(float));
    REQUIRE(layout.size == 3 * sizeof(float));

    layout = std140_layout(NULL, 2 * sizeof(float), &mat3);
    REQUIRE(layout.offset == 4 * sizeof(float));
    REQUIRE(layout.stride == 4 * sizeof(float));
    REQUIRE(layout.size == 3 * 4 * sizeof(float));

    layout = std430_layout(NULL, 2 * sizeof(float), &mat3);
    REQUIRE(layout.offset == 4 * sizeof(float));
    REQUIRE(layout.stride == 3 * sizeof(float));
    REQUIRE(layout.size == 3 * 3 * sizeof(float));

    layout = std140_layout(NULL, 3 * sizeof(float), &vec1);
    REQUIRE(layout.offset == 3 * sizeof(float));
    REQUIRE(layout.stride == sizeof(float));
    REQUIRE(layout.size == sizeof(float));

    struct pl_var vec2a = vec2;
    vec2a.dim_a = 50;

    layout = std140_layout(NULL, sizeof(float), &vec2a);
    REQUIRE(layout.offset == 4 * sizeof(float));
    REQUIRE(layout.stride == 4 * sizeof(float));
    REQUIRE(layout.size == 50 * 4 * sizeof(float));

    layout = std430_layout(NULL, sizeof(float), &vec2a);
    REQUIRE(layout.offset == 2 * sizeof(float));
    REQUIRE(layout.stride == 2 * sizeof(float));
    REQUIRE(layout.size == 50 * 2 * sizeof(float));

    struct pl_var mat2a = mat2;
    mat2a.dim_a = 20;

    layout = std140_layout(NULL, 5 * sizeof(float), &mat2a);
    REQUIRE(layout.offset == 8 * sizeof(float));
    REQUIRE(layout.stride == 4 * sizeof(float));
    REQUIRE(layout.size == 20 * 2 * 4 * sizeof(float));

    layout = std430_layout(NULL, 5 * sizeof(float), &mat2a);
    REQUIRE(layout.offset == 6 * sizeof(float));
    REQUIRE(layout.stride == 2 * sizeof(float));
    REQUIRE(layout.size == 20 * 2 * 2 * sizeof(float));
}
