#include "tests.h"

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
}
