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
}
