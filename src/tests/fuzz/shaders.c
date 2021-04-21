#include "../tests.h"
#include "shaders.h"

__AFL_FUZZ_INIT();

#pragma clang optimize off

int main()
{
    const struct pl_gpu *gpu = pl_gpu_dummy_create(NULL, NULL);

#define WIDTH 64
#define HEIGHT 64
#define COMPS 4

    static const float empty[HEIGHT][WIDTH][COMPS] = {0};

    struct pl_sample_src src = {
        .tex = pl_tex_create(gpu, &(struct pl_tex_params) {
            .format = pl_find_fmt(gpu, PL_FMT_FLOAT, COMPS, 0, 32, PL_FMT_CAP_SAMPLEABLE),
            .initial_data = empty,
            .sampleable = true,
            .w = WIDTH,
            .h = HEIGHT,
        }),
        .new_w = WIDTH * 2,
        .new_h = HEIGHT * 2,
    };

    if (!src.tex)
        return 1;

#ifdef __AFL_HAVE_MANUAL_CONTROL
    __AFL_INIT();
#endif

    unsigned char *buf = __AFL_FUZZ_TESTCASE_BUF;
    while (__AFL_LOOP(10000)) {

#define STACK_SIZE 16
        struct pl_shader *stack[STACK_SIZE] = {0};
        int idx = 0;

        stack[0] = pl_shader_alloc(NULL, &(struct pl_shader_params) {
            .gpu = gpu,
        });

        struct pl_shader *sh = stack[idx];
        struct pl_shader_obj *polar = NULL, *ortho = NULL, *peak = NULL, *dither = NULL;

        size_t len = __AFL_FUZZ_TESTCASE_LEN;
        for (size_t pos = 0; pos < len; pos++) {
            switch (buf[pos]) {
            // Sampling steps
            case 'S':
                pl_shader_sample_direct(sh, &src);
                break;
            case 'D':
                pl_shader_deband(sh, &src, NULL);
                break;
            case 'P':
                pl_shader_sample_polar(sh, &src, &(struct pl_sample_filter_params) {
                    .filter = pl_filter_ewa_lanczos,
                    .lut = &polar,
                });
            case 'O':
                pl_shader_sample_ortho(sh, PL_SEP_VERT, &src, &(struct pl_sample_filter_params) {
                    .filter = pl_filter_spline36,
                    .lut = &ortho,
                });
                break;
            case 'X':
                pl_shader_custom(sh, &(struct pl_custom_shader) {
                    .input = PL_SHADER_SIG_NONE,
                    .output = PL_SHADER_SIG_COLOR,
                    .body = "// merge subpasses",
                });
                break;

            // Colorspace transformation steps
            case 'y': {
                struct pl_color_repr repr = pl_color_repr_jpeg;
                pl_shader_decode_color(sh, &repr, NULL);
                break;
            }
            case 'p':
                pl_shader_detect_peak(sh, pl_color_space_hdr10, &peak, NULL);
                break;
            case 'm':
                pl_shader_color_map(sh, NULL, pl_color_space_bt709,
                                    pl_color_space_monitor, NULL, false);
                break;
            case 't':
                pl_shader_color_map(sh, NULL, pl_color_space_hdr10,
                                    pl_color_space_monitor, &peak, false);
                break;
            case 'd':
                pl_shader_dither(sh, 8, &dither, &(struct pl_dither_params) {
                    // Picked to speed up calculation
                    .method = PL_DITHER_ORDERED_LUT,
                    .lut_size = 2,
                });
                break;

            // Push and pop subshader commands
            case '(':
                if (idx+1 == STACK_SIZE)
                    goto invalid;

                idx++;
                if (!stack[idx]) {
                    stack[idx] = pl_shader_alloc(NULL, &(struct pl_shader_params) {
                        .gpu = gpu,
                        .id = idx,
                    });
                }
                sh = stack[idx];
                break;

            case ')':
                if (idx == 0)
                    goto invalid;

                idx--;
                sh_subpass(stack[idx], stack[idx + 1]);
                pl_shader_reset(stack[idx + 1], &(struct pl_shader_params) {
                    .gpu = gpu,
                    .id = idx + 1,
                });
                sh = stack[idx];
                break;

            default:
                goto invalid;
            }
        }

        // Merge remaining shaders
        while (idx > 0) {
            sh_subpass(stack[idx - 1], stack[idx]);
            idx--;
        }

        pl_shader_finalize(stack[0]);

invalid:
        for (int i = 0; i < STACK_SIZE; i++)
            pl_shader_free(&stack[i]);

        pl_shader_obj_destroy(&polar);
        pl_shader_obj_destroy(&ortho);
        pl_shader_obj_destroy(&peak);
        pl_shader_obj_destroy(&dither);
    }

    pl_tex_destroy(gpu, &src.tex);
    pl_gpu_dummy_destroy(&gpu);
}
