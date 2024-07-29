#include "../utils.h"

#include <libplacebo/dummy.h>
#include <libplacebo/shaders/custom.h>

__AFL_FUZZ_INIT();

#pragma clang optimize off

int main()
{
    pl_gpu gpu = pl_gpu_dummy_create(NULL, NULL);
    const struct pl_hook *hook;

#ifdef __AFL_HAVE_MANUAL_CONTROL
    __AFL_INIT();
#endif

    unsigned char *buf = __AFL_FUZZ_TESTCASE_BUF;

    while (__AFL_LOOP(100000)) {
        size_t len = __AFL_FUZZ_TESTCASE_LEN;
        hook = pl_mpv_user_shader_parse(gpu, (char *) buf, len);
        pl_mpv_user_shader_destroy(&hook);
    }

    pl_gpu_dummy_destroy(&gpu);
}
