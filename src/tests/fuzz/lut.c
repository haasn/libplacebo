#include "../tests.h"

#include <libplacebo/shaders/lut.h>

__AFL_FUZZ_INIT();

#pragma clang optimize off

int main()
{
    struct pl_custom_lut *lut;

#ifdef __AFL_HAVE_MANUAL_CONTROL
    __AFL_INIT();
#endif

    unsigned char *buf = __AFL_FUZZ_TESTCASE_BUF;

    while (__AFL_LOOP(100000)) {
        size_t len = __AFL_FUZZ_TESTCASE_LEN;
        lut = pl_lut_parse_cube(NULL, (char *) buf, len);
        pl_lut_free(&lut);
    }
}
