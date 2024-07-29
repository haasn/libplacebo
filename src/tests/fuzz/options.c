#include "../utils.h"

#include <libplacebo/options.h>

__AFL_FUZZ_INIT();

#pragma clang optimize off

int main()
{
    pl_options opts = pl_options_alloc(NULL);

#ifdef __AFL_HAVE_MANUAL_CONTROL
    __AFL_INIT();
#endif

    unsigned char *buf = __AFL_FUZZ_TESTCASE_BUF;

    while (__AFL_LOOP(100000)) {
        size_t len = __AFL_FUZZ_TESTCASE_LEN;
        buf[len - 1] = '\0'; // ensure proper null termination
        pl_options_load(opts, (const char *) buf);
        pl_options_save(opts);
        pl_options_reset(opts, NULL);
    }
}
