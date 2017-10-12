#include "tests.h"

int main()
{
    struct pl_context *ctx = pl_test_context();
    pl_context_destroy(&ctx);
}
