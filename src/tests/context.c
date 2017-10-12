#include "tests.h"

int main()
{
    struct pl_context *ctx = pl_context_create(&ctx_params, PL_API_VER);
    pl_context_destroy(&ctx);
}
