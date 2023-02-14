#include "tests.h"
#include "libplacebo/utils/dav1d.h"

int main()
{
    // Test enum functions
    for (enum pl_color_system sys = 0; sys < PL_COLOR_SYSTEM_COUNT; sys++) {
        // Exceptions to the rule, due to different handling in dav1d
        if (sys == PL_COLOR_SYSTEM_BT_2100_HLG || sys == PL_COLOR_SYSTEM_XYZ)
            continue;

        enum Dav1dMatrixCoefficients mc = pl_system_to_dav1d(sys);
        enum pl_color_system sys2 = pl_system_from_dav1d(mc);
        if (sys2)
            REQUIRE_CMP(sys, ==, sys2, "u");
    }

    for (enum pl_color_levels lev = 0; lev < PL_COLOR_LEVELS_COUNT; lev++) {
        int range = pl_levels_to_dav1d(lev);
        enum pl_color_levels lev2 = pl_levels_from_dav1d(range);
        if (lev != PL_COLOR_LEVELS_UNKNOWN)
            REQUIRE_CMP(lev, ==, lev2, "u");
    }

    for (enum pl_color_primaries prim = 0; prim < PL_COLOR_PRIM_COUNT; prim++) {
        enum Dav1dColorPrimaries dpri = pl_primaries_to_dav1d(prim);
        enum pl_color_primaries prim2 = pl_primaries_from_dav1d(dpri);
        if (prim2)
            REQUIRE_CMP(prim, ==, prim2, "u");
    }

    for (enum pl_color_transfer trc = 0; trc < PL_COLOR_TRC_COUNT; trc++) {
        enum Dav1dTransferCharacteristics dtrc = pl_transfer_to_dav1d(trc);
        enum pl_color_transfer trc2 = pl_transfer_from_dav1d(dtrc);
        if (trc2)
            REQUIRE_CMP(trc, ==, trc2, "u");
    }

    for (enum pl_chroma_location loc = 0; loc < PL_CHROMA_COUNT; loc++) {
        enum Dav1dChromaSamplePosition dloc = pl_chroma_to_dav1d(loc);
        enum pl_chroma_location loc2 = pl_chroma_from_dav1d(dloc);
        if (loc2)
            REQUIRE_CMP(loc, ==, loc2, "u");
    }
}
