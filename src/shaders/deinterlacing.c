/*
 * This file is part of libplacebo, but also based on vf_yadif_cuda.cu:
 * Copyright (C) 2018 Philip Langdale <philipl@overt.org>
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "shaders.h"

#include <libplacebo/shaders/deinterlacing.h>

const struct pl_deinterlace_params pl_deinterlace_default_params = { PL_DEINTERLACE_DEFAULTS };

void pl_shader_deinterlace(pl_shader sh, const struct pl_deinterlace_source *src,
                           const struct pl_deinterlace_params *params)
{
    params = PL_DEF(params, &pl_deinterlace_default_params);

    const struct pl_tex_params *texparams = &src->cur.top->params;
    if (!sh_require(sh, PL_SHADER_SIG_NONE, texparams->w, texparams->h))
        return;

    sh_describe(sh, "deinterlacing");
    GLSL("vec4 color = vec4(0,0,0,1);   \n"
         "// pl_shader_deinterlace      \n"
         "{                             \n");

    uint8_t comp_mask = PL_DEF(src->component_mask, 0xFu);
    comp_mask &= (1u << texparams->format->num_components) - 1u;
    if (!comp_mask) {
        SH_FAIL(sh, "pl_shader_deinterlace: empty component mask?");
        return;
    }

    const uint8_t num_comps = sh_num_comps(comp_mask);
    const char *swiz = sh_swizzle(comp_mask);
    GLSL("#define T %s \n", sh_float_type(comp_mask));

    ident_t pos, pt;
    ident_t cur = sh_bind(sh, src->cur.top, PL_TEX_ADDRESS_MIRROR,
                          PL_TEX_SAMPLE_NEAREST, "cur", NULL, &pos, &pt);
    if (!cur)
        return;

    GLSL("#define GET(TEX, X, Y)                              \\\n"
         "    (textureLod(TEX, pos + pt * vec2(X, Y), 0.0).%s)  \n"
         "vec2 pos = "$";                                       \n"
         "vec2 pt  = "$";                                       \n"
         "T res;                                                \n",
         swiz, pos, pt);

    if (src->field == PL_FIELD_NONE) {
        GLSL("res = GET("$", 0, 0); \n", cur);
        goto done;
    }

    // Don't modify the primary field
    GLSL("int yh = textureSize("$", 0).y;   \n"
         "int yo = int("$".y * float(yh));  \n"
         "if (yo %% 2 == %d) {              \n"
         "    res = GET("$", 0, 0);         \n"
         "} else {                          \n",
         cur, pos,
         src->field == PL_FIELD_TOP ? 0 : 1,
         cur);

    ident_t prev2 = cur, next2 = cur, prev = cur, next = cur;
    if (params->algo >= PL_DEINTERLACE_YADIF) {
        // Try using a compute shader for these, for the sole reason of
        // optimizing for thread group synchronicity. Otherwise, because we
        // alternate between lines output as-is and lines output deinterlaced,
        // half of our thread group will be mostly idle at any point in time.
        const int bw = PL_DEF(sh_glsl(sh).subgroup_size, 32);
        sh_try_compute(sh, bw, 1, true, 0);

        if (src->prev.top && src->prev.top != src->cur.top) {
            pl_assert(src->prev.top->params.w == texparams->w);
            pl_assert(src->prev.top->params.h == texparams->h);
            prev = sh_bind(sh, src->prev.top, PL_TEX_ADDRESS_MIRROR,
                           PL_TEX_SAMPLE_NEAREST, "prev", NULL, NULL, NULL);
            if (!prev)
                return;
        }

        if (src->next.top && src->next.top != src->cur.top) {
            pl_assert(src->next.top->params.w == texparams->w);
            pl_assert(src->next.top->params.h == texparams->h);
            next = sh_bind(sh, src->next.top, PL_TEX_ADDRESS_MIRROR,
                           PL_TEX_SAMPLE_NEAREST, "next", NULL, NULL, NULL);
            if (!next)
                return;
        }

        enum pl_field first_field = PL_DEF(src->first_field, PL_FIELD_TOP);
        prev2 = src->field == first_field ? prev : cur;
        next2 = src->field == first_field ? cur : next;
    }

    switch (params->algo) {
    case PL_DEINTERLACE_WEAVE:
        GLSL("res = GET("$", 0, 0); \n", cur);
        break;

    case PL_DEINTERLACE_BOB:
        GLSL("res = GET("$", 0, %d); \n", cur,
             src->field == PL_FIELD_TOP ? -1 : 1);
        break;

    case PL_DEINTERLACE_YADIF: {
        // This magic constant is hard-coded in the original implementation as
        // '1' on an 8-bit scale. Since we work with arbitrary bit depth
        // floating point textures, we have to convert this somehow. Hard-code
        // it as 1/255 under the assumption that the original intent was to be
        // roughly 1 unit of brightness increment on an 8-bit source. This may
        // or may not produce suboptimal results on higher-bit-depth content.
        static const float spatial_bias = 1 / 255.0f;

        // Calculate spatial prediction
        ident_t spatial_pred = sh_fresh(sh, "spatial_predictor");
        GLSLH("float "$"(float a, float b, float c, float d, float e, float f, float g, \n"
              "          float h, float i, float j, float k, float l, float m, float n) \n"
              "{                                                                        \n"
              "    float spatial_pred = (d + k) / 2.0;                                  \n"
              "    float spatial_score = abs(c - j) + abs(d - k) + abs(e - l) - %f;     \n"

              "    float score = abs(b - k) + abs(c - l) + abs(d - m);                  \n"
              "    if (score < spatial_score) {                                         \n"
              "        spatial_pred = (c + l) / 2.0;                                    \n"
              "        spatial_score = score;                                           \n"
              "        score = abs(a - l) + abs(b - m) + abs(c - n);                    \n"
              "        if (score < spatial_score) {                                     \n"
              "          spatial_pred = (b + m) / 2.0;                                  \n"
              "          spatial_score = score;                                         \n"
              "        }                                                                \n"
              "    }                                                                    \n"
              "    score = abs(d - i) + abs(e - j) + abs(f - k);                        \n"
              "    if (score < spatial_score) {                                         \n"
              "        spatial_pred = (e + j) / 2.0;                                    \n"
              "        spatial_score = score;                                           \n"
              "        score = abs(e - h) + abs(f - i) + abs(g - j);                    \n"
              "        if (score < spatial_score) {                                     \n"
              "          spatial_pred = (f + i) / 2.0;                                  \n"
              "          spatial_score = score;                                         \n"
              "        }                                                                \n"
              "    }                                                                    \n"
              "    return spatial_pred;                                                 \n"
              "}                                                                        \n",
              spatial_pred, spatial_bias);

        GLSL("T a = GET("$", -3, -1); \n"
             "T b = GET("$", -2, -1); \n"
             "T c = GET("$", -1, -1); \n"
             "T d = GET("$",  0, -1); \n"
             "T e = GET("$", +1, -1); \n"
             "T f = GET("$", +2, -1); \n"
             "T g = GET("$", +3, -1); \n"
             "T h = GET("$", -3, +1); \n"
             "T i = GET("$", -2, +1); \n"
             "T j = GET("$", -1, +1); \n"
             "T k = GET("$",  0, +1); \n"
             "T l = GET("$", +1, +1); \n"
             "T m = GET("$", +2, +1); \n"
             "T n = GET("$", +3, +1); \n",
             cur, cur, cur, cur, cur, cur, cur, cur, cur, cur, cur, cur, cur, cur);

        if (num_comps == 1) {
            GLSL("res = "$"(a, b, c, d, e, f, g, h, i, j, k, l, m, n); \n", spatial_pred);
        } else {
            for (uint8_t i = 0; i < num_comps; i++) {
                char c = "xyzw"[i];
                GLSL("res.%c = "$"(a.%c, b.%c, c.%c, d.%c, e.%c, f.%c, g.%c,  \n"
                     "             h.%c, i.%c, j.%c, k.%c, l.%c, m.%c, n.%c); \n",
                     c, spatial_pred, c, c, c, c, c, c, c, c, c, c, c, c, c, c);
            }
        }

        // Calculate temporal prediction
        ident_t temporal_pred = sh_fresh(sh, "temporal_predictor");
        GLSLH("float "$"(float A, float B, float C, float D, float E, float F,  \n"
              "          float G, float H, float I, float J, float K, float L,  \n"
              "          float spatial_pred)                                    \n"
              "{                                                                \n"
              "    float p0 = (C + H) / 2.0;                                    \n"
              "    float p1 = F;                                                \n"
              "    float p2 = (D + I) / 2.0;                                    \n"
              "    float p3 = G;                                                \n"
              "    float p4 = (E + J) / 2.0;                                    \n"

              "    float tdiff0 = abs(D - I) / 2.0;                             \n"
              "    float tdiff1 = (abs(A - F) + abs(B - G)) / 2.0;              \n"
              "    float tdiff2 = (abs(K - F) + abs(G - L)) / 2.0;              \n"
              "    float diff = max(tdiff0, max(tdiff1, tdiff2));               \n",
              temporal_pred);
        if (!params->skip_spatial_check) {
            GLSLH("float maxi = max(p2 - min(p3, p1), min(p0 - p1, p4 - p3));   \n"
                  "float mini = min(p2 - max(p3, p1), max(p0 - p1, p4 - p3));   \n"
                  "diff = max(diff, max(mini, -maxi));                          \n");
        }
        GLSLH("    if (spatial_pred > p2 + diff)                                \n"
              "      spatial_pred = p2 + diff;                                  \n"
              "    if (spatial_pred < p2 - diff)                                \n"
              "      spatial_pred = p2 - diff;                                  \n"
              "    return spatial_pred;                                         \n"
              "}                                                                \n");

        GLSL("T A = GET("$", 0, -1); \n"
             "T B = GET("$", 0,  1); \n"
             "T C = GET("$", 0, -2); \n"
             "T D = GET("$", 0,  0); \n"
             "T E = GET("$", 0, +2); \n"
             "T F = GET("$", 0, -1); \n"
             "T G = GET("$", 0, +1); \n"
             "T H = GET("$", 0, -2); \n"
             "T I = GET("$", 0,  0); \n"
             "T J = GET("$", 0, +2); \n"
             "T K = GET("$", 0, -1); \n"
             "T L = GET("$", 0, +1); \n",
             prev, prev,
             prev2, prev2, prev2,
             cur, cur,
             next2, next2, next2,
             next, next);

        if (num_comps == 1) {
            GLSL("res = "$"(A, B, C, D, E, F, G, H, I, J, K, L, res); \n", temporal_pred);
        } else {
            for (uint8_t i = 0; i < num_comps; i++) {
                char c = "xyzw"[i];
                GLSL("res.%c = "$"(A.%c, B.%c, C.%c, D.%c, E.%c, F.%c, \n"
                     "             G.%c, H.%c, I.%c, J.%c, K.%c, L.%c, \n"
                     "             res.%c);                            \n",
                     c, temporal_pred, c, c, c, c, c, c, c, c, c, c, c, c, c);
            }
        }
        break;
    }

    case PL_DEINTERLACE_ALGORITHM_COUNT:
        pl_unreachable();
    }

    GLSL("}\n"); // End of primary/secondary field branch

done:
    GLSL("color.%s = res;   \n"
         "#undef T          \n"
         "#undef GET        \n"
         "}                 \n",
         swiz);
}
