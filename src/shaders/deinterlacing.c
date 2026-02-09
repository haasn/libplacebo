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

    const enum pl_field first_field = PL_DEF(src->first_field, PL_FIELD_TOP);

    bool intra_only = params->algo != PL_DEINTERLACE_YADIF;
    if (params->algo == PL_DEINTERLACE_BWDIF) {
        intra_only = (!src->prev.top && src->field == first_field) ||
                     (!src->next.top && src->field != first_field);
    }

    ident_t prev = cur, next = cur;
    if (params->algo >= PL_DEINTERLACE_YADIF) {
        // Try using a compute shader for these, for the sole reason of
        // optimizing for thread group synchronicity. Otherwise, because we
        // alternate between lines output as-is and lines output deinterlaced,
        // half of our thread group will be mostly idle at any point in time.
        const int bw = PL_DEF(sh_glsl(sh).subgroup_size, 32);
        sh_try_compute(sh, bw, 1, true, 0);
    }

    if (!intra_only && src->prev.top && src->prev.top != src->cur.top) {
        pl_assert(src->prev.top->params.w == texparams->w);
        pl_assert(src->prev.top->params.h == texparams->h);
        prev = sh_bind(sh, src->prev.top, PL_TEX_ADDRESS_MIRROR,
                        PL_TEX_SAMPLE_NEAREST, "prev", NULL, NULL, NULL);
        if (!prev)
            return;
    }

    if (!intra_only && src->next.top && src->next.top != src->cur.top) {
        pl_assert(src->next.top->params.w == texparams->w);
        pl_assert(src->next.top->params.h == texparams->h);
        next = sh_bind(sh, src->next.top, PL_TEX_ADDRESS_MIRROR,
                        PL_TEX_SAMPLE_NEAREST, "next", NULL, NULL, NULL);
        if (!next)
            return;
    }

    ident_t prev2 = src->field == first_field ? prev : cur;
    ident_t next2 = src->field == first_field ? cur : next;

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

    case PL_DEINTERLACE_BWDIF: {
        // cur[]         = { mrefs3, mrefs, prefs, prefs3 }
        // prev/next[]   = { mrefs, prefs }
        // prev2/next2[] = { mrefs4, mrefs2, 0, prefs2, prefs4 }
        ident_t process = sh_fresh(sh, "process_bwdif");
        ident_t intra = sh_fresh(sh, "process_intra_bwdif");

#pragma GLSLH                                                                   \
        #define T ${vecType: comp_mask}                                         \
        T $process(T cur[4], T prev[2], T next[2], T prev2[5], T next2[5])      \
        {                                                                       \
            const float lf[2] = float[]( 4309.0/8192.0,  213.0/8192.0 );        \
            const float hf[3] = float[]( 5570.0/8192.0, 3801.0/8192.0, 1016.0/8192.0 ); \
            const float sp[2] = float[]( 5077.0/8192.0,  981.0/8192.0 );        \
                                                                                \
            T s = prev2[2] + next2[2];                                          \
            T d = s / 2.0;                                                      \
            T c = cur[1];                                                       \
            T e = cur[2];                                                       \
                                                                                \
            T tdiff0 = abs(prev2[2] - next2[2]);                                \
            T tdiff1 = abs(prev[0] - c) + abs(prev[1] - e);                     \
            T tdiff2 = abs(next[0] - c) + abs(next[1] - e);                     \
            T diff = max(tdiff0, max(tdiff1, tdiff2)) / 2.0;                    \
                                                                                \
            @if (num_comps > 1)                                                 \
            ${bvecType: comp_mask} diff_mask = equal(diff, T(0.0));             \
            @else                                                               \
            bool diff_mask = diff == 0.0;                                       \
                                                                                \
            T bs = prev2[1] + next2[1];                                         \
            T fs = prev2[3] + next2[3];                                         \
            T b = (bs / 2.0) - c;                                               \
            T f = (fs / 2.0) - c;                                               \
            T dc = d - c;                                                       \
            T de = d - e;                                                       \
            T mmax = max(de, max(dc, min(b, f)));                               \
            T mmin = min(de, min(dc, max(b, f)));                               \
            diff = max(diff, max(mmin, -mmax));                                 \
                                                                                \
            T single = sp[0] * (c+e) - sp[1] * (cur[0] + cur[3]);               \
            T all = (hf[0]*s - hf[1] * (bs+fs) +                                \
            hf[2] * (prev2[0] + next2[0] + prev2[4] + next2[4])) / 4.0;         \
            all += lf[0] * (c + e) - lf[1] * (cur[0] + cur[3]);                 \
                                                                                \
            @if (num_comps > 1)                                                 \
            ${bvecType: comp_mask} mask = greaterThan(abs(c - e), tdiff0);      \
            @else                                                               \
            bool mask = abs(c - e) > tdiff0;                                    \
                                                                                \
            T interpol = mix(single, all, mask);                                \
            interpol = clamp(interpol, d - diff, d + diff);                     \
            return mix(interpol, d, diff_mask);                                 \
        }                                                                       \
                                                                                \
        T $intra(T cur[4])                                                      \
        {                                                                       \
            const float sp[2] = float[]( 5077.0/8192.0, 981.0/8192.0 );         \
            return sp[0] * (cur[1] + cur[2]) - sp[1] * (cur[0] + cur[3]);       \
        }                                                                       \
        #undef T                                                                \

#pragma GLSL /* pl_shader_deinterlace (bwdif) */                                \
        T cur[4];                                                               \
        cur[0] = GET($cur, 0, -3);                                              \
        cur[1] = GET($cur, 0, -1);                                              \
        cur[2] = GET($cur, 0,  1);                                              \
        cur[3] = GET($cur, 0,  3);                                              \
                                                                                \
        @if (!intra_only) {                                                     \
            T prev[2], next[2], prev2[5], next2[5];                             \
            prev[0] = GET($prev, 0, -1);                                        \
            prev[1] = GET($prev, 0,  1);                                        \
            next[0] = GET($next, 0, -1);                                        \
            next[1] = GET($next, 0,  1);                                        \
                                                                                \
            prev2[0] = GET($prev2, 0, -4);                                      \
            prev2[1] = GET($prev2, 0, -2);                                      \
            prev2[2] = GET($prev2, 0,  0);                                      \
            prev2[3] = GET($prev2, 0,  2);                                      \
            prev2[4] = GET($prev2, 0,  4);                                      \
                                                                                \
            next2[0] = GET($next2, 0, -4);                                      \
            next2[1] = GET($next2, 0, -2);                                      \
            next2[2] = GET($next2, 0,  0);                                      \
            next2[3] = GET($next2, 0,  2);                                      \
            next2[4] = GET($next2, 0,  4);                                      \
                                                                                \
            res = $process(cur, prev, next, prev2, next2);                      \
        @} else {                                                               \
            res = $intra(cur);                                                  \
        @}
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
