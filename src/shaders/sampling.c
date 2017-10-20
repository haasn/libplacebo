/*
 * This file is part of libplacebo.
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

#include <math.h>
#include "shaders.h"

const struct pl_deband_params pl_deband_default_params = {
    .iterations = 1,
    .threshold  = 4.0,
    .radius     = 16.0,
    .grain      = 6.0,
};

void pl_shader_deband(struct pl_shader *sh, const struct ra_tex *ra_tex,
                      const struct pl_deband_params *params)
{
    if (!sh_require(sh, PL_SHADER_SIG_NONE, ra_tex->params.w, ra_tex->params.h))
        return;

    GLSL("vec4 color;\n");
    GLSL("// pl_shader_deband\n");
    GLSL("{\n");

    ident_t tex, pos, pt;
    tex = sh_bind(sh, ra_tex, "deband", &pl_transform2x2_identity, &pos, NULL, &pt);
    if (!tex)
        return;

    // Initialize the PRNG. This is friendly for wide usage and returns in
    // a very pleasant-looking distribution across frames even if the difference
    // between input coordinates is very small. Shamelessly stolen from some
    // GLSL tricks forum post years from a decade ago.
    ident_t random = sh_fresh(sh, "random"), permute = sh_fresh(sh, "permute");
    GLSLH("float %s(float x) {                          \n"
          "    x = (34.0 * x + 1.0) * x;                \n"
          "    return x - floor(x * 1.0/289.0) * 289.0; \n" // mod 289
          "}                                            \n"
          "float %s(inout float state) {                \n"
          "    state = %s(state);                       \n"
          "    return fract(state * 1.0/41.0);          \n"
          "}\n", permute, random, permute);

    ident_t seed = sh_var(sh, (struct pl_shader_var) {
        .var  = ra_var_float("seed"),
        .data = &params->seed,
    });

    GLSL("vec3 _m = vec3(%s, %s) + vec3(1.0);          \n"
         "float prng = %s(%s(%s(_m.x) + _m.y) + _m.z); \n"
         "vec4 avg, diff;                              \n"
         "color = texture(%s, %s);                     \n",
         pos, seed, permute, permute, permute, tex, pos);

    // Helper function: Compute a stochastic approximation of the avg color
    // around a pixel, given a specified radius
    ident_t average = sh_fresh(sh, "average");
    GLSLH("vec4 %s(vec2 pos, float range, inout float prng) {   \n"
          // Compute a random angle and distance
          "    float dist = %s(prng) * range;                   \n"
          "    float dir  = %s(prng) * %f;                      \n"
          "    vec2 o = dist * vec2(cos(dir), sin(dir));        \n"
          // Sample at quarter-turn intervals around the source pixel
          "    vec4 sum = vec4(0.0);                            \n"
          "    sum += texture(%s, pos + %s * vec2( o.x,  o.y)); \n"
          "    sum += texture(%s, pos + %s * vec2(-o.x,  o.y)); \n"
          "    sum += texture(%s, pos + %s * vec2(-o.x, -o.y)); \n"
          "    sum += texture(%s, pos + %s * vec2( o.x, -o.y)); \n"
          // Return the (normalized) average
          "    return 0.25 * sum;                               \n"
          "}\n", average, random, random, M_PI * 2,
          tex, pt, tex, pt, tex, pt, tex, pt);

    // For each iteration, compute the average at a given distance and
    // pick it instead of the color if the difference is below the threshold.
    for (int i = 1; i <= params->iterations; i++) {
        GLSL("avg = %s(%s, %f, prng);                               \n"
             "diff = abs(color - avg);                              \n"
             "color = mix(avg, color, greaterThan(diff, vec4(%f))); \n",
             average, pos, i * params->radius, params->threshold / (1000 * i));
    }

    // Add some random noise to smooth out residual differences
    GLSL("vec3 noise = vec3(%s(prng), %s(prng), %s(prng)); \n"
         "color.rgb += %f * (noise - vec3(0.5));           \n",
         random, random, random, params->grain / 1000.0);

    GLSL("}\n");
}
