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

#include "shaders.h"

bool pl_shader_custom(pl_shader sh, const struct pl_custom_shader *params)
{
    if (params->compute) {
        int bw = PL_DEF(params->compute_group_size[0], 16);
        int bh = PL_DEF(params->compute_group_size[1], 16);
        bool flex = !params->compute_group_size[0] ||
                    !params->compute_group_size[1];
        if (!sh_try_compute(sh, bw, bh, flex, params->compute_shmem))
            return false;
    }

    if (!sh_require(sh, params->input, params->output_w, params->output_h))
        return false;

    sh->res.output = params->output;

    // Attach the variables, descriptors etc. directly instead of going via
    // `sh_var` / `sh_desc` etc. to avoid generating fresh names
    for (int i = 0; i < params->num_variables; i++) {
        struct pl_shader_var sv = params->variables[i];
        sv.data = pl_memdup(SH_TMP(sh), sv.data, pl_var_host_layout(0, &sv.var).size);
        sv.var.name = pl_strdup0(SH_TMP(sh), pl_str0(sv.var.name));
        PL_ARRAY_APPEND(sh, sh->vars, sv);
    }

    for (int i = 0; i < params->num_descriptors; i++) {
        struct pl_shader_desc sd = params->descriptors[i];
        size_t bsize = sizeof(sd.buffer_vars[0]) * sd.num_buffer_vars;
        if (bsize)
            sd.buffer_vars = pl_memdup(SH_TMP(sh), sd.buffer_vars, bsize);
        sd.desc.name = pl_strdup0(SH_TMP(sh), pl_str0(sd.desc.name));
        PL_ARRAY_APPEND(sh, sh->descs, sd);
    }

    for (int i = 0; i < params->num_vertex_attribs; i++) {
        struct pl_shader_va sva = params->vertex_attribs[i];
        size_t vsize = sva.attr.fmt->texel_size;
        for (int n = 0; n < PL_ARRAY_SIZE(sva.data); n++)
            sva.data[n] = pl_memdup(SH_TMP(sh), sva.data[n], vsize);
        sva.attr.name = pl_strdup0(SH_TMP(sh), pl_str0(sva.attr.name));
        PL_ARRAY_APPEND(sh, sh->vas, sva);
    }

    for (int i = 0; i < params->num_constants; i++) {
        struct pl_shader_const sc = params->constants[i];
        size_t csize = pl_var_type_size(sc.type);
        sc.data = pl_memdup(SH_TMP(sh), sc.data, csize);
        sc.name = pl_strdup0(SH_TMP(sh), pl_str0(sc.name));
        PL_ARRAY_APPEND(sh, sh->consts, sc);
    }

    if (params->prelude)
        GLSLP("// pl_shader_custom prelude: \n%s\n", params->prelude);
    if (params->header)
        GLSLH("// pl_shader_custom header: \n%s\n", params->header);

    if (params->description)
        sh_describe(sh, pl_strdup0(SH_TMP(sh), pl_str0(params->description)));

    if (params->body) {
        const char *output_decl = "";
        if (params->output != params->input) {
            switch (params->output) {
            case PL_SHADER_SIG_NONE: break;
            case PL_SHADER_SIG_COLOR:
                output_decl = "vec4 color = vec4(0.0);";
                break;

            case PL_SHADER_SIG_SAMPLER:
                pl_unreachable();
            }
        }

        GLSL("// pl_shader_custom \n"
             "%s                  \n"
             "{                   \n"
             "%s                  \n"
             "}                   \n",
             output_decl, params->body);
    }

    return true;
}
