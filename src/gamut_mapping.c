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

#include "common.h"
#include "colorspace.h"
#include "pl_thread.h"

#include <libplacebo/gamut_mapping.h>

#define fclampf(x, lo, hi) fminf(fmaxf(x, lo), hi)
static void fix_constants(struct pl_gamut_map_constants *c)
{
    c->perceptual_deadzone = fclampf(c->perceptual_deadzone, 0.0f, 1.0f);
    c->perceptual_strength = fclampf(c->perceptual_strength, 0.0f, 1.0f);
    c->colorimetric_gamma  = fclampf(c->colorimetric_gamma, 0.0f, 10.0f);
    c->softclip_knee       = fclampf(c->softclip_knee, 0.0f, 1.0f);
    c->softclip_desat      = fclampf(c->softclip_desat, 0.0f, 1.0f);
}

static inline bool constants_equal(const struct pl_gamut_map_constants *a,
                                   const struct pl_gamut_map_constants *b)
{
    pl_static_assert(sizeof(*a) % sizeof(float) == 0);
    return !memcmp(a, b, sizeof(*a));
}

bool pl_gamut_map_params_equal(const struct pl_gamut_map_params *a,
                               const struct pl_gamut_map_params *b)
{
    return a->function      == b->function      &&
           a->min_luma      == b->min_luma      &&
           a->max_luma      == b->max_luma      &&
           a->lut_size_I    == b->lut_size_I    &&
           a->lut_size_C    == b->lut_size_C    &&
           a->lut_size_h    == b->lut_size_h    &&
           a->lut_stride    == b->lut_stride    &&
           constants_equal(&a->constants, &b->constants) &&
           pl_raw_primaries_equal(&a->input_gamut,  &b->input_gamut) &&
           pl_raw_primaries_equal(&a->output_gamut, &b->output_gamut);
}

#define FUN(params) (params->function ? *params->function : pl_gamut_map_clip)

static void noop(float *lut, const struct pl_gamut_map_params *params);
bool pl_gamut_map_params_noop(const struct pl_gamut_map_params *params)
{
    if (FUN(params).map == &noop)
        return true;

    struct pl_raw_primaries src = params->input_gamut, dst = params->output_gamut;
    if (!pl_primaries_compatible(&dst, &src))
        return true;

    bool need_map = !pl_primaries_superset(&dst, &src);
    need_map |= !pl_cie_xy_equal(&src.white, &dst.white);
    if (FUN(params).bidirectional)
        need_map |= !pl_raw_primaries_equal(&dst, &src);

    return !need_map;
}

// For some minimal type safety, and code cleanliness
struct RGB {
    float R, G, B;
};

struct IPT {
    float I, P, T;
};

struct ICh {
    float I, C, h;
};

static inline struct ICh ipt2ich(struct IPT c)
{
    return (struct ICh) {
        .I = c.I,
        .C = sqrtf(c.P * c.P + c.T * c.T),
        .h = atan2f(c.T, c.P),
    };
}

static inline struct IPT ich2ipt(struct ICh c)
{
    return (struct IPT) {
        .I = c.I,
        .P = c.C * cosf(c.h),
        .T = c.C * sinf(c.h),
    };
}

enum { PQ_LUT_SIZE = 1024 };
static const float pq_eotf_lut[1024+1] = {
    0.00000000e+00f, 4.04227185e-09f, 1.31113715e-08f, 2.62368260e-08f, 4.31514948e-08f, 6.37468887e-08f, 8.79823858e-08f, 1.15853616e-07f,
    1.47378188e-07f, 1.82588181e-07f, 2.21525866e-07f, 2.64240981e-07f, 3.10789062e-07f, 3.61230207e-07f, 4.15628222e-07f, 4.74050012e-07f,
    5.36565210e-07f, 6.03245780e-07f, 6.74165676e-07f, 7.49400897e-07f, 8.29028977e-07f, 9.13129156e-07f, 1.00178215e-06f, 1.09507005e-06f,
    1.19307640e-06f, 1.29588602e-06f, 1.40358475e-06f, 1.51625989e-06f, 1.63400000e-06f, 1.75689468e-06f, 1.88503464e-06f, 2.01851185e-06f,
    2.15741920e-06f, 2.30185105e-06f, 2.45190267e-06f, 2.60767024e-06f, 2.76925152e-06f, 2.93674498e-06f, 3.11025065e-06f, 3.28986880e-06f,
    3.47570199e-06f, 3.66785275e-06f, 3.86642569e-06f, 4.07152584e-06f, 4.28326030e-06f, 4.50173548e-06f, 4.72706142e-06f, 4.95934728e-06f,
    5.19870400e-06f, 5.44524437e-06f, 5.69908161e-06f, 5.96032987e-06f, 6.22910557e-06f, 6.50552556e-06f, 6.78970764e-06f, 7.08177140e-06f,
    7.38183780e-06f, 7.69002872e-06f, 8.00646649e-06f, 8.33127706e-06f, 8.66458504e-06f, 9.00651685e-06f, 9.35720254e-06f, 9.71677036e-06f,
    1.00853513e-05f, 1.04630772e-05f, 1.08500826e-05f, 1.12465023e-05f, 1.16524716e-05f, 1.20681298e-05f, 1.24936132e-05f, 1.29290656e-05f,
    1.33746262e-05f, 1.38304395e-05f, 1.42966492e-05f, 1.47734027e-05f, 1.52608463e-05f, 1.57591294e-05f, 1.62684046e-05f, 1.67888211e-05f,
    1.73205335e-05f, 1.78636965e-05f, 1.84184682e-05f, 1.89850052e-05f, 1.95634675e-05f, 2.01540170e-05f, 2.07568173e-05f, 2.13720305e-05f,
    2.19998237e-05f, 2.26403663e-05f, 2.32938255e-05f, 2.39603742e-05f, 2.46401833e-05f, 2.53334292e-05f, 2.60402867e-05f, 2.67609339e-05f,
    2.74955528e-05f, 2.82443198e-05f, 2.90074222e-05f, 2.97850438e-05f, 3.05773719e-05f, 3.13845921e-05f, 3.22068954e-05f, 3.30444782e-05f,
    3.38975333e-05f, 3.47662535e-05f, 3.56508390e-05f, 3.65514898e-05f, 3.74684059e-05f, 3.84017912e-05f, 3.93518530e-05f, 4.03187987e-05f,
    4.13028356e-05f, 4.23041784e-05f, 4.33230380e-05f, 4.43596327e-05f, 4.54141773e-05f, 4.64868936e-05f, 4.75780034e-05f, 4.86877325e-05f,
    4.98163026e-05f, 5.09639467e-05f, 5.21308903e-05f, 5.33173734e-05f, 5.45236253e-05f, 5.57498824e-05f, 5.69963886e-05f, 5.82633838e-05f,
    5.95511119e-05f, 6.08598166e-05f, 6.21897489e-05f, 6.35411634e-05f, 6.49143112e-05f, 6.63094470e-05f, 6.77268254e-05f, 6.91667155e-05f,
    7.06293722e-05f, 7.21150645e-05f, 7.36240690e-05f, 7.51566404e-05f, 7.67130623e-05f, 7.82936040e-05f, 7.98985493e-05f, 8.15281819e-05f,
    8.31827783e-05f, 8.48626223e-05f, 8.65680195e-05f, 8.82992390e-05f, 9.00565938e-05f, 9.18403675e-05f, 9.36508732e-05f, 9.54884017e-05f,
    9.73532660e-05f, 9.92457644e-05f, 1.01166224e-04f, 1.03114944e-04f, 1.05092251e-04f, 1.07098465e-04f, 1.09133900e-04f, 1.11198889e-04f,
    1.13293761e-04f, 1.15418850e-04f, 1.17574484e-04f, 1.19761011e-04f, 1.21978759e-04f, 1.24228085e-04f, 1.26509331e-04f, 1.28822838e-04f,
    1.31168985e-04f, 1.33548107e-04f, 1.35960567e-04f, 1.38406744e-04f, 1.40887001e-04f, 1.43401689e-04f, 1.45951213e-04f, 1.48535924e-04f,
    1.51156215e-04f, 1.53812463e-04f, 1.56505077e-04f, 1.59234420e-04f, 1.62000913e-04f, 1.64804937e-04f, 1.67646896e-04f, 1.70527201e-04f,
    1.73446271e-04f, 1.76404501e-04f, 1.79402312e-04f, 1.82440141e-04f, 1.85518395e-04f, 1.88637510e-04f, 1.91797924e-04f, 1.95000059e-04f,
    1.98244379e-04f, 2.01531308e-04f, 2.04861295e-04f, 2.08234807e-04f, 2.11652281e-04f, 2.15114211e-04f, 2.18621033e-04f, 2.22173214e-04f,
    2.25771248e-04f, 2.29415615e-04f, 2.33106766e-04f, 2.36845211e-04f, 2.40631445e-04f, 2.44465948e-04f, 2.48349243e-04f, 2.52281810e-04f,
    2.56264175e-04f, 2.60296831e-04f, 2.64380331e-04f, 2.68515141e-04f, 2.72701844e-04f, 2.76940962e-04f, 2.81233020e-04f, 2.85578542e-04f,
    2.89978110e-04f, 2.94432277e-04f, 2.98941566e-04f, 3.03506560e-04f, 3.08127812e-04f, 3.12805903e-04f, 3.17541446e-04f, 3.22334934e-04f,
    3.27187037e-04f, 3.32098338e-04f, 3.37069388e-04f, 3.42100830e-04f, 3.47193272e-04f, 3.52347299e-04f, 3.57563549e-04f, 3.62842635e-04f,
    3.68185225e-04f, 3.73591931e-04f, 3.79063364e-04f, 3.84600222e-04f, 3.90203117e-04f, 3.95872747e-04f, 4.01609752e-04f, 4.07414831e-04f,
    4.13288624e-04f, 4.19231830e-04f, 4.25245118e-04f, 4.31329216e-04f, 4.37484821e-04f, 4.43712634e-04f, 4.50013351e-04f, 4.56387730e-04f,
    4.62836440e-04f, 4.69360268e-04f, 4.75959940e-04f, 4.82636213e-04f, 4.89389815e-04f, 4.96221473e-04f, 5.03132062e-04f, 5.10122220e-04f,
    5.17192821e-04f, 5.24344621e-04f, 5.31578436e-04f, 5.38895023e-04f, 5.46295254e-04f, 5.53779886e-04f, 5.61349734e-04f, 5.69005671e-04f,
    5.76748513e-04f, 5.84579073e-04f, 5.92498283e-04f, 6.00506901e-04f, 6.08605857e-04f, 6.16796024e-04f, 6.25078275e-04f, 6.33453543e-04f,
    6.41922641e-04f, 6.50486501e-04f, 6.59146055e-04f, 6.67902175e-04f, 6.76755910e-04f, 6.85708073e-04f, 6.94759656e-04f, 7.03911646e-04f,
    7.13164918e-04f, 7.22520519e-04f, 7.31979439e-04f, 7.41542608e-04f, 7.51211017e-04f, 7.60985771e-04f, 7.70867744e-04f, 7.80858041e-04f,
    7.90957711e-04f, 8.01167742e-04f, 8.11489241e-04f, 8.21923197e-04f, 8.32470774e-04f, 8.43132904e-04f, 8.53910809e-04f, 8.64805537e-04f,
    8.75818194e-04f, 8.86949943e-04f, 8.98201775e-04f, 9.09574970e-04f, 9.21070634e-04f, 9.32689873e-04f, 9.44433908e-04f, 9.56303847e-04f,
    9.68300970e-04f, 9.80426441e-04f, 9.92681365e-04f, 1.00506714e-03f, 1.01758481e-03f, 1.03023578e-03f, 1.04302121e-03f, 1.05594238e-03f,
    1.06900057e-03f, 1.08219695e-03f, 1.09553302e-03f, 1.10900984e-03f, 1.12262904e-03f, 1.13639166e-03f, 1.15029910e-03f, 1.16435287e-03f,
    1.17855414e-03f, 1.19290443e-03f, 1.20740500e-03f, 1.22205738e-03f, 1.23686285e-03f, 1.25182292e-03f, 1.26693910e-03f, 1.28221256e-03f,
    1.29764504e-03f, 1.31323794e-03f, 1.32899266e-03f, 1.34491071e-03f, 1.36099372e-03f, 1.37724308e-03f, 1.39366044e-03f, 1.41024729e-03f,
    1.42700516e-03f, 1.44393567e-03f, 1.46104046e-03f, 1.47832115e-03f, 1.49577926e-03f, 1.51341641e-03f, 1.53123436e-03f, 1.54923473e-03f,
    1.56741915e-03f, 1.58578926e-03f, 1.60434691e-03f, 1.62309385e-03f, 1.64203160e-03f, 1.66116212e-03f, 1.68048707e-03f, 1.70000829e-03f,
    1.71972741e-03f, 1.73964654e-03f, 1.75976718e-03f, 1.78009144e-03f, 1.80062104e-03f, 1.82135787e-03f, 1.84230390e-03f, 1.86346099e-03f,
    1.88483100e-03f, 1.90641591e-03f, 1.92821771e-03f, 1.95023837e-03f, 1.97247975e-03f, 1.99494418e-03f, 2.01763329e-03f, 2.04054941e-03f,
    2.06369464e-03f, 2.08707084e-03f, 2.11068010e-03f, 2.13452475e-03f, 2.15860689e-03f, 2.18292838e-03f, 2.20749178e-03f, 2.23229895e-03f,
    2.25735223e-03f, 2.28265417e-03f, 2.30820640e-03f, 2.33401172e-03f, 2.36007199e-03f, 2.38639000e-03f, 2.41296762e-03f, 2.43980740e-03f,
    2.46691168e-03f, 2.49428279e-03f, 2.52192328e-03f, 2.54983548e-03f, 2.57802196e-03f, 2.60648504e-03f, 2.63522728e-03f, 2.66425102e-03f,
    2.69355904e-03f, 2.72315391e-03f, 2.75303796e-03f, 2.78321397e-03f, 2.81368452e-03f, 2.84445239e-03f, 2.87552015e-03f, 2.90689035e-03f,
    2.93856580e-03f, 2.97054928e-03f, 3.00284335e-03f, 3.03545129e-03f, 3.06837540e-03f, 3.10161896e-03f, 3.13518429e-03f, 3.16907465e-03f,
    3.20329308e-03f, 3.23784212e-03f, 3.27272480e-03f, 3.30794440e-03f, 3.34350369e-03f, 3.37940571e-03f, 3.41565371e-03f, 3.45225050e-03f,
    3.48919933e-03f, 3.52650345e-03f, 3.56416591e-03f, 3.60218994e-03f, 3.64057859e-03f, 3.67933535e-03f, 3.71846370e-03f, 3.75796645e-03f,
    3.79784708e-03f, 3.83810909e-03f, 3.87875596e-03f, 3.91979096e-03f, 3.96121759e-03f, 4.00303910e-03f, 4.04525921e-03f, 4.08788165e-03f,
    4.13090969e-03f, 4.17434704e-03f, 4.21819743e-03f, 4.26246459e-03f, 4.30715177e-03f, 4.35226317e-03f, 4.39780252e-03f, 4.44377353e-03f,
    4.49017994e-03f, 4.53702547e-03f, 4.58431430e-03f, 4.63205064e-03f, 4.68023773e-03f, 4.72887978e-03f, 4.77798143e-03f, 4.82754596e-03f,
    4.87757800e-03f, 4.92808130e-03f, 4.97906050e-03f, 5.03051933e-03f, 5.08246245e-03f, 5.13489358e-03f, 5.18781738e-03f, 5.24123851e-03f,
    5.29516116e-03f, 5.34958951e-03f, 5.40452823e-03f, 5.45998197e-03f, 5.51595492e-03f, 5.57245174e-03f, 5.62947709e-03f, 5.68703609e-03f,
    5.74513292e-03f, 5.80377271e-03f, 5.86295966e-03f, 5.92269935e-03f, 5.98299596e-03f, 6.04385510e-03f, 6.10528141e-03f, 6.16727956e-03f,
    6.22985465e-03f, 6.29301229e-03f, 6.35675760e-03f, 6.42109523e-03f, 6.48603030e-03f, 6.55156886e-03f, 6.61771558e-03f, 6.68447604e-03f,
    6.75185584e-03f, 6.81986008e-03f, 6.88849436e-03f, 6.95776427e-03f, 7.02767540e-03f, 7.09823379e-03f, 7.16944411e-03f, 7.24131335e-03f,
    7.31384614e-03f, 7.38704903e-03f, 7.46092806e-03f, 7.53548881e-03f, 7.61073688e-03f, 7.68667925e-03f, 7.76332151e-03f, 7.84066971e-03f,
    7.91873038e-03f, 7.99750909e-03f, 8.07701331e-03f, 8.15724768e-03f, 8.23822059e-03f, 8.31993762e-03f, 8.40240531e-03f, 8.48563015e-03f,
    8.56961869e-03f, 8.65437835e-03f, 8.73991475e-03f, 8.82623624e-03f, 8.91334750e-03f, 9.00125783e-03f, 9.08997282e-03f, 9.17950086e-03f,
    9.26984660e-03f, 9.36102029e-03f, 9.45302658e-03f, 9.54587478e-03f, 9.63957049e-03f, 9.73412208e-03f, 9.82953794e-03f, 9.92582366e-03f,
    1.00229885e-02f, 1.01210391e-02f, 1.02199847e-02f, 1.03198318e-02f, 1.04205878e-02f, 1.05222631e-02f, 1.06248632e-02f, 1.07283974e-02f,
    1.08328732e-02f, 1.09383008e-02f, 1.10446857e-02f, 1.11520384e-02f, 1.12603670e-02f, 1.13696791e-02f, 1.14799840e-02f, 1.15912911e-02f,
    1.17036076e-02f, 1.18169440e-02f, 1.19313076e-02f, 1.20467097e-02f, 1.21631566e-02f, 1.22806598e-02f, 1.23992264e-02f, 1.25188679e-02f,
    1.26395924e-02f, 1.27614103e-02f, 1.28843300e-02f, 1.30083617e-02f, 1.31335156e-02f, 1.32598002e-02f, 1.33872274e-02f, 1.35158058e-02f,
    1.36455465e-02f, 1.37764579e-02f, 1.39085520e-02f, 1.40418392e-02f, 1.41763287e-02f, 1.43120317e-02f, 1.44489584e-02f, 1.45871202e-02f,
    1.47265280e-02f, 1.48671921e-02f, 1.50091238e-02f, 1.51523352e-02f, 1.52968355e-02f, 1.54426368e-02f, 1.55897513e-02f, 1.57381892e-02f,
    1.58879645e-02f, 1.60390865e-02f, 1.61915664e-02f, 1.63454190e-02f, 1.65006537e-02f, 1.66572854e-02f, 1.68153234e-02f, 1.69747807e-02f,
    1.71356723e-02f, 1.72980074e-02f, 1.74618009e-02f, 1.76270641e-02f, 1.77938100e-02f, 1.79620534e-02f, 1.81318056e-02f, 1.83030814e-02f,
    1.84758920e-02f, 1.86502542e-02f, 1.88261773e-02f, 1.90036781e-02f, 1.91827696e-02f, 1.93634648e-02f, 1.95457805e-02f, 1.97297279e-02f,
    1.99153218e-02f, 2.01025773e-02f, 2.02915110e-02f, 2.04821341e-02f, 2.06744634e-02f, 2.08685137e-02f, 2.10643001e-02f, 2.12618373e-02f,
    2.14611404e-02f, 2.16622259e-02f, 2.18651108e-02f, 2.20698081e-02f, 2.22763345e-02f, 2.24847067e-02f, 2.26949397e-02f, 2.29070522e-02f,
    2.31210571e-02f, 2.33369749e-02f, 2.35548206e-02f, 2.37746127e-02f, 2.39963643e-02f, 2.42200959e-02f, 2.44458262e-02f, 2.46735681e-02f,
    2.49033440e-02f, 2.51351688e-02f, 2.53690612e-02f, 2.56050415e-02f, 2.58431248e-02f, 2.60833316e-02f, 2.63256803e-02f, 2.65701879e-02f,
    2.68168766e-02f, 2.70657632e-02f, 2.73168683e-02f, 2.75702123e-02f, 2.78258119e-02f, 2.80836895e-02f, 2.83438638e-02f, 2.86063571e-02f,
    2.88711879e-02f, 2.91383769e-02f, 2.94079445e-02f, 2.96799131e-02f, 2.99543031e-02f, 3.02311368e-02f, 3.05104330e-02f, 3.07922177e-02f,
    3.10765076e-02f, 3.13633308e-02f, 3.16527039e-02f, 3.19446549e-02f, 3.22392024e-02f, 3.25363688e-02f, 3.28361802e-02f, 3.31386589e-02f,
    3.34438272e-02f, 3.37517112e-02f, 3.40623371e-02f, 3.43757197e-02f, 3.46918926e-02f, 3.50108780e-02f, 3.53326984e-02f, 3.56573798e-02f,
    3.59849520e-02f, 3.63154337e-02f, 3.66488546e-02f, 3.69852409e-02f, 3.73246148e-02f, 3.76670063e-02f, 3.80124450e-02f, 3.83609496e-02f,
    3.87125537e-02f, 3.90672870e-02f, 3.94251682e-02f, 3.97862308e-02f, 4.01505046e-02f, 4.05180156e-02f, 4.08887938e-02f, 4.12628651e-02f,
    4.16402631e-02f, 4.20210138e-02f, 4.24051471e-02f, 4.27926965e-02f, 4.31836881e-02f, 4.35781553e-02f, 4.39761281e-02f, 4.43776362e-02f,
    4.47827131e-02f, 4.51913923e-02f, 4.56037000e-02f, 4.60196733e-02f, 4.64393422e-02f, 4.68627401e-02f, 4.72899042e-02f, 4.77208607e-02f,
    4.81556468e-02f, 4.85942997e-02f, 4.90368493e-02f, 4.94833291e-02f, 4.99337800e-02f, 5.03882356e-02f, 5.08467294e-02f, 5.13092987e-02f,
    5.17759770e-02f, 5.22468053e-02f, 5.27218170e-02f, 5.32010533e-02f, 5.36845475e-02f, 5.41723445e-02f, 5.46644740e-02f, 5.51609769e-02f,
    5.56618981e-02f, 5.61672710e-02f, 5.66771366e-02f, 5.71915358e-02f, 5.77105097e-02f, 5.82340993e-02f, 5.87623455e-02f, 5.92952892e-02f,
    5.98329715e-02f, 6.03754371e-02f, 6.09227270e-02f, 6.14748895e-02f, 6.20319583e-02f, 6.25939891e-02f, 6.31610155e-02f, 6.37330860e-02f,
    6.43102527e-02f, 6.48925528e-02f, 6.54800311e-02f, 6.60727397e-02f, 6.66707307e-02f, 6.72740340e-02f, 6.78827167e-02f, 6.84968159e-02f,
    6.91163763e-02f, 6.97414652e-02f, 7.03721121e-02f, 7.10083768e-02f, 7.16503114e-02f, 7.22979605e-02f, 7.29513839e-02f, 7.36106187e-02f,
    7.42757395e-02f, 7.49467760e-02f, 7.56237954e-02f, 7.63068572e-02f, 7.69959986e-02f, 7.76912868e-02f, 7.83927664e-02f, 7.91005120e-02f,
    7.98145607e-02f, 8.05349797e-02f, 8.12618211e-02f, 8.19951519e-02f, 8.27350169e-02f, 8.34814906e-02f, 8.42346177e-02f, 8.49944726e-02f,
    8.57611075e-02f, 8.65345821e-02f, 8.73149633e-02f, 8.81023109e-02f, 8.88966918e-02f, 8.96981657e-02f, 9.05067995e-02f, 9.13226604e-02f,
    9.21458006e-02f, 9.29763019e-02f, 9.38142315e-02f, 9.46596414e-02f, 9.55126062e-02f, 9.63732079e-02f, 9.72414911e-02f, 9.81175527e-02f,
    9.90014449e-02f, 9.98932421e-02f, 1.00793019e-01f, 1.01700842e-01f, 1.02616794e-01f, 1.03540950e-01f, 1.04473375e-01f, 1.05414145e-01f,
    1.06363349e-01f, 1.07321046e-01f, 1.08287327e-01f, 1.09262258e-01f, 1.10245936e-01f, 1.11238420e-01f, 1.12239808e-01f, 1.13250174e-01f,
    1.14269592e-01f, 1.15298159e-01f, 1.16335951e-01f, 1.17383055e-01f, 1.18439555e-01f, 1.19505540e-01f, 1.20581090e-01f, 1.21666297e-01f,
    1.22761250e-01f, 1.23866037e-01f, 1.24980748e-01f, 1.26105472e-01f, 1.27240300e-01f, 1.28385335e-01f, 1.29540667e-01f, 1.30706385e-01f,
    1.31882578e-01f, 1.33069366e-01f, 1.34266824e-01f, 1.35475054e-01f, 1.36694163e-01f, 1.37924254e-01f, 1.39165416e-01f, 1.40417755e-01f,
    1.41681373e-01f, 1.42956376e-01f, 1.44242868e-01f, 1.45540953e-01f, 1.46850750e-01f, 1.48172364e-01f, 1.49505883e-01f, 1.50851443e-01f,
    1.52209148e-01f, 1.53579101e-01f, 1.54961422e-01f, 1.56356215e-01f, 1.57763630e-01f, 1.59183741e-01f, 1.60616681e-01f, 1.62062585e-01f,
    1.63521558e-01f, 1.64993718e-01f, 1.66479185e-01f, 1.67978108e-01f, 1.69490591e-01f, 1.71016753e-01f, 1.72556743e-01f, 1.74110666e-01f,
    1.75678670e-01f, 1.77260876e-01f, 1.78857431e-01f, 1.80468440e-01f, 1.82094067e-01f, 1.83734432e-01f, 1.85389683e-01f, 1.87059954e-01f,
    1.88745379e-01f, 1.90446094e-01f, 1.92162275e-01f, 1.93894029e-01f, 1.95641518e-01f, 1.97404891e-01f, 1.99184284e-01f, 2.00979874e-01f,
    2.02791780e-01f, 2.04620168e-01f, 2.06465200e-01f, 2.08327025e-01f, 2.10205808e-01f, 2.12101698e-01f, 2.14014858e-01f, 2.15945452e-01f,
    2.17893630e-01f, 2.19859585e-01f, 2.21843481e-01f, 2.23845452e-01f, 2.25865707e-01f, 2.27904394e-01f, 2.29961708e-01f, 2.32037798e-01f,
    2.34132871e-01f, 2.36247078e-01f, 2.38380611e-01f, 2.40533665e-01f, 2.42706403e-01f, 2.44899020e-01f, 2.47111708e-01f, 2.49344662e-01f,
    2.51598060e-01f, 2.53872097e-01f, 2.56166965e-01f, 2.58482873e-01f, 2.60820031e-01f, 2.63178617e-01f, 2.65558839e-01f, 2.67960906e-01f,
    2.70385027e-01f, 2.72831410e-01f, 2.75300294e-01f, 2.77791828e-01f, 2.80306280e-01f, 2.82843858e-01f, 2.85404772e-01f, 2.87989259e-01f,
    2.90597528e-01f, 2.93229818e-01f, 2.95886338e-01f, 2.98567355e-01f, 3.01273078e-01f, 3.04003745e-01f, 3.06759596e-01f, 3.09540868e-01f,
    3.12347800e-01f, 3.15180659e-01f, 3.18039685e-01f, 3.20925146e-01f, 3.23837250e-01f, 3.26776266e-01f, 3.29742491e-01f, 3.32736135e-01f,
    3.35757494e-01f, 3.38806838e-01f, 3.41884404e-01f, 3.44990492e-01f, 3.48125368e-01f, 3.51289302e-01f, 3.54482591e-01f, 3.57705504e-01f,
    3.60958338e-01f, 3.64241362e-01f, 3.67554903e-01f, 3.70899200e-01f, 3.74274611e-01f, 3.77681404e-01f, 3.81119877e-01f, 3.84590358e-01f,
    3.88093114e-01f, 3.91628534e-01f, 3.95196855e-01f, 3.98798466e-01f, 4.02433634e-01f, 4.06102717e-01f, 4.09806043e-01f, 4.13543910e-01f,
    4.17316705e-01f, 4.21124727e-01f, 4.24968332e-01f, 4.28847879e-01f, 4.32763696e-01f, 4.36716169e-01f, 4.40705627e-01f, 4.44732428e-01f,
    4.48796988e-01f, 4.52899605e-01f, 4.57040668e-01f, 4.61220592e-01f, 4.65439737e-01f, 4.69698459e-01f, 4.73997205e-01f, 4.78336304e-01f,
    4.82716203e-01f, 4.87137258e-01f, 4.91599888e-01f, 4.96104538e-01f, 5.00651598e-01f, 5.05241454e-01f, 5.09874523e-01f, 5.14551282e-01f,
    5.19272149e-01f, 5.24037540e-01f, 5.28847933e-01f, 5.33703685e-01f, 5.38605332e-01f, 5.43553352e-01f, 5.48548043e-01f, 5.53590059e-01f,
    5.58679700e-01f, 5.63817561e-01f, 5.69004118e-01f, 5.74239731e-01f, 5.79524994e-01f, 5.84860384e-01f, 5.90246379e-01f, 5.95683515e-01f,
    6.01172268e-01f, 6.06713176e-01f, 6.12306714e-01f, 6.17953420e-01f, 6.23653889e-01f, 6.29408538e-01f, 6.35218024e-01f, 6.41082823e-01f,
    6.47003531e-01f, 6.52980685e-01f, 6.59014821e-01f, 6.65106535e-01f, 6.71256423e-01f, 6.77465081e-01f, 6.83733046e-01f, 6.90060914e-01f,
    6.96449339e-01f, 7.02898860e-01f, 7.09410191e-01f, 7.15983808e-01f, 7.22620487e-01f, 7.29320765e-01f, 7.36085355e-01f, 7.42914796e-01f,
    7.49809861e-01f, 7.56771147e-01f, 7.63799310e-01f, 7.70895064e-01f, 7.78059125e-01f, 7.85292089e-01f, 7.92594731e-01f, 7.99967706e-01f,
    8.07411730e-01f, 8.14927578e-01f, 8.22515965e-01f, 8.30177546e-01f, 8.37913156e-01f, 8.45723510e-01f, 8.53609383e-01f, 8.61571491e-01f,
    8.69610727e-01f, 8.77727747e-01f, 8.85923445e-01f, 8.94198537e-01f, 9.02553856e-01f, 9.10990298e-01f, 9.19508636e-01f, 9.28109646e-01f,
    9.36794281e-01f, 9.45563376e-01f, 9.54417706e-01f, 9.63358283e-01f, 9.72385883e-01f, 9.81501400e-01f, 9.90705848e-01f, 1.00000000e+00f,
    1.0f, // extra padding to avoid out of bounds access
};

static inline float pq_eotf(float x)
{
    float idxf  = fminf(fmaxf(x, 0.0f), 1.0f) * (PQ_LUT_SIZE - 1);
    int ipart   = floorf(idxf);
    float fpart = idxf - ipart;
    return PL_MIX(pq_eotf_lut[ipart], pq_eotf_lut[ipart + 1], fpart);
}

static inline float pq_oetf(float x)
{
    double t = pow(fmax(x, 0.0), PQ_M1);
    t = (PQ_C1 + PQ_C2 * t) / (1.0 + PQ_C3 * t);
    return pow(t, PQ_M2);
}

// Helper struct containing pre-computed cached values describing a gamut
struct gamut {
    pl_matrix3x3 lms2rgb;
    pl_matrix3x3 rgb2lms;
    float min_luma, max_luma;   // pq
    float min_rgb,  max_rgb;    // 10k normalized
    struct ICh *peak_cache;     // 1-item cache for computed peaks (per hue)
};

struct cache {
    struct ICh src_cache;
    struct ICh dst_cache;
};

static void get_gamuts(struct gamut *dst, struct gamut *src, struct cache *cache,
                       const struct pl_gamut_map_params *params)
{
    const float epsilon = 1e-6;
    memset(cache, 0, sizeof(*cache));
    struct gamut base = {
        .min_luma = params->min_luma,
        .max_luma = params->max_luma,
        .min_rgb  = pq_eotf(params->min_luma) - epsilon,
        .max_rgb  = pq_eotf(params->max_luma) + epsilon,
    };

    if (dst) {
        *dst = base;
        dst->lms2rgb = dst->rgb2lms = pl_ipt_rgb2lms(&params->output_gamut);
        dst->peak_cache = &cache->dst_cache;
        pl_matrix3x3_invert(&dst->lms2rgb);
    }

    if (src) {
        *src = base;
        src->lms2rgb = src->rgb2lms = pl_ipt_rgb2lms(&params->input_gamut);
        src->peak_cache = &cache->src_cache;
        pl_matrix3x3_invert(&src->lms2rgb);
    }
}

static inline struct IPT rgb2ipt(struct RGB c, struct gamut gamut)
{
    const float L = gamut.rgb2lms.m[0][0] * c.R +
                    gamut.rgb2lms.m[0][1] * c.G +
                    gamut.rgb2lms.m[0][2] * c.B;
    const float M = gamut.rgb2lms.m[1][0] * c.R +
                    gamut.rgb2lms.m[1][1] * c.G +
                    gamut.rgb2lms.m[1][2] * c.B;
    const float S = gamut.rgb2lms.m[2][0] * c.R +
                    gamut.rgb2lms.m[2][1] * c.G +
                    gamut.rgb2lms.m[2][2] * c.B;
    const float Lp = pq_oetf(L);
    const float Mp = pq_oetf(M);
    const float Sp = pq_oetf(S);
    return (struct IPT) {
        .I = 0.4000f * Lp + 0.4000f * Mp + 0.2000f * Sp,
        .P = 4.4550f * Lp - 4.8510f * Mp + 0.3960f * Sp,
        .T = 0.8056f * Lp + 0.3572f * Mp - 1.1628f * Sp,
    };
}

static inline struct RGB ipt2rgb(struct IPT c, struct gamut gamut)
{
    const float Lp = c.I + 0.097568930514613919f * c.P + 0.2052264331645916f * c.T;
    const float Mp = c.I - 0.11387648547314712f * c.P + 0.13321715836999806f * c.T;
    const float Sp = c.I + 0.032615109917066409f * c.P - 0.67688718306917931f * c.T;
    const float L = pq_eotf(Lp);
    const float M = pq_eotf(Mp);
    const float S = pq_eotf(Sp);
    return (struct RGB) {
        .R = gamut.lms2rgb.m[0][0] * L +
             gamut.lms2rgb.m[0][1] * M +
             gamut.lms2rgb.m[0][2] * S,
        .G = gamut.lms2rgb.m[1][0] * L +
             gamut.lms2rgb.m[1][1] * M +
             gamut.lms2rgb.m[1][2] * S,
        .B = gamut.lms2rgb.m[2][0] * L +
             gamut.lms2rgb.m[2][1] * M +
             gamut.lms2rgb.m[2][2] * S,
    };
}

static inline bool ingamut(struct IPT c, struct gamut gamut)
{
    const float Lp = c.I + 0.097568930514613919f * c.P + 0.2052264331645916f * c.T;
    const float Mp = c.I - 0.11387648547314712f * c.P + 0.13321715836999806f * c.T;
    const float Sp = c.I + 0.032615109917066409f * c.P - 0.67688718306917931f * c.T;
    if (Lp < gamut.min_luma || Lp > gamut.max_luma ||
        Mp < gamut.min_luma || Mp > gamut.max_luma ||
        Sp < gamut.min_luma || Sp > gamut.max_luma)
    {
        // Early exit for values outside legal LMS range
        return false;
    }

    const float L = pq_eotf(Lp);
    const float M = pq_eotf(Mp);
    const float S = pq_eotf(Sp);
    struct RGB rgb = {
        .R = gamut.lms2rgb.m[0][0] * L +
             gamut.lms2rgb.m[0][1] * M +
             gamut.lms2rgb.m[0][2] * S,
        .G = gamut.lms2rgb.m[1][0] * L +
             gamut.lms2rgb.m[1][1] * M +
             gamut.lms2rgb.m[1][2] * S,
        .B = gamut.lms2rgb.m[2][0] * L +
             gamut.lms2rgb.m[2][1] * M +
             gamut.lms2rgb.m[2][2] * S,
    };
    return rgb.R >= gamut.min_rgb && rgb.R <= gamut.max_rgb &&
           rgb.G >= gamut.min_rgb && rgb.G <= gamut.max_rgb &&
           rgb.B >= gamut.min_rgb && rgb.B <= gamut.max_rgb;
}

struct generate_args {
    const struct pl_gamut_map_params *params;
    float *out;
    int start;
    int count;
};

static PL_THREAD_VOID generate(void *priv)
{
    const struct generate_args *args = priv;
    const struct pl_gamut_map_params *params = args->params;

    float *in = args->out;
    const int end = args->start + args->count;
    for (int h = args->start; h < end; h++) {
        for (int C = 0; C < params->lut_size_C; C++) {
            for (int I = 0; I < params->lut_size_I; I++) {
                float Ix = (float) I / (params->lut_size_I - 1);
                float Cx = (float) C / (params->lut_size_C - 1);
                float hx = (float) h / (params->lut_size_h - 1);
                struct IPT ipt = ich2ipt((struct ICh) {
                    .I = PL_MIX(params->min_luma, params->max_luma, Ix),
                    .C = PL_MIX(0.0f, 0.5f, Cx),
                    .h = PL_MIX(-M_PI, M_PI, hx),
                });
                in[0] = ipt.I;
                in[1] = ipt.P;
                in[2] = ipt.T;
                in += params->lut_stride;
            }
        }
    }

    struct pl_gamut_map_params fixed = *params;
    fix_constants(&fixed.constants);
    fixed.lut_size_h = args->count;
    FUN(params).map(args->out, &fixed);
    PL_THREAD_RETURN();
}

void pl_gamut_map_generate(float *out, const struct pl_gamut_map_params *params)
{
    enum { MAX_WORKERS = 32 };
    struct generate_args args[MAX_WORKERS];

    const int num_per_worker = PL_DIV_UP(params->lut_size_h, MAX_WORKERS);
    const int num_workers = PL_DIV_UP(params->lut_size_h, num_per_worker);
    for (int i = 0; i < num_workers; i++) {
        const int start = i * num_per_worker;
        const int count = PL_MIN(num_per_worker, params->lut_size_h - start);
        args[i] = (struct generate_args) {
            .params = params,
            .out    = out,
            .start  = start,
            .count  = count,
        };
        out += count * params->lut_size_C * params->lut_size_I * params->lut_stride;
    }

    pl_thread workers[MAX_WORKERS] = {0};
    for (int i = 0; i < num_workers; i++) {
        if (pl_thread_create(&workers[i], generate, &args[i]) != 0)
            generate(&args[i]); // fallback
    }

    for (int i = 0; i < num_workers; i++) {
        if (!workers[i])
            continue;
        if (pl_thread_join(workers[i]) != 0)
            generate(&args[i]); // fallback
    }
}

void pl_gamut_map_sample(float x[3], const struct pl_gamut_map_params *params)
{
    struct pl_gamut_map_params fixed = *params;
    fix_constants(&fixed.constants);
    fixed.lut_size_I = fixed.lut_size_C = fixed.lut_size_h = 1;
    fixed.lut_stride = 3;

    FUN(params).map(x, &fixed);
}

#define LUT_SIZE(p) (p->lut_size_I * p->lut_size_C * p->lut_size_h * p->lut_stride)
#define FOREACH_LUT(lut, C)                                                     \
    for (struct IPT *_i = (struct IPT *) lut,                                   \
                    *_end = (struct IPT *) (lut + LUT_SIZE(params)),            \
                    C;                                                          \
         _i < _end && ( C = *_i, 1 );                                           \
         *_i = C, _i = (struct IPT *) ((float *) _i + params->lut_stride))

// Something like PL_MIX(base, c, x) but follows an exponential curve, note
// that this can be used to extend 'c' outwards for x > 1
static inline struct ICh mix_exp(struct ICh c, float x, float gamma, float base)
{
    return (struct ICh) {
        .I = base + (c.I - base) * powf(x, gamma),
        .C = c.C * x,
        .h = c.h,
    };
}

// Drop gamma for colors approaching black and achromatic to avoid numerical
// instabilities, and excessive brightness boosting of grain, while also
// strongly boosting gamma for values exceeding the target peak
static inline float scale_gamma(float gamma, struct ICh ich, struct ICh peak,
                                struct gamut gamut)
{
    const float Imin = gamut.min_luma;
    const float Irel = fmaxf((ich.I - Imin) / (peak.I - Imin), 0.0f);
    return gamma * powf(Irel, 3) * fminf(ich.C / peak.C, 1.0f);
}

static const float maxDelta = 5e-5f;

// Find gamut intersection using specified bounds
static inline struct ICh
desat_bounded(float I, float h, float Cmin, float Cmax, struct gamut gamut)
{
    if (I <= gamut.min_luma)
        return (struct ICh) { .I = gamut.min_luma, .C = 0, .h = h };
    if (I >= gamut.max_luma)
        return (struct ICh) { .I = gamut.max_luma, .C = 0, .h = h };

    const float maxDI = I * maxDelta;
    struct ICh res = { .I = I, .C = (Cmin + Cmax) / 2, .h = h };
    do {
        if (ingamut(ich2ipt(res), gamut)) {
            Cmin = res.C;
        } else {
            Cmax = res.C;
        }
        res.C = (Cmin + Cmax) / 2;
    } while (Cmax - Cmin > maxDI);

    return res;
}

// Finds maximally saturated in-gamut color (for given hue)
static inline struct ICh saturate(float hue, struct gamut gamut)
{
    if (gamut.peak_cache->I && fabsf(gamut.peak_cache->h - hue) < 1e-3)
        return *gamut.peak_cache;

    static const float invphi = 0.6180339887498948f;
    static const float invphi2 = 0.38196601125010515f;

    struct ICh lo = { .I = gamut.min_luma, .h = hue };
    struct ICh hi = { .I = gamut.max_luma, .h = hue };
    float de = hi.I - lo.I;
    struct ICh a = { .I = lo.I + invphi2 * de };
    struct ICh b = { .I = lo.I + invphi  * de };
    a = desat_bounded(a.I, hue, 0.0f, 0.5f, gamut);
    b = desat_bounded(b.I, hue, 0.0f, 0.5f, gamut);

    while (de > maxDelta) {
        de *= invphi;
        if (a.C > b.C) {
            hi = b;
            b = a;
            a.I = lo.I + invphi2 * de;
            a = desat_bounded(a.I, hue, lo.C - maxDelta, 0.5f, gamut);
        } else {
            lo = a;
            a = b;
            b.I = lo.I + invphi * de;
            b = desat_bounded(b.I, hue, hi.C - maxDelta, 0.5f, gamut);
        }
    }

    struct ICh peak = a.C > b.C ? a : b;
    *gamut.peak_cache = peak;
    return peak;
}

// Clip a color along the exponential curve given by `gamma`
static inline struct IPT
clip_gamma(struct IPT ipt, float gamma, struct gamut gamut)
{
    if (ipt.I <= gamut.min_luma)
        return (struct IPT) { .I = gamut.min_luma };
    if (ingamut(ipt, gamut))
        return ipt;

    struct ICh ich = ipt2ich(ipt);
    if (!gamma)
        return ich2ipt(desat_bounded(ich.I, ich.h, 0.0f, ich.C, gamut));

    const float maxDI = fmaxf(ich.I * maxDelta, 1e-7f);
    struct ICh peak = saturate(ich.h, gamut);
    gamma = scale_gamma(gamma, ich, peak, gamut);
    float lo = 0.0f, hi = 1.0f, x = 0.5f;
    do {
        struct ICh test = mix_exp(ich, x, gamma, peak.I);
        if (ingamut(ich2ipt(test), gamut)) {
            lo = x;
        } else {
            hi = x;
        }
        x = (lo + hi) / 2.0f;
    } while (hi - lo > maxDI);

    return ich2ipt(mix_exp(ich, x, gamma, peak.I));
}

static float softclip(float value, float source, float target,
                      const struct pl_gamut_map_constants *c)
{
    if (!target)
        return 0.0f;
    const float peak = source / target;
    const float x = fminf(value / target, peak);
    const float j = c->softclip_knee;
    if (x <= j || peak <= 1.0)
        return value;
    // Apply simple mobius function
    const float a = -j*j * (peak - 1.0f) / (j*j - 2.0f * j + peak);
    const float b = (j*j - 2.0f * j * peak + peak) /
                    fmaxf(1e-6f, peak - 1.0f);
    const float scale = (b*b + 2.0f * b*j + j*j) / (b - a);
    return scale * (x + a) / (x + b) * target;
}

static int cmp_float(const void *a, const void *b)
{
    float fa = *(const float*) a;
    float fb = *(const float*) b;
    return PL_CMP(fa, fb);
}

static float wrap(float h)
{
    if (h > M_PI) {
        return h - 2 * M_PI;
    } else if (h < -M_PI) {
        return h + 2 * M_PI;
    } else {
        return h;
    }
}

enum {
    S = 12,    // number of hue shift vertices
    N = S + 2, // +2 for the endpoints
};

// Hue-shift helper struct
struct hueshift {
    float dh[N];
    float dddh[N];
    float K[N];
    float prev_hue;
    float prev_shift;
    struct { float hue, delta; } hueshift[N];
};

static void hueshift_prepare(struct hueshift *s, struct gamut src, struct gamut dst)
{
    const float O = pq_eotf(src.min_luma), X = pq_eotf(src.max_luma);
    const float M = (O + X) / 2.0f;
    const struct RGB refpoints[S] = {
        {X, O, O}, {O, X, O}, {O, O, X},
        {O, X, X}, {X, O, X}, {X, X, O},
        {O, X, M}, {X, O, M}, {X, M, O},
        {O, M, X}, {M, O, X}, {M, X, O},
    };

    memset(s, 0, sizeof(*s));
    for (int i = 0; i < S; i++) {
        struct ICh ich_src = ipt2ich(rgb2ipt(refpoints[i], src));
        struct ICh ich_dst = ipt2ich(rgb2ipt(refpoints[i], dst));
        const float delta = wrap(ich_dst.h - ich_src.h);
        s->hueshift[i+1].hue = ich_src.h;
        s->hueshift[i+1].delta = delta;
    }

    // Sort and wrap endpoints
    qsort(s->hueshift + 1, S, sizeof(*s->hueshift), cmp_float);
    s->hueshift[0]   = s->hueshift[S];
    s->hueshift[S+1] = s->hueshift[1];
    s->hueshift[0].hue   -= 2 * M_PI;
    s->hueshift[S+1].hue += 2 * M_PI;

    // Construction of cubic spline coefficients
    float tmp[N][N] = {0};
    for (int i = N - 1; i > 0; i--) {
        s->dh[i-1] = s->hueshift[i].hue - s->hueshift[i-1].hue;
        s->dddh[i] = (s->hueshift[i].delta - s->hueshift[i-1].delta) / s->dh[i-1];
    }
    for (int i = 1; i < N - 1; i++) {
        tmp[i][i] = 2 * (s->dh[i-1] + s->dh[i]);
        if (i != 1)
            tmp[i][i-1] = tmp[i-1][i] = s->dh[i-1];
        tmp[i][N-1] = 6 * (s->dddh[i+1] - s->dddh[i]);
    }
    for (int i = 1; i < N - 2; i++) {
        const float q = (tmp[i+1][i] / tmp[i][i]);
        for (int j = 1; j <= N - 1; j++)
            tmp[i+1][j] -= q * tmp[i][j];
    }
    for (int i = N - 2; i > 0; i--) {
        float sum = 0.0f;
        for (int j = i; j <= N - 2; j++)
            sum += tmp[i][j] * s->K[j];
        s->K[i] = (tmp[i][N-1] - sum) / tmp[i][i];
    }

    s->prev_hue = -10.0f;
}

static struct ICh hueshift_apply(struct hueshift *s, struct ICh ich)
{
    if (fabsf(ich.h - s->prev_hue) < 1e-6f)
        goto done;

    // Determine perceptual hue shift delta by interpolation of refpoints
    for (int i = 0; i < N - 1; i++) {
        if (s->hueshift[i+1].hue > ich.h) {
            pl_assert(s->hueshift[i].hue <= ich.h);
            float a = (s->K[i+1] - s->K[i]) / (6 * s->dh[i]);
            float b = s->K[i] / 2;
            float c = s->dddh[i+1] - (2 * s->dh[i] * s->K[i] + s->K[i+1] * s->dh[i]) / 6;
            float d = s->hueshift[i].delta;
            float x = ich.h - s->hueshift[i].hue;
            float delta = ((a * x + b) * x + c) * x + d;
            s->prev_shift = ich.h + delta;
            s->prev_hue = ich.h;
            break;
        }
    }

done:
    return (struct ICh) {
        .I = ich.I,
        .C = ich.C,
        .h = s->prev_shift,
    };
}

static void perceptual(float *lut, const struct pl_gamut_map_params *params)
{
    const struct pl_gamut_map_constants *c = &params->constants;
    struct cache cache;
    struct gamut dst, src;
    get_gamuts(&dst, &src, &cache, params);

    FOREACH_LUT(lut, ipt) {
        struct ICh ich = ipt2ich(ipt);
        struct ICh src_peak = saturate(ich.h, src);
        struct ICh dst_peak = saturate(ich.h, dst);
        struct IPT mapped = rgb2ipt(ipt2rgb(ipt, src), dst);

        // Protect in gamut region
        const float maxC = fmaxf(src_peak.C, dst_peak.C);
        float k = pl_smoothstep(c->perceptual_deadzone, 1.0f, ich.C / maxC);
        k *= c->perceptual_strength;
        ipt.I = PL_MIX(ipt.I, mapped.I, k);
        ipt.P = PL_MIX(ipt.P, mapped.P, k);
        ipt.T = PL_MIX(ipt.T, mapped.T, k);

        // Fold over-range channels within the source gamut's extent at this hue
        struct RGB speak = ipt2rgb(ich2ipt(src_peak), dst);
        struct RGB rgb = ipt2rgb(ipt, dst);
        const float mx = fmaxf(rgb.R, fmaxf(rgb.G, rgb.B));
        const float mn = fminf(rgb.R, fminf(rgb.G, rgb.B));
        const float sat = 1.0f - fmaxf(mn, 0.0f) / fmaxf(mx, 1e-6f);
        const float ext = fmaxf(fmaxf(speak.R, fmaxf(speak.G, speak.B)), mx);
        rgb.R = PL_MIX(rgb.R, softclip(rgb.R, ext, dst.max_rgb, c), sat);
        rgb.G = PL_MIX(rgb.G, softclip(rgb.G, ext, dst.max_rgb, c), sat);
        rgb.B = PL_MIX(rgb.B, softclip(rgb.B, ext, dst.max_rgb, c), sat);
        rgb.R = fminf(fmaxf(rgb.R, dst.min_rgb), dst.max_rgb);
        rgb.G = fminf(fmaxf(rgb.G, dst.min_rgb), dst.max_rgb);
        rgb.B = fminf(fmaxf(rgb.B, dst.min_rgb), dst.max_rgb);
        ipt = rgb2ipt(rgb, dst);
    }
}

const struct pl_gamut_map_function pl_gamut_map_perceptual = {
    .name = "perceptual",
    .description = "Perceptual mapping",
    .bidirectional = true,
    .map = perceptual,
};

static void softclip_map(float *lut, const struct pl_gamut_map_params *params)
{
    const struct pl_gamut_map_constants *c = &params->constants;

    // Separate cache after hueshift, because this invalidates previous cache
    struct cache cache_pre, cache_post;
    struct gamut dst_pre, src_pre, src_post, dst_post;
    struct hueshift hueshift;
    get_gamuts(&dst_pre, &src_pre, &cache_pre, params);
    get_gamuts(&dst_post, &src_post, &cache_post, params);
    hueshift_prepare(&hueshift, src_pre, dst_pre);

    FOREACH_LUT(lut, ipt) {
        struct gamut src = src_pre;
        struct gamut dst = dst_pre;

        if (ipt.I <= dst.min_luma) {
            ipt.P = ipt.T = 0.0f;
            continue;
        }

        struct ICh ich = ipt2ich(ipt);
        if (ich.C <= 1e-2f)
            continue; // Fast path for achromatic colors

        float margin = 1.0f;
        struct ICh shifted = hueshift_apply(&hueshift, ich);
        if (fabsf(shifted.h - ich.h) >= 1e-3f) {
            struct ICh src_border = desat_bounded(ich.I, ich.h, 0.0f, 0.5f, src);
            struct ICh dst_border = desat_bounded(ich.I, ich.h, 0.0f, 0.5f, dst);
            const float k = pl_smoothstep(dst_border.C * c->softclip_knee,
                                          src_border.C, ich.C);
            ich.h = PL_MIX(ich.h, shifted.h, k);
            src = src_post;
            dst = dst_post;

            // Expand/contract chromaticity margin to correspond to the altered
            // size of the hue leaf after applying the hue delta
            struct ICh shift_border = desat_bounded(ich.I, ich.h, 0.0f, 0.5f, src);
            margin *= fmaxf(1.0f, src_border.C / shift_border.C);
        }

        // Determine intersections with source and target gamuts, and
        // apply softclip to the chromaticity
        struct ICh source = saturate(ich.h, src);
        struct ICh target = saturate(ich.h, dst);
        struct ICh border = desat_bounded(ich.I, ich.h, 0.0f, target.C, dst);
        const float chromaticity = PL_MIX(target.C, border.C, c->softclip_desat);
        ich.C = softclip(ich.C, margin * source.C, chromaticity, c);

        // Soft-clip the resulting RGB color. This will generally distort
        // hues slightly, but hopefully in an aesthetically pleasing way.
        struct ICh saturated = { ich.I, chromaticity, ich.h };
        struct RGB peak = ipt2rgb(ich2ipt(saturated), dst);
        struct RGB rgb = ipt2rgb(ich2ipt(ich), dst);
        rgb.R = fmaxf(softclip(rgb.R, peak.R, dst.max_rgb, c), dst.min_rgb);
        rgb.G = fmaxf(softclip(rgb.G, peak.G, dst.max_rgb, c), dst.min_rgb);
        rgb.B = fmaxf(softclip(rgb.B, peak.B, dst.max_rgb, c), dst.min_rgb);
        ipt = rgb2ipt(rgb, dst);
    }
}

const struct pl_gamut_map_function pl_gamut_map_softclip = {
    .name = "softclip",
    .description = "Soft clipping",
    .map = softclip_map,
};

static void relative(float *lut, const struct pl_gamut_map_params *params)
{
    const struct pl_gamut_map_constants *c = &params->constants;
    struct cache cache;
    struct gamut dst;
    get_gamuts(&dst, NULL, &cache, params);

    FOREACH_LUT(lut, ipt)
        ipt = clip_gamma(ipt, c->colorimetric_gamma, dst);
}

const struct pl_gamut_map_function pl_gamut_map_relative = {
    .name = "relative",
    .description = "Colorimetric clip",
    .map = relative,
};

static void desaturate(float *lut, const struct pl_gamut_map_params *params)
{
    struct cache cache;
    struct gamut dst;
    get_gamuts(&dst, NULL, &cache, params);

    FOREACH_LUT(lut, ipt)
        ipt = clip_gamma(ipt, 0.0f, dst);
}

const struct pl_gamut_map_function pl_gamut_map_desaturate = {
    .name = "desaturate",
    .description = "Desaturating clip",
    .map = desaturate,
};

static void saturation(float *lut, const struct pl_gamut_map_params *params)
{
    struct cache cache;
    struct gamut dst, src;
    get_gamuts(&dst, &src, &cache, params);

    FOREACH_LUT(lut, ipt)
        ipt = rgb2ipt(ipt2rgb(ipt, src), dst);
}

const struct pl_gamut_map_function pl_gamut_map_saturation = {
    .name = "saturation",
    .description = "Saturation mapping",
    .bidirectional = true,
    .map = saturation,
};

static void absolute(float *lut, const struct pl_gamut_map_params *params)
{
    const struct pl_gamut_map_constants *c = &params->constants;
    struct cache cache;
    struct gamut dst;
    get_gamuts(&dst, NULL, &cache, params);
    pl_matrix3x3 m = pl_get_adaptation_matrix(params->output_gamut.white,
                                              params->input_gamut.white);

    FOREACH_LUT(lut, ipt) {
        struct RGB rgb = ipt2rgb(ipt, dst);
        pl_matrix3x3_apply(&m, (float *) &rgb);
        ipt = rgb2ipt(rgb, dst);
        ipt = clip_gamma(ipt, c->colorimetric_gamma, dst);
    }
}

const struct pl_gamut_map_function pl_gamut_map_absolute = {
    .name = "absolute",
    .description = "Absolute colorimetric clip",
    .map = absolute,
};

static void highlight(float *lut, const struct pl_gamut_map_params *params)
{
    struct cache cache;
    struct gamut dst;
    get_gamuts(&dst, NULL, &cache, params);

    FOREACH_LUT(lut, ipt) {
        if (!ingamut(ipt, dst)) {
            ipt.I = fminf(ipt.I + 0.1f, 1.0f);
            ipt.P = fclampf(-1.2f * ipt.P, -0.5f, 0.5f);
            ipt.T = fclampf(-1.2f * ipt.T, -0.5f, 0.5f);
        }
    }
}

const struct pl_gamut_map_function pl_gamut_map_highlight = {
    .name = "highlight",
    .description = "Highlight out-of-gamut pixels",
    .map = highlight,
};

static void linear(float *lut, const struct pl_gamut_map_params *params)
{
    struct cache cache;
    struct gamut dst, src;
    get_gamuts(&dst, &src, &cache, params);

    float gain = 1.0f;
    for (float hue = -M_PI; hue < M_PI; hue += 0.1f)
        gain = fminf(gain, saturate(hue, dst).C / saturate(hue, src).C);

    FOREACH_LUT(lut, ipt) {
        struct ICh ich = ipt2ich(ipt);
        ich.C *= gain;
        ipt = ich2ipt(ich);
    }
}

const struct pl_gamut_map_function pl_gamut_map_linear = {
    .name = "linear",
    .description = "Linear desaturate",
    .map = linear,
};

static void darken(float *lut, const struct pl_gamut_map_params *params)
{
    const struct pl_gamut_map_constants *c = &params->constants;
    struct cache cache;
    struct gamut dst, src;
    get_gamuts(&dst, &src, &cache, params);

    static const struct RGB points[6] = {
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
        {0, 1, 1}, {1, 0, 1}, {1, 1, 0},
    };

    float gain = 1.0f;
    for (int i = 0; i < PL_ARRAY_SIZE(points); i++) {
        const struct RGB p = ipt2rgb(rgb2ipt(points[i], src), dst);
        const float maxRGB = PL_MAX3(p.R, p.G, p.B);
        gain = fminf(gain, 1.0 / maxRGB);
    }

    FOREACH_LUT(lut, ipt) {
        struct RGB rgb = ipt2rgb(ipt, dst);
        rgb.R *= gain;
        rgb.G *= gain;
        rgb.B *= gain;
        ipt = rgb2ipt(rgb, dst);
        ipt = clip_gamma(ipt, c->colorimetric_gamma, dst);
    }
}

const struct pl_gamut_map_function pl_gamut_map_darken = {
    .name = "darken",
    .description = "Darken and clip",
    .map = darken,
};

static void noop(float *lut, const struct pl_gamut_map_params *params)
{
    return;
}

const struct pl_gamut_map_function pl_gamut_map_clip = {
    .name = "clip",
    .description = "No gamut mapping (hard clip)",
    .map = noop,
};

const struct pl_gamut_map_function * const pl_gamut_map_functions[] = {
    &pl_gamut_map_clip,
    &pl_gamut_map_perceptual,
    &pl_gamut_map_softclip,
    &pl_gamut_map_relative,
    &pl_gamut_map_saturation,
    &pl_gamut_map_absolute,
    &pl_gamut_map_desaturate,
    &pl_gamut_map_darken,
    &pl_gamut_map_highlight,
    &pl_gamut_map_linear,
    NULL
};

const int pl_num_gamut_map_functions = PL_ARRAY_SIZE(pl_gamut_map_functions) - 1;

const struct pl_gamut_map_function *pl_find_gamut_map_function(const char *name)
{
    for (int i = 0; i < pl_num_gamut_map_functions; i++) {
        if (strcmp(name, pl_gamut_map_functions[i]->name) == 0)
            return pl_gamut_map_functions[i];
    }

    return NULL;
}
