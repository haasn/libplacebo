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
#include "pl_thread.h"

#include <libplacebo/gamut_mapping.h>

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

static const float PQ_M1 = 2610./4096 * 1./4,
                   PQ_M2 = 2523./4096 * 128,
                   PQ_C1 = 3424./4096,
                   PQ_C2 = 2413./4096 * 32,
                   PQ_C3 = 2392./4096 * 32;

enum { PQ_LUT_SIZE = 1024 };
static const float pq_eotf_lut[1024+1] = {
    0.0000000e+00f, 4.0422718e-09f, 1.3111372e-08f, 2.6236826e-08f, 4.3151495e-08f, 6.3746885e-08f, 8.7982383e-08f, 1.1585362e-07f,
    1.4737819e-07f, 1.8258818e-07f, 2.2152586e-07f, 2.6424098e-07f, 3.1078907e-07f, 3.6123021e-07f, 4.1562821e-07f, 4.7405001e-07f,
    5.3656521e-07f, 6.0324583e-07f, 6.7416568e-07f, 7.4940095e-07f, 8.2902897e-07f, 9.1312924e-07f, 1.0017822e-06f, 1.0950702e-06f,
    1.1930764e-06f, 1.2958861e-06f, 1.4035847e-06f, 1.5162600e-06f, 1.6340000e-06f, 1.7568948e-06f, 1.8850346e-06f, 2.0185119e-06f,
    2.1574192e-06f, 2.3018509e-06f, 2.4519029e-06f, 2.6076704e-06f, 2.7692516e-06f, 2.9367449e-06f, 3.1102509e-06f, 3.2898690e-06f,
    3.4757019e-06f, 3.6678526e-06f, 3.8664261e-06f, 4.0715262e-06f, 4.2832601e-06f, 4.5017354e-06f, 4.7270617e-06f, 4.9593473e-06f,
    5.1987040e-06f, 5.4452441e-06f, 5.6990819e-06f, 5.9603301e-06f, 6.2291055e-06f, 6.5055251e-06f, 6.7897080e-06f, 7.0817717e-06f,
    7.3818379e-06f, 7.6900283e-06f, 8.0064675e-06f, 8.3312774e-06f, 8.6645849e-06f, 9.0065169e-06f, 9.3572031e-06f, 9.7167704e-06f,
    1.0085351e-05f, 1.0463077e-05f, 1.0850082e-05f, 1.1246501e-05f, 1.1652473e-05f, 1.2068130e-05f, 1.2493614e-05f, 1.2929066e-05f,
    1.3374626e-05f, 1.3830439e-05f, 1.4296648e-05f, 1.4773401e-05f, 1.5260848e-05f, 1.5759132e-05f, 1.6268405e-05f, 1.6788821e-05f,
    1.7320534e-05f, 1.7863697e-05f, 1.8418467e-05f, 1.8985004e-05f, 1.9563470e-05f, 2.0154019e-05f, 2.0756818e-05f, 2.1372031e-05f,
    2.1999824e-05f, 2.2640365e-05f, 2.3293824e-05f, 2.3960372e-05f, 2.4640186e-05f, 2.5333431e-05f, 2.6040288e-05f, 2.6760935e-05f,
    2.7495552e-05f, 2.8244319e-05f, 2.9007421e-05f, 2.9785041e-05f, 3.0577373e-05f, 3.1384594e-05f, 3.2206899e-05f, 3.3044481e-05f,
    3.3897533e-05f, 3.4766253e-05f, 3.5650838e-05f, 3.6551487e-05f, 3.7468409e-05f, 3.8401794e-05f, 3.9351855e-05f, 4.0318799e-05f,
    4.1302836e-05f, 4.2304177e-05f, 4.3323036e-05f, 4.4359629e-05f, 4.5414181e-05f, 4.6486897e-05f, 4.7578006e-05f, 4.8687732e-05f,
    4.9816302e-05f, 5.0963944e-05f, 5.2130889e-05f, 5.3317369e-05f, 5.4523628e-05f, 5.5749886e-05f, 5.6996391e-05f, 5.8263384e-05f,
    5.9551111e-05f, 6.0859816e-05f, 6.2189750e-05f, 6.3541162e-05f, 6.4914307e-05f, 6.6309439e-05f, 6.7726819e-05f, 6.9166705e-05f,
    7.0629384e-05f, 7.2115077e-05f, 7.3624074e-05f, 7.5156646e-05f, 7.6713065e-05f, 7.8293608e-05f, 7.9898553e-05f, 8.1528181e-05f,
    8.3182776e-05f, 8.4862623e-05f, 8.6568012e-05f, 8.8299235e-05f, 9.0056585e-05f, 9.1840360e-05f, 9.3650860e-05f, 9.5488388e-05f,
    9.7353277e-05f, 9.9245779e-05f, 1.0116623e-04f, 1.0311496e-04f, 1.0509226e-04f, 1.0709847e-04f, 1.0913391e-04f, 1.1119889e-04f,
    1.1329376e-04f, 1.1541885e-04f, 1.1757448e-04f, 1.1976100e-04f, 1.2197875e-04f, 1.2422807e-04f, 1.2650931e-04f, 1.2882282e-04f,
    1.3116900e-04f, 1.3354812e-04f, 1.3596059e-04f, 1.3840676e-04f, 1.4088701e-04f, 1.4340170e-04f, 1.4595121e-04f, 1.4853593e-04f,
    1.5115622e-04f, 1.5381247e-04f, 1.5650507e-04f, 1.5923442e-04f, 1.6200090e-04f, 1.6480492e-04f, 1.6764688e-04f, 1.7052718e-04f,
    1.7344629e-04f, 1.7640451e-04f, 1.7940233e-04f, 1.8244015e-04f, 1.8551840e-04f, 1.8863752e-04f, 1.9179792e-04f, 1.9500006e-04f,
    1.9824437e-04f, 2.0153130e-04f, 2.0486129e-04f, 2.0823479e-04f, 2.1165227e-04f, 2.1511419e-04f, 2.1862101e-04f, 2.2217319e-04f,
    2.2577128e-04f, 2.2941563e-04f, 2.3310679e-04f, 2.3684523e-04f, 2.4063146e-04f, 2.4446597e-04f, 2.4834925e-04f, 2.5228182e-04f,
    2.5626417e-04f, 2.6029683e-04f, 2.6438031e-04f, 2.6851514e-04f, 2.7270184e-04f, 2.7694094e-04f, 2.8123299e-04f, 2.8557852e-04f,
    2.8997815e-04f, 2.9443230e-04f, 2.9894159e-04f, 3.0350657e-04f, 3.0812783e-04f, 3.1280593e-04f, 3.1754144e-04f, 3.2233495e-04f,
    3.2718705e-04f, 3.3209833e-04f, 3.3706938e-04f, 3.4210082e-04f, 3.4719324e-04f, 3.5234727e-04f, 3.5756351e-04f, 3.6284261e-04f,
    3.6818526e-04f, 3.7359195e-04f, 3.7906340e-04f, 3.8460024e-04f, 3.9020315e-04f, 3.9587277e-04f, 4.0160977e-04f, 4.0741483e-04f,
    4.1328861e-04f, 4.1923181e-04f, 4.2524511e-04f, 4.3132921e-04f, 4.3748480e-04f, 4.4371260e-04f, 4.5001332e-04f, 4.5638768e-04f,
    4.6283650e-04f, 4.6936032e-04f, 4.7595999e-04f, 4.8263624e-04f, 4.8938982e-04f, 4.9622151e-04f, 5.0313205e-04f, 5.1012223e-04f,
    5.1719283e-04f, 5.2434463e-04f, 5.3157843e-04f, 5.3889502e-04f, 5.4629521e-04f, 5.5377982e-04f, 5.6134968e-04f, 5.6900560e-04f,
    5.7674843e-04f, 5.8457900e-04f, 5.9249818e-04f, 6.0050682e-04f, 6.0860578e-04f, 6.1679595e-04f, 6.2507819e-04f, 6.3345341e-04f,
    6.4192275e-04f, 6.5048661e-04f, 6.5914616e-04f, 6.6790231e-04f, 6.7675600e-04f, 6.8570816e-04f, 6.9475975e-04f, 7.0391171e-04f,
    7.1316500e-04f, 7.2252060e-04f, 7.3197948e-04f, 7.4154264e-04f, 7.5121107e-04f, 7.6098577e-04f, 7.7086777e-04f, 7.8085807e-04f,
    7.9095772e-04f, 8.0116775e-04f, 8.1148922e-04f, 8.2192318e-04f, 8.3247071e-04f, 8.4313287e-04f, 8.5391076e-04f, 8.6480548e-04f,
    8.7581812e-04f, 8.8694982e-04f, 8.9820168e-04f, 9.0957485e-04f, 9.2107048e-04f, 9.3268971e-04f, 9.4443372e-04f, 9.5630368e-04f,
    9.6830115e-04f, 9.8042658e-04f, 9.9268155e-04f, 1.0050673e-03f, 1.0175850e-03f, 1.0302359e-03f, 1.0430213e-03f, 1.0559425e-03f,
    1.0690006e-03f, 1.0821970e-03f, 1.0955331e-03f, 1.1090100e-03f, 1.1226290e-03f, 1.1363917e-03f, 1.1502992e-03f, 1.1643529e-03f,
    1.1785542e-03f, 1.1929044e-03f, 1.2074050e-03f, 1.2220573e-03f, 1.2368628e-03f, 1.2518229e-03f, 1.2669390e-03f, 1.2822125e-03f,
    1.2976449e-03f, 1.3132377e-03f, 1.3289925e-03f, 1.3449105e-03f, 1.3609935e-03f, 1.3772429e-03f, 1.3936602e-03f, 1.4102470e-03f,
    1.4270054e-03f, 1.4439360e-03f, 1.4610407e-03f, 1.4783214e-03f, 1.4957794e-03f, 1.5134166e-03f, 1.5312345e-03f, 1.5492348e-03f,
    1.5674192e-03f, 1.5857894e-03f, 1.6043471e-03f, 1.6230939e-03f, 1.6420317e-03f, 1.6611622e-03f, 1.6804871e-03f, 1.7000083e-03f,
    1.7197275e-03f, 1.7396465e-03f, 1.7597672e-03f, 1.7800914e-03f, 1.8006210e-03f, 1.8213578e-03f, 1.8423038e-03f, 1.8634608e-03f,
    1.8848308e-03f, 1.9064157e-03f, 1.9282175e-03f, 1.9502381e-03f, 1.9724796e-03f, 1.9949439e-03f, 2.0176331e-03f, 2.0405492e-03f,
    2.0636950e-03f, 2.0870711e-03f, 2.1106805e-03f, 2.1345250e-03f, 2.1586071e-03f, 2.1829286e-03f, 2.2074919e-03f, 2.2322992e-03f,
    2.2573525e-03f, 2.2826542e-03f, 2.3082066e-03f, 2.3340118e-03f, 2.3600721e-03f, 2.3863900e-03f, 2.4129676e-03f, 2.4398074e-03f,
    2.4669117e-03f, 2.4942828e-03f, 2.5219233e-03f, 2.5498355e-03f, 2.5780219e-03f, 2.6064849e-03f, 2.6352271e-03f, 2.6642509e-03f,
    2.6935589e-03f, 2.7231536e-03f, 2.7530377e-03f, 2.7832137e-03f, 2.8136843e-03f, 2.8444520e-03f, 2.8755196e-03f, 2.9068898e-03f,
    2.9385662e-03f, 2.9705496e-03f, 3.0028439e-03f, 3.0354517e-03f, 3.0683758e-03f, 3.1016192e-03f, 3.1351846e-03f, 3.1690750e-03f,
    3.2032932e-03f, 3.2378422e-03f, 3.2727250e-03f, 3.3079445e-03f, 3.3435038e-03f, 3.3794058e-03f, 3.4156537e-03f, 3.4522505e-03f,
    3.4891993e-03f, 3.5265034e-03f, 3.5641658e-03f, 3.6021897e-03f, 3.6405785e-03f, 3.6793353e-03f, 3.7184634e-03f, 3.7579661e-03f,
    3.7978468e-03f, 3.8381088e-03f, 3.8787555e-03f, 3.9197904e-03f, 3.9612169e-03f, 4.0030385e-03f, 4.0452587e-03f, 4.0878810e-03f,
    4.1309104e-03f, 4.1743478e-03f, 4.2181981e-03f, 4.2624651e-03f, 4.3071525e-03f, 4.3522639e-03f, 4.3978031e-03f, 4.4437739e-03f,
    4.4901803e-03f, 4.5370259e-03f, 4.5843148e-03f, 4.6320508e-03f, 4.6802379e-03f, 4.7288801e-03f, 4.7779815e-03f, 4.8275461e-03f,
    4.8775780e-03f, 4.9280813e-03f, 4.9790603e-03f, 5.0305191e-03f, 5.0824620e-03f, 5.1348933e-03f, 5.1878172e-03f, 5.2412382e-03f,
    5.2951607e-03f, 5.3495890e-03f, 5.4045276e-03f, 5.4599811e-03f, 5.5159540e-03f, 5.5724510e-03f, 5.6294765e-03f, 5.6870353e-03f,
    5.7451339e-03f, 5.8037735e-03f, 5.8629606e-03f, 5.9227001e-03f, 5.9829968e-03f, 6.0438557e-03f, 6.1052818e-03f, 6.1672799e-03f,
    6.2298552e-03f, 6.2930128e-03f, 6.3567578e-03f, 6.4210953e-03f, 6.4860306e-03f, 6.5515690e-03f, 6.6177157e-03f, 6.6844762e-03f,
    6.7518558e-03f, 6.8198599e-03f, 6.8884942e-03f, 6.9577641e-03f, 7.0276752e-03f, 7.0982332e-03f, 7.1694438e-03f, 7.2413127e-03f,
    7.3138457e-03f, 7.3870486e-03f, 7.4609273e-03f, 7.5354878e-03f, 7.6107361e-03f, 7.6866782e-03f, 7.7633203e-03f, 7.8406684e-03f,
    7.9187312e-03f, 7.9975101e-03f, 8.0770139e-03f, 8.1572490e-03f, 8.2382216e-03f, 8.3199385e-03f, 8.4024059e-03f, 8.4856307e-03f,
    8.5696193e-03f, 8.6543786e-03f, 8.7399153e-03f, 8.8262362e-03f, 8.9133482e-03f, 9.0012582e-03f, 9.0899733e-03f, 9.1795005e-03f,
    9.2698470e-03f, 9.3610199e-03f, 9.4530265e-03f, 9.5458741e-03f, 9.6395701e-03f, 9.7341219e-03f, 9.8295370e-03f, 9.9258231e-03f,
    1.0022988e-02f, 1.0121039e-02f, 1.0219984e-02f, 1.0319830e-02f, 1.0420587e-02f, 1.0522261e-02f, 1.0624862e-02f, 1.0728396e-02f,
    1.0832872e-02f, 1.0938299e-02f, 1.1044684e-02f, 1.1152036e-02f, 1.1260365e-02f, 1.1369677e-02f, 1.1479982e-02f, 1.1591288e-02f,
    1.1703605e-02f, 1.1816941e-02f, 1.1931305e-02f, 1.2046706e-02f, 1.2163153e-02f, 1.2280656e-02f, 1.2399223e-02f, 1.2518864e-02f,
    1.2639596e-02f, 1.2761413e-02f, 1.2884333e-02f, 1.3008365e-02f, 1.3133519e-02f, 1.3259804e-02f, 1.3387231e-02f, 1.3515809e-02f,
    1.3645549e-02f, 1.3776461e-02f, 1.3908555e-02f, 1.4041841e-02f, 1.4176331e-02f, 1.4312034e-02f, 1.4448961e-02f, 1.4587123e-02f,
    1.4726530e-02f, 1.4867194e-02f, 1.5009126e-02f, 1.5152336e-02f, 1.5296837e-02f, 1.5442638e-02f, 1.5589753e-02f, 1.5738191e-02f,
    1.5887965e-02f, 1.6039087e-02f, 1.6191567e-02f, 1.6345419e-02f, 1.6500655e-02f, 1.6657285e-02f, 1.6815323e-02f, 1.6974781e-02f,
    1.7135672e-02f, 1.7298007e-02f, 1.7461800e-02f, 1.7627063e-02f, 1.7793810e-02f, 1.7962053e-02f, 1.8131805e-02f, 1.8303080e-02f,
    1.8475891e-02f, 1.8650252e-02f, 1.8826176e-02f, 1.9003676e-02f, 1.9182767e-02f, 1.9363463e-02f, 1.9545777e-02f, 1.9729724e-02f,
    1.9915319e-02f, 2.0102575e-02f, 2.0291507e-02f, 2.0482131e-02f, 2.0674460e-02f, 2.0868510e-02f, 2.1064296e-02f, 2.1261833e-02f,
    2.1461136e-02f, 2.1662222e-02f, 2.1865105e-02f, 2.2069802e-02f, 2.2276328e-02f, 2.2484700e-02f, 2.2694934e-02f, 2.2907045e-02f,
    2.3121064e-02f, 2.3336982e-02f, 2.3554827e-02f, 2.3774618e-02f, 2.3996370e-02f, 2.4220102e-02f, 2.4445831e-02f, 2.4673574e-02f,
    2.4903349e-02f, 2.5135174e-02f, 2.5369067e-02f, 2.5605046e-02f, 2.5843129e-02f, 2.6083336e-02f, 2.6325684e-02f, 2.6570192e-02f,
    2.6816880e-02f, 2.7065767e-02f, 2.7316872e-02f, 2.7570215e-02f, 2.7825815e-02f, 2.8083692e-02f, 2.8343867e-02f, 2.8606359e-02f,
    2.8871189e-02f, 2.9138378e-02f, 2.9407946e-02f, 2.9679914e-02f, 2.9954304e-02f, 3.0231137e-02f, 3.0510434e-02f, 3.0792217e-02f,
    3.1076508e-02f, 3.1363330e-02f, 3.1652704e-02f, 3.1944653e-02f, 3.2239199e-02f, 3.2536367e-02f, 3.2836178e-02f, 3.3138657e-02f,
    3.3443826e-02f, 3.3751710e-02f, 3.4062333e-02f, 3.4375718e-02f, 3.4691890e-02f, 3.5010874e-02f, 3.5332694e-02f, 3.5657377e-02f,
    3.5984946e-02f, 3.6315428e-02f, 3.6648848e-02f, 3.6985233e-02f, 3.7324608e-02f, 3.7667000e-02f, 3.8012436e-02f, 3.8360942e-02f,
    3.8712547e-02f, 3.9067276e-02f, 3.9425159e-02f, 3.9786223e-02f, 4.0150496e-02f, 4.0518006e-02f, 4.0888783e-02f, 4.1262855e-02f,
    4.1640274e-02f, 4.2021025e-02f, 4.2405159e-02f, 4.2792707e-02f, 4.3183699e-02f, 4.3578166e-02f, 4.3976138e-02f, 4.4377647e-02f,
    4.4782724e-02f, 4.5191401e-02f, 4.5603709e-02f, 4.6019681e-02f, 4.6439350e-02f, 4.6862749e-02f, 4.7289910e-02f, 4.7720867e-02f,
    4.8155654e-02f, 4.8594305e-02f, 4.9036854e-02f, 4.9483336e-02f, 4.9933787e-02f, 5.0388240e-02f, 5.0846733e-02f, 5.1309301e-02f,
    5.1775981e-02f, 5.2246808e-02f, 5.2721821e-02f, 5.3201056e-02f, 5.3684551e-02f, 5.4172344e-02f, 5.4664473e-02f, 5.5160978e-02f,
    5.5661897e-02f, 5.6167269e-02f, 5.6677135e-02f, 5.7191535e-02f, 5.7710508e-02f, 5.8234097e-02f, 5.8762342e-02f, 5.9295285e-02f,
    5.9832968e-02f, 6.0375433e-02f, 6.0922723e-02f, 6.1474882e-02f, 6.2031952e-02f, 6.2593979e-02f, 6.3161006e-02f, 6.3733078e-02f,
    6.4310241e-02f, 6.4892540e-02f, 6.5480021e-02f, 6.6072730e-02f, 6.6670715e-02f, 6.7274023e-02f, 6.7882702e-02f, 6.8496800e-02f,
    6.9116365e-02f, 6.9741447e-02f, 7.0372096e-02f, 7.1008361e-02f, 7.1650293e-02f, 7.2297942e-02f, 7.2951361e-02f, 7.3610602e-02f,
    7.4275756e-02f, 7.4946797e-02f, 7.5623818e-02f, 7.6306873e-02f, 7.6996016e-02f, 7.7691302e-02f, 7.8392787e-02f, 7.9100526e-02f,
    7.9814576e-02f, 8.0534993e-02f, 8.1261837e-02f, 8.1995163e-02f, 8.2735032e-02f, 8.3481501e-02f, 8.4234632e-02f, 8.4994483e-02f,
    8.5761116e-02f, 8.6534592e-02f, 8.7314974e-02f, 8.8102323e-02f, 8.8896702e-02f, 8.9698176e-02f, 9.0506809e-02f, 9.1322665e-02f,
    9.2145810e-02f, 9.2976310e-02f, 9.3814232e-02f, 9.4659643e-02f, 9.5512612e-02f, 9.6373206e-02f, 9.7241496e-02f, 9.8117550e-02f,
    9.9001441e-02f, 9.9893238e-02f, 1.0079301e-01f, 1.0170084e-01f, 1.0261679e-01f, 1.0354094e-01f, 1.0447337e-01f, 1.0541414e-01f,
    1.0636334e-01f, 1.0732104e-01f, 1.0828731e-01f, 1.0926225e-01f, 1.1024592e-01f, 1.1123841e-01f, 1.1223979e-01f, 1.1325016e-01f,
    1.1426958e-01f, 1.1529814e-01f, 1.1633594e-01f, 1.1738304e-01f, 1.1843954e-01f, 1.1950552e-01f, 1.2058107e-01f, 1.2166627e-01f,
    1.2276122e-01f, 1.2386601e-01f, 1.2498072e-01f, 1.2610544e-01f, 1.2724027e-01f, 1.2838531e-01f, 1.2954063e-01f, 1.3070635e-01f,
    1.3188262e-01f, 1.3306940e-01f, 1.3426686e-01f, 1.3547509e-01f, 1.3669420e-01f, 1.3792428e-01f, 1.3916544e-01f, 1.4041778e-01f,
    1.4168140e-01f, 1.4295640e-01f, 1.4424289e-01f, 1.4554098e-01f, 1.4685078e-01f, 1.4817238e-01f, 1.4950591e-01f, 1.5085147e-01f,
    1.5220916e-01f, 1.5357912e-01f, 1.5496144e-01f, 1.5635624e-01f, 1.5776364e-01f, 1.5918375e-01f, 1.6061670e-01f, 1.6206260e-01f,
    1.6352156e-01f, 1.6499372e-01f, 1.6647920e-01f, 1.6797811e-01f, 1.6949059e-01f, 1.7101676e-01f, 1.7255674e-01f, 1.7411067e-01f,
    1.7567867e-01f, 1.7726087e-01f, 1.7885742e-01f, 1.8046844e-01f, 1.8209406e-01f, 1.8373443e-01f, 1.8538967e-01f, 1.8705994e-01f,
    1.8874536e-01f, 1.9044608e-01f, 1.9216225e-01f, 1.9389401e-01f, 1.9564150e-01f, 1.9740486e-01f, 1.9918426e-01f, 2.0097984e-01f,
    2.0279175e-01f, 2.0462014e-01f, 2.0646517e-01f, 2.0832699e-01f, 2.1020577e-01f, 2.1210165e-01f, 2.1401481e-01f, 2.1594540e-01f,
    2.1789359e-01f, 2.1985954e-01f, 2.2184342e-01f, 2.2384540e-01f, 2.2586565e-01f, 2.2790434e-01f, 2.2996165e-01f, 2.3203774e-01f,
    2.3413293e-01f, 2.3624714e-01f, 2.3838068e-01f, 2.4053372e-01f, 2.4270646e-01f, 2.4489908e-01f, 2.4711177e-01f, 2.4934471e-01f,
    2.5159811e-01f, 2.5387214e-01f, 2.5616702e-01f, 2.5848293e-01f, 2.6082007e-01f, 2.6317866e-01f, 2.6555888e-01f, 2.6796095e-01f,
    2.7038507e-01f, 2.7283145e-01f, 2.7530031e-01f, 2.7779186e-01f, 2.8030631e-01f, 2.8284388e-01f, 2.8540479e-01f, 2.8798927e-01f,
    2.9059754e-01f, 2.9322983e-01f, 2.9588635e-01f, 2.9856736e-01f, 3.0127308e-01f, 3.0400374e-01f, 3.0675959e-01f, 3.0954086e-01f,
    3.1234780e-01f, 3.1518066e-01f, 3.1803969e-01f, 3.2092512e-01f, 3.2383723e-01f, 3.2677625e-01f, 3.2974246e-01f, 3.3273611e-01f,
    3.3575747e-01f, 3.3880680e-01f, 3.4188437e-01f, 3.4499045e-01f, 3.4812533e-01f, 3.5128926e-01f, 3.5448255e-01f, 3.5770546e-01f,
    3.6095828e-01f, 3.6424131e-01f, 3.6755483e-01f, 3.7089914e-01f, 3.7427454e-01f, 3.7768132e-01f, 3.8111979e-01f, 3.8459027e-01f,
    3.8809304e-01f, 3.9162844e-01f, 3.9519678e-01f, 3.9879837e-01f, 4.0243354e-01f, 4.0610261e-01f, 4.0980592e-01f, 4.1354380e-01f,
    4.1731681e-01f, 4.2112483e-01f, 4.2496844e-01f, 4.2884798e-01f, 4.3276381e-01f, 4.3671627e-01f, 4.4070572e-01f, 4.4473253e-01f,
    4.4879706e-01f, 4.5289968e-01f, 4.5704076e-01f, 4.6122068e-01f, 4.6543981e-01f, 4.6969854e-01f, 4.7399727e-01f, 4.7833637e-01f,
    4.8271625e-01f, 4.8713731e-01f, 4.9159995e-01f, 4.9610458e-01f, 5.0065162e-01f, 5.0524147e-01f, 5.0987457e-01f, 5.1455133e-01f,
    5.1927219e-01f, 5.2403759e-01f, 5.2884795e-01f, 5.3370373e-01f, 5.3860537e-01f, 5.4355333e-01f, 5.4854807e-01f, 5.5359004e-01f,
    5.5867972e-01f, 5.6381757e-01f, 5.6900408e-01f, 5.7423972e-01f, 5.7952499e-01f, 5.8486037e-01f, 5.9024637e-01f, 5.9568349e-01f,
    6.0117223e-01f, 6.0671311e-01f, 6.1230664e-01f, 6.1795336e-01f, 6.2365379e-01f, 6.2940847e-01f, 6.3521793e-01f, 6.4108273e-01f,
    6.4700342e-01f, 6.5298056e-01f, 6.5901471e-01f, 6.6510643e-01f, 6.7125632e-01f, 6.7746495e-01f, 6.8373290e-01f, 6.9006078e-01f,
    6.9644918e-01f, 7.0289872e-01f, 7.0941001e-01f, 7.1598366e-01f, 7.2262031e-01f, 7.2932059e-01f, 7.3608513e-01f, 7.4291460e-01f,
    7.4981006e-01f, 7.5677134e-01f, 7.6379952e-01f, 7.7089527e-01f, 7.7805929e-01f, 7.8529226e-01f, 7.9259489e-01f, 7.9996786e-01f,
    8.0741191e-01f, 8.1492774e-01f, 8.2251609e-01f, 8.3017769e-01f, 8.3791329e-01f, 8.4572364e-01f, 8.5360950e-01f, 8.6157163e-01f,
    8.6961082e-01f, 8.7772786e-01f, 8.8592352e-01f, 8.9419862e-01f, 9.0255397e-01f, 9.1099038e-01f, 9.1950869e-01f, 9.2810973e-01f,
    9.3679435e-01f, 9.4556340e-01f, 9.5441776e-01f, 9.6335829e-01f, 9.7238588e-01f, 9.8150143e-01f, 9.9070583e-01f, 1.0000000e+00f,
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
    x = powf(fmaxf(x, 0.0f), PQ_M1);
    x = (PQ_C1 + PQ_C2 * x) / (1.0f + PQ_C3 * x);
    return powf(x, PQ_M2);
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
    const float Lp = c.I + 0.0975689f * c.P + 0.205226f * c.T;
    const float Mp = c.I - 0.1138760f * c.P + 0.133217f * c.T;
    const float Sp = c.I + 0.0326151f * c.P - 0.676887f * c.T;
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
    const float Lp = c.I + 0.0975689f * c.P + 0.205226f * c.T;
    const float Mp = c.I - 0.1138760f * c.P + 0.133217f * c.T;
    const float Sp = c.I + 0.0326151f * c.P - 0.676887f * c.T;
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

static const float perceptual_gamma    = 1.80f;
static const float perceptual_knee     = 0.70f;

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

static void perceptual(float *lut, const struct pl_gamut_map_params *params)
{
    // Separate cache after hueshift, because this invalidates previous cache
    struct cache cache_pre, cache_post;
    struct gamut dst_pre, src_pre, src_post, dst_post;
    get_gamuts(&dst_pre, &src_pre, &cache_pre, params);
    get_gamuts(&dst_post, &src_post, &cache_post, params);

    const float O = pq_eotf(params->min_luma), X = pq_eotf(params->max_luma);
    const float M = (O + X) / 2.0f;
    const struct RGB refpoints[] = {
        {X, O, O}, {O, X, O}, {O, O, X},
        {O, X, X}, {X, O, X}, {X, X, O},
        {O, X, M}, {X, O, M}, {X, M, O},
        {O, M, X}, {M, O, X}, {M, X, O},
    };

    enum {
        S = PL_ARRAY_SIZE(refpoints),
        N = S + 2, // +2 for the endpoints
    };

    bool disable_hueshift = false;
    struct { float hue, delta; } hueshift[N];
    for (int i = 0; i < S; i++) {
        struct ICh ich_src = ipt2ich(rgb2ipt(refpoints[i], src_pre));
        struct ICh ich_dst = ipt2ich(rgb2ipt(refpoints[i], dst_pre));
        float delta = wrap(ich_dst.h - ich_src.h);
        if (fabsf(delta) > 1.0f) {
            // Disable hue-shifting becuase one hue vector is rotated too far,
            // probably as the result of this being some sort of synthetic / fake
            // "test" image - preserve hues in this case
            disable_hueshift = true;
            goto hueshift_done;
        }
        hueshift[i+1].hue = ich_src.h;
        hueshift[i+1].delta = delta;
    }

    // Sort and wrap endpoints
    qsort(hueshift + 1, S, sizeof(*hueshift), cmp_float);
    hueshift[0]   = hueshift[S];
    hueshift[S+1] = hueshift[1];
    hueshift[0].hue   -= 2 * M_PI;
    hueshift[S+1].hue += 2 * M_PI;

    // Construction of cubic spline coefficients
    float dh[N], dddh[N], K[N] = {0}, tmp[N][N] = {0};
    for (int i = N - 1; i > 0; i--) {
        dh[i-1] = hueshift[i].hue - hueshift[i-1].hue;
        dddh[i] = (hueshift[i].delta - hueshift[i-1].delta) / dh[i-1];
    }
    for (int i = 1; i < N - 1; i++) {
        tmp[i][i] = 2 * (dh[i-1] + dh[i]);
        if (i != 1)
            tmp[i][i-1] = tmp[i-1][i] = dh[i-1];
        tmp[i][N-1] = 6 * (dddh[i+1] - dddh[i]);
    }
    for (int i = 1; i < N - 2; i++) {
        const float q = (tmp[i+1][i] / tmp[i][i]);
        for (int j = 1; j <= N - 1; j++)
            tmp[i+1][j] -= q * tmp[i][j];
    }
    for (int i = N - 2; i > 0; i--) {
        float sum = 0.0f;
        for (int j = i; j <= N - 2; j++)
            sum += tmp[i][j] * K[j];
        K[i] = (tmp[i][N-1] - sum) / tmp[i][i];
    }

hueshift_done: ;

    float prev_hue = -10.0f, prev_delta = 0.0f;
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

        // Determine perceptual hue shift delta by interpolation of refpoints
        float delta = 0.0f;
        if (disable_hueshift) {
            // do nothing
        } else if (fabsf(ich.h - prev_hue) < 1e-6f) {
            delta = prev_delta;
        } else {
            for (int i = 0; i < N - 1; i++) {
                if (hueshift[i+1].hue > ich.h) {
                    pl_assert(hueshift[i].hue <= ich.h);
                    float a = (K[i+1] - K[i]) / (6 * dh[i]);
                    float b = K[i] / 2;
                    float c = dddh[i+1] - (2 * dh[i] * K[i] + K[i+1] * dh[i]) / 6;
                    float d = hueshift[i].delta;
                    float x = ich.h - hueshift[i].hue;
                    delta = ((a * x + b) * x + c) * x + d;
                    prev_delta = delta;
                    prev_hue = ich.h;
                    break;
                }
            }
        }

        if (fabsf(delta) >= 1e-3f) {
            struct ICh src_border = desat_bounded(ich.I, ich.h, 0.0f, 0.5f, src);
            struct ICh dst_border = desat_bounded(ich.I, ich.h, 0.0f, 0.5f, dst);
            ich.h += delta * pl_smoothstep(dst_border.C * perceptual_knee,
                                           src_border.C, ich.C);
            src = src_post;
            dst = dst_post;
        }

        // Determine intersections with source and target gamuts
        struct ICh source = saturate(ich.h, src);
        struct ICh target = saturate(ich.h, dst);
        const float gamma = scale_gamma(perceptual_gamma, ich, target, dst);

        float lo = 0.0f, x = 1.0f, hi = 1.0f / perceptual_knee + 3 * maxDelta;
        do {
            struct ICh test = mix_exp(ich, x, gamma, target.I);
            if (ingamut(ich2ipt(test), dst)) {
                lo = x;
            } else {
                hi = x;
            }
            x = (lo + hi) / 2.0f;
        } while (hi - lo > maxDelta);

        // Apply simple Mobius tone mapping curve
        const float j = PL_MIX(1.0f, perceptual_knee, ich.C / 0.5f);
        const float peak = fmaxf(source.C / target.C, 1.0f);
        float xx = 1.0f / x;
        if (j < 1.0f && peak >= 1.0f) {
            const float a = -j*j * (peak - 1.0f) / (j*j - 2.0f * j + peak);
            const float b = (j*j - 2.0f * j * peak + peak) /
                            fmaxf(1e-6f, peak - 1.0f);
            const float k = (b*b + 2.0f * b*j + j*j) / (b - a);
            xx = fminf(xx, peak);
            xx = xx <= j ? xx : k * (xx + a) / (xx + b);
        }

        ipt = ich2ipt(mix_exp(ich, xx * x, gamma, target.I));
    }
}

const struct pl_gamut_map_function pl_gamut_map_perceptual = {
    .name = "perceptual",
    .description = "Perceptual soft-clip",
    .map = perceptual,
};

static void relative(float *lut, const struct pl_gamut_map_params *params)
{
    struct cache cache;
    struct gamut dst;
    get_gamuts(&dst, NULL, &cache, params);

    FOREACH_LUT(lut, ipt)
        ipt = clip_gamma(ipt, perceptual_gamma, dst);
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
    struct cache cache;
    struct gamut dst;
    get_gamuts(&dst, NULL, &cache, params);
    pl_matrix3x3 m = pl_get_adaptation_matrix(params->output_gamut.white,
                                              params->input_gamut.white);

    FOREACH_LUT(lut, ipt) {
        struct RGB rgb = ipt2rgb(ipt, dst);
        pl_matrix3x3_apply(&m, (float *) &rgb);
        ipt = rgb2ipt(rgb, dst);
        ipt = clip_gamma(ipt, perceptual_gamma, dst);
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
            ipt.I += 0.1f;
            ipt.P *= -1.2f;
            ipt.T *= -1.2f;
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
        ipt = clip_gamma(ipt, perceptual_gamma, dst);
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
