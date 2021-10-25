/*
 * This file is part of libplacebo, which is normally licensed under the terms
 * of the LGPL v2.1+. However, this file (film_grain_av1.c) is also available
 * under the terms of the more permissive MIT license:
 *
 * Copyright (c) 2018-2019 Niklas Haas
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "film_grain.h"
#include "shaders.h"

// Taken from the spec. Range is [-2048, 2047], mean is 0 and stddev is 512
static const int16_t gaussian_sequence[2048] = {
  56,    568,   -180,  172,   124,   -84,   172,   -64,   -900,  24,   820,
  224,   1248,  996,   272,   -8,    -916,  -388,  -732,  -104,  -188, 800,
  112,   -652,  -320,  -376,  140,   -252,  492,   -168,  44,    -788, 588,
  -584,  500,   -228,  12,    680,   272,   -476,  972,   -100,  652,  368,
  432,   -196,  -720,  -192,  1000,  -332,  652,   -136,  -552,  -604, -4,
  192,   -220,  -136,  1000,  -52,   372,   -96,   -624,  124,   -24,  396,
  540,   -12,   -104,  640,   464,   244,   -208,  -84,   368,   -528, -740,
  248,   -968,  -848,  608,   376,   -60,   -292,  -40,   -156,  252,  -292,
  248,   224,   -280,  400,   -244,  244,   -60,   76,    -80,   212,  532,
  340,   128,   -36,   824,   -352,  -60,   -264,  -96,   -612,  416,  -704,
  220,   -204,  640,   -160,  1220,  -408,  900,   336,   20,    -336, -96,
  -792,  304,   48,    -28,   -1232, -1172, -448,  104,   -292,  -520, 244,
  60,    -948,  0,     -708,  268,   108,   356,   -548,  488,   -344, -136,
  488,   -196,  -224,  656,   -236,  -1128, 60,    4,     140,   276,  -676,
  -376,  168,   -108,  464,   8,     564,   64,    240,   308,   -300, -400,
  -456,  -136,  56,    120,   -408,  -116,  436,   504,   -232,  328,  844,
  -164,  -84,   784,   -168,  232,   -224,  348,   -376,  128,   568,  96,
  -1244, -288,  276,   848,   832,   -360,  656,   464,   -384,  -332, -356,
  728,   -388,  160,   -192,  468,   296,   224,   140,   -776,  -100, 280,
  4,     196,   44,    -36,   -648,  932,   16,    1428,  28,    528,  808,
  772,   20,    268,   88,    -332,  -284,  124,   -384,  -448,  208,  -228,
  -1044, -328,  660,   380,   -148,  -300,  588,   240,   540,   28,   136,
  -88,   -436,  256,   296,   -1000, 1400,  0,     -48,   1056,  -136, 264,
  -528,  -1108, 632,   -484,  -592,  -344,  796,   124,   -668,  -768, 388,
  1296,  -232,  -188,  -200,  -288,  -4,    308,   100,   -168,  256,  -500,
  204,   -508,  648,   -136,  372,   -272,  -120,  -1004, -552,  -548, -384,
  548,   -296,  428,   -108,  -8,    -912,  -324,  -224,  -88,   -112, -220,
  -100,  996,   -796,  548,   360,   -216,  180,   428,   -200,  -212, 148,
  96,    148,   284,   216,   -412,  -320,  120,   -300,  -384,  -604, -572,
  -332,  -8,    -180,  -176,  696,   116,   -88,   628,   76,    44,   -516,
  240,   -208,  -40,   100,   -592,  344,   -308,  -452,  -228,  20,   916,
  -1752, -136,  -340,  -804,  140,   40,    512,   340,   248,   184,  -492,
  896,   -156,  932,   -628,  328,   -688,  -448,  -616,  -752,  -100, 560,
  -1020, 180,   -800,  -64,   76,    576,   1068,  396,   660,   552,  -108,
  -28,   320,   -628,  312,   -92,   -92,   -472,  268,   16,    560,  516,
  -672,  -52,   492,   -100,  260,   384,   284,   292,   304,   -148, 88,
  -152,  1012,  1064,  -228,  164,   -376,  -684,  592,   -392,  156,  196,
  -524,  -64,   -884,  160,   -176,  636,   648,   404,   -396,  -436, 864,
  424,   -728,  988,   -604,  904,   -592,  296,   -224,  536,   -176, -920,
  436,   -48,   1176,  -884,  416,   -776,  -824,  -884,  524,   -548, -564,
  -68,   -164,  -96,   692,   364,   -692,  -1012, -68,   260,   -480, 876,
  -1116, 452,   -332,  -352,  892,   -1088, 1220,  -676,  12,    -292, 244,
  496,   372,   -32,   280,   200,   112,   -440,  -96,   24,    -644, -184,
  56,    -432,  224,   -980,  272,   -260,  144,   -436,  420,   356,  364,
  -528,  76,    172,   -744,  -368,  404,   -752,  -416,  684,   -688, 72,
  540,   416,   92,    444,   480,   -72,   -1416, 164,   -1172, -68,  24,
  424,   264,   1040,  128,   -912,  -524,  -356,  64,    876,   -12,  4,
  -88,   532,   272,   -524,  320,   276,   -508,  940,   24,    -400, -120,
  756,   60,    236,   -412,  100,   376,   -484,  400,   -100,  -740, -108,
  -260,  328,   -268,  224,   -200,  -416,  184,   -604,  -564,  -20,  296,
  60,    892,   -888,  60,    164,   68,    -760,  216,   -296,  904,  -336,
  -28,   404,   -356,  -568,  -208,  -1480, -512,  296,   328,   -360, -164,
  -1560, -776,  1156,  -428,  164,   -504,  -112,  120,   -216,  -148, -264,
  308,   32,    64,    -72,   72,    116,   176,   -64,   -272,  460,  -536,
  -784,  -280,  348,   108,   -752,  -132,  524,   -540,  -776,  116,  -296,
  -1196, -288,  -560,  1040,  -472,  116,   -848,  -1116, 116,   636,  696,
  284,   -176,  1016,  204,   -864,  -648,  -248,  356,   972,   -584, -204,
  264,   880,   528,   -24,   -184,  116,   448,   -144,  828,   524,  212,
  -212,  52,    12,    200,   268,   -488,  -404,  -880,  824,   -672, -40,
  908,   -248,  500,   716,   -576,  492,   -576,  16,    720,   -108, 384,
  124,   344,   280,   576,   -500,  252,   104,   -308,  196,   -188, -8,
  1268,  296,   1032,  -1196, 436,   316,   372,   -432,  -200,  -660, 704,
  -224,  596,   -132,  268,   32,    -452,  884,   104,   -1008, 424,  -1348,
  -280,  4,     -1168, 368,   476,   696,   300,   -8,    24,    180,  -592,
  -196,  388,   304,   500,   724,   -160,  244,   -84,   272,   -256, -420,
  320,   208,   -144,  -156,  156,   364,   452,   28,    540,   316,  220,
  -644,  -248,  464,   72,    360,   32,    -388,  496,   -680,  -48,  208,
  -116,  -408,  60,    -604,  -392,  548,   -840,  784,   -460,  656,  -544,
  -388,  -264,  908,   -800,  -628,  -612,  -568,  572,   -220,  164,  288,
  -16,   -308,  308,   -112,  -636,  -760,  280,   -668,  432,   364,  240,
  -196,  604,   340,   384,   196,   592,   -44,   -500,  432,   -580, -132,
  636,   -76,   392,   4,     -412,  540,   508,   328,   -356,  -36,  16,
  -220,  -64,   -248,  -60,   24,    -192,  368,   1040,  92,    -24,  -1044,
  -32,   40,    104,   148,   192,   -136,  -520,  56,    -816,  -224, 732,
  392,   356,   212,   -80,   -424,  -1008, -324,  588,   -1496, 576,  460,
  -816,  -848,  56,    -580,  -92,   -1372, -112,  -496,  200,   364,  52,
  -140,  48,    -48,   -60,   84,    72,    40,    132,   -356,  -268, -104,
  -284,  -404,  732,   -520,  164,   -304,  -540,  120,   328,   -76,  -460,
  756,   388,   588,   236,   -436,  -72,   -176,  -404,  -316,  -148, 716,
  -604,  404,   -72,   -88,   -888,  -68,   944,   88,    -220,  -344, 960,
  472,   460,   -232,  704,   120,   832,   -228,  692,   -508,  132,  -476,
  844,   -748,  -364,  -44,   1116,  -1104, -1056, 76,    428,   552,  -692,
  60,    356,   96,    -384,  -188,  -612,  -576,  736,   508,   892,  352,
  -1132, 504,   -24,   -352,  324,   332,   -600,  -312,  292,   508,  -144,
  -8,    484,   48,    284,   -260,  -240,  256,   -100,  -292,  -204, -44,
  472,   -204,  908,   -188,  -1000, -256,  92,    1164,  -392,  564,  356,
  652,   -28,   -884,  256,   484,   -192,  760,   -176,  376,   -524, -452,
  -436,  860,   -736,  212,   124,   504,   -476,  468,   76,    -472, 552,
  -692,  -944,  -620,  740,   -240,  400,   132,   20,    192,   -196, 264,
  -668,  -1012, -60,   296,   -316,  -828,  76,    -156,  284,   -768, -448,
  -832,  148,   248,   652,   616,   1236,  288,   -328,  -400,  -124, 588,
  220,   520,   -696,  1032,  768,   -740,  -92,   -272,  296,   448,  -464,
  412,   -200,  392,   440,   -200,  264,   -152,  -260,  320,   1032, 216,
  320,   -8,    -64,   156,   -1016, 1084,  1172,  536,   484,   -432, 132,
  372,   -52,   -256,  84,    116,   -352,  48,    116,   304,   -384, 412,
  924,   -300,  528,   628,   180,   648,   44,    -980,  -220,  1320, 48,
  332,   748,   524,   -268,  -720,  540,   -276,  564,   -344,  -208, -196,
  436,   896,   88,    -392,  132,   80,    -964,  -288,  568,   56,   -48,
  -456,  888,   8,     552,   -156,  -292,  948,   288,   128,   -716, -292,
  1192,  -152,  876,   352,   -600,  -260,  -812,  -468,  -28,   -120, -32,
  -44,   1284,  496,   192,   464,   312,   -76,   -516,  -380,  -456, -1012,
  -48,   308,   -156,  36,    492,   -156,  -808,  188,   1652,  68,   -120,
  -116,  316,   160,   -140,  352,   808,   -416,  592,   316,   -480, 56,
  528,   -204,  -568,  372,   -232,  752,   -344,  744,   -4,    324,  -416,
  -600,  768,   268,   -248,  -88,   -132,  -420,  -432,  80,    -288, 404,
  -316,  -1216, -588,  520,   -108,  92,    -320,  368,   -480,  -216, -92,
  1688,  -300,  180,   1020,  -176,  820,   -68,   -228,  -260,  436,  -904,
  20,    40,    -508,  440,   -736,  312,   332,   204,   760,   -372, 728,
  96,    -20,   -632,  -520,  -560,  336,   1076,  -64,   -532,  776,  584,
  192,   396,   -728,  -520,  276,   -188,  80,    -52,   -612,  -252, -48,
  648,   212,   -688,  228,   -52,   -260,  428,   -412,  -272,  -404, 180,
  816,   -796,  48,    152,   484,   -88,   -216,  988,   696,   188,  -528,
  648,   -116,  -180,  316,   476,   12,    -564,  96,    476,   -252, -364,
  -376,  -392,  556,   -256,  -576,  260,   -352,  120,   -16,   -136, -260,
  -492,  72,    556,   660,   580,   616,   772,   436,   424,   -32,  -324,
  -1268, 416,   -324,  -80,   920,   160,   228,   724,   32,    -516, 64,
  384,   68,    -128,  136,   240,   248,   -204,  -68,   252,   -932, -120,
  -480,  -628,  -84,   192,   852,   -404,  -288,  -132,  204,   100,  168,
  -68,   -196,  -868,  460,   1080,  380,   -80,   244,   0,     484,  -888,
  64,    184,   352,   600,   460,   164,   604,   -196,  320,   -64,  588,
  -184,  228,   12,    372,   48,    -848,  -344,  224,   208,   -200, 484,
  128,   -20,   272,   -468,  -840,  384,   256,   -720,  -520,  -464, -580,
  112,   -120,  644,   -356,  -208,  -608,  -528,  704,   560,   -424, 392,
  828,   40,    84,    200,   -152,  0,     -144,  584,   280,   -120, 80,
  -556,  -972,  -196,  -472,  724,   80,    168,   -32,   88,    160,  -688,
  0,     160,   356,   372,   -776,  740,   -128,  676,   -248,  -480, 4,
  -364,  96,    544,   232,   -1032, 956,   236,   356,   20,    -40,  300,
  24,    -676,  -596,  132,   1120,  -104,  532,   -1096, 568,   648,  444,
  508,   380,   188,   -376,  -604,  1488,  424,   24,    756,   -220, -192,
  716,   120,   920,   688,   168,   44,    -460,  568,   284,   1144, 1160,
  600,   424,   888,   656,   -356,  -320,  220,   316,   -176,  -724, -188,
  -816,  -628,  -348,  -228,  -380,  1012,  -452,  -660,  736,   928,  404,
  -696,  -72,   -268,  -892,  128,   184,   -344,  -780,  360,   336,  400,
  344,   428,   548,   -112,  136,   -228,  -216,  -820,  -516,  340,  92,
  -136,  116,   -300,  376,   -244,  100,   -316,  -520,  -284,  -12,  824,
  164,   -548,  -180,  -128,  116,   -924,  -828,  268,   -368,  -580, 620,
  192,   160,   0,     -1676, 1068,  424,   -56,   -360,  468,   -156, 720,
  288,   -528,  556,   -364,  548,   -148,  504,   316,   152,   -648, -620,
  -684,  -24,   -376,  -384,  -108,  -920,  -1032, 768,   180,   -264, -508,
  -1268, -260,  -60,   300,   -240,  988,   724,   -376,  -576,  -212, -736,
  556,   192,   1092,  -620,  -880,  376,   -56,   -4,    -216,  -32,  836,
  268,   396,   1332,  864,   -600,  100,   56,    -412,  -92,   356,  180,
  884,   -468,  -436,  292,   -388,  -804,  -704,  -840,  368,   -348, 140,
  -724,  1536,  940,   372,   112,   -372,  436,   -480,  1136,  296,  -32,
  -228,  132,   -48,   -220,  868,   -1016, -60,   -1044, -464,  328,  916,
  244,   12,    -736,  -296,  360,   468,   -376,  -108,  -92,   788,  368,
  -56,   544,   400,   -672,  -420,  728,   16,    320,   44,    -284, -380,
  -796,  488,   132,   204,   -596,  -372,  88,    -152,  -908,  -636, -572,
  -624,  -116,  -692,  -200,  -56,   276,   -88,   484,   -324,  948,  864,
  1000,  -456,  -184,  -276,  292,   -296,  156,   676,   320,   160,  908,
  -84,   -1236, -288,  -116,  260,   -372,  -644,  732,   -756,  -96,  84,
  344,   -520,  348,   -688,  240,   -84,   216,   -1044, -136,  -676, -396,
  -1500, 960,   -40,   176,   168,   1516,  420,   -504,  -344,  -364, -360,
  1216,  -940,  -380,  -212,  252,   -660,  -708,  484,   -444,  -152, 928,
  -120,  1112,  476,   -260,  560,   -148,  -344,  108,   -196,  228,  -288,
  504,   560,   -328,  -88,   288,   -1008, 460,   -228,  468,   -836, -196,
  76,    388,   232,   412,   -1168, -716,  -644,  756,   -172,  -356, -504,
  116,   432,   528,   48,    476,   -168,  -608,  448,   160,   -532, -272,
  28,    -676,  -12,   828,   980,   456,   520,   104,   -104,  256,  -344,
  -4,    -28,   -368,  -52,   -524,  -572,  -556,  -200,  768,   1124, -208,
  -512,  176,   232,   248,   -148,  -888,  604,   -600,  -304,  804,  -156,
  -212,  488,   -192,  -804,  -256,  368,   -360,  -916,  -328,  228,  -240,
  -448,  -472,  856,   -556,  -364,  572,   -12,   -156,  -368,  -340, 432,
  252,   -752,  -152,  288,   268,   -580,  -848,  -592,  108,   -76,  244,
  312,   -716,  592,   -80,   436,   360,   4,     -248,  160,   516,  584,
  732,   44,    -468,  -280,  -292,  -156,  -588,  28,    308,   912,  24,
  124,   156,   180,   -252,  944,   -924,  -772,  -520,  -428,  -624, 300,
  -212,  -1144, 32,    -724,  800,   -1128, -212,  -1288, -848,  180,  -416,
  440,   192,   -576,  -792,  -76,   -1080, 80,    -532,  -352,  -132, 380,
  -820,  148,   1112,  128,   164,   456,   700,   -924,  144,   -668, -384,
  648,   -832,  508,   552,   -52,   -100,  -656,  208,   -568,  748,  -88,
  680,   232,   300,   192,   -408,  -1012, -152,  -252,  -268,  272,  -876,
  -664,  -648,  -332,  -136,  16,    12,    1152,  -28,   332,   -536, 320,
  -672,  -460,  -316,  532,   -260,  228,   -40,   1052,  -816,  180,  88,
  -496,  -556,  -672,  -368,  428,   92,    356,   404,   -408,  252,  196,
  -176,  -556,  792,   268,   32,    372,   40,    96,    -332,  328,  120,
  372,   -900,  -40,   472,   -264,  -592,  952,   128,   656,   112,  664,
  -232,  420,   4,     -344,  -464,  556,   244,   -416,  -32,   252,  0,
  -412,  188,   -696,  508,   -476,  324,   -1096, 656,   -312,  560,  264,
  -136,  304,   160,   -64,   -580,  248,   336,   -720,  560,   -348, -288,
  -276,  -196,  -500,  852,   -544,  -236,  -1128, -992,  -776,  116,  56,
  52,    860,   884,   212,   -12,   168,   1020,  512,   -552,  924,  -148,
  716,   188,   164,   -340,  -520,  -184,  880,   -152,  -680,  -208, -1156,
  -300,  -528,  -472,  364,   100,   -744,  -1056, -32,   540,   280,  144,
  -676,  -32,   -232,  -280,  -224,  96,    568,   -76,   172,   148,  148,
  104,   32,    -296,  -32,   788,   -80,   32,    -16,   280,   288,  944,
  428,   -484
};

static inline int get_random_number(int bits, uint16_t *state)
{
    int r = *state;
    uint16_t bit = ((r >> 0) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
    *state = (r >> 1) | (bit << 15);

    return (*state >> (16 - bits)) & ((1 << bits) - 1);
}

static inline int round2(int x, int shift)
{
    if (!shift)
        return x;

    return (x + (1 << (shift - 1))) >> shift;
}

enum {
    BLOCK_SIZE = 32,
    SCALING_LUT_SIZE = 256,

    GRAIN_WIDTH = 82,
    GRAIN_HEIGHT = 73,
    // On the GPU we only need a subsection of this
    GRAIN_WIDTH_LUT = 64,
    GRAIN_HEIGHT_LUT = 64,
    GRAIN_PAD_LUT = 9,

    // For subsampled grain textures
    SUB_GRAIN_WIDTH = 44,
    SUB_GRAIN_HEIGHT = 38,
    SUB_GRAIN_WIDTH_LUT = GRAIN_WIDTH_LUT >> 1,
    SUB_GRAIN_HEIGHT_LUT = GRAIN_HEIGHT_LUT >> 1,
    SUB_GRAIN_PAD_LUT = 6,
};

// Contains the shift by which the offsets are indexed
enum offset {
    OFFSET_TL = 24,
    OFFSET_T  = 16,
    OFFSET_L  = 8,
    OFFSET_N  = 0,
};

// Helper function to compute some common constants
struct grain_scale {
    int grain_center;
    int grain_min;
    int grain_max;
    float texture_scale;
    float grain_scale;
};

static inline int bit_depth(const struct pl_color_repr *repr)
{
    int depth = PL_DEF(repr->bits.color_depth,
                PL_DEF(repr->bits.sample_depth, 8));
    pl_assert(depth >= 8);
    return depth;
}

static struct grain_scale get_grain_scale(const struct pl_film_grain_params *params)
{
    int bits = bit_depth(params->repr);
    struct grain_scale ret = {
        .grain_center = 128 << (bits - 8),
    };

    ret.grain_min = -ret.grain_center;
    ret.grain_max = (256 << (bits - 8)) - 1 - ret.grain_center;

    struct pl_color_repr repr = *params->repr;
    ret.texture_scale = pl_color_repr_normalize(&repr);

    // Since our color samples are normalized to the range [0, 1], we need to
    // scale down grain values from the scale [0, 2^b - 1] to this range.
    ret.grain_scale = 1.0 / ((1 << bits) - 1);

    return ret;
}

// Generates the basic grain table (LumaGrain in the spec).
static void generate_grain_y(float out[GRAIN_HEIGHT_LUT][GRAIN_WIDTH_LUT],
                             int16_t buf[GRAIN_HEIGHT][GRAIN_WIDTH],
                             const struct pl_film_grain_params *params)
{
    const struct pl_av1_grain_data *data = &params->data.params.av1;
    struct grain_scale scale = get_grain_scale(params);
    uint16_t seed = (uint16_t) params->data.seed;
    int bits = bit_depth(params->repr);
    int shift = 12 - bits + data->grain_scale_shift;
    pl_assert(shift >= 0);

    for (int y = 0; y < GRAIN_HEIGHT; y++) {
        for (int x = 0; x < GRAIN_WIDTH; x++) {
            int16_t value = gaussian_sequence[ get_random_number(11, &seed) ];
            buf[y][x] = round2(value, shift);
        }
    }

    const int ar_pad = 3;
    int ar_lag = data->ar_coeff_lag;

    for (int y = ar_pad; y < GRAIN_HEIGHT; y++) {
        for (int x = ar_pad; x < GRAIN_WIDTH - ar_pad; x++) {
            const int8_t *coeff = data->ar_coeffs_y;
            int sum = 0;
            for (int dy = -ar_lag; dy <= 0; dy++) {
                for (int dx = -ar_lag; dx <= ar_lag; dx++) {
                    if (!dx && !dy)
                        break;
                    sum += *(coeff++) * buf[y + dy][x + dx];
                }
            }

            int16_t grain = buf[y][x] + round2(sum, data->ar_coeff_shift);
            grain = PL_CLAMP(grain, scale.grain_min, scale.grain_max);
            buf[y][x] = grain;
        }
    }

    for (int y = 0; y < GRAIN_HEIGHT_LUT; y++) {
        for (int x = 0; x < GRAIN_WIDTH_LUT; x++) {
            int16_t grain = buf[y + GRAIN_PAD_LUT][x + GRAIN_PAD_LUT];
            out[y][x] = grain * scale.grain_scale;
        }
    }
}

static void generate_grain_uv(float *out, int16_t buf[GRAIN_HEIGHT][GRAIN_WIDTH],
                              const int16_t buf_y[GRAIN_HEIGHT][GRAIN_WIDTH],
                              enum pl_channel channel, int sub_x, int sub_y,
                              const struct pl_film_grain_params *params)
{
    const struct pl_av1_grain_data *data = &params->data.params.av1;
    struct grain_scale scale = get_grain_scale(params);
    int bits = bit_depth(params->repr);
    int shift = 12 - bits + data->grain_scale_shift;
    pl_assert(shift >= 0);

    uint16_t seed = params->data.seed;
    if (channel == PL_CHANNEL_CB) {
        seed ^= 0xb524;
    } else if (channel == PL_CHANNEL_CR) {
        seed ^= 0x49d8;
    }

    int chromaW = sub_x ? SUB_GRAIN_WIDTH  : GRAIN_WIDTH;
    int chromaH = sub_y ? SUB_GRAIN_HEIGHT : GRAIN_HEIGHT;

    const int8_t *coeffs[] = {
        [PL_CHANNEL_CB] = data->ar_coeffs_uv[0],
        [PL_CHANNEL_CR] = data->ar_coeffs_uv[1],
    };

    for (int y = 0; y < chromaH; y++) {
        for (int x = 0; x < chromaW; x++) {
            int16_t value = gaussian_sequence[ get_random_number(11, &seed) ];
            buf[y][x] = round2(value, shift);
        }
    }

    const int ar_pad = 3;
    int ar_lag = data->ar_coeff_lag;

    for (int y = ar_pad; y < chromaH; y++) {
        for (int x = ar_pad; x < chromaW - ar_pad; x++) {
            const int8_t *coeff = coeffs[channel];
            pl_assert(coeff);
            int sum = 0;
            for (int dy = -ar_lag; dy <= 0; dy++) {
                for (int dx = -ar_lag; dx <= ar_lag; dx++) {
                    // For the final (current) pixel, we need to add in the
                    // contribution from the luma grain texture
                    if (!dx && !dy) {
                        if (!data->num_points_y)
                            break;
                        int luma = 0;
                        int lumaX = ((x - ar_pad) << sub_x) + ar_pad;
                        int lumaY = ((y - ar_pad) << sub_y) + ar_pad;
                        for (int i = 0; i <= sub_y; i++) {
                            for (int j = 0; j <= sub_x; j++) {
                                luma += buf_y[lumaY + i][lumaX + j];
                            }
                        }
                        luma = round2(luma, sub_x + sub_y);
                        sum += luma * (*coeff);
                        break;
                    }

                    sum += *(coeff++) * buf[y + dy][x + dx];
                }
            }

            int16_t grain = buf[y][x] + round2(sum, data->ar_coeff_shift);
            grain = PL_CLAMP(grain, scale.grain_min, scale.grain_max);
            buf[y][x] = grain;
        }
    }

    int lutW = GRAIN_WIDTH_LUT >> sub_x;
    int lutH = GRAIN_HEIGHT_LUT >> sub_y;
    int padX = sub_x ? SUB_GRAIN_PAD_LUT : GRAIN_PAD_LUT;
    int padY = sub_y ? SUB_GRAIN_PAD_LUT : GRAIN_PAD_LUT;

    for (int y = 0; y < lutH; y++) {
        for (int x = 0; x < lutW; x++) {
            int16_t grain = buf[y + padY][x + padX];
            out[y * lutW + x] = grain * scale.grain_scale;
        }
    }
}

static void generate_offsets(void *pbuf, const struct sh_lut_params *params)
{
    const struct pl_film_grain_data *data = params->priv;
    unsigned int *buf = pbuf;
    pl_static_assert(sizeof(unsigned int) >= sizeof(uint32_t));

    for (int y = 0; y < params->height; y++) {
        uint16_t state = data->seed;
        state ^= ((y * 37 + 178) & 0xFF) << 8;
        state ^= ((y * 173 + 105) & 0xFF);

        for (int x = 0; x < params->width; x++) {
            unsigned int *offsets = &buf[y * params->width + x];

            uint8_t val = get_random_number(8, &state);
            uint8_t val_l = x ? (offsets - 1)[0] : 0;
            uint8_t val_t = y ? (offsets - params->width)[0] : 0;
            uint8_t val_tl = x && y ? (offsets - params->width - 1)[0] : 0;

            // Encode four offsets into a single 32-bit integer for the
            // convenience of the GPU. That way only one LUT fetch is
            // required for the entire block.
            *offsets = ((uint32_t) val_tl << OFFSET_TL)
                     | ((uint32_t) val_t  << OFFSET_T)
                     | ((uint32_t) val_l  << OFFSET_L)
                     | ((uint32_t) val    << OFFSET_N);
        }
    }
}

static void generate_scaling(void *pdata, const struct sh_lut_params *params)
{
    assert(params->width == SCALING_LUT_SIZE && params->comps == 1);
    float *data = pdata;

    struct {
        int num;
        uint8_t (*points)[2];
        const struct pl_av1_grain_data *data;
    } *ctx = params->priv;

    float range = 1 << ctx->data->scaling_shift;

    // Fill up the preceding entries with the initial value
    for (int i = 0; i < ctx->points[0][0]; i++)
        data[i] = ctx->points[0][1] / range;

    // Linearly interpolate the values in the middle
    for (int i = 0; i < ctx->num - 1; i++) {
        int bx = ctx->points[i][0];
        int by = ctx->points[i][1];
        int dx = ctx->points[i + 1][0] - bx;
        int dy = ctx->points[i + 1][1] - by;
        int delta = dy * ((0x10000 + (dx >> 1)) / dx);
        for (int x = 0; x < dx; x++) {
            int v = by + ((x * delta + 0x8000) >> 16);
            data[bx + x] = v / range;
        }
    }

    // Fill up the remaining entries with the final value
    for (int i = ctx->points[ctx->num - 1][0]; i < SCALING_LUT_SIZE; i++)
        data[i] = ctx->points[ctx->num - 1][1] / range;
}

static void sample(pl_shader sh, enum offset off, ident_t lut, int idx,
                   int sub_x, int sub_y)
{
    int dx = (off & OFFSET_L) ? 1 : 0,
        dy = (off & OFFSET_T) ? 1 : 0;

    static const char *index_strs[] = {
        [0] = ".x",
        [1] = ".y",
    };

    GLSL("offset = uvec2(%du, %du) * uvec2((data >> %d) & 0xFu, \n"
         "                                 (data >> %d) & 0xFu);\n"
         "pos = offset + local_id.xy + uvec2(%d, %d);           \n"
         "val = %s(pos)%s;                                      \n",
         sub_x ? 1 : 2, sub_y ? 1 : 2, off + 4, off,
         (BLOCK_SIZE >> sub_x) * dx,
         (BLOCK_SIZE >> sub_y) * dy,
         lut, idx >= 0 ? index_strs[idx] : "");
}

struct grain_obj_av1 {
    // LUT objects for the offsets, grain and scaling luts
    pl_shader_obj lut_offsets;
    pl_shader_obj lut_grain[2];
    pl_shader_obj lut_scaling[3];

    // Previous parameters used to check reusability
    struct pl_film_grain_data data;
    struct pl_color_repr repr;
    bool fg_has_y;
    bool fg_has_u;
    bool fg_has_v;

    // Space to store the temporary arrays, reused
    uint32_t *offsets;
    float grain[2][GRAIN_HEIGHT_LUT][GRAIN_WIDTH_LUT];
    int16_t grain_tmp_y[GRAIN_HEIGHT][GRAIN_WIDTH];
    int16_t grain_tmp_uv[GRAIN_HEIGHT][GRAIN_WIDTH];
};

static void av1_grain_uninit(pl_gpu gpu, void *ptr)
{
    struct grain_obj_av1 *obj = ptr;
    pl_shader_obj_destroy(&obj->lut_offsets);
    for (int i = 0; i < PL_ARRAY_SIZE(obj->lut_grain); i++)
        pl_shader_obj_destroy(&obj->lut_grain[i]);
    for (int i = 0; i < PL_ARRAY_SIZE(obj->lut_scaling); i++)
        pl_shader_obj_destroy(&obj->lut_scaling[i]);
    *obj = (struct grain_obj_av1) {0};
}

bool pl_needs_fg_av1(const struct pl_film_grain_params *params)
{
    const struct pl_av1_grain_data *data = &params->data.params.av1;
    bool has_y = data->num_points_y > 0;
    bool has_u = data->num_points_uv[0] > 0 || data->chroma_scaling_from_luma;
    bool has_v = data->num_points_uv[1] > 0 || data->chroma_scaling_from_luma;

    for (int i = 0; i < 3; i++) {
        enum pl_channel channel = channel_map(i, params);
        if (channel == PL_CHANNEL_Y && has_y)
            return true;
        if (channel == PL_CHANNEL_CB && has_u)
            return true;
        if (channel == PL_CHANNEL_CR && has_v)
            return true;
    }

    return false;
}

static inline bool av1_grain_data_eq(const struct pl_film_grain_data *da,
                                     const struct pl_film_grain_data *db)
{
    const struct pl_av1_grain_data *a = &da->params.av1, *b = &db->params.av1;

    // Only check the fields that are relevant for grain LUT generation
    return da->seed == db->seed &&
           a->chroma_scaling_from_luma == b->chroma_scaling_from_luma &&
           a->scaling_shift == b->scaling_shift &&
           a->ar_coeff_lag == b->ar_coeff_lag &&
           a->ar_coeff_shift == b->ar_coeff_shift &&
           a->grain_scale_shift == b->grain_scale_shift &&
           !memcmp(a->ar_coeffs_y, b->ar_coeffs_y, sizeof(a->ar_coeffs_y)) &&
           !memcmp(a->ar_coeffs_uv, b->ar_coeffs_uv, sizeof(a->ar_coeffs_uv));
}

static void fill_grain_lut(void *data, const struct sh_lut_params *params)
{
    struct grain_obj_av1 *obj = params->priv;
    size_t entries = params->width * params->height * params->comps;
    memcpy(data, obj->grain, entries * sizeof(float));
}

bool pl_shader_fg_av1(pl_shader sh, pl_shader_obj *grain_state,
                      const struct pl_film_grain_params *params)
{
    int sub_x = 0, sub_y = 0;
    int tex_w = params->tex->params.w,
        tex_h = params->tex->params.h;

    if (params->luma_tex) {
        sub_x = params->luma_tex->params.w > tex_w;
        sub_y = params->luma_tex->params.h > tex_h;
    }

    const struct pl_av1_grain_data *data = &params->data.params.av1;
    bool fg_has_y = data->num_points_y > 0;
    bool fg_has_u = data->num_points_uv[0] > 0 || data->chroma_scaling_from_luma;
    bool fg_has_v = data->num_points_uv[1] > 0 || data->chroma_scaling_from_luma;

    bool tex_is_y = false, tex_is_cb = false, tex_is_cr = false;
    for (int i = 0; i < 3; i++) {
        switch (channel_map(i, params)) {
        case PL_CHANNEL_Y:  tex_is_y = true; break;
        case PL_CHANNEL_CB: tex_is_cb = true; break;
        case PL_CHANNEL_CR: tex_is_cr = true; break;
        default: break;
        };
    }

    if (tex_is_y && (sub_x || sub_y)) {
        PL_WARN(sh, "pl_film_grain_params.channels includes PL_CHANNEL_Y but "
                "plane is subsampled, this makes no sense. Continuing anyway "
                "but output is likely incorrect.");
    }

    if (!sh_require(sh, PL_SHADER_SIG_NONE, tex_w, tex_h))
        return false;

    pl_gpu gpu = SH_GPU(sh);
    if (!gpu) {
        PL_ERR(sh, "AV1 film grain synthesis requires a non-NULL pl_gpu!");
        return false;
    }

    if (sh_glsl(sh).version < 130) {
        PL_ERR(sh, "AV1 film grain synthesis requires GLSL >= 130!");
        return false;
    }

    // Disable generation for unneeded component types
    fg_has_y &= tex_is_y;
    fg_has_u &= tex_is_cb;
    fg_has_v &= tex_is_cr;

    int bw = BLOCK_SIZE >> sub_x;
    int bh = BLOCK_SIZE >> sub_y;
    bool is_compute = sh_try_compute(sh, bw, bh, false, sizeof(uint32_t));

    struct grain_obj_av1 *obj;
    obj = SH_OBJ(sh, grain_state, PL_SHADER_OBJ_AV1_GRAIN,
                 struct grain_obj_av1, av1_grain_uninit);
    if (!obj)
        return false;

    // Note: In theory we could check only the parameters related to luma or
    // only related to chroma and skip updating for changes to irrelevant
    // parts, but this is probably not worth it since the seed is expected to
    // change per frame anyway.
    bool needs_update = !av1_grain_data_eq(&params->data, &obj->data) ||
                        !pl_color_repr_equal(params->repr, &obj->repr) ||
                        fg_has_y != obj->fg_has_y ||
                        fg_has_u != obj->fg_has_u ||
                        fg_has_v != obj->fg_has_v;

    if (needs_update) {
        // This is needed even for chroma, so statically generate it
        generate_grain_y(obj->grain[0], obj->grain_tmp_y, params);
    }

    ident_t lut[3];
    int idx[3] = {-1};

    if (fg_has_y) {
        lut[0] = sh_lut(sh, sh_lut_params(
            .object = &obj->lut_grain[0],
            .method = SH_LUT_TEXTURE,
            .type = PL_VAR_FLOAT,
            .width = GRAIN_WIDTH_LUT,
            .height = GRAIN_HEIGHT_LUT,
            .comps = 1,
            .update = needs_update,
            .dynamic = true,
            .fill = fill_grain_lut,
            .priv = obj,
        ));

        if (!lut[0]) {
            SH_FAIL(sh, "Failed generating/uploading luma grain LUT!");
            return false;
        }
    }

    // Try merging the chroma LUTs into a single texture
    int chroma_comps = 0;
    if (fg_has_u) {
        generate_grain_uv(&obj->grain[chroma_comps][0][0], obj->grain_tmp_uv,
                          obj->grain_tmp_y, PL_CHANNEL_CB, sub_x, sub_y,
                          params);
        idx[1] = chroma_comps++;
    }
    if (fg_has_v) {
        generate_grain_uv(&obj->grain[chroma_comps][0][0], obj->grain_tmp_uv,
                          obj->grain_tmp_y, PL_CHANNEL_CR, sub_x, sub_y,
                          params);
        idx[2] = chroma_comps++;
    }

    if (chroma_comps > 0) {
        lut[1] = lut[2] = sh_lut(sh, sh_lut_params(
            .object = &obj->lut_grain[1],
            .method = SH_LUT_TEXTURE,
            .type = PL_VAR_FLOAT,
            .width = GRAIN_WIDTH_LUT >> sub_x,
            .height = GRAIN_HEIGHT_LUT >> sub_y,
            .comps = chroma_comps,
            .update = needs_update,
            .dynamic = true,
            .fill = fill_grain_lut,
            .priv = obj,
        ));

        if (!lut[1]) {
            SH_FAIL(sh, "Failed generating/uploading chroma grain LUT!");
            return false;
        }

        if (chroma_comps == 1)
            idx[1] = idx[2] = -1;
    }

    ident_t offsets = sh_lut(sh, sh_lut_params(
        .object = &obj->lut_offsets,
        .method = SH_LUT_AUTO,
        .type = PL_VAR_UINT,
        .width = PL_ALIGN2(tex_w << sub_x, 128) / 32,
        .height = PL_ALIGN2(tex_h << sub_y, 128) / 32,
        .comps = 1,
        .update = needs_update,
        .dynamic = true,
        .fill = generate_offsets,
        .priv = (void *) &params->data,
    ));

    // For the scaling LUTs, we assume they'll be relatively constant
    // throughout the video so doing some extra work to avoid reinitializing
    // them constantly is probably worth it. Probably.
    const struct pl_av1_grain_data *obj_data = &obj->data.params.av1;
    bool scaling_changed = false;
    if (fg_has_y || data->chroma_scaling_from_luma) {
        scaling_changed |= data->num_points_y != obj_data->num_points_y;
        scaling_changed |= memcmp(data->points_y, obj_data->points_y,
                                  sizeof(data->points_y));
    }

    if (fg_has_u && !data->chroma_scaling_from_luma) {
        scaling_changed |= data->num_points_uv[0] != obj_data->num_points_uv[0];
        scaling_changed |= memcmp(data->points_uv[0],
                                  obj_data->points_uv[0],
                                  sizeof(data->points_uv[0]));
    }

    if (fg_has_v && !data->chroma_scaling_from_luma) {
        scaling_changed |= data->num_points_uv[1] != obj_data->num_points_uv[1];
        scaling_changed |= memcmp(data->points_uv[1],
                                  obj_data->points_uv[1],
                                  sizeof(data->points_uv[1]));
    }

    ident_t scaling[3] = {0};
    for (int i = 0; i < 3; i++) {
        struct {
            int num;
            const uint8_t (*points)[2];
            const struct pl_av1_grain_data *data;
        } priv;

        priv.data = data;
        if (i == 0 || data->chroma_scaling_from_luma) {
            priv.num = data->num_points_y;
            priv.points = &data->points_y[0];
        } else {
            priv.num = data->num_points_uv[i - 1];
            priv.points = &data->points_uv[i - 1][0];
        }

        // Skip scaling for unneeded channels
        bool has_c[3] = { fg_has_y, fg_has_u, fg_has_v };
        if (has_c[i] && priv.num > 0) {
            scaling[i] = sh_lut(sh, sh_lut_params(
                .object = &obj->lut_scaling[i],
                .type = PL_VAR_FLOAT,
                .width = SCALING_LUT_SIZE,
                .comps = 1,
                .linear = true,
                .update = scaling_changed,
                .dynamic = true,
                .fill = generate_scaling,
                .priv = &priv,
            ));

            if (!scaling[i]) {
                SH_FAIL(sh, "Failed generating/uploading scaling LUTs!");
                return false;
            }
        }
    }

    // Done updating LUTs
    obj->data = params->data;
    obj->repr = *params->repr;
    obj->fg_has_y = fg_has_y;
    obj->fg_has_u = fg_has_u;
    obj->fg_has_v = fg_has_v;

    sh_describe(sh, "AV1 film grain");
    GLSL("vec4 color;                   \n"
         "// pl_shader_film_grain (AV1) \n"
         "{                             \n"
         "uvec2 offset;                 \n"
         "uvec2 pos;                    \n"
         "float val;                    \n"
         "float grain;                  \n");

    if (is_compute) {
        GLSL("uvec2 block_id  = gl_WorkGroupID.xy;        \n"
             "uvec2 local_id  = gl_LocalInvocationID.xy;  \n"
             "uvec2 global_id = gl_GlobalInvocationID.xy; \n");
    } else {
        GLSL("uvec2 global_id = uvec2(gl_FragCoord);                  \n"
             "uvec2 block_id  = global_id / uvec2(%d, %d);            \n"
             "uvec2 local_id  = global_id - uvec2(%d, %d) * block_id; \n",
             bw, bh, bw, bh);
    }

    // Load the data vector which holds the offsets
    if (is_compute) {
        GLSLH("shared uint data; \n");
        GLSL("if (gl_LocalInvocationIndex == 0u) \n"
             "    data = uint(%s(block_id));     \n"
             "barrier();                         \n",
             offsets);
    } else {
        GLSL("uint data = uint(%s(block_id)); \n", offsets);
    }

    struct grain_scale scale = get_grain_scale(params);
    pl_color_repr_normalize(params->repr);
    int bits = PL_DEF(params->repr->bits.color_depth, 8);
    pl_assert(bits >= 8);

    ident_t minValue, maxLuma, maxChroma;
    if (pl_color_levels_guess(params->repr) == PL_COLOR_LEVELS_LIMITED) {
        float out_scale = (1 << bits) / ((1 << bits) - 1.0);
        minValue  = SH_FLOAT(16  / 256.0 * out_scale);
        maxLuma   = SH_FLOAT(235 / 256.0 * out_scale);
        maxChroma = SH_FLOAT(240 / 256.0 * out_scale);
        if (!pl_color_system_is_ycbcr_like(params->repr->sys))
            maxChroma = maxLuma;
    } else {
        minValue  = SH_FLOAT(0.0);
        maxLuma   = SH_FLOAT(1.0);
        maxChroma = SH_FLOAT(1.0);
    }

    // Load the color value of the tex itself
    ident_t tex = sh_desc(sh, (struct pl_shader_desc) {
        .binding.object = params->tex,
        .desc = (struct pl_desc) {
            .name = "tex",
            .type = PL_DESC_SAMPLED_TEX,
        },
    });

    ident_t tex_scale = SH_FLOAT(scale.texture_scale);
    GLSL("color = vec4(%s) * texelFetch(%s, ivec2(global_id), 0); \n",
         tex_scale, tex);

    // If we need access to the external luma plane, load it now
    if (tex_is_cb || tex_is_cr) {
        GLSL("float averageLuma; \n");
        if (tex_is_y) {
            // We already have the luma channel as part of the pre-sampled color
            for (int i = 0; i < 3; i++) {
                if (channel_map(i, params) == PL_CHANNEL_Y) {
                    GLSL("averageLuma = color[%s]; \n", SH_INT(i));
                    break;
                }
            }
        } else {
            // Luma channel not present in image, attach it separately
            pl_assert(params->luma_tex);
            ident_t luma = sh_desc(sh, (struct pl_shader_desc) {
                .binding.object = params->luma_tex,
                .desc = (struct pl_desc) {
                    .name = "luma",
                    .type = PL_DESC_SAMPLED_TEX,
                },
            });

            GLSL("pos = global_id * uvec2(%du, %du);                    \n"
                 "averageLuma = %s * texelFetch(%s, ivec2(pos), 0)[%s]; \n",
                 1 << sub_x, 1 << sub_y, tex_scale, luma,
                 SH_INT(params->luma_comp));
        }
    }

    ident_t grain_min = SH_FLOAT(scale.grain_min * scale.grain_scale);
    ident_t grain_max = SH_FLOAT(scale.grain_max * scale.grain_scale);

    for (int i = 0; i < params->components; i++) {
        enum pl_channel c = channel_map(i, params);
        if (c == PL_CHANNEL_NONE)
            continue;
        if (!scaling[c])
            continue;

        sample(sh, OFFSET_N, lut[c], idx[c], sub_x, sub_y);
        GLSL("grain = val; \n");

        if (data->overlap) {
            const char *weights[] = { "vec2(27.0, 17.0)", "vec2(23.0, 22.0)" };

            // X-direction overlapping
            GLSL("if (block_id.x > 0u && local_id.x < %du) {    \n"
                 "vec2 w = %s / 32.0;                           \n"
                 "if (local_id.x == 1u) w.xy = w.yx;            \n",
                 2 >> sub_x, weights[sub_x]);
            sample(sh, OFFSET_L, lut[c], idx[c], sub_x, sub_y);
            GLSL("grain = dot(vec2(val, grain), w);             \n"
                 "}                                             \n");

            // Y-direction overlapping
            GLSL("if (block_id.y > 0u && local_id.y < %du) {    \n"
                 "vec2 w = %s / 32.0;                           \n"
                 "if (local_id.y == 1u) w.xy = w.yx;            \n",
                 2 >> sub_y, weights[sub_y]);

            // We need to special-case the top left pixels since these need to
            // pre-blend the top-left offset block before blending vertically
            GLSL("    if (block_id.x > 0u && local_id.x < %du) {\n"
                 "        vec2 w2 = %s / 32.0;                  \n"
                 "        if (local_id.x == 1u) w2.xy = w2.yx;  \n",
                 2 >> sub_x, weights[sub_x]);
                          sample(sh, OFFSET_TL, lut[c], idx[c], sub_x, sub_y);
            GLSL("        float tmp = val;                      \n");
                          sample(sh, OFFSET_T, lut[c], idx[c], sub_x, sub_y);
            GLSL("        val = dot(vec2(tmp, val), w2);        \n"
                 "    } else {                                  \n");
                          sample(sh, OFFSET_T, lut[c], idx[c], sub_x, sub_y);
            GLSL("    }                                         \n"
                 "grain = dot(vec2(val, grain), w);             \n"
                 "}                                             \n");

            // Correctly clip the interpolated grain
            GLSL("grain = clamp(grain, %s, %s); \n", grain_min, grain_max);
        }

        if (c == PL_CHANNEL_Y) {
            GLSL("color[%d] += %s(color[%d]) * grain;   \n"
                 "color[%d] = clamp(color[%d], %s, %s); \n",
                 i, scaling[c], i,
                 i, i, minValue, maxLuma);
        } else {
            GLSL("val = averageLuma; \n");
            if (!data->chroma_scaling_from_luma) {
                // We need to load some extra variables for the mixing. Do this
                // using sh_var instead of hard-coding them to avoid shader
                // recompilation when these values change.
                ident_t mult = sh_var(sh, (struct pl_shader_var) {
                    .var = pl_var_vec2("mult"),
                    .data = &(float[2]){
                        data->uv_mult_luma[c - 1] / 64.0,
                        data->uv_mult[c - 1] / 64.0,
                    },
                });

                int c_offset = data->uv_offset[c - 1] << (bits - 8);
                ident_t offset = sh_var(sh, (struct pl_shader_var) {
                    .var = pl_var_float("offset"),
                    .data = &(float) { c_offset * scale.grain_scale },
                });

                GLSL("val = dot(vec2(val, color[%d]), %s);  \n", i, mult);
                GLSL("val += %s; \n", offset);
            }
            GLSL("color[%d] += %s(val) * grain;         \n"
                 "color[%d] = clamp(color[%d], %s, %s); \n",
                 i, scaling[c],
                 i, i, minValue, maxChroma);
        }
    }

    GLSL("} \n");
    return true;
}
