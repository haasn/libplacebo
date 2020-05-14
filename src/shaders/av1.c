/*
 * This file is part of libplacebo, which is normally licensed under the terms
 * of the LGPL v2.1+. However, this file (av1.c) is also available under the
 * terms of the more permissive MIT license:
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

#include "shaders.h"

#include <libplacebo/shaders/av1.h>

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

static const char *channel_names[] = {
    [PL_CHANNEL_Y]  = "y",
    [PL_CHANNEL_CB] = "cb",
    [PL_CHANNEL_CR] = "cr",
};

// Helper function to compute some common constants
struct grain_scale {
    int grain_center;
    int grain_min;
    int grain_max;
    float texture_scale;
    float grain_scale;
};

static struct grain_scale get_grain_scale(const struct pl_av1_grain_params *params)
{
    int bit_depth = PL_DEF(params->repr.bits.color_depth, 8);
    pl_assert(bit_depth >= 8);
    struct grain_scale ret = {
        .grain_center = 128 << (bit_depth - 8),
    };

    ret.grain_min = -ret.grain_center;
    ret.grain_max = (256 << (bit_depth - 8)) - 1 - ret.grain_center;

    struct pl_color_repr repr = {
        .levels = PL_COLOR_LEVELS_PC, // the grain is on an absolute scale
        .bits = params->repr.bits,
    };

    ret.texture_scale = pl_color_repr_normalize(&repr);

    // Say we add grain onto a 10-bit channel in a 16-bit texture. The color
    // samples would be in the scale [0, 1/64], whereas the code as written is
    // designed to produce (correctly rounded) values on the scale [0, 2^b). So
    // the net result is that the grain, in this example, is too high by a
    // factor of 64 * (2^b - 1). We need to divide out by the product of both
    // factors in order to allow adding the grain values directly onto the
    // color texture.
    float range = ret.texture_scale * ((1 << bit_depth) - 1);
    ret.grain_scale = 1.0 / range;

    return ret;
}

// Generates the basic grain table (LumaGrain in the spec).
static void generate_grain_y(float out[GRAIN_HEIGHT_LUT][GRAIN_WIDTH_LUT],
                             int16_t buf[GRAIN_HEIGHT][GRAIN_WIDTH],
                             const struct pl_av1_grain_params *params)
{
    const struct pl_av1_grain_data *data = &params->data;
    struct grain_scale scale = get_grain_scale(params);
    uint16_t seed = data->grain_seed;
    int bit_depth = PL_DEF(params->repr.bits.color_depth, 8);
    int shift = 12 - bit_depth + data->grain_scale_shift;
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
            grain = PL_MAX(scale.grain_min, PL_MIN(scale.grain_max, grain));
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
                              enum pl_channel channel,
                              const struct pl_av1_grain_params *params)
{
    const struct pl_av1_grain_data *data = &params->data;
    struct grain_scale scale = get_grain_scale(params);
    int bit_depth = PL_DEF(params->repr.bits.color_depth, 8);
    int shift = 12 - bit_depth + data->grain_scale_shift;
    pl_assert(shift >= 0);

    uint16_t seed = data->grain_seed;
    if (channel == PL_CHANNEL_CB) {
        seed ^= 0xb524;
    } else if (channel == PL_CHANNEL_CR) {
        seed ^= 0x49d8;
    }

    int chromaW = params->sub_x ? SUB_GRAIN_WIDTH  : GRAIN_WIDTH;
    int chromaH = params->sub_y ? SUB_GRAIN_HEIGHT : GRAIN_HEIGHT;

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
            int sum = 0;
            for (int dy = -ar_lag; dy <= 0; dy++) {
                for (int dx = -ar_lag; dx <= ar_lag; dx++) {
                    // For the final (current) pixel, we need to add in the
                    // contribution from the luma grain texture
                    if (!dx && !dy) {
                        if (!data->num_points_y)
                            break;
                        int luma = 0;
                        int lumaX = ((x - ar_pad) << params->sub_x) + ar_pad;
                        int lumaY = ((y - ar_pad) << params->sub_y) + ar_pad;
                        for (int i = 0; i <= params->sub_y; i++) {
                            for (int j = 0; j <= params->sub_x; j++) {
                                luma += buf_y[lumaY + i][lumaX + j];
                            }
                        }
                        luma = round2(luma, params->sub_x + params->sub_y);
                        sum += luma * (*coeff);
                        break;
                    }

                    sum += *(coeff++) * buf[y + dy][x + dx];
                }
            }

            int16_t grain = buf[y][x] + round2(sum, data->ar_coeff_shift);
            grain = PL_MAX(scale.grain_min, PL_MIN(scale.grain_max, grain));
            buf[y][x] = grain;
        }
    }

    int lutW = GRAIN_WIDTH_LUT >> params->sub_x;
    int lutH = GRAIN_HEIGHT_LUT >> params->sub_y;
    int padX = params->sub_x ? SUB_GRAIN_PAD_LUT : GRAIN_PAD_LUT;
    int padY = params->sub_y ? SUB_GRAIN_PAD_LUT : GRAIN_PAD_LUT;

    for (int y = 0; y < lutH; y++) {
        for (int x = 0; x < lutW; x++) {
            int16_t grain = buf[y + padY][x + padX];
            out[y * lutW + x] = grain * scale.grain_scale;
        }
    }
}

static void generate_offsets(uint32_t *buf, int offsets_x, int offsets_y,
                             const struct pl_av1_grain_data *data)
{
    for (int y = 0; y < offsets_y; y++) {
        uint16_t state = data->grain_seed;
        state ^= ((y * 37 + 178) & 0xFF) << 8;
        state ^= ((y * 173 + 105) & 0xFF);

        for (int x = 0; x < offsets_x; x++) {
            uint32_t *offsets = &buf[y * offsets_x + x];

            uint8_t val = get_random_number(8, &state);
            uint8_t val_l = x ? (offsets - 1)[0] : 0;
            uint8_t val_t = y ? (offsets - offsets_x)[0] : 0;
            uint8_t val_tl = x && y ? (offsets - offsets_x - 1)[0] : 0;

            // Encode four offsets into a single 32-bit integer for
            // the convenience of the GPU. That way only one SSBO read is
            // required for the entire block.
            *offsets = ((uint32_t) val_tl << OFFSET_TL)
                     | ((uint32_t) val_t  << OFFSET_T)
                     | ((uint32_t) val_l  << OFFSET_L)
                     | ((uint32_t) val    << OFFSET_N);
        }
    }
}

static void generate_scaling(void *priv, float *data, int w, int h, int d)
{
    assert(w == SCALING_LUT_SIZE);

    struct {
        int num;
        uint8_t (*points)[2];
        const struct pl_av1_grain_data *data;
    } *ctx = priv;

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
    for (int i = ctx->points[ctx->num - 1][0]; i < w; i++)
        data[i] = ctx->points[ctx->num - 1][1] / range;
}

static void sample(struct pl_shader *sh, enum offset off, enum pl_channel c,
                   const struct pl_av1_grain_params *params)
{
    static const struct { int dx, dy; } delta[] = {
        [OFFSET_TL] = {1, 1},
        [OFFSET_T]  = {0, 1},
        [OFFSET_L]  = {1, 0},
        [OFFSET_N]  = {0, 0},
    };

    bool luma = c == PL_CHANNEL_Y;
    int sub_x = luma ? 0 : params->sub_x;
    int sub_y = luma ? 0 : params->sub_y;

    GLSL("offset = %du * uvec2((data >> %d) & 0xFu,   \n"
         "                     (data >> %d) & 0xFu);  \n"
         "pos = offset + local_id.xy + uvec2(%d, %d); \n"
         "val = grain_%s[ pos.y * %du + pos.x ];      \n",
         luma ? 2 : 1, off + 4, off,
         (BLOCK_SIZE >> sub_x) * delta[off].dx,
         (BLOCK_SIZE >> sub_y) * delta[off].dy,
         channel_names[c], GRAIN_WIDTH_LUT >> sub_x);
}

static void get_grain_for_channel(struct pl_shader *sh, enum pl_channel c,
                                  const struct pl_av1_grain_params *params)
{
    if (c == PL_CHANNEL_NONE)
        return;

    const struct pl_av1_grain_data *data = &params->data;
    sample(sh, OFFSET_N, c, params);
    GLSL("grain = val; \n");

    if (data->overlap) {
        int sub_x = c == PL_CHANNEL_Y ? 0 : params->sub_x;
        int sub_y = c == PL_CHANNEL_Y ? 0 : params->sub_y;
        const char *weights[] = { "vec2(27.0, 17.0)", "vec2(23.0, 22.0)" };

        // X-direction overlapping
        GLSL("if (block_id.x > 0u && local_id.x < %du) {    \n"
             "vec2 w = %s / 32.0;                           \n"
             "if (local_id.x == 1u) w.xy = w.yx;            \n",
             2 >> sub_x, weights[sub_x]);
        sample(sh, OFFSET_L, c, params);
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
                      sample(sh, OFFSET_TL, c, params);
        GLSL("        float tmp = val;                      \n");
                      sample(sh, OFFSET_T, c, params);
        GLSL("        val = dot(vec2(tmp, val), w2);        \n"
             "    } else {                                  \n");
                      sample(sh, OFFSET_T, c, params);
        GLSL("    }                                         \n"
             "grain = dot(vec2(val, grain), w);             \n"
             "}                                             \n");

        // Correctly clip the interpolated grain
        struct grain_scale scale = get_grain_scale(params);
        GLSL("grain = clamp(grain, %f, %f); \n",
             scale.grain_min * scale.grain_scale,
             scale.grain_max * scale.grain_scale);
    }
}

struct sh_grain_obj {
    // SSBO state and layout
    struct pl_buf_pool ssbos;
    struct pl_shader_desc desc;
    struct pl_var_layout layout_y;
    struct pl_var_layout layout_cb;
    struct pl_var_layout layout_cr;
    struct pl_var_layout layout_off;
    void *tmp; // to hold `desc`'s contents

    // LUT objects for the scaling luts
    struct pl_shader_obj *scaling[3];

    // Previous parameters used to check reusability
    int num_offsets;
    int chroma_lut_size;
    uint32_t *offsets;
    struct pl_av1_grain_data data;
    int sub_x, sub_y;

    // Space to store the temporary arrays, reused
    float grain[GRAIN_HEIGHT_LUT][GRAIN_WIDTH_LUT];
    int16_t grain_tmp_y[GRAIN_HEIGHT][GRAIN_WIDTH];
    int16_t grain_tmp_uv[GRAIN_HEIGHT][GRAIN_WIDTH];
};

static void sh_grain_uninit(const struct pl_gpu *gpu, void *ptr)
{
    struct sh_grain_obj *obj = ptr;
    pl_buf_pool_uninit(gpu, &obj->ssbos);
    for (int i = 0; i < 3; i++)
        pl_shader_obj_destroy(&obj->scaling[i]);
    *obj = (struct sh_grain_obj) {0};
}

bool pl_needs_av1_grain(const struct pl_av1_grain_params *params)
{
    const struct pl_av1_grain_data *data = &params->data;
    bool has_luma = data->num_points_y > 0;
    bool has_chroma = data->num_points_uv[0] > 0 ||
                      data->num_points_uv[1] > 0 ||
                      data->chroma_scaling_from_luma;

    int chmask = 0;
    for (int i = 0; i < 3; i++) {
        if (params->channels[i] != PL_CHANNEL_NONE)
            chmask |= 1 << params->channels[i];
    }

    bool is_luma = chmask & 0x1;
    bool is_chroma = chmask & 0x6;
    return (has_luma && is_luma) || (has_chroma && is_chroma);
}

bool pl_shader_av1_grain(struct pl_shader *sh,
                         struct pl_shader_obj **grain_state,
                         const struct pl_av1_grain_params *params)
{
    pl_assert(params->luma_tex);
    int width = params->luma_tex->params.w,
        height = params->luma_tex->params.h;

    if (!pl_needs_av1_grain(params)) {
        PL_DEBUG(sh, "pl_shader_av1_grain called but no AV1 grain needs to be "
                 "applied, consider testing with `pl_needs_av1_grain` first!");
        return true; // nothing to do
    }

    const struct pl_av1_grain_data *data = &params->data;
    bool has_luma = data->num_points_y > 0;
    bool has_chroma = data->num_points_uv[0] > 0 ||
                      data->num_points_uv[1] > 0 ||
                      data->chroma_scaling_from_luma;

    int chmask = 0;
    for (int i = 0; i < 3; i++) {
        if (params->channels[i] != PL_CHANNEL_NONE)
            chmask |= 1 << params->channels[i];
    }

    bool is_luma = chmask & 0x1;
    bool is_chroma = chmask & 0x6;
    if (is_luma && is_chroma && (params->sub_x || params->sub_y)) {
        PL_ERR(sh, "pl_shader_av1_grain can't be called on luma and chroma "
               "at the same time with subsampled chroma! Please only call "
               "this function on the correctly sized textures.");
        return false;
    }

    if (!sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0))
        return false;

    const struct pl_gpu *gpu = SH_GPU(sh);
    if (!gpu) {
        PL_ERR(sh, "pl_shader_av1_grain requires a non-NULL pl_gpu!");
        return false;
    }

    if (sh_glsl(sh).version < 130) {
        PL_ERR(sh, "pl_shader_av1_grain requires GLSL >= 130!");
        return false;
    }

    int bw = BLOCK_SIZE >> (is_chroma ? params->sub_x : 0);
    int bh = BLOCK_SIZE >> (is_chroma ? params->sub_y : 0);
    bool is_compute = sh_try_compute(sh, bw, bh, false, 0);

    struct sh_grain_obj *obj;
    obj = SH_OBJ(sh, grain_state, PL_SHADER_OBJ_AV1_GRAIN,
                 struct sh_grain_obj, sh_grain_uninit);
    if (!obj)
        return false;

    int offsets_x = PL_ALIGN2(width,  128) / 32;
    int offsets_y = PL_ALIGN2(height, 128) / 32;

    int chroma_lut_size = (GRAIN_WIDTH_LUT >> params->sub_x)
                        * (GRAIN_HEIGHT_LUT >> params->sub_y);

    // Note: In theory we could check only the parameters related to
    // luma or only related to chroma and skip updating parts of the SSBO,
    // but this is probably not worth it since the grain_seed is expected
    // to change per frame anyway.
    bool needs_update = memcmp(data, &obj->data, sizeof(*data)) != 0 ||
                        params->sub_x != obj->sub_x ||
                        params->sub_y != obj->sub_y;

    // For the scaling LUTs, we assume they'll be relatively constant
    // throughout the video so doing some extra work to avoid reinitializing
    // them constantly is probably worth it. Probably.
    bool scaling_changed = false;
    if (has_luma || data->chroma_scaling_from_luma) {
        scaling_changed |= data->num_points_y != obj->data.num_points_y;
        scaling_changed |= memcmp(data->points_y, obj->data.points_y,
                                  sizeof(data->points_y));
    }

    if (has_chroma && !data->chroma_scaling_from_luma) {
        for (int i = 0; i < 2; i++) {
            scaling_changed |= data->num_points_uv[i] !=
                               obj->data.num_points_uv[i];
            scaling_changed |= memcmp(data->points_uv[i],
                                      obj->data.points_uv[i],
                                      sizeof(data->points_uv[i]));
        }
    }

    if (offsets_x * offsets_y != obj->num_offsets ||
        chroma_lut_size != obj->chroma_lut_size)
    {
        // (Re-)generate the SSBO layout
        if (obj->tmp) {
            talloc_free_children(obj->tmp);
        } else {
            obj->tmp = talloc_new(obj);
        }

        obj->desc = (struct pl_shader_desc) {
            .desc = {
                .name   = "AV1GrainBuf",
                .type   = PL_DESC_BUF_STORAGE,
                .access = PL_DESC_ACCESS_READONLY,
            },
        };

        bool ok = true;

        // Note: We generate these variables unconditionally, because it
        // may be the case that one frame only has luma grain and another
        // only chroma grain, etc.; and we want to avoid thrashing the
        // SSBO layout in this case
        struct pl_var grain_y = pl_var_float("grain_y");
        grain_y.dim_a = GRAIN_WIDTH_LUT * GRAIN_HEIGHT_LUT;
        ok &= sh_buf_desc_append(obj->tmp, gpu, &obj->desc,
                                 &obj->layout_y, grain_y);

        struct pl_var grain_cb = pl_var_float("grain_cb");
        grain_cb.dim_a = chroma_lut_size;
        ok &= sh_buf_desc_append(obj->tmp, gpu, &obj->desc,
                                 &obj->layout_cb, grain_cb);

        struct pl_var grain_cr = pl_var_float("grain_cr");
        grain_cr.dim_a = chroma_lut_size;
        ok &= sh_buf_desc_append(obj->tmp, gpu, &obj->desc,
                                 &obj->layout_cr, grain_cr);

        struct pl_var offsets = pl_var_uint("offsets");
        offsets.dim_a = offsets_x * offsets_y;
        ok &= sh_buf_desc_append(obj->tmp, gpu, &obj->desc,
                                 &obj->layout_off, offsets);

        if (!ok) {
            PL_ERR(sh, "Failed generating SSBO buffer placement: Either GPU "
                   "limits exceeded or width/height nonsensical?");
            return false;
        }

        obj->num_offsets = offsets_x * offsets_y;
        obj->chroma_lut_size = chroma_lut_size;
        TARRAY_GROW(obj, obj->offsets, obj->num_offsets);
        needs_update = true;
    }

    const struct pl_buf *ssbo;
    if (obj->desc.object && !needs_update) {
        ssbo = obj->desc.object;
    } else {
        needs_update = true;

        // Get the next free SSBO buffer, regenerate if necessary
        struct pl_buf_params ssbo_params = {
            .type = PL_BUF_STORAGE,
            .size = sh_buf_desc_size(&obj->desc),
            .host_writable = true,
        };

        const struct pl_buf *last_buf = obj->desc.object;
        if (last_buf && last_buf->params.size > ssbo_params.size)
            ssbo_params.size = last_buf->params.size;

        ssbo = pl_buf_pool_get(gpu, &obj->ssbos, &ssbo_params);
        if (!ssbo) {
            SH_FAIL(sh, "Failed creating/getting SSBO buffer for AV1 grain!");
            return false;
        }

        obj->desc.object = ssbo;
    }

    if (needs_update) {
        // This is needed even for chroma
        generate_grain_y(obj->grain, obj->grain_tmp_y, params);

        if (has_luma) {
            pl_assert(obj->layout_y.stride == sizeof(float));
            pl_buf_write(gpu, ssbo, obj->layout_y.offset, obj->grain,
                         sizeof(obj->grain));
        }

        if (has_chroma) {
            generate_grain_uv(&obj->grain[0][0], obj->grain_tmp_uv,
                              obj->grain_tmp_y, PL_CHANNEL_CB, params);
            pl_assert(obj->layout_cb.stride == sizeof(float));
            pl_buf_write(gpu, ssbo, obj->layout_cb.offset, obj->grain,
                         sizeof(float) * chroma_lut_size);

            generate_grain_uv(&obj->grain[0][0], obj->grain_tmp_uv,
                              obj->grain_tmp_y, PL_CHANNEL_CR, params);
            pl_assert(obj->layout_cr.stride == sizeof(float));
            pl_buf_write(gpu, ssbo, obj->layout_cr.offset, obj->grain,
                         sizeof(float) * chroma_lut_size);
        }

        generate_offsets(obj->offsets, offsets_x, offsets_y, data);
        pl_assert(obj->layout_off.stride == sizeof(uint32_t));
        pl_buf_write(gpu, ssbo, obj->layout_off.offset, obj->offsets,
                     obj->num_offsets * obj->layout_off.stride);

        obj->data = *data;
        obj->sub_x = params->sub_x;
        obj->sub_y = params->sub_y;
    }

    // Update the scaling LUTs
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

        if (priv.num > 0) {
            scaling[i] = sh_lut(sh, &obj->scaling[i], SH_LUT_LINEAR,
                                SCALING_LUT_SIZE, 0, 0, 1, scaling_changed,
                                &priv, generate_scaling);

            if (!scaling[i]) {
                SH_FAIL(sh, "Failed generating/uploading scaling LUTs!");
                return false;
            }
        }
    }

    // Attach the SSBO
    sh_desc(sh, obj->desc);

    GLSL("// pl_shader_av1_grain \n"
         "{                      \n"
         "uvec2 offset;          \n"
         "uvec2 pos;             \n"
         "float val;             \n"
         "float grain;           \n");

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
    GLSL("uint data = offsets[block_id.y * %du + block_id.x]; \n", offsets_x);

    // If we need access to the external luma plane, load it now
    if (is_chroma) {
        GLSL("float averageLuma; \n");
        if (is_luma) {
            // We already have the luma channel as part of the pre-sampled color
            for (int i = 0; i < 3; i++) {
                if (params->channels[i] == PL_CHANNEL_Y) {
                    GLSL("averageLuma = color[%d]; \n", i);
                    break;
                }
            }
        } else {
            // Luma channel not present in image, attach it separately
            ident_t luma = sh_desc(sh, (struct pl_shader_desc) {
                .desc = (struct pl_desc) {
                    .name = "luma",
                    .type = PL_DESC_SAMPLED_TEX,
                },
                .object = params->luma_tex,
            });

            GLSL("pos = local_id * uvec2(%d, %d0);               \n"
                 "averageLuma = texelFetch(%s, ivec2(pos), 0).r; \n",
                 1 << params->sub_x, 1 << params->sub_y,
                 luma);
        }
    }

    struct grain_scale scale = get_grain_scale(params);
    int bits = PL_DEF(params->repr.bits.color_depth, 8);
    pl_assert(bits >= 8);

    float minValue, maxLuma, maxChroma;
    if (pl_color_levels_guess(&params->repr) == PL_COLOR_LEVELS_TV) {
        float out_scale = (1 << bits) / ((1 << bits) - 1.0);
        out_scale /= scale.texture_scale;
        minValue  = 16  / 256.0 * out_scale;
        maxLuma   = 235 / 256.0 * out_scale;
        maxChroma = 240 / 256.0 * out_scale;
        if (!pl_color_system_is_ycbcr_like(params->repr.sys))
            maxChroma = maxLuma;
    } else {
        minValue  = 0.0;
        maxLuma   = 1.0 / scale.texture_scale;
        maxChroma = maxLuma;
    }

    for (int i = 0; i < 3; i++) {
        enum pl_channel c = params->channels[i];
        if (c == PL_CHANNEL_NONE)
            continue;
        if (!scaling[c])
            continue;

        get_grain_for_channel(sh, c, params);

        if (c == PL_CHANNEL_Y) {
            GLSL("color[%d] += %s(color[%d] * %f) * grain; \n"
                 "color[%d] = clamp(color[%d], %f, %f);    \n",
                 i, scaling[c], i, scale.texture_scale,
                 i, i, minValue, maxLuma);
        } else {
            GLSL("val = averageLuma; \n");
            if (!data->chroma_scaling_from_luma) {
                GLSL("val = dot(vec2(val, color[%d]), vec2(%d.0, %d.0)); \n",
                     i, data->uv_mult[c - 1], data->uv_mult[c - 1]);
                int offset = data->uv_offset[c - 1] << (bits - 8);
                GLSL("val = val * 1.0/64.0 + %f; \n",
                     offset / scale.texture_scale);
            }
            GLSL("color[%d] += %s(val * %f) * grain;    \n"
                 "color[%d] = clamp(color[%d], %f, %f); \n",
                 i, scaling[c], scale.texture_scale,
                 i, i, minValue, maxChroma);
        }
    }

    GLSL("} \n");
    return true;
}
