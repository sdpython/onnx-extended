/*----------------------------------------------------------------------------*
 * Copyright (c) 2023, Intel Corporation - All rights reserved.
 * This file is part of FP8-Emulation-Toolkit
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *----------------------------------------------------------------------------*
 * Naveen Mellempudi (Intel Corporation)
 *----------------------------------------------------------------------------*/
// taken from
// https://github.com/IntelLabs/FP8-Emulation-Toolkit/blob/main/mpemu/pytquant/cuda/fpemu_kernels.cu

#include "cuda_fpemu.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CUBLOCK_SIZE 256

namespace fpemu {

enum ROUNDING_MODES {
  ROUND_RTZ = 0,
  ROUND_RNE = 1,
  ROUND_STOCHASTIC = 2,
  ROUND_RNAZ = 3,
  ROUND_RNTZ = 4,
  ROUND_PINF = 5,
  ROUND_NINF = 6
};

typedef union half_t {
  unsigned short u;
  __half f;
} __half_t;

typedef union ufloat32 {
  unsigned u;
  float f;
} __float_t;

enum FpemuMode {
  E4M3_RNE = 1,
};

template <typename scalar_in_t, typename scalar_out_t>
__global__ void E4M3_Kernel(const scalar_in_t *in,
                            scalar_out_t *__restrict__ out, const int size,
                            const scalar_in_t in_scale, bool block_norm,
                            int mbits, int exp_bits, int rmode) {
  int non_mant_bits = exp_bits + 1; /* exponent + sign */
  int lshift = 10 - (mbits - non_mant_bits);

  unsigned short rne_mask = 0;   /* round to nearest even mask */
  unsigned short rnaz_mask = 0;  /* round to nearest away from zero mask */
  unsigned short rntz_mask = 0;  /* round to nearest towards zero mask */
  unsigned short sr_mask = 0;    /* stochastic rounding mask */
  unsigned short rpinf_mask = 0; /* round to +INF */
  unsigned short rminf_mask = 0; /* round to -INF */

  if (rmode == ROUND_RNE)
    rne_mask = 1;
  if (rmode == ROUND_RNAZ)
    rnaz_mask = 1;
  if (rmode == ROUND_RNTZ)
    rntz_mask = 1;
  if (rmode == ROUND_STOCHASTIC)
    sr_mask = 1;
  if (rmode == ROUND_PINF)
    rpinf_mask = 1;
  if (rmode == ROUND_NINF)
    rminf_mask = 1;

  unsigned short mask_mant = (unsigned short)(0xFFFF << lshift);
  unsigned short grs_bitmask = 0x007F;
  unsigned short rne_tie = 0x00C0;

  extern __shared__ float sdata[];
  float scale = in_scale;

  if (block_norm == true) {
    absmax_block(in, sdata, size);
    __float_t f;
    f.f = sdata[0];
    f.u = (f.u & 0x7F800000);
    scale = 2 * f.f;
    scale /= 8.0;
  }
  float scale_reciprocal = 1.0 / scale;

  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size;
       gid += blockDim.x * gridDim.x) {
    __half_t h;
    float inval = in[gid] * scale;

    h.f = __anyfloat2half_rn(inval);
    short exp_h = (short)((h.u & 0x7C00) >> 10) - 15;
    short sign_h = (h.u & 0x8000);
    short mantissa_h = (h.u & 0x03FF);

    unsigned short can_round = ((h.u & 0x7FFF) < 0x5F00) ? 1 : 0;
    unsigned short is_normal =
        (((h.u & 0x7C00) <= 0x7800) && ((h.u & 0x7C00) >= 0x0400)) ? 1 : 0;
    unsigned short is_denorm = ((h.u & 0x7C00) == 0x0) ? 1 : 0;
    unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;

    int dshift = 0;

    if (exp_h > 8 || (can_round == 0)) {
      /* Software : saturate values above to +/-448.0 to +/-448.0 */
      mantissa_h = 0x0300;
      exp_h = 8;
      can_round = 0;
    } else if (exp_h < -9) {
      /* flush values below 1-4-3 subnormal range to zero */
      exp_h = -15;
      mantissa_h = 0;
    } else if (exp_h < -6) {
      dshift = (-6 - exp_h);
      /* handle denormals */
      mantissa_h = mantissa_h >> dshift;
      mantissa_h <<= dshift;
    }
    /* nearest rounding masks */
    unsigned short rnmask = (mantissa_h & grs_bitmask);
    unsigned short rnmask_tie = (mantissa_h & rne_tie);

    if (is_naninf == 0) {
      if (sr_mask) {
        /* stochastic with 16 seeds */
        int seed_index = (gid / 16);
        unsigned short rand = (unsigned short)_rand_xorshft128plus_with_seed(
            sptr[(seed_index % 16)]);
        /* apply stochastic rounding before truncation if sr_mask is enabled */
        mantissa_h += can_round * is_normal * (rand & 0x7F);
        /* stochastic round:  denormals --> rne rounding */
        mantissa_h +=
            can_round * is_denorm *
            (((rnmask > 0x0040) || (rnmask_tie == rne_tie)) << lshift);
      } else {
        /* round to nearest even, if rne_mask is enabled */
        mantissa_h +=
            can_round * rne_mask *
            (((rnmask > 0x0040) || (rnmask_tie == rne_tie)) << lshift);
        /* round to nearest away from zero, if rnaz_mask is enabled */
        mantissa_h += can_round * rnaz_mask * ((rnmask >= 0x0040) << lshift);
        /* round to nearest towards zero, if rntz_mask is enabled */
        mantissa_h += can_round * rntz_mask * ((rnmask > 0x0040) << lshift);
        /* round to +INF, if rpinf_mask is enabled */
        mantissa_h +=
            can_round * rpinf_mask * (h.f > 0) * ((rnmask >= 0x0040) << lshift);
        /* round to -INF, if rminf_mask is enabled */
        mantissa_h +=
            can_round * rminf_mask * (h.f < 0) * ((rnmask >= 0x0040) << lshift);
      }
    }
    /* truncation */
    mantissa_h &= mask_mant;
    mantissa_h += ((exp_h + 15) << 10);
    mantissa_h |= sign_h;
    h.u = mantissa_h;
    __half2anyfloat(h.f * scale_reciprocal, &out[gid]);
  }
}

void fpemu_cuda_forward(const int size, const float *input, uint8_t *output,
                        FpemuMode mode, bool inplace, float scale,
                        bool block_norm, int block_size) {

  int sdata_size = 0;
  int threads = CUBLOCK_SIZE;
  if (block_norm == true && block_size != size) {
    if (size % block_size) {
      block_norm = false;
    } else {
      threads = block_size;
      sdata_size = threads * sizeof(float);
    }
  }
  const dim3 blocks((size + (threads - 1)) / threads);

  if (mode == FpemuMode.E4M3_RNE) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input, FpemuMode.E4M3_RNE, ([&] {
          E4M3_Kernel<float><<<blocks, threads, sdata_size>>>(
              input, output, size, scale, block_norm, 8, 4, ROUND_RNE);
        }));
    checkCudaErrors(cudaDeviceSynchronize());
  }
}

} // namespace fpemu