// This file includes some pieces taken from
// https://github.com/IntelLabs/FP8-Emulation-Toolkit/blob/main/mpemu/pytquant/cuda/fpemu_kernels.cu
// with the following license.
//
/*----------------------------------------------------------------------------*
 * Copyright (c) 2023, Intel Corporation - All rights reserved.
 * This file is part of FP8-Emulation-Toolkit
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *----------------------------------------------------------------------------*
 * Naveen Mellempudi (Intel Corporation)
 *----------------------------------------------------------------------------*/

#include "cuda_fpemu.cuh"
#include "cuda_utils.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CUBLOCK_SIZE 256

namespace cuda_fpemu {

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

/* This implementation of xoroshiro128++ PRNG is borrowed from here:
 *  http://prng.di.unimi.it/xoshiro128plusplus.c
 *  main page: http://prng.di.unimi.it/
 */
__device__ static uint32_t s1[4] = {1387366120, 279844183, 888998500, 1099633400};
__device__ static uint32_t s2[4] = {2034269327, 2125325156, 1209715489, 193165672};
__device__ static uint32_t s3[4] = {1555452618, 650181557, 883695203, 62767784};
__device__ static uint32_t s4[4] = {419524804, 2146478152, 480059239, 1468956197};
__device__ static uint32_t s5[4] = {1252084877, 500390994, 977516591, 1950666000};
__device__ static uint32_t s6[4] = {393659750, 834151069, 1477014702, 734008143};
__device__ static uint32_t s7[4] = {1983400973, 116410309, 2110188261, 2019272068};
__device__ static uint32_t s8[4] = {187709636, 28336299, 419632041, 1774181187};
__device__ static uint32_t s9[4] = {702309618, 407781555, 1512057936, 1868769368};
__device__ static uint32_t s10[4] = {510001215, 966559856, 776583255, 147562106};
__device__ static uint32_t s11[4] = {127180605, 1881312534, 478635452, 814821902};
__device__ static uint32_t s12[4] = {733990058, 1889991804, 1108257970, 1093480892};
__device__ static uint32_t s13[4] = {427374380, 416747337, 558000409, 1594848927};
__device__ static uint32_t s14[4] = {444870959, 1595722866, 1064124488, 363710254};
__device__ static uint32_t s15[4] = {703721499, 389640783, 1002360059, 1427395742};
__device__ static uint32_t s16[4] = {1295231497, 1254972431, 1423497865, 861918264};

__device__ static uint32_t *sptr[16] = {s1, s2,  s3,  s4,  s5,  s6,  s7,  s8,
                                        s9, s10, s11, s12, s13, s14, s15, s16};

__device__ __forceinline__ uint32_t rotl_(const uint32_t x, int k) {
  return (x << k) | (x >> (32 - k));
}

__device__ __forceinline__ uint32_t _rand_xorshft128plus_with_seed(uint32_t *ps) {
  const uint32_t result_plus = ps[0] + ps[3];
  const uint32_t t = ps[1] << 9;

  ps[2] ^= ps[0];
  ps[3] ^= ps[1];
  ps[1] ^= ps[2];
  ps[0] ^= ps[3];

  ps[2] ^= t;

  ps[3] = rotl_(ps[3], 11);

  return result_plus;
}

template <typename scalar_t>
__device__ __forceinline__ void __half2anyfloat(__half h_, scalar_t *out) {
  scalar_t f_;

  if (std::is_same<scalar_t, double>::value) {
    f_ = (scalar_t)__half2float((__half)h_);
  } else if (std::is_same<scalar_t, float>::value) {
    f_ = __half2float(h_);
  } else if (std::is_same<scalar_t, __half>::value) {
    f_ = h_;
  }
  *out = f_;
}

template <typename scalar_t>
__device__ __forceinline__ unsigned short __anyfloat2half_rn(scalar_t f_) {
  unsigned short h_;

  if (std::is_same<scalar_t, double>::value) {
    h_ = __float2half_rn(__double2float_rn(f_));
  } else if (std::is_same<scalar_t, float>::value) {
    h_ = __float2half_rn(f_);
  } else if (std::is_same<scalar_t, __half>::value) {
    unsigned short *ptrh_ = (unsigned short *)&f_;
    h_ = *ptrh_;
  }
  return h_;
}

template <typename scalar_t> __device__ __forceinline__ float __anyfloat2float_rn(scalar_t a_) {
  float f_;

  if (std::is_same<scalar_t, double>::value) {
    f_ = __double2float_rn(a_);
  } else if (std::is_same<scalar_t, float>::value) {
    f_ = a_;
  } else if (std::is_same<scalar_t, __half>::value) {
    f_ = __half2float((__half)a_);
  }
  return f_;
}

template <typename scalar_t>
__device__ void absmax_block(const scalar_t *in, float *sdata, const int size) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = 0.0f;

  if (i < size) {
    sdata[tid] = fmaxf(fabsf(sdata[tid]), fabsf(__anyfloat2float_rn(in[i])));
  }
  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if ((tid % (2 * s)) == 0) {
      sdata[tid] = fmaxf(fabsf(sdata[tid]), fabsf(sdata[tid + s]));
    }
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void E4M3_Kernel(scalar_t *__restrict__ in_out, const int size,
                            const scalar_t in_scale, bool block_norm, int mbits, int exp_bits,
                            int rmode) {
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

  extern __shared__ scalar_t sdata[];
  scalar_t scale = in_scale;

  if (block_norm) {
    absmax_block(in_out, sdata, size);
    __float_t f;
    f.f = sdata[0];
    f.u = (f.u & 0x7F800000);
    scale = 2 * f.f;
    scale /= 8.0;
  }

  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size;
       gid += blockDim.x * gridDim.x) {

    __half_t h;
    scalar_t inval = in_out[gid] * scale;

    h.u = __anyfloat2half_rn(inval);
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
        unsigned short rand =
            (unsigned short)_rand_xorshft128plus_with_seed(sptr[(seed_index % 16)]);
        /* apply stochastic rounding before truncation if sr_mask is enabled */
        mantissa_h += can_round * is_normal * (rand & 0x7F);
        /* stochastic round:  denormals --> rne rounding */
        mantissa_h +=
            can_round * is_denorm * (((rnmask > 0x0040) || (rnmask_tie == rne_tie)) << lshift);
      } else {
        /* round to nearest even, if rne_mask is enabled */
        mantissa_h +=
            can_round * rne_mask * (((rnmask > 0x0040) || (rnmask_tie == rne_tie)) << lshift);
        /* round to nearest away from zero, if rnaz_mask is enabled */
        mantissa_h += can_round * rnaz_mask * ((rnmask >= 0x0040) << lshift);
        /* round to nearest towards zero, if rntz_mask is enabled */
        mantissa_h += can_round * rntz_mask * ((rnmask > 0x0040) << lshift);
        if (h.u & 0x8000 == 0 /* h.f > 0 */) {
          /* round to +INF, if rpinf_mask is enabled */
          mantissa_h += can_round * rpinf_mask * ((rnmask >= 0x0040) << lshift);
        } else if (h.u & 0x8FFF != 0 /* h.f < 0 */) {
          /* round to -INF, if rminf_mask is enabled */
          mantissa_h += can_round * rminf_mask * ((rnmask >= 0x0040) << lshift);
        }
      }
    }
    /* truncation */
    mantissa_h &= mask_mant;
    mantissa_h += ((exp_h + 15) << 10);
    mantissa_h |= sign_h;
    h.u = mantissa_h;
    scalar_t hf;
    __half2anyfloat(h.f, &hf);
    in_out[gid] = hf / scale;
  }
}

void fpemu_cuda_forward(const int size, const float *input, float *output, FpemuMode mode,
                        bool inplace, float scale, bool block_norm, int block_size,
                        int cuda_device) {

  int threads = CUBLOCK_SIZE;
  const dim3 blocks((size + (threads - 1)) / threads);

  float *gpu_ptr;

  checkCudaErrors(cudaSetDevice(cuda_device));
  checkCudaErrors(cudaMalloc(&gpu_ptr, size * sizeof(float)));
  checkCudaErrors(cudaMemcpy(gpu_ptr, input, size * sizeof(float), cudaMemcpyHostToDevice));

  if (mode == FpemuMode::E4M3_RNE) {
    E4M3_Kernel<float><<<blocks, threads>>>(gpu_ptr, size, scale, block_norm, 8, 4, ROUND_RNE);
    checkCudaErrors(cudaDeviceSynchronize());
  } else {
    NVTE_CHECK(false, onnx_extended_helpers::MakeString("Unsupported mode ", mode,
                                                        " for this function."));
  }
  checkCudaErrors(cudaMemcpy(output, gpu_ptr, size * sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(gpu_ptr));
}

} // namespace cuda_fpemu