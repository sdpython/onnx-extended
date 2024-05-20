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

#pragma once

#if defined(__SSSE3__)

#include <immintrin.h>

#endif

namespace cpu_fpemu {

#if defined(__SSSE3__)

inline float __double2float_rn(double inval) {
  float out[4] = {0};
  __m128 vout = _mm_cvtpd_ps(_mm_set1_pd(inval));

  _mm_store_ps(&out[0], vout);
  return out[0];
}

#ifdef _WIN32

inline unsigned short __float2half_rn(float inval) {
  __m128i m = _mm_cvtps_ph(_mm_set_ss(inval), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  return _mm_extract_epi16(m, 0);
}

inline float __half2float(unsigned short h_val) {
  __m128i m = _mm_cvtsi32_si128(h_val);
  return _mm_cvtss_f32(_mm_cvtph_ps(m));
}

#else

inline unsigned short __float2half_rn(float inval) {
  return _cvtss_sh(inval, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

inline float __half2float(unsigned short h_val) { return _cvtsh_ss(h_val); }

#endif

#endif

} // namespace cpu_fpemu
