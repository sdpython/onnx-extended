#pragma once

#include <cstdint>
#include <cstring>
#include <omp.h>

inline uint8_t float_to_e4m3fn(float v, bool saturate = true) {
  uint32_t b;
  std::memcpy(&b, &v, sizeof(b));

  uint8_t val = static_cast<uint8_t>((b & 0x80000000) >> 24); // sign
  if ((b & 0x7fc00000) == 0x7fc00000) {
    val |= 0x7f;
  } else if ((b & 0x7fffffff) == 0x7f800000) {
    if (saturate) {
      val |= 126;
    } else {
      val |= 0x7f;
    }
  } else {
    uint8_t e = static_cast<uint8_t>((b & 0x7F800000) >> 23); // exponent
    uint32_t m = static_cast<uint32_t>(b & 0x007FFFFF);       // mantissa
    if (e != 0) {
      if (e < 117) {        // 0b1110101
      } else if (e < 118) { // 0b1110110
        val |= 1;
        if ((m >> 23) & 1) {
          // rounding
          val += 1;
        }
      } else if (e < 121) { // 127 - 7 + 1 // 0b1111001
        auto d = 120 - e;   // 0b1111000
        val |= 1 << (2 - d);
        val |= m >> (21 + d);
        if ((m >> (20 + d)) & 1) {
          // rounding
          val += 1;
        }
      } else if (e < 136) { // 127 + 8 + 1 // 0b10001000
        auto ex = e - 120;  // 127 - 7
        if (ex == 0) {
          val |= 0x4;
          val |= m >> 21;
        } else {
          val |= ex << 3;
          val |= m >> 20;
          if ((val & 0x7F) == 0x7F) {
            val &= 0xFE;
          }
        }
        if ((m & 0x80000) && ((m & 0x100000) || (m & 0x7C000))) {
          if ((val & 0x7F) < 0x7E) {
            // rounding
            val += 1;
          } else if (!saturate) {
            val |= 0x7F;
          }
        }
      } else if (saturate) {
        val |= 126; // 0b01111110
      } else {
        val |= 0x7F;
      }
    }
  }
  return val;
}

inline void float_to_e4m3fn(int64_t n, const float *src, uint8_t *dst,
                            bool saturate = true) {
#pragma omp parallel for
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = float_to_e4m3fn(src[i], saturate);
  }
}

inline void float_to_e4m3fn(int64_t n, const float *src, uint8_t *dst,
                            float scale, bool saturate = true) {
#pragma omp parallel for
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = float_to_e4m3fn(src[i] / scale, saturate);
  }
}

inline float e4m3fn_to_float(uint8_t val) {
  uint32_t res;
  if (val == 255) {
    res = 0xffc00000;
  } else if (val == 127) {
    res = 0x7fc00000;
  } else {
    uint32_t expo = (val & 0x78) >> 3;
    uint32_t mant = val & 0x07;
    uint32_t sign = val & 0x80;
    res = sign << 24;
    if (expo == 0) {
      if (mant > 0) {
        expo = 0x7F - 7;
        if ((mant & 0x4) == 0) {
          mant &= 0x3;
          mant <<= 1;
          expo -= 1;
        }
        if ((mant & 0x4) == 0) {
          mant &= 0x3;
          mant <<= 1;
          expo -= 1;
        }
        res |= (mant & 0x3) << 21;
        res |= expo << 23;
      }
    } else {
      res |= mant << 20;
      expo -= 0x7;
      expo += 0x7F;
      res |= expo << 23;
    }
  }
  float float_res;
  std::memcpy(&float_res, &res, sizeof(float));
  return float_res;
}

inline void e4m3fn_to_float(int64_t n, const uint8_t *src, float *dst) {
#pragma omp parallel for
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = e4m3fn_to_float(src[i]);
  }
}
