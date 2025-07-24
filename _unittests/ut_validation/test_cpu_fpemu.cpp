#include <gtest/gtest.h>
#include "onnx_extended/validation/cpu/cpu_fpemu.hpp"
#include "onnx_extended_helpers.h"
#include "onnx_extended_test_common.h"

using namespace cpu_fpemu;

TEST(validation, cast) {
#if defined(__SSSE3__)
  float f = 1.f;
  double d = 1.f;
  float ff = __double2float_rn(d);
  EXPECT_EQ(f, ff);
  unsigned short u = __float2half_rn(f);
  float bu = __half2float(u);
  EXPECT_EQ(f, bu);
#endif
}
