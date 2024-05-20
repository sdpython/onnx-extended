#include "onnx_extended/validation/cpu/cpu_fpemu.hpp"
#include "onnx_extended_helpers.h"
#include "onnx_extended_test_common.h"

using namespace cpu_fpemu;

void test_cast() {

#if defined(__SSSE3__)

  float f = 1.f;
  double d = 1.f;
  float ff = __double2float_rn(d);
  ASSERT_THROW(f == ff);
  unsigned short u = __float2half_rn(f);
  float bu = __half2float(u);
  ASSERT_THROW(f == bu);

#endif

}

int main(int, char**) {
  test_cast();
}
