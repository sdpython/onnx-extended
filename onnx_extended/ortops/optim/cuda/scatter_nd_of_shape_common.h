#pragma once

namespace ortops {

enum class Reduction : int {
  None = 0,
  Add = 1,
  Mul = 2,
  Min = 3,
  Max = 4,
};

enum class Strategy : int {
  None = 0,
  Optimize = 1,
};

struct Shape2 {
  int64_t dims[12];
};

} // namespace ortops
