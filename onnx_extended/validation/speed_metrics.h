#pragma once

#include <cstddef>
#include <stdint.h>
#include <vector>

namespace validation {

#if defined(_WIN32) || defined(WIN32)

inline bool _isnan_(float x) { return _isnanf(x); }
inline bool _isnan_(double x) { return _isnan(x); }

#elif defined(__MACOSX__) || defined(__APPLE__)

#include <cmath>
#include <math.h>

inline bool _isnan_(float x) { return (float)::isnan((double)x); }
inline bool _isnan_(double x) { return ::isnan(x); }

#else

// See
// https://stackoverflow.com/questions/2249110/how-do-i-make-a-portable-isnan-isinf-function
inline bool _isnan_(double x) {
  union {
    uint64_t u;
    double f;
  } ieee754;
  ieee754.f = x;
  return ((unsigned)(ieee754.u >> 32) & 0x7fffffff) +
             ((unsigned)ieee754.u != 0) >
         0x7ff00000;
}

inline bool _isnan_(float x) { return _isnan_((double)x); }

#endif

typedef struct ElementTime {
  int64_t trial;
  int64_t row;
  double time;
  inline ElementTime() {}
  inline ElementTime(int64_t n, int64_t r, double t) {
    trial = n;
    row = r;
    time = t;
  }
} ElementTime;

double benchmark_cache(int64_t arr_size, bool verbose);
std::vector<ElementTime>
benchmark_cache_tree(int64_t n_rows, int64_t n_features, int64_t n_trees,
                     int64_t tree_size, int64_t max_depth,
                     int64_t search_step = 64);

} // namespace validation
