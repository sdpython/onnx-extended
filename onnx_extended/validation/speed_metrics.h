#pragma once

#include <cstddef>
#include <vector>

#if !defined(__APPLE__)
#ifndef _SSIZE_T_DEFINED
typedef int64_t ssize_t;
#define _SSIZE_T_DEFINED
#endif
#endif

#if defined(_WIN32) || defined(WIN32)

inline bool _isnan_(float x) { return _isnanf(x); }
inline bool _isnan_(double x) { return _isnan(x); }

#elif defined(__MACOSX__) || defined(__APPLE__)

inline bool _isnan_(float x) { return (float)::isnan((double)x); }
inline bool _isnan_(double x) { return ::isnan(x); }

#else

// See https://stackoverflow.com/questions/2249110/how-do-i-make-a-portable-isnan-isinf-function
inline bool _isnan_(double x) {
    union { uint64_t u; double f; } ieee754;
    ieee754.f = x;
    return ((unsigned)(ieee754.u >> 32) & 0x7fffffff) +
        ((unsigned)ieee754.u != 0) > 0x7ff00000;
}

inline bool _isnan_(float x) { return _isnan_((double)x); }

#endif



typedef struct elem_time {
    int64_t n_trial;
    int64_t row;
    double time;
    inline elem_time() { }
    inline elem_time(int64_t n, int64_t r, double t) { 
        n_trial=t; row=r; time=r; 
    }
} elem_time;

double benchmark_cache(int64_t arr_size, bool verbose);
std::vector<elem_time> benchmark_cache_tree(
        int64_t n_rows, int64_t n_features, int64_t n_trees, int64_t tree_size,
        int64_t max_depth, int64_t n_trials);
