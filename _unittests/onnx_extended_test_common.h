#pragma once

#include "test_constants.h"
#include <cstddef>
#include <stdint.h>
#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

#define ASSERT_THROW( condition ) {                                 \
  if( !( condition ) ) {                                            \
    throw std::runtime_error(   std::string( __FILE__ )             \
                              + std::string( ":" )                  \
                              + std::to_string( __LINE__ )          \
                              + std::string( " in " )               \
                              + std::string( __FUNCTION__ )         \
    );                                                              \
  }                                                                 \
}

#define ASSERT_EQUAL( a, b ) {                                      \
  if( a != b ) {                                                    \
    throw std::runtime_error(   std::string( __FILE__ )             \
                              + std::string( ":" )                  \
                              + std::to_string( __LINE__ )          \
                              + std::string( " in " )               \
                              + std::string( __FUNCTION__ )         \
                              + std::string("\n")                   \
                              + std::string("a != b")               \
    );                                                              \
  }                                                                 \
}

template <typename T>
bool check_equal(int n, T* pa, T* pb) {
    for (int i = 0; i < n; ++i) {
        if (pa[i] != pb[i])
            return false;
    }
    return true;
}

#define ASSERT_EQUAL_VECTOR(n, pa, pb) ASSERT_THROW(check_equal(n, pa, pb))

template <typename T>
bool check_almost_equal(int n, T* pa, T* pb, T precision=1e-5) {
    for (int i = 0; i < n; ++i) {
        if (pa[i] == pb[i])
            continue;
        T d = pa[i] - pb[i];
        d = d > 0 ? d : -d;
        if (d > precision)
            return false;
    }
    return true;
}

#define ASSERT_ALMOST_VECTOR(n, pa, pb, prec) ASSERT_THROW(check_almost_equal(n, pa, pb, prec))
