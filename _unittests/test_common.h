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
