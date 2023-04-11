#pragma once

#include <cstddef>
#include <stdint.h>
#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

#define ASSERT_THROW( condition ) \
{                                                                   \
  if( !( condition ) ) {                                            \
    throw std::runtime_error(   std::string( __FILE__ )             \
                              + std::string( ":" )                  \
                              + std::to_string( __LINE__ )          \
                              + std::string( " in " )               \
                              + std::string( __FUNCTION__ )  \
    );                                                              \
  }                                                                 \
}
