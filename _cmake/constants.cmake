#
# python extension
#
if(MSVC)
  set(DLLEXT "dll")
elseif(APPLE)
  set(DLLEXT "dylib")
else()
  set(DLLEXT "so")
endif()

#
# C++ 14 or C++ 17
#
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
elseif(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
  if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "15")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  elseif(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "11")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  elseif(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "9")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  elseif(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "6")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  else()
    message(FATAL_ERROR "gcc>=6.0 is needed but "
                        "${CMAKE_C_COMPILER_VERSION} was detected.")
  endif()
  # needed to build many linux build
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lm")

  execute_process(
    COMMAND ldd --version | grep 'ldd (.*)'
    OUTPUT_VARIABLE ldd_version_output
    ERROR_VARIABLE ldd_version_error
    RESULT_VARIABLE ldd_version_result)
  message(STATUS "GLIBC_VERSION=${ldd_version_output}")
endif()

set(TEST_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/../_unittests")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test_constants.h.in
  ${TEST_FOLDER}/test_constants.h
)

#
# Compiling options
#

# AVX instructions
if(MSVC)
  # disable warning for #pragma unroll
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
  add_compile_options(/wd4068)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

if(APPLE)
  message(STATUS "APPLE: set env var for open mp: CC, CCX, LDFLAGS, CPPFLAGS")
  set(ENV{CC} "/usr/local/opt/llvm/bin/clang")
  set(ENV{CXX} "/usr/local/opt/llvm/bin/clang++")
  set(ENV(LDFLAGS) "-L/usr/local/opt/llvm/lib")
  set(ENV(CPPFLAGS) "-I/usr/local/opt/llvm/include")
endif()

message(STATUS "**********************************")
message(STATUS "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "CMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "LDFLAGS=${LDFLAGS}")
message(STATUS "CPPFLAGS=${CPPFLAGS}")
message(STATUS "DLL_EXT=${DLL_EXT}")
message(STATUS "TEST_FOLDER=${TEST_FOLDER}")
message(STATUS "CMAKE_C_COMPILER_VERSION=${CMAKE_C_COMPILER_VERSION}")
message(STATUS "**********************************")
