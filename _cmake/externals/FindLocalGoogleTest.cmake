#
# initialization
#
# defines LocalGoogleTest

#
# googletest
#

set(googletest_TAG "1.17.0")

set(LOCAL_GTEST_URL "https://github.com/google/googletest/releases/download/v${googletest_TAG}/googletest-${googletest_TAG}.tar.gz")
FetchContent_Declare(googletest URL ${LOCAL_GTEST_URL})
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_MakeAvailable(googletest)
  message(STATUS "googletest_SOURCE_DIR=${googletest_SOURCE_DIR}")
  message(STATUS "googletest_BINARY_DIR=${googletest_BINARY_DIR}")
  # add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
  include(GoogleTest)
else()
  message(FATAL_ERROR "GoogleTest was not found.")
endif()

set(googletest_VERSION ${googletest_TAGTAG})
message(STATUS "googletest_INCLUDE_DIR=${googletest_INCLUDE_DIR}")
message(STATUS "googletest_VERSION=${googletest_VERSION}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  LocalGoogleTest
  VERSION_VAR googletest_VERSION
  REQUIRED_VARS googletest_SOURCE_DIR googletest_BINARY_DIR)
