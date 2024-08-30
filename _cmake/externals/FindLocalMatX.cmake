#
# initialization
#
# defines matx matx_SOURCE_DIR matx_BINARY_DIR

#
# matx
#

set(matx_TAG "v0.8.0")

include(FetchContent)
FetchContent_Declare(
  matx
  GIT_REPOSITORY https://github.com/NVIDIA/matx
  GIT_TAG ${matx_TAG})

FetchContent_MakeAvailable(matx)
FetchContent_GetProperties(matx)

set(matx_VERSION ${matx_TAG})
set(MATX_INCLUDE_DIR "${matx_SOURCE_DIR}/include")
message(STATUS "matx_BINARY_DIR=${matx_BINARY_DIR}")
message(STATUS "matx_SOURCE_DIR=${matx_SOURCE_DIR}")
message(STATUS "MATX_INCLUDE_DIR=${MATX_INCLUDE_DIR}")
message(STATUS "matx_VERSION=${matx_VERSION}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  LocalMatX
  VERSION_VAR matx_VERSION
  REQUIRED_VARS matx_SOURCE_DIR matx_BINARY_DIR)
