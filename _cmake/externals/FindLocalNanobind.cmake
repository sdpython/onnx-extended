# Fichier pour trouver et configurer nanobind

include(FetchContent)

set(NANOBIND_TAG "v2.11.0")
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        ${NANOBIND_TAG}
)
FetchContent_GetProperties(nanobind)
if(NOT nanobind_POPULATED)
  FetchContent_Populate(nanobind)
  message(STATUS "nanobind_SOURCE_DIR=${nanobind_SOURCE_DIR}")
  message(STATUS "nanobind_BINARY_DIR=${nanobind_BINARY_DIR}")
  add_subdirectory(${nanobind_SOURCE_DIR} ${nanobind_BINARY_DIR})
else()
  message(FATAL_ERROR "Nanobind was not found.")
endif()

set(nanobind_VERSION ${nanobind_TAG})
message(STATUS "NANOBIND_OPT_SIZE=${NANOBIND_OPT_SIZE}")
message(STATUS "nanobind_INCLUDE_DIR=${nanobind_INCLUDE_DIR}")
message(STATUS "nanobind_VERSION=${nanobind_VERSION}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  LocalNanobind
  VERSION_VAR nanobind_VERSION
  REQUIRED_VARS nanobind_SOURCE_DIR nanobind_BINARY_DIR)

#! local_nanobind_add_module : compile a nanobind extension
#
# \arg:name extension name
# \arg:omp_lib omp library to link with
# \argn: additional c++ files to compile
#
function(local_nanobind_add_module name omp_lib)
  message(STATUS "nanobind module '${name}': ++ ${ARGN}")
  nanobind_add_module(${name} NB_STATIC STABLE_ABI LTO FREE_THREADED ${ARGN})
  target_include_directories(
    ${name} PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${PYTHON3_INCLUDE_DIR}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${nanobind_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIR}
    ${OMP_INCLUDE_DIR})
  target_include_directories(nanobind-static PRIVATE ${Python3_INCLUDE_DIRS})
  target_link_libraries(
    ${name} PRIVATE
    nanobind-static
    ${Python3_LIBRARY_RELEASE}  # use ${Python3_LIBRARIES} if python debug
    ${Python3_NumPy_LIBRARIES}
    ${omp_lib})
  # if(MSVC) target_link_libraries(${target_name} PRIVATE
  # nanobind::windows_extras nanobind::lto) endif()
  # target_include_directories(nanobind PRIVATE ${Python3_INCLUDE_DIRS})
  set_target_properties(
    ${name} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION ON
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")
  message(STATUS "nanobind added module '${name}'")
  get_target_property(prop ${name} BINARY_DIR)
  message(STATUS "nanobind added into '${prop}'.")
endfunction()

#
#! cuda_nanobind_add_module : compile a pyx file into cpp
#
# \arg:name extension name
# \arg:nanofine nanobind extension
# \argn: additional c++ files to compile as the cuda extension
#
function(cuda_nanobind_add_module name nanofine)
  local_nanobind_add_module(${name} OpenMP::OpenMP_CXX ${nanofine} ${ARGN})
  target_compile_definitions(
    ${name}
    PRIVATE
    CUDA_VERSION=${CUDA_VERSION_INT}
    PYTHON_MANYLINUX=${PYTHON_MANYLINUX})
  target_include_directories(${name} PRIVATE ${CUDA_INCLUDE_DIRS})
  message(STATUS "    LINK ${name} <- stdc++ ${CUDA_LIBRARIES}")
  target_link_libraries(${name} PRIVATE stdc++ ${CUDA_LIBRARIES})
  if(USE_NVTX)
    message(STATUS "    LINK ${name} <- nvtx3-cpp")
    target_link_libraries(${name} PRIVATE nvtx3-cpp)
  endif()

  # add property --use_fast_math to cu files
  # set(NEW_LIST ${name}_src_files)
  # list(APPEND ${name}_cu_files ${ARGN})
  # list(FILTER ${name}_cu_files INCLUDE REGEX ".+[.]cu$")
  # set_source_files_properties(
  #   ${name}_cu_files PROPERTIES COMPILE_OPTIONS "--use_fast_math")
endfunction()
