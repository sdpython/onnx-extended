#
# initialization
#
# defines LocalPyBind11 pybind11_SOURCE_DIR pybind11_BINARY_DIR define
# local_pybind11_add_module

#
# pybind11
#

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v2.10.4)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
  FetchContent_Populate(pybind11)
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
else()
  message(FATAL_ERROR "Pybind11 was not found.")
endif()

message(STATUS "PYBIND11_OPT_SIZE=${PYBIND11_OPT_SIZE}")
message(STATUS "pybind11_INCLUDE_DIR=${pybind11_INCLUDE_DIR}")
message(STATUS "pybind11_VERSION=${pybind11_VERSION}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  LocalPyBind11
  VERSION_VAR pybind11_VERSION
  REQUIRED_VARS pybind11_SOURCE_DIR pybind11_BINARY_DIR)

#
#! local_pybind11_add_module : compile a pybind11 extension
#
# \arg:name extension name
# \arg:omp_lib omp library to link with
# \argn: additional c++ files to compile
#
function(local_pybind11_add_module name omp_lib)
  message(STATUS "pybind11 module '${name}': ${pyx_file} ++ ${ARGN}")
  python3_add_library(${name} MODULE ${ARGN})
  target_include_directories(
    ${name} PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${PYTHON3_INCLUDE_DIR}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIR}
    ${OMP_INCLUDE_DIR})
  target_link_libraries(
    ${name} PRIVATE
    pybind11::headers
    ${Python3_LIBRARY_RELEASE}  # use ${Python3_LIBRARIES} if python debug
    ${Python3_NumPy_LIBRARIES}
    ${omp_lib})
  # if(MSVC) target_link_libraries(${target_name} PRIVATE
  # pybind11::windows_extras pybind11::lto) endif()
  set_target_properties(
    ${name} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION ON
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")
  message(STATUS "pybind11 added module '${name}'")
  get_target_property(prop ${name} BINARY_DIR)
  message(STATUS "cython added into '${prop}'.")
endfunction()
