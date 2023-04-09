
################
# initialization
################
# defines LocalPyBind11 pybind11_SOURCE_DIR pybind11_BINARY_DIR
# define local_pybind11_add_module 


find_package(Python COMPONENTS Interpreter Development)
if (NOT Python_FOUND)
    message(FATAL_ERROR "Python was not found.")
endif()

##########
# pybind11
##########

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.10.4
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
else()
    message(FATAL_ERROR "Pybind11 was not found.")
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(
    LocalPyBind11
    VERSION_VAR pybind11_VERSION
    REQUIRED_VARS pybind11_SOURCE_DIR pybind11_BINARY_DIR)

##########
# function
##########

function(local_pybind11_add_module name omp_lib)
    message(STATUS "pybind11 module '${name}': ${pyx_file} ++ ${ARGN}")
    Python_add_library(${name} MODULE ${ARGN})
    target_include_directories(${name} PRIVATE
        ${Python_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIR}
        ${Python_NumPy_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIR})
    target_link_libraries(${name} PRIVATE
        pybind11::headers
        ${Python_LIBRARIES}
        ${Python_NumPy_LIBRARIES}
        ${omp_lib})
    set_target_properties(${name}
        PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON
                   CXX_VISIBILITY_PRESET ON
                   VISIBILITY_INLINES_HIDDEN ON)
    set_target_properties(${name}
        PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                   SUFFIX "${PYTHON_MODULE_EXTENSION}")
    message(STATUS "pybind11 added module '${name}'")
endfunction()
