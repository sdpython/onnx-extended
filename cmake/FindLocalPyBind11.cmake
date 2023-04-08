
################
# initialization
################
# defines LocalPyBind11 pybind11_SOURCE_DIR pybind11_BINARY_DIR
# define local_pybind11_add_module 


find_package(Python COMPONENTS Interpreter Development)
if (Python_FOUND)
    message("-- Found Python ${Python_VERSION}.")
    message("-- Python_INCLUDE_DIRS=${Python_INCLUDE_DIRS}")
    message("-- Python Libraries: ${Python_LIBRARIES}")
else()
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
    REQUIRED_VARS pybind11_SOURCE_DIR pybind11_BINARY_DIR)

##########
# function
##########

function(local_pybind11_add_module name)
    message("-- pybind11 module '${name}': ${pyx_file}")
    message("-- pybind11 files: ${ARGN}")
    Python_add_library(${name} MODULE ${ARGN})
    target_link_libraries(${name} PRIVATE pybind11::headers)
    set_target_properties(${name}
        PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON
                   CXX_VISIBILITY_PRESET ON
                   VISIBILITY_INLINES_HIDDEN ON)
    set_target_properties(${name}
        PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                   SUFFIX "${PYTHON_MODULE_EXTENSION}")

endfunction()
