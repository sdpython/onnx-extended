
################
# initialization
################
# output variables Cython_FOUND, Cython_VERSION
# function cython_add_module

if(MSVC)
  find_package(Python)
else()
  find_package(Python REQUIRED COMPONENTS NumPy)
endif()

execute_process(COMMAND ${Python_EXECUTABLE} -m cython --version
              OUTPUT_VARIABLE CYTHON_version_output
              ERROR_VARIABLE CYTHON_version_error
              RESULT_VARIABLE CYTHON_version_result
              OUTPUT_STRIP_TRAILING_WHITESPACE
              ERROR_STRIP_TRAILING_WHITESPACE)

if(NOT ${CYTHON_version_result} EQUAL 0)
    set(Cython_found_var 0)
    set(Cython_VERSION "?")
else()
    set(Cython_VERSION ${CYTHON_version_error})
    set(Cython_found_var 1)
endif()

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NUMPY_NOT_FOUND
)
if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "Numpy headers not found.")
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(
    Cython
    VERSION_VAR Cython_VERSION
    REQUIRED_VARS NUMPY_INCLUDE_DIR)

##########################
# function compile_cython
##########################

function(compile_cython filename pyx_file_cpp)
  message(STATUS "cython cythonize '${filename}'")
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${pyx_file_cpp}
    COMMAND ${Python_EXECUTABLE} -m cython -3 --cplus ${CMAKE_CURRENT_SOURCE_DIR}/${filename}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${filename})
  message(STATUS "cython cythonize '${filename}' - done")
endfunction()

############################
# function cython_add_module
############################

function(cython_add_module name pyx_file omp_lib)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SOURCES DEPS)
  message(STATUS "cython module '${name}': ${pyx_file} ++ ${ARGN}")
  get_filename_component(pyx_dir ${pyx_file} DIRECTORY)
  
  # cythonize

  compile_cython(${pyx_file} ${pyx_dir}/${name}.cpp)
  list(APPEND ARGN ${pyx_dir}/${name}.cpp)
  
  # adding the library
  
  message(STATUS "cython all files: ${ARGN}")
  add_library(${name} MODULE ${ARGN})

  target_include_directories(${name} PRIVATE
    ${Python_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
    ${Python_NumPy_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR})
  target_link_libraries(${name} PRIVATE
    ${Python_LIBRARIES}
    ${Python_NumPy_LIBRARIES}
    ${omp_lib})
  
  set_target_properties(${name}
    PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
               SUFFIX "${PYTHON_MODULE_EXTENSION}")

  #install(TARGETS ${name} LIBRARY DESTINATION ${pyx_dir})

  message(STATUS "cython added module '${name}'")
endfunction()
