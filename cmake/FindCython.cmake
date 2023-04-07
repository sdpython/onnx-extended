#.rst:
#
# Find ``cython`` executable.
#
# This module will set the following variables in your project:
#
#  ``CYTHON_EXECUTABLE``
#    path to the ``cython`` program
#
#  ``CYTHON_VERSION``
#    version of ``cython``
#
#  ``CYTHON_FOUND``
#    true if the program was found
#
# For more information on the Cython project, see https://cython.org/.
#
# *Cython is a language that makes writing C extensions for the Python language
# as easy as Python itself.*
#
#=============================================================================
# Copyright 2011 Kitware, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# Use the Cython executable that lives next to the Python executable
# if it is a local installation.
if(Python_EXECUTABLE)
  get_filename_component(_python_path ${Python_EXECUTABLE} PATH)
elseif(Python3_EXECUTABLE)
  get_filename_component(_python_path ${Python3_EXECUTABLE} PATH)
elseif(DEFINED PYTHON_EXECUTABLE)
  get_filename_component(_python_path ${PYTHON_EXECUTABLE} PATH)
endif()

if(DEFINED _python_path)
  find_program(CYTHON_EXECUTABLE
               NAMES cython cython.bat cython3
               HINTS ${_python_path}
               DOC "path to the cython executable")
else()
  find_program(CYTHON_EXECUTABLE
               NAMES cython cython.bat cython3
               DOC "path to the cython executable")
endif()

if(CYTHON_EXECUTABLE)
  message("-- CYTHON_EXECUTABLE=${CYTHON_EXECUTABLE}")
  set(CYTHON_version_command ${CYTHON_EXECUTABLE} --version)

  execute_process(COMMAND ${CYTHON_version_command}
                  OUTPUT_VARIABLE CYTHON_version_output
                  ERROR_VARIABLE CYTHON_version_error
                  RESULT_VARIABLE CYTHON_version_result
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_STRIP_TRAILING_WHITESPACE)

  if(NOT ${CYTHON_version_result} EQUAL 0)
    set(_error_msg "Command \"${CYTHON_version_command}\" failed with")
    set(_error_msg "${_error_msg} output:\n${CYTHON_version_error}")
    message(SEND_ERROR "${_error_msg}")
  else()
    if("${CYTHON_version_output}" MATCHES "^[Cc]ython version ([^,]+)")
      set(CYTHON_VERSION "${CMAKE_MATCH_1}")
    else()
      if("${CYTHON_version_error}" MATCHES "^[Cc]ython version ([^,]+)")
        set(CYTHON_VERSION "${CMAKE_MATCH_1}")
      endif()
    endif()
  endif()
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Cython REQUIRED_VARS CYTHON_EXECUTABLE)

mark_as_advanced(CYTHON_EXECUTABLE)

##########################
# function compile_cython
##########################

function(compile_cython filename)
  message("-- Cythonize '${filename}'")
  execute_process(
    COMMAND ${CYTHON_EXECUTABLE} -3 --cplus ${CMAKE_CURRENT_SOURCE_DIR}/${filename}
    OUTPUT_VARIABLE cpl
    ERROR_VARIABLE err
    RESULT_VARIABLE ret)
  message("${cpl}")
  message("${err}")
  if (ret)
    message(FATAL_ERROR "Cython failed with ${ret}.")
  endif()
  message("-- Cythonized '${filename}'.")
endfunction()

############################
# function cython_add_module
############################

function(cython_add_module name ${pyx_file})
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SOURCES DEPS)
  message("-- Add cython module '${name}': ${pyx_file}")
  message("-- Additional files: ${ARGN}")
  get_filename_component(pyx_dir ${pyx_file} DIRECTORY)
  
  # cythonize

  file(MODIFIED_TIME ${pyx_file} MODIFIED_TIMESTAMP)
  CHECK_IF_MODIFIED_SINCE(${MODIFIED_TIMESTAMP} IS_MODIFIED)
  if(IS_MODIFIED)
    compile_cython(${pyx_file} CXX)
  endif()
  
  list(APPEND ARGN ${pyx_dir}/${name}.cpp)
  
  # adding the library
  
  message("-- All files: ${ARGN}")
  add_library(${name} SHARED ${ARGN})

  target_include_directories(${name} PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
  target_link_libraries(${name} PRIVATE ${Python_LIBRARIES} ${Python_NumPy_LIBRARIES})
  
  set_target_properties(${name} PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    OUTPUT_NAME "${name}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")

  install(TARGETS ${name} LIBRARY DESTINATION ${pyx_dir})

  message("-- Added cython module '${name}'")
endfunction()
