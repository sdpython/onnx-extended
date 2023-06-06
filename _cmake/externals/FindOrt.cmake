#
# initialization
#
# downloads onnxruntime as a binary
# functions ort_add_dependency, ort_add_custom_op

if(NOT ORT_VERSION)
  set(ORT_VERSION 1.15.0)
endif()
string(LENGTH "${ORT_VERSION}" ORT_VERSION_LENGTH)

if(CUDA_FOUND)
  if(APPLE)
    message(WARNING "onnxruntime-gpu not available on MacOsx")
  endif()
  set(ORT_GPU "-gpu")
else()
  set(ORT_GPU "")
endif()

if(ORT_VERSION_LENGTH LESS_EQUAL 12)
  message(STATUS "ORT - retrieve release version ${ORT_VERSION}")
  if(MSVC)
    set(ORT_NAME "onnxruntime-win-x64${ORT_GPU}-${ORT_VERSION}.zip")
    set(ORT_FOLD "onnxruntime-win-x64${ORT_GPU}-${ORT_VERSION}")
  elseif(APPLE)
    set(ORT_NAME "onnxruntime-osx-universal2-${ORT_VERSION}.tgz")
    set(ORT_FOLD "onnxruntime-osx-universal2-${ORT_VERSION}")
  else()
    set(ORT_NAME "onnxruntime-linux-x64${ORT_GPU}-${ORT_VERSION}.tgz")
    set(ORT_FOLD "onnxruntime-linux-x64${ORT_GPU}-${ORT_VERSION}")
  endif()
  set(ORT_ROOT "https://github.com/microsoft/onnxruntime/releases/download/")
  set(ORT_URL "${ORT_ROOT}v${ORT_VERSION}/${ORT_NAME}")
  set(ORT_DEST "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-download/${ORT_NAME}")
  set(ORT_DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-bin/")

  FetchContent_Declare(onnxruntime URL ${ORT_URL})
  FetchContent_makeAvailable(onnxruntime)
  set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
  set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib)
else()
  message(STATUS "ORT - retrieve development version from '${ORT_VERSION}'")
  set(ONNXRUNTIME_LIB_DIR "${ORT_VERSION}")
  set(ONNXRUNTIME_INCLUDE_DIR "${ORT_VERSION}/../../../include/onnxruntime/core/session")
  set(ORT_URL ${ORT_VERSION})
endif()

if(MSVC)
  set(DLLEXT "dll")
elseif(APPLE)
  set(DLLEXT "dylib")
else()
  set(DLLEXT "so")
endif()

find_library(ONNXRUNTIME onnxruntime HINTS "${ONNXRUNTIME_LIB_DIR}")
if(ONNXRUNTIME-NOTFOUND)
  message(FATAL_ERROR "onnxruntime cannot be found at '${ONNXRUNTIME_LIB_DIR}'")
endif()

file(GLOB ORT_LIB_FILES ${ONNXRUNTIME_LIB_DIR}/*.${DLLEXT}*)
file(GLOB ORT_LIB_HEADER ${ONNXRUNTIME_INCLUDE_DIR}/*.h)

list(LENGTH ORT_LIB_FILES ORT_LIB_FILES_LENGTH)
if (ORT_LIB_FILES_LENGTH LESS_EQUAL 1)
  message(FATAL_ERROR "No file found in '${ONNXRUNTIME_LIB_DIR}' "
                      "from url '${ORT_URL}', "
                      "found files [${ORT_LIB_FILES}].")
endif()

list(LENGTH ORT_LIB_HEADER ORT_LIB_HEADER_LENGTH)
if (ORT_LIB_HEADER_LENGTH LESS_EQUAL 1)
  message(FATAL_ERROR "No file found in '${ONNXRUNTIME_INCLUDE_DIR}' "
                      "from url '${ORT_URL}', "
                      "found files [${ORT_LIB_HEADER}]")
endif()

#
#! ort_add_dependency : copies necessary onnxruntime assembly
#                       to the location a target is build
#
# \arg:name target name
#
function(ort_add_dependency name folder_copy)
  get_target_property(target_output_directory ${name} BINARY_DIR)
  message(STATUS "ort copy ${ORT_LIB_FILES_LENGTH} files from '${ONNXRUNTIME_LIB_DIR}'")
  if(MSVC)
    set(destination_dir ${target_output_directory}/${CMAKE_BUILD_TYPE})
  else()
    set(destination_dir ${target_output_directory})
  endif()
  message(STATUS "ort copy to '${destination_dir}'")
  if(folder_copy)
    message(STATUS "ort copy to '${folder_copy}'")
  endif()
  foreach(file_i ${ORT_LIB_FILES})
    if(NOT EXISTS ${destination_dir}/${file_i})
      message(STATUS "ort copy '${file_i}' to '${destination_dir}'")
      add_custom_command(
        TARGET ${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${destination_dir})
    endif()
    if(folder_copy)
      if(NOT EXISTS ${folder_copy}/${file_i})
        message(STATUS "ort copy '${file_i}' to '${folder_copy}'")
        add_custom_command(
          TARGET ${name} POST_BUILD
          COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${folder_copy})
      endif()
    endif()
  endforeach()
  # file(COPY ${ORT_LIB_FILES} DESTINATION ${target_output_directory})
endfunction()

file(WRITE "../_setup_ext.txt" "")

#
#! ort_add_custom_op : compile a pyx file into cpp
#
# \arg:name project name
# \arg:folder where to copy the library
# \arg:provider CUDA if a cuda lib, CUBLAS to use CUDA with CUBLAS empty or CPU for CPU
# \argn: C++ file to compile
#
function(ort_add_custom_op name provider folder)
  if (provider STREQUAL "CUDA" OR provider STREQUAL "CUBLAS")
    if (provider STREQUAL "CUBLAS")
      set(link_options "${CUBLAS_LIBRARY}")
    else()
      set(link_options "")
    endif()
    message(STATUS "ort custom op CUDA: '${name}': ${ARGN}")
    set(cuda_name ${name}_cuda)
    cuda_add_library_ext(${cuda_name} STATIC ${ARGN})
    add_library(${name} SHARED ${ARGN})
    target_compile_definitions(${name} PRIVATE
      CUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR}
      CUDA_VERSION_MINOR=${CUDA_VERSION_MINOR})
    if(USE_NVTX)
      target_link_libraries(
        ${name}
        PRIVATE
        ${cuda_name}
        stdc++
        nvtx3-cpp
        ${link_options})
    else()
      target_link_libraries(
        ${name}
        PRIVATE
        ${cuda_name}
        stdc++
        ${link_options})
    endif()
    target_include_directories(
      ${cuda_name}
      PRIVATE
      ${ONNXRUNTIME_INCLUDE_DIR})
    target_include_directories(
      ${name}
      PRIVATE
      ${ONNXRUNTIME_INCLUDE_DIR}
      ${CUDA_INCLUDE_DIRS})
  else()
    message(STATUS "ort custom op CPU: '${name}': ${ARGN}")
    add_library(${name} SHARED ${ARGN})
    target_include_directories(${name} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  endif()
  set_property(TARGET ${name} PROPERTY POSITION_INDEPENDENT_CODE ON)
  get_target_property(target_file ${name} LIBRARY_OUTPUT_NAME)
  # add_custom_command(
  #   TARGET ${name} POST_BUILD
  #   COMMAND ${CMAKE_COMMAND} ARGS -E copy $<TARGET_FILE_NAME:${name}> ${folder})
  # $<TARGET_FILE_NAME:${name}> does not seem to work.
  # The following step adds a line in '_setup.txt' to tell setup.py
  # to copy an additional file.
  # if (provider STREQUAL "CUDA" OR provider STREQUAL "CUBLAS")
  #   file(APPEND "../_setup_ext.txt" "copy,${cuda_name},${folder}\n")
  # endif()
  file(APPEND "../_setup_ext.txt" "copy,${name},${folder}\n")
endfunction()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Ort
  VERSION_VAR ORT_VERSION
  REQUIRED_VARS ORT_URL ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIB_DIR
                ORT_LIB_FILES ORT_LIB_HEADER)
