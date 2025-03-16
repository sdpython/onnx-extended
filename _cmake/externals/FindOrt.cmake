#
# initialization
#
# downloads onnxruntime as a binary
# functions ort_add_dependency, ort_add_custom_op

if(NOT ORT_VERSION)
  set(ORT_VERSION 1.16.1)
  set(ORT_VERSION_INT 1161)
endif()
string(LENGTH "${ORT_VERSION}" ORT_VERSION_LENGTH)

if(CUDAToolkit_FOUND)
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

  string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" ORT_VERSION_MATCH ${ORT_VERSION})
  set(ORT_VERSION_MAJOR ${CMAKE_MATCH_1})
  set(ORT_VERSION_MINOR ${CMAKE_MATCH_2})
  set(ORT_VERSION_PATCH ${CMAKE_MATCH_3})
  math(
    EXPR
    ORT_VERSION_INT
    "${ORT_VERSION_MAJOR} * 1000 + ${ORT_VERSION_MINOR} * 10 + ${ORT_VERSION_PATCH}"
    OUTPUT_FORMAT DECIMAL)

  message(STATUS "ORT - ORT_URL=${ORT_URL}")
  FetchContent_Declare(onnxruntime URL ${ORT_URL})
  FetchContent_makeAvailable(onnxruntime)
  set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
  set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib)

  if ("${ORT_VERSION}" LESS "1.17.0")
    message(FATAL_ERROR "onnxruntime version must be >= 1.17.0 but it ${ORT_VERSION}.")
  endif()

else()
  message(STATUS "ORT - retrieve development version from '${ORT_VERSION}'")
  set(ORT_VERSION_INT 99999)
  set(ONNXRUNTIME_LIB_DIR "${ORT_VERSION}")
  if(MSVC)
    set(ONNXRUNTIME_INCLUDE_DIR
        "${ORT_VERSION}\\..\\..\\..\\..\\include\\onnxruntime\\core\\session")
  else()
    set(ONNXRUNTIME_INCLUDE_DIR
        "${ORT_VERSION}/../../../include/onnxruntime/core/session")
  endif()
  set(ORT_URL ${ORT_VERSION})
  message(STATUS "ORT - retrieve from '${ONNXRUNTIME_INCLUDE_DIR}'")
endif()

find_library(ONNXRUNTIME onnxruntime HINTS "${ONNXRUNTIME_LIB_DIR}")
if(ONNXRUNTIME-NOTFOUND)
  message(FATAL_ERROR "onnxruntime cannot be found at '${ONNXRUNTIME_LIB_DIR}'")
endif()

file(GLOB ORT_LIB_FILES ${ONNXRUNTIME_LIB_DIR}/*.${DLLEXT}*)
file(GLOB ORT_LIB_HEADER ${ONNXRUNTIME_INCLUDE_DIR}/*.h)

if(MSVC)
  if(NOT EXISTS "${ONNXRUNTIME_LIB_DIR}\\onnxruntime.dll")
    message(FATAL_ERROR "Unable to find '${ONNXRUNTIME_LIB_DIR}\\onnxruntime.dll'.")
  endif()
endif()

list(LENGTH ORT_LIB_FILES ORT_LIB_FILES_LENGTH)
if (ORT_LIB_FILES_LENGTH LESS_EQUAL 1)
  message(FATAL_ERROR "No file found in '${ONNXRUNTIME_LIB_DIR}' "
                      "from path or url '${ORT_URL}', "
                      "found files [${ORT_LIB_FILES}] "
                      "with extension '${DLLEXT}'.")
endif()

list(LENGTH ORT_LIB_HEADER ORT_LIB_HEADER_LENGTH)
if (ORT_LIB_HEADER_LENGTH LESS_EQUAL 1)
  message(FATAL_ERROR "No file found in '${ONNXRUNTIME_INCLUDE_DIR}' "
                      "from path or url '${ORT_URL}', "
                      "found files [${ORT_LIB_HEADER}]")
endif()

if(MSVC)
  file(GLOB ORT_LIB_FILES_PDB ${ONNXRUNTIME_LIB_DIR}/*.pdb)
  list(LENGTH ORT_LIB_FILES_PDB ORT_LIB_FILES_PDB_LENGTH)
  if (ORT_LIB_FILES_PDB_LENGTH LESS_EQUAL 1)
    message(FATAL_ERROR "No pdb file found in '${ONNXRUNTIME_LIB_DIR}' "
                        "from path or url '${ORT_URL}', "
                        "found files [${ORT_LIB_FILES}] "
                        "with extension 'pdb'.")
  endif()
  list(APPEND ORT_LIB_FILES ${ORT_LIB_FILES_PDB})

  file(GLOB ORT_LIB_FILES_LIB ${ONNXRUNTIME_LIB_DIR}/*.lib)
  list(LENGTH ORT_LIB_FILES_LIB ORT_LIB_FILES_LIB_LENGTH)
  if (ORT_LIB_FILES_LIB_LENGTH LESS_EQUAL 1)
    message(FATAL_ERROR "No pdb file found in '${ONNXRUNTIME_LIB_DIR}' "
                        "from path or url '${ORT_URL}', "
                        "found files [${ORT_LIB_FILES}] "
                        "with extension 'lib'.")
  endif()
  list(APPEND ORT_LIB_FILES ${ORT_LIB_FILES_LIB})
endif()

#
#! ort_add_dependency : copies necessary onnxruntime assembly
#                       to the location a target is build
#
# \arg:name target name
# \arg:folder_copy where to copy the assembly
#
function(ort_add_dependency name folder_copy)
  get_target_property(target_output_directory ${name} BINARY_DIR)
  if(MSVC)
    set(destination_dir ${target_output_directory}/${CMAKE_BUILD_TYPE})
  else()
    set(destination_dir ${target_output_directory})
  endif()
  message(STATUS "ort: copy ${ORT_LIB_FILES_LENGTH} files from '${ONNXRUNTIME_LIB_DIR}'")
  foreach(file_i ${ORT_LIB_FILES})
    message(STATUS "ort: copy ${file_i} to '${destination_dir}'")
    add_custom_command(
      TARGET ${name} PRE_BUILD
      COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${destination_dir})
    if(folder_copy)
      message(STATUS "ort: copy '${file_i}' to '${ROOT_PROJECT_PATH}/${folder_copy}'")
      add_custom_command(
        TARGET ${name} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy "${file_i}" "${ROOT_PROJECT_PATH}/${folder_copy}")
      message(STATUS "ort: copy '${file_i}' to '${SETUP_BUILD_LIB}/${folder_copy}'")
      add_custom_command(
        TARGET ${name} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E make_directory "${SETUP_BUILD_LIB}/${folder_copy}")
      add_custom_command(
        TARGET ${name} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy "${file_i}" "${SETUP_BUILD_LIB}/${folder_copy}")
    endif()
  endforeach()
  # file(COPY ${ORT_LIB_FILES} DESTINATION ${target_output_directory})
endfunction()

#
#! ort_add_custom_op : compile a pyx file into cpp
#
# \arg:name project name
# \arg:folder where to copy the library
# \arg:provider CUDA if a cuda lib, CPU if CPU
# \argn: C++ file to compile
#
function(ort_add_custom_op name provider folder)
  if (WIN32)
    file(WRITE "${folder}/${name}.def" "LIBRARY "
               "\"${name}.dll\"\nEXPORTS\n  RegisterCustomOps @1")
    list(APPEND ARGN "${folder}/${name}.def")
  endif()
  if (provider STREQUAL "CUDA")
    message(STATUS "ort: custom op ${provider}: '${name}' in '${folder}'")
    add_library(${name} SHARED ${ARGN})

    # add property --use_fast_math to cu files
    # set(NEW_LIST ${name}_src_files)
    # list(APPEND ${name}_cu_files ${ARGN})
    # list(FILTER ${name}_cu_files INCLUDE REGEX ".+[.]cu$")
    # set_source_files_properties(
    #  ${name}_cu_files PROPERTIES COMPILE_OPTIONS "--use_fast_math")

    target_compile_definitions(
      ${name}
      PRIVATE
      CUDA_VERSION=${CUDA_VERSION_INT}
      ORT_VERSION=${ORT_VERSION_INT}
      PYTHON_MANYLINUX=${PYTHON_MANYLINUX})
    if(USE_NVTX)
      message(STATUS "    LINK ${name} <- stdc++ nvtx3-cpp ${CUDA_LIBRARIES}")
      target_link_libraries(
        ${name}
        PRIVATE
        stdc++
        nvtx3-cpp
        ${CUDA_LIBRARIES})
    else()
      message(STATUS "    LINK ${name} <- stdc++ ${CUDA_LIBRARIES}")
      target_link_libraries(
        ${name}
        PRIVATE
        stdc++
        ${CUDA_LIBRARIES})
    endif()
    target_include_directories(
      ${name}
      PRIVATE
      ${ONNXRUNTIME_INCLUDE_DIR})
  else()
    message(STATUS "ort: custom op CPU: '${name}' in '${folder}'")
    add_library(${name} SHARED ${ARGN})
    target_include_directories(${name} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_compile_definitions(
      ${name}
      PRIVATE
      ORT_VERSION=${ORT_VERSION_INT}
      PYTHON_MANYLINUX=${PYTHON_MANYLINUX})
  endif()
  set_property(TARGET ${name} PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(${name} PROPERTIES CXX_VISIBILITY_PRESET hidden)
  set_target_properties(${name} PROPERTIES VISIBILITY_INLINES_HIDDEN 1)  
  get_target_property(target_file ${name} LIBRARY_OUTPUT_NAME)
  message(STATUS "ort: copy after build '$<TARGET_FILE:${name}>' to '${ROOT_PROJECT_PATH}/${folder}'")
  add_custom_command(
    TARGET ${name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E copy "$<TARGET_FILE:${name}>" "${ROOT_PROJECT_PATH}/${folder}")
  message(STATUS "ort: copy after build '$<TARGET_FILE:${name}>' to '${SETUP_BUILD_LIB}//${folder}'")
  add_custom_command(
    TARGET ${name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E make_directory  "${SETUP_BUILD_LIB}/${folder}")
  add_custom_command(
    TARGET ${name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E copy "$<TARGET_FILE:${name}>" "${SETUP_BUILD_LIB}/${folder}")
endfunction()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Ort
  VERSION_VAR ORT_VERSION
  REQUIRED_VARS ORT_URL ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIB_DIR
                ORT_LIB_FILES ORT_LIB_HEADER ORT_VERSION_INT)
