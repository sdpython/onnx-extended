#
# initialization
#
# downloads onnxruntime as a binary
#

#
# 
#

if(NOT ORT_VERSION)
  set(ORT_VERSION 1.14.1)
endif()
set(Ort_VERSION "${ORT_VERSION}")

if(CUDA_FOUND)
  if(APPLE)
    message(WARNING "onnxruntime-gpu not available on MacOsx")
  endif()
  set(ORT_GPU "-gpu")
else()
  set(ORT_GPU "")
endif()
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

if(EXISTS "${ORT_DEST}")
  message(STATUS "onnxruntime - already downloaded '${ORT_DEST}'")
else()
  message(STATUS "onnxruntime - download '${ORT_URL}' into '${ORT_DEST}'")
  file(DOWNLOAD "${ORT_URL}" "${ORT_DEST}" SHOW_PROGRESS)
endif()
file(SIZE "${ORT_DEST}" ORT_DEST_SIZE)
if(ORT_DEST_SIZE LESS 1024)
  message(FATAL_ERROR "Empty file, unable to download '${ORT_URL}'.")
endif()
if(EXISTS "${ORT_DEST}")
  if(EXISTS "${ORT_DEST_DIR}/${ORT_FOLD}/VERSION_NUMBER")
  message(STATUS "onnxruntime - extracted '${ORT_DEST}'")
  else()
    message(STATUS "onnxruntime - extract '${ORT_DEST}'")
    file(ARCHIVE_EXTRACT INPUT "${ORT_DEST}" DESTINATION "${ORT_DEST_DIR}" TOUCH VERBOSE)
  endif()
else()
  message(FATAL_ERROR "Unable to download '${ORT_URL}'.")
endif()
if(EXISTS "${ORT_DEST_DIR}/${ORT_FOLD}/VERSION_NUMBER")
  message(STATUS "onnxruntime - found ${ORT_DEST_DIR}/${ORT_FOLD}/VERSION_NUMBER'.")
else()
  message(FATAL_ERROR "Unable to extract '${ORT_DEST}'.")
endif()
set(ORT_DIR "${ORT_DEST_DIR}${ORT_FOLD}")


add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ORT_DIR}/libonnxruntime.so")


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Ort
  VERSION_VAR Ort_VERSION
  REQUIRED_VARS ORT_URL ORT_DEST ORT_DEST_DIR ORT_DIR)
