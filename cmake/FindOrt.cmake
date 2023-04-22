#
# initialization
#
# downloads onnxruntime as a binary
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

FetchContent_Declare(onnxruntime URL ${ORT_URL})
FetchContent_makeAvailable(onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib)

find_library(ONNXRUNTIME onnxruntime HINTS "${ONNXRUNTIME_LIB_DIR}")
if(ONNXRUNTIME-NOTFOUND)
    message(FATAL_ERROR "onnxruntime cannot be found at ${ONNXRUNTIME_LIB_DIR}.")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Ort
  VERSION_VAR Ort_VERSION
  REQUIRED_VARS ORT_URL ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIB_DIR)
