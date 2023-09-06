
#
# Packages
#

message(STATUS "-------------------")

if(USE_CUDA)
  find_package(CudaExtension)
  if(CUDAToolkit_FOUND)
    message(STATUS "CUDA_AVAILABLE=${CUDA_AVAILABLE}")
    message(STATUS "CUDA_VERSION=${CUDA_VERSION}")
    message(STATUS "CUDA_VERSION_INT=${CUDA_VERSION_INT}")
    message(STATUS "CUDA version=${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR}")
    message(STATUS "CUDA_HAS_FP16=${CUDA_HAS_FP16}")
    message(STATUS "CUDA_INCLUDE_DIRS=${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES=${CUDA_LIBRARIES}")
    message(STATUS "CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA_cudart_static_LIBRARY=${CUDA_cudart_static_LIBRARY}")
    message(STATUS "CUDA_cudadevrt_LIBRARY=${CUDA_cudadevrt_LIBRARY}")
    message(STATUS "CUDA_cupti_LIBRARY=${CUDA_cupti_LIBRARY}")
    message(STATUS "CUDA_curand_LIBRARY=${CUDA_curand_LIBRARY}")
    message(STATUS "CUDA_cusolver_LIBRARY=${CUDA_cusolver_LIBRARY}")
    message(STATUS "CUDA_cusparse_LIBRARY=${CUDA_cusparse_LIBRARY}")
    message(STATUS "CUDA_nvToolsExt_LIBRARY=${CUDA_nvToolsExt_LIBRARY}")
    message(STATUS "CUDA_OpenCL_LIBRARY=${CUDA_OpenCL_LIBRARY}")
    message(STATUS "CUDA NVTX_LINK_C=${NVTX_LINK_C}")
    message(STATUS "CUDA NVTX_LINK_CPP=${NVTX_LINK_CPP}")
    message(STATUS "CUDA CMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
    message(STATUS "CUDA CMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}")
    message(STATUS "CUDA CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA CMAKE_CUDA_COMPILER_ID=${CMAKE_CUDA_COMPILER_ID}")
    message(STATUS "CUDA CMAKE_LIBRARY_ARCHITECTURE=${CMAKE_LIBRARY_ARCHITECTURE}")
    message(STATUS "CUDA CUDA_NVCC_FLAGS=${CUDA_NVCC_FLAGS}")
    message(STATUS "CUDA CUDAARCHS=${CUDAARCHS}")
    message(STATUS "CUDA CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}")
    message(STATUS "CUDA CUDAToolkit_NVCC_EXECUTABLE=${CUDAToolkit_NVCC_EXECUTABLE}")
    message(STATUS "CUDA CUDAToolkit_BIN_DIR=${CUDAToolkit_BIN_DIR}")
    message(STATUS "CUDA CUDAToolkit_LIBRARY_DIR=${CUDAToolkit_LIBRARY_DIR}")
    message(STATUS "CUDA NVCC_VERSION=${NVCC_VERSION}")
    set(CUDA_AVAILABLE 1)
  else()
    message(STATUS "Module CudaExtension is not installed.")
    set(CUDA_AVAILABLE 0)
  endif()
else()
  message(STATUS "Module CudaExtension is disabled.")
  set(CUDA_AVAILABLE 0)
endif()

message(STATUS "-------------------")
find_package(MyPython)
if(NOT ${PYTHON_VERSION} MATCHES ${Python3_VERSION})
  string(LENGTH PYTHON_VERSION_MM PYTHON_VERSION_MM_LENGTH)
  string(SUBSTRING Python3_VERSION
         0 PYTHON_VERSION_MM_LENGTH
         Python3_VERSION_MM)
  if(${PYTHON_VERSION_MM} MATCHES ${Python3_VERSION_MM})
    message(WARNING
            "cmake selects a different python micr  o version "
            "${Python3_VERSION} than ${PYTHON_VERSION}.")
  else()
    message(FATAL_ERROR
            "cmake selects a different python minor version "
            "${Python3_VERSION_MM} than ${PYTHON_VERSION_MM}.")
  endif()
  # installation of cython, numpy
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pip install cython numpy
    OUTPUT_VARIABLE install_version_output
    ERROR_VARIABLE install_version_error
    RESULT_VARIABLE install_version_result)
  message(STATUS "install_version_output=${install_version_output}")
  message(STATUS "install_version_error=${install_version_error}")
  message(STATUS "install_version_result=${install_version_result}")
endif()
if(MyPython_FOUND)
  message(STATUS "Python3_VERSION=${Python3_VERSION}")
  message(STATUS "Python3_LIBRARY=${Python3_LIBRARY}")
  message(STATUS "Python3_LIBRARY_RELEASE=${Python3_LIBRARY_RELEASE}")
else()
  message(FATAL_ERROR "Unable to find Python through MyPython.")
endif()

message(STATUS "-------------------")
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "Found OpenMP ${OpenMP_CXX_VERSION}")
  set(OMP_INCLUDE_DIR "")
else()
  # see https://github.com/microsoft/LightGBM/blob/master/CMakeLists.txt#L148
  execute_process(COMMAND brew --prefix libomp
                  OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(MAC_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_C_FLAGS "${MAC_FLAGS} -I${HOMEBREW_LIBOMP_PREFIX}/include")
  set(OpenMP_CXX_FLAGS "${MAC_FLAGS} -I${HOMEBREW_LIBOMP_PREFIX}/include")
  set(OpenMP_C_LIB_NAMES omp)
  set(OpenMP_CXX_LIB_NAMES omp)
  set(OMP_INCLUDE_DIR ${HOMEBREW_LIBOMP_PREFIX}/include)
  set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    message(STATUS "Found(2) OpenMP ${OpenMP_CXX_VERSION}")
  else()
    message(FATAL_ERROR "OpenMP cannot be found.")
  endif()
endif()

message(STATUS "-------------------")
find_package(Cython REQUIRED)
if(Cython_FOUND)
  message(STATUS "Found Cython ${Cython_VERSION}")
  message(STATUS "NUMPY_INCLUDE_DIR: ${NUMPY_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "Module cython is not installed.")
endif()

message(STATUS "-------------------")
find_package(LocalPyBind11 REQUIRED)
if(LocalPyBind11_FOUND)
  message(STATUS "Found LocalPyBind11, pybind11 at ${pybind11_SOURCE_DIR}")
  message(STATUS "Found pybind11 ${pybind11_VERSION}")
else()
  message(FATAL_ERROR "Module pybind11 is not installed.")
endif()

message(STATUS "-------------------")
find_package(Ort REQUIRED)
if(Ort_FOUND)
  message(STATUS "ORT_VERSION=${ORT_VERSION}")
  message(STATUS "ORT_VERSION_INT=${ORT_VERSION_INT}")
  message(STATUS "ORT_URL=${ORT_URL}")
  message(STATUS "ONNXRUNTIME_INCLUDE_DIR=${ONNXRUNTIME_INCLUDE_DIR}")
  message(STATUS "ONNXRUNTIME_LIB_DIR=${ONNXRUNTIME_LIB_DIR}")
  message(STATUS "ORT_LIB_FILES=${ORT_LIB_FILES}")
  message(STATUS "ORT_LIB_HEADER=${ORT_LIB_HEADER}")
else()
  message(FATAL_ERROR "onnxruntime is not installed.")
endif()

message(STATUS "-------------------")
find_package(LocalEigen REQUIRED)
if(LocalEigen_FOUND)
  message(STATUS "Found Eigen ${LocalEigen_VERSION}")
  message(STATUS "LOCAL_EIGEN_URL=${LOCAL_EIGEN_URL}")
  message(STATUS "LOCAL_EIGEN_SOURCE=${LOCAL_EIGEN_SOURCE}")
  message(STATUS "EIGEN_INCLUDE=${EIGEN_INCLUDE}")
  message(STATUS "EIGEN_INCLUDE_DIRS=${EIGEN_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "Module eigen is not installed.")
endif()

message(STATUS "-------------------")

if(CUDA_AVAILABLE)
  set(
    config_content
    "HAS_CUDA = 1\nCUDA_VERSION = '${CUDA_VERSION}'"
    "\nCUDA_VERSION_INT = ${CUDA_VERSION_INT}"
    "\nCXX_FLAGS = '${CMAKE_CXX_FLAGS}'\n")
else()
  set(config_content "HAS_CUDA = 0\nCXX_FLAGS='${CMAKE_CXX_FLAGS}\n")
endif()
