#
# initialization
#
# Defines USE_NTVX to enable profiling with NVIDIA profiler.
# CUDA_VERSION must be defined as well.

if(CUDA_VERSION)
  find_package(CUDAToolkit ${CUDA_VERSION} EXACT)
else()
  find_package(CUDAToolkit)
endif()

message(STATUS "CUDAToolkit_FOUND=${CUDAToolkit_FOUND}")

if(CUDAToolkit_FOUND)

  if(USE_NVTX)
    # see https://github.com/NVIDIA/NVTX
    include(CPM.cmake)

    CPMAddPackage(
        NAME NVTX
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.1.0-c-cpp
        GIT_SHALLOW TRUE)

    message(STATUS "CUDA NTVX_FOUND=${NTVX_FOUND}")
    set(NVTX_LINK_C "nvtx3-c")
    set(NVTX_LINK_CPP "nvtx3-cpp")
    add_compile_definitions("ENABLE_NVTX")
  else()
    set(NVTX_LINK_C "")
    set(NVTX_LINK_CPP "")
    message(STATUS "CUDA NTVX not added.")
  endif()

  execute_process(
    COMMAND nvcc --version
    OUTPUT_VARIABLE NVCC_version_output
    ERROR_VARIABLE NVCC_version_error
    RESULT_VARIABLE NVCC_version_result
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  # If the version is not the same, something like the following can be tried:
  # export PATH=/usr/local/cuda-11-8/bin:$PATH
  if(NOT NVCC_version_output MATCHES ".*${CUDA_VERSION}.*")
    message(FATAL_ERROR "CUDA_VERSION=${CUDA_VERSION} does not match nvcc "
                        "version=${NVCC_version_output}, try\n"
                        "export PATH=/usr/local/cuda-"
                        "${CUDAToolkit_VERSION_MAJOR}."
                        "${CUDAToolkit_VERSION_MINOR}/bin:$PATH")
  endif()
  set(NVCC_VERSION "${NVCC_version_output}")
  math(
    EXPR
    CUDA_VERSION_INT
    "${CUDAToolkit_VERSION_MAJOR} * 1000 + ${CUDAToolkit_VERSION_MINOR} * 10"
    OUTPUT_FORMAT DECIMAL)

  set(CUDA_AVAILABLE 1)
  set(CUDA_VERSION ${CUDAToolkit_VERSION})
  set(CUDA_LIBRARIES CUDA::cudart_static
                     CUDA::cufft_static CUDA::cufftw_static
                     CUDA::curand_static
                     CUDA::cublas_static CUDA::cublasLt_static
                     CUDA::cusolver_static
                     CUDA::cupti_static
                     CUDA::nvToolsExt)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    CudaExtension
    VERSION_VAR "0.1"
    REQUIRED_VARS CUDAToolkit_FOUND CUDA_VERSION
                  CUDA_VERSION_INT CUDA_LIBRARIES NVCC_VERSION
                  CUDA_AVAILABLE)

else()

  if(CUDA_VERSION)
    message(FATAL_ERROR "Unable to find CUDA=${CUDA_VERSION}, you can do\n"
                        "export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:$PATH\n"
                        "PATH=$ENV{PATH}")
  endif()
  set(CUDA_VERSION_INT 0)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    CudaExtension
    VERSION_VAR "0.1"
    REQUIRED_VARS CUDAToolkit_FOUND CUDA_VERSION CUDA_VERSION_INT "" "" 0)

endif()
