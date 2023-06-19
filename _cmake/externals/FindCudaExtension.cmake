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

  enable_language(CUDA)
  message(STATUS "------------- CUDA settings")
  message(STATUS "CUDA_VERSION=${CUDA_VERSION}")
  message(STATUS "CUDA_BUILD=${CUDA_BUILD}")
  message(STATUS "CUDAARCHS=${CUDAARCHS}")
  message(STATUS "CMAKE_CUDA_COMPILER_VERSION=${CMAKE_CUDA_COMPILER_VERSION}")
  message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "CMAKE_LIBRARY_ARCHITECTURE=${CMAKE_LIBRARY_ARCHITECTURE}")
  message(STATUS "CMAKE_CUDA_COMPILER_ID=${CMAKE_CUDA_COMPILER_ID}")
  message(STATUS "CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}")
  message(STATUS "------------- end of CUDA settings")
  if (NOT CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL CUDA_VERSION)
    message(FATAL_ERROR "CMAKE_CUDA_COMPILER_VERSION=${CMAKE_CUDA_COMPILER_VERSION} "
                        "< ${CUDA_VERSION}, nvcc is not setup properly. "
                        "Try 'whereis nvcc' and chack the version.")
  endif()
  
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # CUDA flags
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

  if(CUDA_BUILD STREQUAL "H100opt")

    # see https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    set(CMAKE_CUDA_ARCHITECTURES 90)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_90,code=sm_90")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_90a,code=sm_90a")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_90a,code=compute_90a")

  else()  # H100, DEFAULT

    if(CUDA_BUILD STREQUAL "H100")
      set(CMAKE_CUDA_ARCHITECTURES 52 70 80 90)
    elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      if(NOT CUDA_BUILD STREQUAL "DEFAULT")
        message(FATAL_ERROR "Unexpected value for CUDA_BUILD='${CUDA_BUILD}'.")
      endif()
      set(CMAKE_CUDA_ARCHITECTURES 52 70 80 90)
    else()
      if(NOT CUDA_BUILD STREQUAL "DEFAULT")
        message(FATAL_ERROR "Unexpected value for CUDA_BUILD='${CUDA_BUILD}'.")
      endif()
    endif()

    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11)
      message(FATAL_ERROR "CUDA verions must be >= 11 but is ${CMAKE_CUDA_COMPILER_VERSION}.")
    endif()
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12)
      # 37, 50 still work in CUDA 11 but are marked deprecated and will be removed in future CUDA version.
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_37,code=sm_37") # K80
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50") # M series
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_52,code=sm_52") # M60
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60") # P series
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_61,code=sm_61") # P series
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70") # V series
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75") # T series
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80") # A series
      # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_86,code=sm_86") # A series
      # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_87,code=sm_87") # A series
    endif()
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_90,code=sm_90") # H series
    endif()
  endif()

  if (NOT WIN32)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fPIC")
  endif()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 4")

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
