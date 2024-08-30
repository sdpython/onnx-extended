Build with onnxruntime
======================

This package includes a wrapper for :epkg:`onnxruntime` based on
:epkg:`cython`. The standard one relies on :epkg:`pybind11`.
For that purpose, it includes the onnxruntime binaries released
on github (see :epkg:`onnxruntime releases`).

build onnxruntime
+++++++++++++++++

::

    clear&&CUDA_VERSION=11.8 CUDACXX=/usr/local/cuda-11.8/bin/nvcc python ./tools/ci_build/build.py \
            --config Release --build_wheel --build_dir ./build/linux_cuda \
            --build_shared_lib --use_cuda --cuda_home /usr/local/cuda-11.8/ \
            --cudnn_home /usr/local/cuda-11.8/ --cuda_version=11.8 --enable_training --enable_training_ops \
            --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=61" \
            --parallel --skip_tests

    clear&&CUDA_VERSION=12.1 CUDACXX=/usr/local/cuda-12.1/bin/nvcc python ./tools/ci_build/build.py \
            --config Release --build_wheel --build_dir ./build/linux_cuda \
            --build_shared_lib --use_cuda --cuda_home /usr/local/cuda-12.1/ \
            --cudnn_home /usr/local/cuda-12.1/ --cuda_version=12.1 --enable_training --enable_training_ops \
            --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=70;72" \
            --parallel --skip_tests

cmake
+++++

The first step is to load the extension `FindOrt.cmake
<https://github.com/sdpython/onnx-extended/blob/main/_cmake/externals/FindOrt.cmake>`_
with `find_package(Ort REQUIRED)`. This file exposes two functions.
The first one `ort_add_dependency(name folder_copy)` copies the binaries
into folder *folder_copy* and links target *name* with onnxruntime.

The second function `ort_add_custom_op(name folder "CPU")` creates a library with 
several custom kernels for onnxruntime and links it with onnxruntime.
*name* is the project name, *folder* its location.

::

    ort_add_custom_op(
        ortops_tutorial_cpu                                             # name
        "CPU"
        ../onnx_extended/ortops/tutorial/cpu                            # folder
        ../onnx_extended/ortops/tutorial/cpu/my_kernel.cc               # source file
        ../onnx_extended/ortops/tutorial/cpu/my_kernel_attr.cc          # source file
        ../onnx_extended/ortops/tutorial/cpu/ort_tutorial_cpu_lib.cc)   # source file

Every new kernel can be added by adding new source file. A line must be added
in file `ort_tutorial_cpu_lib.cc` to register the kernel. That file also defines
the domain the kernel belongs to.
These project define constant `ORT_VERSION`. For example, version 1.15 becomes
`1150`.
