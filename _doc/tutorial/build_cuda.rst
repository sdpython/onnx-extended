Build with CUDA
===============

The build may include pybind11 extension building with CUDA.
The setup is more complex as CUDA is not always available.
The profiler may be enabled as well.

cmake
+++++

The first step is to load the extension `FindCudaExtension.cmake
<https://github.com/sdpython/onnx-extended/blob/main/_cmake/externals/FindCudaExtension.cmake>`_
with `find_package(CudaExtension)`. This file exposes function
`cuda_pybind11_add_module(name pybindfile)` called for
every extension to build and used as follows:

::

    if(CUDA_AVAILABLE)

        cuda_pybind11_add_module(
            cuda_example_py                                             # name
            ../onnx_extended/validation/cuda/cuda_example_py.cpp        # pybind11 file
            ../onnx_extended/validation/cuda/cuda_example.cu            # CUDA code
            ../onnx_extended/validation/cuda/cuda_example_reduce.cu)    # CUDA code

    endif()

The function accepts many source files whether they have extension c, cpp, cc, cu.
Other link dependencies can be added as well
by adding an instructions like `target_link_libraries(name PRIVATE lib_name)`.
These project define constant `CUDA_VERSION`. For example, version 11.8 becomes
`11080`.

setup.py
++++++++

`setup.py <https://github.com/sdpython/onnx-extended/blob/main/setup.py>`_
defines a custom command to call cmake. Another line must be added
to register the extension in the setup.

::

    if platform.system() == "Windows":
        ext = "pyd"
    elif platform.system() == "Darwin"
        ext = "dylib"
    else:
        ext = "so"

    if find_cuda():

        setup(
            ...
            ext_modules = [
                ...
                CMakeExtension(
                    "onnx_extended.validation.cuda.cuda_example_py",
                    f"onnx_extended/validation/cuda/cuda_example_py.{ext}",
                ),
            ]
        )

Function `find_cuda()` executes :epkg:`nvidia-smi` to check
the installation of CUDA.

Possible errors
+++++++++++++++

CMAKE_CUDA_COMPILER_VERSION=11.5.119 < 12.1, nvcc is not setup properly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On Linux, the following error may happen:

::

    CMake Error at externals/FindCudaExtension.cmake:60 (message):
    CMAKE_CUDA_COMPILER_VERSION=11.5.119 < 12.1, nvcc is not setup properly.
    Try 'whereis nvcc' and chack the version.
    Call Stack (most recent call first):
    load_externals.cmake:9 (find_package)
    CMakeLists.txt:19 (include)

It can be fixed by adding `--cuda-nvcc=<path ot nvcc>`. An example:
`--cuda-nvcc=/usr/local/cuda-12.1/bin/nvcc`.