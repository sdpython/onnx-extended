Build with onnxruntime
======================

This package includes a wrapper for :epkg:`onnxruntime` based on
:epkg:`cython`. The standard one relies on :epkg:`pybind11`.
For that purpose, it includes the onnxruntime binaries released
on github (see :epkg:`onnxruntime releases`).

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

This function is subject to change. It creates a file `_setup_ext.txt` to indicate
which file to copy from the build directory to the package directory.
This file is loaded by `setup.py` after cmake is done with the compilation.
