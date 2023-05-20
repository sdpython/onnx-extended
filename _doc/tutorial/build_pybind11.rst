Build with pybind11
===================

Any :epkg:`pybind11` extension is built by cmake.
Using cmake + pybind11 instead of pybind11
only make it easier to link with static
libraries and write unit tests in C++.

cmake
+++++

The first step is to load the extension `FindLocalPyBind11
<https://github.com/sdpython/onnx-extended/blob/main/_cmake/externals/FindLocalPyBind11.cmake>`_
with ``find_package(LocalPyBind11 REQUIRED)``.
This extension fetches the content of pybind11 and builds it with
`FetchContent_Populate(pybind11)`. The version is registered there.
It must be done once.
It defines a function `local_pybind11_add_module(name omp_lib)` called for
every extension to build and used as follows:

::

    local_pybind11_add_module(
    _validation                                         # name
    OpenMP::OpenMP_CXX                                  # link with this library
    ../onnx_extended/validation/cpu/_validation.cpp     # source file
    ../onnx_extended/validation/cpu/vector_sum.cpp)     # source file

Additional libraries can be added with `target_link_libraries(name PRIVATE lib_name)`.

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

    setup(
        ...
        ext_modules = [
            ...
            CMakeExtension(
                "onnx_extended.validation.cpu._validation",
                f"onnx_extended/validation/cpu/_validation.{ext}",
            ),
        ]
    )
