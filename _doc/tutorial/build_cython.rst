Build with cython
=================

Any :epkg:`cython` extension is built by cmake.
It first calls cython to convert a pyx file into a C++ file
before it is compiled and linked. Using cmake + cython
instead of cython only make it easier to link with static
libraries and write unit tests in C++.

cmake
+++++

The first step is to load the extension `FindCython.cmake
<https://github.com/sdpython/onnx-extended/blob/main/_cmake/externals/FindCython.cmake>`_
with `find_package(Cython REQUIRED)`. This file exposes function
`cython_add_module(name pyx_file omp_lib)` called for
every extension to build and used as follows:

::

    cython_add_module(
        vector_function_cy                                          # name
        ../onnx_extended/validation/cython/vector_function_cy.pyx   # pyx_file
        OpenMP::OpenMP_CXX                                          # link with this target
        ../onnx_extended/validation/cpu/vector_function.cpp)        # sources files

The function accepts many source files. Other link dependencies can be added as well
by adding an instructions like `target_link_libraries(name PRIVATE lib_name)`.
This function *cythonize* the *pyx_file* into a cpp file before building
the dynamic library.

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
                "onnx_extended.validation.cython.vector_function_cy",
                f"onnx_extended/validation/cython/vector_function_cy.{ext}",
            ),
        ]
    )
