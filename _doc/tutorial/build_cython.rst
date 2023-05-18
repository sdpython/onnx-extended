
Build with cython
=================

Any :epkg:`cython` extension is built by cmake.
It first calls cython to convert a pyx file into a C++ file
before it is compiled and linked.

cmake
+++++

The first step is to load the extension `FindCython.cmake
<https://github.com/sdpython/onnx-extended/blob/main/_cmake/externals/FindCython.cmake>`_
with ``find_package(Cython REQUIRED)``.
