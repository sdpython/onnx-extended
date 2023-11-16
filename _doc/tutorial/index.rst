
Tutorial
========

This package is mostly used to validate custom implementations
of a specific onnx operator or **kernel**.
The same code is used to either implement a custom kernel for the
reference implementation from :epkg:`onnx` package or a custom kernel
for :epkg:`onnxruntime`. The last section
describe how to build the package and to add a new implementation
depending the technology it relies on (CPU, openmp, CUDA, eigen, ...).
The last section is a sorted index of the examples.

.. toctree::
    :maxdepth: 1
    :caption: Kernels

    reference_evaluator
    cython_binding
    custom_ops
    ops
    many_tools
    build

.. toctree::
    :maxdepth: 1
    :caption: Deprecated

    parallelization
