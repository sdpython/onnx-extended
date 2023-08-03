

===============
ONNX Benchmarks
===============

Shows the list of benchmarks implemented the :ref:`l-example-gallery`.

CPU
===

plot_optim_tree_ensemble
++++++++++++++++++++++++

See :ref:`l-plot-optim-tree-ensemble`.

This packages implements a custom kernel for
:epkg:`TreeEnsembleRegressor` and :epkg:`TreeEnsembleClassifier`
and let the users choose the parallelization parameters.
This scripts tries many values to select the best one
for trees trains with :epkg:`scikit-learn` and a 
:class:`sklearn.ensemble.RandomForestRegressor`.

CUDA
====

These tests only works if they are run a computer
with CUDA enabled.

plot_bench_gemm_f8
++++++++++++++++++

See :ref:`l-example-gemm-f8`.

The script checks the speed of :epkg:`cublasLtMatmul`
for various types and dimensions on square matricies. The code is implementation
in C++ and does not involve *onnxruntime*. It checks configurations implemented
in :epkg:`cuda_gemm.cu`.

.. autofunction:: onnx_extended.validation.cuda.cuda_example_py.gemm_benchmark_test

plot_bench_gemm_ort
+++++++++++++++++++

See :ref:`l-example-gemm-ort-f8`.

The script checks the speed of :epkg:`cublasLtMatmul` with a
custom operator for :epkg:`onnxruntime` and implemented in
:epkg:`custom_gemm.cu`.

plot_profile_gemm_ort
+++++++++++++++++++++

See :ref:`l-example-plot-profile-gemm`.

The benchmark profiles the execution of Gemm for different
types and configuration. That includes a custom operator
only available on CUDA calling function :epkg:`cublasLtMatmul`.

No specific provider
====================

plot_bench_cypy_ort
+++++++++++++++++++

See :ref:`l-cython-pybind11-ort-bindings`.

The python package for :epkg:`onnxruntime` is implemented with
:epkg:`pybind11`. It is less efficient than :epkg:`cython`
which makes direct calls to the :epkg:`Python C API`.
The benchmark evaluates that cost.
