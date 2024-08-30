

===============
ONNX Benchmarks
===============

Shows the list of benchmarks implemented the :ref:`l-example-gallery`.

CPU
===

plot_bench_tfidf
++++++++++++++++

See :ref:`l-plot-optim-tfidf`.

This benchmark measures the computation time when the kernel outputs
sparse tensors.


plot_op_tree_ensemble_optim
+++++++++++++++++++++++++++

See :ref:`l-plot-optim-tree-ensemble`.

This packages implements a custom kernel for
:epkg:`TreeEnsembleRegressor` and :epkg:`TreeEnsembleClassifier`
and let the users choose the parallelization parameters.
This scripts tries many values to select the best one
for trees trains with :epkg:`scikit-learn` and a 
:class:`sklearn.ensemble.RandomForestRegressor`.

plot_op_tree_ensemble_sparse
++++++++++++++++++++++++++++

See :ref:`l-plot-optim-tree-ensemble-sparse`.

This packages implements a custom kernel for
:epkg:`TreeEnsembleRegressor` and :epkg:`TreeEnsembleClassifier`
and let the users choose the parallelization parameters.
This scripts tries many values to select the best one
for trees trains with :epkg:`scikit-learn` and a 
:class:`sklearn.ensemble.RandomForestRegressor`.

plot_op_tree_ensemble_implementations
+++++++++++++++++++++++++++++++++++++

Test several implementations of TreeEnsemble is more simple way,
see :ref:`l-plot_op_tree_ensemble_implementations`.

plot_op_einsum
++++++++++++++

See :ref:`l-plot-op-einsum`.

Function einsum can be decomposed into a matrix multiplication and
other transpose operators. What is the best decomposition?

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
in :epkg:`cuda_gemm.cu`. See function `gemm_benchmark_test` in
`onnx_extended.validation.cuda.cuda_example_py`.

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

plot_op_gemm2_cuda
++++++++++++++++++

See :ref:`l-example-op-gemm2_cuda`.

One big Gemm or two smaller gemm.

plot_op_mul_cuda
++++++++++++++++

See :ref:`l-example-op-mul_cuda`.

The benchmark compares two operators Mul profiles
with their fusion into a single operator.

plot_op_scatternd_cuda
++++++++++++++++++++++

See :ref:`l-example-op-scatternd_cuda`.

The benchmark compares two operators ScatterND, using
atomic, no atomic.

plot_op_scatternd_mask_cuda
+++++++++++++++++++++++++++

See :ref:`l-example-op-scatternd_mask_cuda`.

The benchmark compares three operators ScatterND to update
a matrix.

plot_op_transpose2dcast_cuda
++++++++++++++++++++++++++++

See :ref:`l-example-op-transpose2dcast_cuda`.

The benchmark looks into the fusion to Transpose + Cast.

No specific provider
====================

plot_bench_cypy_ort
+++++++++++++++++++

See :ref:`l-cython-pybind11-ort-bindings`.

The python package for :epkg:`onnxruntime` is implemented with
:epkg:`pybind11`. It is less efficient than :epkg:`cython`
which makes direct calls to the :epkg:`Python C API`.
The benchmark evaluates that cost.
