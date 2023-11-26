
======================
validation.bench_trees
======================

Design
======

The sparse format defined here is structure storing indices and values (float)
in a single float array. The beginning of the structures
stores the shape (1D to 5D), the element type and the number of stored
elements. The two following functions are used to convert
from dense from/to sparse.

Functions
=========

onnx_extended.validation.cpu._validation.sparse_struct_to_dense
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. autofunction:: onnx_extended.validation.cpu._validation.sparse_struct_to_dense

onnx_extended.validation.cpu._validation.dense_to_sparse_struct
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. autofunction:: onnx_extended.validation.cpu._validation.dense_to_sparse_struct
