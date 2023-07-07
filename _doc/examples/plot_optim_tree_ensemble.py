"""
TreeEnsemble optimization
=========================

The execution of a TreeEnsembleRegressor can lead to very different results
depending on how the computation is parallelized. By trees,
by rows, by both, for only one row, for a short batch of rows, a longer one.
The implementation in :epkg:`onnxruntime` does not let the user changed
the predetermined settings but a custom kernel might. That's what this example
is measuring.
"""
