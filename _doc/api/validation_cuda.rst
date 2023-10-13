
===============
validation.cuda
===============

C API
=====

cuda_example_py
+++++++++++++++

.. ifconfig:: HAS_CUDA in ("1", )

    .. autoclass:: onnx_extended.validation.cuda.cuda_example_py.FpemuMode
        :members:

    .. autofunction:: onnx_extended.validation.cuda.cuda_example_py.fpemu_cuda_forward

    .. autofunction:: onnx_extended.validation.cuda.cuda_example_py.gemm_benchmark_test

    .. autofunction:: onnx_extended.validation.cuda.cuda_example_py.vector_add

    .. autofunction:: onnx_extended.validation.cuda.cuda_example_py.vector_sum0

    .. autofunction:: onnx_extended.validation.cuda.cuda_example_py.vector_sum6

.. ifconfig:: HAS_CUDA in ("0", )

    The documentation was not compiled with CUDA enabled and cannot
    expose the CUDA functions.
