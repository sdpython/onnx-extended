
===============
validation.cuda
===============

C API
=====

cuda_example_py
+++++++++++++++

.. runpython::
    :rst:

    from onnx_extended import has_cuda

    if not has_cuda():
        print(
            "The documentation was not compiled with CUDA enabled "
            "and cannot expose the CUDA functions."
        )

    names = [
        "FpemuMode",
        "fpemu_cuda_forward",
        "gemm_benchmark_test",
        "vector_add",
        "vector_sum0",
        "vector_sum6",
    ]
    classes = {"FpemuMode"}
    names.sort()
    if has_cuda():
        fct_template = ".. autofunction:: onnx_extended.validation.cuda.cuda_example_py.%s"
        cls_template = (
            ".. autoclass:: onnx_extended.validation.cuda.cuda_example_py.%s\n    :members:"
        )
    else:
        fct_template = (
            "Unable to document function `onnx_extended.validation.cuda.cuda_example_py.%s`"
        )
        cls_template = (
            "Unable to document class `onnx_extended.validation.cuda.cuda_example_py.%s`"
        )

    for name in names:
        tpl = cls_template if name in classes else fct_template
        print(tpl % name)
        print()
