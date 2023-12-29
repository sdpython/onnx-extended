
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
        "cuda_device_count",
        "cuda_device_memory",
        "cuda_devices_memory",
        "cuda_version",
        "gemm_benchmark_test",
        "FpemuMode",
        "fpemu_cuda_forward",
    ]
    names.sort()
    classes = {"FpemuMode"}
    noindex = {"gemm_benchmark_test"}

    prefix = "onnx_extended.validation.cuda.cuda_example_py."
    if has_cuda():
        fct_template = f".. autofunction:: {prefix}%s"
        fct_template_no = f".. autofunction:: {prefix}%s\n    :noindex:"
        cls_template = f".. autoclass:: {prefix}%s\n    :members:"
    else:
        fct_template = f"Unable to document function `{prefix}%s`"
        fct_template_no = fct_template
        cls_template = f"Unable to document class `{prefix}%s`"

    for name in names:
        tpl = cls_template if name in classes else (
            fct_template_no if name in noindex else fct_template
        )
        print(tpl % name)
        print()

cuda_monitor
++++++++++++

.. runpython::
    :rst:

    from onnx_extended import has_cuda

    if not has_cuda():
        print(
            "The documentation was not compiled with CUDA enabled "
            "and cannot expose the CUDA functions."
        )

    names = [
        "cuda_version",
        "nvml_device_get_count",
        "nvml_device_get_memory_info",
        "nvml_init",
        "nvml_shutdown",
    ]
    names.sort()
    noindex = {}
    classes = {}

    prefix = "onnx_extended.validation.cuda.cuda_monitor."
    if has_cuda():
        fct_template = f".. autofunction:: {prefix}%s"
        fct_template_no = f".. autofunction:: {prefix}%s\n    :noindex:"
        cls_template = f".. autoclass:: {prefix}%s\n    :members:"
    else:
        fct_template = f"Unable to document function `{prefix}%s`"
        fct_template_no = fct_template
        cls_template = f"Unable to document class `{prefix}%s`"

    for name in names:
        tpl = cls_template if name in classes else (
            fct_template_no if name in noindex else fct_template
        )
        print(tpl % name)
        print()
