Custom Kernels for onnxruntime
==============================

:epkg:`onnxruntime` implements a C API which allows the user
to add custom implementation for any new operator.
This mechanism is described on onnxruntime documentation
`Custom operators <https://onnxruntime.ai/docs/reference/operators/add-custom-op.html>`_.
This packages implements a couple of custom operators for CPU and
GPU (NVIDIA). The first steps is to register an assembly to let
onnxruntime use them.

.. code-block:: python

    from onnxruntime import InferenceSession, SessionOptions
    from onnx_extended.ortops.optim.cpu import get_ort_ext_libs

    opts = SessionOptions()
    opts.register_custom_ops_library(get_ort_ext_libs()[0])

    sess = InferenceSession(
        "<model_name_or_bytes>", opts, providers=[..., "CPUExecutionProvider"]
    )

Next section introduces the list of operators and assemblies this package
implements.

onnx_extended.ortops.tutorial.cpu
+++++++++++++++++++++++++++++++++

.. runpython::
    :showcode:

    from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

    print(get_ort_ext_libs)

.. runpython::
    :rst:

    from onnx_extended.ortops.tutorial.cpu import documentation
    print(documentation())

onnx_extended.ortops.tutorial.cuda
++++++++++++++++++++++++++++++++++

.. runpython::
    :showcode:

    from onnx_extended.ortops.tutorial.cuda import get_ort_ext_libs

    print(get_ort_ext_libs)

.. runpython::
    :rst:

    from onnx_extended.ortops.tutorial.cuda import documentation
    print(documentation())

onnx_extended.ortops.optim.cpu
++++++++++++++++++++++++++++++

.. runpython::
    :showcode:

    from onnx_extended.ortops.optim.cpu import get_ort_ext_libs

    print(get_ort_ext_libs)

.. runpython::
    :rst:

    from onnx_extended.ortops.optim.cpu import documentation
    print(documentation())

