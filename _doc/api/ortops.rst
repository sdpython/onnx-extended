
====================
onnx_extended.ortops
====================

It supports any onnxruntime C API greater than version:

.. runpython::
    :showcode:

    from onnx_extended.ortcy.wrap.ortinf import get_ort_c_api_supported_version
    
    print(get_ort_c_api_supported_version())

.. toctree::
    :maxdepth: 2

    ortops_tutorial_cpu
    ortops_tutorial_cuda
    ortops_optim
