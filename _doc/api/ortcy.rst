
===================
onnx_extended.ortcy
===================

It supports any onnxruntime C API greater than version:

.. runpython::
    :showcode:

    from onnx_extended.ortcy.wrap.ortinf import get_ort_c_api_supported_version
    
    print(get_ort_c_api_supported_version())

get_ort_c_api_supported_version
+++++++++++++++++++++++++++++++

.. autofunction:: onnx_extended.ortcy.wrap.ortinf.get_ort_c_api_supported_version

ort_get_available_providers
===========================

.. autofunction:: onnx_extended.ortcy.wrap.ortinf.ort_get_available_providers

OrtSession
==========

.. autoclass:: onnx_extended.ortcy.wrap.ortinf.OrtSession
    :members:
