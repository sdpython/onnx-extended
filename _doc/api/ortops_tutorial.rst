
===============
ortops.tutorial
===============

CPU: onnx_extented.ortops.tutorial.cpu
======================================

.. autofunction:: onnx_extended.ortops.tutorial.cpu.get_ort_ext_libs

**List of implemented kernels**

.. runpython::
    :showcode:
    :rst:

    from onnx_extended.ortops.tutorial.cpu import documentation
    print("\n".join(documentation()))

CUDA: onnx_extented.ortops.tutorial.cuda
========================================

.. autofunction:: onnx_extended.ortops.tutorial.cuda.get_ort_ext_libs

**List of implemented kernels**

.. runpython::
    :showcode:
    :rst:

    from onnx_extended.ortops.tutorial.cuda import documentation
    print("\n".join(documentation()))
