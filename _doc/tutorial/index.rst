
Tutorial
========

Introduction
++++++++++++

.. toctree::
    :maxdepth: 1

    usefulcmd
    install_cuda_wsl

Operators
+++++++++

.. toctree::
    :maxdepth: 1

    ../auto_examples/plot_conv
    ../auto_examples/plot_conv_denorm

Build
+++++

The packages relies on :epkg:`cmake` to build the C++ extensions.
whether it wrapped with :epkg:`pybind11` or :epkg:`cython`.
Both options are available and can be linked with :epkg:`openmp`,
:epkg:`eigen`, :epkg:`onnxruntime`, :epkg:`CUDA`.
*cmake* is called from `setup.py
<https://github.com/sdpython/onnx-extended/blob/main/setup.py#L198>`_
with two instructions:

* ``python setup.py build_ext --inplace``, the legacy way
* ``pip install -e .``, the new way

By default, *cmake* builds with CUDA if it is available. It can be disabled:

* ``python setup.py build_ext --inplace --with-cuda=0``, the legacy way
* ``pip install -e . --config-settings="--with-cuda=0"``, the new way (not fully working yet)

In case there are multiple versions of CUDA installed, option `cuda-version`
can be specified:

::

    python setup.py build_ext --inplace --cuda-version=11.8

The development versions of :epkg:`onnxruntime` can be used if it was already build
``--ort-version=<version or build path>``. Example:

::

    python setup.py build_ext --inplace --cuda-version=11.8 --ort-version=/home/github/onnxruntime/build/linux_cuda/Release

.. toctree::
    :maxdepth: 1    
    
    build_cython
    build_pybind11
    build_cuda
    build_ortext

Validation, Experiments
+++++++++++++++++++++++

.. toctree::
    :maxdepth: 1

    ../auto_examples/plot_bench_cpu
    ../auto_examples/plot_bench_cpu_vector_sum
    ../auto_examples/plot_bench_cpu_vector_sum_parallel
    ../auto_examples/plot_bench_cpu_vector_sum_avx_parallel
    ../auto_examples/plot_bench_gpu_vector_sum_gpu
    ../auto_examples/plot_bench_cypy_ort
    ../auto_examples/plot_bench_gemm_f8
    ../auto_examples/plot_bench_gemm_ort
    ../auto_examples/plot_profile_gemm_ort

Technical details in practice
+++++++++++++++++++++++++++++

.. toctree::
    :maxdepth: 1

    ../auto_examples/plot_conv_denorm
