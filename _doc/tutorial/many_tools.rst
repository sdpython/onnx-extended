
==========
Many Tools
==========

Developpers write many lines of code, many are part of a package,
many are used to investigate what the first line produces.
This section gathers some tools occasionally needed 
to write converters in :epkg:`sklearn-onnx`, to implement
kernels in :epkg:`onnxruntime`, to add new operators in :epkg:`onnx`.
The first series is used to play with :epkg:`onnx` files.
A couple of the helpers described below are available
through command lines.

.. toctree::
    :maxdepth: 2
    :caption: onnx

    external_data
    onnx_manipulations
    quantize

The second series is used to investigate C++ implementations
in :epkg:`onnxruntime`.

.. toctree::
    :maxdepth: 2
    :caption: onnxruntime

    profiling
    ort_debug
    old_version
    trees
