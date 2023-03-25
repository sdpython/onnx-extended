
onnx-extended: more operator for onnx
=====================================

.. image:: https://dev.azure.com/xavierdupre3/onnx-extended/_apis/build/status/sdpython.onnx-extended
    :target: https://dev.azure.com/xavierdupre3/onnx-extended/

.. image:: https://badge.fury.io/py/onnx-extended.svg
    :target: http://badge.fury.io/py/onnx-extended

.. image:: http://img.shields.io/github/issues/sdpython/onnx-extended.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/onnx-extended/issues

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: http://opensource.org/licenses/MIT

.. image:: https://img.shields.io/github/repo-size/sdpython/onnx-extended
    :target: https://github.com/sdpython/onnx-extended/
    :alt: size

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

**onnx-extended** extends the list of supported operators in onnx
reference implementation, or implements faster versions in C++.

.. toctree::
    :maxdepth: 2

    tutorial/index
    api/index
    auto_examples/index

Sources available on
`github/onnx-extended <https://github.com/sdpython/onnx-extended>`_,
see also `code coverage <cov/index.html>`_.

.. runpython::
    :showcode:

    import numpy as np
    from onnx import TensorProto
    from onnx.helper import (
        make_graph,
        make_model,
        make_node,
        make_opsetid,
        make_tensor_value_info,
    )
    from onnx.reference import ReferenceEvaluator
    from onnxruntime import InferenceSession
    from onnx_extended.ext_test_case import measure_time
    from onnx_extended.reference import CReferenceEvaluator


    X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None, None])
    Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None, None])
    B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None, None, None])
    W = make_tensor_value_info("W", TensorProto.FLOAT, [None, None, None, None])
    node = make_node(
        "Conv",
        ["X", "W", "B"],
        ["Y"],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        strides=[2, 2],
    )
    graph = make_graph([node], "g", [X, W, B], [Y])
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])

    sH, sW = 64, 64
    X = np.arange(sW * sH).reshape((1, 1, sH, sW)).astype(np.float32)
    W = np.ones((1, 1, 3, 3), dtype=np.float32)
    B = np.array([[[[0]]]], dtype=np.float32)

    sess1 = ReferenceEvaluator(onnx_model)
    sess2 = CReferenceEvaluator(onnx_model)  # 100 times faster

    expected = sess1.run(None, {"X": X, "W": W, "B": B})[0]
    got = sess2.run(None, {"X": X, "W": W, "B": B})[0]
    diff = np.abs(expected - got).max()
    print(f"difference: {diff}")
