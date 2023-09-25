Cython Binding of onnxruntime
=============================

:epkg:`onnxruntime` implements a python API based on :epkg:`pybind11`.
This API is custom and does not leverage the C API.
This package implements class
:class:`OrtSession <onnx_extended.ortcy.wrap.ortinf.OrtSession>`.
The bindings is based on :epkg:`cython` which faster.
The difference is significant when onnxruntime deals with small tensors.

.. runpython::
    :showcode:

    import numpy
    from onnx import TensorProto
    from onnx.helper import (
        make_model,
        make_node,
        make_graph,
        make_tensor_value_info,
        make_opsetid,
    )
    from onnx_extended.ortcy.wrap.ortinf import OrtSession

    X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
    Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])
    node = make_node("Add", ["X", "Y"], ["Z"])
    graph = make_graph([node], "add", [X, Y], [Z])
    onnx_model = make_model(
        graph, opset_imports=[make_opsetid("", 18)], ir_version=8
    )
    check_model(onnx_model)
    with open("model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    self.assertExists(name)

    session = OrtSession("model.onnx")
    x = numpy.random.randn(2, 3).astype(numpy.float32)
    y = numpy.random.randn(2, 3).astype(numpy.float32)
    got =session.run([x, y])

    print(got)

The signature is different compare to onnxruntime
``session.run(None, {"X": x, "Y": y})`` to increase performance.
This binding supports custom operators as well.
A benchmark :ref:`l-cython-pybind11-ort-bindings` compares
:epkg:`onnxruntime` to this new binding.
