
Debug Intermediate Results
==========================

The reference evaluation (:class:`onnx_extended.reference.CReferenceEvaluator`)
can return all intermediate results. :epkg:`onnxruntime` does not
unless the onnx model is split to extract the intermediate results.
Function :func:`enumerate_ort_run <onnx_extended.tools.ort_debug.enumerate_ort_run>`
creates many models, inputs are always the same, new outputs are intermediate
results of an original model.

.. runpython::
    :showcode:

    import logging
    import numpy as np
    from onnx import TensorProto
    from onnx.helper import (
        make_model,
        make_node,
        make_graph,
        make_tensor_value_info,
        make_opsetid,
    )
    from onnx.checker import check_model
    from onnx_extended.tools.ort_debug import enumerate_ort_run

    logging.getLogger("onnx-extended").setLevel(logging.ERROR)

    def get_model():
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info("Z", TensorProto.INT64, [None, None])
        graph = make_graph(
            [
                make_node("Add", ["X", "Y"], ["z1"]),
                make_node("Mul", ["X", "z1"], ["z2"]),
                make_node("Cast", ["z2"], ["Z"], to=TensorProto.INT64),
            ],
            "add",
            [X, Y],
            [Z],
        )
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 18)], ir_version=8
        )
        check_model(onnx_model)
        return onnx_model

    model = get_model()
    feeds = {
        "X": np.arange(4).reshape((2, 2)).astype(np.float32),
        "Y": np.arange(4).reshape((2, 2)).astype(np.float32),
    }

    for names, outs, node in enumerate_ort_run(model, feeds, verbose=2):
        print(f"node: {op_type}")
        for n, o in zip(names, outs):
            print(f"   {n}:{o.dtype}:{o.shape}")
