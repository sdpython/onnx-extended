
Profiling onnxruntime
=====================

After the profiling comes the analyze.
Here are two kinds.

Graph 1
+++++++

.. plot::

    import numpy as np
    from onnx import TensorProto
    from onnx.checker import check_model
    from onnx.helper import (
        make_model,
        make_graph,
        make_node,
        make_opsetid,
        make_tensor_value_info,
    )
    from onnx.numpy_helper import from_array
    from onnxruntime import InferenceSession, SessionOptions
    import matplotlib.pyplot as plt
    from onnx_extended.tools.js_profile import (
        js_profile_to_dataframe,
        plot_ort_profile,
        _process_shape,
    )

    def get_model():
        model_def0 = make_model(
            make_graph(
                [
                    make_node("Add", ["X", "init1"], ["X1"]),
                    make_node("Abs", ["X"], ["X2"]),
                    make_node("Add", ["X", "init3"], ["inter"]),
                    make_node("Mul", ["X1", "inter"], ["Xm"]),
                    make_node("Sub", ["X2", "Xm"], ["final"]),
                ],
                "test",
                [make_tensor_value_info("X", TensorProto.FLOAT, [None])],
                [make_tensor_value_info("final", TensorProto.FLOAT, [None])],
                [
                    from_array(np.array([1], dtype=np.float32), name="init1"),
                    from_array(np.array([3], dtype=np.float32), name="init3"),
                ],
            ),
            opset_imports=[make_opsetid("", 18)],
        )
        check_model(model_def0)
        return model_def0

    sess_options = SessionOptions()
    sess_options.enable_profiling = True
    sess = InferenceSession(
        get_model().SerializeToString(),
        sess_options,
        providers=["CPUExecutionProvider"],
    )
    for _ in range(11):
        sess.run(None, dict(X=np.arange(10).astype(np.float32)))
    prof = sess.end_profiling()

    df = js_profile_to_dataframe(prof, first_it_out=True)
    print(df.head())

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plot_ort_profile(df, ax[0], ax[1], "test_title")

Aggregated Graph
++++++++++++++++

.. plot::

    import numpy as np
    from onnx import TensorProto
    from onnx.checker import check_model
    from onnx.helper import (
        make_model,
        make_graph,
        make_node,
        make_opsetid,
        make_tensor_value_info,
    )
    from onnx.numpy_helper import from_array
    from onnxruntime import InferenceSession, SessionOptions
    import matplotlib.pyplot as plt
    from onnx_extended.tools.js_profile import (
        js_profile_to_dataframe,
        plot_ort_profile,
        _process_shape,
    )

    def get_model():
        model_def0 = make_model(
            make_graph(
                [
                    make_node("Add", ["X", "init1"], ["X1"]),
                    make_node("Abs", ["X"], ["X2"]),
                    make_node("Add", ["X", "init3"], ["inter"]),
                    make_node("Mul", ["X1", "inter"], ["Xm"]),
                    make_node("Sub", ["X2", "Xm"], ["final"]),
                ],
                "test",
                [make_tensor_value_info("X", TensorProto.FLOAT, [None])],
                [make_tensor_value_info("final", TensorProto.FLOAT, [None])],
                [
                    from_array(np.array([1], dtype=np.float32), name="init1"),
                    from_array(np.array([3], dtype=np.float32), name="init3"),
                ],
            ),
            opset_imports=[make_opsetid("", 18)],
        )
        check_model(model_def0)
        return model_def0

    sess_options = SessionOptions()
    sess_options.enable_profiling = True
    sess = InferenceSession(
        get_model().SerializeToString(),
        sess_options,
        providers=["CPUExecutionProvider"],
    )
    for _ in range(11):
        sess.run(None, dict(X=np.arange(10).astype(np.float32)))
    prof = sess.end_profiling()

    df = js_profile_to_dataframe(prof, first_it_out=True, agg=True)
    print(df.head())

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_ort_profile(df, ax, title="test_title")
    fig.tight_layout()    
