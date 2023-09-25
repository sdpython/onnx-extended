Trees
=====

Parallelization parameters
++++++++++++++++++++++++++

The latency of a tree ensemble depends on the tree size
and the machine it runs on. The following example
takes a model using a TreeEnsembleRegressor and replaces
it with a custom node and additional parameters to tune
the parallelization. The kernel is a custom operator
for :epkg:`onnxruntime`.

.. runpython::
    :showcode:

    import os
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    from skl2onnx import to_onnx
    from onnxruntime import InferenceSession, SessionOptions
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
    from onnx_extended.ortops.optim.cpu import get_ort_ext_libs
    from onnx_extended.ortops.optim.optimize import (
        change_onnx_operator_domain,
        get_node_attribute,
    )


    # The dimension of the problem.
    batch_size = 20
    n_features = 4
    n_trees = 2
    max_depth = 3

    # Let's create model.
    X, y = make_regression(batch_size * 2, n_features=n_features, n_targets=1)
    X, y = X.astype(np.float32), y.astype(np.float32)
    model = RandomForestRegressor(n_trees, max_depth=max_depth, verbose=0)
    model.fit(X[:batch_size], y[:batch_size])
    onx = to_onnx(model, X[:1], target_opset=17)

    # onnx-extended implements custom kernels
    # to tune the parallelization parameters.
    # It requires to replace the onnx node by another
    # one including the optimization parameters.

    optim_params = {
        "parallel_tree": 80,
        "parallel_tree_N": 80,
        "parallel_N": 80,
        "batch_size_tree": 2,
        "batch_size_rows": 2,
        "use_node3": 0,
    }

    # Let's replace the node TreeEnsembleRegressor with a new one
    # and additional parameters.


    def transform_model(onx, op_name, **kwargs):
        att = get_node_attribute(onx.graph.node[0], "nodes_modes")
        modes = ",".join(map(lambda s: s.decode("ascii"), att.strings))
        return change_onnx_operator_domain(
            onx,
            op_type=op_name,
            op_domain="ai.onnx.ml",
            new_op_domain="onnx_extented.ortops.optim.cpu",
            nodes_modes=modes,
            **kwargs,
        )


    modified_onx = transform_model(onx, "TreeEnsembleRegressor", **optim_params)
    print(onnx_simple_text_plot(modified_onx))

    # Let's check it is working.

    opts = SessionOptions()
    r = get_ort_ext_libs()
    opts.register_custom_ops_library(r[0])

    sess = InferenceSession(
        modified_onx.SerializeToString(),
        opts,
        providers=["CPUExecutionProvider"],
    )
    feeds = {"X": X}
    print(sess.run(None, feeds))
