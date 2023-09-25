
Compare multiple versions of onnxruntime
========================================

One important task is check the onnxruntime does not run
slower for any new version. The following tools were developped
for that purpose.

Step 1: save a test
+++++++++++++++++++

We need to first to save the model and the input onnxruntime must
be evaluated on. This is done with function :func:`save_for_benchmark_or_test
<onnx_extended.tools.run_onnx.save_for_benchmark_or_test>`.

.. runpython::
    :showcode:

    import os
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    from skl2onnx import to_onnx
    from onnx_extended.tools.run_onnx import save_for_benchmark_or_test

    # The dimension of the problem.
    batch_size = 100
    n_features = 10
    n_trees = 2
    max_depth = 3

    # Let's create model.
    X, y = make_regression(batch_size * 2, n_features=n_features, n_targets=1)
    X, y = X.astype(np.float32), y.astype(np.float32)
    model = RandomForestRegressor(n_trees, max_depth=max_depth, verbose=0)
    model.fit(X[:batch_size], y[:batch_size])

    # target_opset is used to select opset an old version of onnxruntime can process.
    onx = to_onnx(model, X[:1], target_opset=17)

    # Let's save the model and the inputs on disk.
    folder = "test_ort_version"
    if not os.path.exists(folder):
        os.mkdir(folder)

    inputs = [X]
    save_for_benchmark_or_test(folder, "rf", onx, inputs)

    # Let's see what was saved.
    for r, d, f in os.walk(folder):
        for name in f:
            full_name = os.path.join(r, name)
            print(f"{os.stat(full_name).st_size / 2 ** 10:1.1f} Kb: {full_name}")

The output are not used to measure the performance but it can be
used to evaluate the discrepancies.

Step 2: evaluate multiple versions of onnxruntime
+++++++++++++++++++++++++++++++++++++++++++++++++

It calls function :func:`bench_virtual <onnx_extended.tools.run_onnx.bench_virtual>`.

.. code-block:: python

    import os
    from onnx_extended.tools.run_onnx import bench_virtual

    folder = os.path.abspath("test_ort_version/rf")
    virtual_env = os.path.abspath("venv")

    runtimes = ["ReferenceEvaluator", "CReferenceEvaluator", "onnxruntime"]
    modules = [
        {"onnx-extended": "0.2.1", "onnx": "1.14.1", "onnxruntime": "1.16.0"},
        {"onnx-extended": "0.2.1", "onnx": "1.14.1", "onnxruntime": "1.15.1"},
        {"onnx-extended": "0.2.1", "onnx": "1.14.1", "onnxruntime": "1.14.1"},
        {"onnx-extended": "0.2.1", "onnx": "1.14.1", "onnxruntime": "1.13.1"},
        {"onnx-extended": "0.2.1", "onnx": "1.14.1", "onnxruntime": "1.12.1"},
    ]
    filter_fct = (
        lambda rt, modules: rt == "onnxruntime" or modules["onnxruntime"] == "1.16.0"
    )

    df = bench_virtual(
        folder,
        virtual_env,
        verbose=1,
        modules=modules,
        runtimes=runtimes,
        warmup=5,
        repeat=10,
        save_as_dataframe="results.csv",
        filter_fct=filter_fct,
    )

    columns = ["runtime", "b_avg_time", "runtime", "v_onnxruntime"]
    print(df[columns])

The output would look like:

::

    [bench_virtual] 1/5 18:01:02 onnx==1.14.1 onnx-extended==0.2.1 onnxruntime==1.16.0
    [bench_virtual] 2/5 18:01:06 onnx==1.14.1 onnx-extended==0.2.1 onnxruntime==1.15.1
    [bench_virtual] 3/5 18:01:09 onnx==1.14.1 onnx-extended==0.2.1 onnxruntime==1.14.1
    [bench_virtual] 4/5 18:01:12 onnx==1.14.1 onnx-extended==0.2.1 onnxruntime==1.13.1
    [bench_virtual] 5/5 18:01:15 onnx==1.14.1 onnx-extended==0.2.1 onnxruntime==1.12.1
                runtime  b_avg_time              runtime v_onnxruntime
    0   ReferenceEvaluator    0.001879   ReferenceEvaluator        1.16.0
    1  CReferenceEvaluator    0.000042  CReferenceEvaluator        1.16.0
    2          onnxruntime    0.000013          onnxruntime        1.16.0
    3          onnxruntime    0.000012          onnxruntime        1.15.1
    4          onnxruntime    0.000017          onnxruntime        1.14.1
    5          onnxruntime    0.000012          onnxruntime        1.13.1
    6          onnxruntime    0.000011          onnxruntime        1.12.1

The differences are not significant on such small model except for
the python runtime.
