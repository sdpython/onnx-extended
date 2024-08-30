
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
    :process:

    import os
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    from skl2onnx import to_onnx
    from onnx_extended.tools.run_onnx import save_for_benchmark_or_test
    from onnx_extended.args import get_parsed_args

    # The dimension of the problem.

    args = get_parsed_args(
        "create_bench",
        **dict(
            batch_size=(10, "batch size"),
            n_features=(10, "number of features"),
            n_trees=(10, "number of trees"),
            max_depth=(3, "max detph"),
        ),
    )

    batch_size = args.batch_size
    n_features = args.n_features
    n_trees = args.n_trees
    max_depth = args.max_depth

    # Let's create model.
    X, y = make_regression(
        batch_size + 2**max_depth * 2, n_features=n_features, n_targets=1
    )
    X, y = X.astype(np.float32), y.astype(np.float32)

    print(
        f"train RandomForestRegressor n_trees={n_trees} "
        f"n_features={n_features} batch_size={batch_size} "
        f"max_depth={max_depth}"
    )
    model = RandomForestRegressor(n_trees, max_depth=max_depth, n_jobs=-1, verbose=1)
    model.fit(X[:-batch_size], y[:-batch_size])

    # target_opset is used to select opset an old version of onnxruntime can process.
    print("conversion to onnx")
    onx = to_onnx(model, X[:1], target_opset=17)

    print(f"size: {len(onx.SerializeToString())}")

    # Let's save the model and the inputs on disk.
    folder = f"test_ort_version-F{n_features}-T{n_trees}-D{max_depth}-B{batch_size}"
    if not os.path.exists(folder):
        os.mkdir(folder)

    print("create the benchmark")
    inputs = [X[:batch_size]]
    save_for_benchmark_or_test(folder, "rf", onx, inputs)

    print("end")
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
    import platform
    import psutil
    from onnx_extended.tools.run_onnx import bench_virtual
    from onnx_extended.args import get_parsed_args

    args = get_parsed_args(
        "run_bench",
        **dict(
            test_name=(
                "test_ort_version-F10-T10-D3-B10",
                "folder containing the benchmark to run",
            ),
        ),
    )

    name = args.test_name
    folder = os.path.abspath(f"{name}/rf")
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Unable to find {folder!r}.")
    virtual_env = os.path.abspath("venv")

    runtimes = ["onnxruntime"]
    modules = [
        {"onnx-extended": "0.3.0", "onnx": "1.15.0", "onnxruntime": "1.19.0"},
        {"onnx-extended": "0.3.0", "onnx": "1.15.0", "onnxruntime": "1.18.0"},
        {"onnx-extended": "0.2.3", "onnx": "1.15.0", "onnxruntime": "1.17.3"},
        {"onnx-extended": "0.2.3", "onnx": "1.15.0", "onnxruntime": "1.16.3"},
        {"onnx-extended": "0.2.3", "onnx": "1.15.0", "onnxruntime": "1.15.1"},
        {"onnx-extended": "0.2.3", "onnx": "1.15.0", "onnxruntime": "1.14.1"},
        {"onnx-extended": "0.2.3", "onnx": "1.15.0", "onnxruntime": "1.13.1"},
        {"onnx-extended": "0.2.3", "onnx": "1.15.0", "onnxruntime": "1.12.1"},
    ]

    print("--------------------------")
    print(platform.machine(), platform.version(), platform.platform())
    print(platform.processor())
    print(f"RAM: {psutil.virtual_memory().total / (1024.0 **3):1.3f} GB")
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Total cores:", psutil.cpu_count(logical=True))
    print("--------------------------")
    print(name)
    for t in range(3):
        print("--------------------------")
        df = bench_virtual(
            folder,
            virtual_env,
            verbose=1,
            modules=modules,
            runtimes=runtimes,
            warmup=5,
            repeat=10,
            save_as_dataframe=f"result-{name}.t{t}.csv",
            filter_fct=lambda rt, modules: True,
        )

        columns = ["runtime", "b_avg_time", "runtime", "v_onnxruntime"]
        df[columns].to_csv(f"summary-{name}.t{t}.csv")
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
