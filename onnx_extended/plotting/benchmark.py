def vhistograms(
    df: "pandas.DataFrame",  # noqa: F821
    metric: str = "time",
    name: str = "name",
    batch_size: str = "batch_size",
    voc_size: str = "voc_size",
    sup_title: str = "Compares Implementations of TfIdfVectorizer",
):
    """
    Histograms with error bars.

    :param df: data
    :param metric: metric to show
    :param name: experiment name
    :param batch_size: first column for the variations
    :param voc_size: second column for the variations
    :param sub_title: figure title
    :return: axes

    .. runpython::

        from onnx_extended.plotting.data import histograms_data

        df = pandas.DataFrame(histograms_data())
        print(df.head())

    .. plot::

        from onnx_extended.plotting.data import histograms_data
        from onnx_extended.plotting.benchmark import vhistograms

        df = pandas.DataFrame(histograms_data())
        ax = vhistograms(df)
    """
    import matplotlib.pyplot as plt

    batch_sizes = list(sorted(set(df[batch_size])))
    voc_sizes = list(sorted(set(df[voc_size])))
    B = len(batch_sizes)
    V = len(voc_sizes)

    fig, ax = plt.subplots(V, B, figsize=(B * 2, V * 2), sharex=True, sharey=True)
    fig.suptitle(sup_title)

    for b in range(B):
        for v in range(V):
            aa = ax[v, b]
            sub = df[
                (df[batch_size] == batch_sizes[b]) & (df[voc_size] == voc_sizes[v])
            ][[name, metric]].set_index(name)
            if 0 in sub.shape:
                continue
            sub[metric].plot.bar(
                ax=aa, logy=True, rot=0, color=["blue", "orange", "green"]
            )
            if b == 0:
                aa.set_ylabel(f"vocabulary={voc_sizes[v]}")
            if v == V - 1:
                aa.set_xlabel(f"batch_size={batch_sizes[b]}")
            aa.grid(True)

    if ax is None:
        fig.tight_layout()
    return ax
