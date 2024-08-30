from typing import Tuple, Union


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
    :param sup_title: figure title
    :return: axes

    .. runpython::

        import pandas
        from onnx_extended.plotting.data import vhistograms_data

        df = pandas.DataFrame(vhistograms_data())
        print(df.head())

    .. plot::

        import pandas
        from onnx_extended.plotting.data import vhistograms_data
        from onnx_extended.plotting.benchmark import vhistograms

        df = pandas.DataFrame(vhistograms_data())
        vhistograms(df)
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


def hhistograms(
    df: "pandas.DataFrame",  # noqa: F821
    keys: Union[str, Tuple[str, ...]] = "name",
    metric: str = "average",
    baseline: str = "baseline",
    title: str = "Benchmark",
    limit: int = 50,
    ax=None,
):
    """
    Histograms with error bars.
    Shows the first best performances.

    :param df: data
    :param keys: columns to graph by
    :param metric: metric to display
    :param baseline: column `keys[-1]`, no matter what it should be displayed
    :param title: graph title
    :param limit: number of performances to display
    :param ax: existing axes
    :return: axes

    .. runpython::

        import pandas
        from onnx_extended.plotting.data import hhistograms_data

        df = pandas.DataFrame(hhistograms_data())
        print(df.head())

    .. plot::

        import pandas
        from onnx_extended.plotting.data import hhistograms_data
        from onnx_extended.plotting.benchmark import hhistograms

        df = pandas.DataFrame(hhistograms_data())
        hhistograms(df, keys=("input", "name"))
    """
    import pandas

    if not isinstance(keys, (tuple, list)):
        keys = (keys,)

    dfm = (
        df[[*keys, metric]]
        .groupby(list(keys), as_index=False)
        .agg(["mean", "min", "max"])
        .copy()
    )
    if dfm.shape[1] == 3:
        dfm = dfm.reset_index(drop=False)
    dfm.columns = [*keys, metric, "min", "max"]
    dfi = dfm.sort_values(metric).reset_index(drop=True)
    base = dfi[dfi[keys[-1]].str.contains(baseline)]
    not_base = dfi[~dfi[keys[-1]].str.contains(baseline)].reset_index(drop=True)
    if not_base.shape[0] > limit:
        not_base = not_base[:limit]
    merged = pandas.concat([base, not_base], axis=0)
    merged = merged.sort_values(metric).reset_index(drop=True).set_index(list(keys))

    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, merged.shape[0] / 2))

    err_min = merged[metric] - merged["min"]
    err_max = merged["max"] - merged[metric]
    merged[[metric]].plot.barh(
        ax=ax,
        title=title,
        xerr=[err_min, err_max],
    )
    b = df.loc[df[keys[-1]] == baseline, metric].mean()
    ax.plot([b, b], [0, df.shape[0]], "r--")
    ax.set_xlim(
        [
            (df["min_exec"].min() + df[metric].min()) / 2,
            (df[metric].max() + df[metric].max()) / 2,
        ]
    )
    # ax.set_xscale("log")

    fig.tight_layout()
    return ax
