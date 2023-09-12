import json
from typing import List, Optional, Tuple, Union
from pandas import DataFrame


def post_process_df_profile(
    df: DataFrame,
    first_it_out: bool = False,
    agg: bool = False,
    agg_op_name: bool = True,
) -> DataFrame:
    """
    Post-processed a dataframe obtained after profiling onnxruntime.
    It adds a column for a more explicit event name and adds
    a column for the iteration number

    :param agg: aggregate the result
    :param first_it_out: leave the first iteration
        out of the aggregation
    :param agg_op_name: aggregate on operator name or operator index
    :return: DataFrame
    """
    events = {"kernel_time", "fence_after", "fence_before"}

    def sep_event(s):
        for e in events:
            if s.endswith(e):
                return e
        return s

    df = df.copy()
    df["event_name"] = df["name"].apply(sep_event)
    df["iteration"] = -1
    current = -1
    for i in range(df.shape[0]):
        if df.loc[i, "name"] == "SequentialExecutor::Execute":
            current += 1
        df.loc[i, "iteration"] = current

    if not agg:
        return df

    agg_cols = ["cat", "args_node_index", "args_op_name", "args_provider", "event_name"]
    if first_it_out:
        df["it==0"] = (df["iteration"] <= 0).astype(int)
        agg_cols.insert(0, "it==0")
    if agg_op_name:
        del agg_cols[agg_cols.index("args_node_index")]
    for c in agg_cols:
        df[c] = df[c].fillna("")
    df["dur"] = df["dur"].fillna(0)
    agg = df[agg_cols + ["dur"]].groupby(agg_cols).sum()
    return agg


def js_profile_to_dataframe(
    filename: str,
    as_df: bool = True,
    first_it_out: bool = False,
    agg: bool = False,
    agg_op_name: bool = False,
) -> Union[List, DataFrame]:
    """
    Profiles the execution of an onnx graph with onnxruntime.

    :param filename: filename holding the profiling stored in json format
    :param as_df: returns the
    :param first_it_out: if aggregated, leaves the first iteration out
    :param agg: aggregate by event
    :param agg_op_name: aggregate on operator name or operator index
    :return: DataFrame or dictionary
    """
    with open(filename, "r") as f:
        content = f.read()
    js = json.loads(content)

    suffixes = ["_kernel_time", "_fence_before", "_fence_after"]
    rows = []
    for row in js:
        if "args" in row and isinstance(row["args"], dict):
            for k, v in row["args"].items():
                row[f"args_{k}"] = v
            del row["args"]
        name = row["name"]
        for suf in suffixes:
            if name.endswith(suf):
                changed = name[: -len(suf)]
                row["op_name"] = changed
                break
        rows.append(row)
    if as_df:
        return post_process_df_profile(
            DataFrame(rows), first_it_out=first_it_out, agg=agg, agg_op_name=agg_op_name
        )
    return rows


def plot_ort_profile(
    df: DataFrame,
    ax0: Optional["matplotlib.axes.Axes"] = None,
    ax1: Optional["matplotlib.axes.Axes"] = None,
    title: Optional[str] = None,
) -> Tuple["matplotlib.axes.Axes", DataFrame]:
    """
    Plots time spend in computation based on dataframe
    produced by function :func:`js_profile_to_dataframe`.

    :param df: dataframe
    :param ax0: first axis to draw time
    :param ax1: second axis to draw occurences
    :param title: graph title
    :return: the graph, the data of the graph
    """
    if ax0 is None:
        import matplotlib as plt

        ax0 = plt.gca()  # pragma: no cover

    if "args_provider" in df.columns:
        # Aggregation by operator
        df = df.copy()
        df["args_provider"] = df["args_provider"].apply(
            lambda s: s.replace("ExecutionProvider", "") if isinstance(s, str) else s
        )
        gr_dur = (
            df[["dur", "args_op_name", "args_provider"]]
            .groupby(["args_provider", "args_op_name"])
            .sum()
            .sort_values("dur")
        )
        gr_dur.plot.barh(ax=ax0)
        ax0.get_yaxis().set_label_text("")
        ax0.set_yticklabels(ax0.get_yticklabels(), rotation=45, ha="right")
        if title is not None:
            ax0.set_title(title)
        if ax1 is not None:
            gr_n = (
                df[["dur", "args_op_name", "args_provider"]]
                .groupby(["args_provider", "args_op_name"])
                .count()
                .sort_values("dur")
            )
            gr_n = gr_n.loc[gr_dur.index, :]
            gr_n.plot.barh(ax=ax1)
            ax1.set_title("n occurences")
            ax1.get_yaxis().set_label_text("")
            ax1.set_yticklabels(ax1.get_yticklabels(), rotation=45, ha="right")
        return ax0, gr_dur

    df = df.reset_index(drop=False).copy()
    df["args_provider"] = df["args_provider"].apply(
        lambda s: s.replace("ExecutionProvider", "") if isinstance(s, str) else s
    )
    df = df[
        (df["it==0"] == 0) & (df["cat"] == "Node") & (df["event_name"] == "kernel_time")
    ]
    df = (
        df[["args_node_index", "args_op_name", "args_provider", "dur"]]
        .groupby(
            [
                "args_node_index",
                "args_provider",
                "args_op_name",
            ]
        )
        .sum()
    )
    df = df.sort_index(ascending=False)
    df.plot.barh(ax=ax0)
    ax0.get_yaxis().set_label_text("")
    if title is not None:
        ax0.set_title(title)
    return ax0, df
