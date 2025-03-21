import json
import warnings
from typing import List, Optional, Union
from pandas import DataFrame


_mapping_types = {
    "float": "F",
    "double": "D",
    "float16": "H",
    "uint8": "U8",
    "uint16": "U16",
    "uint32": "U32",
    "uint64": "U64",
    "int8": "I8",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
}


def _process_shape(shape_df):
    if isinstance(shape_df, float) or len(shape_df) == 0:
        return ""
    values = []
    for val in shape_df:
        if len(val) != 1:
            raise ValueError(f"Unable to process shape {val!r} from {values!r}.")
        for _k, _v in val.items():
            k, v = _k, _v
            break
        if v:
            vs = "x".join(map(str, v))
            values.append(f"{_mapping_types.get(k,k)}[{vs}]")
        else:
            values.append(f"{_mapping_types.get(k,k)}")
    return "+".join(values)


def post_process_df_profile(
    df: DataFrame,
    first_it_out: bool = False,
    agg: bool = False,
    agg_op_name: bool = True,
    with_shape: bool = False,
) -> DataFrame:
    """
    Post-processed a dataframe obtained after profiling onnxruntime.
    It adds a column for a more explicit event name and adds
    a column for the iteration number

    :param agg: aggregate the result
    :param first_it_out: leave the first iteration
        out of the aggregation
    :param agg_op_name: aggregate on operator name or operator index
    :param with_shape: keep the shape to aggregate
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
        if with_shape:
            df["args_input_type_shape"] = df["args_input_type_shape"].apply(
                _process_shape
            )
            df["args_output_type_shape"] = df["args_output_type_shape"].apply(
                _process_shape
            )
        else:
            df = df.drop(["args_input_type_shape", "args_output_type_shape"], axis=1)
        if first_it_out:
            df["it==0"] = (df["iteration"] <= 0).astype(int)
        return df

    agg_cols = ["cat", "args_node_index", "args_op_name", "args_provider", "event_name"]
    if with_shape:
        agg_cols.append("args_input_type_shape")
        df["args_input_type_shape"] = df["args_input_type_shape"].apply(_process_shape)
        df["args_output_type_shape"] = df["args_output_type_shape"].apply(
            _process_shape
        )
    else:
        df = df.drop(["args_input_type_shape", "args_output_type_shape"], axis=1)

    if first_it_out:
        df["it==0"] = (df["iteration"] <= 0).astype(int)
        agg_cols.insert(0, "it==0")
    if agg_op_name:
        del agg_cols[agg_cols.index("args_node_index")]
    for c in agg_cols:
        df[c] = df[c].fillna("")
    df["dur"] = df["dur"].fillna(0)
    agg = df[[*agg_cols, "dur"]].groupby(agg_cols).sum()
    return agg


def js_profile_to_dataframe(
    filename: str,
    as_df: bool = True,
    first_it_out: bool = False,
    agg: bool = False,
    agg_op_name: bool = False,
    with_shape: bool = False,
) -> Union[List, DataFrame]:
    """
    Profiles the execution of an onnx graph with onnxruntime.

    :param filename: filename holding the profiling stored in json format
    :param as_df: returns the
    :param first_it_out: if aggregated, leaves the first iteration out
    :param agg: aggregate by event
    :param agg_op_name: aggregate on operator name or operator index
    :param with_shape: keep the shape before aggregating
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
            DataFrame(rows),
            first_it_out=first_it_out,
            agg=agg,
            agg_op_name=agg_op_name,
            with_shape=with_shape,
        )
    return rows


def _preprocess_graph1(df):
    df = df.copy()
    df["args_provider"] = df["args_provider"].apply(
        lambda s: s.replace("ExecutionProvider", "") if isinstance(s, str) else s
    )
    agg_cols = ["dur", "args_op_name", "args_provider"]
    for c in ["it==0", "args_input_type_shape"]:
        if c in df.columns:
            agg_cols.append(c)
    if "it==0" in df.columns:
        vs = ["t>=1", "t=0"]
        df["it==0"] = df["it==0"].apply(lambda v: vs[v])
    gr_dur = df[agg_cols].groupby(agg_cols[1:]).sum().sort_values("dur")
    gr_n = df[agg_cols].groupby(agg_cols[1:]).count()
    gr_n = gr_n.loc[gr_dur.index, :]
    gr_n.columns = ["count"]
    gr = gr_dur.merge(gr_n, left_index=True, right_index=True, how="outer")
    gr["ratio"] = gr["dur"] / gr["dur"].sum()
    return gr_dur, gr_n, gr


def _preprocess_graph2(df):
    df = df.reset_index(drop=False).copy()
    df["args_node_index"] = df["args_node_index"].apply(
        lambda i: int(i) if i not in {None, ""} else -1
    )
    df["args_provider"] = df["args_provider"].apply(
        lambda s: s.replace("ExecutionProvider", "") if isinstance(s, str) else s
    )
    df = df[(df["cat"] == "Node") & (df["event_name"] == "kernel_time")]
    agg_cols = ["dur", "args_node_index", "args_op_name", "args_provider"]
    for c in ["it==0", "args_input_type_shape"]:
        if c in df.columns:
            agg_cols.append(c)
    if "it==0" in df.columns:
        vs = ["t>=1", "t=0"]
        df["it==0"] = df["it==0"].apply(lambda v: vs[v])
    df = df[agg_cols].groupby(agg_cols[1:]).sum()
    df = df.sort_index(ascending=False)
    df["ratio"] = df["dur"] / df["dur"].sum()
    return df


def plot_ort_profile(
    df: DataFrame,
    ax0: Optional["matplotlib.axes.Axes"] = None,
    ax1: Optional["matplotlib.axes.Axes"] = None,
    title: Optional[str] = None,
) -> "matplotlib.axes.Axes":
    """
    Plots time spend in computation based on a dataframe
    produced by function :func:`js_profile_to_dataframe`.

    :param df: dataframe
    :param ax0: first axis to draw time
    :param ax1: second axis to draw occurences
    :param title: graph title
    :return: the graph
    """
    fontsize = 10
    if ax0 is None:
        import matplotlib.pyplot as plt

        ax0 = plt.gca()

    if "args_provider" in df.columns:
        # Aggregation by operator
        gr_dur, gr_n, _ = _preprocess_graph1(df)
        gr_dur.plot.barh(ax=ax0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax0.set_xticklabels(ax0.get_xticklabels(), fontsize=fontsize)
            ax0.get_yaxis().set_label_text("")
            ax0.set_yticklabels(
                ax0.get_yticklabels(), rotation=45, ha="right", fontsize=fontsize
            )
        if title is not None:
            ax0.set_title(title)
        if ax1 is not None:
            gr_n.plot.barh(ax=ax1)
            ax1.set_title("n occurences")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=fontsize)
                ax1.get_yaxis().set_label_text("")
                ax1.set_yticklabels(
                    ax1.get_yticklabels(), rotation=45, ha="right", fontsize=fontsize
                )
        return ax0

    df = _preprocess_graph2(df)
    df[["dur"]].plot.barh(ax=ax0)
    if title is not None:
        ax0.set_title(title)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax0.set_xticklabels(ax0.get_xticklabels(), fontsize=fontsize)
        ax0.get_yaxis().set_label_text("")
        ax0.set_yticklabels(ax0.get_yticklabels(), fontsize=fontsize)
    return ax0


def plot_ort_profile_timeline(
    df: DataFrame,
    ax: Optional["matplotlib.axes.Axes"] = None,
    iteration: int = -2,
    title: Optional[str] = None,
    quantile: float = 0.5,
    fontsize: int = 12,
) -> "matplotlib.axes.Axes":
    """
    Creates a timeline based on a dataframe
    produced by function :func:`js_profile_to_dataframe`.

    :param df: dataframe
    :param ax: first axis to draw time
    :param iteration: iteration to plot, negative value to start from the end
    :param title: graph title
    :param quantile: draw the 10% less consuming operators in a different color
    :param fontsize: font size
    :return: the graph
    """
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()

    df = df.copy()
    df["iteration"] = df["iteration"].astype(int)
    iterations = set(df["iteration"])
    n_iter = iteration if iteration >= 0 else max(iterations) + 1 + iteration
    dfi = df[df["iteration"] == n_iter]
    assert dfi.shape[0] > 0, f"Iteration {iteration} cannot be found in {iterations}."

    if "fence_before" in set(dfi["event_name"]):
        started = {}
        data = []
        for irow in dfi.iterrows():
            assert isinstance(
                irow, tuple
            ), f"pandas has changed its api, type is {type(row)}"
            assert len(irow) == 2, f"pandas has changed its api, row is {row}"
            row = irow[1]
            it = row["iteration"]
            op_type = row["args_op_name"]
            op_name = row["op_name"]
            event_name = row["event_name"]
            provider = row["args_provider"]
            ts = float(row["ts"])
            dur = float(row["dur"])
            if event_name == "fence_before":
                started[op_type, op_name, it] = dict(
                    op_name=op_name, op_type=op_type, begin=ts
                )
            elif event_name == "kernel_time":
                obs = started[op_type, op_name, it]
                obs["duration"] = dur
                obs["begin_kernel"] = ts
                obs["provider"] = provider
            elif event_name == "fence_after":
                obs = started[op_type, op_name, it]
                obs["end"] = ts
                data.append(obs)
                del started[op_type, op_name, it]
            else:
                assert event_name in {
                    "SequentialExecutor::Execute",
                    "model_run",
                }, f"Unexpected event_name={event_name!r}, row={row}"
    else:
        # New format
        data = []
        for irow in dfi.iterrows():
            row = irow[1]
            if row["event_name"] != "kernel_time":
                continue
            obs = dict(
                duration=float(row["dur"]),
                op_name=row["op_name"],
                op_type=row["args_op_name"],
                provider=row["args_provider"],
                begin=float(row["ts"]),
                end=float(row["ts"]) + float(row["dur"]),
                begin_kernel=float(row["ts"]),
            )
            data.append(obs)

    # durations
    data_dur = list(sorted(d["duration"] for d in data))
    threshold = data_dur[int(quantile * len(data_dur))]
    origin = dfi["ts"].min()

    colors = ["blue", "green", "red", "orange"]

    import matplotlib.patches as mpatches

    cs = [0, 0]
    for i, obs in enumerate(data):
        dur = obs["duration"]
        cat = int(dur >= threshold)

        # color
        color = colors[cat * 2 + cs[cat] % 2]
        cs[cat] += 1

        # rectangle
        t1 = obs["begin"] - origin
        t2 = obs["end"] - origin
        shape = mpatches.Rectangle((0, t1), 1, t2 - t1, ec="none", color=color)
        ax.add_artist(shape)
        tk1 = obs["begin_kernel"] - origin
        tk2 = (obs["begin_kernel"] + obs["duration"]) - origin
        ax.plot([0, 1], [tk1, tk1], "b--")
        ax.plot([0, 1], [tk2, tk2], "b--")
        if i == 0:
            ax.plot([0, 2], [tk1, tk1], "b")
        elif i == len(data) - 1:
            ax.plot([0, 2], [tk2, tk2], "b")

        # text
        y = (tk1 + tk2) / 2
        text = obs["op_type"]
        prov = obs["provider"].replace("ExecutionProvider", "")
        name = obs["op_name"]
        if len(name) >= 10:
            name = name[:5] + "..." + name[5:]
        ax.text(1, y, f"{i}:{prov}:{text}-{name}", fontsize=fontsize, va="center")

    ax.invert_yaxis()
    return ax
