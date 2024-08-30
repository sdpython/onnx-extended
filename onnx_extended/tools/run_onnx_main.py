import json
import sys
from textwrap import dedent
from argparse import ArgumentParser
from onnx_extended.tools.run_onnx import TestRun


def get_parser():
    parser = ArgumentParser(
        prog="run_onnx_main",
        description=dedent(
            """
        Runs a Benchmark.
        """
        ),
    )
    parser.add_argument(
        "-p", "--path", type=str, required=True, help="path to the test definition"
    )
    parser.add_argument("-i", "--index", type=int, default=0)
    parser.add_argument("-r", "--repeat", type=int, default=10)
    parser.add_argument("-w", "--warmup", type=int, default=5)
    parser.add_argument(
        "-e",
        "--runtime",
        default="ReferenceEvaluator",
        help="""
        A choice among
        CReferenceEvaluator, ReferenceEvaluator, onnxruntime, CustomTreeEnsemble.x.x.x,
        CustomTreeEnsemble.x.x.x is based on a custom node,
        with different settings.
        """,
    )
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    tr = TestRun(args.path)
    if args.runtime == "CReferenceEvaluator":
        from onnx_extended.reference import CReferenceEvaluator

        f_build = lambda proto: CReferenceEvaluator(proto)
        f_run = lambda rt, feeds: rt.run(None, feeds)
    elif args.runtime == "ReferenceEvaluator":
        from onnx.reference import ReferenceEvaluator

        f_build = lambda proto: ReferenceEvaluator(proto)
        f_run = lambda rt, feeds: rt.run(None, feeds)
    elif args.runtime == "onnxruntime":
        from onnxruntime import InferenceSession

        f_build = lambda proto: InferenceSession(
            proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        f_run = lambda rt, feeds: rt.run(None, feeds)
    elif args.runtime.startswith("CustomTreeEnsemble"):
        from onnx_extended.ortops.optim.cpu import get_ort_ext_libs
        from onnx_extended.ortops.optim.optimize import (
            change_onnx_operator_domain,
            get_node_attribute,
        )
        from onnxruntime import InferenceSession, SessionOptions

        spl = args.runtime.split(".")
        op_name = spl[0].replace("Custom", "")
        params = [int(a) for a in spl[1:]]
        assert len(params) <= 6, "Unexpected runtime {args.runtime!r}."

        optim_params = {}
        for i, k in enumerate(
            [
                "parallel_tree",
                "parallel_tree_N",
                "parallel_N",
                "batch_size_tree",
                "batch_size_rows",
                "use_node3",
            ]
        ):
            if i >= len(params):
                break
            optim_params[k] = params[i]

        def transform_model(onx, **kwargs):
            att = get_node_attribute(onx.graph.node[0], "nodes_modes")
            modes = ",".join([s.decode("ascii") for s in att.strings])
            return change_onnx_operator_domain(
                onx,
                op_type=op_name,
                op_domain="ai.onnx.ml",
                new_op_domain="onnx_extended.ortops.optim.cpu",
                nodes_modes=modes,
                **kwargs,
            )

        opts = SessionOptions()
        r = get_ort_ext_libs()
        opts.register_custom_ops_library(r[0])

        f_build = lambda proto, opts=opts: InferenceSession(
            transform_model(proto, **optim_params).SerializeToString(),
            opts,
            providers=["CPUExecutionProvider"],
        )
        f_run = lambda rt, feeds: rt.run(None, feeds)

    else:
        raise ValueError(f"Unexpected value {args.runtime!r} for runtime.")

    check = tr.test(
        exc=False,
        f_build=f_build,
        f_run=f_run,
        index=args.index,
    )
    bench = tr.bench(
        f_build=f_build,
        f_run=f_run,
        warmup=args.warmup,
        repeat=args.repeat,
        index=args.index,
    )
    output = {"test": check, "bench": bench, "runtime": args.runtime}
    js = json.dumps(output)
    print(js)


if __name__ == "__main__":
    main(sys.argv[1:])
