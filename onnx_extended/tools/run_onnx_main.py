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
        choices=["CReferenceEvaluator", "ReferenceEvaluator", "onnxruntime"],
        default="ReferenceEvaluator",
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
            proto, providers=["CPUExecutionProvider"]
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
    output = {"test": check, "bench": bench}
    js = json.dumps(output)
    print(js)


if __name__ == "__main__":
    main(sys.argv[1:])
