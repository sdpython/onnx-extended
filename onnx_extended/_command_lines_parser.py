import sys
from argparse import ArgumentParser
from textwrap import dedent


def get_main_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="onnx-extended",
        description="onnx-extended main command line.",
        epilog="Type 'onnx-extended <cmd> --help' to get help for a specific command.",
    )
    parser.add_argument(
        "cmd",
        choices=["store", "check"],
        help=dedent(
            """
        Select a command.
        
        'store' executes a model with class CReferenceEvaluator and stores every
        intermediate results on disk with a short onnx to execute the node.
        'check' checks a runtime on stored intermediate results.
        """
        ),
    )
    return parser


def get_parser_store() -> ArgumentParser:
    parser = ArgumentParser(
        prog="onnx-extended",
        description=dedent(
            """
        Executes a model with class CReferenceEvaluator and stores every
        intermediate results on disk with a short onnx to execute the node.
        """
        ),
        epilog="Type 'onnx-extended <cmd> --help' to get help for a specific command.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="onnx model to test",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=".",
        help="path where to store the outputs, default is '.'",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="input, it can be a path, or a string like "
        "'float32(4,5)' to generate a random input of this shape, "
        "'rnd' works as well if the model precisely defines the inputs",
        nargs="?",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose, default is False",
    )
    parser.add_argument(
        "-r",
        "--runtime",
        choices=["CReferenceEvaluator"],
        default="CReferenceEvaluator",
        help="Runtime to use to generate the intermediate results, "
        "default is 'CReferenceEvaluator'",
    )
    parser.add_argument(
        "-p",
        "--providers",
        default="CPU",
        help="Execution providers, multiple values can separated with a comma",
    )
    return parser


def main():
    argv = sys.argv
    if len(argv) <= 1 or argv[-1] == "--help":
        if len(argv) < 3:
            parser = get_main_parser()
            parser.parse_args()
        else:
            parser = get_parser_store()
            parser.parse_args(argv[1:])
        raise RuntimeError("The programme should have exited before.")

    cmd = argv[1]
    if cmd == "store":
        from ._command_lines import store_intermediate_results

        parser = get_parser_store()
        args = parser.parse_args(sys.argv[1:])
        store_intermediate_results(
            model=args.model,
            runtime=args.runtime,
            verbose=args.verbose,
            inputs=args.input,
            out=args.out,
            providers=args.providers,
        )
    else:
        raise ValueError(
            f"Unknown command {cmd!r}, use --help to get the list of known command."
        )
