import sys
from typing import Any, List, Optional
from argparse import ArgumentParser
from textwrap import dedent


def get_main_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="onnx-extended",
        description="onnx-extended main command line.",
        epilog="Type 'python -m onnx_extended <cmd> --help' "
        "to get help for a specific command.",
    )
    parser.add_argument(
        "cmd",
        choices=["store", "check", "display", "print"],
        help=dedent(
            """
        Select a command.
        
        'store' executes a model with class CReferenceEvaluator and stores every
        intermediate results on disk with a short onnx to execute the node.
        'check' checks a runtime on stored intermediate results.
        'display' displays the shapes inferences results,
        'print' prints out a model or a protobuf file on the standard output
        """
        ),
    )
    return parser


def get_parser_store() -> ArgumentParser:
    parser = ArgumentParser(
        prog="store",
        description=dedent(
            """
        Executes a model with class CReferenceEvaluator and stores every
        intermediate results on disk with a short onnx to execute the node.
        """
        ),
        epilog="Type 'python -m onnx_extended store --help' "
        "to get help for this command.",
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
        action="append",
        help="input, it can be a path, or a string like "
        "'float32(4,5)' to generate a random input of this shape, "
        "'rnd' works as well if the model precisely defines the inputs, "
        "'float32(4,5):U10' generates a tensor with a uniform law",
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


def get_parser_display() -> ArgumentParser:
    parser = ArgumentParser(
        prog="display",
        description=dedent(
            """
        Executes shape inference on an ONNX model and display the inferred shape.
        """
        ),
        epilog="Type 'python -m onnx_extended display --help' "
        "to get help for this command.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="onnx model to display",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        required=False,
        help="saved the data as a dataframe",
    )
    parser.add_argument(
        "-t",
        "--tab",
        type=int,
        required=False,
        default=12,
        help="column size when printed on standard output",
    )
    return parser


def get_parser_print() -> ArgumentParser:
    parser = ArgumentParser(
        prog="print",
        description=dedent(
            """
        Shows an onnx model or a protobuf string on stdout.
        Extension '.onnx' is considered a model,
        extension '.proto' or '.pb' is a protobuf string.
        """
        ),
        epilog="Type 'python -m onnx_extended print --help' "
        "to get help for this command.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="onnx model or protobuf file to print",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["raw", "nodes"],
        default="raw",
        help="format ot use to display the graph",
    )
    return parser


def main(argv: Optional[List[Any]] = None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) <= 1 or argv[-1] in ("--help", "-h"):
        if len(argv) < 2:
            parser = get_main_parser()
            parser.parse_args(argv)
        else:
            parsers = dict(
                store=get_parser_store,
                print=get_parser_print,
                display=get_parser_display,
            )
            cmd = argv[0]
            if cmd not in parsers:
                raise ValueError(
                    f"Unknown command {cmd!r}, it should be in {list(sorted(parsers))}."
                )
            parser = parsers[cmd]()
            parser.parse_args(argv[:1])
        raise RuntimeError("The programme should have exited before.")

    cmd = argv[0]
    if cmd == "store":
        from ._command_lines import store_intermediate_results

        parser = get_parser_store()
        args = parser.parse_args(argv[1:])
        store_intermediate_results(
            model=args.model,
            runtime=args.runtime,
            verbose=args.verbose,
            inputs=args.input,
            out=args.out,
            providers=args.providers,
        )
    elif cmd == "display":
        from ._command_lines import display_intermediate_results

        parser = get_parser_display()
        args = parser.parse_args(argv[1:])
        display_intermediate_results(model=args.model, save=args.save, tab=args.tab)
    elif cmd == "print":
        from ._command_lines import print_proto

        parser = get_parser_print()
        args = parser.parse_args(argv[1:])
        print_proto(proto=args.input, fmt=args.format)
    else:
        raise ValueError(
            f"Unknown command {cmd!r}, use --help to get the list of known command."
        )
