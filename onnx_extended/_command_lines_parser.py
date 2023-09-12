import os
import sys
from typing import Any, Dict, List, Optional
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
        choices=[
            "store",
            "check",
            "display",
            "print",
            "quantize",
            "select",
            "external",
            "plot",
        ],
        help=dedent(
            """
        Select a command.
        
        'store' executes a model with class CReferenceEvaluator and stores every
        intermediate results on disk with a short onnx to execute the node.
        'check' checks a runtime on stored intermediate results.
        'display' displays the shapes inferences results,
        'print' prints out a model or a protobuf file on the standard output,
        'quantize' quantizes an onnx model in simple ways,
        'select' selects a subgraph inside a bigger models,
        'external' saves the coefficients in a different files for an onnx model,
        'plot' plots a graph like a profiling
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
        epilog="This is inspired from PR https://github.com/onnx/onnx/pull/5413. "
        "This command may disappear if this functionnality is not used.",
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
        Executes shape inference on an ONNX model and displays the inferred shapes.
        """
        ),
        epilog="This helps looking at a model from a terminal.",
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
        epilog="The command can be used on short models, mostly coming "
        "from unittests. Big models are far too large to make this command "
        "useful. Use command display instead.",
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


def get_parser_quantize() -> ArgumentParser:
    parser = ArgumentParser(
        prog="quantize",
        description=dedent(
            """
        Quantizes a model in simple ways.
        """
        ),
        epilog="The implementation quantization are mostly experimental. "
        "Once finalized, the functionality might move to another package.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="onnx model or protobuf file to print",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="output model to write",
    )
    parser.add_argument(
        "-k",
        "--kind",
        choices=["fp8", "fp16"],
        required=True,
        help="Kind of quantization to do. 'fp8' "
        "quantizes weights to float 8 e4m3fn whenever possible. "
        "It replaces MatMul by Transpose + DynamicQuantizeLinear + GemmFloat8. "
        "'fp16' casts all float weights to float 16, it does the same for "
        "inputs and outputs. It changes all operators Cast when they "
        "cast into float 32.",
    )
    parser.add_argument(
        "-s",
        "--scenario",
        choices=["onnxruntime", "onnx-extended"],
        required=False,
        default="onnxruntime",
        help="Possible versions for fp8 quantization. "
        "'onnxruntime' uses operators implemented by onnxruntime, "
        "'onnx-extended' uses experimental operators from this package.",
    )
    parser.add_argument(
        "-e",
        "--early-stop",
        required=False,
        help="stops after N modifications",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="do not stop if an exception is raised",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="enable logging, can be repeated",
    )
    parser.add_argument(
        "-t",
        "--transpose",
        default=2,
        help="which input(s) to transpose, 1 for the first one, "
        "2 for the second, 3 for both, 0 for None",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        type=str,
        help="to avoid quantizing nodes if their names belongs "
        "to that list (comma separated)",
    )
    return parser


def get_parser_select() -> ArgumentParser:
    parser = ArgumentParser(
        prog="select",
        description=dedent(
            """
        Selects a subpart of an onnx model.
        """
        ),
        epilog="The function removes the unused nodes.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="onnx model",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        required=True,
        help="saves into that file",
    )
    parser.add_argument(
        "-i",
        "--inputs",
        type=str,
        required=False,
        help="list of inputs to keep, comma separated values, "
        "leave empty to keep the model inputs",
    )
    parser.add_argument(
        "-o",
        "--outputs",
        type=str,
        required=False,
        help="list of outputs to keep, comma separated values, "
        "leave empty to keep the model outputs",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose, default is False",
    )
    parser.add_argument(
        "-t",
        "--transpose",
        type=int,
        required=False,
        default=1,
        help="which input to transpose before calling gemm: "
        "0 (none), 1 (first), 2 (second), 3 for both",
    )
    return parser


def get_parser_external() -> ArgumentParser:
    parser = ArgumentParser(
        prog="external",
        description=dedent(
            """
        Takes an onnx model and split the model and the coefficients.    
        """
        ),
        epilog="The functions stores the coefficients as external data. "
        "It calls the function convert_model_to_external_data.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="onnx model",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        required=True,
        help="saves into that file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="display sizes",
    )
    return parser


def _cmd_store(argv):
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


def _cmd_display(argv):
    from ._command_lines import display_intermediate_results

    parser = get_parser_display()
    args = parser.parse_args(argv[1:])
    display_intermediate_results(model=args.model, save=args.save, tab=args.tab)


def _cmd_print(argv):
    from ._command_lines import print_proto

    parser = get_parser_print()
    args = parser.parse_args(argv[1:])
    print_proto(proto=args.input, fmt=args.format)


def _process_exceptions(text: Optional[str]) -> List[Dict[str, str]]:
    if text is None:
        return []
    names = text.split(",")
    return [dict(name=n) for n in names]


def _cmd_quantize(argv):
    from ._command_lines import cmd_quantize

    parser = get_parser_quantize()
    args = parser.parse_args(argv[1:])
    processed_exceptions = _process_exceptions(args.exclude)
    cmd_quantize(
        model=args.input,
        output=args.output,
        verbose=args.verbose,
        scenario=args.scenario,
        kind=args.kind,
        early_stop=args.early_stop,
        quiet=args.quiet,
        index_transpose=args.transpose,
        exceptions=processed_exceptions,
    )


def _cmd_select(argv):
    from ._command_lines import cmd_select

    parser = get_parser_select()
    args = parser.parse_args(argv[1:])
    cmd_select(
        model=args.model,
        save=args.save,
        inputs=args.inputs,
        outputs=args.outputs,
        verbose=args.verbose,
    )


def _cmd_external(argv):
    from onnx import load
    from onnx.external_data_helper import (
        convert_model_to_external_data,
        write_external_data_tensors,
    )

    parser = get_parser_external()
    args = parser.parse_args(argv[1:])
    model = args.model
    save = args.save

    if args.verbose:
        size = os.stat(model).st_size
        print(f"Load model {model!r}, size is {size / 2 ** 10:1.3f} kb")
    with open(model, "rb") as f:
        proto = load(f)

    if args.verbose:
        print("convert_model_to_external_data")
    convert_model_to_external_data(
        proto,
        all_tensors_to_one_file=True,
        location=os.path.split(save)[-1] + ".data",
        convert_attribute=True,
        size_threshold=1024,
    )

    dirname = os.path.dirname(save)
    proto = write_external_data_tensors(proto, dirname)
    with open(save, "wb") as f:
        f.write(proto.SerializeToString())
    if args.verbose:
        size = os.stat(save).st_size
        print(f"Saved model {save!r}, size is {size / 2 ** 10:1.3f} kb")


def get_parser_plot() -> ArgumentParser:
    parser = ArgumentParser(
        prog="plot",
        description=dedent(
            """
        Plots a graph reprsenting the data loaded from a filename.
        """
        ),
        epilog="Plots a graph",
    )
    parser.add_argument(
        "-k",
        "--kind",
        choices=[
            "profile_op",
            "profile_node",
        ],
        help=dedent(
            """
        Kind of plot to draw.
        
        'profile_op' shows the time spent in every kernel per operator type,
        'profile_node' shows the time spent in every kernel per operator node,
        """
        ),
        required=True,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="input file",
    )
    parser.add_argument(
        "-c",
        "--ocsv",
        type=str,
        required=False,
        help="saves the data used to plot the graph as csv file",
    )
    parser.add_argument(
        "-o",
        "--opng",
        type=str,
        required=False,
        help="saves the plot as png into that file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="display sizes",
    )
    parser.add_argument(
        "-w",
        "--with-shape",
        action="store_true",
        help="keep the shape before aggregating the results",
    )
    parser.add_argument(
        "-t",
        "--title",
        required=False,
        help="plot title",
    )
    return parser


def _cmd_plot(argv):
    from ._command_lines import cmd_plot

    parser = get_parser_plot()
    args = parser.parse_args(argv[1:])
    cmd_plot(
        kind=args.kind,
        filename=args.input,
        out_csv=args.ocsv,
        out_png=args.opng,
        title=args.title,
        verbose=args.verbose,
        with_shape=args.with_shape,
    )


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
                quantize=get_parser_quantize,
                select=get_parser_select,
                external=get_parser_external,
                plot=get_parser_plot,
            )
            cmd = argv[0]
            if cmd not in parsers:
                raise ValueError(
                    f"Unknown command {cmd!r}, it should be in {list(sorted(parsers))}."
                )
            parser = parsers[cmd]()
            parser.parse_args(argv[1:])
        raise RuntimeError("The programme should have exited before.")

    cmd = argv[0]
    fcts = dict(
        store=_cmd_store,
        display=_cmd_display,
        print=_cmd_print,
        quantize=_cmd_quantize,
        select=_cmd_select,
        external=_cmd_external,
        plot=_cmd_plot,
    )
    if cmd in fcts:
        fcts[cmd](argv)
    else:
        raise ValueError(
            f"Unknown command {cmd!r}, use --help to get the list of known command."
        )
