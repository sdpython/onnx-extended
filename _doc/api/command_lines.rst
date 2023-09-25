
.. _l-command-lines:

=============
command lines
=============

display
=======

Displays information from the shape inference on the standard output
and in a csv file.

.. runpython::

    from onnx_extended._command_lines_parser import get_parser_display
    get_parser_display().print_help()

.. autofunction:: onnx_extended._command_lines.display_intermediate_results

external
========

Split the model and the coefficients. The coefficients goes to an external file.

.. runpython::

    from onnx_extended._command_lines_parser import get_parser_external
    get_parser_external().print_help()

plot
====

Plots a graph like a profiling.

.. runpython::

    from onnx_extended._command_lines_parser import get_parser_plot
    get_parser_plot().print_help()

.. autofunction:: onnx_extended._command_lines.cmd_plot

print
=====

Prints a model or a tensor on the standard output.

.. runpython::

    from onnx_extended._command_lines_parser import get_parser_print
    get_parser_print().print_help()

.. autofunction:: onnx_extended._command_lines.print_proto

quantize
========

Prints a model or a tensor on the standard output.

.. runpython::

    from onnx_extended._command_lines_parser import get_parser_quantize
    get_parser_quantize().print_help()

Example::

    python3 -m onnx_extended quantize -i bertsquad-12.onnx -o bertsquad-12-fp8-1.onnx -v -v -k fp8 -q

.. autofunction:: onnx_extended._command_lines.cmd_quantize

select
======

Extracts a subpart of an existing model.

.. runpython::

    from onnx_extended._command_lines_parser import get_parser_select
    get_parser_select().print_help()

.. autofunction:: onnx_extended._command_lines.cmd_select

store
=====

Stores intermediate outputs on disk.
See also :class:`CReferenceEvaluator <onnx_extended.reference.CReferenceEvaluator>`.

.. runpython::

    from onnx_extended._command_lines_parser import get_parser_store
    get_parser_store().print_help()

.. autofunction:: onnx_extended._command_lines.store_intermediate_results
