
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

store
=====

Stores intermediate outputs on disk.
See also :class:`CReferenceEvaluator <onnx_extended.reference.CReferenceEvaluator>`.

.. runpython::

    from onnx_extended._command_lines_parser import get_parser_store
    get_parser_store().print_help()

.. autofunction:: onnx_extended._command_lines.store_intermediate_results
