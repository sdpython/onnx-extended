
=============
command lines
=============

store
=====

Stores intermediate outputs on disk.
See :func:`store_intermediate_results <onnx_extended._command_lines.store_intermediate_results>`
or :class:`CReferenceEvaluator <onnx_extended.reference.CReferenceEvaluator>`.

.. runpython::

    from onnx_extended._command_lines_parser import get_parser_store
    get_parser_store().print_help()
