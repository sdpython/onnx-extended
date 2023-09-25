
Onnx Manipulations
==================

Extract a subgraph
++++++++++++++++++

Both functions below are usually to extract a small piece of an existing
model to create unit tests.

Function :func:`onnx_remove_node_unused
<onnx_extended.tools.onnx_manipulations.onnx_remove_node_unused>`
removes every node whose outputs are not used.

Function :func:`select_model_inputs_outputs
<onnx_extended.tools.onnx_manipulations.select_model_inputs_outputs>`
creates an onnx graph taking any intermediate results as new inputs
or new outputs.

Analyze
+++++++

Loops or tests are based on onnx `GraphProto`. These
subgraphs takes inputs but can also use any intermediated
results computed so far. These results are part of the local
context but they are not explicit mentioned and that sometimes
makes it difficult to understand what subgraph is doing or needs.
Function :func:`get_hidden_inputs
<onnx_extended.tools.onnx_manipulations.get_hidden_inputs>`
retrieves that information.

Function :func:`enumerate_onnx_node_types
<onnx_extended.tools.onnx_tools.enumerate_onnx_node_types>`
quickly gives the list of operators a model uses.
