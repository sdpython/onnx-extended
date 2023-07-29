
=========
reference
=========

CReferenceEvaluator
===================

.. autoclass:: onnx_extended.reference.CReferenceEvaluator
    :members: input_names, output_names, opsets, run

Backend
=======

.. autofunction:: onnx_extended.reference.c_reference_backend.create_reference_backend

.. autoclass:: onnx_extended.reference.c_reference_backend.CReferenceEvaluatorBackend
    :members: 

.. autoclass:: onnx_extended.reference.c_reference_backend.CReferenceEvaluatorBackendRep
    :members: 

.. autoclass:: onnx_extended.reference.c_reference_backend.Runner
    :members: 

Tools
=====

.. autofunction:: onnx_extended.reference.c_reference_evaluator.from_array_extended

Operators
=========

ai.onnx
+++++++

.. autoclass:: onnx_extended.reference.c_ops.c_op_conv.Conv

ai.onnx.ml
++++++++++

.. autoclass:: onnx_extended.reference.c_ops.c_op_tree_ensemble_classifier.TreeEnsembleClassifier_1

.. autoclass:: onnx_extended.reference.c_ops.c_op_tree_ensemble_classifier.TreeEnsembleClassifier_3

.. autoclass:: onnx_extended.reference.c_ops.c_op_tree_ensemble_regressor.TreeEnsembleRegressor_1

.. autoclass:: onnx_extended.reference.c_ops.c_op_tree_ensemble_regressor.TreeEnsembleRegressor_3
