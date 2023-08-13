Change Logs
===========

0.2.0
+++++

* :pr:`59`: add local functions to quantize
* :pr:`57`: add C implementation for DynamicQuantizeLinear (for experimentation)
* :pr:`56`: add C implementation to cast a float into float 8
* :pr:`55`, :pr:`58`: add basic functionality to transform a graph, starts with basic quantization
* :pr:`51`: fix optmized TreeEnsembleRegressor and adds TreeEnsembleClassifier as custom ops
* :pr:`50`: add command line store to store intermediate outputs
* :pr:`49`: add option to save intermediate results in CReferenceEvaluator
* :pr:`45`: add option cuda-link to setup.py to specify how to link with CUDA library
* :pr:`41`: implements a custom kernel for RandomForestRegressor easier to optimize
* :pr:`34`: update to onnxruntime v1.15.1
* :pr:`31`: implement a custom CUDA kernel (gemm)
* :pr:`32`: update to onnxruntime v1.15.0
* :pr:`27`: add a custom kernel with parameters to onnxruntime
* :pr:`26`: add a custom kernel to onnxruntime
* :pr:`24`: use Eigen to implement Conv operator
* :pr:`23`: make `pip wheel .` work
* :pr:`22`: rename cmake into _cmake to avoid warnings related to cmake package
* :pr:`19`: minimal settings to use onnxruntime
* :pr:`14`: minimal setting to use CUDA
* :pr:`8`: support for C++ unit test
