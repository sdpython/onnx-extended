Change Logs
===========

0.2.4
+++++

* :pr:`111`: adds C implementation of TfIdfVectorizer + python implementation of Tokenizer
* :pr:`110`: allows LEQ as an alias for BRANCH_LEQ for nodes_modes in TreeEnsemble* operators
* :pr:`108`: improves command lines documentation, fix an issue in command line stats
* :pr:`103`: add methods to compute statistics on TreeEnsemble and initializers

0.2.3
+++++

* :pr:`99`: use onnxruntime==1.16.1 as default
* :pr:`96`: implements a fonction to convert a ModelProto into string (not bytes),
  add a function to multiply the number of trees in a TreeEnsemble
* :pr:`75`: add an implementation of murmurhash3 to validate some options
* :pr:`93`: validates the wheels in CI
* :pr:`89`: add a function to merge models and update them if both have different opsets

0.2.2
+++++

* :pr:`87`: update the quantization tools to use a simplified dynamic linear quantization into float 8
* :pr:`85`: add load_model, save_model to help saving with/without external data
* :pr:`82`: fixes benchmark on multiple versions of onnxruntime

0.2.1
+++++

* :pr:`79`: update to onnxruntime v1.16.0
* :pr:`77`: helpers to benchmark a model
* :pr:`74`: add a function to enumerate all intermediate results with onnxruntime
* :pr:`71`, :pr:`72`, :pr:`73`: add function to analyse a profile produce by onnxruntime
* :pr:`68`, :pr:`69`, :pr:`70`: add CPU implementation for CustomGemmFloat8
* :pr:`67`: add a function to extract a subgraph of a model
* :pr:`59`, :pr:`60`, :pr:`61`, :pr:`62`, :pr:`63`, :pr:`65`,
  :pr:`66`, :pr:`68`, :pr:`69`, :pr:`70`:
  add local functions to quantize into float 8, float 16
* :pr:`57`: add C implementation for DynamicQuantizeLinear (for experimentation)
* :pr:`56`: add C implementation to cast a float into float 8
* :pr:`55`, :pr:`58`: add basic functionality to transform a graph, starts with basic quantization
* :pr:`51`: fix optimized TreeEnsembleRegressor and adds TreeEnsembleClassifier as custom ops
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
