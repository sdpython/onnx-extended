Change Logs
===========

0.3.0
+++++

* :pr:`185`: adds custom operator MulMulSigmoid on CUDA
* :pr:`184`: use onnxruntime==1.18.0 as default
* :pr:`181`: adds MaskedScatterNDOfShape custom operator
* :pr:`175`: adds custom operator MulSub and SubMul on CUDA
* :pr:`173`: adds custom operator AddSharedInput, MulSharedInput on CUDA
* :pr:`170`: adds custom operator TriMatrix on CUDA
* :pr:`169`: adds custom operator ReplaceZero on CUDA
* :pr:`168`: adds custom operator MulSigmoid on CUDA
* :pr:`167`: adds custom operator Rotary on CUDA
* :pr:`166`, :pr:`178`: adds custom operators AddMul, MulAdd on CUDA
* :pr:`165`: adds custom operators AddAddAdd, MulMulMul on CUDA
* :pr:`163`: use onnxruntime==1.17.3 as default
* :pr:`162`: add ScatterNDOfShape implementation on CUDA without atomics
* :pr:`159`: add AddAdd custom operator on CUDA
* :pr:`158`: add MulMul custom operator on CUDA
* :pr:`157`: add ScatterNDOfShape custom operator
* :pr:`155`: add a function to draw a timeline from a profile
* :pr:`154`: improves ploting legend for profiling
* :pr:`151`: refactoring of TreeEnsemble code to make them faster
* :pr:`129`, :pr:`132`: support sparse features for TreeEnsemble

0.2.4
+++++

* :pr:`120`: use onnxruntime==1.16.3 as default
* :pr:`115`, :pr:`116`, :pr:`118`: adds C implementation of SVMRegressor, SVMClassifier
  reference operator based on it, and custom kernels for onnxruntime as well
* :pr:`111`, :pr:`117`, :pr:`119`: adds C implementation of TfIdfVectorizer +
  python implementation of Tokenizer + custom kernel for onnxruntime
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
