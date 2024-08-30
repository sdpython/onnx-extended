from itertools import permutations
from typing import Any, Dict, Iterator, List, Optional, Tuple
import time
import math
import numpy
from onnx import helper, ModelProto, TensorProto
from ...reference import to_array_extended
from ..stats_nodes import extract_attributes
from .einsum_config import DEFAULT_OPSET, DEFAULT_IR_VERSION, guess_proto_dtype
from .einsum_impl import decompose_einsum_equation, apply_einsum_sequence
from .einsum_ml import predict_transposition_cost


_einsum_cache: Dict[int, Any] = {}


class OnnxMicroRuntime:
    """
    Implements a micro runtime for ONNX graphs.
    It does not implements all the operator types.

    :param model_onnx: ONNX model
    """

    def __init__(self, model_onnx):
        assert hasattr(
            model_onnx, "graph"
        ), f"model_onnx is not an ONNX graph but {type(model_onnx)!r}."
        self.model_onnx = model_onnx

    @property
    def input_names(self):
        "Returns input names."
        return [i.name for i in self.model_onnx.graph.input]

    @property
    def output_names(self):
        "Returns output names."
        return [i.name for i in self.model_onnx.graph.output]

    def run(
        self, unused: Optional[List[str]], inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Computes the outputs of the graph.

        :param unused: unused (the list of desired outputs)
        :param inputs: dictionary
        :return: all intermediates results and output as a dictionary
        """
        assert isinstance(
            inputs, dict
        ), f"inputs must be a dictionary not {type(inputs)!r}."
        results = inputs.copy()

        for init in self.model_onnx.graph.initializer:
            name = init.name
            mat = to_array_extended(init)
            results[name] = mat

        for node in self.model_onnx.graph.node:
            op_type = node.op_type
            inp = [results[n] for n in node.input]
            meth_name = f"_op_{op_type.lower()}"
            if not hasattr(self, meth_name):
                raise NotImplementedError(
                    f"OnnxMicroRuntime does not implement operator {op_type!r}."
                )
            kwargs = {k: v[1] for k, v in extract_attributes(node).items()}
            out = getattr(self, meth_name)(*inp, **kwargs)
            for n, o in zip(node.output, out):
                results[n] = o

        return results

    ########################
    # Runtime for operators
    ########################

    def _op_abs(self, x):
        "Runtime for operator :epkg:`Op:Abs`."
        return (numpy.abs(x),)

    def _op_add(self, x, y):
        "Runtime for operator :epkg:`Op:Add`."
        return (x + y,)

    def _op_concat(self, *args, axis=None):
        "Runtime for operator :epkg:`Op:Concat`."

        def _preprocess(a, axis):
            if axis >= len(a.shape):
                new_shape = a.shape + (1,) * (axis + 1 - len(a.shape))
                return a.reshape(new_shape)
            return a

        targs = tuple(_preprocess(a, axis) for a in args)
        return (numpy.concatenate(targs, axis),)

    def _op_gemm(self, a, b, c=None, alpha=None, beta=None, transA=False, transB=False):
        "Runtime for operator :epkg:`Op:Gemm`."

        def _gemm00(a, b, c, alpha, beta):
            o = numpy.dot(a, b) * alpha
            if beta != 0:
                o += c * beta
            return o

        def _gemm01(a, b, c, alpha, beta):
            o = numpy.dot(a, b.T) * alpha
            if beta != 0:
                o += c * beta
            return o

        def _gemm10(a, b, c, alpha, beta):
            o = numpy.dot(a.T, b) * alpha
            if beta != 0:
                o += c * beta
            return o

        def _gemm11(a, b, c, alpha, beta):
            o = numpy.dot(a.T, b.T) * alpha
            if beta != 0:
                o += c * beta
            return o

        assert isinstance(
            transA, (int, bool, numpy.int64)
        ), f"Unexpected type for transA: {type(transA)!r}."
        assert isinstance(
            transB, (int, bool, numpy.int64)
        ), f"Unexpected type for transA: {type(transB)!r}."
        if transA:
            fct = _gemm11 if transB else _gemm10
        else:
            fct = _gemm01 if transB else _gemm00
        return (fct(a, b, c, alpha=alpha, beta=beta),)

    def _op_gather(self, x, indices, axis=None):
        "Runtime for operator :epkg:`Op:Gather`."
        if not x.flags["C_CONTIGUOUS"]:
            x = numpy.ascontiguousarray(x)
        if not indices.flags["C_CONTIGUOUS"]:
            indices = indices.ascontiguousarray()
        return (numpy.take(x, indices, axis=axis),)

    def _op_identity(self, x):
        "Runtime for operator :epkg:`Op:Identity`."
        return (x,)

    def _op_matmul(self, x, y):
        "Runtime for operator :epkg:`Op:MatMul`."
        return (numpy.matmul(x, y),)

    def _op_max(self, *inps):
        "Runtime for operator :epkg:`Op:Max`."
        return (numpy.maximum(*inps),)

    def _op_mul(self, x, y):
        "Runtime for operator :epkg:`Op:Mul`."
        return (x * y,)

    def _op_reduceprod(self, data, axes=None, keepdims=None):
        "Runtime for operator :epkg:`Op:ReduceProd`."
        if axes is not None and not isinstance(axes, int):
            if isinstance(axes, numpy.ndarray) and len(axes.shape) == 0:
                axes = int(axes)
            else:
                axes = tuple(axes) if len(axes) > 0 else None
        return (numpy.prod(data, axis=axes, keepdims=keepdims, dtype=data.dtype),)

    def _op_reducesum(self, data, axes, keepdims=None, noop_with_empty_axes=None):
        "Runtime for operator :epkg:`Op:ReduceSum`."
        if axes is None and noop_with_empty_axes:
            return (data,)
        if axes is not None and not isinstance(axes, int):
            if isinstance(axes, numpy.ndarray) and len(axes.shape) == 0:
                axes = int(axes)
            else:
                axes = tuple(axes) if len(axes) > 0 else None
        return (numpy.sum(data, axis=axes, keepdims=keepdims, dtype=data.dtype),)

    def _op_reshape(self, x, shape):
        "Runtime for operator :epkg:`Op:Reshape`."
        return (x.reshape(shape),)

    def _op_shape(self, x):
        "Runtime for operator :epkg:`Op:Shape`."
        return (numpy.array(list(x.shape), dtype=numpy.int64),)

    def _op_squeeze(self, x, axes=None):
        "Runtime for operator :epkg:`Op:Squeeze`."
        if axes is None:
            return (x,)
        if hasattr(axes, "__iter__"):
            return (numpy.squeeze(x, axis=tuple(axes)),)
        return (numpy.squeeze(x, axis=axes),)

    def _op_transpose(self, x, perm=None):
        "Runtime for operator :epkg:`Op:Transpose`."
        return (numpy.transpose(x, perm),)

    def _op_unsqueeze(self, x, axes=None):
        "Runtime for operator :epkg:`Op:Unsqueeze`."
        if axes is None:
            return (x,)
        if hasattr(axes, "__iter__"):
            return (numpy.expand_dims(x, axis=tuple(axes)),)
        return (numpy.expand_dims(x, axis=axes),)


def enumerate_cached_einsum() -> Iterator[Tuple[int, Any]]:
    """
    Enumerates all cached einsum function.
    """
    global _einsum_cache
    yield from _einsum_cache.items()


class CachedEinsum:
    """
    Stores all the necessary information to cache the preprocessing
    of a an einsum equation.

    :param equation: numpy equation
    :param runtime: see :func:`einsum
        <onnx_extended.tools.einsum.einsum_fct.einsum>`
    :param opset: ONNX opset
    :param optimize: finds the best letter permutation
    :param dtype: dtype
    :param decompose: to decompose Einsum operator or to keep it as is
    :param key: key used to cache this class
    :param strategy: optimization strategy
    :param verbose: displays progress information

    The class creates the following attributes:

    * `equation_` corresponding to the best equivalent equation
    * `graph_`: the corresponding graph returned by function
        :func:`decompose_einsum_equation
        <onnx_extended.tools.einsum.einsum_impl.decompose_einsum_equation>`
    * `onnx_`: if a conversion to onnx is used, stores the onnx graph
    * `runtime_`: a function used by `__call__`, calls the runtime
    """

    def __init__(
        self,
        equation: str,
        runtime: str = "batch_dot",
        opset: Optional[int] = None,
        optimize: bool = False,
        dtype: Any = numpy.float64,
        decompose: bool = True,
        strategy: Optional[str] = None,
        verbose: Optional[bool] = None,
        key: Optional[int] = None,
    ):
        self.equation = equation
        self.runtime = runtime
        self.opset = opset
        self.optimize = optimize
        self.dtype = dtype
        self.decompose = decompose
        self.strategy = strategy
        self.verbose = verbose
        self.key = key

    def __repr__(self):
        "usual"
        return "%s(%r, %r, %r, %r, %r, %r, %r, key=%r)" % (
            self.__class__.__name__,
            self.equation,
            self.runtime,
            self.opset,
            self.optimize,
            self.dtype,
            self.decompose,
            self.strategy,
            self.key,
        )

    def default_inputs(self, N: Optional[int] = None) -> List[numpy.ndarray]:
        """
        Returns default inputs (reshaped numpy.arange + 0.7i).

        :param N: dimension (all dimension have the same size)

        If *N is None*, N is given a size depending on the number of letters
        to avoid spending too much time on optimization.
        """
        if N is None:
            letters = set(
                c for c in self.equation if "a" <= c <= "z" or "A" <= c <= "Z"
            )
            nn = math.factorial(len(letters))
            N = max(int(2**11 / nn), 4)
            N = min(N, 15)
        inps = self.equation.split("->")[0].split(",")
        lens = [len(s) for s in inps]
        inputs = [numpy.arange(N**d).reshape((N,) * d) for d in lens]
        inputs = [(i + 0.7 * ii).astype(self.dtype) for ii, i in enumerate(inputs)]
        return inputs

    def build(self):
        """
        Preprocesses the equation builds whatever is necessary
        to compute the result of the einsum equation.
        """
        if not self.optimize and not hasattr(self, "equation_"):
            self.equation_ = self.equation
        elif self.strategy is None:
            self.equation_ = self._build_optimize()
        elif self.strategy == "ml":
            self.equation_ = self._build_optimize_ml()
        else:
            raise ValueError(f"Unknown strategy {self.strategy!r}.")
        self.build_runtime()

    def _build_optimize(self) -> str:
        # loops over all permutations
        assert (
            self.equation.lower() == self.equation
        ), f"Only lower equation can be optimized, {self.equation!r} is not."
        letters = list(sorted(set(c for c in self.equation if "a" <= c <= "z")))
        possible = list(permutations(letters))
        possible.insert(0, letters)
        if self.verbose:
            from tqdm import tqdm

            subset = tqdm(possible)
        else:
            subset = possible
        best: List[Tuple[float, str]] = []
        confs = []
        very_best = None
        inputs = None
        for perm in subset:
            replace = {d: c for c, d in zip(letters, perm)}
            eq = self.equation
            for k, v in replace.items():
                eq = eq.replace(k, v.upper())
            eq = eq.lower()
            inst = CachedEinsum(
                eq,
                runtime=self.runtime,
                opset=self.opset,
                optimize=False,
                dtype=self.dtype,
                decompose=self.decompose,
            )
            inst.build()
            if inputs is None:
                inputs = inst.default_inputs()
                inst(*inputs)
            ts = time.perf_counter()
            for _ in range(10):
                inst(*inputs)
            delta = time.perf_counter() - ts
            confs.append((delta, eq))
            if len(best) < 10:
                best.append((delta, eq))
                best.sort()
            elif delta < best[-1][0]:
                best[-1] = (delta, eq)
                best.sort()
            if self.verbose and (very_best is None or very_best != best[0][0]):
                very_best = best[0][0]
                subset.set_description("%1.2g rtbest=%r" % best[0])
        self.optimized_ = best
        self.timed_permutations_ = confs
        return best[0][1]

    def _build_optimize_ml(self) -> str:
        # loops over all permutations
        assert (
            self.equation.lower() == self.equation
        ), f"Only lower equation can be optimized, {self.equation!r} is not."
        letters = list(sorted(set(c for c in self.equation if "a" <= c <= "z")))
        possible = list(permutations(letters))
        possible.insert(0, letters)
        if self.verbose:
            from tqdm import tqdm

            subset = tqdm(possible)
        else:
            subset = possible
        best: List[Tuple[float, str]] = []
        confs = []
        very_best = None
        inputs = None
        for perm in subset:
            replace = {d: c for c, d in zip(letters, perm)}
            eq = self.equation
            for k, v in replace.items():
                eq = eq.replace(k, v.upper())
            eq = eq.lower()
            inst = CachedEinsum(
                eq,
                runtime=self.runtime,
                opset=self.opset,
                optimize=False,
                dtype=self.dtype,
                decompose=self.decompose,
            )
            inst.build()
            if inputs is None:
                inputs = inst.default_inputs()
            if hasattr(inst, "onnx_"):
                onx = inst.onnx_
            else:
                inits = [
                    ("X%d" % i, (TensorProto.FLOAT, inputs[i].shape))
                    for i in range(len(inputs))
                ]
                onx = inst.graph_.to_onnx("Y", *inits, opset=self.opset)

            rt = OnnxMicroRuntime(onx)
            dict_inputs = {"X%d" % i: inp for i, inp in enumerate(inputs)}
            out = rt.run(None, dict_inputs)

            transposes = []
            for node in onx.graph.node:
                if node.op_type == "Transpose":
                    shape = [(d * 10 if d > 1 else d) for d in out[node.input[0]].shape]
                    transposes.append([shape, list(node.attribute[0].ints)])

            delta = sum(max(0, predict_transposition_cost(*v)) for v in transposes)

            confs.append((delta, eq))
            if len(best) < 10:
                best.append((delta, eq))
                best.sort()
            elif delta < best[-1][0]:
                best[-1] = (delta, eq)
                best.sort()
            if self.verbose and (very_best is None or very_best != best[0][0]):
                very_best = best[0][0]
                subset.set_description("%1.2g mlbest=%r" % best[0])
        self.optimized_ = best
        self.timed_permutations_ = confs
        return best[0][1]

    def build_onnx_einsum(self, input_names: List[str]) -> ModelProto:
        """
        Builds an ONNX graph with a single einsum operator.
        """
        opset = self.opset if self.opset is not None else DEFAULT_OPSET
        ir_version = DEFAULT_IR_VERSION
        proto_type = guess_proto_dtype(
            numpy.float32 if self.dtype is None else self.dtype
        )

        model = helper.make_model(
            opset_imports=[helper.make_operatorsetid("", opset)],
            ir_version=ir_version,
            producer_name="onnx_extended",
            producer_version="0.0.1",
            graph=helper.make_graph(
                name="einsum",
                inputs=[
                    helper.make_tensor_value_info(n, proto_type, None)
                    for n in input_names
                ],
                outputs=[helper.make_tensor_value_info("Y", proto_type, None)],
                nodes=[
                    helper.make_node(
                        "Einsum", input_names, ["Y"], equation=self.equation_
                    )
                ],
            ),
        )
        return model

    def build_runtime(self):
        """
        Builds the runtime associated to the
        equation `self.equation_`.
        """
        if self.runtime == "python":
            from ...reference import CReferenceEvaluator

            cls = CReferenceEvaluator
        elif self.runtime == "onnxruntime":
            from onnxruntime import InferenceSession

            cls = lambda obj: InferenceSession(  # noqa: E731
                obj, providers=["CPUExecutionProvider"]
            )
        elif self.runtime == "batch_dot":
            cls = None
        else:
            raise TypeError(f"Unexpected runtime {self.runtime!r}.")

        if self.decompose:
            self.graph_ = decompose_einsum_equation(
                self.equation_, strategy="numpy", clean=True
            )
            if self.runtime == "batch_dot":
                self.runtime_ = lambda *inputs: apply_einsum_sequence(
                    self.graph_, *inputs
                )
            else:
                n_inputs = len(self.graph_.metadata["lengths"]) - 1
                input_names = ["X%d" % i for i in range(n_inputs)]
                self.onnx_names_ = input_names
                onx = self.graph_.to_onnx(
                    "Y", *input_names, opset=self.opset, dtype=self.dtype
                )
                self.onnx_ = onx
                self.oinf_ = cls(self.onnx_.SerializeToString())
                self.runtime_ = lambda *inputs: self.oinf_.run(
                    None, dict(zip(self.onnx_names_, inputs))
                )[0]
        else:
            n_inputs = len(self.equation.split("->")[0].split(","))
            input_names = ["X%d" % i for i in range(n_inputs)]
            self.onnx_ = self.build_onnx_einsum(input_names)
            self.onnx_names_ = input_names
            self.oinf_ = cls(self.onnx_.SerializeToString())
            self.runtime_ = lambda *inputs: self.oinf_.run(
                None, dict(zip(self.onnx_names_, inputs))
            )[0]

    def __call__(self, *inputs: List[numpy.ndarray]) -> List[numpy.ndarray]:
        """
        Calls the runtime `self.runtime_`.
        """
        assert hasattr(self, "runtime_"), "Method 'build_runtime' was not called."
        return self.runtime_(*inputs)

    @staticmethod
    def build_einsum(
        equation: str,
        runtime: str,
        opset: int,
        optimize: bool,
        dtype: Any,
        decompose: bool = True,
        strategy: Optional[str] = None,
        verbose: Optional[bool] = None,
        key: Optional[int] = None,
    ) -> "CachedEinsum":
        """
        Creates an instance of *CachedEinsum*.
        """
        inst = CachedEinsum(
            equation,
            runtime=runtime,
            opset=opset,
            optimize=optimize,
            dtype=dtype,
            decompose=decompose,
            strategy=strategy,
            verbose=verbose,
            key=key,
        )
        inst.build()
        return inst


def _einsum(
    equation: str,
    dtype: Any,
    optimize: bool = False,
    runtime: str = "batch_dot",
    cache: bool = True,
    opset: Optional[int] = None,
    decompose: bool = True,
    strategy: Optional[str] = None,
    verbose: Optional[bool] = None,
) -> CachedEinsum:
    global _einsum_cache
    cached = None
    if cache:
        key = equation, runtime, opset, optimize, dtype, decompose, strategy
        cached = _einsum_cache.get(key, None)
    else:
        key = None
    if cached is None:
        cached = CachedEinsum.build_einsum(
            equation,
            runtime,
            opset,
            optimize,
            dtype,
            decompose=decompose,
            strategy=strategy,
            verbose=verbose,
            key=key,
        )
    else:
        cache = False
    if cache:
        _einsum_cache[key] = cached
    return cached


def optimize_decompose_einsum_equation(
    equation: str,
    dtype: Any,
    optimize: bool = False,
    runtime: str = "batch_dot",
    cache: bool = True,
    opset: Optional[int] = None,
    decompose: bool = True,
    strategy: Optional[str] = None,
    verbose: Optional[bool] = None,
) -> CachedEinsum:
    """
    Proposes a new implementation of :func:`numpy.einsum`.
    It does not allow expresion using `...` and expects
    a right member.

    :param equation: einsum equation
    :param dtype: numpy dtype used for the computation
    :param optimize: permutes all letters to find the best
        permutation
    :param runtime: runtime used to compute the results once the
        computation graph is produced (see below)
    :param cache: if True, the function stores the preprocessing
        done for a specific equation, the second call with the same
        equation is much faster
    :param opset: ONNX opset to use for some runtimes
    :param decompose: by default, the function decomposes
        the equation into more simple operators but it can keep
        the original ONNX einsum operator.
    :param strategy: optimisation strategy (see below)
    :param verbose: display progress if optimize is True
    :return: einsum result

    The available runtimes are:

    * `batch_dot`: the runtime is :func:`apply_einsum_sequence
      <onnx_extended.tools.einsum.einsum_impl.apply_einsum_sequence>`,
    * `python`: one ONNX graph executed with a python runtime,
    * `onnxruntime`: one ONNX graph executed with :epkg:`onnxruntime`.

    The optimisation strategy can be:

    * `None`: the same runtime is used to find the best permutation of letters
    * `'ml'`: a machine learned model is used to predict the
        best permutation of letters.

    The function works in two steps:

    * first step analyses the equation to produce a computation graph,
      this graph can also be converted into ONNX,
    * second step runs the graph whatever the graph is.

    The function returns an object of type :class:`CachedEinsum`
    which has the following members after optimization:

    * `equation_` corresponding to the best equivalent equation
    * `graph_`: the corresponding graph returned by function
        :func:`decompose_einsum_equation
        <onnx_extended.tools.einsum.einsum_impl.decompose_einsum_equation>`
    * `onnx_`: if a conversion to onnx is used, stores the onnx graph
    * `runtime_`: a function used by `__call__`, calls the runtime
    * `oinf_`: an object of type :class:`CReferenceEvaluator
      <onnx_extended.reference.CReferenceEvaluator>`
    * `timed_permutations_`: memorizes the results of the optimization

    .. runpython::
        :showcode:

        import numpy
        from onnx_extended.tools.einsum import optimize_decompose_einsum_equation

        seq_opt = optimize_decompose_einsum_equation(
            "bsnh,btnh->bnts", numpy.float64, strategy='ml', verbose=1,
            runtime="python", optimize=True)

        print("best equation:", seq_opt.equation_)

    """
    res = _einsum(
        equation,
        dtype,
        optimize=optimize,
        runtime=runtime,
        cache=cache,
        opset=opset,
        decompose=decompose,
        strategy=strategy,
        verbose=verbose,
    )
    return res


def einsum(
    equation: str,
    *inputs: List[numpy.ndarray],
    optimize: bool = False,
    runtime: str = "batch_dot",
    cache: bool = True,
    opset: Optional[int] = None,
    decompose: bool = True,
    strategy: Optional[str] = None,
    verbose: Optional[bool] = None,
) -> numpy.ndarray:
    """
    Proposes a new implementation of :func:`numpy.einsum`.
    It does not allow expresion using `...` and expects
    a right member.

    :param equation: einsum equation
    :param inputs: inputs
    :param optimize: permutes all letters to find the best
        permutation
    :param runtime: runtime used to compute the results once the
        computation graph is produced (see below)
    :param cache: if True, the function stores the preprocessing
        done for a specific equation, the second call with the same
        equation is much faster
    :param opset: ONNX opset to use for some runtimes
    :param decompose: by default, the function decomposes
        the equation into more simple operators but it can keep
        the original ONNX einsum operator.
    :param strategy: optimisation strategy (see below)
    :param verbose: display progress if optimize is True
    :return: einsum result

    The available runtimes are:

    * `batch_dot`: the runtime is :func:`apply_einsum_sequence
      <onnx_extended.tools.einsum.einsum_impl.apply_einsum_sequence>`,
    * `python`: one ONNX graph executed with a python runtime,
    * `onnxruntime`: one ONNX graph executed with :epkg:`onnxruntime`.

    The optimisation strategy can be:

    * `None`: the same runtime is used to find the best permutation of letters
    * `'ml'`: a machine learned model is used to predict the
        best permutation of letters.

    The function works in two steps:

    * first step analyses the equation to produce a computation graph,
      this graph can also be converted into ONNX,
    * second step runs the graph whatever the graph is.

    Further details are available in the documentation of function
    :func:`optimize_decompose_einsum_equation`.
    The function works the same way as :func:`numpy.einsum`:

    .. runpython::
        :showcode:

        import numpy
        from onnx_extended.tools.einsum import einsum

        equation = "abc,cd->abd"

        m1 = numpy.random.randn(2, 2, 2)
        m2 = numpy.random.randn(2, 2)

        np = numpy.einsum(equation, m1, m2)
        print('numpy.einsum')
        print(np)

        print('onnx_extended.tools.einsum')
        mp = einsum(equation, m1, m2)
        print(mp)

    In some case, the einsum implementation can be optimized by looping
    on possible permutation:

    .. runpython::
        :showcode:
        :process:

        import timeit
        import numpy
        from onnx_extended.tools.einsum import einsum
        from onnx_extended.tools.einsum.einsum_fct import enumerate_cached_einsum

        equation = "cab,cd->ad"

        m1 = numpy.random.randn(20, 20, 20)
        m2 = numpy.random.randn(20, 20)

        print(
            "numpy.einsum",
            timeit.timeit(
                "numpy.einsum(equation, m1, m2)",
                number=200, globals=globals()
            ),
        )

        einsum(equation, m1, m2)
        print(
            "einsum", timeit.timeit(
                "einsum(equation, m1, m2)", number=200, globals=globals()
            )
        )

        einsum(equation, m1, m2, runtime="python")
        print(
            "einsum-python",
            timeit.timeit(
                'einsum(equation, m1, m2, runtime="python")',
                number=200, globals=globals()
            ),
        )

        einsum(equation, m1, m2, runtime="onnxruntime")
        print(
            "einsum-onnxruntime",
            timeit.timeit(
                'einsum(equation, m1, m2, runtime="onnxruntime")',
                number=200, globals=globals()
            ),
        )

        einsum(equation, m1, m2, runtime="onnxruntime", optimize=True, verbose=1)
        print(
            "einsum-onnxruntime",
            timeit.timeit(
                'einsum(equation, m1, m2, runtime="onnxruntime", optimize=True)',
                number=200,
                globals=globals(),
            ),
        )

        print("list of cached einsum equations")
        for k, v in enumerate_cached_einsum():
            print(k, v.equation, v.equation_)

    The last example shows the time taken by every function:

    .. runpython::
        :showcode:
        :process:

        import logging
        import os
        import cProfile
        from io import StringIO
        from pstats import Stats
        import numpy
        from onnx_extended.tools.einsum import einsum
        from onnx_extended.tools.einsum.einsum_fct import enumerate_cached_einsum
        from onnx_extended import __file__ as path


        def profile(fct, sort="cumulative", **kwargs):
            pr = cProfile.Profile(**kwargs)
            pr.enable()
            fct_res = fct()
            pr.disable()
            s = StringIO()
            ps = Stats(pr, stream=s).sort_stats(sort)
            ps.print_stats()
            res = s.getvalue()
            return ps, res


        root = os.path.dirname(path)
        logging.getLogger("onnx-extended").setLevel(logging.ERROR)

        equation = "cab,cd->ad"

        m1 = numpy.random.randn(200, 20, 20)
        m2 = numpy.random.randn(200, 20)


        def clean(txt):
            txt = txt.replace(root, "onnx_extended")
            return "\\n".join(txt.split("\\n")[:30])


        def fct1():
            for i in range(100):
                einsum(equation, m1, m2, cache=False)


        print("Profile cache with default runtime.")
        res = profile(fct1)
        print(root)
        print(clean(res[1]))


        def fct2():
            for i in range(100):
                einsum(equation, m1, m2, cache=False, runtime="python")


        print("Profile cache with runtime='python'.")
        res = profile(fct2)
        print(root)
        print(clean(res[1]))


        def fct3():
            for i in range(100):
                einsum(equation, m1, m2, cache=True)


        einsum(equation, m1, m2, cache=True)
        print("Profile execution with default runtime.")
        res = profile(fct3)
        print(root)
        print(clean(res[1]))


        def fct4():
            for i in range(100):
                einsum(equation, m1, m2, cache=True, runtime="python")


        einsum(equation, m1, m2, cache=True, runtime="python")
        print("Profile execution with runtime='python'.")
        res = profile(fct4)
        print(root)
        print(clean(res[1]))


        def fct5():
            for i in range(100):
                einsum(equation, m1, m2, cache=True, runtime="onnxruntime")


        einsum(equation, m1, m2, cache=True, runtime="onnxruntime")
        print("Profile execution with runtime='onnxruntime'.")
        res = profile(fct5)
        print(root)
        print(clean(res[1]))
    """
    assert inputs, "No inputs found."
    dtypes = set(i.dtype for i in inputs)
    assert len(dtypes) == 1, (
        f"All inputs do not have the same type ({dtypes!r}), "
        f"all of them should be cast before called einsum."
    )
    cached = optimize_decompose_einsum_equation(
        equation,
        inputs[0].dtype,
        optimize=optimize,
        runtime=runtime,
        cache=cache,
        opset=opset,
        decompose=decompose,
        strategy=strategy,
        verbose=verbose,
    )
    return cached(*inputs)
