import os
import re
import unittest
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unittest.case import SkipTest
import numpy
import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx import FunctionProto, ModelProto, NodeProto
from onnx.backend.test import BackendTest
from onnx.backend.test.runner import Pattern, TestItem
from onnx.backend.test.loader import load_model_tests
from onnx.backend.base import Backend, Device, DeviceType
from .c_reference_evaluator import CReferenceEvaluator


class Runner:
    """
    Collects tests and run them as unit tests.

    :param backend: a subclass of :class:`onnx.backend.base.Backend`
    :param path_to_test: folder to look at
    :param kind: subfolder to test
    :param test_kwargs: additional test parameters
    """

    _add_model_test = BackendTest._add_model_test
    _add_test = BackendTest._add_test
    _load_proto = BackendTest._load_proto
    assert_similar_outputs = BackendTest.assert_similar_outputs

    def __init__(
        self,
        backend: type[Backend],
        path_to_test: Optional[str] = None,
        kind: Optional[Union[str, List[str]]] = None,
        test_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.backend = backend
        self._include_patterns: Set[Pattern[str]] = set()
        self._exclude_patterns: Set[Pattern[str]] = set()
        self._xfail_patterns: Set[Pattern[str]] = set()
        self._test_kwargs: dict = test_kwargs or {}

        if path_to_test is None or not os.path.exists(path_to_test):
            raise FileNotFoundError(f"Unable to find path {path_to_test!r}.")
        if isinstance(kind, str):
            kind = [kind]
        elif kind is None:
            kind = [
                k
                for k in os.listdir(path_to_test)
                if os.path.isdir(os.path.join(path_to_test, k))
            ]

        self._test_items: Dict[str, Dict[str, TestItem]] = {
            f"{k}Model": {} for k in kind
        }

        for k in kind:
            for ot in load_model_tests(path_to_test, kind=k):
                self._add_model_test(ot, k)

    def include(self, pattern: str) -> "Runner":
        self._include_patterns.add(re.compile(pattern))
        return self

    def exclude(self, pattern: str) -> "Runner":
        self._exclude_patterns.add(re.compile(pattern))
        return self

    def xfail(self, pattern: str) -> "Runner":
        self._xfail_patterns.add(re.compile(pattern))
        return self

    def _filtered_test_items(self) -> dict[str, dict[str, TestItem]]:
        filtered: dict[str, dict[str, TestItem]] = {}
        for category, items_map in self._test_items.items():
            filtered[category] = {}
            for name, item in items_map.items():
                if self._include_patterns and (
                    not any(include.search(name) for include in self._include_patterns)
                ):
                    item.func = unittest.skip("no matched include pattern")(item.func)
                for exclude in self._exclude_patterns:
                    if exclude.search(name):
                        item.func = unittest.skip(
                            f"matched exclude pattern '{exclude.pattern}'"
                        )(item.func)
                for xfail in self._xfail_patterns:
                    if xfail.search(name):
                        item.func = unittest.expectedFailure(item.func)
                filtered[category][name] = item
        return filtered

    def tests(self, name: str = "CustomTestCase") -> type[unittest.TestCase]:
        """
        Returns a subclass of `unittest.TestCase`.

        :param name: name of the subclass
        """
        tests = type("CustomTestCase", (unittest.TestCase,), {})
        for items_map in sorted(
            self._filtered_test_items().values(), key=lambda cl: cl.__class__.__name__
        ):
            for name, item in sorted(items_map.items()):
                setattr(tests, name, item.func)
        return tests

    def run(self, verbose: int = 0, exc_cls: Optional[type] = AssertionError) -> Tuple[
        List[Tuple[str, Callable]],
        List[Tuple[str, Callable, Any]],
        List[Tuple[str, Callable, Exception]],
    ]:
        """
        Runs all tests.

        :param verbose: verbosity, use :epkg:`tqdm`
        :param exc_cls: exception to raise when a test fails, if None,
            no exception is raised
        :return: list of run tests, list of skipped tests, list of failed tests
        """
        tests = self.tests()
        methods = []
        for att in dir(tests):
            if att.startswith("test_"):
                test = getattr(tests, att)
                methods.append((att, test))
        assert (
            methods
        ), f"No test was detected. Available tests are:\n{', '.join(dir(tests))}"

        if verbose:
            from tqdm import tqdm

            loop = tqdm(methods)
        else:
            loop = methods

        ran = []
        skipped = []
        failed = []
        for i, (name, f) in enumerate(loop):
            if verbose:
                loop.set_description(f"{i+1}/{len(methods)}-{name}")
            try:
                f(tests)
            except SkipTest as es:
                skipped.append((name, f, es))
                continue
            except Exception as e:
                if exc_cls is not None:
                    raise exc_cls(f"Test {i}-{name!r} failed.") from e
                failed.append((name, f, e))
            ran.append((name, f))
        return ran, skipped, failed


class CReferenceEvaluatorBackendRep(onnx.backend.base.BackendRep):
    """
    See :class:`onnx_extended.reference.CReferenceEvaluator`
    for an example.

    :param session: any runtime with the same interface as
        :class:`onnx.reference.ReferenceEvaluator`
    """

    def __init__(self, session: CReferenceEvaluator):
        self._session = session

    def run(self, inputs: List[numpy.ndarray], **kwargs) -> List[numpy.ndarray]:
        if isinstance(inputs, numpy.ndarray):
            inputs = [inputs]
        if isinstance(inputs, list):
            if len(inputs) == len(self._session.input_names):
                feeds = dict(zip(self._session.input_names, inputs))
            else:
                feeds = {}
                pos_inputs = 0
                for inp, tshape in zip(
                    self._session.input_names, self._session.input_types
                ):
                    shape = tuple(d.dim_value for d in tshape.tensor_type.shape.dim)
                    if shape == inputs[pos_inputs].shape:
                        feeds[inp] = inputs[pos_inputs]
                        pos_inputs += 1
                        if pos_inputs >= len(inputs):
                            break
        elif isinstance(inputs, dict):
            feeds = inputs
        else:
            raise TypeError(f"Unexpected input type {type(inputs)!r}.")
        outs = self._session.run(None, feeds)
        return outs


class CReferenceEvaluatorBackend(onnx.backend.base.Backend):
    """
    See :class:`onnx_extended.reference.CReferenceEvaluator`
    for an example.
    """

    cls_inference = CReferenceEvaluator

    @classmethod
    def __class_getitem__(cls, cls_inference: type, name: Optional[str] = None) -> type:
        """
        Creates a new class inheriting from this one but with
        static attribute `cls_inference` equal to *cls_inference*.
        The goal is to make it easier to evaluate a runtime
        sharing the same API as the :class:`CReferenceEvaluator`
        on CPU.
        """
        if name is None:
            name = f"{cls.__name__}{cls_inference.__name__}"
        return type(name, (cls,), {"cls_inference": cls_inference})

    @classmethod
    def is_opset_supported(cls, model):
        """
        Tells which opsets are supported.
        """
        return True, ""

    @classmethod
    def supports_device(cls, device: str) -> bool:
        """
        Tells if a specific device is supported.
        """
        d = Device(device)
        return d.type == DeviceType.CPU

    @classmethod
    def create_inference_session(
        cls, model: Union[str, bytes, ModelProto, NodeProto, FunctionProto]
    ):
        """
        Creates an instance of the class running a model.
        """
        return cls.cls_inference(model)

    @classmethod
    def prepare(
        cls, model: Any, device: str = "CPU", **kwargs: Dict[str, Any]
    ) -> CReferenceEvaluatorBackendRep:
        if isinstance(model, cls.cls_inference):
            return CReferenceEvaluatorBackendRep(model)
        if isinstance(model, (str, bytes, ModelProto)):
            inf = cls.create_inference_session(model)
            return cls.prepare(inf, device, **kwargs)
        raise TypeError(f"Unexpected type {type(model)} for model.")

    @classmethod
    def run_model(
        cls,
        model,
        inputs: List[Any],
        device: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ):
        """
        Called if the onnx proto is a `ModelProto`.
        """
        rep = cls.prepare(model, device or "cpu", **kwargs)
        return rep.run(inputs, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        """
        Called if the onnx proto is a `NodeProto`.
        """
        raise NotImplementedError("Unable to run the model node by node.")


def create_reference_backend(
    backend: Optional[type[Backend]] = None,
    path_to_test: Optional[str] = None,
    kind: Optional[str] = None,
) -> Runner:
    return Runner(
        backend or CReferenceEvaluatorBackend,
        path_to_test=path_to_test,
        kind=kind,
    )
