from typing import Any, List, Optional
import numpy
import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx import ModelProto
from onnx.backend.test.loader import load_model_tests
from onnx.backend.base import Backend, Device, DeviceType
from .c_reference_evaluator import CReferenceEvaluator


class Runner(onnx.backend.test.BackendTest):
    def __init__(
        self,
        backend: type[Backend],
        parent_module: Optional[str] = None,
        path_to_test: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> None:
        onnx.backend.test.BackendTest.__init__(
            self, backend=backend, parent_module=parent_module
        )

        if path_to_test is None:
            if kind is None:
                for rt in load_model_tests(kind="node"):
                    self._add_model_test(rt, "Node")
        if kind is None:
            raise ValueError("path_to_test is defined, so kind must be as well.")

        for ot in load_model_tests(path_to_test, kind=kind):
            self._add_model_test(ot, kind)


class CReferenceEvaluatorBackendRep(onnx.backend.base.BackendRep):
    """
    See :class:`onnx_extended.reference.CReferenceEvaluator`
    for an example.
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
        return True, ""

    @classmethod
    def supports_device(cls, device: str) -> bool:
        d = Device(device)
        return d.type == DeviceType.CPU

    @classmethod
    def create_inference_session(cls, model):
        return cls.cls_inference(model)

    @classmethod
    def prepare(
        cls, model: Any, device: str = "CPU", **kwargs: Any
    ) -> CReferenceEvaluatorBackendRep:
        if isinstance(model, cls.cls_inference):
            return CReferenceEvaluatorBackendRep(model)
        if isinstance(model, (str, bytes, ModelProto)):
            inf = cls.create_inference_session(model)
            return cls.prepare(inf, device, **kwargs)
        raise TypeError(f"Unexpected type {type(model)} for model.")

    @classmethod
    def run_model(cls, model, inputs, device=None, **kwargs):
        rep = cls.prepare(model, device, **kwargs)
        return rep.run(inputs, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        raise NotImplementedError("Unable to run the model node by node.")


def create_reference_backend(
    backend: Optional[type] = None,
    path_to_test: Optional[str] = None,
    kind: Optional[str] = None,
) -> CReferenceEvaluatorBackend:
    return Runner(
        backend or CReferenceEvaluatorBackend,
        __name__,
        path_to_test=path_to_test,
        kind=kind,
    )
