from typing import Any, Dict

import numpy as np

from onnx import NodeProto
from onnx.reference.op_run import OpRun
from .cpu.c_op_conv_ import ConvDouble, ConvFloat


class Conv(OpRun):
    def __init__(
        self, onnx_node: NodeProto, run_params: Dict[str, Any], schema: Any = None
    ):
        OpRun.__init__(self, onnx_node, run_params, schema)
        self.cache_: Dict[type, Any] = {}

    def _run(
        self,
        X,
        W,
        B=None,
        auto_pad=None,
        dilations=None,
        group=None,
        kernel_shape=None,
        pads=None,
        strides=None,
    ):
        if X.dtype not in self.cache_:
            if X.dtype == np.float32:
                rt = ConvFloat()
            elif X.dtype == np.float64:
                rt = ConvDouble()
            else:
                raise TypeError(
                    f"No C implementation C for operator 'Conv' and dtype={X.dtype}."
                )
            self.cache_[X.dtype] = rt

            rt.init(
                auto_pad,
                np.array(dilations or [], dtype=np.int64),
                group,
                np.array(kernel_shape or [], dtype=np.int64),
                np.array(pads or [], dtype=np.int64),
                np.array(strides or [], dtype=np.int64),
            )

        rt = self.cache_[X.dtype]

        assert X is not None, f"X cannot be None for operator {type(self)}."
        assert (
            min(X.shape) != 0
        ), f"Unable to run operator Conv on an empty matrix. X.shape={X.shape!r}."
        assert (
            B is None or min(B.shape) != 0
        ), f"Unable to run operator Conv on an empty matrix. B.shape={B.shape!r}."
        cv = rt.compute(X, W, B)
        return (cv,)
