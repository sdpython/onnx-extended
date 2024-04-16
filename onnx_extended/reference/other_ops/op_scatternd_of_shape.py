import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_scatternd import _scatter_nd_impl


class ScatterNDOfShape(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, shape, indices, updates, reduction=None, strategy=None):
        data = np.zeros(shape, dtype=updates.dtype)
        y = _scatter_nd_impl(data, indices, updates, reduction=reduction)
        return (y,)
