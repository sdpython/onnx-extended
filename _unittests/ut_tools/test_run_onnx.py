import unittest
import os
import tempfile
import numpy as np
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_opsetid,
)
from onnx.checker import check_model
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.reference import CReferenceEvaluator
from onnx_extended.tools.run_onnx import save_for_benchmark_or_test, TestRun


class TestRunOnnx(ExtTestCase):
    def _get_model(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info("Z", TensorProto.INT64, [None, None])
        graph = make_graph(
            [
                make_node("Add", ["X", "Y"], ["z1"]),
                make_node("Mul", ["X", "z1"], ["z2"]),
                make_node("Cast", ["z2"], ["Z"], to=TensorProto.INT64),
            ],
            "add",
            [X, Y],
            [Z],
        )
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 18)], ir_version=8
        )
        check_model(onnx_model)
        return onnx_model

    def test_save_for_benchmark_or_test(self):
        model = self._get_model()
        inputs = [
            np.arange(4).reshape((2, 2)).astype(np.float32),
            np.arange(4).reshape((2, 2)).astype(np.float32),
        ]

        with tempfile.TemporaryDirectory() as temp:
            save_for_benchmark_or_test(temp, "t1", model, inputs)
            self.assertExists(os.path.join(temp, "t1"))
            self.assertExists(os.path.join(temp, "t1", "model.onnx"))

            tr = TestRun(os.path.join(temp, "t1"))
            res = tr.test(
                f_build=lambda proto: CReferenceEvaluator(proto),
                f_run=lambda rt, feeds: rt.run(None, feeds),
            )
            self.assertEmpty(res)
            bench = tr.bench(
                f_build=lambda proto: CReferenceEvaluator(proto),
                f_run=lambda rt, feeds: rt.run(None, feeds),
            )
            examples = {
                "build_time": 0.0012805999995180173,
                "shapes": {"X": (2, 2), "Y": (2, 2)},
                "dtypes": {"X": np.float32, "Y": np.float32},
                "warmup_time": 0.0002456999991409248,
                "warmup": 5,
                "name": "/tmp/tmpz2tw4rqj/t1",
                "index": 0,
                "repeat": 10,
                "avg_time": 3.1990000206860716e-05,
                "min_time": 3.1000001399661414e-05,
                "max_time": 3.319999996165279e-05,
                "max1_time": 3.2699999792384915e-05,
                "min1_time": 3.130000004603062e-05,
                "input_size": 8,
            }
            for k, v in examples.items():
                self.assertIn(k, bench)
                if "_time" in k:
                    self.assertIsInstance(v, float)
                elif k == "name":
                    self.assertEqual(os.path.join(temp, "t1"), bench[k])
                else:
                    self.assertEqual(examples[k], bench[k])


if __name__ == "__main__":
    unittest.main(verbosity=2)
