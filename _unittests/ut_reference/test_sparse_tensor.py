import unittest
import numpy as np
from scipy.sparse import coo_matrix
from onnx import TensorProto, helper, checker
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.reference import to_array_extended
from onnx_extended.reference import CReferenceEvaluator


class TestSparseTensor(ExtTestCase):
    def test_sparse_tensor_1d(self):
        dense_shape = [3, 3]
        sparse_values = [1.764052391052246, 0.40015721321105957, 0.978738009929657]
        values_tensor = helper.make_tensor(
            name="sparse_values",
            data_type=TensorProto.FLOAT,
            dims=[len(sparse_values)],
            vals=np.array(sparse_values).astype(np.float32),
            raw=False,
        )

        linear_indices = [2, 3, 5]
        indices_tensor = helper.make_tensor(
            name="indices",
            data_type=TensorProto.INT64,
            dims=[len(linear_indices)],
            vals=np.array(linear_indices).astype(np.int64),
            raw=False,
        )
        sparse_tensor = helper.make_sparse_tensor(
            values_tensor, indices_tensor, dense_shape
        )

        attr = helper.make_attribute("sparse_attr", sparse_tensor)
        self.assertEqual(attr.name, "sparse_attr")
        checker.check_sparse_tensor(helper.get_attribute_value(attr))
        checker.check_attribute(attr)

        back = to_array_extended(sparse_tensor)
        self.assertEqualArray(
            np.array(
                [
                    [0.0, 0.0, 1.764052391052246],
                    [0.40015721321105957, 0.0, 0.978738009929657],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            back.toarray(),
        )

    def test_sparse_tensor_2d(self):
        dense_shape = [3, 3]
        sparse_values = [1.764052391052246, 0.40015721321105957, 0.978738009929657]
        values_tensor = helper.make_tensor(
            name="sparse_values",
            data_type=TensorProto.FLOAT,
            dims=[len(sparse_values)],
            vals=np.array(sparse_values).astype(np.float32),
            raw=False,
        )

        linear_indices = np.array([[0, 2], [1, 0], [1, 2]], dtype=np.int64)
        indices_tensor = helper.make_tensor(
            name="indices",
            data_type=TensorProto.INT64,
            dims=list(linear_indices.shape),
            vals=linear_indices.ravel(),
            raw=False,
        )
        sparse_tensor = helper.make_sparse_tensor(
            values_tensor, indices_tensor, dense_shape
        )

        attr = helper.make_attribute("sparse_attr", sparse_tensor)
        self.assertEqual(attr.name, "sparse_attr")
        checker.check_sparse_tensor(helper.get_attribute_value(attr))
        checker.check_attribute(attr)

        back = to_array_extended(sparse_tensor)
        self.assertEqualArray(
            np.array(
                [
                    [0.0, 0.0, 1.764052391052246],
                    [0.40015721321105957, 0.0, 0.978738009929657],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            back.toarray(),
        )

    def test_reference_implementation(self):
        onx = helper.make_model(
            helper.make_graph(
                [helper.make_node("Shape", ["X"], ["Y"])],
                "name",
                [
                    helper.make_sparse_tensor_value_info(
                        "X", TensorProto.FLOAT, [None, None]
                    )
                ],
                [helper.make_tensor_value_info("Y", TensorProto.INT64, [None])],
            )
        )
        ref = CReferenceEvaluator(onx)
        sp = coo_matrix(
            np.array(
                [
                    [0.0, 0.0, 1.764052391052246],
                    [0.40015721321105957, 0.0, 0.978738009929657],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        got = ref.run(None, {"X": sp})
        self.assertEqualArray(np.array([3, 3], dtype=np.int64), got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
