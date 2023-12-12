import unittest
import numpy as np
from onnx_extended.ext_test_case import ExtTestCase, skipif_ci_apple


class TestSparseStruct(ExtTestCase):
    @skipif_ci_apple("crash")
    def test_sparse_struct(self):
        from onnx_extended.validation.cpu._validation import (
            sparse_struct_to_dense,
            dense_to_sparse_struct,
        )

        dense = np.zeros((10, 10), dtype=np.float32)
        dense[0, 0] = 777
        dense[9, 9] = 888
        dense[6, 3] = 555
        sp = dense_to_sparse_struct(dense)
        self.assertLess(sp.size, dense.size)
        dense2 = sparse_struct_to_dense(sp)
        self.assertEqualArray(dense, dense2)

    @skipif_ci_apple("crash")
    def test_sparse_struct_maps(self):
        from onnx_extended.validation.cpu._validation import (
            dense_to_sparse_struct,
            sparse_struct_to_maps,
        )

        dense = np.zeros((10, 11), dtype=np.float32)
        dense[0, 0] = 777
        dense[9, 9] = 888
        dense[6, 3] = 555
        sp = dense_to_sparse_struct(dense)
        self.assertLess(sp.size, dense.size)
        maps = sparse_struct_to_maps(sp)
        self.assertIsInstance(maps, list)
        self.assertIsInstance(maps[0], dict)
        self.assertEqual(
            maps, [{0: 777.0}, {}, {}, {}, {}, {}, {3: 555.0}, {}, {}, {9: 888.0}]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
