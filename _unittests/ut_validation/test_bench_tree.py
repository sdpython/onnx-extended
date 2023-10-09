import unittest
from contextlib import redirect_stdout
from io import StringIO
from onnx import ModelProto
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.validation.bench_trees import create_decision_tree, bench_trees
from onnx_extended.validation._tree_d14_f100 import tree_d14_f100
from onnx_extended.tools.onnx_io import onnx2string


class TestBenchTree(ExtTestCase):
    def test_create_decision_tree(self):
        tree = create_decision_tree(max_depth=2)
        code = onnx2string(tree, as_code=True)
        self.assertNotIn("import textwrap", code)
        # with open("onnx_extended/validation/_tree_d14_f100.py", "w") as f:
        #     f.write(code)

    def test_tree14(self):
        model = tree_d14_f100()
        self.assertIsInstance(model, ModelProto)

    def test_bench_tree(self):
        res = bench_trees(
            max_depth=2,
            n_estimators=10,
            n_features=4,
            batch_size=100,
            number=10,
            warmup=2,
            verbose=0,
            engine_names=["onnxruntime", "CReferenceEvaluator"],
        )
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 4)

    def test_bench_tree_verbose(self):
        st = StringIO()
        with redirect_stdout(st):
            res = bench_trees(
                max_depth=2,
                n_estimators=10,
                n_features=4,
                batch_size=100,
                number=10,
                warmup=2,
                engine_names=["CReferenceEvaluator"],
                verbose=2,
            )
        text = st.getvalue()
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 2)
        self.assertIn("test 'CReferenceEvaluator' duration=", text)

    def test_bench_tree_all_engines(self):
        res = bench_trees(
            max_depth=2,
            n_estimators=10,
            n_features=4,
            batch_size=100,
            number=10,
            warmup=2,
            repeat=1,
            engine_names=["onnxruntime", "onnxruntime-customops"],
        )
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
