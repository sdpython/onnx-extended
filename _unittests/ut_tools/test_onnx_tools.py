import unittest
from io import StringIO
from contextlib import redirect_stdout
from typing import Callable, List, Optional, Tuple
import numpy
from onnx import GraphProto, ModelProto, TensorProto
from onnx.checker import check_model
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_opsetid,
)
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from onnx.parser import parse_model
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.tools.onnx_nodes import (
    enumerate_onnx_node_types,
    onnx_merge_models,
    convert_onnx_model,
)
from onnx_extended.tools.onnx_io import onnx2string, string2onnx


class TestOnnxTools(ExtTestCase):
    def _get_model(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])
        graph = make_graph(
            [
                make_node("Add", ["X", "Y"], ["z1"]),
                make_node("Mul", ["X", "z1"], ["Z"]),
            ],
            "add",
            [X, Y],
            [Z],
        )
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 18)], ir_version=8
        )
        check_model(onnx_model)
        return onnx_model, [
            {
                "level": 0,
                "name": "X",
                "kind": "input",
                "type": "tensor",
                "elem_type": "FLOAT",
                "shape": "?x?",
            },
            {
                "level": 0,
                "name": "Y",
                "kind": "input",
                "type": "tensor",
                "elem_type": "FLOAT",
                "shape": "?x?",
            },
            {
                "level": 0,
                "name": "",
                "kind": "Op",
                "domain": "",
                "type": "Add",
                "inputs": "X,Y",
                "outputs": "z1",
                "input_types": ",",
                "output_types": "FLOAT",
            },
            {
                "name": "z1",
                "kind": "result",
                "level": 0,
                "type": "tensor",
                "elem_type": "FLOAT",
                "shape": "unk__0xunk__1",
            },
            {
                "level": 0,
                "name": "",
                "kind": "Op",
                "domain": "",
                "type": "Mul",
                "inputs": "X,z1",
                "outputs": "Z",
                "input_types": ",FLOAT",
                "output_types": "FLOAT",
            },
            {
                "name": "Z",
                "kind": "result",
                "level": 0,
                "type": "tensor",
                "elem_type": "FLOAT",
                "shape": "?x?",
            },
            {
                "level": 0,
                "name": "Z",
                "kind": "output",
                "type": "tensor",
                "elem_type": "FLOAT",
                "shape": "?x?",
            },
        ]

    def test_enumerate_onnx_node_types(self):
        model, expected = self._get_model()
        res = list(enumerate_onnx_node_types(model))
        self.assertEqual(len(expected), len(res))
        for i, (a, b) in enumerate(zip(expected, res)):
            self.assertEqual(len(a), len(b), msg=f"Item(1) {i} failed a={a}, b={b}.")
            self.assertEqual(set(a), set(b), msg=f"Item(2) {i} failed a={a}, b={b}.")
            for k in sorted(a):
                self.assertEqual(a[k], b[k], msg=f"Item(3) {i} failed a={a}, b={b}.")

    def _load_model(self, m_def: str) -> ModelProto:
        """
        Parses a model from a string representation,
        including checking the model for correctness
        """
        m = parse_model(m_def)
        check_model(m)
        return m

    def _test_merge_models(
        self,
        m1def: str,
        m2def: str,
        io_map: List[Tuple[str, str]],
        check_expectations: Callable[[GraphProto, GraphProto, GraphProto], None],
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        prefix1: Optional[str] = None,
        prefix2: Optional[str] = None,
    ) -> None:
        m1, m2 = self._load_model(m1def), self._load_model(m2def)
        m3, stdout, stderr = self.capture(
            lambda: onnx_merge_models(m1, m2, io_map=io_map, verbose=1)
        )
        check_model(m3)
        check_expectations(m1.graph, m2.graph, m3.graph)
        if stderr:
            raise AssertionError(stderr)
        if stdout:
            self.assertIn("[onnx_merge_models]", stdout)

    def test_merge(self):
        M1_DEF = """
            <
                ir_version: 7,
                opset_import: [ "": 10, "com.microsoft": 1]
            >
            agraph (float[N, M] A0, float[N, M] A1, float[N, M] _A
                    ) => (float[N, M] B00, float[N, M] B10, float[N, M] B20)
            {
                B00 = Add(A0, A1)
                B10 = Sub(A0, A1)
                B20 = Mul(A0, A1)
            }
            """

        M2_DEF = """
            <
                ir_version: 7,
                opset_import: [ "": 10, "com.microsoft": 1]
            >
            agraph (float[N, M] B01, float[N, M] B11, float[N, M] B21
                    ) => (float[N, M] M1)
            {
                C0 = Add(B01, B11)
                C1 = Sub(B11, B21)
                M1 = Mul(C0, C1)
            }
            """

        def check_expectations(g1: GraphProto, g2: GraphProto, g3: GraphProto) -> None:
            self.assertEqual(g3.input, g1.input)
            self.assertEqual(g3.output, g2.output)
            self.assertEqual(
                ["Add", "Sub", "Mul", "Add", "Sub", "Mul"],
                [item.op_type for item in g3.node],
            )

        io_map = [("B00", "B01"), ("B10", "B11"), ("B20", "B21")]
        self._test_merge_models(M1_DEF, M2_DEF, io_map, check_expectations)

    def test_merge_opset1(self):
        M1_DEF = """
            <
                ir_version: 7,
                opset_import: [ "": 9]
            >
            agraph (float[N, M] A0, float[N, M] A1, float[N, M] _A
                    ) => (float[N, M] B00, float[N, M] B10, float[N, M] B20)
            {
                B00 = Add(A0, A1)
                B10 = Sub(A0, A1)
                B20 = Mul(A0, A1)
            }
            """

        M2_DEF = """
            <
                ir_version: 7,
                opset_import: [ "": 10]
            >
            agraph (float[N, M] B01, float[N, M] B11, float[N, M] B21
                    ) => (float[N, M] M1)
            {
                C0 = Add(B01, B11)
                C1 = Sub(B11, B21)
                M1 = Mul(C0, C1)
            }
            """

        def check_expectations(g1: GraphProto, g2: GraphProto, g3: GraphProto) -> None:
            self.assertEqual(g3.input, g1.input)
            self.assertEqual(g3.output, g2.output)
            self.assertEqual(
                ["Add", "Sub", "Mul", "Add", "Sub", "Mul"],
                [item.op_type for item in g3.node],
            )

        io_map = [("B00", "B01"), ("B10", "B11"), ("B20", "B21")]
        self._test_merge_models(M1_DEF, M2_DEF, io_map, check_expectations)

    def test_merge_opset2(self):
        M1_DEF = """
            <
                ir_version: 7,
                opset_import: [ "": 10]
            >
            agraph (float[N, M] A0, float[N, M] A1, float[N, M] _A
                    ) => (float[N, M] B00, float[N, M] B10, float[N, M] B20)
            {
                B00 = Add(A0, A1)
                B10 = Sub(A0, A1)
                B20 = Mul(A0, A1)
            }
            """

        M2_DEF = """
            <
                ir_version: 7,
                opset_import: [ "": 9]
            >
            agraph (float[N, M] B01, float[N, M] B11, float[N, M] B21
                    ) => (float[N, M] M1)
            {
                C0 = Add(B01, B11)
                C1 = Sub(B11, B21)
                M1 = Mul(C0, C1)
            }
            """

        def check_expectations(g1: GraphProto, g2: GraphProto, g3: GraphProto) -> None:
            self.assertEqual(g3.input, g1.input)
            self.assertEqual(g3.output, g2.output)
            self.assertEqual(
                ["Add", "Sub", "Mul", "Add", "Sub", "Mul"],
                [item.op_type for item in g3.node],
            )

        io_map = [("B00", "B01"), ("B10", "B11"), ("B20", "B21")]
        self._test_merge_models(M1_DEF, M2_DEF, io_map, check_expectations)

    def test_random_forest_regressor_as_tensor(self):
        from skl2onnx import to_onnx

        X, y = make_regression(100, 2, n_informative=1, random_state=32)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)

        rf = RandomForestRegressor(3, max_depth=2, random_state=32, n_jobs=-1)
        rf.fit(X[:80], y[:80])
        onx = to_onnx(rf, X[:1], target_opset={"ai.onnx.ml": 3, "": 18})
        st = StringIO()
        with redirect_stdout(st):
            new_onx = convert_onnx_model(
                onx,
                opsets={"": 18, "ai.onnx.ml": 3},
                use_as_tensor_attributes=True,
                verbose=3,
            )
        text = st.getvalue()
        self.assertIn("TreeEnsembleRegressor", text)
        self.assertIn("_as_tensor", str(new_onx))

    def test_model2string(self):
        model = self._get_model()[0]
        s = onnx2string(model)
        expected = (
            "CAg6WgoPCgFYCgFZEgJ6MSIDQWRkCg8KAVgKAnoxEgFaIg"
            "NNdWwSA2FkZFoPCgFYEgoKCAgBEgQKAAoAWg8KAVkSCgo"
            "ICAESBAoACgBiDwoBWhIKCggIARIECgAKAEIECgAQEg=="
        )
        self.assertEqual(expected, s)
        model2 = string2onnx(s)
        self.assertEqual(
            [n.op_type for n in model.graph.node],
            [n.op_type for n in model2.graph.node],
        )
        code = onnx2string(model, as_code=True)
        self.assertIn("model = string2onnx(text)", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
