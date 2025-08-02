import unittest
import onnx
import onnx.helper as xoh
import onnx_extended.onnx2.helper as xoh2
import onnx_extended.onnx2 as onnx2
from onnx_extended.ext_test_case import ExtTestCase


class TestOnnx2Helper(ExtTestCase):
    def assertEqualModelProto(self, model1, model2):
        self.assertEqual(type(model1), type(model2))
        search = 'domain: ""'
        s1 = model1.SerializeToString()
        s2 = model2.SerializeToString()
        spl1 = str(model1).split(search)
        spl2 = str(model2).split(search)
        if len(spl1) != len(spl2) or s1 != s2:
            n1 = self.get_dump_file("model1.onnx.txt")
            with open(n1, "w") as f:
                f.write(str(model1))
            n2 = self.get_dump_file("model2.onnx.txt")
            with open(n2, "w") as f:
                f.write(str(model2))
        self.assertEqual(len(spl1), len(spl2))
        self.assertEqual(s1, s2)

    @classmethod
    def make_model_gemm(cls, oh, tp):
        itype = tp.FLOAT
        return oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Gemm", ["X", "Y"], ["XY"]),
                    oh.make_node("Gemm", ["X", "Z"], ["XZ"]),
                    oh.make_node("Concat", ["XY", "XZ"], ["XYZ"], axis=1),
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", itype, [None, None]),
                    oh.make_tensor_value_info("Y", itype, [None, None]),
                    oh.make_tensor_value_info("Z", itype, [None, None]),
                ],
                [oh.make_tensor_value_info("XYZ", itype, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

    def test_model_gemm_onnx_to_onnx2(self):
        name = self.get_dump_file("test_model_gemm_onnx_to_onnx2.onnx")
        model = self.make_model_gemm(xoh, onnx.TensorProto)
        onnx.save(model, name)
        model2 = onnx2.load(name)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name2 = self.get_dump_file("test_model_gemm_onnx_to_onnx2_2.onnx")
        onnx2.save(model2, name2)
        model3 = onnx.load(name2)
        self.assertEqualModelProto(model, model3)

    def test_model_gemm_onnx2_to_onnx(self):
        name2 = self.get_dump_file("test_model_gemm_onnx2_to_onnx_2.onnx")
        model2 = self.make_model_gemm(xoh2, onnx2.TensorProto)
        onnx2.save(model2, name2)
        model = onnx.load(name2)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name = self.get_dump_file("test_model_gemm_onnx2_to_onnx.onnx")
        onnx.save(model, name)
        model3 = onnx.load(name)
        self.assertEqualModelProto(model, model3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
