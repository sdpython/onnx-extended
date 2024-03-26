import contextlib
import os
import unittest
from io import StringIO
import numpy
from onnx.checker import check_model
from onnx import TensorProto, helper, load
from onnx_extended.ext_test_case import ExtTestCase, skipif_ci_windows
from onnx_extended.reference import CReferenceEvaluator
from onnx.inliner import inline_local_functions
from onnx_extended.tools.onnx_inline import onnx_inline_function

# from onnx_extended.tools.onnx_inline import inline_local_functions


class TestOnnxInline(ExtTestCase):
    def test_onnx_inline_subgraph(self, log=False):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N"])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, ["N"])
        one = helper.make_tensor_value_info("one", TensorProto.FLOAT, ["N"])

        graph1 = helper.make_graph([], "then", [], [X])
        graph2 = helper.make_graph([], "else", [], [one])

        model_def = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Constant", [], ["one"], value_floats=[1.0]),
                    helper.make_node("Greater", ["X", "one"], ["cond"]),
                    helper.make_node(
                        "If", ["cond"], ["Z"], then_branch=graph1, else_branch=graph2
                    ),
                ],
                "test",
                [X],
                [Z],
            ),
            ir_version=7,
            opset_imports=[helper.make_operatorsetid("", 15)],
        )
        check_model(model_def)

        feeds = {"X": numpy.array([-5], dtype=numpy.float32)}
        oinf = CReferenceEvaluator(model_def)
        got = oinf.run(None, feeds)
        for fi in [onnx_inline_function, inline_local_functions]:
            with self.subTest(f=fi):
                inlined = fi(model_def)
                if isinstance(inlined, tuple):
                    inlined = inlined[0]
                check_model(inlined)
                oinf = CReferenceEvaluator(inlined)
                goti = oinf.run(None, feeds)
                self.assertEqualArray(got[0], goti[0])

    def test_onnx_inline_subgraph_function(self, log=False):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N"])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, ["N"])
        one = helper.make_tensor_value_info("one", TensorProto.FLOAT, ["N"])

        graph1 = helper.make_graph([], "then", [], [X])
        graph2 = helper.make_graph([], "else", [], [one])

        func_def = helper.make_function(
            "this",
            "fct",
            ["X"],
            ["Z"],
            [
                helper.make_node("Constant", [], ["one"], value_floats=[1.0]),
                helper.make_node("Greater", ["X", "one"], ["cond"]),
                helper.make_node(
                    "If", ["cond"], ["Z"], then_branch=graph1, else_branch=graph2
                ),
            ],
            opset_imports=[helper.make_operatorsetid("", 15)],
        )

        model_def = helper.make_model(
            helper.make_graph(
                [helper.make_node("fct", ["X"], ["Z"], domain="this")], "test", [X], [Z]
            ),
            ir_version=7,
            opset_imports=[
                helper.make_operatorsetid("", 15),
                helper.make_operatorsetid("this", 1),
            ],
            functions=[func_def],
        )
        check_model(model_def)

        feeds = {"X": numpy.array([-5], dtype=numpy.float32)}
        oinf = CReferenceEvaluator(model_def)
        got = oinf.run(None, feeds)
        for fi in [onnx_inline_function, inline_local_functions]:
            with self.subTest(f=fi):
                inlined = fi(model_def)
                if isinstance(inlined, tuple):
                    inlined = inlined[0]
                check_model(inlined)
                self.assertNotIn("functions {", str(inlined))
                try:
                    oinf = CReferenceEvaluator(inlined)
                    goti = oinf.run(None, feeds)
                except RuntimeError:
                    if fi is inline_local_functions:
                        # bug in onnx
                        continue
                self.assertEqualArray(got[0], goti[0])
                self.assertEqualArray(got[0], numpy.array([1], dtype=numpy.float32))

    @skipif_ci_windows("crash")
    def test_onnx_inline_subgraph_function_double(self, log=False):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N"])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N"])
        Z = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N"])

        func_def_add = helper.make_function(
            "this",
            "fctadd",
            ["input2"],
            ["output"],
            [
                helper.make_node("Constant", [], ["one"], value_floats=[1.0]),
                helper.make_node("Add", ["input2", "one"], ["output"]),
            ],
            opset_imports=[helper.make_operatorsetid("", 15)],
        )

        graph1 = helper.make_graph(
            [helper.make_node("fctadd", ["input"], ["output"], domain="this")],
            "then",
            [],
            [out],
        )
        graph2 = helper.make_graph(
            [helper.make_node("fctadd", ["input"], ["output"], domain="this")],
            "else",
            [],
            [out],
        )

        func_def = helper.make_function(
            "this",
            "fct",
            ["input"],
            ["output"],
            [
                helper.make_node("Constant", [], ["one"], value_floats=[1.0]),
                helper.make_node("Greater", ["input", "one"], ["cond"]),
                helper.make_node(
                    "If", ["cond"], ["output"], then_branch=graph1, else_branch=graph2
                ),
            ],
            opset_imports=[
                helper.make_operatorsetid("", 15),
                helper.make_operatorsetid("this", 1),
            ],
        )

        model_def = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("fct", ["X"], ["ztmp"], domain="this"),
                    helper.make_node("fct", ["ztmp"], ["output"], domain="this"),
                ],
                "test",
                [X],
                [Z],
            ),
            ir_version=7,
            opset_imports=[
                helper.make_operatorsetid("", 15),
                helper.make_operatorsetid("this", 1),
            ],
            functions=[func_def_add, func_def],
        )

        feeds = {"X": numpy.array([-5], dtype=numpy.float32)}
        import onnxruntime

        oinf = onnxruntime.InferenceSession(model_def.SerializeToString())
        got = oinf.run(None, feeds)
        for fi in [onnx_inline_function, inline_local_functions]:
            with self.subTest(f=fi):
                inlined = fi(model_def)
                if isinstance(inlined, tuple):
                    inlined = inlined[0]
                self.assertNotIn("functions {", str(inlined))
                try:
                    oinf = onnxruntime.InferenceSession(inlined.SerializeToString())
                    goti = oinf.run(None, feeds)
                except onnxruntime.capi.onnxruntime_pybind11_state.Fail:
                    if fi is inline_local_functions:
                        # bug in onnx
                        continue
                self.assertEqualArray(got[0], goti[0])
                self.assertEqualArray(got[0], numpy.array([-3], dtype=numpy.float32))

    def test_onnx_inline_subgraph_function2(self, log=False):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N"])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, ["N"])
        one = helper.make_tensor_value_info("one", TensorProto.FLOAT, ["N"])

        graph1 = helper.make_graph([], "then", [], [X])
        graph2 = helper.make_graph([], "else", [], [one])
        g1 = helper.make_graph(
            [
                helper.make_node("Greater", ["X", "one"], ["cond"]),
                helper.make_node(
                    "If", ["cond"], ["Z"], then_branch=graph1, else_branch=graph2
                ),
            ],
            "test",
            [],
            [Z],
        )

        graph1 = helper.make_graph([], "then", [], [X])
        graph2 = helper.make_graph([], "else", [], [one])
        g2 = helper.make_graph(
            [
                helper.make_node("Greater", ["X", "one"], ["cond"]),
                helper.make_node(
                    "If", ["cond"], ["Z"], then_branch=graph1, else_branch=graph2
                ),
            ],
            "test",
            [],
            [Z],
        )

        func_def = helper.make_function(
            "this",
            "fct",
            ["X"],
            ["Z"],
            [
                helper.make_node("Constant", [], ["one"], value_floats=[1.0]),
                helper.make_node("Greater", ["X", "one"], ["cond"]),
                helper.make_node("If", ["cond"], ["Z"], then_branch=g1, else_branch=g2),
            ],
            opset_imports=[helper.make_operatorsetid("", 15)],
        )

        model_def = helper.make_model(
            helper.make_graph(
                [helper.make_node("fct", ["X"], ["Z"], domain="this")], "test", [X], [Z]
            ),
            ir_version=7,
            opset_imports=[
                helper.make_operatorsetid("", 15),
                helper.make_operatorsetid("this", 1),
            ],
            functions=[func_def],
        )

        feeds = {"X": numpy.array([-5], dtype=numpy.float32)}
        oinf = CReferenceEvaluator(model_def)
        got = oinf.run(None, feeds)
        for fi in [onnx_inline_function, inline_local_functions]:
            with self.subTest(f=fi):
                if fi == onnx_inline_function:
                    with contextlib.redirect_stdout(StringIO()):
                        inlined = fi(model_def, verbose=10)
                else:
                    inlined = fi(model_def)
                if isinstance(inlined, tuple):
                    inlined = inlined[0]
                self.assertNotIn("functions {", str(inlined))
                oinf = CReferenceEvaluator(inlined)
                try:
                    oinf = CReferenceEvaluator(inlined)
                    goti = oinf.run(None, feeds)
                except RuntimeError:
                    if fi is inline_local_functions:
                        # bug in onnx
                        continue
                self.assertEqualArray(got[0], goti[0])
                self.assertEqualArray(got[0], numpy.array([1], dtype=numpy.float32))

    @unittest.skipIf(True, reason="bug in onnxruntime")
    def test_onnx_inline_subgraph_function3_fct(self, log=False):
        # subfct
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N"])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, ["N"])
        one = helper.make_tensor_value_info("one", TensorProto.FLOAT, ["N"])

        graph1 = helper.make_graph([], "then", [], [X])
        graph2 = helper.make_graph([], "else", [], [one])
        g1 = helper.make_graph(
            [
                helper.make_node("Greater", ["X", "one"], ["cond"]),
                helper.make_node(
                    "If", ["cond"], ["Z"], then_branch=graph1, else_branch=graph2
                ),
            ],
            "test",
            [],
            [Z],
        )

        graph1 = helper.make_graph([], "then", [], [X])
        graph2 = helper.make_graph([], "else", [], [one])
        g2 = helper.make_graph(
            [
                helper.make_node("Greater", ["X", "one"], ["cond"]),
                helper.make_node(
                    "If", ["cond"], ["Z"], then_branch=graph1, else_branch=graph2
                ),
            ],
            "test",
            [],
            [Z],
        )

        func_def1 = helper.make_function(
            "this",
            "subfct",
            ["X"],
            ["Z"],
            [
                helper.make_node("Constant", [], ["one"], value_floats=[1.0]),
                helper.make_node("Greater", ["X", "one"], ["cond"]),
                helper.make_node("If", ["cond"], ["Z"], then_branch=g1, else_branch=g2),
            ],
            opset_imports=[helper.make_operatorsetid("", 15)],
        )

        # mainfct
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N"])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, ["N"])
        one = helper.make_tensor_value_info("one", TensorProto.FLOAT, ["N"])

        gg1 = helper.make_graph(
            [helper.make_node("subfct", ["X"], ["Z"], domain="this")], "then", [], [Z]
        )
        gg2 = helper.make_graph(
            [
                helper.make_node("subfct", ["X"], ["T"], domain="this"),
                helper.make_node("Neg", ["T"], ["Z"]),
            ],
            "else",
            [],
            [Z],
        )

        func_def2 = helper.make_function(
            "this",
            "mainfct",
            ["X"],
            ["Z"],
            [
                helper.make_node("Constant", [], ["one"], value_floats=[1.0]),
                helper.make_node("Greater", ["X", "one"], ["cond"]),
                helper.make_node(
                    "If", ["cond"], ["Z"], then_branch=gg1, else_branch=gg2
                ),
            ],
            opset_imports=[
                helper.make_operatorsetid("", 15),
                helper.make_operatorsetid("this", 1),
            ],
            ir_version=9,
        )

        model_def = helper.make_model(
            helper.make_graph(
                [helper.make_node("mainfct", ["X"], ["Z"], domain="this")],
                "test",
                [X],
                [Z],
            ),
            ir_version=7,
            opset_imports=[
                helper.make_operatorsetid("", 15),
                helper.make_operatorsetid("this", 1),
            ],
            functions=[func_def1, func_def2],
        )

        import onnxruntime

        feeds = {"X": numpy.array([-5], dtype=numpy.float32)}
        oinf = onnxruntime.InferenceSession(model_def.SerializeToString())
        got = oinf.run(None, feeds)
        for fi in [onnx_inline_function, inline_local_functions]:
            with self.subTest(f=fi):
                inlined = fi(model_def)
                if isinstance(inlined, tuple):
                    inlined = inlined[0]
                self.assertNotIn("functions {", str(inlined))
                oinf2 = onnxruntime.InferenceSession(model_def.SerializeToString())
                oinf2.check_onnx()
                got2 = oinf2.run(feeds)
                self.assertEqualArray(got[0], got2[0])
                oinf3 = onnxruntime.InferenceSession(inlined.SerializeToString())
                oinf3.check_onnx()
                got3 = oinf3.run(feeds)
                self.assertEqualArray(got[0], got3[0])

    def test_inline_model(self):
        import onnxruntime

        model_def = os.path.join(
            os.path.dirname(__file__), "data", "debug_4700-CPUep.onnx"
        )
        oinf = onnxruntime.InferenceSession(model_def)
        feeds = {}
        for i in oinf.get_inputs():
            feeds[i.name] = numpy.random.rand(*i.shape).astype(numpy.float32)
        model = load(model_def)
        got = oinf.run(None, feeds)
        for fi in [onnx_inline_function, inline_local_functions]:
            with self.subTest(f=fi):
                inlined = fi(model)
                if isinstance(inlined, tuple):
                    inlined = inlined[0]
                check_model(inlined)
                # from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
                # print(onnx_simple_text_plot(inlined))
                oinf = onnxruntime.InferenceSession(inlined.SerializeToString())
                goti = oinf.run(None, feeds)
                self.assertEqualArray(got[0], goti[0])

    def test_inline_model_optim(self):
        import onnxruntime

        model_def = load(
            os.path.join(os.path.dirname(__file__), "data", "debug_4700-CPUep.onnx")
        )
        new_output = [o for o in model_def.graph.output if o.name == "addmm_2"]
        assert len(new_output) == 1
        del model_def.graph.output[:]
        model_def.graph.output.extend(new_output)
        oinf = onnxruntime.InferenceSession(model_def.SerializeToString())
        feeds = {}
        for i in oinf.get_inputs():
            feeds[i.name] = numpy.random.rand(*i.shape).astype(numpy.float32)
        got = oinf.run(None, feeds)
        for fi in [onnx_inline_function, inline_local_functions]:
            with self.subTest(f=fi):
                inlined = fi(model_def)
                if isinstance(inlined, tuple):
                    inlined = inlined[0]
                check_model(inlined)
                try:
                    from onnx_array_api.graph_api import GraphBuilder

                    use_builder = True
                except ImportError:
                    use_builder = False
                if use_builder:
                    with open(f"debug.{fi.__name__}.0.onnx", "wb") as f:
                        f.write(inlined.SerializeToString())
                    gr = GraphBuilder(inlined)
                    gr.optimize()
                    inlined = gr.to_onnx()
                    with open(f"debug.{fi.__name__}.1.onnx", "wb") as f:
                        f.write(inlined.SerializeToString())
                oinf = onnxruntime.InferenceSession(inlined.SerializeToString())
                goti = oinf.run(None, feeds)
                self.assertEqualArray(got[0], goti[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
