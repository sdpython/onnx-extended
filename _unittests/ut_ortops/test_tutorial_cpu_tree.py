import multiprocessing
import os
import subprocess
import unittest
from typing import Optional, Tuple
import numpy
from onnx import ModelProto
from onnx.helper import make_attribute
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from onnx_extended.ext_test_case import ExtTestCase

try:
    from onnxruntime import InferenceSession, SessionOptions
except ImportError:
    SessionOptions, InferenceSession = None, None


def make_tree(n_features: int, n_trees: int, max_depth: int) -> ModelProto:
    from skl2onnx import to_onnx

    X, y = make_regression(max_depth * 1000, n_features)
    X = X.astype(numpy.float32)
    rf = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth)
    rf.fit(X, y)
    onx = to_onnx(rf, X[:1])
    return onx


def compile_tree(
    llc_exe: str,
    file_name: str,
    onx: ModelProto,
    batch_size: int,
    tree_tile_size: int = 8,
) -> str:
    import treebeard

    compiler_options = treebeard.CompilerOptions(batch_size, tree_tile_size)

    compiler_options.SetNumberOfCores(multiprocessing.cpu_count())
    compiler_options.SetMakeAllLeavesSameDepth(1)
    compiler_options.SetReorderTreesByDepth(True)
    assert 8 < batch_size
    compiler_options.SetPipelineWidth(8)

    with open(file_name, "wb") as f:
        f.write(onx.SerializeToString())
    onnx_model_path = os.path.abspath(file_name)
    tbContext = treebeard.TreebeardContext(onnx_model_path, "", compiler_options)
    tbContext.SetRepresentationType("sparse")
    tbContext.SetInputFiletype("onnx_file")

    llvm_file_path = f"{os.path.splitext(onnx_model_path)[0]}.ll"
    dumped = tbContext.DumpLLVMIR(llvm_file_path)

    if not dumped:
        raise RuntimeError(f"Failed to dump LLVM IR in {llvm_file_path!r}.")

    asm_file_path = f"{os.path.splitext(onnx_model_path)[0]}.s"
    so_file_path = f"{os.path.splitext(onnx_model_path)[0]}.so"

    # Run LLC
    subprocess.run(
        [
            llc_exe,
            llvm_file_path,
            "-O3",
            "-march=x86-64",
            "-mcpu=native",
            "--relocation-model=pic",
            "-o",
            asm_file_path,
        ]
    )

    # Run CLANG
    subprocess.run(
        ["clang", "-shared", asm_file_path, "-fopenmp=libomp", "-o", so_file_path]
    )

    return so_file_path


def make_ort_session(onx: ModelProto, assembly_name: Optional[str] = None) -> Tuple:
    from onnxruntime import InferenceSession, SessionOptions
    from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs as lib_tuto
    from onnx_extended.ortops.optim.cpu import get_ort_ext_libs as lib_optim
    from onnx_extended.ortops.optim.optimize import (
        change_onnx_operator_domain,
        get_node_attribute,
    )

    # baseline
    sess_check = InferenceSession(
        onx.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    # first optimization
    onx2 = change_onnx_operator_domain(
        onx,
        op_type="TreeEnsembleRegressor",
        op_domain="ai.onnx.ml",
        new_op_domain="onnx_extented.ortops.optim.cpu",
        nodes_modes=",".join(
            map(
                lambda s: s.decode("ascii"),
                get_node_attribute(onx.graph.node[0], "nodes_modes").strings,
            )
        ),
    )

    r = lib_optim()
    opts = SessionOptions()
    opts.register_custom_ops_library(r[0])
    sess_opt = InferenceSession(
        onx2.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )

    if assembly_name is None:
        return sess_check, sess_opt, None

    # assembly
    for node in onx.graph.node:
        if node.op_type == "TreeEnsembleRegressor":
            node.domain = "onnx_extented.ortops.tutorial.cpu"
            del node.attribute[:]
            new_add = make_attribute("assembly", assembly_name)
            node.attribute.append(new_add)

    d = onx.opset_import.add()
    d.domain = "onnx_extented.ortops.tutorial.cpu"
    d.version = 1

    r = lib_tuto()
    opts = SessionOptions()
    opts.register_custom_ops_library(r[0])
    sess_assembly = InferenceSession(
        onx.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )

    return sess_check, sess_opt, sess_assembly


class TestOrtOpTutorialCpuTree(ExtTestCase):
    def test_get_ort_ext_libs(self):
        from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

        r = get_ort_ext_libs()
        self.assertEqual(len(r), 1)

    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
    def test_custom_tree_ensemble(self):
        n_features = 3
        batch_size = 10
        onx = make_tree(n_features, 3, 5)
        llc_exe = os.environ.get("TEST_LLC_EXE", "SKIP")
        if llc_exe == "SKIP":
            sessions = make_ort_session(onx)

        elif not os.path.exists(llc_exe):
            raise FileNotFoundError(f"Unable to find {llc_exe}.")
        else:
            assembly_name = compile_tree(
                llc_exe, "custom_tree_ensemble.onnx", onx, batch_size
            )
            sessions = make_ort_session(onx, assembly_name)

        feeds = {"X": numpy.random.randn(batch_size, n_features).astype(numpy.float32)}
        results = []
        for sess in sessions:
            if sess is None:
                continue
            results.append(sess.run(None, feeds)[0])

        self.assertEqualArray(results[0], results[1])
        if len(results) > 2:
            self.assertEqualArray(results[0], results[2])


if __name__ == "__main__":
    # TestOrtOpTutorialCpu().test_dynamic_quantize_linear()
    unittest.main(verbosity=2)
