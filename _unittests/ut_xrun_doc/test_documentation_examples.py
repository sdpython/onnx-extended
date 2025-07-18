import unittest
import warnings
import os
import sys
import importlib
import subprocess
import time
from onnx_extended import __file__ as onnx_extended_file, has_cuda
from onnx_extended.ext_test_case import ExtTestCase, is_windows

VERBOSE = 0
ROOT = os.path.realpath(os.path.abspath(os.path.join(onnx_extended_file, "..", "..")))

try:
    from onnx_extended.ortcy.wrap.ortinf import OrtSession
except ImportError as e:
    msg = "libonnxruntime.so.1.22.0: cannot open shared object file"
    if msg in str(e):
        from onnx_extended.ortcy.wrap import __file__ as loc

        all_files = os.listdir(os.path.dirname(loc))
        warnings.warn(
            f"Unable to find onnxruntime {e!r}, found files in {os.path.dirname(loc)}: "
            f"{all_files}.",
            stacklevel=0,
        )
        OrtSession = None
        here = os.path.dirname(__file__)
    else:
        OrtSession = "OrtSession is not initialized"


def import_source(module_file_path, module_name):
    if not os.path.exists(module_file_path):
        raise FileNotFoundError(module_file_path)
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    if module_spec is None:
        raise FileNotFoundError(
            "Unable to find '{}' in '{}'.".format(module_name, module_file_path)
        )
    module = importlib.util.module_from_spec(module_spec)
    return module_spec.loader.exec_module(module)


class TestDocumentationExamples(ExtTestCase):
    def run_test(self, fold: str, name: str, verbose=0) -> int:
        ppath = os.environ.get("PYTHONPATH", "")
        if not ppath:
            os.environ["PYTHONPATH"] = ROOT
        elif ROOT not in ppath:
            sep = ";" if is_windows() else ":"
            os.environ["PYTHONPATH"] = ppath + sep + ROOT
        perf = time.perf_counter()
        os.environ["UNITTEST_GOING"] = "1"
        try:
            mod = import_source(fold, os.path.splitext(name)[0])
            assert mod is not None
        except FileNotFoundError:
            # try another way
            cmds = [sys.executable, "-u", os.path.join(fold, name)]
            p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            res = p.communicate()
            out, err = res
            st = err.decode("ascii", errors="ignore")
            if st and "Traceback" in st:
                if "No module named 'onnxruntime'" in st:
                    if verbose:
                        print(f"failed: {name!r} due to missing onnxruntime.")
                    return 1
                raise AssertionError(  # noqa: B904
                    "Example '{}' (cmd: {} - exec_prefix='{}') "
                    "failed due to\n{}"
                    "".format(name, cmds, sys.exec_prefix, st)
                )
        dt = time.perf_counter() - perf
        if verbose:
            print(f"{dt:.3f}: run {name!r}")
        return 1

    @classmethod
    def add_test_methods(cls):
        this = os.path.abspath(os.path.dirname(__file__))
        fold = os.path.normpath(os.path.join(this, "..", "..", "_doc", "examples"))
        found = os.listdir(fold)
        for name in found:
            if not name.startswith("plot_") or not name.endswith(".py"):
                continue
            reason = None

            if OrtSession is None and name in {"plot_bench_cypy_ort.py"}:
                reason = "wrong build"

            elif name in {"plot_op_tfidfvectorizer_sparse.py"}:
                if sys.platform in {"darwin", "win32"}:
                    reason = "stuck due to the creation of a secondary process"

            elif not has_cuda() and name in {"plot_op_mul_cuda.py"}:
                reason = "cuda required"

            if reason:

                @unittest.skip(reason)
                def _test_(self, name=name):
                    res = self.run_test(fold, name, verbose=VERBOSE)
                    self.assertTrue(res)

            else:

                def _test_(self, name=name):
                    res = self.run_test(fold, name, verbose=VERBOSE)
                    self.assertTrue(res)

            short_name = os.path.split(os.path.splitext(name)[0])[-1]
            setattr(cls, f"test_{short_name}", _test_)


TestDocumentationExamples.add_test_methods()

if __name__ == "__main__":
    unittest.main(verbosity=2)
