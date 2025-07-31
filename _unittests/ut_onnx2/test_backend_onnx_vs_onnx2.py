import os
import re
import shutil
import unittest
import onnx
import onnx_extended.onnx2 as onnx2
from onnx.backend.test.loader import load_model_tests
from onnx_extended.ext_test_case import ExtTestCase


class TestOnnxVsOnnx2(ExtTestCase):
    regs = [(re.compile("(adagrad|adam)"), "training")]

    @classmethod
    def filter_out(cls, model_name):
        for reg, reason in cls.regs:
            if reg.search(model_name):
                return reason
        return False

    @classmethod
    def add_test_methods(cls):
        tests = load_model_tests(kind="node")
        for test in tests:
            model = os.path.join(test.model_dir, "model.onnx")
            if not os.path.exists(model):
                continue
            reason = cls.filter_out(model)
            if reason:
                @unittest.skip(reason)
                def _test_(self, name=model):
                    self.run_test(name)
            else:
                def _test_(self, name=model):
                    self.run_test(name)

            short_name = os.path.split(test.model_dir)[-1].replace("test_", "")
            setattr(cls, f"test_vs_{short_name}", _test_)

    def run_test(self, model_name):
        onx = onnx.load(model_name)
        try:
            onx2 = onnx2.load(model_name)
        except RuntimeError as e:
            name = self.get_dump_file(f"{os.path.split(os.path.split(model_name)[0])[-1]}.onnx")
            shutil.copy(model_name, name)
            with open(name + ".txt", "w") as f:
                f.write(str(onx))
            with open(model_name, "rb") as f:
                content = f.read()
            rows = []
            for i in range(0, len(content), 10):
                rows.append(f"{i:03d}: {content[i:min(i+10,len(content))]}")
            msg = "\n".join(rows)            
            raise AssertionError(f"Unable to load {model_name!r} with onnx2.\n---\n{msg}") from e
        self.assertEqual(len(onx.graph.node), len(onx2.graph.node))

        # compare the serialized string with onnx2 format
        with self.subTest(fmt="onnx2"):
            s = onx.SerializeToString()
            onx_onnx2 = onnx2.ModelProto()
            onx_onnx2.ParseFromString(s)
            b = onx_onnx2.SerializeToString()
            a = onx2.SerializeToString()
            if a != b:
                short_name = os.path.splitext(
                    os.path.split(os.path.split(model_name)[0])[-1]
                )[0]
                f1 = self.get_dump_file(short_name + ".original2.onnx")
                with open(f1, "wb") as f:
                    f.write(a)
                with open(f1 + ".txt", "w") as f:
                    f.write(str(onx2))
                f2 = self.get_dump_file(short_name + ".original2_to_onnx.onnx")
                with open(f2, "wb") as f:
                    f.write(b)
                with open(f2 + ".txt", "w") as f:
                    f.write(str(onx_onnx2))
            self.assertEqual(a, b)

        # compare the serialized string with onnx format
        with self.subTest(fmt="onnx"):
            s2 = onx2.SerializeToString()
            onx2_onnx = onnx.ModelProto()
            onx2_onnx.ParseFromString(s2)
            a = onx.SerializeToString()
            b = onx2_onnx.SerializeToString()
            if a != b:
                short_name = os.path.splitext(
                    os.path.split(os.path.split(model_name)[0])[-1]
                )[0]
                f1 = self.get_dump_file(short_name + ".original.onnx")
                with open(f1, "wb") as f:
                    f.write(a)
                with open(f1 + ".txt", "w") as f:
                    f.write(str(onx))
                f2 = self.get_dump_file(short_name + ".original_to_onnx2.onnx")
                with open(f2, "wb") as f:
                    f.write(b)
                with open(f2 + ".txt", "w") as f:
                    f.write(str(onx2_onnx))
            self.assertEqual(a, b)


TestOnnxVsOnnx2.add_test_methods()

if __name__ == "__main__":
    unittest.main(verbosity=2, exit=False)
