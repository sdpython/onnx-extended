import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx_extended.ext_test_case import ExtTestCase
import onnx_extended.validation.cpu._validation as onnx2


class TestOnnx2(ExtTestCase):
    def test_onnx2_tensorproto(self):
        a = onh.from_array(
            np.array([[10, 20, 30, 40, 50, 60]]).reshape((2, 3, 1, 1)).astype(np.int16),
            name="AAAê€AAA",
        )
        a.doc_string = "help"
        s = a.SerializeToString()
        self.assertEqual(onnx2.utils_onnx2_read_varint64(b"\xac\x02"), (150, 2))
        i = onnx2.utils_onnx2_read_varint64(s[0:])
        self.assertEqual(i, (4, 1))
        t2 = onnx2.TensorProto()
        t2.ParseFromString(s)
        self.assertEqual(a.name, t2.name)
        self.assertEqual(a.doc_string, t2.doc_string)
        self.assertEqual(tuple(a.dims), tuple(t2.dims))
        self.assertEqual(a.data_type, int(t2.data_type))
        self.assertEqual(a.raw_data, t2.raw_data)
        a.raw_data = b"012345"
        self.assertEqual(a.raw_data, b"012345")

    def test_onnx2_tensorproto_metadata(self):
        a = onh.from_array(
            np.array([[10, 20, 30, 40, 50, 60]]).reshape((2, 3, 1, 1)).astype(np.int16),
            name="AAAê€AAA",
        )
        a.doc_string = "help"
        entry = a.metadata_props.add()
        entry.key = "k1"
        entry.value = "vv1"
        entry = a.metadata_props.add()
        entry.key = "k2"
        entry.value = "vv2"
        s = a.SerializeToString()
        self.assertEqual(onnx2.utils_onnx2_read_varint64(b"\xac\x02"), (150, 2))
        i = onnx2.utils_onnx2_read_varint64(s[0:])
        self.assertEqual(i, (4, 1))
        t2 = onnx2.TensorProto()
        t2.ParseFromString(s)
        self.assertEqual(a.name, t2.name)
        self.assertEqual(a.doc_string, t2.doc_string)
        self.assertEqual(tuple(a.dims), tuple(t2.dims))
        self.assertEqual(a.data_type, int(t2.data_type))
        self.assertEqual(a.raw_data, t2.raw_data)
        a.raw_data = b"012345"
        self.assertEqual(a.raw_data, b"012345")
        kv = list(t2.metadata_props)
        self.assertEqual(len(kv), 2)
        self.assertEqual(
            [kv[0].key, kv[0].value, kv[1].key, kv[1].value], ["k1", "vv1", "k2", "vv2"]
        )

    def test_string_string_entry_proto(self):
        p = onnx.StringStringEntryProto()
        p.key = "hk"
        p.value = "zoo"
        s = p.SerializeToString()
        p2 = onnx2.StringStringEntryProto()
        p2.ParseFromString(s)
        self.assertEqual((p2.key, p2.value), (p.key, p.value))

    def test_tensor_shape_proto(self):
        vts = oh.make_tensor_value_info(
            "iname", onnx.TensorProto.FLOAT, (4, "dyn"), "hello", ["D1", "D2"]
        )
        ts = vts.type.tensor_type.shape
        bin = ts.SerializeToString()
        ts2 = onnx2.TensorShapeProto()
        ts2.ParseFromString(bin)
        self.assertEqual(len(ts.dim), len(ts2.dim))
        for d1, d2 in zip(ts.dim, ts2.dim):
            self.assertEqual(d1.dim_value, d2.dim_value)
            self.assertEqual(d1.dim_param, d2.dim_param)
            self.assertEqual(d1.denotation, d2.denotation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
