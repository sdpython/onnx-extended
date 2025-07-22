import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx_extended.ext_test_case import ExtTestCase
import onnx_extended.onnx2.cpu._onnx2py as onnx2


class TestOnnx2(ExtTestCase):
    def test_onnx2_tensorproto(self):
        a = onh.from_array(
            np.array([[10, 20, 30, 40, 50, 60]]).reshape((2, 3, 1, 1)).astype(np.int16),
            name="AAAê€AAA",
        )
        a.doc_string = "help"
        s = a.SerializeToString()
        self.assertEqual(onnx2.utils_onnx2_read_varint64(b"\xac\x02"), (300, 2))
        i = onnx2.utils_onnx2_read_varint64(s[0:])
        self.assertEqual(i, (8, 1))
        t2 = onnx2.TensorProto()
        t2.ParseFromString(s)
        self.assertEqual(a.name, t2.name)
        self.assertEqual(a.doc_string, t2.doc_string)
        self.assertEqual(tuple(a.dims), tuple(t2.dims))
        self.assertEqual(a.data_type, int(t2.data_type))
        self.assertEqual(a.raw_data, t2.raw_data)
        a.raw_data = b"012345"
        self.assertEqual(a.raw_data, b"012345")

        # way back
        s2 = t2.SerializeToString()
        t = onnx.TensorProto()
        t.ParseFromString(s2)
        self.assertEqual(t.raw_data, t2.raw_data)
        self.assertEqual(t.name, t2.name)
        self.assertEqual(tuple(t.dims), tuple(t2.dims))

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
        self.assertEqual(onnx2.utils_onnx2_read_varint64(b"\xac\x02"), (300, 2))
        i = onnx2.utils_onnx2_read_varint64(s[0:])
        self.assertEqual(i, (8, 1))
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
        del t2.metadata_props[:]

    def test_string_string_entry_proto(self):
        p = onnx.StringStringEntryProto()
        p.key = "hk"
        p.value = "zoo"
        s = p.SerializeToString()
        p2 = onnx2.StringStringEntryProto()
        p2.ParseFromString(s)
        self.assertEqual((p2.key, p2.value), (p.key, p.value))
        # way back
        s2 = p2.SerializeToString()
        p = onnx.StringStringEntryProto()
        p.ParseFromString(s2)
        self.assertEqual((p2.key, p2.value), (p.key, p.value))

    def test_tensor_shape_proto(self):
        vts = oh.make_tensor_value_info(
            "iname",
            onnx.TensorProto.FLOAT,
            (4, "dyndyn"),
            "hellohello",
            ["DDDDD1", "DDDD2"],
        )
        ts = vts.type.tensor_type.shape
        bin = ts.SerializeToString()
        ts2 = onnx2.TensorShapeProto()
        ts2.ParseFromString(bin)
        self.assertEqual(len(ts.dim), len(ts2.dim))
        for d1, d2 in zip(ts.dim, ts2.dim):
            self.assertEqual(d1.dim_value, d2.dim_value or 0)
            self.assertEqual(d1.dim_param, d2.dim_param)
            self.assertEqual(d1.denotation, d2.denotation)
        # way back
        s2 = ts2.SerializeToString()
        onnx2.TensorShapeProto().ParseFromString(s2)
        ts = onnx.TensorShapeProto()
        ts.ParseFromString(s2)
        self.assertEqual(len(ts.dim), len(ts2.dim))
        for d1, d2 in zip(ts.dim, ts2.dim):
            self.assertEqual(d1.dim_value, d2.dim_value or 0)
            self.assertEqual(d1.dim_param, d2.dim_param)
            self.assertEqual(d1.denotation, d2.denotation)

    def test_operator_set_id(self):
        p = oh.make_opsetid("ai.onnx.ml", 5)
        s = p.SerializeToString()
        p2 = onnx2.OperatorSetIdProto()
        p2.ParseFromString(s)
        self.assertEqual((p2.domain, p2.version), (p.domain, p.version))
        # way back
        s2 = p2.SerializeToString()
        p = onnx.OperatorSetIdProto()
        p.ParseFromString(s2)
        self.assertEqual((p2.domain, p2.version), (p.domain, p.version))

    def test_operator_set_id_negative(self):
        p = oh.make_opsetid("ai.onnx.ml", -7)
        s = p.SerializeToString()
        p0 = onnx.OperatorSetIdProto()
        p0.ParseFromString(s)
        self.assertEqual((p0.domain, p0.version), (p.domain, p.version))
        p2 = onnx2.OperatorSetIdProto()
        p2.ParseFromString(s)
        self.assertEqual((p2.domain, p2.version), (p.domain, p.version))
        # way back
        s2 = p2.SerializeToString()
        p = onnx.OperatorSetIdProto()
        p.ParseFromString(s2)
        self.assertEqual((p2.domain, p2.version), (p.domain, p.version))

    def test_tensor_proto_double_data(self):
        p = onnx.TensorProto()
        p.name = "test"
        p.dims.extend([2])
        p.double_data.extend((4.0, 5.0))
        p.data_type = onnx.TensorProto.DOUBLE
        s = p.SerializeToString()

        p2 = onnx2.TensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.double_data), tuple(p2.double_data))

        s2 = p2.SerializeToString()
        p0 = onnx.TensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(tuple(p0.double_data), tuple(p2.double_data))
        self.assertEqual(s, s0)

    def test_tensor_proto_float_data(self):
        p = onnx.TensorProto()
        p.name = "test"
        p.dims.extend([2])
        p.float_data.extend((4.0, 5.0))
        p.data_type = onnx.TensorProto.FLOAT
        s = p.SerializeToString()

        p2 = onnx2.TensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.float_data), tuple(p2.float_data))

        s2 = p2.SerializeToString()
        p0 = onnx.TensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(tuple(p0.float_data), tuple(p2.float_data))
        self.assertEqual(s, s0)

    def test_tensor_proto_int32_data(self):
        p = onnx.TensorProto()
        p.name = "test"
        p.dims.extend([7])
        p.int32_data.extend((4, 5, 6, 7, 8, 9, 10))
        p.data_type = onnx.TensorProto.INT32
        s = p.SerializeToString()

        p2 = onnx2.TensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.int32_data), tuple(p2.int32_data))

        s2 = p2.SerializeToString()
        p0 = onnx.TensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(tuple(p0.int32_data), tuple(p2.int32_data))
        self.assertEqual(s, s0)

    def test_tensor_proto_int64_data(self):
        p = onnx.TensorProto()
        p.name = "test"
        p.dims.extend([2])
        p.int64_data.extend((4, 5))
        p.data_type = onnx.TensorProto.INT64
        s = p.SerializeToString()

        p2 = onnx2.TensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.int64_data), tuple(p2.int64_data))

        s2 = p2.SerializeToString()
        p0 = onnx.TensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(tuple(p0.int64_data), tuple(p2.int64_data))
        self.assertEqual(s, s0)

    def test_tensor_proto_uint64_data(self):
        p = onnx.TensorProto()
        p.name = "test"
        p.dims.extend([2])
        p.uint64_data.extend((4, 5))
        p.data_type = onnx.TensorProto.UINT64
        s = p.SerializeToString()

        p2 = onnx2.TensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.uint64_data), tuple(p2.uint64_data))

        s2 = p2.SerializeToString()
        p0 = onnx.TensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(tuple(p0.uint64_data), tuple(p2.uint64_data))
        self.assertEqual(s, s0)

    def test_tensor_proto_uint64_data_reverse(self):
        p = onnx2.TensorProto()
        p.name = "test"
        p.dims.extend([2])
        p.uint64_data.extend((4, 5))
        p.data_type = onnx.TensorProto.UINT64
        s = p.SerializeToString()

        p2 = onnx.TensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.uint64_data), tuple(p2.uint64_data))

        s2 = p2.SerializeToString()
        p0 = onnx2.TensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(tuple(p0.uint64_data), tuple(p2.uint64_data))
        self.assertEqual(s, s0)

    def test_tensor_proto_string_data(self):
        p = onnx.TensorProto()
        p.name = "test"
        p.dims.extend([2])
        p.string_data.extend((b"s4", b"s5"))
        p.data_type = onnx.TensorProto.STRING
        s = p.SerializeToString()

        p2 = onnx2.TensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.string_data), tuple(p2.string_data))

        s2 = p2.SerializeToString()
        p0 = onnx.TensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(tuple(p0.string_data), tuple(p.string_data))
        self.assertEqual(s, s0)

    def test_tensor_proto_string_data_reverse(self):
        p = onnx2.TensorProto()
        p.name = "test"
        p.dims.extend([2])
        p.string_data.extend((b"s4", b"s5"))
        p.data_type = onnx.TensorProto.STRING
        s = p.SerializeToString()

        p2 = onnx.TensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.string_data), tuple(p2.string_data))

        s2 = p2.SerializeToString()
        p0 = onnx2.TensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(tuple(p0.string_data), tuple(p.string_data))
        self.assertEqual(s, s0)

    def test_sparse_tensor_proto(self):
        dense_shape = [3, 3]
        sparse_values = [1.764052391052246, 0.40015721321105957, 0.978738009929657]
        values_tensor = oh.make_tensor(
            name="sparse_values",
            data_type=onnx.TensorProto.FLOAT,
            dims=[len(sparse_values)],
            vals=np.array(sparse_values).astype(np.float32),
            raw=False,
        )

        linear_indices = [2, 3, 5]
        indices_tensor = oh.make_tensor(
            name="indices",
            data_type=onnx.TensorProto.INT64,
            dims=[len(linear_indices)],
            vals=np.array(linear_indices).astype(np.int64),
            raw=False,
        )
        p = oh.make_sparse_tensor(values_tensor, indices_tensor, dense_shape)
        s = p.SerializeToString()
        self.assertEqual(p.__class__.__name__, "SparseTensorProto")

        p2 = onnx2.SparseTensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))

        s2 = p2.SerializeToString()
        p0 = onnx.SparseTensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(s, s0)

    def test_tensor_annotation(self):
        p = onnx.TensorAnnotation()
        p.tensor_name = "T"
        e = p.quant_parameter_tensor_names.add()
        e.key = "K"
        e.value = "V"
        e = p.quant_parameter_tensor_names.add()
        e.key = "K2"
        e.value = "V2"

        s = p.SerializeToString()
        p2 = onnx2.TensorAnnotation()
        p2.ParseFromString(s)
        self.assertEqual(p.tensor_name, p2.tensor_name)

        s2 = p2.SerializeToString()
        p0 = onnx.TensorAnnotation()
        p0.ParseFromString(s2)
        self.assertEqual(p0.tensor_name, p2.tensor_name)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_tensor_annotation_reverse(self):
        p = onnx2.TensorAnnotation()
        p.tensor_name = "T"
        e = p.quant_parameter_tensor_names.add()
        e.key = "K"
        e.value = "V"
        e = p.quant_parameter_tensor_names.add()
        e.key = "K2"
        e.value = "V2"

        s = p.SerializeToString()
        p2 = onnx.TensorAnnotation()
        p2.ParseFromString(s)
        self.assertEqual(p.tensor_name, p2.tensor_name)

        s2 = p2.SerializeToString()
        p0 = onnx2.TensorAnnotation()
        p0.ParseFromString(s2)
        self.assertEqual(p0.tensor_name, p2.tensor_name)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_int_int_list_entry_proto(self):
        p = onnx.IntIntListEntryProto()
        p.key = 1
        p.value.extend([3, 4])

        s = p.SerializeToString()
        p2 = onnx2.IntIntListEntryProto()
        p2.ParseFromString(s)
        self.assertEqual(p.key, p2.key)

        s2 = p2.SerializeToString()
        p0 = onnx.IntIntListEntryProto()
        p0.ParseFromString(s2)
        self.assertEqual(p0.key, p2.key)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_int_int_list_entry_proto_reverse(self):
        p = onnx2.IntIntListEntryProto()
        p.key = 1
        p.value.extend([3, 4])

        s = p.SerializeToString()
        p2 = onnx.IntIntListEntryProto()
        p2.ParseFromString(s)
        self.assertEqual(p.key, p2.key)

        s2 = p2.SerializeToString()
        p0 = onnx2.IntIntListEntryProto()
        p0.ParseFromString(s2)
        self.assertEqual(p0.key, p2.key)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_device_configuration_proto(self):
        for x, x2 in [(onnx, onnx2), (onnx2, onnx)]:
            with self.subTest(start=x.__name__):
                p = x.DeviceConfigurationProto()
                p.name = "R"
                p.num_devices = 3
                p.device.extend(["T3", "G4"])

                s = p.SerializeToString()
                p2 = x2.DeviceConfigurationProto()
                p2.ParseFromString(s)

                s2 = p2.SerializeToString()
                p0 = x.DeviceConfigurationProto()
                p0.ParseFromString(s2)
                self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_simple_shared_dim_proto(self):
        for x, x2 in [(onnx, onnx2), (onnx2, onnx)]:
            with self.subTest(start=x.__name__):
                p = x.SimpleShardedDimProto()
                p.dim_value = 3
                # p.dim_param = "rt"
                p.num_shards = 4
                self.assertEqual(p.dim_value, 3)
                self.assertEqual(p.dim_param, "")

                s = p.SerializeToString()
                p2 = x2.SimpleShardedDimProto()
                p2.ParseFromString(s)

                s2 = p2.SerializeToString()
                p0 = x.SimpleShardedDimProto()
                p0.ParseFromString(s2)
                self.assertEqual(p.SerializeToString(), p0.SerializeToString())

        for x, x2 in [(onnx, onnx2), (onnx2, onnx)]:
            with self.subTest(start=x.__name__):
                p = x.SimpleShardedDimProto()
                # p.dim_value = 3
                p.dim_param = "rt"
                p.num_shards = 4
                self.assertIn(p.dim_value, (0, None))
                self.assertEqual(p.dim_param, "rt")

                s = p.SerializeToString()
                p2 = x2.SimpleShardedDimProto()
                p2.ParseFromString(s)

                s2 = p2.SerializeToString()
                p0 = x.SimpleShardedDimProto()
                p0.ParseFromString(s2)
                self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_sharded_dim_proto(self):
        for x, x2 in [(onnx, onnx2), (onnx2, onnx)]:
            with self.subTest(start=x.__name__):
                p = x.ShardedDimProto()
                p.axis = 3
                a = p.simple_sharding.add()
                a.dim_value = 4
                a.num_shards = 5

                s = p.SerializeToString()
                p2 = x2.ShardedDimProto()
                p2.ParseFromString(s)

                s2 = p2.SerializeToString()
                p0 = x.ShardedDimProto()
                p0.ParseFromString(s2)
                self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_sharding_spec_proto(self):
        for x, x2 in [(onnx, onnx2), (onnx2, onnx)]:
            with self.subTest(start=x.__name__):
                p = x.ShardingSpecProto()
                p.tensor_name = "erty"
                p.device.extend([4, 5])
                a = p.index_to_device_group_map.add()
                a.key = 10
                a.value.extend([6, 7])
                a = p.index_to_device_group_map.add()
                a.key = 11
                a.value.extend([61, 71])
                b = p.sharded_dim.add()
                b.axis = 3
                c = b.simple_sharding.add()
                c.dim_value = 4
                c.num_shards = 5

                s = p.SerializeToString()
                p2 = x2.ShardingSpecProto()
                p2.ParseFromString(s)

                s2 = p2.SerializeToString()
                p0 = x.ShardingSpecProto()
                p0.ParseFromString(s2)
                self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_node_device_configuration_proto(self):
        for x, x2 in [(onnx, onnx2), (onnx2, onnx)]:
            with self.subTest(start=x.__name__):
                p = x.NodeDeviceConfigurationProto()
                p.configuration_id = "cid"
                p.pipeline_stage = 5

                ps = p.sharding_spec.add()
                ps.tensor_name = "erty"
                ps.device.extend([4, 5])
                a = ps.index_to_device_group_map.add()
                a.key = 10
                a.value.extend([6, 7])
                a = ps.index_to_device_group_map.add()
                a.key = 11
                a.value.extend([61, 71])
                b = ps.sharded_dim.add()
                b.axis = 3
                c = b.simple_sharding.add()
                c.dim_value = 4
                c.num_shards = 5

                s = p.SerializeToString()
                p2 = x2.NodeDeviceConfigurationProto()
                p2.ParseFromString(s)

                s2 = p2.SerializeToString()
                p0 = x.NodeDeviceConfigurationProto()
                p0.ParseFromString(s2)
                self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_tensor_proto_data_type(self):
        self.assertEqual(onnx2.TensorProto.UNDEFINED, onnx.TensorProto.UNDEFINED)
        self.assertEqual(onnx2.TensorProto.FLOAT, onnx.TensorProto.FLOAT)
        self.assertEqual(onnx2.TensorProto.UINT8, onnx.TensorProto.UINT8)
        self.assertEqual(onnx2.TensorProto.INT8, onnx.TensorProto.INT8)
        self.assertEqual(onnx2.TensorProto.UINT16, onnx.TensorProto.UINT16)
        self.assertEqual(onnx2.TensorProto.INT16, onnx.TensorProto.INT16)
        self.assertEqual(onnx2.TensorProto.INT32, onnx.TensorProto.INT32)
        self.assertEqual(onnx2.TensorProto.INT64, onnx.TensorProto.INT64)
        self.assertEqual(onnx2.TensorProto.STRING, onnx.TensorProto.STRING)
        self.assertEqual(onnx2.TensorProto.BOOL, onnx.TensorProto.BOOL)
        self.assertEqual(onnx2.TensorProto.FLOAT16, onnx.TensorProto.FLOAT16)
        self.assertEqual(onnx2.TensorProto.DOUBLE, onnx.TensorProto.DOUBLE)
        self.assertEqual(onnx2.TensorProto.UINT32, onnx.TensorProto.UINT32)
        self.assertEqual(onnx2.TensorProto.UINT64, onnx.TensorProto.UINT64)
        self.assertEqual(onnx2.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX64)
        self.assertEqual(onnx2.TensorProto.COMPLEX128, onnx.TensorProto.COMPLEX128)
        self.assertEqual(onnx2.TensorProto.BFLOAT16, onnx.TensorProto.BFLOAT16)
        self.assertEqual(onnx2.TensorProto.FLOAT8E4M3FN, onnx.TensorProto.FLOAT8E4M3FN)
        self.assertEqual(
            onnx2.TensorProto.FLOAT8E4M3FNUZ, onnx.TensorProto.FLOAT8E4M3FNUZ
        )
        self.assertEqual(onnx2.TensorProto.FLOAT8E5M2, onnx.TensorProto.FLOAT8E5M2)
        self.assertEqual(
            onnx2.TensorProto.FLOAT8E5M2FNUZ, onnx.TensorProto.FLOAT8E5M2FNUZ
        )
        self.assertEqual(onnx2.TensorProto.UINT4, onnx.TensorProto.UINT4)
        self.assertEqual(onnx2.TensorProto.INT4, onnx.TensorProto.INT4)
        self.assertEqual(onnx2.TensorProto.FLOAT4E2M1, onnx.TensorProto.FLOAT4E2M1)
        self.assertEqual(onnx2.TensorProto.FLOAT8E8M0, onnx.TensorProto.FLOAT8E8M0)

        for k in dir(onnx.TensorProto):
            v = getattr(onnx.TensorProto, k)
            if isinstance(v, int):
                with self.subTest(attr=k):
                    self.assertEqual(v, getattr(onnx2.TensorProto, k))

    def test_type_proto_tensor_type(self):
        t = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["a", "b", "c"])
        p = t.type

        s = p.SerializeToString()
        p2 = onnx2.TypeProto()
        p2.ParseFromString(s)

        s2 = p2.SerializeToString()
        p0 = onnx.TypeProto()
        p0.ParseFromString(s2)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_type_proto_tensor_type_reverse(self):
        p = onnx2.TypeProto()
        p.tensor_type=onnx2.TypeProto.Tensor()
        p.tensor_type.elem_type= onnx2.TensorProto.FLOAT
        p.tensor_type.shape.dim.add().dim_param="a"
        p.tensor_type.shape.dim.add().dim_param="b"
        p.tensor_type.shape.dim.add().dim_param="c"
        
        s = p.SerializeToString()
        p2 = onnx.TypeProto()
        p2.ParseFromString(s)

        s2 = p2.SerializeToString()
        p0 = onnx2.TypeProto()
        p0.ParseFromString(s2)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

if __name__ == "__main__":
    unittest.main(verbosity=2)
