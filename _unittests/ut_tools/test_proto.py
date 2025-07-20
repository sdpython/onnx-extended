import textwrap
import unittest
from onnx_extended.ext_test_case import ExtTestCase, hide_stdout
from onnx_extended.tools.protoc import parse_proto


class TestProto(ExtTestCase):
    def test_parse_proto(self):
        text = textwrap.dedent(
            """
        message Company {
            string name = 1;
            message Employee {
                string name = 1;
                int32 id = 2;
            }
            repeated Employee employees = 2;
        }
        """
        )
        out = parse_proto(text)
        self.assertIn("std::vector<Employee> employees;", out)
        self.assertIn("struct Employee {", out)

    @hide_stdout()
    def test_onnx_ml_proto(self):
        text = textwrap.dedent(
            """
            message AttributeProto {
                reserved 12, 16 to 19;
                reserved "v";

                enum AttributeType {
                    UNDEFINED = 0;
                    FLOAT = 1;
                    INT = 2;
                    STRING = 3;
                    TENSOR = 4;
                    GRAPH = 5;
                    SPARSE_TENSOR = 11;
                    TYPE_PROTO = 13;
                    FLOATS = 6;
                    INTS = 7;
                    STRINGS = 8;
                    TENSORS = 9;
                    GRAPHS = 10;
                    SPARSE_TENSORS = 12;
                    TYPE_PROTOS = 14;
                }

                string name = 1;
                string ref_attr_name = 21;
                string doc_string = 13;
                AttributeType type = 20;
                optional float f = 2;
                optional int64 i = 3;
                optional bytes s = 4;
                optional TensorProto t = 5;
                optional GraphProto g = 6;
                optional SparseTensorProto sparse_tensor = 22;
                optional TypeProto tp = 14;
                repeated float floats = 7;
                repeated int64 ints = 8;
                repeated bytes strings = 9;
                repeated TensorProto tensors = 10;
                repeated GraphProto graphs = 11;
                repeated SparseTensorProto sparse_tensors = 23;
                repeated TypeProto type_protos = 15;
            }

            message ValueInfoProto {
                string name = 1;
                optional TypeProto type = 2;
                string doc_string = 3;
                repeated StringStringEntryProto metadata_props = 4;
            }

            message NodeProto {
                repeated string input = 1;
                repeated string output = 2;
                string name = 3;
                string op_type = 4;
                string domain = 7;
                string overload = 8;
                repeated AttributeProto attribute = 5;
                string doc_string = 6;
                repeated StringStringEntryProto metadata_props = 9;
                repeated NodeDeviceConfigurationProto device_configurations = 10;
            }

            message IntIntListEntryProto {
                int64 key = 1;
                repeated int64 value = 2;
            }

            message NodeDeviceConfigurationProto {
                string configuration_id = 1;
                repeated ShardingSpecProto sharding_spec = 2;
                optional int32 pipeline_stage = 3;
            }

            message ShardingSpecProto {
                string tensor_name = 1;
                repeated int64 device = 2;
                repeated IntIntListEntryProto index_to_device_group_map = 3;
                repeated ShardedDimProto sharded_dim = 4;
            }

            message ShardedDimProto {
                int64 axis = 1;
                repeated SimpleShardedDimProto simple_sharding = 2;
            }

            message SimpleShardedDimProto {
                oneof dim {
                    int64 dim_value = 1;
                    string dim_param = 2;
                }
                int64 num_shards = 3;
            }

            message ModelProto {
                int64 ir_version = 1;
                repeated OperatorSetIdProto opset_import = 8;
                string producer_name = 2;
                string producer_version = 3;
                string domain = 4;
                optional int64 model_version = 5;
                string doc_string = 6;
                GraphProto graph = 7;
                repeated StringStringEntryProto metadata_props = 14;
                repeated TrainingInfoProto training_info = 20;
                repeated FunctionProto functions = 25;
                repeated DeviceConfigurationProto configuration = 26;
            }

            message DeviceConfigurationProto {
                string name = 1;
                int32 num_devices = 2;
                repeated string device = 3;
            }

            message StringStringEntryProto {
                string key = 1;
                string value = 2;
            }

            message TensorAnnotation {
                string tensor_name = 1;
                repeated StringStringEntryProto quant_parameter_tensor_names = 2;
            }

            message GraphProto {
                repeated NodeProto node = 1;
                string name = 2;
                repeated TensorProto initializer = 5;
                repeated SparseTensorProto sparse_initializer = 15;
                string doc_string = 10;
                repeated ValueInfoProto input = 11;
                repeated ValueInfoProto output = 12;
                repeated ValueInfoProto value_info = 13;
                repeated TensorAnnotation quantization_annotation = 14;
                repeated StringStringEntryProto metadata_props = 16;
                reserved 3, 4, 6 to 9;
                reserved "ir_version", "producer_version", "producer_tag", "domain";
            }

            message TensorProto {
                enum DataType {
                    UNDEFINED = 0;
                    FLOAT = 1;
                    UINT8 = 2;
                    INT8 = 3;
                    UINT16 = 4;
                    INT16 = 5;
                    INT32 = 6;
                    INT64 = 7;
                    STRING = 8;
                    BOOL = 9;
                    FLOAT16 = 10;
                    DOUBLE = 11;
                    UINT32 = 12;
                    UINT64 = 13;
                    COMPLEX64 = 14;
                    COMPLEX128 = 15;
                    BFLOAT16 = 16;
                    FLOAT8E4M3FN = 17;
                    FLOAT8E4M3FNUZ = 18;
                    FLOAT8E5M2 = 19;
                    FLOAT8E5M2FNUZ = 20;
                    UINT4 = 21;
                    INT4 = 22;
                    FLOAT4E2M1 = 23;
                    FLOAT8E8M0 = 24;
                }

                repeated int64 dims = 1;
                int32 data_type = 2;
                optional Segment segment = 3;
                repeated float float_data = 4 [packed = true];
                repeated int32 int32_data = 5 [packed = true];
                repeated bytes string_data = 6;
                repeated int64 int64_data = 7 [packed = true];
                string name = 8;
                string doc_string = 12;
                bytes raw_data = 9;
                repeated StringStringEntryProto external_data = 13;
                optional DataLocation data_location = 14;
                repeated double double_data = 10 [packed = true];
                repeated uint64 uint64_data = 11 [packed = true];
                repeated StringStringEntryProto metadata_props = 16;
            }

            message SparseTensorProto {
                TensorProto values = 1;
                TensorProto indices = 2;
                repeated int64 dims = 3;
            }

            message TensorShapeProto {
                message Dimension {
                    oneof value {
                        int64 dim_value = 1;
                        string dim_param = 2;
                    };
                    string denotation = 3;
                };
                repeated Dimension dim = 1;
            }

            message TypeProto {
                message Tensor {
                    int32 elem_type = 1;
                    TensorShapeProto shape = 2;
                }

                message Sequence {
                    TypeProto elem_type = 1;
                };

                message Map {
                    int32 key_type = 1;
                    TypeProto value_type = 2;
                };

                message Optional {
                    TypeProto elem_type = 1;
                };

                message SparseTensor {
                    int32 elem_type = 1;
                    TensorShapeProto shape = 2;
                }

                message Opaque {
                    string domain = 1;
                    string name = 2;
                }

                oneof value {
                    Tensor tensor_type = 1;
                    Sequence sequence_type = 4;
                    Map map_type = 5;
                    Optional optional_type = 9;
                    SparseTensor sparse_tensor_type = 8;
                    Opaque opaque_type = 7;
                }

                string denotation = 6;
            }

            message OperatorSetIdProto {
                string domain = 1;
                int64 version = 2;
            }

            enum OperatorStatus {
                EXPERIMENTAL = 0;
                STABLE = 1;
            }

            message FunctionProto {
                string name = 1;
                reserved 2;
                reserved "since_version";
                reserved 3;
                reserved "status";
                repeated string input = 4;
                repeated string output = 5;
                repeated string attribute = 6;
                repeated AttributeProto attribute_proto = 11;
                repeated NodeProto node = 7;
                string doc_string = 8;
                repeated OperatorSetIdProto opset_import = 9;
                string domain = 10;
                string overload = 13;
                repeated ValueInfoProto value_info = 12;
                repeated StringStringEntryProto metadata_props = 14;
            }
        """
        )
        out = parse_proto(text)
        self.assertIn("struct SparseTensorProto {", out)
        print(out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
