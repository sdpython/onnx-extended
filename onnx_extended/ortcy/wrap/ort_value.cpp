#include "ort_value.h"
#include "helper.h"
#include "ortapi.h"
#include "ortapi_inline.h"

namespace ortapi {

DLDataType GetDlpackDataType(OrtValueType *value) {

  size_t elem_type, n_dims;
  OrtTensorTypeAndShapeInfo *info;
  ThrowOnError(GetOrtApi()->GetTensorTypeAndShape((OrtValue *)value, &info));
  ThrowOnError(GetOrtApi()->GetTensorElementType(info, &elem_type));
  GetOrtApi()->ReleaseTensorTypeAndShapeInfo(info);

  DLDataType dtype;
  dtype.lanes = 1;
  switch (elem_type) {
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    dtype.code = DLDataTypeCode::kDLFloat;
    dtype.bits = sizeof(double);
    break;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    dtype.code = DLDataTypeCode::kDLFloat;
    dtype.bits = sizeof(float);
    break;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    dtype.code = DLDataTypeCode::kDLInt;
    dtype.bits = sizeof(int8_t);
    break;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    dtype.code = DLDataTypeCode::kDLInt;
    dtype.bits = sizeof(int16_t);
    break;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    dtype.code = DLDataTypeCode::kDLInt;
    dtype.bits = sizeof(int);
    break;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    dtype.code = DLDataTypeCode::kDLInt;
    dtype.bits = sizeof(int64_t);
    break;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    dtype.code = DLDataTypeCode::kDLFloat;
    dtype.bits = sizeof(uint16_t); // sizeof(half);
    break;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    dtype.code = DLDataTypeCode::kDLBool;
    dtype.bits = sizeof(bool);
    break;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    dtype.code = DLDataTypeCode::kDLUInt;
    dtype.bits = sizeof(uint8_t);
    break;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    dtype.code = DLDataTypeCode::kDLUInt;
    dtype.bits = sizeof(uint16_t);
    break;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    dtype.code = DLDataTypeCode::kDLUInt;
    dtype.bits = sizeof(uint32_t);
    break;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    dtype.code = DLDataTypeCode::kDLUInt;
    dtype.bits = sizeof(uint64_t);
    break;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    dtype.code = DLDataTypeCode::kDLBfloat;
    dtype.bits = sizeof(uint16_t); // sizeof(BFloat16);
    break;
  default:
    EXT_THROW("Unexpected data type of ", elem_type);
  }

  dtype.bits *= 8; // bits.
  return dtype;
}

ONNXTensorElementDataType GetOrtValueDataType(const DLDataType &dtype) {
  if (dtype.lanes != 1)
    EXT_THROW("OrtValue does not support lanes != 1.");
  switch (dtype.code) {
  case DLDataTypeCode::kDLBool:
    return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  case DLDataTypeCode::kDLUInt:
    switch (dtype.bits) {
    case 8:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNINT8;
    case 16:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case 32:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case 64:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    default:
      EXT_THROW("OrtValue does not support bits=", dtype.bits,
                " (unsigned int).");
    }
  case DLDataTypeCode::kDLInt:
    switch (dtype.bits) {
    case 8:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case 16:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case 32:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case 64:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      EXT_THROW("OrtValue does not support bits=", dtype.bits,
                " (signed int).");
    }
  case DLDataTypeCode::kDLFloat:
    switch (dtype.bits) {
    case 16:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case 32:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case 64:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    default:
      EXT_THROW("OrtValue does not support bits=", dtype.bits, " (float).");
    }
  case DLDataTypeCode::kDLBfloat:
    switch (dtype.bits) {
    case 16:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    default:
      EXT_THROW("OrtValue does not support bits=", dtype.bits, " (bfloat).");
    }
  default:
    EXT_THROW("OrtValue does not support code=", dtype.code, ".");
  }
}

DLDevice GetDlpackDevice(OrtValue *value, const int64_t &device_id) {
  DLDevice device;
  device.device_id = static_cast<int>(device_id);

  OrtMemoryInfo *info;
  ThrowOnError(GetOrtApi()->GetTensorMemoryInfo((OrtValue *)value, &mem_info));
  // OrtAllocatorType alloc_type;
  // ThrowOnError(GetOrtApi()->MemoryInfoGetType(mem_info, ));
  // OrtMemType mem_type;
  // ThrowOnError(GetOrtApi()->MemoryInfoGetMemType(mem_info, &mem_type));
  OrtMemoryInfoDeviceType device_type;
  ThrowOnError(GetOrtApi()->MemoryInfoGetDeviceType(mem_info, &device_type));
  int it;
  ThrowOnError(GetOrtApi()->MemoryInfoGetId(mem_info, &id));

  switch (device_type) {
  case OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU:
    device.device_type = DLDeviceType::kDLCPU;
    break;
  case OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU:
    // #ifdef USE_ROCM
    // device.device_type = DLDeviceType::kDLROCM;
    // #else
    device.device_type = DLDeviceType::kDLCUDA;
    // #endif
    break;
  default:
    EXT_THROW("OrtValue does not support device type=", device_type);
  }

  return device;
}

/*
OrtDevice GetOrtDevice(const DLDevice& device) {
  switch (device.device_type) {
    case DLDeviceType::kDLCPU:
      return OrtDevice();
    case DLDeviceType::kDLCUDA:
    case DLDeviceType::kDLROCM:
      return OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT,
static_cast<OrtDevice::DeviceId>(device.device_id)); default:
      ORT_THROW("Unsupported device type");
  }
}


bool IsContiguousTensor(const DLTensor& tensor) {
  if (!tensor.strides) {
    return true;
  }

  int64_t running_size = 1;
  for (int i = tensor.ndim - 1; i >= 0; i--) {
    if (tensor.shape[i] == 0) {
      return true;
    }

    if (tensor.shape[i] != 1 && tensor.strides[i] != running_size) {
      return false;
    }

    running_size *= tensor.shape[i];
  }

  return true;
}

}  // namespace
*/

struct OrtDLManagedTensor {
  DlpackToOrtValue *handle;
  DLManagedTensor tensor;
};

static void DlpackDeleter(DLManagedTensor *arg) {
  delete static_cast<OrtDLManagedTensor *>(arg->manager_ctx);
}

// This function should use smart pointers inside
// #if defined(_MSC_VER) && !defined(__clang__)
// #pragma warning(push)
// #pragma warning(disable : 26409)
// #pragma warning(disable : 26400)
// #endif

// This function returns a pointer to DLManagedTensor constructed from an
// OrtValue The OrtValue inside OrtDLManagedTensor will increase its own
// buffer's ref count by one When the consumer of DLManagedTensor is done with
// the tensor, it should invoke the deleter.
DLManagedTensor *OrtValueToDlpack(OrtValue *ort_value) {
  OrtDLManagedTensor *ort_dlmanaged_tensor(new OrtDLManagedTensor);

  Tensor &tensor = *ort_value.GetMutable<Tensor>();
  ort_dlmanaged_tensor->handle = ort_value;
  ort_dlmanaged_tensor->tensor.manager_ctx = ort_dlmanaged_tensor;
  ort_dlmanaged_tensor->tensor.deleter = &DlpackDeleter;
  ort_dlmanaged_tensor->tensor.dl_tensor.data = (tensor.MutableDataRaw());
  ort_dlmanaged_tensor->tensor.dl_tensor.device =
      GetDlpackDevice(ort_value, tensor.Location().device.Id());
  ort_dlmanaged_tensor->tensor.dl_tensor.ndim =
      static_cast<int>(tensor.Shape().NumDimensions());
  ort_dlmanaged_tensor->tensor.dl_tensor.dtype = GetDlpackDataType(ort_value);
  ort_dlmanaged_tensor->tensor.dl_tensor.shape =
      tensor.Shape().NumDimensions() > 0
          ? &const_cast<TensorShape &>(tensor.Shape())[0]
          : nullptr;
  ort_dlmanaged_tensor->tensor.dl_tensor.strides = nullptr;
  ort_dlmanaged_tensor->tensor.dl_tensor.byte_offset = 0;
  return &(ort_dlmanaged_tensor->tensor);
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
OrtValue DlpackToOrtValue(DLManagedTensor *dlpack, bool is_bool_tensor) {
  // ORT only supports contiguous tensor for now.
  ORT_ENFORCE(IsContiguousTensor(dlpack->dl_tensor),
              "ORT only supports contiguous tensor for now.");
  OrtDevice device = GetOrtDevice(dlpack->dl_tensor.device);
  MLDataType data_type =
      GetOrtValueDataType(dlpack->dl_tensor.dtype, is_bool_tensor);
  OrtMemoryInfo info(GetOrtDeviceName(device), OrtDeviceAllocator, device,
                     device.Id());
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(
      data_type,
      TensorShape(dlpack->dl_tensor.shape,
                  static_cast<size_t>(dlpack->dl_tensor.ndim)),
      dlpack->dl_tensor.data, info);

  OrtValue ort_value;
  std::function<void(void *)> deleter = [dlpack](void *p) {
    ORT_ENFORCE(dlpack->deleter != NULL,
                "A dlpack structure must have a deleter.");
    dlpack->deleter(dlpack);
    auto deleter =
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_Tensor >
        ()->GetDeleteFunc();
    if (deleter != NULL)
      deleter(p);
  };

  ort_value.Init(
      p_tensor.release(),
      ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_Tensor > (),
      deleter);
  return ort_value;
}

void DlpackCapsuleDestructor(PyObject *data) {
  DLManagedTensor *dlmanaged_tensor = reinterpret_cast<DLManagedTensor *>(
      PyCapsule_GetPointer(data, "dltensor"));
  if (dlmanaged_tensor) {
    // The dlmanaged_tensor has not been consumed, call deleter ourselves.
    dlmanaged_tensor->deleter(const_cast<DLManagedTensor *>(dlmanaged_tensor));
  } else {
    // The dlmanaged_tensor has been consumed,
    // PyCapsule_GetPointer has set an error indicator.
    PyErr_Clear();
  }
}

// Allocate a new Capsule object, which takes the ownership of OrtValue.
// Caller is responsible for releasing.
// This function calls OrtValueToDlpack(...).
PyObject *ToDlpack(OrtValue ort_value) {
  DLManagedTensor *dlmanaged_tensor = dlpack::OrtValueToDlpack(ort_value);
  return PyCapsule_New(dlmanaged_tensor, "dltensor", DlpackCapsuleDestructor);
}

// Consume a Capsule object and claims the ownership of its underlying tensor to
// create a OrtValue. This function calls DlpackToOrtValue(...) to do the
// conversion.
OrtValue FromDlpack(PyObject *dlpack_tensor, const bool is_bool_tensor) {
  // Extract DLPack tensor pointer from the capsule carrier.
  DLManagedTensor *dlmanaged_tensor =
      (DLManagedTensor *)PyCapsule_GetPointer(dlpack_tensor, "dltensor");
  OrtValue ort_value =
      dlpack::DlpackToOrtValue(dlmanaged_tensor, is_bool_tensor);
  // Make sure this capsule will never be used again.
  PyCapsule_SetName(dlpack_tensor, "used_dltensor");
  return ort_value;
}

.def_static(
    "from_dlpack",
    [](py::object data, bool is_bool_tensor) {
      return FromDlpack(data.ptr(), is_bool_tensor);
    },
    py::arg("data"), py::arg("is_bool_tensor") = false,
    "Converts a tensor from a external library into an OrtValue by means of "
    "the __dlpack__ protocol.")
    .def(
        "__dlpack__",
        [](OrtValue *ort_value, py::object /* stream */) -> py::object {
          return py::reinterpret_steal<py::object>(ToDlpack(*ort_value));
        },
        py::arg("stream") = py::none(),
        "Returns a DLPack representing the tensor (part of __dlpack__ "
        "protocol). "
        "This method does not copy the pointer shape, instead, it copies the "
        "pointer value. "
        "The OrtValue must persist until the dlpack structure is consumed.")
    .def(
        "__dlpack_device__",
        [](const OrtValue *ort_value) -> py::tuple {
          ORT_ENFORCE(ort_value->IsTensor(),
                      "Only tensor type OrtValues are supported");
          const onnxruntime::Tensor &tensor = ort_value->Get<Tensor>();
          DLDevice device = onnxruntime::dlpack::GetDlpackDevice(
              *ort_value, tensor.Location().device.Id());
          return py::make_tuple(static_cast<int>(device.device_type),
                                device.device_id);
        },
        "Returns a tuple of integers, (device, device index) (part of "
        "__dlpack__ protocol).")

} // namespace ortapi