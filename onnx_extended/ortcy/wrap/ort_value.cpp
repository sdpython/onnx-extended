#include "ort_value.h"
#include "helpers.h"
#include "ortapi.h"
#include "ortapi_inline.h"
#include <functional>

namespace ortapi {

DLDataType GetDlpackDataType(ONNXTensorElementDataType elem_type) {
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
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
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

typedef void (*type_deleter)(struct DLPackOrtValue *self);

DLDevice GetDlpackDevice(OrtValue *value) {
  DLDevice device;

  const OrtMemoryInfo *mem_info;
  ThrowOnError(GetOrtApi()->GetTensorMemoryInfo((OrtValue *)value, &mem_info));
  // OrtAllocatorType alloc_type;
  // ThrowOnError(GetOrtApi()->MemoryInfoGetType(mem_info, ));
  // OrtMemType mem_type;
  // ThrowOnError(GetOrtApi()->MemoryInfoGetMemType(mem_info, &mem_type));
  OrtMemoryInfoDeviceType device_type;
  GetOrtApi()->MemoryInfoGetDeviceType(mem_info, &device_type);
  GetOrtApi()->MemoryInfoGetId(mem_info, &device.device_id);

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

bool IsContiguousTensor(const DLTensor &tensor, int64_t &size) {
  if (!tensor.strides) {
    size = 1;
    for (int i = tensor.ndim - 1; i >= 0; i--) {
      size *= tensor.shape[i];
    }
    return true;
  }

  int64_t running_size = 1;
  for (int i = tensor.ndim - 1; i >= 0; i--) {
    if (tensor.shape[i] == 0) {
      size = 0;
      return true;
    }

    if (tensor.shape[i] != 1 && tensor.strides[i] != running_size) {
      size = -1;
      return false;
    }
    running_size *= tensor.shape[i];
  }
  size = running_size;
  return true;
}

DLPackOrtValue *DlpackToOrtValue(DLManagedTensor *dlpack) {
  int64_t size;
  bool contiguous = IsContiguousTensor(dlpack->dl_tensor, size);
  EXT_ENFORCE(
      contiguous,
      "OrtValue only supports contiguous tensor in this implementation.");
  EXT_ENFORCE(dlpack->dl_tensor.byte_offset == 0,
              "byte_offset != 0 is not supported yet.");

  OrtMemoryInfo *memory_info;
  switch (dlpack->dl_tensor.device.device_type) {
  case DLDeviceType::kDLCPU:
    ThrowOnError(GetOrtApi()->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    break;
  case DLDeviceType::kDLCUDA:
  case DLDeviceType::kDLROCM:
    ThrowOnError(GetOrtApi()->CreateMemoryInfo(
        "Cuda",
        OrtDeviceAllocator,
        dlpack->dl_tensor.device.device_id,
        OrtMemTypeDefault,
        &memory_info));
    break;
  default:
    EXT_THROW("Unsupported device type=", dlpack->dl_tensor.device, ".");
  }

  ONNXTensorElementDataType elem_type =
      GetOrtValueDataType(dlpack->dl_tensor.dtype);

  OrtValue *value;
  ThrowOnError(GetOrtApi()->CreateTensorWithDataAsOrtValue(
      memory_info, dlpack->dl_tensor.data, size * ElementSize(elem_type),
      dlpack->dl_tensor.shape, static_cast<size_t>(dlpack->dl_tensor.ndim),
      elem_type, &value));

  std::function<void(void *)> deleter = [dlpack](void *p) {
    EXT_ENFORCE(dlpack->deleter != NULL,
                "A dlpack structure must have a deleter.");
    dlpack->deleter(dlpack);
    DLPackOrtValue *dl = (DLPackOrtValue *)p;
    GetOrtApi()->ReleaseValue((OrtValue*)dl->ort_value);
    GetOrtApi()->ReleaseMemoryInfo((OrtMemoryInfo*)dl->memory_info);
    if (dl->shape != nullptr) {
      delete dl->shape;
    }
    delete dl;
  };

  // GetDimensions does not return a pointer pointing on the shape definition
  // stored by the OrtValue. It needs to be copied again.
  int64_t *shape = new int64_t[dlpack->dl_tensor.ndim];
  memcpy(shape, dlpack->dl_tensor.shape, dlpack->dl_tensor.ndim * sizeof(int64_t));
  DLPackOrtValue *dl_value = new DLPackOrtValue;
  dl_value->ort_value= (void *)value;
  dl_value->memory_info = (void *)memory_info;
  dl_value->shape = shape;
  dl_value->deleter = deleter;
  return dl_value;
}

struct OrtDLManagedTensor {
  DLPackOrtValue *handle;
  DLManagedTensor tensor;
};

static void DlpackDeleter(DLManagedTensor *arg) {
  delete static_cast<OrtDLManagedTensor *>(arg->manager_ctx);
}

DLManagedTensor *OrtValueToDlpack(DLPackOrtValue *value) {
  OrtDLManagedTensor *ort_dlmanaged_tensor(new OrtDLManagedTensor);

  size_t n_dims, size;
  ONNXTensorElementDataType elem_type;
  void *data;
  OrtTensorTypeAndShapeInfo *info;
  ThrowOnError(
      GetOrtApi()->GetTensorTypeAndShape((OrtValue *)value->ort_value, &info));
  ThrowOnError(GetOrtApi()->GetTensorElementType(info, &elem_type));

  ThrowOnError(GetOrtApi()->GetTensorShapeElementCount(info, &size));
  ThrowOnError(GetOrtApi()->GetTensorMutableData((OrtValue*)value->ort_value, &data));
  ThrowOnError(GetOrtApi()->GetDimensionsCount(info, &n_dims));
  /* typedef void copy_allocate(size_t output, int elem_type, size_t size,
                                OrtShape shape, void* data, void* args); */
  GetOrtApi()->ReleaseTensorTypeAndShapeInfo(info);

  ort_dlmanaged_tensor->handle = value;
  ort_dlmanaged_tensor->tensor.manager_ctx = ort_dlmanaged_tensor;
  ort_dlmanaged_tensor->tensor.deleter = &DlpackDeleter;
  ort_dlmanaged_tensor->tensor.dl_tensor.data = data;
  ort_dlmanaged_tensor->tensor.dl_tensor.device =
      GetDlpackDevice((OrtValue *)value->ort_value);
  ort_dlmanaged_tensor->tensor.dl_tensor.ndim = n_dims;
  ort_dlmanaged_tensor->tensor.dl_tensor.dtype = GetDlpackDataType(elem_type);
  ort_dlmanaged_tensor->tensor.dl_tensor.shape =
      n_dims > 0 ? value->shape : nullptr;
  ort_dlmanaged_tensor->tensor.dl_tensor.strides = nullptr;
  ort_dlmanaged_tensor->tensor.dl_tensor.byte_offset = 0;
  return &(ort_dlmanaged_tensor->tensor);
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
PyObject *ToDlpack(DLPackOrtValue *ort_value) {
  DLManagedTensor *dlmanaged_tensor = OrtValueToDlpack(ort_value);
  return PyCapsule_New(dlmanaged_tensor, "dltensor", DlpackCapsuleDestructor);
}

// Consume a Capsule object and claims the ownership of its underlying tensor to
// create a OrtValue. This function calls DlpackToOrtValue(...) to do the
// conversion.
DLPackOrtValue *FromDlpack(PyObject *dlpack_tensor) {
  // Extract DLPack tensor pointer from the capsule carrier.
  DLManagedTensor *dlmanaged_tensor =
      (DLManagedTensor *)PyCapsule_GetPointer(dlpack_tensor, "dltensor");
  DLPackOrtValue *ort_value = DlpackToOrtValue(dlmanaged_tensor);
  // Make sure this capsule will never be used again.
  PyCapsule_SetName(dlpack_tensor, "used_dltensor");
  return ort_value;
}

int64_t *dlpack_ort_value_get_shape_type(DLPackOrtValue *value, size_t &n_dims,
                                         ONNXTensorElementDataType &elem_type) {
  OrtTensorTypeAndShapeInfo *info;
  ThrowOnError(GetOrtApi()->GetTensorTypeAndShape((OrtValue*)value->ort_value, &info));
  ThrowOnError(GetOrtApi()->GetTensorElementType(info, &elem_type));
  ThrowOnError(GetOrtApi()->GetDimensionsCount(info, &n_dims));
  GetOrtApi()->ReleaseTensorTypeAndShapeInfo(info);
  return value->shape;
}

void delete_dlpack_ort_value(DLPackOrtValue *p) { p->deleter(p); }

void GetDlPackDevice(DLPackOrtValue *value, int &dev_type, int &dev_id) {
  DLDevice dev = GetDlpackDevice((OrtValue*)value->ort_value);
  dev_type = dev.device_type;
  dev_id = dev.device_id;
}

} // namespace ortapi