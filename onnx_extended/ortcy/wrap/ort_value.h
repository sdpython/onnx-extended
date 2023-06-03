#pragma once

#include "onnxruntime_c_api.h"
#if _WIN32
#if not defined(_MSC_VER)
#define _MSC_VER
#include <dlpack/dlpack.h>
#undef _MSC_VER
#endif
#else
#include <dlpack/dlpack.h>
#endif
#include <Python.h>

namespace ortapi {

struct DLPackOrtValue {
  void *ort_value;
  void *memory_info;
  int64_t *shape;
  void (*deleter)(void *self);
};

int64_t* dlpack_ort_value_get_shape_type(DLPackOrtValue *value, size_t &n_dims,
                                         ONNXTensorElementDataType &elem_type);
void delete_dlpack_ort_value(DLPackOrtValue *);
void GetDlPackDevice(DLPackOrtValue *, int &dev_type, int &dev_id);
PyObject *ToDlpack(DLPackOrtValue *ort_value);
DLPackOrtValue *FromDlpack(PyObject *dlpack_tensor);

} // namespace ortapi