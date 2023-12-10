#pragma once

#include "common/c_op_helpers.h"
#include "cpu/c_op_conv.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace onnx_c_ops {

class ConvPoolCommonShape {
protected:
  AutoPadType auto_pad_;
  std::vector<int64_t> kernel_shape_;

public:
  ConvPoolCommonShape() { auto_pad_ = AutoPadType::NOTSET; }

  void init(const std::string &auto_pad, py_array_t<int64_t> kernel_shape);
  void initcpp(const std::string &auto_pad, std::vector<int64_t> kernel_shape);
  void compute_kernel_shape(const std::vector<int64_t> &weight_shape,
                            std::vector<int64_t> &kernel_shape) const;

  void infer_output_shape(const std::vector<int64_t> &input_shape,
                          const std::vector<int64_t> &kernel_shape,
                          const std::vector<int64_t> &strides_p,
                          const std::vector<int64_t> &dilations_p, std::vector<int64_t> &pads_p,
                          std::vector<int64_t> &output_shape,
                          bool ForceSymmetricAutoPadding) const;
};

class ConvPoolCommon : public ConvPoolCommonShape {
protected:
  std::vector<int64_t> dilations_;
  int64_t group_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;

public:
  void init(const std::string &auto_pad, py_array_t<int64_t> dilations, int64_t group,
            py_array_t<int64_t> kernel_shape, py_array_t<int64_t> pads,
            py_array_t<int64_t> strides);

  void initcpp(const std::string &auto_pad, std::vector<int64_t> dilations, int64_t group,
               std::vector<int64_t> kernel_shape, std::vector<int64_t> pads,
               std::vector<int64_t> strides);
};

void ConvPoolCommonShape::init(const std::string &auto_pad, py_array_t<int64_t> kernel_shape) {
  auto_pad_ = to_AutoPadType(auto_pad);
  array2vector(kernel_shape_, kernel_shape, int64_t);
}

void ConvPoolCommonShape::initcpp(const std::string &auto_pad,
                                  std::vector<int64_t> kernel_shape) {
  auto_pad_ = to_AutoPadType(auto_pad);
  kernel_shape_ = kernel_shape;
}

void ConvPoolCommonShape::compute_kernel_shape(const std::vector<int64_t> &weight_shape,
                                               std::vector<int64_t> &kernel_shape) const {
  if (kernel_shape_.size() > 0) {
    kernel_shape = kernel_shape_;
    if (kernel_shape.size() + 2 != weight_shape.size())
      throw std::invalid_argument(
          "kernel_shape num_dims is not compatible with W num_dims (1).");

    for (std::size_t i = 0; i < kernel_shape.size(); ++i)
      if (kernel_shape[i] != weight_shape[i + 2])
        throw std::invalid_argument(
            "kernel_shape num_dims is not compatible with W num_dims (2).");
  } else {
    auto &weight_dims = weight_shape;
    kernel_shape = std::vector<int64_t>(weight_dims.begin() + 2, weight_dims.end());
  }
}

void ConvPoolCommonShape::infer_output_shape(const std::vector<int64_t> &input_shape,
                                             const std::vector<int64_t> &kernel_shape,
                                             const std::vector<int64_t> &strides_p,
                                             const std::vector<int64_t> &dilations_p,
                                             std::vector<int64_t> &pads_p,
                                             std::vector<int64_t> &output_shape,
                                             bool ForceSymmetricAutoPadding) const {
  conv_infer_output_shape(input_shape, kernel_shape, strides_p, dilations_p, pads_p,
                          output_shape, ForceSymmetricAutoPadding, auto_pad_);
}

void ConvPoolCommon::init(const std::string &auto_pad, py_array_t<int64_t> dilations,
                          int64_t group, py_array_t<int64_t> kernel_shape,
                          py_array_t<int64_t> pads, py_array_t<int64_t> strides) {
  ConvPoolCommonShape::init(auto_pad, kernel_shape);
  array2vector(dilations_, dilations, int64_t);
  group_ = group;
  array2vector(pads_, pads, int64_t);
  array2vector(strides_, strides, int64_t);
}

void ConvPoolCommon::initcpp(const std::string &auto_pad, std::vector<int64_t> dilations,
                             int64_t group, std::vector<int64_t> kernel_shape,
                             std::vector<int64_t> pads, std::vector<int64_t> strides) {
  ConvPoolCommonShape::initcpp(auto_pad, kernel_shape);
  dilations_ = dilations;
  group_ = group;
  pads_ = pads;
  strides_ = strides;
}

template <typename T> class Conv : public ConvPoolCommon {
public:
  Conv();

  py::array_t<T> compute(py::array_t<T, py::array::c_style | py::array::forcecast> X,
                         py::array_t<T, py::array::c_style | py::array::forcecast> W,
                         py::array_t<T, py::array::c_style | py::array::forcecast> B) const;

protected:
  void compute_gil_free(py::array_t<T, py::array::c_style | py::array::forcecast> X,
                        py::array_t<T, py::array::c_style | py::array::forcecast> W,
                        py::array_t<T, py::array::c_style | py::array::forcecast> B,
                        py::array_t<T, py::array::c_style | py::array::forcecast> &Y,
                        const std::vector<int64_t> &input_shape,
                        const std::vector<int64_t> &output_shape,
                        const std::vector<int64_t> &kernel_shape,
                        const std::vector<int64_t> &pads, const std::vector<int64_t> &dilations,
                        const std::vector<int64_t> &strides, const std::vector<int64_t> &x_dims,
                        const std::vector<int64_t> &y_dims,
                        const std::vector<int64_t> &w_dims) const;
};

template <typename T> Conv<T>::Conv() : ConvPoolCommon() {}

template <typename T>
py::array_t<T>
Conv<T>::compute(py::array_t<T, py::array::c_style | py::array::forcecast> X,
                 py::array_t<T, py::array::c_style | py::array::forcecast> W,
                 py::array_t<T, py::array::c_style | py::array::forcecast> B) const {
  std::vector<int64_t> x_dims;
  arrayshape2vector(x_dims, X);
  std::vector<int64_t> w_dims;
  arrayshape2vector(w_dims, W);

  const int64_t N = x_dims[0];
  const int64_t M = w_dims[0];

  std::vector<int64_t> kernel_shape;
  compute_kernel_shape(w_dims, kernel_shape);

  std::vector<int64_t> pads(pads_);
  if (pads.empty())
    pads.resize(kernel_shape.size() * 2, 0);

  std::vector<int64_t> dilations(dilations_);
  if (dilations.empty())
    dilations.resize(kernel_shape.size(), 1);

  std::vector<int64_t> strides(strides_);
  if (strides.empty())
    strides.resize(kernel_shape.size(), 1);

  std::vector<int64_t> y_dims;
  y_dims.insert(y_dims.begin(), {N, M});
  std::vector<int64_t> input_shape(x_dims.begin() + 2, x_dims.end());
  infer_output_shape(input_shape, kernel_shape, strides, dilations, pads, y_dims, false);
  std::vector<int64_t> output_shape(y_dims.begin() + 2, y_dims.end());

  py::array_t<T, py::array::c_style | py::array::forcecast> Y(y_dims);
  {
    py::gil_scoped_release release;
    compute_gil_free(X, W, B, Y, input_shape, output_shape, kernel_shape, pads, dilations,
                     strides, x_dims, y_dims, w_dims);
  }
  return Y;
}

template <typename T>
void Conv<T>::compute_gil_free(
    py::array_t<T, py::array::c_style | py::array::forcecast> X,
    py::array_t<T, py::array::c_style | py::array::forcecast> W,
    py::array_t<T, py::array::c_style | py::array::forcecast> B,
    py::array_t<T, py::array::c_style | py::array::forcecast> &Y,
    const std::vector<int64_t> &input_shape, const std::vector<int64_t> &output_shape,
    const std::vector<int64_t> &kernel_shape, const std::vector<int64_t> &pads,
    const std::vector<int64_t> &dilations, const std::vector<int64_t> &strides,
    const std::vector<int64_t> &x_dims, const std::vector<int64_t> &y_dims,
    const std::vector<int64_t> &w_dims) const {
  std::vector<int64_t> b_dims;
  arrayshape2vector(b_dims, B);

  const int64_t N = x_dims[0];
  const int64_t C = x_dims[1];
  const int64_t M = w_dims[0];

  const int64_t input_image_size = flattened_dimension(input_shape);
  const int64_t output_image_size = flattened_dimension(output_shape);
  const int64_t y_size = flattened_dimension(y_dims);
  const int64_t kernel_size = flattened_dimension(kernel_shape);
  const int64_t X_offset = C / group_ * input_image_size;
  const int64_t Y_offset = flattened_dimension(y_dims) / y_dims[0] / group_;
  const int64_t W_offset = flattened_dimension(w_dims) / group_;
  const int64_t kernel_dim = C / group_ * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;

  std::vector<T> _col_data(col_buffer_size);
  auto col_buffer_data = &_col_data[0];

  const T *Xdata = X.data(0);
  T *Ydata = (T *)Y.data(0);
  T *yptr;
  std::size_t k2;

  std::fill(Ydata, Ydata + y_size, (T)0);

  std::vector<int64_t> image_shape(x_dims.begin() + 1, x_dims.end());
  std::vector<int64_t> col_buffer_shape{kernel_dim};
  col_buffer_shape.insert(col_buffer_shape.end(), output_shape.begin(), output_shape.end());

  const std::size_t kernel_rank = kernel_shape.size();

  for (int image_id = 0; image_id < N; ++image_id) {
    for (int group_id = 0; group_id < group_; ++group_id) {
      if (kernel_rank == 2) {
        Im2col_NCHW<T>(Xdata + group_id * X_offset, C / group_, input_shape[0], input_shape[1],
                       kernel_shape[0], kernel_shape[1], dilations[0], dilations[1], pads[0],
                       pads[1], pads[2], pads[3], strides[0], strides[1], col_buffer_data);
      } else {
        Im2colNd_NCHW<T>(Xdata + group_id * X_offset, &image_shape[0], col_buffer_shape.data(),
                         C * input_image_size, col_buffer_size, &kernel_shape[0],
                         strides.data(), &dilations[0], &pads[0],
                         static_cast<int>(kernel_shape.size()), col_buffer_data);
      }

      gemm<T>(false, false,
              (std::size_t)(M / group_),                  // m
              (std::size_t)(output_image_size),           // n
              (std::size_t)kernel_dim,                    // k
              (T)1,                                       // alpha
              (const T *)W.data(0) + group_id * W_offset, // *a
              (const T *)col_buffer_data,                 // *b
              (T)0,                                       // beta
              (T *)Ydata + group_id * Y_offset            // *c
      );
    }

    if (b_dims.size() != 0 && b_dims[0] != 0) {
      const T *ptrb = B.data(0);
      for (std::size_t k = 0; k < (std::size_t)M; ++k, ++ptrb) {
        yptr = Ydata + output_image_size * k;
        for (k2 = 0; k2 < (std::size_t)output_image_size; ++k2, ++yptr)
          *yptr += *ptrb;
      }
    }

    Xdata += X_offset * group_;
    Ydata += Y_offset * group_;
  }
}

class ConvFloat : public Conv<float> {
public:
  ConvFloat() : Conv<float>() {}
};

class ConvDouble : public Conv<double> {
public:
  ConvDouble() : Conv<double>() {}
};

}; // namespace onnx_c_ops
