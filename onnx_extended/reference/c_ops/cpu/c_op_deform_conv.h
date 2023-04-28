#include "c_op_conv.h"
#include "c_op_conv_common.h"

#define py_array_T py::array_t<T, py::array::c_style | py::array::forcecast>
#define py_array_int64                                                         \
  py::array_t<int64_t, py::array::c_style | py::array::forcecast>

namespace onnx_c_ops {

namespace py = pybind11;

template <typename T>
T dmcn_im2col_bilinear(const T *bottom_data, const int data_width,
                       const int height, const int width, T h, T w) {
  int h_low = static_cast<int>(std::floor(h));
  int w_low = static_cast<int>(std::floor(w));
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh, hw = 1 - lw;

  T v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  T v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  T v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  T v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename T>
T dmcn_get_gradient_weight(T argmax_h, T argmax_w, const int h, const int w,
                           const int height, const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = static_cast<int>(std::floor(argmax_h));
  int argmax_w_low = static_cast<int>(std::floor(argmax_w));
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename T>
T dmcn_get_coordinate_weight(T argmax_h, T argmax_w, const int height,
                             const int width, const T *im_data,
                             const int data_width, const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = static_cast<int>(std::floor(argmax_h));
  int argmax_w_low = static_cast<int>(std::floor(argmax_w));
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename T>
void deformable_im2col_gpu_kernel(
    const int n, const T *data_im, const T *data_offset, const int height,
    const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int num_channels, const int deformable_group, const int height_col,
    const int width_col, T *data_col) {

  // launch channels * batch_size * height_col * width_col cores
  for (int index = 0; index < n; ++index) {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation,
    // col_buffer is of shape (c*kw*kh, N, oh, ow) here columns is of shape (N,
    // c*kw*kh, oh * ow), need to adapt axis NOTE(Jiarui XU): different from
    // CharlesShang's implementation, col_buffer is of shape (N, c*kw*kh, oh *
    // ow) here columns is of shape (c*kw*kh, N, oh, ow), need to adapt axis

    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    T *data_col_ptr =
        data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    // const T* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height
    // + h_in) * width + w_in;
    const T *data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const T *data_offset_ptr =
        data_offset + (b_col * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const T offset_h = data_offset_ptr[data_offset_h_ptr];
        const T offset_w = data_offset_ptr[data_offset_w_ptr];
        T val = static_cast<T>(0);
        const T h_im = h_in + i * dilation_h + offset_h;
        const T w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          // const T map_h = i * dilation_h + offset_h;
          // const T map_w = j * dilation_w + offset_w;
          // const int cur_height = height - h_in;
          // const int cur_width = width - w_in;
          // val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height,
          // cur_width, map_h, map_w);
          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im,
                                     w_im);
        }
        *data_col_ptr = val;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template <typename T>
void deform_im2col_forward(const T *data_im, const T *data_offset,
                           const int batch_size, const int channels,
                           const int height_im, const int width_im,
                           const int height_col, const int width_col,
                           const int kernel_h, const int kernel_w,
                           const int pad_h, const int pad_w, const int stride_h,
                           const int stride_w, const int dilation_h,
                           const int dilation_w, const int deformable_group,
                           T *data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;

  deformable_im2col_gpu_kernel<T>(
      num_kernels, data_im, data_offset, height_im, width_im, kernel_h,
      kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      channel_per_deformable_group, batch_size, channels, deformable_group,
      height_col, width_col, data_col);
}

template <typename T>
void deform_im2col_forward(
    std::vector<T> &output, std::vector<int64_t> &output_shape, const T *input,
    const std::vector<int64_t> &input_shape, const int64_t *offset,
    const std::vector<int64_t> &offset_shape,
    // const int channels_out,
    // const int channels_kernel,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int group, const int deformable_group,
    const int im2col_step) {
  const int batch = input_shape[0];
  const int channels = input_shape[1];
  const int height = input_shape[2];
  const int width = input_shape[3];
  const int im2col_step_ = std::min(batch, im2col_step);

  // limitations of the current implementation
  EXT_ENFORCE(im2col_step_ == 1, "only support im2col_step == 1");
  EXT_ENFORCE(group == 1, "only support group == 1");
  EXT_ENFORCE(deformable_group == 1, "only support deformable_group == 1");
  EXT_ENFORCE(batch % im2col_step_ == 0, "batch(", batch,
              ") must divide im2col_step(", im2col_step_, ")");
  EXT_ENFORCE(channels % group == 0, "channels(", channels,
              ") must divide group(", group, ")");

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  const int batch_n = im2col_step_;
  const int per_input_size = channels * height * width;
  const int per_offset_size =
      offset_shape[1] * offset_shape[2] * offset_shape[3];

  output_shape = std::move(
      std::vector<int64_t>{batch / im2col_step_, channels * kernel_h * kernel_w,
                           batch_n * height_out * width_out});
  int64_t total_size = (batch / im2col_step_) * channels * kernel_h * kernel_w *
                       batch_n * height_out * width_out;
  int64_t column_size = total_size / (batch / im2col_step_);
  output.resize(total_size);

  for (int n = 0; n < batch / im2col_step_; ++n) {
    deformable_im2col_forward(
        input + n * im2col_step_ * per_input_size,
        offset + n * im2col_step_ * per_offset_size, batch_n, channels, height,
        width, height_out, width_out, kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w, deformable_group,
        output.data() + column_size * n);
  }
}

template <typename T> class DeformConv : public ConvPoolCommon {
public:
  DeformConv() : ConvPoolCommon() {}

  py::array_t<T> compute(py_array_T X, py_array_T W, py_array_int64 offset,
                         py_array_T B, py_array_T mask) const {
    std::vector<int64_t> mask_dims;
    arrayshape2vector(mask_dims, mask);
    if (mask_dims.size() != 0) {
      throw std::runtime_error("Not implemented when mask is specified.")
    }

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
    infer_output_shape(input_shape, kernel_shape, strides, dilations, pads,
                       y_dims, false);
    std::vector<int64_t> output_shape(y_dims.begin() + 2, y_dims.end());

    py_array_T Y(y_dims);
    {
      py::gil_scoped_release release;
      compute_gil_free(X, W, offset, B, mask, Y, input_shape, output_shape,
                       kernel_shape, pads, dilations, strides, x_dims, y_dims,
                       w_dims);
    }
    return Y;
  }

protected:
  void compute_gil_free(py_array_T X, py_array_T W, py_array_int64 offset,
                        py_array_T B, py_array_T mask, py_array_T &Y,
                        const std::vector<int64_t> &input_shape,
                        const std::vector<int64_t> &output_shape,
                        const std::vector<int64_t> &kernel_shape,
                        const std::vector<int64_t> &pads,
                        const std::vector<int64_t> &dilations,
                        const std::vector<int64_t> &strides,
                        const std::vector<int64_t> &x_dims,
                        const std::vector<int64_t> &y_dims,
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
    const int64_t *offsetdata = offset.data(0);
    T *Ydata = (T *)Y.data(0);
    T *yptr;
    size_t k2;

    std::fill(Ydata, Ydata + y_size, (T)0);

    std::vector<int64_t> image_shape(x_dims.begin() + 1, x_dims.end());
    std::vector<int64_t> col_buffer_shape{kernel_dim};
    col_buffer_shape.insert(col_buffer_shape.end(), output_shape.begin(),
                            output_shape.end());

    const size_t kernel_rank = kernel_shape.size();

    for (int image_id = 0; image_id < N; ++image_id) {
      for (int group_id = 0; group_id < group_; ++group_id) {
        if (kernel_rank == 2) {
          deform_im2col_forward(
              Xdata + group_id * X_offset, offsetdata, C / group_,
              input_shape[0], input_shape[1], kernel_shape[0], kernel_shape[1],
              dilations[0], dilations[1], pads[0], pads[1], pads[2], pads[3],
              strides[0], strides[1], deformable_group, col_buffer_data);
        } else {
          throw std::runtime_error(
              "DeformConv is only implemented for rank==2.");
        }

        gemm<T>(false, false,
                (size_t)(M / group_),                       // m
                (size_t)(output_image_size),                // n
                (size_t)kernel_dim,                         // k
                (T)1,                                       // alpha
                (const T *)W.data(0) + group_id * W_offset, // *a
                (const T *)col_buffer_data,                 // *b
                (T)0,                                       // beta
                (T *)Ydata + group_id * Y_offset            // *c
        );
      }

      if (b_dims.size() != 0 && b_dims[0] != 0) {
        const T *ptrb = B.data(0);
        for (size_t k = 0; k < (size_t)M; ++k, ++ptrb) {
          yptr = Ydata + output_image_size * k;
          for (k2 = 0; k2 < (size_t)output_image_size; ++k2, ++yptr)
            *yptr += *ptrb;
        }
      }

      Xdata += X_offset * group_;
      Ydata += Y_offset * group_;
    }
  }
};

class DeformConvFloat : public DeformConv<float> {
public:
  DeformConvFloat() : DeformConv<float>() {}
};

class DeformConvDouble : public DeformConv<double> {
public:
  DeformConvDouble() : DeformConv<double>() {}
};

}; // namespace onnx_c_ops
