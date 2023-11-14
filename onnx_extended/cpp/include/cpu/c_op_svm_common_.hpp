#pragma once
// Inspired from
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_classifier.cc.

#include <cmath>
#include <iterator>
#include <thread>
#include <vector>

#include <omp.h>

#include "common/c_op_common_parameters.h"
#include "common/c_op_helpers.h"

namespace onnx_c_ops {

template <typename NTYPE>
NTYPE vector_dot_product_pointer(const NTYPE *pa, const NTYPE *pb,
                                 std::size_t len) {
  NTYPE s = 0;
  for (; len > 0; ++pa, ++pb, --len)
    s += *pa * *pb;
  return s;
}

template <typename NTYPE> class RuntimeSVMCommon {
public:
  KERNEL kernel_type_;
  NTYPE gamma_;
  NTYPE coef0_;
  int64_t degree_;
  bool one_class_;

  // svm_regressor.h
  int64_t feature_count_;
  int64_t vector_count_;
  std::vector<NTYPE> rho_;
  std::vector<NTYPE> coefficients_;
  std::vector<NTYPE> support_vectors_;
  POST_EVAL_TRANSFORM post_transform_;
  SVM_TYPE mode_; // how are we computing SVM? 0=LibSVC, 1=LibLinear

public:
  RuntimeSVMCommon() {}
  ~RuntimeSVMCommon() {}

  void init(const std::vector<NTYPE> &coefficients,
            const std::vector<NTYPE> &kernel_params,
            const std::string &kernel_type, int64_t n_supports,
            int64_t one_class, const std::string &post_transform,
            const std::vector<NTYPE> &rho,
            const std::vector<NTYPE> &support_vectors) {
    kernel_type_ = to_KERNEL(kernel_type);
    support_vectors_ = support_vectors;
    post_transform_ = to_POST_EVAL_TRANSFORM(post_transform);
    rho_ = rho;
    coefficients_ = coefficients;
    one_class_ = one_class != 0;
    vector_count_ = n_supports;

    if (!kernel_params.empty()) {
      gamma_ = kernel_params[0];
      coef0_ = kernel_params[1];
      degree_ = static_cast<int64_t>(kernel_params[2]);
    } else {
      gamma_ = (NTYPE)0;
      coef0_ = (NTYPE)0;
      degree_ = 0;
    }
  }

  void compute_svm(const std::vector<int64_t> &x_dims, int64_t N,
                   int64_t stride, const NTYPE *x_data, NTYPE *z_data) const {

#pragma omp parallel for
    for (int64_t n = 0; n < N; ++n) {
      int64_t current_weight_0 = n * stride;
      NTYPE sum = 0;
      if (this->mode_ == SVM_TYPE::SVM_SVC) {
        for (int64_t j = 0; j < this->vector_count_; ++j) {
          sum +=
              this->coefficients_[j] *
              this->kernel_dot(x_data, current_weight_0, this->support_vectors_,
                               this->feature_count_ * j, this->feature_count_,
                               this->kernel_type_);
        }
        sum += this->rho_[0];
      } else if (this->mode_ == SVM_TYPE::SVM_LINEAR) {
        sum = this->kernel_dot(x_data, current_weight_0, this->coefficients_, 0,
                               this->feature_count_, this->kernel_type_);
        sum += this->rho_[0];
      }
      z_data[n] = one_class_ ? (sum > 0 ? 1 : -1) : sum;
    }
  }

protected:
  NTYPE kernel_dot(const NTYPE *A, int64_t a, const std::vector<NTYPE> &B,
                   int64_t b, int64_t len, KERNEL k) const {
    double sum = 0;
    double val;
    const NTYPE *pA = A + a;
    const NTYPE *pB = B.data() + b;
    switch (k) {
    case KERNEL::POLY:
      sum = vector_dot_product_pointer(pA, pB, static_cast<std::size_t>(len));
      sum = gamma_ * sum + coef0_;
      switch (degree_) {
      case 2:
        sum = sum * sum;
        break;
      case 3:
        sum = sum * sum * sum;
        break;
      case 4:
        val = sum * sum;
        sum = val * val;
        break;
      default:
        sum = std::pow(sum, degree_);
        break;
      }
      break;
    case KERNEL::SIGMOID:
      sum = vector_dot_product_pointer(pA, pB, static_cast<std::size_t>(len));
      sum = gamma_ * sum + coef0_;
      sum = std::tanh(sum);
      break;
    case KERNEL::RBF:
      for (int64_t i = len; i > 0; --i, ++pA, ++pB) {
        val = *pA - *pB;
        sum += val * val;
      }
      sum = std::exp(-gamma_ * sum);
      break;
    case KERNEL::LINEAR:
      sum = vector_dot_product_pointer(pA, pB, static_cast<std::size_t>(len));
      break;
    }
    return (NTYPE)sum;
  }
};

} // namespace onnx_c_ops
