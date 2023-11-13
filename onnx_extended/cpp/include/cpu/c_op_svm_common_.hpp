#pragma once
// Inspired from
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_classifier.cc.

#include <iterator>
#include <thread>
#include <vector>

#include <omp.h>

#include "common/c_op_common_.hpp"
#include "common/c_op_common_num_.hpp"

template <typename T> struct SimpleTensor {
  std::vector<int64_t> shape;
  std::vector<T> values;
};

template <typename NTYPE> class RuntimeSVMCommon {
public:
  KERNEL kernel_type_;
  NTYPE gamma_;
  NTYPE coef0_;
  int64_t degree_;

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

  void init(const SimpleTensor<NTYPE> &coefficients,
            const SimpleTensor<NTYPE> &kernel_params,
            const std::string &kernel_type, const std::string &post_transform,
            const SimpleTensor<NTYPE> &rho,
            const SimpleTensor<NTYPE> &support_vectors) {
    kernel_type_ = to_KERNEL(kernel_type);
    array2vector(support_vectors_, support_vectors, NTYPE);
    post_transform_ = to_POST_EVAL_TRANSFORM(post_transform);
    array2vector(rho_, rho, NTYPE);
    array2vector(coefficients_, coefficients, NTYPE);

    std::vector<NTYPE> kernel_params_local;
    array2vector(kernel_params_local, kernel_params, NTYPE);

    if (!kernel_params_local.empty()) {
      gamma_ = kernel_params_local[0];
      coef0_ = kernel_params_local[1];
      degree_ = static_cast<int64_t>(kernel_params_local[2]);
    } else {
      gamma_ = (NTYPE)0;
      coef0_ = (NTYPE)0;
      degree_ = 0;
    }
  }

  void compute(const SimpleTensor<NTYPE> &coefficients,
               SimpleTensor<NTYPE> &output) const;

protected:
  NTYPE kernel_dot(const NTYPE *A, int64_t a, const std::vector<NTYPE> &B,
                   int64_t b, int64_t len, KERNEL k) const {
    double sum = 0;
    double val;
    const NTYPE *pA = A + a;
    const NTYPE *pB = B.data() + b;
    switch (k) {
    case KERNEL::POLY:
      sum = vector_dot_product_pointer_sse(pA, pB, (size_t)len);
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
      sum = vector_dot_product_pointer_sse(pA, pB, (size_t)len);
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
      sum = vector_dot_product_pointer_sse(pA, pB, (size_t)len);
      break;
    }
    return (NTYPE)sum;
  }

  void compute_gil_free(const std::vector<int64_t> &x_dims, int64_t N,
                        int64_t stride, const SimpleTensor<NTYPE> &X,
                        SimpleTensor<NTYPE> &Z) const {

    const NTYPE *x_data = X.values.data();
    NTYPE *z_data = Z.values.data();

#pragma omp parallel for
    for (int64_t n = 0; n < N; ++n) {
      int64_t current_weight_0 = n * stride;
      NTYPE sum = 0;
      if (this->mode_ == SVM_TYPE::SVM_SVC) {
        for (int64_t j = 0; j < this->vector_count_; ++j) {
          sum += this->coefficients_[j] *
                 this->kernel_dot_gil_free(
                     x_data, current_weight_0, this->support_vectors_,
                     this->feature_count_ * j, this->feature_count_,
                     this->kernel_type_);
        }
        sum += this->rho_[0];
      } else if (this->mode_ == SVM_TYPE::SVM_LINEAR) {
        sum = this->kernel_dot_gil_free(
            x_data, current_weight_0, this->coefficients_, 0,
            this->feature_count_, this->kernel_type_);
        sum += this->rho_[0];
      }
      z_data[n] = one_class_ ? (sum > 0 ? 1 : -1) : sum;
    }
  }
};
