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
#include "common/c_op_math.h"
#include "onnx_extended_helpers.h"

namespace onnx_c_ops {

template <typename NTYPE>
NTYPE vector_dot_product_pointer(const NTYPE *pa, const NTYPE *pb, std::size_t len) {
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

  std::vector<NTYPE> rho_;
  std::vector<NTYPE> coefficients_;
  std::vector<NTYPE> support_vectors_;
  POST_EVAL_TRANSFORM post_transform_;
  SVM_TYPE mode_; // how are we computing SVM? 0=LibSVC, 1=LibLinear
  int64_t feature_count_;
  int64_t vector_count_;

  // classifier
  std::vector<NTYPE> proba_;
  std::vector<NTYPE> probb_;
  bool weights_are_all_positive_;
  std::vector<int64_t> classlabels_ints_;

  int64_t class_count_;
  std::vector<int64_t> vectors_per_class_;
  std::vector<int64_t> starting_vector_;

public:
  RuntimeSVMCommon() {}
  ~RuntimeSVMCommon() {}

  void init(const std::vector<NTYPE> &coefficients, const std::vector<NTYPE> &kernel_params,
            const std::string &kernel_type, const std::string &post_transform,
            const std::vector<NTYPE> &rho, const std::vector<NTYPE> &support_vectors,
            // regressor
            int64_t n_supports, int64_t one_class,
            // classifier
            const std::vector<NTYPE> &proba, const std::vector<NTYPE> &probb,
            const std::vector<int64_t> &classlabels_ints,
            const std::vector<int64_t> &vectors_per_class) {
    kernel_type_ = to_KERNEL(kernel_type);
    support_vectors_ = support_vectors;
    post_transform_ = to_POST_EVAL_TRANSFORM(post_transform);
    rho_ = rho;
    coefficients_ = coefficients;

    if (!kernel_params.empty()) {
      gamma_ = kernel_params[0];
      coef0_ = kernel_params[1];
      degree_ = static_cast<int64_t>(kernel_params[2]);
    } else {
      gamma_ = (NTYPE)0;
      coef0_ = (NTYPE)0;
      degree_ = 0;
    }

    if (classlabels_ints.empty()) {
      // regressor
      one_class_ = one_class != 0;
      vector_count_ = n_supports;

      if (vector_count_ > 0) {
        feature_count_ =
            support_vectors_.size() / vector_count_; // length of each support vector
        mode_ = SVM_TYPE::SVM_SVC;
      } else {
        feature_count_ = coefficients_.size();
        mode_ = SVM_TYPE::SVM_LINEAR;
        kernel_type_ = KERNEL::LINEAR;
      }
      class_count_ = 0;
    } else {
      // classifier
      proba_ = proba;
      probb_ = probb;
      classlabels_ints_ = classlabels_ints;
      vectors_per_class_ = vectors_per_class;

      vector_count_ = 0;
      class_count_ = 0;
      starting_vector_.reserve(vectors_per_class_.size());
      for (std::size_t i = 0; i < vectors_per_class_.size(); ++i) {
        starting_vector_.push_back(vector_count_);
        vector_count_ += vectors_per_class_[i];
      }

      class_count_ = classlabels_ints_.size() > 0 ? classlabels_ints_.size() : 1;
      if (vector_count_ > 0) {
        feature_count_ =
            support_vectors_.size() / vector_count_; // length of each support vector
        mode_ = SVM_TYPE::SVM_SVC;
      } else {
        feature_count_ = coefficients_.size() / class_count_; // liblinear mode
        mode_ = SVM_TYPE::SVM_LINEAR;
        kernel_type_ = KERNEL::LINEAR;
      }
      weights_are_all_positive_ = true;
      for (std::size_t i = 0; i < coefficients_.size(); i++) {
        if (coefficients_[i] >= 0)
          continue;
        weights_are_all_positive_ = false;
        break;
      }
    }
  }

  void compute_regressor(const std::vector<int64_t> & /* x_dims */, int64_t N, int64_t stride,
                         const NTYPE *x_data, NTYPE *z_data) const {
#pragma omp parallel for
    for (int64_t n = 0; n < N; ++n) {
      int64_t current_weight_0 = n * stride;
      NTYPE sum = 0;
      if (mode_ == SVM_TYPE::SVM_SVC) {
        for (int64_t j = 0; j < vector_count_; ++j) {
          sum +=
              coefficients_[j] * kernel_dot(x_data, current_weight_0, support_vectors_,
                                            feature_count_ * j, feature_count_, kernel_type_);
        }
        sum += rho_[0];
      } else if (mode_ == SVM_TYPE::SVM_LINEAR) {
        sum = kernel_dot(x_data, current_weight_0, coefficients_, 0, feature_count_,
                         kernel_type_);
        sum += rho_[0];
      }
      z_data[n] = one_class_ ? (sum > 0 ? 1 : -1) : sum;
    }
  }

  int64_t get_n_columns() const {
    int64_t n_columns = class_count_;
    if (proba_.size() == 0 && vector_count_ > 0) {
      n_columns = class_count_ > 2 ? class_count_ * (class_count_ - 1) / 2 : 2;
    }
    return n_columns;
  }

  void compute_classifier(const std::vector<int64_t> & /* x_dims */, int64_t N, int64_t stride,
                          const NTYPE *x_data0, int64_t *y_data0, NTYPE *z_data0,
                          int64_t n_columns) const {

#pragma omp parallel for
    for (int64_t n = 0; n < N; ++n) {
      const NTYPE *x_data = x_data0 + n * stride;
      int64_t *y_data = y_data0 + n;
      NTYPE *z_data = z_data0 + n_columns * n;

      int64_t maxclass = -1;
      std::vector<NTYPE> decisions;
      std::vector<NTYPE> scores;
      std::vector<NTYPE> kernels;
      std::vector<int64_t> votes;

      if (vector_count_ == 0 && mode_ == SVM_TYPE::SVM_LINEAR) {
        scores.resize(class_count_);
        for (int64_t j = 0; j < class_count_; j++) { // for each class
          scores[j] = rho_[0] + kernel_dot(x_data, 0, coefficients_, feature_count_ * j,
                                           feature_count_, kernel_type_);
        }
      } else {
        EXT_ENFORCE(vector_count_ > 0, "No support vectors.");
        int evals = 0;

        kernels.resize(vector_count_);
        for (int64_t j = 0; j < vector_count_; j++) {
          kernels[j] = kernel_dot(x_data, 0, support_vectors_, feature_count_ * j,
                                  feature_count_, kernel_type_);
        }
        votes.resize(class_count_, 0);
        scores.reserve(class_count_ * (class_count_ - 1) / 2);
        for (int64_t i = 0; i < class_count_; i++) {   // for each class
          int64_t start_index_i = starting_vector_[i]; // *feature_count_;
          int64_t class_i_support_count = vectors_per_class_[i];
          int64_t pos2 = (vector_count_) * (i);
          for (int64_t j = i + 1; j < class_count_; j++) { // for each class
            NTYPE sum = 0;
            int64_t start_index_j = starting_vector_[j]; // *feature_count_;
            int64_t class_j_support_count = vectors_per_class_[j];

            int64_t pos1 = (vector_count_) * (j - 1);
            const NTYPE *val1 = &(coefficients_[pos1 + start_index_i]);
            const NTYPE *val2 = &(kernels[start_index_i]);
            for (int64_t m = 0; m < class_i_support_count; ++m, ++val1, ++val2)
              sum += *val1 * *val2;

            val1 = &(coefficients_[pos2 + start_index_j]);
            val2 = &(kernels[start_index_j]);
            for (int64_t m = 0; m < class_j_support_count; ++m, ++val1, ++val2)
              sum += *val1 * *val2;

            sum += rho_[evals];
            scores.push_back((NTYPE)sum);
            ++(votes[sum > 0 ? i : j]);
            ++evals; // index into rho
          }
        }
      }

      if (proba_.size() > 0 && mode_ == SVM_TYPE::SVM_SVC) {
        // compute probabilities from the scores
        int64_t num = class_count_ * class_count_;
        std::vector<NTYPE> probsp2(num, 0.f);
        std::vector<NTYPE> estimates(class_count_, 0.f);
        int64_t index = 0;
        NTYPE val1, val2;
        for (int64_t i = 0; i < class_count_; ++i) {
          int64_t p1 = i * class_count_ + i + 1;
          int64_t p2 = (i + 1) * class_count_ + i;
          for (int64_t j = i + 1; j < class_count_; ++j, ++index, ++p1, p2 += class_count_) {
            val1 = sigmoid_probability(scores[index], proba_[index], probb_[index]);
            val2 = std::max(val1, (NTYPE)1.0e-7);
            val2 = std::min(val2, (NTYPE)(1 - 1.0e-7));
            probsp2[p1] = val2;
            probsp2[p2] = 1 - val2;
          }
        }
        multiclass_probability(class_count_, probsp2, estimates);
        // copy probabilities back into scores
        scores.resize(estimates.size());
        std::copy(estimates.begin(), estimates.end(), scores.begin());
      }

      NTYPE max_weight = 0;
      if (votes.size() > 0) {
        auto it_maxvotes = std::max_element(votes.begin(), votes.end());
        maxclass = std::distance(votes.begin(), it_maxvotes);
      } else {
        auto it_max_weight = std::max_element(scores.begin(), scores.end());
        maxclass = std::distance(scores.begin(), it_max_weight);
        max_weight = *it_max_weight;
      }

      // write top class
      // onnx specs expects one column per class.
      int write_additional_scores = -1;
      if (rho_.size() == 1) {
        write_additional_scores = _set_score_svm(y_data, max_weight, maxclass, 0, 1, 0);
      } else if (classlabels_ints_.size() > 0) { // multiclass
        *y_data = classlabels_ints_[maxclass];
      } else {
        *y_data = maxclass;
      }

      write_scores(scores, post_transform_, z_data, write_additional_scores);
    }
  }

protected:
  NTYPE kernel_dot(const NTYPE *A, int64_t a, const std::vector<NTYPE> &B, int64_t b,
                   int64_t len, KERNEL k) const {
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

  int _set_score_svm(int64_t *output_data, NTYPE max_weight, const int64_t maxclass,
                     const int64_t n, int64_t posclass, int64_t negclass) const {
    int write_additional_scores = -1;
    if (classlabels_ints_.size() == 2) {
      write_additional_scores = post_transform_ == POST_EVAL_TRANSFORM::NONE ? 2 : 0;
      if (proba_.size() == 0) {
        if (weights_are_all_positive_ && max_weight >= 0.5)
          output_data[n] = classlabels_ints_[1];
        else if (max_weight > 0 && !weights_are_all_positive_)
          output_data[n] = classlabels_ints_[1];
        else
          output_data[n] = classlabels_ints_[maxclass];
      } else {
        output_data[n] = classlabels_ints_[maxclass];
      }
    } else if (max_weight > 0) {
      output_data[n] = posclass;
    } else {
      output_data[n] = negclass;
    }
    return write_additional_scores;
  }

  void multiclass_probability(int64_t classcount, const std::vector<NTYPE> &r,
                              std::vector<NTYPE> &p) const {
    int64_t sized2 = classcount * classcount;
    std::vector<NTYPE> Q(sized2, 0);
    std::vector<NTYPE> Qp(classcount, 0);
    NTYPE eps = 0.005f / static_cast<NTYPE>(classcount);
    int64_t ii, ij, ji, j;
    NTYPE t;
    for (int64_t i = 0; i < classcount; i++) {
      p[i] = 1.0f / static_cast<NTYPE>(classcount); // Valid if k = 1
      ii = i * classcount + i;
      ji = i;
      ij = i * classcount;
      for (j = 0; j < i; ++j, ++ij, ji += classcount) {
        t = r[ji];
        Q[ii] += t * t;
        Q[ij] = Q[ji];
      }
      ++j;
      ++ij;
      ji += classcount;
      for (; j < classcount; ++j, ++ij, ji += classcount) {
        t = r[ji];
        Q[ii] += t * t;
        Q[ij] = -t * r[ij];
      }
    }
    NTYPE pQp, max_error, error, diff;
    for (int64_t loop = 0; loop < 100; loop++) {
      // stopping condition, recalculate QP,pQP for numerical accuracy
      pQp = 0;
      for (int64_t i = 0; i < classcount; i++) {
        t = 0;
        ij = i * classcount;
        for (int64_t j = 0; j < classcount; ++j, ++ij) {
          t += Q[ij] * p[j];
        }
        Qp[i] = t;
        pQp += p[i] * t;
      }
      max_error = 0;
      for (int64_t i = 0; i < classcount; i++) {
        error = std::fabs(Qp[i] - pQp);
        if (error > max_error) {
          max_error = error;
        }
      }
      if (max_error < eps)
        break;

      for (int64_t i = 0; i < classcount; ++i) {
        ii = i * classcount + i;
        diff = (-Qp[i] + pQp) / Q[ii];
        p[i] += diff;
        pQp = (pQp + diff * (diff * Q[ii] + 2 * Qp[i])) / (1 + diff) / (1 + diff);
        ij = i * classcount;
        for (int64_t j = 0; j < classcount; ++j, ++ij) {
          Qp[j] = (Qp[j] + diff * Q[ij]) / (1 + diff);
          p[j] /= (1 + diff);
        }
      }
    }
  }
};

} // namespace onnx_c_ops
