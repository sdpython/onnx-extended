#pragma once

#include <algorithm>
#include <float.h>
#include <iostream> // cout
#include <iterator>
#include <sstream>
#include <thread>
#include <vector>
#if _WIN32
// #include <cmath>
#else
#include <cmath>
#endif

namespace onnx_c_ops {

void *AllocatorDefaultAlloc(size_t size);
void AllocatorDefaultFree(void *p);

class Status {
public:
  int code;
  Status() : code(1) {}
  Status(int code) : code(code) {}
  Status &operator=(const Status &other) {
    code = other.code;
    return *this;
  }
  bool IsOK() const { return code == 1; }
  int Code() const { return code; }
  bool operator==(const Status &other) const { return code == other.code; }
  bool operator!=(const Status &other) const { return !(*this == other); }
  static Status OK() { return Status(1); }
};

#define InlinedVector std::vector
#define InlinedHashSet std::unordered_set

#if defined(_WIN32) || defined(WIN32)

inline bool _isnan_(float x) { return _isnanf(x); }
inline bool _isnan_(double x) { return _isnan(x); }

#elif defined(__MACOSX__) || defined(__APPLE__)

inline bool _isnan_(float x) { return (float)::isnan((double)x); }
inline bool _isnan_(double x) { return ::isnan(x); }

#else

// See
// https://stackoverflow.com/questions/2249110/how-do-i-make-a-portable-isnan-isinf-function
inline bool _isnan_(double x) {
  union {
    uint64_t u;
    double f;
  } ieee754;
  ieee754.f = x;
  return ((unsigned)(ieee754.u >> 32) & 0x7fffffff) +
             ((unsigned)ieee754.u != 0) >
         0x7ff00000;
}

inline bool _isnan_(float x) { return _isnan_((double)x); }

#endif

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

#if !defined(__APPLE__)
#ifndef _SSIZE_T_DEFINED
typedef int64_t ssize_t;
#define _SSIZE_T_DEFINED
#endif
#endif

enum class POST_EVAL_TRANSFORM {
  NONE = 0,
  LOGISTIC = 1,
  SOFTMAX = 2,
  SOFTMAX_ZERO = 3,
  PROBIT = 4
};

POST_EVAL_TRANSFORM to_POST_EVAL_TRANSFORM(const std::string &value);

enum NODE_MODE : uint8_t {
  LEAF = 1,
  BRANCH_LEQ = 2,
  BRANCH_LT = 4,
  BRANCH_GTE = 6,
  BRANCH_GT = 8,
  BRANCH_EQ = 10,
  BRANCH_NEQ = 12
};

NODE_MODE to_NODE_MODE(const std::string &value);

const char *to_str(NODE_MODE mode);

enum class AGGREGATE_FUNCTION { AVERAGE, SUM, MIN, MAX };

AGGREGATE_FUNCTION to_AGGREGATE_FUNCTION(const std::string &input);

enum class SVM_TYPE { SVM_LINEAR, SVM_SVC };

SVM_TYPE to_SVM_TYPE(const std::string &value);

enum KERNEL { LINEAR, POLY, RBF, SIGMOID };

KERNEL to_KERNEL(const std::string &value);

enum StorageOrder {
  UNKNOWN = 0,
  NHWC = 1,
  NCHW = 2,
};

StorageOrder to_StorageOrder(const std::string &value);

enum class AutoPadType {
  NOTSET = 0,
  VALID = 1,
  SAME_UPPER = 2,
  SAME_LOWER = 3,
};

AutoPadType to_AutoPadType(const std::string &value);

static inline float ErfInv(float x) {
  float sgn = x < 0 ? -1.0f : 1.0f;
  x = (1 - x) * (1 + x);
  float log = std::log(x);
  float v = 2 / (3.14159f * 0.147f) + 0.5f * log;
  float v2 = 1 / (0.147f) * log;
  float v3 = -v + std::sqrt(v * v - v2);
  x = sgn * std::sqrt(v3);
  return x;
}

static inline double ErfInv(double x) {
  double sgn = x < 0 ? -1.0 : 1.0;
  x = (1 - x) * (1 + x);
  double log = std::log(x);
  double v = 2. / (3.14159f * 0.147f) + 0.5f * log;
  double v2 = 1. / (0.147f) * log;
  double v3 = std::sqrt(v * v - v2) - v;
  return sgn * std::sqrt(v3);
}

static inline float ComputeLogistic(float val) {
  float v = 1 / (1 + std::exp(-std::abs(val)));
  return (val < 0) ? (1 - v) : v;
}

static inline double ComputeLogistic(double val) {
  double v = 1. / (1. + std::exp(-std::abs(val)));
  return (val < 0) ? (1. - v) : v;
}

static const float ml_sqrt2 = 1.41421356f;

template <class NTYPE> static inline NTYPE ComputeProbit(NTYPE val) {
  return ml_sqrt2 * ErfInv(val * 2 - 1);
}

template <class NTYPE>
static inline NTYPE sigmoid_probability(NTYPE score, NTYPE proba, NTYPE probb) {
  // ref:
  // https://github.com/arnaudsj/libsvm/blob/eaaefac5ebd32d0e07902e1ae740e038eaaf0826/svm.cpp#L1818
  NTYPE val = score * proba + probb;
  return 1 - ComputeLogistic(val);
}

template <typename NTYPE> void ComputeSoftmax(NTYPE *begin, NTYPE *end) {
  NTYPE v_max = -FLT_MAX;
  NTYPE *it;
  for (it = begin; it != end; ++it) {
    if (*it > v_max)
      v_max = *it;
  }
  NTYPE this_sum = 0;
  for (it = begin; it != end; ++it) {
    *it = std::exp(*it - v_max);
    this_sum += *it;
  }
  for (it = begin; it != end; ++it)
    *it /= this_sum;
}

template <typename NTYPE> void ComputeSoftmax(std::vector<NTYPE> &values) {
  ComputeSoftmax(values.data(), values.data() + values.size());
}

template <typename NTYPE> void ComputeSoftmaxZero(NTYPE *begin, NTYPE *end) {
  NTYPE v_max = -std::numeric_limits<NTYPE>::max();
  NTYPE *it;
  for (it = begin; it != end; ++it) {
    if (*it > v_max)
      v_max = *it;
  }
  NTYPE exp_neg_v_max = std::exp(-v_max);
  NTYPE this_sum = (NTYPE)0;
  for (it = begin; it != end; ++it) {
    if (*it > 0.0000001f || *it < -0.0000001f) {
      *it = std::exp(*it - v_max);
      this_sum += *it;
    } else {
      *it *= exp_neg_v_max;
    }
  }
  for (it = begin; it != end; ++it)
    *it /= this_sum;
}

template <typename NTYPE> void ComputeSoftmaxZero(std::vector<NTYPE> &values) {
  ComputeSoftmaxZero(values.data(), values.data() + values.size());
}

template <typename NTYPE, typename T>
size_t write_scores(std::vector<NTYPE> &scores,
                    POST_EVAL_TRANSFORM post_transform, T *Z,
                    int add_second_class) {
  if ((scores.size() == 1) && add_second_class) {
    scores.push_back(scores[0]);
    scores[1] = 0.f;
    return write_scores(1, scores.data(), post_transform, Z, add_second_class);
  }
  return write_scores(scores.size(), scores.data(), post_transform, Z,
                      add_second_class);
}

template <typename NTYPE, typename T>
size_t write_scores(size_t n_classes, NTYPE *scores,
                    POST_EVAL_TRANSFORM post_transform, T *Z,
                    int add_second_class) {
  if (n_classes >= 2) {
    NTYPE *end = scores + n_classes;
    switch (post_transform) {
    case POST_EVAL_TRANSFORM::PROBIT:
      for (auto it = scores; it != end; ++it, ++Z)
        *Z = ComputeProbit((T)*it);
      break;
    case POST_EVAL_TRANSFORM::LOGISTIC:
      for (auto it = scores; it != end; ++it, ++Z)
        *Z = ComputeLogistic((T)*it);
      break;
    case POST_EVAL_TRANSFORM::SOFTMAX:
      ComputeSoftmax(scores, end);
      for (auto it = scores; it != end; ++it, ++Z)
        *Z = (T)*it;
      break;
    case POST_EVAL_TRANSFORM::SOFTMAX_ZERO:
      ComputeSoftmaxZero(scores, end);
      for (auto it = scores; it != end; ++it, ++Z)
        *Z = (T)*it;
      break;
    default:
    case POST_EVAL_TRANSFORM::NONE:
      for (auto it = scores; it != end; ++it, ++Z)
        *Z = (T)*it;
      break;
    }
  } else if (n_classes == 1) { // binary case
    if (post_transform == POST_EVAL_TRANSFORM::PROBIT) {
      scores[0] = ComputeProbit((T)scores[0]);
      *Z = scores[0];
    } else {
      switch (add_second_class) {
      case 0: // 0=all positive weights, winning class is positive
        scores[1] = (T)scores[0];
        scores[0] = 1.f - (T)scores[0]; // put opposite score in positive slot
        *Z = (T)scores[0];
        *(Z + 1) = (T)scores[1];
        ++n_classes;
        break;
      case 1: // 1 = all positive weights, winning class is negative
        scores[1] = (T)scores[0];
        scores[0] = 1.f - (T)scores[0]; // put opposite score in positive slot
        *Z = (T)scores[0];
        *(Z + 1) = (T)scores[1];
        ++n_classes;
        break;
      case 2:
      case 3: // 2 = mixed weights, winning class is positive
        if (post_transform == POST_EVAL_TRANSFORM::LOGISTIC) {
          scores[1] = ComputeLogistic((T)scores[0]); // ml_logit(scores[k]);
          scores[0] = ComputeLogistic((T)(-scores[0]));
        } else {
          scores[1] = (T)scores[0];
          scores[0] = (T)(-scores[0]);
        }
        *Z = scores[0];
        *(Z + 1) = scores[1];
        ++n_classes;
        break;
      default:
        *Z = scores[0];
        break;
      }
    }
  }
  return n_classes;
}

template <typename NTYPE, typename T>
size_t write_scores2(NTYPE *scores, POST_EVAL_TRANSFORM post_transform, T *Z,
                     int add_second_class) {
  switch (post_transform) {
  case POST_EVAL_TRANSFORM::PROBIT:
    Z[0] = ComputeProbit(scores[0]);
    Z[1] = ComputeProbit(scores[1]);
    break;
  case POST_EVAL_TRANSFORM::LOGISTIC:
    Z[0] = ComputeLogistic(scores[0]);
    Z[1] = ComputeLogistic(scores[1]);
    break;
  case POST_EVAL_TRANSFORM::SOFTMAX:
    ComputeSoftmax(scores, scores + 2);
    Z[0] = (T)scores[0];
    Z[1] = (T)scores[1];
    break;
  case POST_EVAL_TRANSFORM::SOFTMAX_ZERO:
    ComputeSoftmaxZero(scores, scores + 2);
    Z[0] = (T)scores[0];
    Z[1] = (T)scores[1];
    break;
  default:
  case POST_EVAL_TRANSFORM::NONE:
    Z[0] = (T)scores[0];
    Z[1] = (T)scores[1];
    break;
  }
  return 2;
}

#define py_array_t py::array_t
#define py_array_style py::array::c_style | py::array::forcecast

#define is_a_ge_zero_and_a_lt_b(a, b)                                          \
  (static_cast<uint64_t>(a) < static_cast<uint64_t>(b))

#define array2vector(vec, arr, dtype)                                          \
  {                                                                            \
    if (arr.size() > 0) {                                                      \
      auto n = arr.size();                                                     \
      auto p = (dtype *)arr.data(0);                                           \
      vec = std::vector<dtype>(p, p + n);                                      \
    }                                                                          \
  }

#define arrayshape2vector(vec, arr)                                            \
  {                                                                            \
    if (arr.size() > 0) {                                                      \
      vec.resize(arr.ndim());                                                  \
      for (size_t i = 0; i < vec.size(); ++i)                                  \
        vec[i] = (int64_t)arr.shape(i);                                        \
    }                                                                          \
  }

template <class NTYPE>
NTYPE flattened_dimension(const std::vector<NTYPE> &values) {
  NTYPE r = 1;
  for (auto it = values.begin(); it != values.end(); ++it)
    r *= *it;
  return r;
}

template <class NTYPE>
NTYPE flattened_dimension(const std::vector<NTYPE> &values, int64_t first) {
  NTYPE r = 1;
  auto end = values.begin() + first;
  for (auto it = values.begin(); it != end; ++it)
    r *= *it;
  return r;
}

template <class DIMTYPE, class NTYPE>
void shape2strides(const std::vector<DIMTYPE> &shape,
                   std::vector<DIMTYPE> &strides, NTYPE cst) {
  strides.resize(shape.size());
  strides[strides.size() - 1] = sizeof(NTYPE);
  for (int64_t i = (int64_t)strides.size() - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
}

template <class DIMTYPE>
DIMTYPE SizeFromDimension(const std::vector<DIMTYPE> &shape, size_t start,
                          size_t end) {
  DIMTYPE size = 1;
  for (size_t i = start; i < end; i++) {
    if (shape[i] < 0)
      return -1;
    size *= shape[i];
  }
  return size;
}

template <typename T, T b> constexpr T roundUpPow2(T a) {
  return (a + (b - 1)) & (~(b - 1));
}

inline int64_t HandleNegativeAxis(int64_t axis, int64_t tensor_rank) {
  return axis < 0 ? axis + tensor_rank : axis;
}

template <typename T>
void debug_print(const std::string &msg, size_t size, const T *value) {
  std::cout << msg << " - size:" << size << " :: ";
  size_t i = size > 10 ? 10 : size;
  for (size_t j = 0; j < i; ++j)
    std::cout << value[j] << " ";
  std::cout << "\n";
}

template <typename T>
void debug_print(const std::string &msg, const std::vector<T> &value) {
  auto size = value.size();
  std::cout << msg << " - size:" << size << " :: ";
  size_t i = size > 10 ? 10 : size;
  for (size_t j = 0; j < i; ++j)
    std::cout << value[j] << " ";
  std::cout << "\n";
}

void debug_print(const std::string &msg, float value);
void debug_print(const std::string &msg, double value);
void debug_print(const std::string &msg, int64_t value);
void debug_print(const std::string &msg, size_t value);
void debug_print(const std::string &msg, int64_t iter, int64_t end);
void debug_print(const std::string &msg, size_t i, size_t j, size_t k, float pa,
                 float pb, float val);
void debug_print(const std::string &msg, size_t i, size_t j, size_t k,
                 double pa, double pb, double val);

inline void MakeStringInternal(std::ostringstream &ss) noexcept {}

template <typename T>
inline void MakeStringInternal(std::ostringstream &ss, const T &t) noexcept {
  ss << t;
}

template <>
inline void MakeStringInternal(std::ostringstream &ss,
                               const std::vector<int32_t> &t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <>
inline void MakeStringInternal(std::ostringstream &ss,
                               const std::vector<uint32_t> &t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <>
inline void MakeStringInternal(std::ostringstream &ss,
                               const std::vector<int64_t> &t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <>
inline void MakeStringInternal(std::ostringstream &ss,
                               const std::vector<uint64_t> &t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <>
inline void MakeStringInternal(std::ostringstream &ss,
                               const std::vector<int16_t> &t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <>
inline void MakeStringInternal(std::ostringstream &ss,
                               const std::vector<uint16_t> &t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <typename T, typename... Args>
inline void MakeStringInternal(std::ostringstream &ss, const T &t,
                               const Args &...args) noexcept {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args> inline std::string MakeString(const Args &...args) {
  std::ostringstream ss;
  MakeStringInternal(ss, args...);
  return std::string(ss.str());
}

#if !defined(_THROW_DEFINED)
#define EXT_THROW(...) throw std::runtime_error(MakeString(__VA_ARGS__));
#define _THROW_DEFINED
#endif

#if !defined(_ENFORCE_DEFINED)
#define EXT_ENFORCE(cond, ...)                                                    \
  if (!(cond))                                                                 \
    throw std::runtime_error(                                                  \
        MakeString("`", #cond, "` failed. ", MakeString(__VA_ARGS__)));
#define _ENFORCE_DEFINED
#endif

} // namespace onnx_c_ops
