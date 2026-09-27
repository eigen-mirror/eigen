// SPDX-FileCopyrightText: 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// SPDX-FileCopyrightText: 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_NUMERICAL_TEST_HELPERS_H
#define EIGEN_TEST_NUMERICAL_TEST_HELPERS_H

#include <algorithm>
#include <complex>
#include <iostream>
#include <type_traits>

#include <Eigen/Core>

#define VERIFY_IS_EQUAL(a, b) VERIFY(test_is_equal(a, b, true))
#define VERIFY_IS_NOT_EQUAL(a, b) VERIFY(test_is_equal(a, b, false))
#define VERIFY_IS_APPROX(a, b) VERIFY(verifyIsApprox(a, b))
#define VERIFY_IS_NOT_APPROX(a, b) VERIFY(!test_isApprox(a, b))
#define VERIFY_IS_MUCH_SMALLER_THAN(a, b) VERIFY(test_isMuchSmallerThan(a, b))
#define VERIFY_IS_NOT_MUCH_SMALLER_THAN(a, b) VERIFY(!test_isMuchSmallerThan(a, b))
#define VERIFY_IS_APPROX_OR_LESS_THAN(a, b) VERIFY(test_isApproxOrLessThan(a, b))
#define VERIFY_IS_CWISE_EQUAL(a, b) VERIFY(verifyIsCwiseApprox(a, b, true))
#define VERIFY_IS_CWISE_APPROX(a, b) VERIFY(verifyIsCwiseApprox(a, b, false))
#define VERIFY_IS_APPROX_SCALED(a, b, scale) VERIFY(verifyIsApproxScaled(a, b, scale))

#define VERIFY_IS_UNITARY(a) VERIFY(test_isUnitary(a))

namespace Eigen {

#if EIGEN_COMP_ICC
template <typename T, typename U>
bool test_is_equal(const T& actual, const U& expected, bool expect_equal = true);
#endif

// Legacy defaults for broad consistency checks, not accuracy bounds. Numerical tests should use explicit
// bounds derived from epsilon, the operation count and conditioning, or a documented ULP budget.
template <typename T>
inline typename NumTraits<T>::Real test_precision() {
  return NumTraits<T>::dummy_precision();
}
template <>
inline float test_precision<float>() {
  return 1e-3f;
}
template <>
inline double test_precision<double>() {
  return 1e-6;
}
template <>
inline long double test_precision<long double>() {
  return 1e-6l;
}
template <>
inline float test_precision<std::complex<float>>() {
  return test_precision<float>();
}
template <>
inline double test_precision<std::complex<double>>() {
  return test_precision<double>();
}
template <>
inline long double test_precision<std::complex<long double>>() {
  return test_precision<long double>();
}

#define EIGEN_TEST_SCALAR_TEST_OVERLOAD(TYPE)                                                                    \
  inline bool test_isCwiseApprox(TYPE a, TYPE b, bool exact,                                                     \
                                 typename NumTraits<TYPE>::Real precision = test_precision<TYPE>()) {            \
    return numext::equal_strict(a, b) || ((numext::isnan)(a) && (numext::isnan)(b)) ||                           \
           (!exact && (numext::isfinite)(a) && (numext::isfinite)(b) && internal::isApprox(a, b, precision));    \
  }                                                                                                              \
  inline bool test_isApprox(TYPE a, TYPE b) { return test_isCwiseApprox(a, b, false); }                          \
  inline bool test_isMuchSmallerThan(TYPE a, TYPE b) {                                                           \
    return (numext::isfinite)(a) && internal::isMuchSmallerThan(a, b, test_precision<TYPE>());                   \
  }                                                                                                              \
  inline bool test_isApproxOrLessThan(TYPE a, TYPE b) {                                                          \
    return a <= b ||                                                                                             \
           ((numext::isfinite)(a) && (numext::isfinite)(b) && internal::isApprox(a, b, test_precision<TYPE>())); \
  }

EIGEN_TEST_SCALAR_TEST_OVERLOAD(short)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(bool)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(unsigned short)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(int)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(unsigned int)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(long)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(unsigned long)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(long long)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(unsigned long long)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(float)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(double)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(half)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(bfloat16)

#ifndef EIGEN_TEST_NO_LONGDOUBLE
EIGEN_TEST_SCALAR_TEST_OVERLOAD(long double)
#endif

#undef EIGEN_TEST_SCALAR_TEST_OVERLOAD

#ifndef EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_COMPLEX_TEST_OVERLOAD(REAL)                                                               \
  inline bool test_isApprox(const std::complex<REAL>& a, const std::complex<REAL>& b) {                      \
    using Vector = Matrix<std::complex<REAL>, 1, 1>;                                                         \
    return Vector::Constant(a).isApprox(Vector::Constant(b), test_precision<std::complex<REAL>>());          \
  }                                                                                                          \
  inline bool test_isMuchSmallerThan(const std::complex<REAL>& a, const std::complex<REAL>& b) {             \
    using Vector = Matrix<std::complex<REAL>, 1, 1>;                                                         \
    return Vector::Constant(a).isMuchSmallerThan(Vector::Constant(b), test_precision<std::complex<REAL>>()); \
  }                                                                                                          \
  inline bool test_isCwiseApprox(const std::complex<REAL>& a, const std::complex<REAL>& b, bool exact,       \
                                 REAL precision = test_precision<std::complex<REAL>>()) {                    \
    using Vector = Matrix<std::complex<REAL>, 1, 1>;                                                         \
    return numext::equal_strict(a, b) || ((numext::isnan)(a) && (numext::isnan)(b)) ||                       \
           (!exact && Vector::Constant(a).isApprox(Vector::Constant(b), precision));                         \
  }

EIGEN_TEST_COMPLEX_TEST_OVERLOAD(float)
EIGEN_TEST_COMPLEX_TEST_OVERLOAD(double)
#ifndef EIGEN_TEST_NO_LONGDOUBLE
EIGEN_TEST_COMPLEX_TEST_OVERLOAD(long double)
#endif
#undef EIGEN_TEST_COMPLEX_TEST_OVERLOAD
#endif

template <typename Scalar, bool Scaled>
struct test_relative_error_impl {
  template <typename X, typename Y>
  static typename NumTraits<Scalar>::Real run(const X& a, const Y& b) {
    return numext::sqrt((a.matrix() - b.matrix()).cwiseAbs2().sum() /
                        numext::mini(a.cwiseAbs2().sum(), b.cwiseAbs2().sum()));
  }
};

template <typename Scalar>
struct test_relative_error_impl<Scalar, true> {
  template <typename X, typename Y>
  static typename NumTraits<Scalar>::Real run(const X& a, const Y& b) {
    using Real = typename NumTraits<Scalar>::Real;
    const auto na = internal::scaled_comparison_norm(a);
    const auto nb = internal::scaled_comparison_norm(b);
    const auto denominator = na <= nb ? na : nb;
    const auto difference = internal::scaled_comparison_distance(a, b);
    if (difference.fraction == 0 && denominator.fraction == 0) return Real(0);
    return Real(numext::ldexp(difference.fraction / denominator.fraction, difference.exponent - denominator.exponent));
  }
};

// Promote integers before subtraction and squaring, avoiding Boolean subtraction and integer overflow. Other
// scalars keep their type: a custom complex scalar can inherit a real NonInteger from NumTraits<Real>.
template <typename Scalar>
using test_difference_scalar_t =
    std::conditional_t<NumTraits<Scalar>::IsInteger, typename NumTraits<Scalar>::NonInteger, Scalar>;

// test_relative_error returns the relative difference between a and b as a real scalar as used in isApprox.
template <typename T1, typename T2>
typename NumTraits<typename T1::RealScalar>::NonInteger test_relative_error(const EigenBase<T1>& a,
                                                                            const EigenBase<T2>& b) {
  using DiffScalar1 = test_difference_scalar_t<typename T1::Scalar>;
  using DiffScalar2 = test_difference_scalar_t<typename T2::Scalar>;
  typename internal::nested_eval<T1, 2>::type ea(a.derived());
  typename internal::nested_eval<T2, 2>::type eb(b.derived());
  // Exponent scaling needs binary floating-point on both sides.
  constexpr bool kScaled =
      internal::use_scaled_comparison<DiffScalar1>::value && internal::use_scaled_comparison<DiffScalar2>::value;
  return test_relative_error_impl<DiffScalar1, kScaled>::run(ea.template cast<DiffScalar1>(),
                                                             eb.template cast<DiffScalar2>());
}

template <typename T1, typename T2>
typename T1::RealScalar test_relative_error(const T1& a, const T2& b, const typename T1::Coefficients* = 0) {
  return test_relative_error(a.coeffs(), b.coeffs());
}

template <typename T1, typename T2>
typename T1::Scalar test_relative_error(const T1& a, const T2& b, const typename T1::MatrixType* = 0) {
  return test_relative_error(a.matrix(), b.matrix());
}

template <typename S, int D>
S test_relative_error(const Translation<S, D>& a, const Translation<S, D>& b) {
  return test_relative_error(a.vector(), b.vector());
}

template <typename S, int D, int O>
S test_relative_error(const ParametrizedLine<S, D, O>& a, const ParametrizedLine<S, D, O>& b) {
  return (std::max)(test_relative_error(a.origin(), b.origin()), test_relative_error(a.direction(), b.direction()));
}

template <typename S, int D>
S test_relative_error(const AlignedBox<S, D>& a, const AlignedBox<S, D>& b) {
  return (std::max)(test_relative_error((a.min)(), (b.min)()), test_relative_error((a.max)(), (b.max)()));
}

template <typename Derived>
class SparseMatrixBase;
template <typename T1, typename T2>
typename T1::RealScalar test_relative_error(const MatrixBase<T1>& a, const SparseMatrixBase<T2>& b) {
  return test_relative_error(a, b.toDense());
}

template <typename Derived>
class SparseMatrixBase;
template <typename T1, typename T2>
typename T1::RealScalar test_relative_error(const SparseMatrixBase<T1>& a, const MatrixBase<T2>& b) {
  return test_relative_error(a.toDense(), b);
}

template <typename Derived>
class SparseMatrixBase;
template <typename T1, typename T2>
typename T1::RealScalar test_relative_error(const SparseMatrixBase<T1>& a, const SparseMatrixBase<T2>& b) {
  return test_relative_error(a.toDense(), b.toDense());
}

template <typename T1, typename T2,
          std::enable_if_t<internal::is_arithmetic<typename NumTraits<T1>::Real>::value, int> = 0>
typename NumTraits<typename NumTraits<T1>::Real>::NonInteger test_relative_error(const T1& a, const T2& b) {
  using Scalar = decltype(a - b);
  using Real = typename NumTraits<typename NumTraits<T1>::Real>::NonInteger;
  using Vector = Matrix<Scalar, 1, 1>;
  return Real(test_relative_error(Vector::Constant(Scalar(a)), Vector::Constant(Scalar(b))));
}

template <typename T>
T test_relative_error(const Rotation2D<T>& a, const Rotation2D<T>& b) {
  return test_relative_error(a.angle(), b.angle());
}

template <typename T>
T test_relative_error(const AngleAxis<T>& a, const AngleAxis<T>& b) {
  return (std::max)(test_relative_error(a.angle(), b.angle()), test_relative_error(a.axis(), b.axis()));
}

template <typename Type1, typename Type2>
inline bool test_isApprox(const Type1& a, const Type2& b, typename Type1::Scalar* = 0)  // Enabled for Eigen's type only
{
  return a.isApprox(b, test_precision<typename Type1::Scalar>());
}

// get_test_precision is a small wrapper to test_precision allowing to return the scalar precision for either scalars or
// expressions
template <typename T>
typename NumTraits<typename T::Scalar>::Real get_test_precision(const T&, const typename T::Scalar* = 0) {
  return test_precision<typename NumTraits<typename T::Scalar>::Real>();
}

template <typename T, std::enable_if_t<internal::is_arithmetic<typename NumTraits<T>::Real>::value, int> = 0>
typename NumTraits<T>::Real get_test_precision(const T&) {
  return test_precision<typename NumTraits<T>::Real>();
}

// verifyIsApprox is a wrapper to test_isApprox that outputs the relative difference magnitude if the test fails.
template <typename Type1, typename Type2>
inline bool verifyIsApprox(const Type1& a, const Type2& b) {
  bool ret = test_isApprox(a, b);
  if (!ret) {
    std::cerr << "Difference too large wrt tolerance " << get_test_precision(a)
              << ", relative error is: " << test_relative_error(a, b) << std::endl;
  }
  return ret;
}

// verifyIsCwiseApprox is a wrapper to test_isCwiseApprox that outputs the relative difference magnitude if the test
// fails.
template <typename Type1, typename Type2>
inline bool verifyIsCwiseApprox(const Type1& a, const Type2& b, bool exact) {
  bool ret = test_isCwiseApprox(a, b, exact);
  if (!ret) {
    if (exact) {
      std::cerr << "Values are not an exact match";
    } else {
      std::cerr << "Difference too large wrt tolerance " << get_test_precision(a);
    }
    std::cerr << ", relative error is: " << test_relative_error(a, b) << std::endl;
  }
  return ret;
}

// Largest coefficient magnitude of a matrix or of an array expression.
template <typename Derived>
typename NumTraits<typename Derived::Scalar>::Real max_abs_coeff(const DenseBase<Derived>& m) {
  using Real = typename NumTraits<typename Derived::Scalar>::Real;
  return m.size() == 0 ? Real(0) : m.derived().matrix().cwiseAbs().template maxCoeff<PropagateNaN>();
}

// Compares two expressions that are mathematically equal but whose evaluations differ by rounding proportional to
// `scale` rather than to the result. verifyIsApprox measures the error relative to the result, which no
// implementation can meet once the result is formed by cancellation.
template <typename Type1, typename Type2>
inline bool verifyIsApproxScaled(const Type1& a, const Type2& b,
                                 const typename NumTraits<typename Type1::Scalar>::Real& scale) {
  typedef typename NumTraits<typename Type1::Scalar>::Real RealScalar;
  const RealScalar error = max_abs_coeff((a - b).eval());
  const RealScalar tolerance = test_precision<typename Type1::Scalar>() * scale;
  // An infinite or NaN bound would admit any result, so treat it as a failure of the caller rather
  // than as a passing comparison.
  if (!((numext::isfinite)(tolerance))) {
    std::cerr << "Invalid tolerance " << tolerance << " for scale " << scale << std::endl;
    return false;
  }
  if (!(error <= tolerance)) {
    std::cerr << "Difference " << error << " too large wrt tolerance " << tolerance << std::endl;
    return false;
  }
  return true;
}

template <typename Derived1, typename Derived2>
inline bool test_isMuchSmallerThan(const DenseBase<Derived1>& m1, const DenseBase<Derived2>& m2) {
  return m1.isMuchSmallerThan(m2, test_precision<typename internal::traits<Derived1>::Scalar>());
}

template <typename Derived>
inline bool test_isMuchSmallerThan(const DenseBase<Derived>& m,
                                   const typename NumTraits<typename internal::traits<Derived>::Scalar>::Real& s) {
  return m.isMuchSmallerThan(s, test_precision<typename internal::traits<Derived>::Scalar>());
}

template <typename Derived>
inline bool test_isUnitary(const MatrixBase<Derived>& m) {
  return m.isUnitary(test_precision<typename internal::traits<Derived>::Scalar>());
}

// Checks component-wise, works with infs and nans. A scalar without a dedicated overload above, such as a custom
// complex type over a built-in real, delegates to its own test_isApprox.
template <typename Scalar,
          std::enable_if_t<
              !std::is_base_of<EigenBase<Scalar>, Scalar>::value && !internal::is_arithmetic<Scalar>::value, int> = 0>
bool test_isCwiseApprox(const Scalar& a, const Scalar& b, bool exact) {
  return numext::equal_strict(a, b) || ((numext::isnan)(a) && (numext::isnan)(b)) || (!exact && test_isApprox(a, b));
}

template <typename Derived1, typename Derived2>
bool test_isCwiseApprox(const DenseBase<Derived1>& m1, const DenseBase<Derived2>& m2, bool exact) {
  if (m1.rows() != m2.rows()) {
    return false;
  }
  if (m1.cols() != m2.cols()) {
    return false;
  }
  for (Index r = 0; r < m1.rows(); ++r) {
    for (Index c = 0; c < m1.cols(); ++c) {
      if (!test_isCwiseApprox(m1(r, c), m2(r, c), exact)) {
        return false;
      }
    }
  }
  return true;
}

template <typename Derived1, typename Derived2>
bool test_isCwiseApprox(const SparseMatrixBase<Derived1>& m1, const SparseMatrixBase<Derived2>& m2, bool exact) {
  return test_isCwiseApprox(m1.toDense(), m2.toDense(), exact);
}

template <typename T, typename U>
bool test_is_equal(const T& actual, const U& expected, bool expect_equal) {
  if (numext::equal_strict(actual, expected) == expect_equal) return true;
  // false:
  std::cerr << "\n    actual   = " << actual << "\n    expected " << (expect_equal ? "= " : "!=") << expected << "\n\n";
  return false;
}

/**
 * Check if number is "not a number" (NaN).
 *
 * @tparam T input type
 * @param x input value
 * @return true, if input value is "not a number" (NaN)
 */
template <typename T>
bool isNotNaN(const T& x) {
  return x == x;
}

/**
 * Check if number is plus infinity.
 *
 * @tparam T input type
 * @param x input value
 * @return true, if input value is plus infinity
 */
template <typename T>
bool isPlusInf(const T& x) {
  return x > NumTraits<T>::highest();
}

/**
 * Check if number is minus infinity.
 *
 * @tparam T input type
 * @param x input value
 * @return true, if input value is minus infinity
 */
template <typename T>
bool isMinusInf(const T& x) {
  return x < NumTraits<T>::lowest();
}

}  // end namespace Eigen

#endif  // EIGEN_TEST_NUMERICAL_TEST_HELPERS_H
