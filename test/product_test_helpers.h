// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_PRODUCT_TEST_HELPERS_H
#define EIGEN_TEST_PRODUCT_TEST_HELPERS_H

#include <iostream>
#include <vector>

#include <Eigen/Core>

namespace Eigen {

// Rounding error bounds for matrix products, based on:
//
//   Deterministic: Higham, "Accuracy and Stability of Numerical Algorithms",
//     Thm 3.5: |fl(A*B) - A*B| <= gamma_k * |A| * |B|,  gamma_k ~ k * epsilon.
//
//   Probabilistic: Higham & Mary, "A New Approach to Probabilistic Rounding
//     Error Analysis", SISC 2019, Thm 3.4: under the assumption that rounding
//     errors are independent with mean zero:
//       |fl(A*B) - A*B| <= gamma_tilde_k * |A| * |B|,
//       gamma_tilde_k ~ lambda * sqrt(k) * epsilon,
//     holding with probability >= 1 - 2*exp(-lambda^2/2) per inner product.
//
// The bound is absolute rather than relative on purpose: || |A|*|B| ||_F /
// ||A*B||_F is unbounded, so no multiple of epsilon bounds the relative error
// of a product whose result cancels. Random operands do reach that regime — for
// A = m1*m1^T, B = m2, the result vanishes as m1^T*m2 does.
//
// Parameters:
//   num_products: number of independent products contributing error (default 1).
//                 Use 2 when comparing two different evaluations of A*B.
//   lambda:       probability parameter; P(lambda) = 1 - 2*exp(-lambda^2/2).
//                 lambda=5 gives P > 0.9999 per inner product.

// Returns (|A|*|B|)_ij.
//
// The bounds are accumulated in double with an explicit loop rather than formed as
// (A.cwiseAbs() * B.cwiseAbs()): a bound built with the product implementation would inherit
// that implementation's defects, and one that overflowed to infinity would admit any result at all.
template <typename DerivedA, typename DerivedB>
double abs_inner_product(const MatrixBase<DerivedA>& A, const MatrixBase<DerivedB>& B, Index i, Index j) {
  double sum = 0.0;
  for (Index l = 0; l < A.cols(); ++l)
    sum += static_cast<double>(numext::abs(A.coeff(i, l))) * static_cast<double>(numext::abs(B.coeff(l, j)));
  return sum;
}

// Returns || |alpha|*|A|*|B| + |C0| ||_F, with |C0| given coefficient-wise by c0(i, j).
template <typename DerivedA, typename DerivedB, typename C0Coeff>
double abs_product_update_norm(double abs_alpha, const MatrixBase<DerivedA>& A, const MatrixBase<DerivedB>& B,
                               C0Coeff c0) {
  double squared_norm = 0.0;
  for (Index i = 0; i < A.rows(); ++i)
    for (Index j = 0; j < B.cols(); ++j) {
      const double sum = abs_alpha * abs_inner_product(A, B, i, j) + c0(i, j);
      squared_norm += sum * sum;
    }
  return numext::sqrt(squared_norm);
}

// The type a product sums in: Accumulator, or the operand scalar when Accumulator is void.
template <typename Scalar, typename Accumulator>
using product_accumulator_t = std::conditional_t<std::is_void<Accumulator>::value, Scalar, Accumulator>;

// Returns lambda * sqrt(k) * epsilon * num_products * || |A|*|B| ||_F, with epsilon that of the
// accumulator (by default the operand scalar) but the value kept in double: narrowed to half, a
// triple-product bound already overflows at k = 100 for all-one operands.
template <typename Accumulator = void, typename DerivedA, typename DerivedB>
double product_error_bound(const MatrixBase<DerivedA>& A, const MatrixBase<DerivedB>& B, int num_products = 1,
                           double lambda = 5) {
  using Scalar = product_accumulator_t<typename DerivedA::Scalar, Accumulator>;
  const double norm = abs_product_update_norm(1.0, A, B, [](Index, Index) { return 0.0; });
  return lambda * numext::sqrt(double(A.cols())) * static_cast<double>(NumTraits<Scalar>::epsilon()) *
         double(num_products) * norm;
}

// Returns, for C = C0 + alpha*A*B (for beta*C0 + alpha*A*B, pass beta*C0 as C0),
//   lambda * sqrt(k+2) * epsilon * num_products * || |alpha|*|A|*|B| + |C0| ||_F.
// k+2: a k-term inner product, then the alpha scale and the sum with c0 (real arithmetic; complex
// scalars use the same depth by extension, as in product_error_bound). The |C0| term is needed: a
// bound on A*B alone fails once |alpha|*|A|*|B| << |C0|.
template <typename Accumulator = void, typename DerivedA, typename DerivedB, typename DerivedC0>
double product_update_error_bound(const typename DerivedA::Scalar& alpha, const MatrixBase<DerivedA>& A,
                                  const MatrixBase<DerivedB>& B, const MatrixBase<DerivedC0>& C0, int num_products = 1,
                                  double lambda = 5) {
  using Scalar = product_accumulator_t<typename DerivedA::Scalar, Accumulator>;
  eigen_assert(C0.rows() == A.rows() && C0.cols() == B.cols());
  const double norm = abs_product_update_norm(static_cast<double>(numext::abs(alpha)), A, B, [&C0](Index i, Index j) {
    return static_cast<double>(numext::abs(C0.coeff(i, j)));
  });
  return lambda * numext::sqrt(double(A.cols() + 2)) * static_cast<double>(NumTraits<Scalar>::epsilon()) *
         double(num_products) * norm;
}

// Returns ||M||_F accumulated in double, so that it overflows only when the norm itself does.
template <typename Derived>
double frobenius_norm_in_double(const MatrixBase<Derived>& M) {
  double squared_norm = 0.0;
  for (Index j = 0; j < M.cols(); ++j)
    for (Index i = 0; i < M.rows(); ++i) {
      const double d = static_cast<double>(numext::abs(M.coeff(i, j)));
      squared_norm += d * d;
    }
  return numext::sqrt(squared_norm);
}

// A product that sums in an Accumulator wider than its operand Scalar rounds each result coefficient
// once to Scalar, with error at most u|e| + denorm_min/2 (u = epsilon/2 of Scalar; the second term
// covers subnormal results): num_products * (u ||expected||_F + denorm_min/2 sqrt(mn)) more. Summed
// in Scalar, that rounding is one of the k+2 operations the bound already counts.
template <typename Scalar, typename Accumulator, typename Derived>
double result_rounding_bound(const MatrixBase<Derived>& expected, int num_products) {
  if (std::is_same<product_accumulator_t<Scalar, Accumulator>, Scalar>::value) return 0.0;
  return 0.5 * double(num_products) *
         (static_cast<double>(NumTraits<Scalar>::epsilon()) * frobenius_norm_in_double(expected) +
          static_cast<double>((std::numeric_limits<typename NumTraits<Scalar>::Real>::denorm_min)()) *
              numext::sqrt(double(expected.size())));
}

// Returns true if ||actual - expected||_F <= bound.
//
// The error norm is accumulated in double like the bound, so that it overflows only when the
// difference itself does. A non-finite error or bound fails: inf <= inf would otherwise accept
// an overflowed result.
template <typename D1, typename D2>
bool verify_within_product_bound(const MatrixBase<D1>& actual, const MatrixBase<D2>& expected, double bound) {
  const double error = frobenius_norm_in_double((actual - expected).eval());
  if (!((numext::isfinite)(bound) && error <= bound)) {
    std::cerr << "Product verification failed: error " << error << " exceeds bound " << bound << std::endl;
    return false;
  }
  return true;
}

// Verify that two computations of A*B agree within the Higham-Mary bound.
// Returns true if ||actual - expected||_F <= product_error_bound<Accumulator>(A, B, ...), plus the
// result_rounding_bound when Accumulator is wider than the operand scalar. Against an exact
// expected value, pass num_products = 1.
template <typename Accumulator = void, typename D1, typename D2, typename DA, typename DB>
inline bool verifyProduct(const MatrixBase<D1>& actual, const MatrixBase<D2>& expected, const MatrixBase<DA>& A,
                          const MatrixBase<DB>& B, int num_products = 2, double lambda = 5) {
  return verify_within_product_bound(
      actual, expected,
      product_error_bound<Accumulator>(A, B, num_products, lambda) +
          result_rounding_bound<typename DA::Scalar, Accumulator>(expected, num_products));
}

// Verify that two computations of C0 + alpha*A*B agree within the Higham-Mary bound.
// Returns true if ||actual - expected||_F <= product_update_error_bound<Accumulator>(alpha, A, B, C0,
// ...), plus the result_rounding_bound when Accumulator is wider than the operand scalar.
template <typename Accumulator = void, typename D1, typename D2, typename DA, typename DB, typename DC0>
bool verifyProductUpdate(const MatrixBase<D1>& actual, const MatrixBase<D2>& expected, const typename DA::Scalar& alpha,
                         const MatrixBase<DA>& A, const MatrixBase<DB>& B, const MatrixBase<DC0>& C0,
                         int num_products = 2, double lambda = 5) {
  return verify_within_product_bound(
      actual, expected,
      product_update_error_bound<Accumulator>(alpha, A, B, C0, num_products, lambda) +
          result_rounding_bound<typename DA::Scalar, Accumulator>(expected, num_products));
}

// Verify C = C0 + alpha*A*B coefficient-wise against the Higham-Mary bound of each inner product:
//   |actual_ij - expected_ij| <= num_products * lambda * sqrt(k+2) * epsilon * (|alpha|*|A|*|B| + |C0|)_ij.
// Unlike verifyProductUpdate, which bounds the Frobenius norm of the whole error, one wrong coefficient cannot hide
// in the slack of the others. For products that sum in the operand scalar.
template <typename D1, typename D2, typename DA, typename DB, typename DC0>
bool verifyProductUpdateCoeffwise(const MatrixBase<D1>& actual, const MatrixBase<D2>& expected,
                                  const typename DA::Scalar& alpha, const MatrixBase<DA>& A, const MatrixBase<DB>& B,
                                  const MatrixBase<DC0>& C0, int num_products = 2, double lambda = 5) {
  using Scalar = typename DA::Scalar;
  eigen_assert(actual.rows() == A.rows() && actual.cols() == B.cols());
  const double gamma = double(num_products) * lambda * numext::sqrt(double(A.cols() + 2)) *
                       static_cast<double>(NumTraits<Scalar>::epsilon());
  const double abs_alpha = static_cast<double>(numext::abs(alpha));
  for (Index j = 0; j < actual.cols(); ++j)
    for (Index i = 0; i < actual.rows(); ++i) {
      const double bound =
          gamma * (abs_alpha * abs_inner_product(A, B, i, j) + static_cast<double>(numext::abs(C0.coeff(i, j))));
      const double error = static_cast<double>(numext::abs(actual.coeff(i, j) - expected.coeff(i, j)));
      if (!((numext::isfinite)(bound) && error <= bound)) {
        std::cerr << "Product verification failed at (" << i << ", " << j << "): error " << error << " exceeds bound "
                  << bound << std::endl;
        return false;
      }
    }
  return true;
}

// Bitwise equality of every coefficient, so that +0 and -0 differ and NaNs with equal payloads agree.
template <typename A, typename B>
bool bitwise_equal(const DenseBase<A>& a, const DenseBase<B>& b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
  for (Index j = 0; j < a.cols(); ++j)
    for (Index i = 0; i < a.rows(); ++i)
      if (!test::biteq(a.derived().coeff(i, j), b.derived().coeff(i, j))) return false;
  return true;
}

// Returns true if every coefficient of `backing` outside the block that a strided Map placed on it addresses
// compares equal to `original`. The block's coefficient (inner i, outer o) sits at
// offset + i * inner_stride + o * outer_stride (a vector is a block with outer_size 1). Pass a constant
// expression as `original` for a sentinel fill.
template <typename Backing, typename Original>
bool strided_block_gaps_untouched(const DenseBase<Backing>& backing, const DenseBase<Original>& original, Index offset,
                                  Index inner_size, Index outer_size, Index inner_stride, Index outer_stride) {
  eigen_assert(backing.size() == original.size());
  std::vector<bool> addressed(static_cast<std::size_t>(backing.size()), false);
  for (Index o = 0; o < outer_size; ++o)
    for (Index i = 0; i < inner_size; ++i)
      addressed[static_cast<std::size_t>(offset + i * inner_stride + o * outer_stride)] = true;
  for (Index k = 0; k < backing.size(); ++k)
    if (!addressed[static_cast<std::size_t>(k)] && !(backing.derived().coeff(k) == original.derived().coeff(k))) {
      std::cerr << "Strided block verification failed: coefficient " << k << " outside the block was written"
                << std::endl;
      return false;
    }
  return true;
}

// y = alpha lhs rhs, y += ... and y -= ..., into contiguous and strided destinations, against a long double reference:
//   |y_i - ref_i| <= factor eps (|c0_i| + |alpha| sum_j |lhs_ij rhs_j|) + (2k + 3) denorm_min,  k = lhs.cols(),
// with factor the caller's error model; the second term covers subnormal roundings (two per term, alpha, c0).
template <typename Lhs, typename Rhs>
bool gemv_matches_long_double(const Lhs& lhs, const Rhs& rhs, double factor) {
  using Scalar = typename ScalarBinaryOpTraits<typename Lhs::Scalar, typename Rhs::Scalar>::ReturnType;
  using Real = typename NumTraits<Scalar>::Real;
  using Wide = std::conditional_t<NumTraits<Scalar>::IsComplex, std::complex<long double>, long double>;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  const Index rows = lhs.rows();
  const Scalar sentinel(7);
  const Vector initial = Vector::Random(rows);
  const long double epsilon = static_cast<long double>(NumTraits<Real>::epsilon());
  const long double underflow = static_cast<long double>(2 * lhs.cols() + 3) *
                                static_cast<long double>((std::numeric_limits<Real>::denorm_min)());
  bool ok = true;
  for (const Scalar alpha : {Scalar(1), internal::random<Scalar>(), Scalar(0)}) {
    for (int operation = 0; operation < 3; ++operation) {
      std::vector<Wide> expected(static_cast<std::size_t>(rows));
      std::vector<long double> bound(static_cast<std::size_t>(rows));
      for (Index i = 0; i < rows; ++i) {
        Wide sum(0);
        long double magnitude = 0;
        for (Index j = 0; j < lhs.cols(); ++j) {
          const Wide term = internal::cast<typename Lhs::Scalar, Wide>(lhs.coeff(i, j)) *
                            internal::cast<typename Rhs::Scalar, Wide>(rhs.coeff(j, 0));
          sum += term;
          magnitude += numext::abs(term);
        }
        const Wide c0 = operation == 0 ? Wide(0) : internal::cast<Scalar, Wide>(initial(i));
        const Wide wide_alpha = internal::cast<Scalar, Wide>(alpha);
        expected[static_cast<std::size_t>(i)] = c0 + (operation == 2 ? -wide_alpha : wide_alpha) * sum;
        bound[static_cast<std::size_t>(i)] =
            static_cast<long double>(factor) * epsilon * (numext::abs(c0) + numext::abs(wide_alpha) * magnitude) +
            underflow;
      }
      auto matches = [&](const auto& result) {
        for (Index i = 0; i < rows; ++i)
          if (!(numext::abs(internal::cast<Scalar, Wide>(result(i)) - expected[static_cast<std::size_t>(i)]) <=
                bound[static_cast<std::size_t>(i)])) {
            std::cerr << "GEMV verification failed at row " << i << std::endl;
            return false;
          }
        return true;
      };
      for (int form = 0; form < 2; ++form) {
        auto evaluate = [&](auto& target) {
          target = initial;
          if (operation == 0) {
            if (form == 0)
              target.noalias() = alpha * (lhs * rhs);
            else
              target.noalias() = alpha * lhs * rhs;
          } else if (operation == 1) {
            if (form == 0)
              target.noalias() += alpha * (lhs * rhs);
            else
              target.noalias() += alpha * lhs * rhs;
          } else {
            if (form == 0)
              target.noalias() -= alpha * (lhs * rhs);
            else
              target.noalias() -= alpha * lhs * rhs;
          }
        };
        Vector contiguous(rows);
        evaluate(contiguous);
        ok = ok && matches(contiguous);
        // One extra coefficient before the destination and one after, or two with stride 2.
        for (Index stride : {Index(1), Index(2)}) {
          Vector backing = Vector::Constant(stride * rows + 2, sentinel);
          Map<Vector, Unaligned, InnerStride<Dynamic>> dynamic_stride(backing.data() + 1, rows,
                                                                      InnerStride<Dynamic>(stride));
          evaluate(dynamic_stride);
          ok = ok && matches(dynamic_stride) &&
               strided_block_gaps_untouched(backing, Vector::Constant(backing.size(), sentinel), 1, rows, 1, stride, 0);
        }
        Vector backing = Vector::Constant(2 * rows + 2, sentinel);
        Map<Vector, Unaligned, InnerStride<2>> static_stride(backing.data() + 1, rows);
        evaluate(static_stride);
        ok = ok && matches(static_stride) &&
             strided_block_gaps_untouched(backing, Vector::Constant(backing.size(), sentinel), 1, rows, 1, 2, 0);
      }
    }
  }
  return ok;
}

}  // namespace Eigen

#endif  // EIGEN_TEST_PRODUCT_TEST_HELPERS_H
