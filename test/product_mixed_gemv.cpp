// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

template <typename Lhs, typename Rhs>
void check_mixed_gemv(const Lhs& lhs, const Rhs& rhs) {
  using Scalar = typename Product<Lhs, Rhs>::Scalar;
  using Real = typename NumTraits<Scalar>::Real;
  using Wide = std::complex<long double>;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  const Vector initial = Vector::Random(lhs.rows());
  for (Scalar alpha : {Scalar(1), Scalar(Real(0.75), Real(-0.5)), Scalar(0)}) {
    for (Index stride : {Index(1), Index(2)}) {
      Vector backing = Vector::Constant(stride * lhs.rows() + 2, Scalar(19, 23));
      Map<Vector, Unaligned, InnerStride<Dynamic>> result(backing.data() + 1, lhs.rows(), InnerStride<Dynamic>(stride));
      for (int operation = 0; operation < 3; ++operation) {
        result = initial;
        if (operation == 0)
          result.noalias() = alpha * (lhs * rhs);
        else if (operation == 1)
          result.noalias() += alpha * (lhs * rhs);
        else
          result.noalias() -= alpha * (lhs * rhs);
        for (Index i = 0; i < lhs.rows(); ++i) {
          Wide sum(0);
          long double magnitude = 0;
          for (Index j = 0; j < lhs.cols(); ++j) {
            const Wide term = Wide(lhs.coeff(i, j)) * Wide(rhs.coeff(j));
            sum += term;
            magnitude += std::abs(term);
          }
          const Wide expected =
              (operation == 0 ? Wide(0) : Wide(initial(i))) + (operation == 2 ? -Wide(alpha) : Wide(alpha)) * sum;
          const long double bound = 16 * (lhs.cols() + 1) * NumTraits<Real>::epsilon() *
                                    (1 + std::abs(Wide(initial(i))) + std::abs(Wide(alpha)) * magnitude);
          VERIFY(std::abs(Wide(result(i)) - expected) <= bound);
        }
        VERIFY_IS_EQUAL(backing(0), Scalar(19, 23));
        VERIFY_IS_EQUAL(backing(backing.size() - 1), Scalar(19, 23));
        if (stride == 2)
          for (Index i = 2; i < backing.size(); i += 2) VERIFY_IS_EQUAL(backing(i), Scalar(19, 23));
      }
    }
  }
}

template <typename Scalar, int Order>
struct MixedGemvValueMapper {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  const MatrixType& matrix;
  Index outer_stride;
  Scalar operator()(Index i, Index j) const { return matrix(i, j); }
  Index stride() const { return outer_stride; }
  template <typename Packet, int Alignment>
  Packet load(Index i, Index j) const {
    Scalar values[internal::unpacket_traits<Packet>::size];
    for (Index k = 0; k < internal::unpacket_traits<Packet>::size; ++k)
      values[k] = matrix(i + (Order == ColMajor ? k : 0), j + (Order == RowMajor ? k : 0));
    return internal::ploadu<Packet>(values);
  }
};

template <typename Real, int Order, bool RealLeft>
void mixed_gemv_value_mapper() {
  using Complex = std::complex<Real>;
  using LhsScalar = std::conditional_t<RealLeft, Real, Complex>;
  using RhsScalar = std::conditional_t<RealLeft, Complex, Real>;
  using LhsMapper = MixedGemvValueMapper<LhsScalar, Order>;
  using RhsMapper = MixedGemvValueMapper<RhsScalar, Order == ColMajor ? RowMajor : ColMajor>;
  const typename LhsMapper::MatrixType lhs = LhsMapper::MatrixType::Random(9, 7);
  const typename RhsMapper::MatrixType rhs = RhsMapper::MatrixType::Random(7, 1);
  Matrix<Complex, 9, 1> result = Matrix<Complex, 9, 1>::Zero();
  using Alpha = std::conditional_t<Order == ColMajor, RhsScalar, Complex>;
  internal::general_matrix_vector_product<Index, LhsScalar, LhsMapper, Order, true, RhsScalar, RhsMapper, true>::run(
      9, 7, LhsMapper{lhs, lhs.outerStride()}, RhsMapper{rhs, 1}, result.data(), 1, Alpha(1));
  for (Index i = 0; i < 9; ++i) {
    std::complex<long double> reference(0);
    long double magnitude = 0;
    for (Index j = 0; j < 7; ++j) {
      const auto term =
          std::conj(std::complex<long double>(lhs(i, j))) * std::conj(std::complex<long double>(rhs(j, 0)));
      reference += term;
      magnitude += std::abs(term);
    }
    VERIFY(std::abs(std::complex<long double>(result(i)) - reference) <=
           128 * NumTraits<Real>::epsilon() * (1 + magnitude));
  }
}

template <typename Real, int Order, bool RealLeft>
void mixed_gemv_layout() {
  using Complex = std::complex<Real>;
  using LhsScalar = std::conditional_t<RealLeft, Real, Complex>;
  using RhsScalar = std::conditional_t<RealLeft, Complex, Real>;
  using Lhs = Matrix<LhsScalar, Dynamic, Dynamic, Order>;
  using Rhs = Matrix<RhsScalar, Dynamic, 1>;
  for (Index rows : {Index(0), Index(1), Index(3), Index(8), Index(17), Index(33)}) {
    for (Index cols : {Index(0), Index(1), Index(3), Index(7), Index(8), Index(17), Index(129)}) {
      const Lhs lhs = Lhs::Random(rows, cols);
      const Rhs rhs = Rhs::Random(cols);
      check_mixed_gemv(lhs, rhs);
      // Conjugating the real operand is the identity, so only the complex one varies.
      EIGEN_IF_CONSTEXPR (RealLeft) {
        check_mixed_gemv(lhs, rhs.conjugate());
      } else {
        check_mixed_gemv(lhs.conjugate(), rhs);
      }
      Rhs storage = Rhs::Zero(2 * cols + 1);
      Map<Rhs, Unaligned, InnerStride<2>> strided_rhs(storage.data(), cols);
      strided_rhs = rhs;
      check_mixed_gemv(lhs, strided_rhs);
      Lhs padded = Lhs::Zero(rows + 2, cols + 2);
      padded.block(1, 1, rows, cols) = lhs;
      check_mixed_gemv(padded.block(1, 1, rows, cols), rhs);
    }
  }
  Matrix<LhsScalar, 8, 8, Order> fixed = Matrix<LhsScalar, 8, 8, Order>::Random();
  const Matrix<RhsScalar, 8, 1> fixed_rhs = Matrix<RhsScalar, 8, 1>::Random();
  check_mixed_gemv(fixed, fixed_rhs);

  fixed.setConstant(LhsScalar(NumTraits<Real>::quiet_NaN()));
  Matrix<Complex, 8, 1> result = Matrix<Complex, 8, 1>::Constant(Complex(7));
  using LhsMapper = internal::const_blas_data_mapper<LhsScalar, Index, Order>;
  using RhsMapper = internal::const_blas_data_mapper<RhsScalar, Index, Order == ColMajor ? RowMajor : ColMajor>;
  using Alpha = std::conditional_t<Order == ColMajor, RhsScalar, Complex>;
  internal::general_matrix_vector_product<Index, LhsScalar, LhsMapper, Order, false, RhsScalar, RhsMapper, false>::run(
      8, 8, LhsMapper(fixed.data(), fixed.outerStride()), RhsMapper(fixed_rhs.data(), 1), result.data(), 1, Alpha(0));
  VERIFY((result.array() == Complex(7)).all());
  EIGEN_IF_CONSTEXPR (!RealLeft) {
    result.noalias() += Complex(0) * (fixed * fixed_rhs);
    VERIFY((result.array() == Complex(7)).all());
  }
#if !defined(EIGEN_VECTORIZE_ALTIVEC) && !defined(EIGEN_VECTORIZE_VSX)
  // Dedicated AltiVec GEMV specializations require BLAS-style mappers.
  mixed_gemv_value_mapper<Real, Order, RealLeft>();
#endif
}

template <typename Real>
void mixed_gemv() {
  using Complex = std::complex<Real>;
  STATIC_CHECK((bool(internal::gemv_traits<Complex, Real>::Vectorizable) ==
                bool(internal::packet_traits<Complex>::Vectorizable)));
  STATIC_CHECK((bool(internal::gemv_traits<Real, Complex>::Vectorizable) ==
                bool(internal::packet_traits<Complex>::Vectorizable)));
  mixed_gemv_layout<Real, ColMajor, false>();
  mixed_gemv_layout<Real, ColMajor, true>();
  mixed_gemv_layout<Real, RowMajor, false>();
  mixed_gemv_layout<Real, RowMajor, true>();
}

EIGEN_DECLARE_TEST(product_mixed_gemv) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1(mixed_gemv<float>());
    CALL_SUBTEST_2(mixed_gemv<double>());
  }
}
