// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include "product_test_helpers.h"
#define EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW
#include "AnnoyingScalar.h"

template <typename LhsScalar, typename RhsScalar = LhsScalar>
void degenerate_gemv() {
  // Depth at most 4, plus the scalings of the operands and of alpha.
  const double factor = 32;
  using MatrixType = Matrix<LhsScalar, Dynamic, Dynamic>;
  using RhsVector = Matrix<RhsScalar, Dynamic, 1>;
  for (Index n : {0, 1, 3, 8, 17, 257}) {
    for (Index depth : {0, 1, 2, 4}) {
      MatrixType a = MatrixType::Random(depth, n);
      RhsVector rhsStorage = RhsVector::Random(3 * depth + 1);
      Map<RhsVector, Unaligned, InnerStride<3>> rhs(rhsStorage.data(), depth);
      VERIFY(gemv_matches_long_double(a.transpose(), rhs, factor));
      VERIFY(gemv_matches_long_double(a.adjoint(), rhs.conjugate(), factor));
      VERIFY(gemv_matches_long_double(LhsScalar(0.75) * a.adjoint(), RhsScalar(0.5) * rhs, factor));
      MatrixType columnMajor = a.transpose();
      VERIFY(gemv_matches_long_double(columnMajor, rhs, factor));
      VERIFY(gemv_matches_long_double(columnMajor.conjugate(), rhs.conjugate(), factor));
      VERIFY(gemv_matches_long_double(LhsScalar(0.75) * columnMajor.conjugate(), RhsScalar(0.5) * rhs, factor));
    }
    Matrix<LhsScalar, Dynamic, 1> lhsStorage = Matrix<LhsScalar, Dynamic, 1>::Random(3 * n + 1);
    using RowMatrix = Matrix<LhsScalar, Dynamic, Dynamic, RowMajor>;
    Map<RowMatrix, Unaligned, OuterStride<3>> lhs(lhsStorage.data(), n, 1);
    RhsVector rhs = RhsVector::Random(1);
    VERIFY(gemv_matches_long_double(lhs, rhs, factor));
    Map<RowMatrix, Unaligned, OuterStride<>> runtimeLhs(lhsStorage.data(), n, 1, OuterStride<>(3));
    VERIFY(gemv_matches_long_double(runtimeLhs, rhs, factor));
    Matrix<LhsScalar, 1, Dynamic> row = Matrix<LhsScalar, 1, Dynamic>::Random(n);
    VERIFY(gemv_matches_long_double(row.transpose(), rhs, factor));
    Matrix<LhsScalar, 2, Dynamic> fixedDepth = Matrix<LhsScalar, 2, Dynamic>::Random(2, n);
    RhsVector fixedDepthRhs = RhsVector::Random(2);
    VERIFY(gemv_matches_long_double(fixedDepth.transpose(), fixedDepthRhs, factor));
  }
}

template <typename Scalar, int StorageOrder>
void degenerate_gemv_zero_alpha() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  using Real = typename NumTraits<Scalar>::Real;
  MatrixType lhs(17, 1);
  Vector rhs(1);
  Vector backing(2 * lhs.rows() + 1);
  Map<Vector, Unaligned, InnerStride<2>> strided(backing.data(), lhs.rows());
  Map<Vector, Unaligned, InnerStride<Dynamic>> dynamicStrided(backing.data(), lhs.rows(), InnerStride<Dynamic>(2));
  Vector contiguous(lhs.rows());
  for (const Real exceptional : {NumTraits<Real>::quiet_NaN(), NumTraits<Real>::infinity()}) {
    for (int operand = 0; operand < 2; ++operand) {
      lhs.setConstant(operand == 0 ? Scalar(exceptional) : Scalar(2));
      rhs.setConstant(operand == 1 ? Scalar(exceptional) : Scalar(3));
      for (const Real zero : {Real(0), -Real(0)}) {
        for (int factor = 0; factor < 3; ++factor) {
          for (int operation = 0; operation < 2; ++operation) {
            auto check = [&](auto& destination) {
              backing.setConstant(Scalar(7));
              contiguous.setConstant(Scalar(7));
              auto update = [&](const auto& product) {
                if (operation == 0)
                  destination.noalias() += product;
                else
                  destination.noalias() -= product;
              };
              if (factor == 0) update(Scalar(zero) * (lhs * rhs));
              if (factor == 1) update((Scalar(zero) * lhs) * rhs);
              if (factor == 2) update(lhs * (Scalar(zero) * rhs));
              for (Index i = 0; i < destination.size(); ++i) VERIFY_IS_EQUAL(destination[i], Scalar(7));
              for (Index i = 1; i < backing.size(); i += 2) VERIFY_IS_EQUAL(backing[i], Scalar(7));
              VERIFY_IS_EQUAL(backing[backing.size() - 1], Scalar(7));
            };
            check(contiguous);
            check(strided);
            check(dynamicStrided);
          }
        }
      }
    }
  }
}

template <typename Scalar, int StorageOrder>
void degenerate_gemv_array_destinations() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  using ArrayVector = Array<Scalar, Dynamic, 1>;
  const Index size = 17;
  MatrixType lhs(size, 1);
  ArrayVector expected(size);
  for (Index i = 0; i < size; ++i) {
    lhs(i, 0) = Scalar(2 * (i + 1));
    expected[i] = Scalar(3 * (i + 1));
  }
  const Vector rhs = Vector::Constant(1, Scalar(3));
  const Scalar alpha(0.5);
  const ArrayVector contiguous = alpha * lhs * rhs;
  VERIFY((contiguous == expected).all());

  ArrayVector backing = ArrayVector::Constant(2 * size + 1, Scalar(7));
  Map<ArrayVector, Unaligned, InnerStride<2>> strided(backing.data(), size);
  Map<ArrayVector, Unaligned, InnerStride<Dynamic>> dynamicStrided(backing.data(), size, InnerStride<Dynamic>(2));
  auto check = [&](auto& destination) {
    backing.setConstant(Scalar(7));
    destination = alpha * lhs * rhs;
    VERIFY((destination == expected).all());
    for (Index i = 1; i < backing.size(); i += 2) VERIFY_IS_EQUAL(backing[i], Scalar(7));
    VERIFY_IS_EQUAL(backing[backing.size() - 1], Scalar(7));
  };
  check(strided);
  check(dynamicStrided);

  const Array<Scalar, 1, Dynamic> row = alpha * rhs.transpose() * lhs.transpose();
  VERIFY((row.transpose() == expected).all());
  Array<Scalar, Dynamic, Dynamic> rows = Array<Scalar, Dynamic, Dynamic>::Constant(2, size + 2, Scalar(7));
  rows.row(1).segment(1, size) = alpha * rhs.transpose() * lhs.transpose();
  VERIFY((rows.row(1).segment(1, size).transpose() == expected).all());
  VERIFY((rows.row(0) == Scalar(7)).all());
  VERIFY_IS_EQUAL(rows(1, 0), Scalar(7));
  VERIFY_IS_EQUAL(rows(1, size + 1), Scalar(7));
}

EIGEN_DECLARE_TEST(gemv_degenerate) {
  // The disabled C++14 dispatch must not require coefficient access or instantiate the update.
  int placeholder = 0;
  VERIFY((!internal::gemv_single_column<false, false, false>::run(placeholder, placeholder, placeholder, placeholder,
                                                                  true)));
  degenerate_gemv_array_destinations<float, RowMajor>();
  degenerate_gemv_array_destinations<float, ColMajor>();
  degenerate_gemv_array_destinations<double, RowMajor>();
  degenerate_gemv_array_destinations<double, ColMajor>();
  degenerate_gemv_array_destinations<std::complex<float>, RowMajor>();
  degenerate_gemv_array_destinations<std::complex<float>, ColMajor>();
  degenerate_gemv_array_destinations<std::complex<double>, RowMajor>();
  degenerate_gemv_array_destinations<std::complex<double>, ColMajor>();

  degenerate_gemv_zero_alpha<float, RowMajor>();
  degenerate_gemv_zero_alpha<double, RowMajor>();
  degenerate_gemv_zero_alpha<std::complex<float>, RowMajor>();
  degenerate_gemv_zero_alpha<std::complex<double>, RowMajor>();
  degenerate_gemv_zero_alpha<float, ColMajor>();
  degenerate_gemv_zero_alpha<double, ColMajor>();
  STATIC_CHECK(!std::is_trivially_copyable<AnnoyingScalar>::value);
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    degenerate_gemv<float>();
    degenerate_gemv<double>();
    degenerate_gemv<half>();
    degenerate_gemv<bfloat16>();
    degenerate_gemv<std::complex<float>>();
    degenerate_gemv<std::complex<double>>();
    degenerate_gemv<double, std::complex<double>>();
    degenerate_gemv<std::complex<double>, double>();
  }
  Matrix<double, Dynamic, Dynamic, RowMajor> lhs = MatrixXd::Constant(2, 1, 1e-200);
  VectorXd rhs = VectorXd::Constant(1, 1e308);
  VectorXd result = VectorXd::Zero(2);
  result.noalias() += 2.0 * lhs * rhs;
  for (Index i = 0; i < result.size(); ++i) {
    VERIFY((numext::isfinite)(result[i]));
    VERIFY(numext::abs(result[i] / 2e108 - 1.0) <= 8 * NumTraits<double>::epsilon());
  }
  Matrix<AnnoyingScalar, Dynamic, Dynamic, RowMajor> custom =
      Matrix<AnnoyingScalar, Dynamic, Dynamic, RowMajor>::Constant(5, 1, AnnoyingScalar(2));
  Matrix<AnnoyingScalar, Dynamic, 1> customRhs = Matrix<AnnoyingScalar, Dynamic, 1>::Constant(1, AnnoyingScalar(3));
  Matrix<AnnoyingScalar, Dynamic, 1> customResult(5);
  customResult.noalias() = custom * customRhs;
  for (Index i = 0; i < customResult.size(); ++i) VERIFY_IS_EQUAL(customResult[i], AnnoyingScalar(6));

  MatrixXd column = MatrixXd::Constant(20000, 1, 2.0);
  VectorXd factor = VectorXd::Constant(1, 3.0);
  VectorXd backing = VectorXd::Zero(2 * column.rows());
  Map<VectorXd, Unaligned, InnerStride<2>> strided(backing.data(), column.rows());
  internal::set_is_malloc_allowed(false);
  strided.noalias() += 0.75 * column * factor;
  internal::set_is_malloc_allowed(true);
  for (Index i = 0; i < column.rows(); ++i) {
    VERIFY_IS_EQUAL(strided[i], 4.5);
    VERIFY_IS_EQUAL(backing[2 * i + 1], 0.0);
  }
}
