// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#define EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW
#include "AnnoyingScalar.h"

template <typename Lhs, typename Rhs>
void check_degenerate_gemv(const Lhs& lhs, const Rhs& rhs) {
  using Scalar = typename ScalarBinaryOpTraits<typename Lhs::Scalar, typename Rhs::Scalar>::ReturnType;
  using Real = typename NumTraits<Scalar>::Real;
  using Wide = std::conditional_t<NumTraits<Scalar>::IsComplex, std::complex<long double>, long double>;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  const Scalar alpha = internal::random<Scalar>();
  const Vector initial = Vector::Random(lhs.rows());
  Vector contiguous(initial);
  Vector storage = Vector::Constant(2 * lhs.rows() + 1, Scalar(7));
  Map<Vector, Unaligned, InnerStride<2>> result(storage.data(), lhs.rows());
  Vector dynamicStorage(storage);
  Map<Vector, Unaligned, InnerStride<Dynamic>> dynamicResult(dynamicStorage.data(), lhs.rows(),
                                                             InnerStride<Dynamic>(2));
  for (int operation = 0; operation < 3; ++operation) {
    result = initial;
    contiguous = initial;
    dynamicResult = initial;
    auto evaluate = [&](auto& target) {
      if (operation == 0) target.noalias() = alpha * lhs * rhs;
      if (operation == 1) target.noalias() += alpha * lhs * rhs;
      if (operation == 2) target.noalias() -= alpha * lhs * rhs;
    };
    evaluate(result);
    evaluate(contiguous);
    evaluate(dynamicResult);
    for (Index i = 0; i < lhs.rows(); ++i) {
      Wide sum(0);
      long double magnitude = 0;
      for (Index k = 0; k < lhs.cols(); ++k) {
        const Wide term = Wide(alpha) * Wide(lhs.coeff(i, k)) * Wide(rhs.coeff(k, 0));
        sum += term;
        magnitude += numext::abs(term);
      }
      const Wide expected = operation == 0 ? sum : Wide(initial[i]) + (operation == 1 ? sum : -sum);
      const long double bound =
          32 * static_cast<long double>(NumTraits<Real>::epsilon()) * (magnitude + numext::abs(Wide(initial[i])));
      VERIFY(numext::abs(Wide(result[i]) - expected) <= bound);
      VERIFY(numext::abs(Wide(contiguous[i]) - expected) <= bound);
      VERIFY(numext::abs(Wide(dynamicResult[i]) - expected) <= bound);
    }
    for (Index i = 1; i < storage.size(); i += 2) VERIFY_IS_EQUAL(storage[i], Scalar(7));
    VERIFY_IS_EQUAL(storage[storage.size() - 1], Scalar(7));
    for (Index i = 1; i < dynamicStorage.size(); i += 2) VERIFY_IS_EQUAL(dynamicStorage[i], Scalar(7));
    VERIFY_IS_EQUAL(dynamicStorage[dynamicStorage.size() - 1], Scalar(7));
  }
}

template <typename LhsScalar, typename RhsScalar = LhsScalar>
void degenerate_gemv() {
  using MatrixType = Matrix<LhsScalar, Dynamic, Dynamic>;
  using RhsVector = Matrix<RhsScalar, Dynamic, 1>;
  for (Index n : {0, 1, 3, 8, 17, 257}) {
    for (Index depth : {0, 1, 2, 4}) {
      MatrixType a = MatrixType::Random(depth, n);
      RhsVector rhsStorage = RhsVector::Random(3 * depth + 1);
      Map<RhsVector, Unaligned, InnerStride<3>> rhs(rhsStorage.data(), depth);
      check_degenerate_gemv(a.transpose(), rhs);
      check_degenerate_gemv(a.adjoint(), rhs.conjugate());
      check_degenerate_gemv(LhsScalar(0.75) * a.adjoint(), RhsScalar(0.5) * rhs);
      MatrixType columnMajor = a.transpose();
      check_degenerate_gemv(columnMajor, rhs);
      check_degenerate_gemv(columnMajor.conjugate(), rhs.conjugate());
      check_degenerate_gemv(LhsScalar(0.75) * columnMajor.conjugate(), RhsScalar(0.5) * rhs);
    }
    Matrix<LhsScalar, Dynamic, 1> lhsStorage = Matrix<LhsScalar, Dynamic, 1>::Random(3 * n + 1);
    using RowMatrix = Matrix<LhsScalar, Dynamic, Dynamic, RowMajor>;
    Map<RowMatrix, Unaligned, OuterStride<3>> lhs(lhsStorage.data(), n, 1);
    RhsVector rhs = RhsVector::Random(1);
    check_degenerate_gemv(lhs, rhs);
    Map<RowMatrix, Unaligned, OuterStride<>> runtimeLhs(lhsStorage.data(), n, 1, OuterStride<>(3));
    check_degenerate_gemv(runtimeLhs, rhs);
    Matrix<LhsScalar, 1, Dynamic> row = Matrix<LhsScalar, 1, Dynamic>::Random(n);
    check_degenerate_gemv(row.transpose(), rhs);
    Matrix<LhsScalar, 2, Dynamic> fixedDepth = Matrix<LhsScalar, 2, Dynamic>::Random(2, n);
    RhsVector fixedDepthRhs = RhsVector::Random(2);
    check_degenerate_gemv(fixedDepth.transpose(), fixedDepthRhs);
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
