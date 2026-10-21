// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

std::vector<Index> blas_singleton_strides(Index count) {
  std::vector<Index> strides{-1, 0, 1, count - 1, count, count + 3};
  EIGEN_IF_CONSTEXPR (std::numeric_limits<Index>::digits > std::numeric_limits<int>::digits) {
    Index large_stride = (std::numeric_limits<int>::max)();
    strides.push_back(++large_stride);
  }
  return strides;
}

void blas_stride_helpers() {
  constexpr Index count = 24;
  for (Index stride : blas_singleton_strides(count)) {
    VERIFY_IS_EQUAL(internal::blas_leading_dim<ColMajor>(count, Index(1), stride), count);
    VERIFY_IS_EQUAL(internal::blas_leading_dim<RowMajor>(Index(1), count, stride), count);
    VERIFY_IS_EQUAL(internal::blas_vector_increment(Index(1), stride), 1);
    VERIFY(internal::blas_compatible_stride<ColMajor>(count, Index(1), stride));
    VERIFY(internal::blas_compatible_stride<RowMajor>(Index(1), count, stride));
    VERIFY_IS_EQUAL(internal::blas_compatible_stride<ColMajor>(count, count, stride), stride >= count);
  }
  VERIFY_IS_EQUAL(internal::blas_leading_dim<ColMajor>(Index(0), Index(1), Index(0)), 1);
  VERIFY_IS_EQUAL(internal::blas_leading_dim<RowMajor>(Index(1), Index(0), Index(0)), 1);
  VERIFY_IS_EQUAL(internal::blas_leading_dim<ColMajor>(count, count, count + 3), count + 3);
  VERIFY_IS_EQUAL(internal::blas_vector_increment(count, Index(-2)), -2);
}

template <typename Scalar>
Scalar blas_stride_value(Index i) {
  return Scalar(i + 1);
}
template <>
std::complex<float> blas_stride_value<std::complex<float>>(Index i) {
  return {float(i + 1), float(i % 3 - 1)};
}
template <>
std::complex<double> blas_stride_value<std::complex<double>>(Index i) {
  return {double(i + 1), double(i % 3 - 1)};
}

template <typename Scalar, int Order>
void blas_singleton_stride() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using VectorType = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  constexpr Index count = 24;
  const Index rows = Order == ColMajor ? count : 1;
  const Index cols = Order == ColMajor ? 1 : count;
  Scalar storage[count];
  for (Index i = 0; i < count; ++i) storage[i] = Scalar(i + 1);

  for (Index stride : blas_singleton_strides(count)) {
    // With one outer vector, even a zero or negative outer stride is unused.
    Map<const MatrixType, 0, OuterStride<>> matrix(storage, rows, cols, OuterStride<>(stride));
    const Scalar alpha = internal::random<Scalar>();
    const VectorType rhs = VectorType::Constant(cols, Scalar(2));
    VectorType expected(rows);
    for (Index i = 0; i < rows; ++i) {
      Scalar value = Scalar(0);
      for (Index j = 0; j < cols; ++j) value += matrix(i, j) * rhs(j);
      expected(i) = value;
    }
    VectorType result = matrix * rhs;
    VERIFY_IS_CWISE_EQUAL(result, expected);
    EIGEN_IF_CONSTEXPR (Order == RowMajor) {
      Scalar singleton_storage = Scalar(0);
      Map<VectorType, 0, InnerStride<>> singleton_result(&singleton_storage, 1, InnerStride<>(stride));
      singleton_result.noalias() = matrix * rhs;
      VERIFY_IS_EQUAL(singleton_storage, expected(0));
    }
    result.setOnes();
    result.noalias() += alpha * matrix * rhs;
    const RealScalar bound = RealScalar(8 * count) * NumTraits<RealScalar>::epsilon();
    VERIFY((result - (VectorType::Ones(rows) + alpha * expected)).norm() <= bound * (RealScalar(1) + expected.norm()));

    const VectorType transpose_rhs = VectorType::Constant(rows, Scalar(2));
    VectorType transpose_expected(cols);
    for (Index j = 0; j < cols; ++j) {
      Scalar value = Scalar(0);
      for (Index i = 0; i < rows; ++i) value += matrix(i, j) * transpose_rhs(i);
      transpose_expected(j) = value;
    }
    result = matrix.transpose() * transpose_rhs;
    VERIFY_IS_CWISE_EQUAL(result, transpose_expected);
  }
}

template <typename Scalar, int Order>
void blas_singleton_outer_product() {
  using ColMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using RowMatrix = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;
  using DstMatrix = Matrix<Scalar, Dynamic, Dynamic, Order>;
  constexpr Index count = 24;
  Scalar lhs_storage[count], rhs_storage[count];
  for (Index i = 0; i < count; ++i) {
    lhs_storage[i] = Scalar(i + 1);
    rhs_storage[i] = Scalar(2 * i + 1);
  }
  DstMatrix expected(count, count), result(count, count);
  for (Index j = 0; j < count; ++j)
    for (Index i = 0; i < count; ++i) expected(i, j) = lhs_storage[i] * rhs_storage[j];
  for (Index stride : blas_singleton_strides(count)) {
    Map<const ColMatrix, 0, OuterStride<>> lhs(lhs_storage, count, 1, OuterStride<>(stride));
    Map<const RowMatrix, 0, OuterStride<>> rhs(rhs_storage, 1, count, OuterStride<>(stride));
    result.noalias() = lhs * rhs;
    VERIFY_IS_CWISE_EQUAL(result, expected);
    result.setOnes();
    result.noalias() += Scalar(2) * lhs * rhs;
    VERIFY_IS_CWISE_EQUAL(result, (DstMatrix::Ones(count, count) + Scalar(2) * expected).eval());

    Scalar scalar_storage = Scalar(2);
    Map<const RowMatrix, 0, OuterStride<>> scalar(&scalar_storage, 1, 1, OuterStride<>(stride));
    result.noalias() = lhs * scalar;
    for (Index i = 0; i < count; ++i) VERIFY_IS_EQUAL(result(i, 0), Scalar(2) * lhs_storage[i]);
  }
}

#ifdef EIGEN_USE_BLAS
// Call the adapters directly: public products redirect singleton outputs through dot products or GEMV.
template <typename Scalar, int Order>
void blas_direct_singleton_gemv() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using VectorType = Matrix<Scalar, Dynamic, 1>;
  constexpr Index count = 24;
  const Index rows = Order == ColMajor ? count : 1;
  const Index cols = Order == ColMajor ? 1 : count;
  MatrixType lhs(rows, cols);
  VectorType rhs(cols);
  for (Index i = 0; i < lhs.size(); ++i) lhs.data()[i] = blas_stride_value<Scalar>(i);
  for (Index i = 0; i < cols; ++i) rhs(i) = blas_stride_value<Scalar>(i % 3);
  for (Index stride : blas_singleton_strides(count)) {
    VectorType result = VectorType::Ones(rows), expected = result;
    for (Index i = 0; i < rows; ++i)
      for (Index j = 0; j < cols; ++j) expected(i) += Scalar(2) * lhs(i, j) * rhs(j);
    internal::general_matrix_vector_product_gemv<Index, Scalar, Order, false, Scalar, false>::run(
        rows, cols, lhs.data(), stride, rhs.data(), cols == 1 ? stride : Index(1), result.data(),
        rows == 1 ? stride : Index(1), Scalar(2));
    VERIFY_IS_CWISE_EQUAL(result, expected);
  }
  using RealScalar = typename NumTraits<Scalar>::Real;
  RealScalar nan = NumTraits<RealScalar>::quiet_NaN();
  EIGEN_FAST_MATH_CONSTANT_BARRIER(nan);
  lhs.setConstant(Scalar(nan));
  for (Index stride : {Index(0), lhs.outerStride()}) {
    VectorType result = VectorType::Ones(rows);
    internal::general_matrix_vector_product_gemv<Index, Scalar, Order, false, Scalar, false>::run(
        rows, cols, lhs.data(), stride, rhs.data(), Index(1), result.data(), Index(1), Scalar(0));
    VERIFY_IS_CWISE_EQUAL(result, VectorType::Ones(rows));
  }
}

template <typename Scalar, int LhsOrder, int RhsOrder>
void blas_direct_singleton_gemm() {
  using Lhs = Matrix<Scalar, Dynamic, Dynamic, LhsOrder>;
  using Rhs = Matrix<Scalar, Dynamic, Dynamic, RhsOrder>;
  using Dst = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  const Index shapes[][3] = {{24, 1, 24}, {1, 24, 24}, {24, 24, 1}, {1, 1, 24}};
  for (const auto& shape : shapes) {
    const Index rows = shape[0], cols = shape[1], depth = shape[2];
    Lhs lhs(rows, depth);
    Rhs rhs(depth, cols);
    for (Index i = 0; i < lhs.size(); ++i) lhs.data()[i] = blas_stride_value<Scalar>(i % 3);
    for (Index i = 0; i < rhs.size(); ++i) rhs.data()[i] = blas_stride_value<Scalar>(i % 5);
    Dst expected = Dst::Ones(rows, cols);
    for (Index j = 0; j < cols; ++j)
      for (Index i = 0; i < rows; ++i)
        for (Index k = 0; k < depth; ++k) expected(i, j) += Scalar(2) * lhs(i, k) * rhs(k, j);
    for (Index stride : blas_singleton_strides(Index(24))) {
      Dst result = Dst::Ones(rows, cols);
      const Index lhs_stride = lhs.outerSize() == 1 ? stride : lhs.outerStride();
      const Index rhs_stride = rhs.outerSize() == 1 ? stride : rhs.outerStride();
      internal::gemm_blocking_space<ColMajor, Scalar, Scalar, Dynamic, Dynamic, Dynamic> blocking(rows, cols, depth, 1,
                                                                                                  false);
      internal::general_matrix_matrix_product<Index, Scalar, LhsOrder, false, Scalar, RhsOrder, false, ColMajor,
                                              1>::run(rows, cols, depth, lhs.data(), lhs_stride, rhs.data(), rhs_stride,
                                                      result.data(), Index(1),
                                                      cols == 1 ? stride : result.outerStride(), Scalar(2), blocking,
                                                      nullptr);
      VERIFY_IS_CWISE_EQUAL(result, expected);
    }
    using RealScalar = typename NumTraits<Scalar>::Real;
    RealScalar nan = NumTraits<RealScalar>::quiet_NaN();
    EIGEN_FAST_MATH_CONSTANT_BARRIER(nan);
    lhs.setConstant(Scalar(nan));
    for (Index stride : {Index(0), lhs.outerStride()}) {
      Dst result = Dst::Ones(rows, cols);
      internal::gemm_blocking_space<ColMajor, Scalar, Scalar, Dynamic, Dynamic, Dynamic> blocking(rows, cols, depth, 1,
                                                                                                  false);
      internal::general_matrix_matrix_product<Index, Scalar, LhsOrder, false, Scalar, RhsOrder, false, ColMajor,
                                              1>::run(rows, cols, depth, lhs.data(), stride, rhs.data(),
                                                      rhs.outerStride(), result.data(), Index(1), result.outerStride(),
                                                      Scalar(0), blocking, nullptr);
      VERIFY_IS_CWISE_EQUAL(result, Dst::Ones(rows, cols));
    }
  }
}
template <typename Scalar, int Order>
void blas_direct_trimmed_trmv() {
  using Vector = Matrix<Scalar, Dynamic, 1>;
  constexpr Index count = 24;
  constexpr int mode = Order == ColMajor ? Lower : Upper;
  const Index rows = Order == ColMajor ? 1 : count;
  const Index cols = Order == ColMajor ? count : 1;
  // Only (0,0) belongs to the triangular part; the other coefficients are implicit zeros.
  const Scalar coefficient = blas_stride_value<Scalar>(0);
  const Vector rhs = Vector::Ones(cols);
  Vector expected = Vector::Ones(rows);
  expected(0) += Scalar(2) * coefficient;
  for (Index stride : blas_singleton_strides(count)) {
    Vector result = Vector::Ones(rows);
    internal::triangular_matrix_vector_product_trmv<Index, mode, Scalar, false, Scalar, false, Order>::run(
        rows, cols, &coefficient, stride, rhs.data(), Index(1), result.data(), Index(1), Scalar(2));
    VERIFY_IS_CWISE_EQUAL(result, expected);
  }
}

template <typename Scalar, int Order, int Mode, bool Conjugate>
void blas_direct_strided_trmv() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  const Index shapes[][2] = {{3, 3}, {3, 5}, {5, 3}, {19, 21}, {21, 19}};
  for (const auto& shape : shapes) {
    const Index rows = shape[0], cols = shape[1];
    const Index inner = Order == ColMajor ? rows : cols;
    const Index outer = Order == ColMajor ? cols : rows;
    for (Index stride : {Index(-2), Index(0), Index(1), inner + 2}) {
      Vector storage((outer - 1) * numext::abs(stride) + inner);
      for (Index i = 0; i < storage.size(); ++i) storage(i) = blas_stride_value<Scalar>(i % 5);
      const Scalar* data = storage.data() + (stride < 0 ? (outer - 1) * -stride : 0);
      Map<const MatrixType, 0, OuterStride<>> matrix(data, rows, cols, OuterStride<>(stride));
      for (Index rhs_incr : {Index(-2), Index(0), Index(1), Index(2)}) {
        Vector rhs_storage((cols - 1) * numext::abs(rhs_incr) + 1);
        for (Index i = 0; i < rhs_storage.size(); ++i) rhs_storage(i) = blas_stride_value<Scalar>(i % 3);
        const Scalar* rhs = rhs_storage.data() + (rhs_incr < 0 ? rhs_storage.size() - 1 : 0);
        for (Index res_incr : {Index(-2), Index(0), Index(1), Index(2)}) {
          Vector result = Vector::Constant((rows - 1) * numext::abs(res_incr) + 1, Scalar(-17));
          Vector expected = result;
          const Index offset = res_incr < 0 ? result.size() - 1 : 0;
          for (Index i = 0; i < rows; ++i) {
            Scalar sum = Scalar(0);
            for (Index j = 0; j < cols; ++j) {
              if ((Mode & Lower) ? i < j : i > j) continue;
              const Scalar value = i == j && (Mode & (UnitDiag | ZeroDiag)) ? Scalar((Mode & UnitDiag) ? 1 : 0)
                                   : Conjugate                              ? numext::conj(matrix(i, j))
                                                                            : matrix(i, j);
              sum += value * (Conjugate ? numext::conj(rhs[j * rhs_incr]) : rhs[j * rhs_incr]);
            }
            expected(offset + i * res_incr) += Scalar(2) * sum;
          }
          internal::triangular_matrix_vector_product_trmv<Index, Mode, Scalar, Conjugate, Scalar, Conjugate,
                                                          Order>::run(rows, cols, data, stride, rhs, rhs_incr,
                                                                      result.data() + offset, res_incr, Scalar(2));
          // Compare the entire backing store, including gaps and implicit-zero output rows.
          VERIFY_IS_CWISE_EQUAL(result, expected);
        }
      }
    }
  }
}

#endif

template <typename Scalar, int Order>
void blas_structured_singleton_stride() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using ColMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  constexpr Index count = 24;
  const ColMatrix diagonal = Scalar(2) * ColMatrix::Identity(count, count);
  MatrixType storage(Order == ColMajor ? count : 1, Order == ColMajor ? 1 : count);
  for (Index i = 0; i < count; ++i) storage.data()[i] = blas_stride_value<Scalar>(i);
  for (Index stride : blas_singleton_strides(count)) {
    Map<const MatrixType, 0, OuterStride<>> input(storage.data(), storage.rows(), storage.cols(),
                                                  OuterStride<>(stride));
    MatrixType output_storage = storage;
    Map<MatrixType, 0, OuterStride<>> output(output_storage.data(), storage.rows(), storage.cols(),
                                             OuterStride<>(stride));
    EIGEN_IF_CONSTEXPR (Order == ColMajor) {
      diagonal.template triangularView<Lower>().solveInPlace(output);
      VERIFY_IS_CWISE_EQUAL(output, (storage * Scalar(0.5)).eval());
      ColMatrix result = diagonal.template selfadjointView<Lower>() * input;
      VERIFY_IS_CWISE_EQUAL(result, (storage * Scalar(2)).eval());
      result = ColMatrix::Zero(count, count);
      result.template selfadjointView<Lower>().rankUpdate(input);
      for (Index j = 0; j < count; ++j)
        for (Index i = j; i < count; ++i) VERIFY_IS_EQUAL(result(i, j), storage(i, 0) * numext::conj(storage(j, 0)));
    } else {
      diagonal.template triangularView<Lower>().template solveInPlace<OnTheRight>(output);
      VERIFY_IS_CWISE_EQUAL(output, (storage * Scalar(0.5)).eval());
      ColMatrix result = input * diagonal.template selfadjointView<Lower>();
      VERIFY_IS_CWISE_EQUAL(result, (storage * Scalar(2)).eval());
    }
  }
}

template <typename Scalar, int Order>
void blas_overlapping_input_stride() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using VectorType = Matrix<Scalar, Dynamic, 1>;
  constexpr Index count = 24;
  VectorType storage(2 * count * count);
  for (Index i = 0; i < storage.size(); ++i) storage(i) = blas_stride_value<Scalar>(i % 5);
  const MatrixType rhs = MatrixType::Ones(count, count);
  for (Index stride : {-25, -1, 0, 1, 23}) {
    const Scalar* data = storage.data() + (stride < 0 ? (count - 1) * -stride : 0);
    Map<const MatrixType, 0, OuterStride<>> lhs(data, count, count, OuterStride<>(stride));
    VectorType expected(count);
    for (Index i = 0; i < count; ++i) {
      expected(i) = Scalar(0);
      for (Index j = 0; j < count; ++j) expected(i) += lhs(i, j);
    }
    VectorType result = lhs * VectorType::Ones(count);
    VERIFY_IS_CWISE_EQUAL(result, expected);
    MatrixType product = lhs * rhs;
    for (Index j = 0; j < count; ++j) VERIFY_IS_CWISE_EQUAL(product.col(j), expected);

    VectorType hermitian_rows = VectorType::Zero(count), hermitian_cols = VectorType::Zero(count);
    VectorType lower_rows = VectorType::Zero(count), lower_cols = VectorType::Zero(count);
    for (Index j = 0; j < count; ++j)
      for (Index i = 0; i < count; ++i) {
        const Scalar value = i == j ? Scalar(numext::real(lhs(i, i))) : i > j ? lhs(i, j) : numext::conj(lhs(j, i));
        hermitian_rows(i) += value;
        hermitian_cols(j) += value;
        if (i >= j) {
          lower_rows(i) += lhs(i, j);
          lower_cols(j) += lhs(i, j);
        }
      }
    result = lhs.template selfadjointView<Lower>() * VectorType::Ones(count);
    VERIFY_IS_CWISE_EQUAL(result, hermitian_rows);
    product = lhs.template selfadjointView<Lower>() * rhs;
    for (Index j = 0; j < count; ++j) VERIFY_IS_CWISE_EQUAL(product.col(j), hermitian_rows);
    product = rhs * lhs.template selfadjointView<Lower>();
    for (Index i = 0; i < count; ++i) VERIFY_IS_CWISE_EQUAL(product.row(i).transpose(), hermitian_cols);
    result = lhs.template triangularView<Lower>() * VectorType::Ones(count);
    VERIFY_IS_CWISE_EQUAL(result, lower_rows);
    product = lhs.template triangularView<Lower>() * rhs;
    for (Index j = 0; j < count; ++j) VERIFY_IS_CWISE_EQUAL(product.col(j), lower_rows);
    product = rhs * lhs.template triangularView<Lower>();
    for (Index i = 0; i < count; ++i) VERIFY_IS_CWISE_EQUAL(product.row(i).transpose(), lower_cols);
    product.setZero();
    product.template selfadjointView<Lower>().rankUpdate(lhs);
    for (Index j = 0; j < count; ++j)
      for (Index i = j; i < count; ++i) {
        Scalar value = Scalar(0);
        for (Index k = 0; k < count; ++k) value += lhs(i, k) * numext::conj(lhs(j, k));
        VERIFY_IS_EQUAL(product(i, j), value);
      }

    MatrixType solved = MatrixType::Ones(4, 2);
    for (Index j = 0; j < solved.cols(); ++j)
      for (Index i = 0; i < solved.rows(); ++i) {
        solved(i, j) = Scalar(0);
        for (Index k = 0; k <= i; ++k) solved(i, j) += lhs(i, k);
      }
    lhs.topLeftCorner(4, 4).template triangularView<Lower>().solveInPlace(solved);
    VERIFY_IS_APPROX(solved, MatrixType::Ones(4, 2));
  }
  Scalar two = Scalar(2);
  Map<const VectorType, 0, InnerStride<>> broadcast(&two, count, InnerStride<>(0));
  const VectorType result = rhs * broadcast;
  VERIFY_IS_CWISE_EQUAL(result, VectorType::Constant(count, Scalar(2 * count)));
}

template <typename Scalar, int Order>
void blas_stride_tests() {
  blas_singleton_stride<Scalar, Order>();
  blas_singleton_outer_product<Scalar, Order>();
  blas_structured_singleton_stride<Scalar, Order>();
  blas_overlapping_input_stride<Scalar, Order>();
#ifdef EIGEN_USE_BLAS
  blas_direct_singleton_gemv<Scalar, Order>();
  blas_direct_trimmed_trmv<Scalar, Order>();
  blas_direct_strided_trmv<Scalar, Order, Lower, false>();
  blas_direct_strided_trmv<Scalar, Order, Upper | UnitDiag, true>();
  blas_direct_strided_trmv<Scalar, Order, Lower | ZeroDiag, false>();
  blas_direct_singleton_gemm<Scalar, Order, ColMajor>();
  blas_direct_singleton_gemm<Scalar, Order, RowMajor>();
#endif
}

EIGEN_DECLARE_TEST(blas_singleton_stride) {
  blas_stride_helpers();
  blas_stride_tests<float, ColMajor>();
  blas_stride_tests<float, RowMajor>();
  blas_stride_tests<double, ColMajor>();
  blas_stride_tests<double, RowMajor>();
  blas_stride_tests<std::complex<float>, ColMajor>();
  blas_stride_tests<std::complex<float>, RowMajor>();
  blas_stride_tests<std::complex<double>, ColMajor>();
  blas_stride_tests<std::complex<double>, RowMajor>();
}
