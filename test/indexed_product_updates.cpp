// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define TEST_ENABLE_TEMPORARY_TRACKING
#define TEST_IGNORE_STACK_ALLOCATED_TEMPORARY
#include "main.h"
#include <Eigen/SparseCore>

template <typename Matrix, typename Expression>
void check_indexed_update(const Matrix& input, const Matrix& expected, const Expression& expression, int allocations) {
  Matrix dst = Matrix::Constant(expected.rows(), expected.cols(), typename Matrix::Scalar(3));
  const Matrix initial = dst;
  VERIFY_EVALUATION_COUNT((dst.noalias() += expression), allocations);
  VERIFY_IS_APPROX(dst, (initial + expected).eval());
  VERIFY_EVALUATION_COUNT((dst.noalias() -= expression), allocations);
  VERIFY_IS_APPROX(dst, initial);
  Matrix padded = Matrix::Zero(expected.rows() + 2, expected.cols() + 2);
  auto block = padded.block(1, 1, expected.rows(), expected.cols());
  VERIFY_EVALUATION_COUNT((block.noalias() += expression), allocations);
  VERIFY_IS_APPROX(block, expected);
  VERIFY_EVALUATION_COUNT((block.noalias() -= expression), allocations);
  VERIFY(padded.isZero(0));
  dst.noalias() = expression;
  VERIFY_IS_APPROX(dst, expected);
  EIGEN_UNUSED_VARIABLE(input);
}

template <typename Scalar, int Order>
void permutation_updates() {
  using Matrix = Eigen::Matrix<Scalar, Dynamic, Dynamic, Order>;
  for (Index n : {Index(0), Index(1), Index(7), Index(24)}) {
    PermutationMatrix<Dynamic> p(n);
    p.setIdentity();
    for (Index i = 0; i < n; ++i) p.applyTranspositionOnTheRight(i, internal::random<Index>(i, n - 1));
    const Matrix a = Matrix::Random(n, n + 3), b = Matrix::Random(n + 3, n);
    Matrix left(n, n + 3), inverseLeft(n, n + 3), right(n + 3, n), inverseRight(n + 3, n);
    for (Index i = 0; i < n; ++i)
      for (Index j = 0; j < n + 3; ++j) {
        left(p.indices()(i), j) = a(i, j);
        inverseLeft(i, j) = a(p.indices()(i), j);
        right(j, i) = b(j, p.indices()(i));
        inverseRight(j, p.indices()(i)) = b(j, i);
      }
    const auto check = [&](const Matrix& input, const Matrix& expected, const auto& makeProduct) {
      check_indexed_update(input, expected, makeProduct(input), 0);
      Matrix alias = input;
      alias += makeProduct(alias);
      VERIFY_IS_APPROX(alias, (input + expected).eval());
      alias = input;
      alias -= makeProduct(alias);
      VERIFY_IS_APPROX(alias, (input - expected).eval());
    };
    check(a, left, [&](const auto& x) { return p * x; });
    check(a, inverseLeft, [&](const auto& x) { return p.inverse() * x; });
    check(b, right, [&](const auto& x) { return x * p; });
    check(b, inverseRight, [&](const auto& x) { return x * p.inverse(); });
    check_indexed_update(a, left, p * a.block(0, 0, n, n + 3), 0);
    check_indexed_update(a, Matrix(inverseRight.transpose()), p * b.transpose(), 0);
    check_indexed_update(b, Matrix(inverseLeft.transpose()), a.transpose() * p, 0);
    check_indexed_update(a, Matrix(left + left), p * (a + a), 0);
    const Map<const Matrix, Unaligned, Stride<Dynamic, Dynamic>> mapped(
        a.data(), a.rows(), a.cols(), Stride<Dynamic, Dynamic>(a.outerStride(), a.innerStride()));
    check_indexed_update(a, left, p * mapped, 0);
    SparseMatrix<Scalar, Order> sparse = a.sparseView();
    Matrix dst = Matrix::Zero(n, n + 3);
    dst += p * sparse;
    VERIFY_IS_APPROX(dst, left);
  }
}

template <typename Scalar, int Order, typename Tr>
void transposition_updates(const Tr& tr) {
  using Matrix = Eigen::Matrix<Scalar, Dynamic, Dynamic, Order>;
  const Index n = tr.size();
  constexpr bool FitsStack = Tr::IndicesType::MaxSizeAtCompileTime != Dynamic &&
                             (EIGEN_STACK_ALLOCATION_LIMIT == 0 || std::size_t(Tr::IndicesType::MaxSizeAtCompileTime) <=
                                                                       EIGEN_STACK_ALLOCATION_LIMIT / sizeof(Index));
  const int allocations = n != 0 && !FitsStack ? 1 : 0;
  const Matrix a = Matrix::Random(n, n + 3), b = Matrix::Random(n + 3, n);
  const auto reference = [&](Matrix result, bool left, bool inverse) {
    for (Index step = 0; step < n; ++step) {
      const Index k = inverse ? n - 1 - step : step;
      for (Index j = 0; j < n + 3; ++j) {
        if (left)
          numext::swap(result(k, j), result(tr.coeff(k), j));
        else
          numext::swap(result(j, k), result(j, tr.coeff(k)));
      }
    }
    return result;
  };
  const auto check = [&](const Matrix& input, const Matrix& expected, const auto& makeProduct) {
    check_indexed_update(input, expected, makeProduct(input), allocations);
    Matrix alias = input;
    alias += makeProduct(alias);
    VERIFY_IS_APPROX(alias, (input + expected).eval());
    alias = input;
    alias -= makeProduct(alias);
    VERIFY_IS_APPROX(alias, (input - expected).eval());
    alias = input;
    VERIFY_EVALUATION_COUNT((alias = makeProduct(alias)), 0);
    VERIFY_IS_APPROX(alias, expected);
    Matrix overlap = Matrix::Random(input.rows() + 1, input.cols() + 1);
    const Matrix saved = overlap.block(0, 0, input.rows(), input.cols());
    const Matrix product = makeProduct(saved);
    Matrix expectedOverlap = overlap;
    expectedOverlap.block(1, 1, input.rows(), input.cols()) += product;
    overlap.block(1, 1, input.rows(), input.cols()) += makeProduct(overlap.block(0, 0, input.rows(), input.cols()));
    VERIFY_IS_APPROX(overlap, expectedOverlap);
  };
  const auto inverse = tr.inverse();
  check(a, reference(a, true, false), [&](const auto& x) { return tr * x; });
  check(a, reference(a, true, true), [&](const auto& x) { return inverse * x; });
  check(b, reference(b, false, false), [&](const auto& x) { return x * tr; });
  check(b, reference(b, false, true), [&](const auto& x) { return x * inverse; });
}

template <typename Scalar, int Order>
void transposition_update_cases() {
  for (Index n : {Index(0), Index(1), Index(7), Index(24)}) {
    Transpositions<Dynamic> dynamic(n);
    Transpositions<Dynamic, 32, Index> bounded(n);
    for (Index i = 0; i < n; ++i) {
      dynamic.coeffRef(i) = internal::convert_index<int>(internal::random<Index>(0, n - 1));
      bounded.coeffRef(i) = dynamic.coeff(i);
    }
    transposition_updates<Scalar, Order>(dynamic);
    transposition_updates<Scalar, Order>(bounded);
    const Map<Transpositions<Dynamic, 32, Index>> mapped(bounded.indices().data(), n);
    transposition_updates<Scalar, Order>(mapped);
  }
  Transpositions<7, 7, short> fixed;
  for (Index i = 0; i < fixed.size(); ++i) fixed.coeffRef(i) = internal::random<short>(0, 6);
  transposition_updates<Scalar, Order>(fixed);
  Transpositions<Dynamic, Dynamic, signed char> narrow(129);
  narrow.indices().setZero();
  transposition_updates<Scalar, Order>(narrow);
  Transpositions<Dynamic, 129, signed char> narrowBounded(129);
  narrowBounded.indices().setZero();
  transposition_updates<Scalar, Order>(narrowBounded);
}

template <typename Scalar, int Order>
void transposition_row_wrappers() {
  RowVector3i fixed;
  fixed << 2, 2, 2;
  const TranspositionsWrapper<RowVector3i> fixedWrapper(fixed);
  transposition_updates<Scalar, Order>(fixedWrapper);
  const TranspositionsWrapper<const RowVector3i> constWrapper(fixed);
  transposition_updates<Scalar, Order>(constWrapper);
  for (Index n : {Index(0), Index(1), Index(7)}) {
    RowVectorXi dynamic(n);
    for (Index i = 0; i < n; ++i) dynamic(i) = int(n - 1 - i);
    const TranspositionsWrapper<RowVectorXi> dynamicWrapper(dynamic);
    transposition_updates<Scalar, Order>(dynamicWrapper);
    Map<RowVectorXi> mapped(dynamic.data(), n);
    const TranspositionsWrapper<Map<RowVectorXi>> mappedWrapper(mapped);
    transposition_updates<Scalar, Order>(mappedWrapper);
    Eigen::Matrix<int, 1, Dynamic, RowMajor, 1, 7> bounded = dynamic;
    const TranspositionsWrapper<decltype(bounded)> boundedWrapper(bounded);
    transposition_updates<Scalar, Order>(boundedWrapper);
  }
}

template <int Order>
void transposition_widened_capacity() {
  // The input fits the stack limit; widening its maximum capacity to Index may not.
  constexpr int Capacity = EIGEN_STACK_ALLOCATION_LIMIT == 0 ? 20000 : int(EIGEN_STACK_ALLOCATION_LIMIT / sizeof(int));
  const Index n = (std::min)(Index(3), Index(Capacity));
  // Wrappers avoid the owning transpositions' square-matrix traits, whose capacity squared can overflow int.
  Eigen::Matrix<int, Dynamic, 1, 0, Capacity, 1> indices(n);
  for (Index i = 0; i < n; ++i) indices(i) = int(n - 1 - i);
  const TranspositionsWrapper<decltype(indices)> tr(indices);
  transposition_updates<double, Order>(tr);
  Eigen::Matrix<int, 1, Dynamic, RowMajor, 1, Capacity> row(n);
  for (Index i = 0; i < n; ++i) row(i) = tr.coeff(i);
  const TranspositionsWrapper<decltype(row)> wrapper(row);
  transposition_updates<double, Order>(wrapper);

  Eigen::Matrix<int, Capacity, 1> fixedIndices;
  fixedIndices.setZero();
  const TranspositionsWrapper<decltype(fixedIndices)> fixed(fixedIndices);
  using Matrix = Eigen::Matrix<double, Dynamic, Dynamic, Order>;
  const Matrix input = Matrix::Random(Capacity, 2);
  Matrix expected = input, dst = Matrix::Zero(Capacity, 2);
  for (Index i = 0; i < Capacity; ++i)
    for (Index j = 0; j < 2; ++j) numext::swap(expected(i, j), expected(0, j));
  constexpr bool FitsStack =
      EIGEN_STACK_ALLOCATION_LIMIT == 0 || std::size_t(Capacity) <= EIGEN_STACK_ALLOCATION_LIMIT / sizeof(Index);
  const int allocations = Capacity != 0 && !FitsStack ? 1 : 0;
  VERIFY_EVALUATION_COUNT((dst.noalias() += fixed * input), allocations);
  VERIFY_IS_EQUAL(dst, expected);
  VERIFY_EVALUATION_COUNT((dst.noalias() -= fixed * input), allocations);
  VERIFY(dst.isZero(0));
}

EIGEN_DECLARE_TEST(indexed_product_updates) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1((permutation_updates<double, ColMajor>()));
    CALL_SUBTEST_1((permutation_updates<double, RowMajor>()));
    CALL_SUBTEST_2((permutation_updates<std::complex<double>, ColMajor>()));
    CALL_SUBTEST_2((permutation_updates<std::complex<double>, RowMajor>()));
    CALL_SUBTEST_3((transposition_update_cases<double, ColMajor>()));
    CALL_SUBTEST_3((transposition_update_cases<double, RowMajor>()));
    CALL_SUBTEST_4((transposition_update_cases<std::complex<double>, ColMajor>()));
    CALL_SUBTEST_4((transposition_update_cases<std::complex<double>, RowMajor>()));
    CALL_SUBTEST_9((transposition_row_wrappers<double, ColMajor>()));
    CALL_SUBTEST_9((transposition_row_wrappers<double, RowMajor>()));
    CALL_SUBTEST_9((transposition_row_wrappers<std::complex<double>, ColMajor>()));
    CALL_SUBTEST_9((transposition_row_wrappers<std::complex<double>, RowMajor>()));
    CALL_SUBTEST_10((transposition_widened_capacity<ColMajor>()));
    CALL_SUBTEST_10((transposition_widened_capacity<RowMajor>()));
  }
}
