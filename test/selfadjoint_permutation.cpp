// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define TEST_ENABLE_TEMPORARY_TRACKING
#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include <Eigen/Core>
#include <algorithm>
#include <cstdint>

template <unsigned int UpLo, typename MatrixType, typename Transformation>
void check_selfadjoint_permutation(MatrixType& matrix, const Transformation& transformation) {
  using IndicesType = typename Transformation::IndicesType;
  using Permutation = PermutationMatrix<IndicesType::SizeAtCompileTime, IndicesType::MaxSizeAtCompileTime,
                                        typename IndicesType::Scalar>;
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization): Transpositions inputs need this conversion.
  const Permutation permutation(transformation);
  using PlainObject = typename MatrixType::PlainObject;
  const PlainObject original = matrix;
  const Index size = matrix.rows();
  PlainObject full(size, size);
  for (Index j = 0; j < size; ++j) {
    for (Index i = 0; i < size; ++i) {
      full(Index(permutation.indices()[i]), Index(permutation.indices()[j])) =
          (UpLo == Lower ? i >= j : i <= j) ? original(i, j) : numext::conj(original(j, i));
    }
  }
  PlainObject expected = original;
  expected.template triangularView<UpLo>() = full;
  auto view = matrix.template selfadjointView<UpLo>();
  VERIFY(&view.applyPermutation(transformation) == &view);
  VERIFY_IS_EQUAL(matrix, expected);
  VERIFY_IS_EQUAL(view.toDenseMatrix(), full);
  view.applyPermutation(transformation.inverse());
  VERIFY_IS_EQUAL(matrix, original);
  view.applyPermutation(transformation.transpose()).applyPermutation(transformation);
  VERIFY_IS_EQUAL(matrix, original);
}

template <unsigned int UpLo, typename MatrixType>
void check_selfadjoint_transposition(MatrixType& matrix, Index i, Index j) {
  using PlainObject = typename MatrixType::PlainObject;
  const PlainObject original = matrix;
  PlainObject full = original.template selfadjointView<UpLo>();
  full.row(i).swap(full.row(j));
  full.col(i).swap(full.col(j));
  PlainObject expected = original;
  expected.template triangularView<UpLo>() = full;
  auto view = matrix.template selfadjointView<UpLo>();
  internal::set_is_malloc_allowed(false);
  VERIFY(&view.applyTransposition(i, j) == &view);
  internal::set_is_malloc_allowed(true);
  VERIFY_IS_EQUAL(matrix, expected);
  view.applyTransposition(j, i);
  VERIFY_IS_EQUAL(matrix, original);
}

template <typename Scalar, int Order>
void selfadjoint_permutation_cycles() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  for (Index size = 0; size <= 5; ++size) {
    MatrixType matrix = MatrixType::Random(size, size);
    matrix.diagonal() = matrix.diagonal().real().template cast<Scalar>();
    PermutationMatrix<Dynamic> permutation(size);
    permutation.setIdentity();
    do {
      check_selfadjoint_permutation<Lower>(matrix, permutation);
      check_selfadjoint_permutation<Upper>(matrix, permutation);
    } while (size > 1 && std::next_permutation(permutation.indices().data(), permutation.indices().data() + size));
    for (Index i = 0; i < size; ++i) {
      for (Index j = 0; j < size; ++j) {
        check_selfadjoint_transposition<Lower>(matrix, i, j);
        check_selfadjoint_transposition<Upper>(matrix, i, j);
      }
    }
  }
}

template <typename Scalar, int Order>
void selfadjoint_permutation_views() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  const Index size = internal::random<Index>(2, 30);
  MatrixType storage = MatrixType::Random(2 * size + 3, 2 * size + 3);
  const MatrixType original = storage;
  auto block = storage.block(1, 2, size, size);
  PermutationMatrix<Dynamic, Dynamic, std::int64_t> permutation(size);
  randomPermutationVector(permutation.indices(), size);
  const Map<PermutationMatrix<Dynamic, Dynamic, std::int64_t>> mapped_permutation(permutation.indices().data(), size);
  check_selfadjoint_permutation<Lower>(block, mapped_permutation);
  check_selfadjoint_permutation<Upper>(block, permutation.indices().asPermutation());
  VERIFY_IS_EQUAL(storage, original);

  Map<MatrixType, Unaligned, Stride<Dynamic, Dynamic>> strided(storage.data(), size, size,
                                                               Stride<Dynamic, Dynamic>(2 * storage.outerStride(), 2));
  check_selfadjoint_permutation<Lower>(strided, permutation);
  check_selfadjoint_permutation<Upper>(strided, permutation);
  for (Index i = 0; i < size; ++i) {
    check_selfadjoint_transposition<Lower>(block, i, size - 1 - i);
    check_selfadjoint_transposition<Upper>(strided, i, size - 1 - i);
  }
  VERIFY_IS_EQUAL(storage, original);
}

template <typename Scalar, int Size>
void selfadjoint_permutation_fixed() {
  Matrix<Scalar, Size, Size> matrix = Matrix<Scalar, Size, Size>::Random();
  PermutationMatrix<Size> permutation;
  randomPermutationVector(permutation.indices(), Size);
  internal::set_is_malloc_allowed(false);
  check_selfadjoint_permutation<Lower>(matrix, permutation);
  check_selfadjoint_permutation<Upper>(matrix, permutation);
  Transpositions<Size> transpositions;
  for (Index i = 0; i < Size; ++i) transpositions.indices()[i] = int(Size - 1 - i);
  check_selfadjoint_permutation<Lower>(matrix, transpositions);
  check_selfadjoint_permutation<Upper>(matrix, transpositions);
  internal::set_is_malloc_allowed(true);
}

void selfadjoint_permutation_preconditions() {
  MatrixXd matrix = MatrixXd::Random(3, 3);
  PermutationMatrix<Dynamic> permutation(2);
  permutation.setIdentity();
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Lower>().applyPermutation(permutation));
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Upper>().applyPermutation(permutation.inverse()));
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Lower>().applyTransposition(-1, 0));
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Upper>().applyTransposition(0, 3));
  Transpositions<Dynamic> transpositions(2);
  transpositions.setIdentity();
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Lower>().applyPermutation(transpositions));
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Upper>().applyPermutation(transpositions.inverse()));
  // Indices that do not form a permutation: a repeat would otherwise loop forever.
  PermutationMatrix<Dynamic> invalid(3);
  invalid.indices() << 1, 1, 2;
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Lower>().applyPermutation(invalid));
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Upper>().applyPermutation(invalid));
  invalid.indices() << 2, 0, 0;
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Lower>().applyPermutation(invalid));
  invalid.indices() << 3, 0, 1;
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Upper>().applyPermutation(invalid));
  matrix.resize(2, 3);
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Lower>().applyPermutation(permutation));
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Upper>().applyPermutation(permutation));
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Lower>().applyTransposition(0, 1));
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Upper>().applyPermutation(transpositions));
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Lower>().applyPermutation(transpositions.transpose()));
}

void selfadjoint_permutation_aliased_indices() {
  Matrix3i matrix;
  matrix << 1, 2, 0, 4, 5, 6, 7, 8, 9;
  const PermutationMatrix<3> permutation(matrix.row(0).transpose());
  Matrix3i expected = matrix;
  expected.selfadjointView<Upper>().applyPermutation(permutation);
  matrix.selfadjointView<Upper>().applyPermutation(matrix.row(0).transpose().asPermutation());
  VERIFY_IS_EQUAL(matrix, expected);
  matrix << 1, 2, 0, 4, 5, 6, 7, 8, 9;
  expected = matrix;
  expected.selfadjointView<Upper>().applyPermutation(permutation.inverse());
  matrix.selfadjointView<Upper>().applyPermutation(matrix.row(0).transpose().asPermutation().inverse());
  VERIFY_IS_EQUAL(matrix, expected);

  for (bool inverse : {false, true}) {
    matrix << 1, 2, 0, 4, 5, 6, 7, 8, 9;
    auto indices = matrix.row(0).transpose();
    TranspositionsWrapper<decltype(indices)> wrapped(indices);
    const Transpositions<3> transpositions(wrapped);
    expected = matrix;
    if (inverse) {
      expected.selfadjointView<Upper>().applyPermutation(transpositions.inverse());
      matrix.selfadjointView<Upper>().applyPermutation(wrapped.inverse());
    } else {
      expected.selfadjointView<Upper>().applyPermutation(transpositions);
      matrix.selfadjointView<Upper>().applyPermutation(wrapped);
    }
    VERIFY_IS_EQUAL(matrix, expected);
  }
}

template <typename Scalar, int Order>
void selfadjoint_transpositions() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  for (Index size = 0; size <= 4; ++size) {
    MatrixType matrix = MatrixType::Random(size, size);
    matrix.diagonal() = matrix.diagonal().real().template cast<Scalar>();
    Transpositions<Dynamic, Dynamic, std::int64_t> transpositions(size);
    transpositions.indices().setZero();
    // All n^n sequences, including swaps with earlier indices and repeated targets.
    bool finished = false;
    do {
      check_selfadjoint_permutation<Lower>(matrix, transpositions);
      check_selfadjoint_permutation<Upper>(matrix, transpositions);
      Index i = 0;
      while (i < size && ++transpositions.indices()[i] == size) transpositions.indices()[i++] = 0;
      finished = i == size;
    } while (!finished);
    transpositions.setIdentity();
    const Map<Transpositions<Dynamic, Dynamic, std::int64_t>> mapped(transpositions.indices().data(), size);
    check_selfadjoint_permutation<Lower>(matrix, mapped);
    check_selfadjoint_permutation<Upper>(matrix, mapped);
  }
}

void selfadjoint_permutation_temporary_count() {
  MatrixXd matrix = MatrixXd::Random(8, 8);
  PermutationMatrix<Dynamic> permutation(8);
  randomPermutationVector(permutation.indices(), 8);
  VERIFY_EVALUATION_COUNT(matrix.selfadjointView<Lower>().applyPermutation(permutation), 1);
  VERIFY_EVALUATION_COUNT(matrix.selfadjointView<Upper>().applyPermutation(permutation.inverse()), 1);
  VERIFY_EVALUATION_COUNT(matrix.selfadjointView<Lower>().applyPermutation(permutation.transpose()), 1);
  Transpositions<Dynamic> transpositions(8);
  transpositions.setIdentity();
  VERIFY_EVALUATION_COUNT(matrix.selfadjointView<Lower>().applyPermutation(transpositions), 1);
  VERIFY_EVALUATION_COUNT(matrix.selfadjointView<Upper>().applyPermutation(transpositions.transpose()), 1);
  VERIFY_EVALUATION_COUNT(matrix.selfadjointView<Lower>().applyTransposition(1, 7), 0);
}

EIGEN_DECLARE_TEST(selfadjoint_permutation) {
  CALL_SUBTEST_1((selfadjoint_permutation_cycles<float, ColMajor>()));
  CALL_SUBTEST_1((selfadjoint_permutation_cycles<float, RowMajor>()));
  CALL_SUBTEST_2((selfadjoint_permutation_cycles<double, ColMajor>()));
  CALL_SUBTEST_2((selfadjoint_permutation_cycles<double, RowMajor>()));
  CALL_SUBTEST_3((selfadjoint_permutation_cycles<std::complex<float>, ColMajor>()));
  CALL_SUBTEST_3((selfadjoint_permutation_cycles<std::complex<float>, RowMajor>()));
  CALL_SUBTEST_4((selfadjoint_permutation_cycles<std::complex<double>, ColMajor>()));
  CALL_SUBTEST_4((selfadjoint_permutation_cycles<std::complex<double>, RowMajor>()));
  CALL_SUBTEST_5((selfadjoint_permutation_cycles<int, ColMajor>()));
  CALL_SUBTEST_5((selfadjoint_permutation_cycles<short, RowMajor>()));
  CALL_SUBTEST_5((selfadjoint_permutation_cycles<half, ColMajor>()));
  CALL_SUBTEST_5((selfadjoint_permutation_cycles<bfloat16, RowMajor>()));
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1((selfadjoint_permutation_views<float, ColMajor>()));
    CALL_SUBTEST_2((selfadjoint_permutation_views<double, RowMajor>()));
    CALL_SUBTEST_3((selfadjoint_permutation_views<std::complex<float>, RowMajor>()));
    CALL_SUBTEST_4((selfadjoint_permutation_views<std::complex<double>, ColMajor>()));
    CALL_SUBTEST_1((selfadjoint_permutation_fixed<float, 0>()));
    CALL_SUBTEST_1((selfadjoint_permutation_fixed<float, 1>()));
    CALL_SUBTEST_2((selfadjoint_permutation_fixed<double, 4>()));
    CALL_SUBTEST_3((selfadjoint_permutation_fixed<std::complex<float>, 5>()));
  }
  CALL_SUBTEST_6((selfadjoint_transpositions<double, ColMajor>()));
  CALL_SUBTEST_6((selfadjoint_transpositions<double, RowMajor>()));
  CALL_SUBTEST_6((selfadjoint_transpositions<std::complex<double>, ColMajor>()));
  CALL_SUBTEST_6((selfadjoint_transpositions<std::complex<double>, RowMajor>()));
  CALL_SUBTEST_6((selfadjoint_transpositions<int, RowMajor>()));
  CALL_SUBTEST_6(selfadjoint_permutation_temporary_count());
  CALL_SUBTEST_1(selfadjoint_permutation_preconditions());
  CALL_SUBTEST_5(selfadjoint_permutation_aliased_indices());
}
