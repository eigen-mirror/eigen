// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

// det(A) can be representable where the product of its pivots, taken in order, is not. The matrices are S D and
// D S D with S well conditioned and D = diag(2^(+-s)), half of each sign: the scaling is exact, so det(A) = det(S).
// The rank-revealing QR factorizations stop at a column below epsilon times the largest, so they get a D within
// that ratio and enough columns for the big pivots to overflow together.

namespace {

enum Symmetry { General, HermitianPositive, HermitianIndefinite, SymmetricComplex };

template <typename Scalar>
struct ScaledProblem {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;

  // An exponent of D whose square stays in range, so that a column norm and an entry of D S D do.
  static int steepExponent() { return 2 * (std::numeric_limits<RealScalar>::max_exponent / 4 - 4); }
  // An exponent of D that keeps every column within epsilon^(1/2) of the largest.
  static int gentleExponent() { return std::numeric_limits<RealScalar>::digits / 4; }
  // The number of columns at which half of them, scaled by 2^exponent, overflow together.
  static Index overflowingSize(int exponent) {
    return 2 * (std::numeric_limits<RealScalar>::max_exponent / exponent + 2);
  }

  ScaledProblem(Index size, Symmetry symmetry, int exponent) : scaleExponent(exponent) {
    // S = L diag(u) M with L and M unit triangular, off-diagonal entries of magnitude at most 1 / (2 n) and
    // 1 <= |u_i| <= 2, so that det(S) = prod(u), ||L||, ||M|| <= 3/2 and ||S^-1|| <= 4 in the 2-norm.
    const RealScalar offDiagonal = RealScalar(1) / RealScalar(2 * size);
    MatrixType lower = MatrixType::Identity(size, size);
    lower.template triangularView<StrictlyLower>() = offDiagonal * MatrixType::Random(size, size);
    MatrixType upper;
    if (symmetry == General) {
      upper = MatrixType::Identity(size, size);
      upper.template triangularView<StrictlyUpper>() = offDiagonal * MatrixType::Random(size, size);
    } else if (symmetry == SymmetricComplex) {
      upper = lower.transpose();
    } else {
      upper = lower.adjoint();
    }
    determinant = Scalar(1);
    Matrix<Scalar, Dynamic, 1> pivots(size);
    for (Index i = 0; i < size; ++i) {
      Scalar pivot = Scalar(internal::random<RealScalar>(RealScalar(1), RealScalar(2)));
      // A random sign or phase, except where L L^T needs a real square root.
      if (symmetry == General || (symmetry == SymmetricComplex && NumTraits<Scalar>::IsComplex))
        pivot *= numext::sign(internal::random<Scalar>());
      if (symmetry == HermitianIndefinite && i % 3 == 0) pivot = -pivot;
      pivots[i] = pivot;
      determinant *= pivot;
    }
    S = lower * pivots.asDiagonal() * upper;
    // To first order, det(S + E) = det(S) (1 + tr(S^-1 E)) and |tr(S^-1 E)| <= n ||S^-1|| ||E||. Forming S and
    // factoring it each leave ||E|| <= n epsilon ||L|| ||u|| ||M|| <= 9/2 n epsilon (Higham, Accuracy and Stability of
    // Numerical Algorithms, 2nd ed., Theorem 9.3), column scaling aside: 36 n^2 epsilon in all.
    tolerance = RealScalar(64 * size * size) * NumTraits<RealScalar>::epsilon();
    VERIFY(tolerance < RealScalar(1) / RealScalar(16));

    const RealScalar big = std::ldexp(RealScalar(1), scaleExponent);
    const RealScalar bigRoot = std::ldexp(RealScalar(1), scaleExponent / 2);
    columnScaled = S;
    symmetricScaled = S;
    allBig = big * S;
    for (Index j = 0; j < size; ++j) {
      const bool isBig = j < size / 2;
      columnScaled.col(j) *= isBig ? big : RealScalar(1) / big;
      symmetricScaled.col(j) *= isBig ? bigRoot : RealScalar(1) / bigRoot;
      symmetricScaled.row(j) *= isBig ? bigRoot : RealScalar(1) / bigRoot;
    }
  }

  // Whether the product of half of the pivots, each scaled by 2^exponentPerPivot, overflows.
  bool bigPivotsOverflow(int exponentPerPivot) const {
    return (S.rows() / 2) * exponentPerPivot > std::numeric_limits<RealScalar>::max_exponent;
  }

  int scaleExponent;
  MatrixType S, columnScaled, symmetricScaled, allBig;
  Scalar determinant;
  RealScalar tolerance;
};

// absDeterminant() and logAbsDeterminant() return Scalar in SparseLU.
template <typename Solver, typename Scalar>
void check_determinant(const Solver& solver, const Scalar& expected,
                       const typename NumTraits<Scalar>::Real& tolerance) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  const RealScalar expectedAbs = numext::abs(expected);
  VERIFY(numext::abs(Scalar(solver.determinant()) - expected) <= tolerance * expectedAbs);
  VERIFY(numext::abs(numext::real(Scalar(solver.absDeterminant())) - expectedAbs) <= tolerance * expectedAbs);
  // d log|det| = d|det| / |det|, and the log itself rounds.
  const RealScalar expectedLog = numext::log(expectedAbs);
  VERIFY(numext::abs(numext::real(Scalar(solver.logAbsDeterminant())) - expectedLog) <=
         tolerance + RealScalar(4) * NumTraits<RealScalar>::epsilon() * numext::abs(expectedLog));
}

// det(2^s S) overflows, and only it: the log is that of det(S) plus n s log(2).
template <typename Solver, typename Scalar>
void check_overflowing_determinant(const Solver& solver, const ScaledProblem<Scalar>& problem) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  const Scalar det = solver.determinant();
  VERIFY(!(numext::isfinite)(numext::abs(det)));
  VERIFY(!(numext::isnan)(numext::real(det)) && !(numext::isnan)(numext::imag(det)));
  VERIFY_IS_EQUAL(numext::real(Scalar(solver.absDeterminant())), NumTraits<RealScalar>::infinity());
  const RealScalar expectedLog = numext::log(numext::abs(problem.determinant)) +
                                 RealScalar(problem.S.rows() * problem.scaleExponent) * numext::log(RealScalar(2));
  VERIFY(numext::abs(numext::real(Scalar(solver.logAbsDeterminant())) - expectedLog) <=
         problem.tolerance + RealScalar(4) * NumTraits<RealScalar>::epsilon() * expectedLog);
}

template <typename Scalar>
void determinant_scaling_general(Index size) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  const ScaledProblem<Scalar> problem(size, General, ScaledProblem<Scalar>::steepExponent());
  const MatrixType& A = problem.columnScaled;
  const Scalar det = problem.determinant;
  const RealScalar tolerance = problem.tolerance;
  VERIFY(problem.bigPivotsOverflow(problem.scaleExponent));

  check_determinant(PartialPivLU<MatrixType>(A), det, tolerance);
  check_determinant(HouseholderQR<MatrixType>(A), det, tolerance);
  VERIFY(numext::abs(A.determinant() - det) <= tolerance * numext::abs(det));
  // The small columns are below any rank threshold relative to the big ones.
  FullPivLU<MatrixType> fullPivLU(size, size);
  fullPivLU.setThreshold(RealScalar(0));
  fullPivLU.compute(A);
  check_determinant(fullPivLU, det, tolerance);

  check_overflowing_determinant(PartialPivLU<MatrixType>(problem.allBig), problem);
  check_overflowing_determinant(FullPivLU<MatrixType>(problem.allBig), problem);
  check_overflowing_determinant(HouseholderQR<MatrixType>(problem.allBig), problem);
  check_overflowing_determinant(ColPivHouseholderQR<MatrixType>(problem.allBig), problem);

  // U of P A = L U holds the pivots; det(P) = +-1.
  const PartialPivLU<MatrixType> lu(A);
  const Scalar triangular = lu.matrixLU().template triangularView<Upper>().determinant();
  VERIFY(numext::abs(triangular * Scalar(RealScalar(lu.permutationP().determinant())) - det) <=
         tolerance * numext::abs(det));

  SparseLU<SparseMatrix<Scalar>> sparseLU(A.sparseView());
  VERIFY_IS_EQUAL(sparseLU.info(), Success);
  check_determinant(sparseLU, det, tolerance);
  SparseLU<SparseMatrix<Scalar>> sparseBig(problem.allBig.sparseView());
  VERIFY_IS_EQUAL(sparseBig.info(), Success);
  check_overflowing_determinant(sparseBig, problem);
}

template <typename Scalar>
void determinant_scaling_rank_revealing() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  const int exponent = ScaledProblem<Scalar>::gentleExponent();
  const ScaledProblem<Scalar> problem(ScaledProblem<Scalar>::overflowingSize(exponent), General, exponent);
  VERIFY(problem.bigPivotsOverflow(exponent));

  const ColPivHouseholderQR<MatrixType> colPiv(problem.columnScaled);
  VERIFY(colPiv.isInjective());
  check_determinant(colPiv, problem.determinant, problem.tolerance);
  const FullPivHouseholderQR<MatrixType> fullPiv(problem.columnScaled);
  VERIFY(fullPiv.isInjective());
  check_determinant(fullPiv, problem.determinant, problem.tolerance);
  check_determinant(CompleteOrthogonalDecomposition<MatrixType>(problem.columnScaled), problem.determinant,
                    problem.tolerance);
  const RandColPivHouseholderQR<MatrixType> randColPiv(problem.columnScaled);
  VERIFY(randColPiv.isInjective());
  check_determinant(randColPiv, problem.determinant, problem.tolerance);
  const FullPivLU<MatrixType> fullPivLU(problem.columnScaled);
  VERIFY(fullPivLU.isInjective());
  check_determinant(fullPivLU, problem.determinant, problem.tolerance);
}

template <typename Scalar>
void determinant_scaling_hermitian(Index size) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using SparseType = SparseMatrix<Scalar>;
  {
    const ScaledProblem<Scalar> problem(size, HermitianPositive, ScaledProblem<Scalar>::steepExponent());
    const MatrixType& A = problem.symmetricScaled;
    const Scalar det = problem.determinant;
    const RealScalar tolerance = problem.tolerance;
    // The pivots of L are 2^(s / 2) l_ii and those of D are 2^s d_i.
    VERIFY(problem.bigPivotsOverflow(problem.scaleExponent / 2));

    const LLT<MatrixType> llt(A);
    VERIFY_IS_EQUAL(llt.info(), Success);
    check_determinant(llt, det, tolerance);
    const LDLT<MatrixType> ldlt(A);
    VERIFY_IS_EQUAL(ldlt.info(), Success);
    check_determinant(ldlt, det, tolerance);
    check_determinant(BunchKaufman<MatrixType>(A), det, tolerance);

    const SparseType sparse = A.sparseView();
    const SimplicialLLT<SparseType> simplicialLLT(sparse);
    VERIFY_IS_EQUAL(simplicialLLT.info(), Success);
    check_determinant(simplicialLLT, det, tolerance);
    const SimplicialLDLT<SparseType> simplicialLDLT(sparse);
    VERIFY_IS_EQUAL(simplicialLDLT.info(), Success);
    check_determinant(simplicialLDLT, det, tolerance);
    const SimplicialCholesky<SparseType> simplicialCholesky(sparse);
    VERIFY_IS_EQUAL(simplicialCholesky.info(), Success);
    VERIFY(numext::abs(simplicialCholesky.determinant() - det) <= tolerance * numext::abs(det));

    const LLT<MatrixType> big(problem.allBig);
    VERIFY_IS_EQUAL(big.info(), Success);
    check_overflowing_determinant(big, problem);
    check_overflowing_determinant(LDLT<MatrixType>(problem.allBig), problem);
    check_overflowing_determinant(SimplicialLLT<SparseType>(problem.allBig.sparseView()), problem);
    check_overflowing_determinant(SimplicialLDLT<SparseType>(problem.allBig.sparseView()), problem);
  }
  {
    const ScaledProblem<Scalar> problem(size, HermitianIndefinite, ScaledProblem<Scalar>::steepExponent());
    check_determinant(BunchKaufman<MatrixType, Lower>(problem.symmetricScaled), problem.determinant, problem.tolerance);
    check_determinant(BunchKaufman<MatrixType, Upper>(problem.symmetricScaled), problem.determinant, problem.tolerance);
    check_overflowing_determinant(BunchKaufman<MatrixType>(problem.allBig), problem);
  }
  {
    // A = L L^T and A = L D L^T of a symmetric matrix, which is not Hermitian for a complex scalar.
    const ScaledProblem<Scalar> problem(size, SymmetricComplex, ScaledProblem<Scalar>::steepExponent());
    const SparseType sparse = problem.symmetricScaled.sparseView();
    const SimplicialNonHermitianLLT<SparseType> llt(sparse);
    VERIFY_IS_EQUAL(llt.info(), Success);
    check_determinant(llt, problem.determinant, problem.tolerance);
    const SimplicialNonHermitianLDLT<SparseType> ldlt(sparse);
    VERIFY_IS_EQUAL(ldlt.info(), Success);
    check_determinant(ldlt, problem.determinant, problem.tolerance);
  }
}

// A zero pivot that follows pivots whose product overflows gives a zero determinant, not inf * 0.
template <typename Scalar>
void determinant_scaling_singular() {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using VectorType = Matrix<Scalar, Dynamic, 1>;
  const RealScalar big = std::ldexp(RealScalar(1), ScaledProblem<Scalar>::steepExponent());
  VectorType diagonal = VectorType::Constant(6, Scalar(big));
  diagonal[5] = Scalar(0);
  const MatrixType A = diagonal.asDiagonal();

  VERIFY_IS_EQUAL(PartialPivLU<MatrixType>(A).determinant(), Scalar(0));
  VERIFY_IS_EQUAL(PartialPivLU<MatrixType>(A).absDeterminant(), RealScalar(0));
  VERIFY_IS_EQUAL(PartialPivLU<MatrixType>(A).logAbsDeterminant(), -NumTraits<RealScalar>::infinity());
  VERIFY_IS_EQUAL(FullPivLU<MatrixType>(A).determinant(), Scalar(0));
  VERIFY_IS_EQUAL(HouseholderQR<MatrixType>(A).determinant(), Scalar(0));
  VERIFY_IS_EQUAL(HouseholderQR<MatrixType>(A).absDeterminant(), RealScalar(0));
  VERIFY_IS_EQUAL(A.determinant(), Scalar(0));
  VERIFY_IS_EQUAL(A.template triangularView<Upper>().determinant(), Scalar(0));
  VERIFY_IS_EQUAL(A.template triangularView<Lower>().determinant(), Scalar(0));
  VERIFY_IS_EQUAL(BunchKaufman<MatrixType>(A).determinant(), Scalar(0));
}

template <typename Scalar>
void determinant_scaling_views() {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  const RealScalar big = std::ldexp(RealScalar(1), ScaledProblem<Scalar>::steepExponent());
  const RealScalar small = RealScalar(1) / big;

  Matrix<Scalar, 6, 6> triangular = Matrix<Scalar, 6, 6>::Random();
  triangular.diagonal() << Scalar(big), Scalar(3 * big), Scalar(-big), Scalar(small), Scalar(small), Scalar(5 * small);
  VERIFY_IS_EQUAL(triangular.template triangularView<Upper>().determinant(), Scalar(-15));
  VERIFY_IS_EQUAL(triangular.template triangularView<Lower>().determinant(), Scalar(-15));
  VERIFY_IS_EQUAL(triangular.template triangularView<UnitUpper>().determinant(), Scalar(1));
  VERIFY_IS_EQUAL(MatrixType(triangular).template triangularView<Upper>().determinant(), Scalar(-15));

  Matrix<int, 6, 1> indices;
  indices << 1, 0, 2, 3, 4, 5;
  const PermutationMatrix<6> transposition(indices);
  const DiagonalMatrix<Scalar, 6> scales(triangular.diagonal());
  const ScaledPermutationMatrix<Scalar, 6> scaled = transposition * scales;
  VERIFY_IS_EQUAL(scaled.determinant(), Scalar(15));
}

// A scalar without an exact frexp keeps the plain product and the sum of logs.
template <typename = void>
void determinant_scaling_plain_scalar() {
  using Scalar = Eigen::half;
  STATIC_CHECK(!internal::has_scaled_product<Scalar>::value);
  Matrix<Scalar, 3, 3> A;
  A << Scalar(2.0f), Scalar(1.0f), Scalar(0.0f), Scalar(0.0f), Scalar(-4.0f), Scalar(1.0f), Scalar(0.0f), Scalar(0.0f),
      Scalar(0.5f);
  VERIFY_IS_EQUAL(A.template triangularView<Upper>().determinant(), Scalar(-4.0f));
  const PartialPivLU<Matrix<Scalar, 3, 3>> lu(A);
  VERIFY_IS_EQUAL(lu.determinant(), Scalar(-4.0f));
  VERIFY_IS_EQUAL(lu.absDeterminant(), Scalar(4.0f));
  VERIFY_IS_APPROX(float(lu.logAbsDeterminant()), std::log(4.0f));
}

// BunchKaufman balances the product of the blocks of D for the scalars that scaled_product multiplies directly.
template <typename Scalar>
void determinant_scaling_bunch_kaufman_plain_scalar() {
  STATIC_CHECK(!internal::has_scaled_product<Scalar>::value);
  using MatrixType = Matrix<Scalar, 4, 4>;
  // big^2 overflows and small^2 underflows.
  const int exponent = (std::numeric_limits<Scalar>::max_exponent + 1) / 2 + 1;
  const Scalar big(std::ldexp(1.0f, exponent)), small(std::ldexp(1.0f, -exponent));
  VERIFY(!(numext::isfinite)(big * big));

  MatrixType A = MatrixType::Zero();
  A.diagonal() << big, -big, small, small;
  const BunchKaufman<MatrixType> pivots1x1(A);
  VERIFY_IS_EQUAL(pivots1x1.info(), Success);
  VERIFY_IS_EQUAL(pivots1x1.determinant(), Scalar(-1.0f));
  VERIFY_IS_EQUAL(pivots1x1.absDeterminant(), Scalar(1.0f));
  VERIFY_IS_EQUAL(pivots1x1.logAbsDeterminant(), Scalar(0.0f));

  // Two 2x2 pivot blocks with determinants -big^2 and -small^2.
  A.setZero();
  A(0, 1) = A(1, 0) = big;
  A(2, 3) = A(3, 2) = small;
  const BunchKaufman<MatrixType> pivots2x2(A);
  VERIFY_IS_EQUAL(pivots2x2.info(), Success);
  VERIFY_IS_EQUAL(pivots2x2.determinant(), Scalar(1.0f));
  VERIFY_IS_EQUAL(pivots2x2.absDeterminant(), Scalar(1.0f));

  A.diagonal() << big, big, big, Scalar(0.0f);
  A(0, 1) = A(1, 0) = A(2, 3) = A(3, 2) = Scalar(0.0f);
  VERIFY_IS_EQUAL(BunchKaufman<MatrixType>(A).determinant(), Scalar(0.0f));
}

}  // namespace

EIGEN_DECLARE_TEST(determinant_scaling) {
  for (int i = 0; i < g_repeat; ++i) {
    // The pivots of half the columns must overflow together, including those of L in L L^*, which scale by 2^(s / 2).
    const Index size = 2 * internal::random<Index>(6, 10);
    CALL_SUBTEST_1(determinant_scaling_general<float>(size));
    CALL_SUBTEST_1(determinant_scaling_hermitian<float>(size));
    CALL_SUBTEST_2(determinant_scaling_general<double>(size));
    CALL_SUBTEST_2(determinant_scaling_hermitian<double>(size));
    CALL_SUBTEST_3(determinant_scaling_general<std::complex<float>>(size));
    CALL_SUBTEST_3(determinant_scaling_hermitian<std::complex<float>>(size));
    CALL_SUBTEST_4(determinant_scaling_general<std::complex<double>>(size));
    CALL_SUBTEST_4(determinant_scaling_hermitian<std::complex<double>>(size));
    CALL_SUBTEST_5(determinant_scaling_general<long double>(size));
    CALL_SUBTEST_5(determinant_scaling_hermitian<long double>(size));
  }
  CALL_SUBTEST_1(determinant_scaling_rank_revealing<float>());
  CALL_SUBTEST_2(determinant_scaling_rank_revealing<double>());
  CALL_SUBTEST_3(determinant_scaling_rank_revealing<std::complex<float>>());
  CALL_SUBTEST_4(determinant_scaling_rank_revealing<std::complex<double>>());
  CALL_SUBTEST_1(determinant_scaling_singular<float>());
  CALL_SUBTEST_2(determinant_scaling_singular<double>());
  CALL_SUBTEST_3(determinant_scaling_singular<std::complex<float>>());
  CALL_SUBTEST_4(determinant_scaling_singular<std::complex<double>>());
  CALL_SUBTEST_5(determinant_scaling_singular<long double>());
  CALL_SUBTEST_1(determinant_scaling_views<float>());
  CALL_SUBTEST_2(determinant_scaling_views<double>());
  CALL_SUBTEST_3(determinant_scaling_views<std::complex<float>>());
  CALL_SUBTEST_4(determinant_scaling_views<std::complex<double>>());
  CALL_SUBTEST_5(determinant_scaling_views<long double>());
  CALL_SUBTEST_1(determinant_scaling_plain_scalar());
  CALL_SUBTEST_1(determinant_scaling_bunch_kaufman_plain_scalar<Eigen::half>());
  CALL_SUBTEST_1(determinant_scaling_bunch_kaufman_plain_scalar<Eigen::bfloat16>());
}
