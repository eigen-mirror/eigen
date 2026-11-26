// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "fp_control.h"
#include <algorithm>
#include <limits>
#include <vector>
#include <Eigen/Eigenvalues>

template <typename MatrixType>
void schur(int size = MatrixType::ColsAtCompileTime) {
  using ComplexScalar = typename ComplexSchur<MatrixType>::ComplexScalar;
  using ComplexMatrixType = typename ComplexSchur<MatrixType>::ComplexMatrixType;

  // Test basic functionality: T is triangular and A = U T U*
  for (int counter = 0; counter < g_repeat; ++counter) {
    MatrixType A = MatrixType::Random(size, size);
    ComplexSchur<MatrixType> schurOfA(A);
    VERIFY_IS_EQUAL(schurOfA.info(), Success);
    const ComplexMatrixType& U = schurOfA.matrixU();
    const ComplexMatrixType& T = schurOfA.matrixT();
    for (int row = 1; row < size; ++row) {
      for (int col = 0; col < row; ++col) {
        VERIFY(T(row, col) == (typename MatrixType::Scalar)0);
      }
    }
    VERIFY_IS_APPROX(A.template cast<ComplexScalar>(), U * T * U.adjoint());
  }

  // Test asserts when not initialized
  ComplexSchur<MatrixType> csUninitialized;
  VERIFY_RAISES_ASSERT(csUninitialized.matrixT());
  VERIFY_RAISES_ASSERT(csUninitialized.matrixU());
  VERIFY_RAISES_ASSERT(csUninitialized.info());

  // Test whether compute() and constructor returns same result
  MatrixType A = MatrixType::Random(size, size);
  ComplexSchur<MatrixType> cs1;
  cs1.compute(A);
  ComplexSchur<MatrixType> cs2(A);
  VERIFY_IS_EQUAL(cs1.info(), Success);
  VERIFY_IS_EQUAL(cs2.info(), Success);
  VERIFY_IS_EQUAL(cs1.matrixT(), cs2.matrixT());
  VERIFY_IS_EQUAL(cs1.matrixU(), cs2.matrixU());

  // Test maximum number of iterations
  ComplexSchur<MatrixType> cs3;
  cs3.setMaxIterations(ComplexSchur<MatrixType>::m_maxIterationsPerRow * size).compute(A);
  VERIFY_IS_EQUAL(cs3.info(), Success);
  VERIFY_IS_EQUAL(cs3.matrixT(), cs1.matrixT());
  VERIFY_IS_EQUAL(cs3.matrixU(), cs1.matrixU());
  cs3.setMaxIterations(1).compute(A);
  // The schur decomposition does often converge with a single iteration.
  // VERIFY_IS_EQUAL(cs3.info(), size > 1 ? NoConvergence : Success);
  VERIFY_IS_EQUAL(cs3.getMaxIterations(), 1);

  MatrixType Atriangular = A;
  Atriangular.template triangularView<StrictlyLower>().setZero();
  cs3.setMaxIterations(1).compute(Atriangular);  // triangular matrices do not need any iterations
  VERIFY_IS_EQUAL(cs3.info(), Success);
  VERIFY_IS_EQUAL(cs3.matrixT(), Atriangular.template cast<ComplexScalar>());
  VERIFY_IS_EQUAL(cs3.matrixU(), ComplexMatrixType::Identity(size, size));

  // Test computation of only T, not U
  ComplexSchur<MatrixType> csOnlyT(A, false);
  VERIFY_IS_EQUAL(csOnlyT.info(), Success);
  VERIFY_IS_EQUAL(cs1.matrixT(), csOnlyT.matrixT());
  VERIFY_RAISES_ASSERT(csOnlyT.matrixU());

  if (size > 1 && size < 20) {
    // Test matrix with NaN
    A(0, 0) = std::numeric_limits<typename MatrixType::RealScalar>::quiet_NaN();
    ComplexSchur<MatrixType> csNaN(A);
    VERIFY_IS_EQUAL(csNaN.info(), NoConvergence);
  }
}

// The Hessenberg reduction treats a subdiagonal tail whose squared norm underflows as already zero. ComplexSchur used
// to reduce the caller's matrix directly, so at this scale it returned the diagonal of the input as the Schur form.
template <typename MatrixType>
void schur_underflow_scale(Index size) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using ComplexScalar = typename ComplexSchur<MatrixType>::ComplexScalar;
  using ComplexMatrixType = typename ComplexSchur<MatrixType>::ComplexMatrixType;

  // Representable, but every product of two coefficients underflows.
  const RealScalar scale = numext::sqrt((std::numeric_limits<RealScalar>::min)()) / RealScalar(1024);
  const MatrixType A = MatrixType::Random(size, size) * Scalar(scale);

  ComplexSchur<MatrixType> schurOfA(A);
  VERIFY_IS_EQUAL(schurOfA.info(), Success);
  const ComplexMatrixType& U = schurOfA.matrixU();
  const ComplexMatrixType& T = schurOfA.matrixT();

  // isApprox squares its operands, and every square here is subnormal: under flush-to-zero both sides of the
  // reconstruction collapse to zero and any T passes. scale is an exact power of two, so multiplying by its
  // reciprocal is lossless and puts the comparison back in the normal range.
  const ComplexScalar invScale = ComplexScalar(RealScalar(1) / scale);
  VERIFY_IS_APPROX(ComplexMatrixType(A.template cast<ComplexScalar>() * invScale),
                   ComplexMatrixType(U * T * U.adjoint() * invScale));
  VERIFY_IS_APPROX(U * U.adjoint(), ComplexMatrixType::Identity(size, size));
  for (Index row = 1; row < size; ++row)
    for (Index col = 0; col < row; ++col) VERIFY(numext::is_exactly_zero(T(row, col)));
}

// Under flush-to-zero, targets with runtime control see what ARMv7 NEON always does to subnormal SIMD results, and Arm
// flushes subnormal inputs as well. The decomposition runs inside the scope; callers verify outside it.
template <typename MatrixType>
bool compute_schur(ComplexSchur<MatrixType>& schur, const MatrixType& A, bool flushToZero) {
  if (!flushToZero) {
    schur.compute(A);
    return true;
  }
  ScopedFlushToZero scope;
  if (scope.isSupported()) schur.compute(A);
  return scope.isSupported();
}

// A matrix whose coefficients are all subnormal is still a matrix with a Schur decomposition, not the zero matrix.
// Scaling by a power of two moves it into the normal range exactly, so the factorization stays accurate to the
// quantization of the subnormal input.
template <typename MatrixType>
void schur_subnormal_scale(Index size, bool flushToZero) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using ComplexMatrixType = typename ComplexSchur<MatrixType>::ComplexMatrixType;
  using WideScalar = std::complex<long double>;
  using WideMatrix = Matrix<WideScalar, Dynamic, Dynamic>;

  const RealScalar denormalMin = std::numeric_limits<RealScalar>::denorm_min();
  const RealScalar normalMin = (std::numeric_limits<RealScalar>::min)();
  if (!(denormalMin < normalMin)) return;  // Target has no subnormals, or flushes them to zero.

  // Every coefficient lands in the top of the subnormal range: none of them is normal, yet each still carries most of
  // its mantissa, so an accurate decomposition remains possible. Maxima and reconstructions are taken in long double,
  // whose arithmetic is scalar: a SIMD unit that flushes subnormals (ARMv7 NEON) would zero them.
  const MatrixType A = MatrixType::Random(size, size) * Scalar(normalMin / RealScalar(2));
  const WideMatrix wideA = A.template cast<WideScalar>();
  const long double maxA = wideA.cwiseAbs().maxCoeff();
  VERIFY(maxA > 0 && maxA < static_cast<long double>(normalMin));

  ComplexSchur<MatrixType> schurOfA(size);
  if (!compute_schur(schurOfA, A, flushToZero)) return;
  VERIFY_IS_EQUAL(schurOfA.info(), Success);
  const ComplexMatrixType& U = schurOfA.matrixU();
  const ComplexMatrixType& T = schurOfA.matrixT();
  const WideMatrix wideU = U.template cast<WideScalar>();
  const WideMatrix wideT = T.template cast<WideScalar>();

  // Unitary invariance gives ||T||_F = ||A||_F, so the largest coefficient of T cannot be more than a factor of the
  // dimension below the largest coefficient of A. Returning the zero matrix, as the unscaled reduction did, fails
  // this; isZero() would not, because every coefficient here is much smaller than one.
  VERIFY(wideT.cwiseAbs().maxCoeff() >= maxA / static_cast<long double>(size));
  VERIFY_IS_APPROX(U * U.adjoint(), ComplexMatrixType::Identity(size, size));
  // T rounds back into the subnormal range, where the spacing is denorm_min rather than eps times the coefficient, so
  // bound the residual by that spacing instead of by a relative tolerance.
  const long double residual = (wideA - wideU * wideT * wideU.adjoint()).cwiseAbs().maxCoeff();
  VERIFY(residual <= static_cast<long double>(4 * size * size) * static_cast<long double>(denormalMin));

  // A triangular matrix is its own Schur form and power-of-two scaling is exact, so T reproduces it bit for bit.
  MatrixType triangular = A;
  triangular.template triangularView<StrictlyLower>().setZero();
  compute_schur(schurOfA, triangular, flushToZero);
  VERIFY_IS_EQUAL(schurOfA.info(), Success);
  VERIFY_IS_EQUAL(U, ComplexMatrixType::Identity(size, size));
  VERIFY_IS_EQUAL(WideMatrix(T.template cast<WideScalar>()), WideMatrix(triangular.template cast<WideScalar>()));
}

// Scaling divides by a power of two, so coefficients more than the exponent range below the largest one flush to zero.
// That perturbation is bounded by the smallest subnormal relative to the largest coefficient, far below eps, and the
// unscaled reduction squares such a matrix into overflow anyway.
template <typename MatrixType>
void schur_dynamic_range(Index size) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using ComplexScalar = typename ComplexSchur<MatrixType>::ComplexScalar;
  using ComplexMatrixType = typename ComplexSchur<MatrixType>::ComplexMatrixType;

  const RealScalar huge = numext::sqrt((std::numeric_limits<RealScalar>::max)());
  MatrixType A = MatrixType::Random(size, size) * Scalar(huge);
  A(size - 1, 0) = Scalar(std::numeric_limits<RealScalar>::denorm_min());

  ComplexSchur<MatrixType> schurOfA(A);
  VERIFY_IS_EQUAL(schurOfA.info(), Success);
  const ComplexMatrixType& U = schurOfA.matrixU();
  const ComplexMatrixType& T = schurOfA.matrixT();

  VERIFY(U.allFinite() && T.allFinite());
  VERIFY_IS_APPROX(U * U.adjoint(), ComplexMatrixType::Identity(size, size));
  // Both Frobenius norms overflow at this scale, so isApprox would compare infinity against infinity and any T would
  // pass. Compare at unit scale instead; huge is not a power of two, so this costs one rounding per coefficient.
  const ComplexScalar invHuge = ComplexScalar(RealScalar(1) / huge);
  VERIFY_IS_APPROX(ComplexMatrixType(A.template cast<ComplexScalar>() * invHuge),
                   ComplexMatrixType(U * T * U.adjoint() * invHuge));
}

// A finite coefficient whose modulus overflows made the scale infinite, so the matrix was left unscaled, and the
// iteration overflowed into NoConvergence for about a third of these matrices. Both the copying and the in-place
// decomposition scale. A column of T can overflow where a part of U^* A U exceeds highest; every other column must
// satisfy A u_j = U t_j, checked at the scale 2^-8, where the squared coefficients do not overflow.
template <typename MatrixType>
void schur_overflowing_modulus(Index size) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  const RealScalar big = RealScalar(0.9) * NumTraits<RealScalar>::highest();
  const RealScalar down = numext::ldexp(RealScalar(1), -8);
  // O(n eps) rounding in the unitary transformations.
  const RealScalar tolerance = RealScalar(64 * size) * NumTraits<RealScalar>::epsilon();
  for (int trial = 0; trial < 30; ++trial) {
    MatrixType A = MatrixType::Random(size, size) * Scalar(numext::ldexp(big, -10));
    A(0, size - 1) = Scalar(big, big);
    const MatrixType scaledA = A * down;
    const auto verify = [&](const MatrixType& U, const MatrixType& T) {
      VERIFY(U.isUnitary(tolerance));
      VERIFY(T.template triangularView<StrictlyLower>().toDenseMatrix().isZero(0));
      for (Index j = 0; j < size; ++j) {
        if (!T.col(j).allFinite()) continue;
        VERIFY((scaledA * U.col(j) - U * (T.col(j) * down)).stableNorm() <= tolerance * scaledA.stableNorm());
      }
    };
    ComplexSchur<MatrixType> schurOfA(A);
    VERIFY_IS_EQUAL(schurOfA.info(), Success);
    verify(schurOfA.matrixU(), schurOfA.matrixT());

    MatrixType inPlace = A;
    Ref<MatrixType> ref(inPlace);
    ComplexSchur<Ref<MatrixType>> schurInPlace(ref);
    VERIFY_IS_EQUAL(schurInPlace.info(), Success);
    verify(schurInPlace.matrixU(), inPlace);
  }
}

// A NaN must not be scaled away. maxCoeff has to propagate it, or a matrix that is zero apart from a NaN looks like
// the zero matrix and reports a successful decomposition. Which coefficient carries the NaN matters, because the
// default max reduction keeps whichever operand it visits first.
template <typename MatrixType>
void schur_nan_only(Index size) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;

  for (Index k = 0; k < size; ++k) {
    MatrixType A = MatrixType::Zero(size, size);
    A(k, k) = Scalar(std::numeric_limits<RealScalar>::quiet_NaN());

    ComplexSchur<MatrixType> schurOfA(A);
    VERIFY_IS_NOT_EQUAL(schurOfA.info(), Success);
  }
}

// The zero matrix has no exponent to normalize by; it must still decompose trivially.
template <typename MatrixType>
void schur_zero(Index size) {
  using ComplexMatrixType = typename ComplexSchur<MatrixType>::ComplexMatrixType;

  const MatrixType A = MatrixType::Zero(size, size);
  ComplexSchur<MatrixType> schurOfA(A);
  VERIFY_IS_EQUAL(schurOfA.info(), Success);
  VERIFY(schurOfA.matrixT().isZero());
  VERIFY_IS_EQUAL(schurOfA.matrixU(), ComplexMatrixType::Identity(size, size));
}

template <typename RealScalar>
void schur_shift_scaling() {
  using Scalar = std::complex<RealScalar>;
  using MatrixType = Matrix<Scalar, 2, 2>;
  MatrixType base;
  base << Scalar(1, 1), Scalar(2, -1), Scalar(-3, 1), Scalar(4, -2);
  const MatrixType identity = MatrixType::Identity();
  const int exponents[] = {std::numeric_limits<RealScalar>::min_exponent + 4,
                           std::numeric_limits<RealScalar>::min_exponent / 2 - 8,
                           -1,
                           0,
                           1,
                           std::numeric_limits<RealScalar>::max_exponent / 2 - 8,
                           std::numeric_limits<RealScalar>::max_exponent - 5};
  for (int exponent : exponents) {
    const RealScalar scale = numext::ldexp(RealScalar(1), exponent);
    const MatrixType matrix = base * scale;
    ComplexSchur<MatrixType> schur;
    // Bypass compute()'s input normalization. The nonzero subdiagonal requires a QR shift at the requested scale.
    schur.setMaxIterations(0).computeFromHessenberg(matrix, identity);
    VERIFY_IS_EQUAL(schur.info(), NoConvergence);
    schur.setMaxIterations(ComplexSchur<MatrixType>::m_maxIterationsPerRow * 2);
    schur.computeFromHessenberg(matrix, identity);
    VERIFY_IS_EQUAL(schur.info(), Success);
    const MatrixType unitary = schur.matrixU();
    const MatrixType normalizedT = schur.matrixT() * (RealScalar(1) / scale);
    VERIFY(normalizedT.allFinite());
    VERIFY(unitary.allFinite());
    VERIFY_IS_EQUAL(normalizedT(1, 0), Scalar(0));
    // Two-dimensional QR and reconstruction incur a bounded number of rounding errors per iteration.
    const RealScalar tolerance = RealScalar(128) * NumTraits<RealScalar>::epsilon();
    VERIFY((base - unitary * normalizedT * unitary.adjoint()).norm() <= tolerance * base.norm());
    VERIFY((unitary * unitary.adjoint() - identity).norm() <= tolerance);
    schur.computeFromHessenberg(matrix, identity, false);
    VERIFY_IS_EQUAL(schur.info(), Success);
    VERIFY_IS_EQUAL(schur.matrixT() * (RealScalar(1) / scale), normalizedT);
  }
}

// Reordering must keep A = U T U^*, leave T triangular, move the selected eigenvalues to the leading diagonal positions
// in their previous relative order, and make the leading columns of U a basis of their invariant subspace. The selected
// set is chosen as the half of the spectrum of largest modulus, split at a gap wider than the perturbation the
// rotations introduce, so that the expected order is unambiguous.
template <typename MatrixType>
void schur_reorder(Index size) {
  using ComplexScalar = typename ComplexSchur<MatrixType>::ComplexScalar;
  using ComplexMatrixType = typename ComplexSchur<MatrixType>::ComplexMatrixType;
  using RealScalar = typename MatrixType::RealScalar;

  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  // n^2 rotations, each contributing a rounding of order eps ||A|| with a small constant.
  const RealScalar tolerance = RealScalar(64) * size * size * eps;
  // Orthogonality loses only the rounding of the n^2 plane rotations themselves, which is linear in n per entry.
  const RealScalar orthogonalityTolerance = RealScalar(64) * size * eps;

  for (int counter = 0; counter < g_repeat; ++counter) {
    const MatrixType A = MatrixType::Random(size, size);
    ComplexSchur<MatrixType> schur(A);
    VERIFY_IS_EQUAL(schur.info(), Success);
    const ComplexMatrixType T0 = schur.matrixT();
    const ComplexMatrixType U0 = schur.matrixU();

    const Index wanted = (size + 1) / 2;
    if (wanted >= size) continue;
    std::vector<RealScalar> moduli(size);
    for (Index i = 0; i < size; ++i) moduli[i] = numext::abs(T0.coeff(i, i));
    std::sort(moduli.begin(), moduli.end());
    const RealScalar largestUnselected = moduli[size - wanted - 1];
    const RealScalar smallestSelected = moduli[size - wanted];
    if (!(smallestSelected - largestUnselected > RealScalar(1024) * size * eps * (T0.stableNorm() + 1))) continue;
    const RealScalar threshold = (smallestSelected + largestUnselected) / RealScalar(2);
    const auto select = [threshold](const ComplexScalar& lambda) { return numext::abs(lambda) > threshold; };

    const Index selected = schur.reorder(select);
    VERIFY_IS_EQUAL(selected, wanted);
    const ComplexMatrixType& U = schur.matrixU();
    const ComplexMatrixType& T = schur.matrixT();

    VERIFY((A.template cast<ComplexScalar>() - U * T * U.adjoint()).norm() <= tolerance * A.norm());
    VERIFY((U * U.adjoint() - ComplexMatrixType::Identity(size, size)).norm() <= orthogonalityTolerance);
    VERIFY(T.template triangularView<StrictlyLower>().toDenseMatrix().isZero(0));

    // The leading eigenvalues are the selected ones of the unreordered form, exactly, in the same relative order.
    Index next = 0;
    for (Index i = 0; i < size; ++i) {
      if (!select(T0.coeff(i, i))) continue;
      VERIFY(next < selected);
      VERIFY_IS_EQUAL(T.coeff(next, next), T0.coeff(i, i));
      ++next;
    }
    VERIFY_IS_EQUAL(next, selected);
    for (Index i = selected; i < size; ++i) VERIFY(!select(T.coeff(i, i)));

    // The leading columns span the invariant subspace of the selected eigenvalues: A U1 = U1 T11.
    if (selected > 0) {
      VERIFY((A.template cast<ComplexScalar>() * U.leftCols(selected) -
              U.leftCols(selected) * T.topLeftCorner(selected, selected))
                 .norm() <= tolerance * A.norm());
    }

    // Reordering an already reordered form performs no rotation, so it neither moves nor perturbs anything.
    const ComplexMatrixType Tbefore = T;
    const ComplexMatrixType Ubefore = U;
    VERIFY_IS_EQUAL(schur.reorder(select), selected);
    VERIFY_IS_EQUAL(schur.matrixT(), Tbefore);
    VERIFY_IS_EQUAL(schur.matrixU(), Ubefore);

    // Without U the same rotations are applied to T alone.
    ComplexSchur<MatrixType> withoutU(A, false);
    VERIFY_IS_EQUAL(withoutU.reorder(select), selected);
    VERIFY_IS_EQUAL(withoutU.matrixT(), Tbefore);

    // Selecting nothing and selecting everything are both admissible and leave the form unchanged.
    ComplexSchur<MatrixType> none(A);
    VERIFY_IS_EQUAL(none.reorder([](const ComplexScalar&) { return false; }), Index(0));
    VERIFY_IS_EQUAL(none.matrixT(), T0);
    VERIFY_IS_EQUAL(none.matrixU(), U0);
    ComplexSchur<MatrixType> everything(A);
    VERIFY_IS_EQUAL(everything.reorder([](const ComplexScalar&) { return true; }), size);
    VERIFY_IS_EQUAL(everything.matrixT(), T0);
    VERIFY_IS_EQUAL(everything.matrixU(), U0);
  }
}

// Repeated eigenvalues make the selected set indistinguishable by value, and the rotation that swaps them is
// degenerate: selecting one diagonal position must still perform the real swaps and leave a valid decomposition
// behind. The diagonal must be an exact permutation of the diagonal before the call.
template <typename ComplexScalar>
void schur_reorder_repeated() {
  using MatrixType = Matrix<ComplexScalar, 3, 3>;
  using ComplexMatrixType = typename ComplexSchur<MatrixType>::ComplexMatrixType;
  using RealScalar = typename MatrixType::RealScalar;

  MatrixType jordan = MatrixType::Zero();
  jordan.diagonal().setOnes();
  jordan(0, 1) = jordan(1, 2) = 1;
  const MatrixType identity = MatrixType::Identity();
  MatrixType diagonal = MatrixType::Zero();
  diagonal.diagonal() << ComplexScalar(1), ComplexScalar(2), ComplexScalar(3);
  const MatrixType* inputs[] = {&jordan, &identity, &diagonal};
  for (const MatrixType* input : inputs) {
    for (Index target = 0; target < 3; ++target) {
      ComplexSchur<MatrixType> schur(*input);
      VERIFY_IS_EQUAL(schur.info(), Success);
      int calls = 0;
      const Index selected =
          schur.reorder([&calls, target](const ComplexScalar&) { return ++calls == target + 1; });
      VERIFY_IS_EQUAL(selected, Index(1));
      const ComplexMatrixType& U = schur.matrixU();
      const ComplexMatrixType& T = schur.matrixT();
      const RealScalar tolerance = RealScalar(64) * NumTraits<RealScalar>::epsilon();
      VERIFY((*input).template cast<ComplexScalar>().isApprox(ComplexMatrixType(U * T * U.adjoint()), tolerance));
      VERIFY((U * U.adjoint() - ComplexMatrixType::Identity()).norm() <= RealScalar(64) * tolerance);
      VERIFY(T.template triangularView<StrictlyLower>().toDenseMatrix().isZero(0));
      // The selected diagonal entry leads, and the rest keep their relative order.
      Index next = 1;
      for (Index i = 0; i < 3; ++i) {
        if (i == target) continue;
        VERIFY_IS_EQUAL(T.coeff(next, next), input->coeff(i, i));
        ++next;
      }
      VERIFY_IS_EQUAL(T.coeff(0, 0), input->coeff(target, target));
    }
  }
}

// A finite triangular matrix whose diagonal difference overflows must still swap without producing NaN.
template <typename Scalar>
void schur_reorder_overflow() {
  using ComplexScalar = std::complex<Scalar>;
  using MatrixType = Matrix<ComplexScalar, 2, 2>;
  const Scalar highest = NumTraits<Scalar>::highest();
  MatrixType a = MatrixType::Zero();
  a(0, 0) = ComplexScalar(-Scalar(0.9) * highest, 0);
  a(0, 1) = ComplexScalar(1, 0);
  a(1, 1) = ComplexScalar(Scalar(0.9) * highest, 0);

  ComplexSchur<MatrixType> schur;
  schur.computeFromHessenberg(a, MatrixType::Identity(), true);
  VERIFY_IS_EQUAL(schur.info(), Success);
  int calls = 0;
  const Index selected = schur.reorder([&calls](const ComplexScalar&) { return ++calls == 2; });
  VERIFY_IS_EQUAL(selected, Index(1));
  VERIFY(schur.matrixT().allFinite());
  VERIFY(schur.matrixU().allFinite());
  VERIFY_IS_EQUAL(schur.matrixT()(0, 0), a(1, 1));
  VERIFY_IS_EQUAL(schur.matrixT()(1, 1), a(0, 0));
}

EIGEN_DECLARE_TEST(schur_complex) {
  CALL_SUBTEST_1((schur<Matrix4cd>()));
  CALL_SUBTEST_2((schur<MatrixXcf>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4))));
  CALL_SUBTEST_3((schur<Matrix<std::complex<float>, 1, 1> >()));
  CALL_SUBTEST_4((schur<Matrix<float, 3, 3, Eigen::RowMajor> >()));

  // Test problem size constructors
  CALL_SUBTEST_5(ComplexSchur<MatrixXf>(10));
  CALL_SUBTEST_11((schur<MatrixXcd>(128)));

  CALL_SUBTEST_6((schur_underflow_scale<Matrix4cd>(4)));
  CALL_SUBTEST_6((schur_underflow_scale<MatrixXcf>(8)));
  CALL_SUBTEST_6((schur_underflow_scale<MatrixXf>(8)));
  {
    // Every square this reconstruction compares is subnormal, so run it again with subnormals flushed to zero: that
    // is the configuration in which the unscaled reduction it guards against also passes.
    Eigen::ScopedFlushToZero flush_to_zero;
    if (flush_to_zero.isSupported()) {
      CALL_SUBTEST_6((schur_underflow_scale<Matrix4cd>(4)));
      CALL_SUBTEST_6((schur_underflow_scale<MatrixXcf>(8)));
      CALL_SUBTEST_6((schur_underflow_scale<MatrixXf>(8)));
    }
  }

  CALL_SUBTEST_7((schur_subnormal_scale<Matrix4cd>(4, false)));
  CALL_SUBTEST_7((schur_subnormal_scale<MatrixXcf>(8, false)));
  CALL_SUBTEST_7((schur_subnormal_scale<MatrixXf>(8, false)));
  CALL_SUBTEST_7((schur_subnormal_scale<Matrix4cd>(4, true)));
  CALL_SUBTEST_7((schur_subnormal_scale<MatrixXcf>(8, true)));
  CALL_SUBTEST_7((schur_subnormal_scale<MatrixXf>(8, true)));

  CALL_SUBTEST_8((schur_dynamic_range<Matrix4cd>(4)));
  CALL_SUBTEST_8((schur_dynamic_range<MatrixXcf>(8)));
  CALL_SUBTEST_8((schur_dynamic_range<MatrixXf>(8)));
  CALL_SUBTEST_8((schur_overflowing_modulus<MatrixXcf>(8)));
  CALL_SUBTEST_8((schur_overflowing_modulus<MatrixXcd>(8)));

  CALL_SUBTEST_9((schur_nan_only<Matrix4cd>(4)));
  CALL_SUBTEST_9((schur_nan_only<MatrixXcf>(8)));
  CALL_SUBTEST_9((schur_nan_only<MatrixXf>(8)));

  CALL_SUBTEST_10((schur_zero<Matrix4cd>(4)));
  CALL_SUBTEST_10((schur_zero<MatrixXcf>(8)));
  CALL_SUBTEST_10((schur_zero<MatrixXf>(8)));
  CALL_SUBTEST_10(schur_shift_scaling<float>());
  CALL_SUBTEST_10(schur_shift_scaling<double>());

  CALL_SUBTEST_12((schur_reorder<Matrix4cd>(4)));
  CALL_SUBTEST_12((schur_reorder<MatrixXcd>(internal::random<int>(2, EIGEN_TEST_MAX_SIZE / 4))));
  CALL_SUBTEST_12((schur_reorder<MatrixXcf>(8)));
  CALL_SUBTEST_12((schur_reorder<MatrixXd>(7)));
  CALL_SUBTEST_12((schur_reorder<Matrix<std::complex<float>, 3, 3, Eigen::RowMajor>>(3)));
  CALL_SUBTEST_13(schur_reorder_repeated<std::complex<double>>());
  CALL_SUBTEST_13(schur_reorder_repeated<std::complex<float>>());
  CALL_SUBTEST_14(schur_reorder_overflow<double>());
  CALL_SUBTEST_14(schur_reorder_overflow<float>());
}
