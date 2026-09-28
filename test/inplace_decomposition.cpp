// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>

// This file test inplace decomposition through Ref<>, as supported by the Cholesky, LU, and QR decompositions, the
// Hessenberg, tridiagonal, Schur and QZ reductions, and the dense eigensolvers.

template <typename DecType, typename MatrixType>
void inplace(bool square = false, bool SPD = false) {
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> RhsType;
  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> ResType;

  Index rows = MatrixType::RowsAtCompileTime == Dynamic ? internal::random<Index>(2, EIGEN_TEST_MAX_SIZE / 2)
                                                        : Index(MatrixType::RowsAtCompileTime);
  Index cols = MatrixType::ColsAtCompileTime == Dynamic ? (square ? rows : internal::random<Index>(2, rows))
                                                        : Index(MatrixType::ColsAtCompileTime);

  MatrixType A = MatrixType::Random(rows, cols);
  RhsType b = RhsType::Random(rows);
  ResType x(cols);

  if (SPD) {
    assert(square);
    A.topRows(cols) = A.topRows(cols).adjoint() * A.topRows(cols);
    A.diagonal().array() += 1e-3;
  }

  MatrixType A0 = A;
  MatrixType A1 = A;

  DecType dec(A);

  // Check that the content of A has been modified
  VERIFY_IS_NOT_APPROX(A, A0);

  // Check that the decomposition is correct:
  if (rows == cols) {
    VERIFY_IS_APPROX(A0 * (x = dec.solve(b)), b);
  } else {
    VERIFY_IS_APPROX(A0.transpose() * A0 * (x = dec.solve(b)), A0.transpose() * b);
  }

  // Check that modifying A breaks the current dec:
  A.setRandom();
  if (rows == cols) {
    VERIFY_IS_NOT_APPROX(A0 * (x = dec.solve(b)), b);
  } else {
    VERIFY_IS_NOT_APPROX(A0.transpose() * A0 * (x = dec.solve(b)), A0.transpose() * b);
  }

  // Check that calling compute(A1) does not modify A1:
  A = A0;
  dec.compute(A1);
  VERIFY_IS_EQUAL(A0, A1);
  VERIFY_IS_NOT_APPROX(A, A0);
  if (rows == cols) {
    VERIFY_IS_APPROX(A0 * (x = dec.solve(b)), b);
  } else {
    VERIFY_IS_APPROX(A0.transpose() * A0 * (x = dec.solve(b)), A0.transpose() * b);
  }
}

template <typename MatrixType>
void inplace_fullpivlu_subspaces() {
  using RealScalar = typename MatrixType::RealScalar;
  MatrixType input(3, 3);
  input << 1, 0, 1, 0, 1, 1, 0, 0, 0;
  MatrixType working = input;
  FullPivLU<Ref<MatrixType>> lu(working);
  Ref<MatrixType> original(input);
  const auto kernel = lu.kernel().eval();
  const auto image = lu.image(original).eval();
  STATIC_CHECK((int(decltype(kernel)::IsRowMajor) == int(MatrixType::IsRowMajor)));
  STATIC_CHECK((int(decltype(image)::IsRowMajor) == int(MatrixType::IsRowMajor)));
  VERIFY_IS_EQUAL(kernel.cols(), 1);
  VERIFY_IS_EQUAL(image.cols(), 2);
  const RealScalar tolerance = RealScalar(128 * input.rows()) * NumTraits<RealScalar>::epsilon();
  VERIFY((input * kernel).norm() <= tolerance * input.norm() * kernel.norm());
  VERIFY_IS_EQUAL(kernel.fullPivLu().rank(), kernel.cols());
  VERIFY_IS_EQUAL(image.fullPivLu().rank(), image.cols());
  VERIFY((image * image.fullPivLu().solve(input) - input).norm() <= tolerance * input.norm());
}

template <typename MatrixType>
MatrixType random_selfadjoint(Index n) {
  MatrixType a = MatrixType::Random(n, n);
  return a + a.adjoint();
}

template <typename MatrixType, typename BasisType, typename FormType>
void verify_inplace_similarity(const MatrixType& input, const BasisType& basis, const FormType& form) {
  using RealScalar = typename MatrixType::RealScalar;
  const RealScalar tolerance = RealScalar(128 * input.rows()) * NumTraits<RealScalar>::epsilon();
  VERIFY((input - basis * form * basis.adjoint()).norm() <= tolerance * input.norm());
  VERIFY(basis.isUnitary(tolerance));
}

template <typename MatrixType, typename QZType>
void verify_inplace_qz(const MatrixType& a, const MatrixType& b, const QZType& qz) {
  using RealScalar = typename MatrixType::RealScalar;
  const RealScalar tolerance = RealScalar(128 * a.rows()) * NumTraits<RealScalar>::epsilon();
  VERIFY_IS_EQUAL(qz.info(), Success);
  VERIFY((a - qz.matrixQ() * qz.matrixS() * qz.matrixZ()).norm() <= tolerance * a.norm());
  VERIFY((b - qz.matrixQ() * qz.matrixT() * qz.matrixZ()).norm() <= tolerance * b.norm());
  VERIFY(qz.matrixQ().isUnitary(tolerance));
  VERIFY(qz.matrixZ().isUnitary(tolerance));
}

template <typename MatrixType>
void inplace_reductions(Index size) {
  using Scalar = typename MatrixType::Scalar;
  {
    MatrixType A = MatrixType::Random(size, size), A0 = A;
    HessenbergDecomposition<Ref<MatrixType>> hess(A);
    VERIFY(internal::is_same_dense(hess.packedMatrix(), A));
    MatrixType Q = hess.matrixQ(), H = hess.matrixH();
    verify_inplace_similarity(A0, Q, H);
    MatrixType twiceH = Scalar(2) * hess.matrixH();
    VERIFY_IS_EQUAL(twiceH, Scalar(2) * H);

    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization): Snapshot verifies compute() preserves its input.
    MatrixType A1 = MatrixType::Random(size, size), A1c = A1;
    hess.compute(A1);
    VERIFY_IS_EQUAL(A1, A1c);
    VERIFY(internal::is_same_dense(hess.packedMatrix(), A));
    Q = hess.matrixQ();
    H = hess.matrixH();
    verify_inplace_similarity(A1, Q, H);
  }
  {
    MatrixType A = random_selfadjoint<MatrixType>(size), A0 = A;
    Tridiagonalization<Ref<MatrixType>> tri(A);
    VERIFY(internal::is_same_dense(tri.packedMatrix(), A));
    MatrixType Q = tri.matrixQ(), T = tri.matrixT().eval().template cast<Scalar>();
    verify_inplace_similarity(A0, Q, T);
  }
}

template <typename MatrixType, template <typename> class SchurType>
void inplace_schur(Index size) {
  using RealScalar = typename MatrixType::RealScalar;
  const RealScalar tolerance = RealScalar(128 * size) * NumTraits<RealScalar>::epsilon();
  MatrixType A = MatrixType::Random(size, size), A0 = A;
  SchurType<Ref<MatrixType>> schur(A);
  VERIFY_IS_EQUAL(schur.info(), Success);
  VERIFY(internal::is_same_dense(schur.matrixT(), A));
  verify_inplace_similarity(A0, schur.matrixU(), schur.matrixT());

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization): Snapshot verifies compute() preserves its input.
  MatrixType A1 = MatrixType::Random(size, size), A1c = A1;
  schur.compute(A1);
  VERIFY_IS_EQUAL(schur.info(), Success);
  VERIFY_IS_EQUAL(A1, A1c);
  VERIFY(internal::is_same_dense(schur.matrixT(), A));
  verify_inplace_similarity(A1, schur.matrixU(), schur.matrixT());
  MatrixType T = schur.matrixT();
  schur.compute(A1, false);
  VERIFY_IS_EQUAL(schur.info(), Success);
  VERIFY_IS_EQUAL(A1, A1c);
  VERIFY((schur.matrixT() - T).norm() <= tolerance * A1.norm());
}

template <typename MatrixType>
void inplace_selfadjoint_eigensolver(Index size) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  const RealScalar tolerance = RealScalar(128 * size) * NumTraits<RealScalar>::epsilon();
  MatrixType A = random_selfadjoint<MatrixType>(size), A0 = A;
  A.template triangularView<StrictlyUpper>().setConstant(Scalar(std::numeric_limits<RealScalar>::quiet_NaN()));
  SelfAdjointEigenSolver<Ref<MatrixType>> es(A);
  VERIFY_IS_EQUAL(es.info(), Success);
  VERIFY(internal::is_same_dense(es.eigenvectors(), A));
  VERIFY((A0 * A - A * es.eigenvalues().asDiagonal()).norm() <= tolerance * A0.norm());
  VERIFY(A.isUnitary(tolerance));

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization): Snapshot verifies compute() preserves its input.
  MatrixType A1 = random_selfadjoint<MatrixType>(size), A1c = A1;
  es.compute(A1, EigenvaluesOnly);
  VERIFY_IS_EQUAL(es.info(), Success);
  VERIFY_IS_EQUAL(A1, A1c);
  const auto values = es.eigenvalues().eval();
  es.compute(A1);
  VERIFY_IS_EQUAL(es.info(), Success);
  VERIFY_IS_EQUAL(A1, A1c);
  VERIFY(internal::is_same_dense(es.eigenvectors(), A));
  VERIFY((es.eigenvalues() - values).norm() <= tolerance * A1.norm());
  VERIFY((A1 * A - A * es.eigenvalues().asDiagonal()).norm() <= tolerance * A1.norm());
  VERIFY(A.isUnitary(tolerance));

  if (MatrixType::RowsAtCompileTime == Dynamic) {
    MatrixType big = MatrixType::Random(size + 3, size + 2);
    Ref<MatrixType> B = big.block(2, 1, size, size);
    B = random_selfadjoint<MatrixType>(size);
    MatrixType Bin = B;
    SelfAdjointEigenSolver<Ref<MatrixType>> esb(B);
    VERIFY_IS_EQUAL(esb.info(), Success);
    VERIFY(internal::is_same_dense(esb.eigenvectors(), B));
    VERIFY((Bin * B - B * esb.eigenvalues().asDiagonal()).norm() <= tolerance * Bin.norm());
    VERIFY(B.isUnitary(tolerance));
  }

  MatrixType Ag = random_selfadjoint<MatrixType>(size);
  MatrixType Bg = MatrixType::Random(size, size);
  Bg = Bg * Bg.adjoint() + RealScalar(size) * MatrixType::Identity(size, size);
  for (int type : {int(Ax_lBx), int(ABx_lx), int(BAx_lx)}) {
    MatrixType A2 = Ag, B2 = Bg;
    GeneralizedSelfAdjointEigenSolver<Ref<MatrixType>> ges(A2, B2, ComputeEigenvectors | type);
    VERIFY_IS_EQUAL(ges.info(), Success);
    VERIFY(internal::is_same_dense(ges.eigenvectors(), A2));
    const MatrixType& V = ges.eigenvectors();
    MatrixType VD = V * ges.eigenvalues().asDiagonal();
    if (type == Ax_lBx) {
      VERIFY((Ag * V - Bg * VD).norm() <= tolerance * (Ag.norm() * V.norm() + Bg.norm() * VD.norm()));
    } else if (type == ABx_lx) {
      VERIFY((Ag * (Bg * V) - VD).norm() <= tolerance * (Ag.norm() * Bg.norm() * V.norm() + VD.norm()));
    } else {
      VERIFY((Bg * (Ag * V) - VD).norm() <= tolerance * (Ag.norm() * Bg.norm() * V.norm() + VD.norm()));
    }
    MatrixType metric;
    if (type == BAx_lx) {
      metric = V.adjoint() * Bg.llt().solve(V);
    } else {
      metric = V.adjoint() * Bg * V;
    }
    VERIFY((metric - MatrixType::Identity(size, size)).norm() <= tolerance * RealScalar(size));

    MatrixType A3 = Ag, B3 = Bg;
    GeneralizedSelfAdjointEigenSolver<Ref<MatrixType>> gesv(A3, B3, EigenvaluesOnly | type);
    VERIFY_IS_EQUAL(gesv.info(), Success);
    VERIFY((gesv.eigenvalues() - ges.eigenvalues()).norm() <= tolerance * ges.eigenvalues().norm());
  }
}

template <typename MatrixType>
void inplace_special_values(Index size) {
  using RealScalar = typename MatrixType::RealScalar;
  for (Index kind = 0; kind <= 2 * size; ++kind) {
    MatrixType A0 = MatrixType::Zero(size, size);
    if (kind > 0) {
      const Index index = (kind - 1) % size;
      A0(index, index) =
          kind <= size ? std::numeric_limits<RealScalar>::quiet_NaN() : std::numeric_limits<RealScalar>::infinity();
    }
    MatrixType A = A0;
    RealSchur<Ref<MatrixType>> schur(A);
    RealSchur<MatrixType> schur0(A0);
    VERIFY_IS_EQUAL(schur.info(), schur0.info());
    VERIFY_IS_EQUAL(schur.info(), kind == 0 ? Success : NoConvergence);
    if (schur.info() == Success && kind == 0) verify_inplace_similarity(A0, schur.matrixU(), schur.matrixT());

    MatrixType B = A0;
    SelfAdjointEigenSolver<Ref<MatrixType>> saes(B);
    SelfAdjointEigenSolver<MatrixType> saes0(A0);
    VERIFY_IS_EQUAL(saes.info(), saes0.info());
    VERIFY_IS_EQUAL(saes.info(), kind == 0 ? Success : NoConvergence);
    if (saes.info() == Success && kind == 0) {
      VERIFY_IS_EQUAL(saes.eigenvalues(), saes0.eigenvalues());
      VERIFY(saes.eigenvectors().isUnitary(RealScalar(128 * size) * NumTraits<RealScalar>::epsilon()));
    }

    MatrixType C = A0;
    EigenSolver<Ref<MatrixType>> es(C);
    EigenSolver<MatrixType> es0(A0);
    VERIFY_IS_EQUAL(es.info(), es0.info());
    VERIFY_IS_EQUAL(es.info(), kind == 0 ? Success : NumericalIssue);
    if (es.info() == Success && kind == 0) {
      VERIFY_IS_EQUAL(es.eigenvalues(), es0.eigenvalues());
      VERIFY((es.eigenvectors().colwise().norm().array() > RealScalar(0)).all());
    }
  }
}

template <typename MatrixType, template <typename> class SolverType>
void inplace_eigensolver(Index size) {
  using RealScalar = typename MatrixType::RealScalar;
  using ComplexScalar = std::complex<RealScalar>;
  using ComplexMatrix = typename EigenSolver<MatrixType>::EigenvectorsType;
  const RealScalar tolerance = RealScalar(128 * size) * NumTraits<RealScalar>::epsilon();
  MatrixType A = MatrixType::Random(size, size), A0 = A;
  SolverType<Ref<MatrixType>> es(A);
  VERIFY_IS_EQUAL(es.info(), Success);
  ComplexMatrix V = es.eigenvectors();
  VERIFY((A0.template cast<ComplexScalar>() * V - V * es.eigenvalues().asDiagonal()).norm() <=
         tolerance * A0.norm() * V.norm());
  VERIFY((V.colwise().norm().array() - RealScalar(1)).abs().maxCoeff() <= tolerance);

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization): Snapshot verifies compute() preserves its input.
  MatrixType A1 = MatrixType::Random(size, size), A1c = A1;
  es.compute(A1);
  VERIFY_IS_EQUAL(es.info(), Success);
  VERIFY_IS_EQUAL(A1, A1c);
  V = es.eigenvectors();
  VERIFY((A1.template cast<ComplexScalar>() * V - V * es.eigenvalues().asDiagonal()).norm() <=
         tolerance * A1.norm() * V.norm());
  const auto values = es.eigenvalues().eval();
  es.compute(A1, false);
  VERIFY_IS_EQUAL(es.info(), Success);
  VERIFY_IS_EQUAL(A1, A1c);
  VERIFY((es.eigenvalues() - values).norm() <= tolerance * A1.norm());
}

template <typename MatrixType, template <typename> class QZType>
void inplace_qz(Index size) {
  MatrixType A = MatrixType::Random(size, size), B = MatrixType::Random(size, size), A0 = A, Bin = B;
  QZType<Ref<MatrixType>> qz(A, B);
  VERIFY(internal::is_same_dense(qz.matrixS(), A));
  VERIFY(internal::is_same_dense(qz.matrixT(), B));
  verify_inplace_qz(A0, Bin, qz);

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization): Snapshots verify compute() preserves both inputs.
  MatrixType A1 = MatrixType::Random(size, size), B1 = MatrixType::Random(size, size), A1c = A1, B1c = B1;
  qz.compute(A1, B1);
  VERIFY_IS_EQUAL(A1, A1c);
  VERIFY_IS_EQUAL(B1, B1c);
  VERIFY(internal::is_same_dense(qz.matrixS(), A));
  VERIFY(internal::is_same_dense(qz.matrixT(), B));
  verify_inplace_qz(A1, B1, qz);
  const MatrixType S = qz.matrixS(), T = qz.matrixT();
  qz.compute(A1, B1, false);
  VERIFY_IS_EQUAL(qz.info(), Success);
  VERIFY_IS_EQUAL(A1, A1c);
  VERIFY_IS_EQUAL(B1, B1c);
  using RealScalar = typename MatrixType::RealScalar;
  const RealScalar tolerance = RealScalar(128 * size) * NumTraits<RealScalar>::epsilon();
  VERIFY((qz.matrixS() - S).norm() <= tolerance * A1.norm());
  VERIFY((qz.matrixT() - T).norm() <= tolerance * B1.norm());
}

template <typename MatrixType>
void inplace_generalized_eigensolver(Index size) {
  using RealScalar = typename MatrixType::RealScalar;
  using ComplexScalar = std::complex<RealScalar>;
  using EigenvectorsType = typename GeneralizedEigenSolver<MatrixType>::EigenvectorsType;
  MatrixType A = MatrixType::Random(size, size), B = MatrixType::Random(size, size), A0 = A, Binput = B;
  GeneralizedEigenSolver<Ref<MatrixType>> ges(A, B);
  VERIFY_IS_EQUAL(ges.info(), Success);
  EigenvectorsType V = ges.eigenvectors();
  const EigenvectorsType lhs =
      A0.template cast<ComplexScalar>() * V * ges.betas().template cast<ComplexScalar>().asDiagonal();
  const EigenvectorsType rhs = Binput.template cast<ComplexScalar>() * V * ges.alphas().asDiagonal();
  const RealScalar tolerance = RealScalar(128 * size) * NumTraits<RealScalar>::epsilon();
  VERIFY((lhs - rhs).norm() <= tolerance * (lhs.norm() + rhs.norm()));
  VERIFY((V.colwise().norm().array() - RealScalar(1)).abs().maxCoeff() <= tolerance);
}

template <typename Scalar, template <typename> class QZType, int Options>
void inplace_qz_inner_stride() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Options>;
  using StridedRef = Ref<MatrixType, 0, Stride<Dynamic, Dynamic>>;
  for (Index inner : {Index(1), Index(2)}) {
    MatrixType storageA = MatrixType::Random(8, 5), storageB = MatrixType::Random(8, 5);
    const Stride<Dynamic, Dynamic> stride(8, inner);
    Map<MatrixType, 0, Stride<Dynamic, Dynamic>> a(storageA.data(), 4, 4, stride), b(storageB.data(), 4, 4, stride);
    const MatrixType A0 = a, Binput = b;
    const MatrixType savedA = storageA, savedB = storageB;
    StridedRef ar(a), br(b);
    QZType<StridedRef> qz(ar, br);
    VERIFY(internal::is_same_dense(qz.matrixS(), a));
    VERIFY(internal::is_same_dense(qz.matrixT(), b));
    verify_inplace_qz(A0, Binput, qz);
    // The Map's strides determine its footprint in either storage order.
    for (Index offset = 0; offset < storageA.size(); ++offset) {
      const Index outer = offset / stride.outer();
      const Index innerOffset = offset % stride.outer();
      if (outer < 4 && innerOffset % inner == 0 && innerOffset / inner < 4) continue;
      VERIFY_IS_EQUAL(storageA.data()[offset], savedA.data()[offset]);
      VERIFY_IS_EQUAL(storageB.data()[offset], savedB.data()[offset]);
    }
  }
}

struct InplaceLowerOnly {
  double operator()(Index row, Index col) const {
    VERIFY(row >= col);
    return row == col ? 4.0 : 1.0;
  }
};

void inplace_plain_lower_triangle() {
  auto input = MatrixXd::NullaryExpr(3, 3, InplaceLowerOnly());
  SelfAdjointEigenSolver<MatrixXd> mutableSolver(input);
  const auto constInput = input;
  SelfAdjointEigenSolver<MatrixXd> constSolver(constInput);
  VERIFY_IS_EQUAL(mutableSolver.info(), Success);
  VERIFY_IS_EQUAL(mutableSolver.eigenvalues(), constSolver.eigenvalues());
}

EIGEN_DECLARE_TEST(inplace_decomposition) {
  CALL_SUBTEST_4((inplace_fullpivlu_subspaces<Matrix<double, Dynamic, Dynamic, ColMajor>>()));
  CALL_SUBTEST_4((inplace_fullpivlu_subspaces<Matrix<double, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST_4((inplace_fullpivlu_subspaces<Matrix<double, 3, 3, RowMajor | DontAlign>>()));
  CALL_SUBTEST_10((inplace_special_values<MatrixXd>(2)));
  CALL_SUBTEST_10((inplace_special_values<MatrixXd>(1)));
  CALL_SUBTEST_11(inplace_plain_lower_triangle());
  CALL_SUBTEST_12((inplace_eigensolver<MatrixXd, EigenSolver>(128)));
  CALL_SUBTEST_12((inplace_eigensolver<MatrixXd, EigenSolver>(129)));
  CALL_SUBTEST_14((inplace_qz_inner_stride<double, RealQZ, ColMajor>()));
  CALL_SUBTEST_14((inplace_qz_inner_stride<double, RealQZ, RowMajor>()));
  CALL_SUBTEST_14((inplace_qz_inner_stride<std::complex<double>, ComplexQZ, ColMajor>()));
  CALL_SUBTEST_14((inplace_qz_inner_stride<std::complex<double>, ComplexQZ, RowMajor>()));
  EIGEN_UNUSED typedef Matrix<double, 4, 3> Matrix43d;
  for (int i = 0; i < g_repeat; i++) {
    EIGEN_UNUSED const Index size = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE / 4);

    CALL_SUBTEST_1((inplace<LLT<Ref<MatrixXd> >, MatrixXd>(true, true)));
    CALL_SUBTEST_1((inplace<LLT<Ref<Matrix4d> >, Matrix4d>(true, true)));

    CALL_SUBTEST_2((inplace<LDLT<Ref<MatrixXd> >, MatrixXd>(true, true)));
    CALL_SUBTEST_2((inplace<LDLT<Ref<Matrix4d> >, Matrix4d>(true, true)));

    CALL_SUBTEST_3((inplace<PartialPivLU<Ref<MatrixXd> >, MatrixXd>(true, false)));
    CALL_SUBTEST_3((inplace<PartialPivLU<Ref<Matrix4d> >, Matrix4d>(true, false)));

    CALL_SUBTEST_4((inplace<FullPivLU<Ref<MatrixXd> >, MatrixXd>(true, false)));
    CALL_SUBTEST_4((inplace<FullPivLU<Ref<Matrix4d> >, Matrix4d>(true, false)));

    CALL_SUBTEST_5((inplace<HouseholderQR<Ref<MatrixXd> >, MatrixXd>(false, false)));
    CALL_SUBTEST_5((inplace<HouseholderQR<Ref<Matrix43d> >, Matrix43d>(false, false)));

    CALL_SUBTEST_6((inplace<ColPivHouseholderQR<Ref<MatrixXd> >, MatrixXd>(false, false)));
    CALL_SUBTEST_6((inplace<ColPivHouseholderQR<Ref<Matrix43d> >, Matrix43d>(false, false)));

    CALL_SUBTEST_7((inplace<FullPivHouseholderQR<Ref<MatrixXd> >, MatrixXd>(false, false)));
    CALL_SUBTEST_7((inplace<FullPivHouseholderQR<Ref<Matrix43d> >, Matrix43d>(false, false)));

    CALL_SUBTEST_8((inplace<CompleteOrthogonalDecomposition<Ref<MatrixXd> >, MatrixXd>(false, false)));
    CALL_SUBTEST_8((inplace<CompleteOrthogonalDecomposition<Ref<Matrix43d> >, Matrix43d>(false, false)));

    CALL_SUBTEST_2((inplace<BunchKaufman<Ref<MatrixXd>>, MatrixXd>(true, true)));
    CALL_SUBTEST_2((inplace<BunchKaufman<Ref<Matrix4d>>, Matrix4d>(true, true)));

    CALL_SUBTEST_9((inplace_reductions<MatrixXd>(size)));
    CALL_SUBTEST_9((inplace_reductions<MatrixXcd>(size)));
    CALL_SUBTEST_9((inplace_reductions<MatrixXd>(1)));
    CALL_SUBTEST_9((inplace_reductions<Matrix4f>(4)));

    CALL_SUBTEST_10((inplace_schur<MatrixXd, RealSchur>(size)));
    CALL_SUBTEST_10((inplace_schur<MatrixXd, RealSchur>(1)));
    CALL_SUBTEST_10((inplace_schur<Matrix4f, RealSchur>(4)));
    CALL_SUBTEST_10((inplace_schur<MatrixXcd, ComplexSchur>(size)));
    CALL_SUBTEST_10((inplace_schur<MatrixXcd, ComplexSchur>(1)));
    CALL_SUBTEST_10((inplace_schur<Matrix4cf, ComplexSchur>(4)));
    CALL_SUBTEST_10((inplace_special_values<MatrixXd>(size)));
    CALL_SUBTEST_10((inplace_special_values<Matrix4f>(4)));

    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<MatrixXd>(size)));
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<MatrixXcd>(size)));
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<MatrixXd>(1)));
    // Fixed sizes whose columns are packet-aligned take the vectorized rotation kernel.
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<Matrix2d>(2)));
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<Matrix3d>(3)));
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<Matrix4f>(4)));
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<Matrix4d>(4)));
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<Matrix<float, Dynamic, Dynamic, RowMajor>>(size)));
    // Large enough for the blocked tridiagonalization, on strided and row-major storage.
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<MatrixXd>(100)));
    CALL_SUBTEST_11((inplace_selfadjoint_eigensolver<Matrix<float, Dynamic, Dynamic, RowMajor>>(100)));

    CALL_SUBTEST_12((inplace_eigensolver<MatrixXd, EigenSolver>(size)));
    CALL_SUBTEST_12((inplace_eigensolver<MatrixXd, EigenSolver>(1)));
    CALL_SUBTEST_12((inplace_eigensolver<Matrix4f, EigenSolver>(4)));
    CALL_SUBTEST_12((inplace_eigensolver<MatrixXcd, ComplexEigenSolver>(size)));
    CALL_SUBTEST_12((inplace_eigensolver<MatrixXcd, ComplexEigenSolver>(1)));
    CALL_SUBTEST_12((inplace_eigensolver<Matrix4cf, ComplexEigenSolver>(4)));

    CALL_SUBTEST_13((inplace_qz<MatrixXd, RealQZ>(size)));
    CALL_SUBTEST_13((inplace_generalized_eigensolver<MatrixXd>(size)));
    CALL_SUBTEST_13((inplace_generalized_eigensolver<Matrix4d>(4)));
    CALL_SUBTEST_13((inplace_qz<MatrixXd, RealQZ>(1)));
    CALL_SUBTEST_13((inplace_qz<Matrix4d, RealQZ>(4)));
    CALL_SUBTEST_13((inplace_qz<MatrixXcd, ComplexQZ>(size)));
    CALL_SUBTEST_13((inplace_qz<MatrixXcd, ComplexQZ>(1)));
  }
}
