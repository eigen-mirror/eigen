// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "lapack_common.h"
#include <Eigen/QR>

// xGEQRF returns A = Q R, Q = H_0 H_1 ... H_{k-1}, H_i = I - tau_i v_i v_i^* with v_i(0:i-1) = 0 and v_i(i) = 1
// implicit, where H_i^* x = beta e_1. Eigen's QR computes the same v_i and beta with (I - tau_i^E v_i v_i^*) x =
// beta e_1, hence tau_i = conj(tau_i^E). Given coefficients c, the block kernels apply H_0 H_1 ... (forward) or
// ... H_1 H_0 (backward) with H_i = I - c_i v_i v_i^*: Q is forward over tau, and Q^* = ... H_1^* H_0^* is backward
// over conj(tau).

namespace {

// ORGQR and ORMQR call the block kernels directly: HouseholderSequence applies fewer than 48 reflectors one at a time.
constexpr Eigen::Index kQrBlockSize = 32;

using QrConstVector = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> >;

// C := Q C or Q^* C, Q = H_0 ... H_{k-1} given by V (nq x k, unit lower trapezoidal) and tau, applied as
// Q = Q_0 Q_1 ... with Q_b = I - V_b T_b V_b^* the product of kQrBlockSize reflectors.
void qr_apply_left(const ConstMatrixType& V, const QrConstVector& tau, MatrixType& C, bool adjoint) {
  const Eigen::Index nq = V.rows(), k = V.cols();
  const Eigen::Index nblocks = (k + kQrBlockSize - 1) / kQrBlockSize;
  for (Eigen::Index b = 0; b < nblocks; ++b) {
    const Eigen::Index i = (adjoint ? b : nblocks - 1 - b) * kQrBlockSize;
    const Eigen::Index ib = Eigen::numext::mini(kQrBlockSize, k - i);
    const auto vectors = V.block(i, i, nq - i, ib);
    const auto coeffs = tau.segment(i, ib);
    auto sub = C.bottomRows(nq - i);
    if (adjoint)
      Eigen::internal::apply_block_householder_on_the_left(sub, vectors, coeffs.conjugate(), false);
    else
      Eigen::internal::apply_block_householder_on_the_left(sub, vectors, coeffs, true);
  }
}

// C := C Q or C Q^*, as above.
void qr_apply_right(const ConstMatrixType& V, const QrConstVector& tau, MatrixType& C, bool adjoint) {
  const Eigen::Index nq = V.rows(), k = V.cols();
  const Eigen::Index nblocks = (k + kQrBlockSize - 1) / kQrBlockSize;
  for (Eigen::Index b = 0; b < nblocks; ++b) {
    const Eigen::Index i = (adjoint ? nblocks - 1 - b : b) * kQrBlockSize;
    const Eigen::Index ib = Eigen::numext::mini(kQrBlockSize, k - i);
    const auto vectors = V.block(i, i, nq - i, ib);
    const auto coeffs = tau.segment(i, ib);
    auto sub = C.rightCols(nq - i);
    if (adjoint)
      Eigen::internal::apply_block_householder_on_the_right(sub, vectors, coeffs.conjugate(), false);
    else
      Eigen::internal::apply_block_householder_on_the_right(sub, vectors, coeffs, true);
  }
}

// A := the first n columns of Q = H_0 ... H_{k-1}, k = tau.size(), over the reflectors in A(:, 0:k). Blocks go last
// to first. On reaching block [i, i+ib), A(:, i+ib:n) holds those columns of the later blocks' product, which vanish
// in rows 0:i+ib; Q_b is applied to their rows i:m, and Q_b(i:m, i:i+ib) replaces the block's reflectors.
void qr_generate_q(MatrixType& A, const QrConstVector& tau) {
  const Eigen::Index m = A.rows(), n = A.cols(), k = tau.size();
  A.rightCols(n - k).setZero();
  A.diagonal().tail(n - k).setOnes();
  const Eigen::Index nblocks = (k + kQrBlockSize - 1) / kQrBlockSize;
  PlainMatrixType panel(m, Eigen::numext::mini(kQrBlockSize, k));
  for (Eigen::Index b = nblocks - 1; b >= 0; --b) {
    const Eigen::Index i = b * kQrBlockSize;
    const Eigen::Index ib = Eigen::numext::mini(kQrBlockSize, k - i);
    const auto V = A.block(i, i, m - i, ib);
    const auto coeffs = tau.segment(i, ib);
    if (i + ib < n) {
      auto trailing = A.block(i, i + ib, m - i, n - i - ib);
      Eigen::internal::apply_block_householder_on_the_left(trailing, V, coeffs, true);
    }
    auto qb = panel.topLeftCorner(m - i, ib);
    qb.setIdentity();
    Eigen::internal::apply_block_householder_on_the_left(qb, V, coeffs, true);
    A.block(i, i, m - i, ib) = qb;
    A.block(0, i, i, ib).setZero();
  }
}

// ORGQR / UNGQR.
void qr_generate_entry(const char* name, EIGEN_BLAS_INT* m, EIGEN_BLAS_INT* n, EIGEN_BLAS_INT* k, RealScalar* pa,
                       EIGEN_BLAS_INT* lda, RealScalar* ptau, RealScalar* pwork, EIGEN_BLAS_INT* lwork,
                       EIGEN_BLAS_INT* info) {
  Scalar* work = reinterpret_cast<Scalar*>(pwork);
  const bool query = *lwork == -1;
  *info = 0;
  work[0] = Scalar(roundup_lwork(std::max<EIGEN_BLAS_INT>(1, *n)));
  if (*m < 0)
    *info = -1;
  else if (*n < 0 || *n > *m)
    *info = -2;
  else if (*k < 0 || *k > *n)
    *info = -3;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *m))
    *info = -5;
  else if (*lwork < std::max<EIGEN_BLAS_INT>(1, *n) && !query)
    *info = -8;
  if (*info != 0) {
    EIGEN_BLAS_INT e = -*info;
    return xerbla_(name, &e, kBlasNameLength);
  }
  if (query) return;
  MatrixType A(reinterpret_cast<Scalar*>(pa), *m, *n, *lda);
  qr_generate_q(A, QrConstVector(reinterpret_cast<const Scalar*>(ptau), *k));
}

// ORMQR / UNMQR.
void qr_multiply_entry(const char* name, const char* side, const char* trans, EIGEN_BLAS_INT* m, EIGEN_BLAS_INT* n,
                       EIGEN_BLAS_INT* k, RealScalar* pa, EIGEN_BLAS_INT* lda, RealScalar* ptau, RealScalar* pc,
                       EIGEN_BLAS_INT* ldc, RealScalar* pwork, EIGEN_BLAS_INT* lwork, EIGEN_BLAS_INT* info) {
  Scalar* work = reinterpret_cast<Scalar*>(pwork);
  const bool query = *lwork == -1;
  const bool left = SIDE(*side) == LEFT;
  const EIGEN_BLAS_INT nq = left ? *m : *n;
  const EIGEN_BLAS_INT nw = std::max<EIGEN_BLAS_INT>(1, left ? *n : *m);
  const bool notrans = OP(*trans) == NOTR;
  const bool adjoint = OP(*trans) == (ISCOMPLEX ? ADJ : TR);
  *info = 0;
  if (SIDE(*side) == INVALID)
    *info = -1;
  else if (!notrans && !adjoint)
    *info = -2;
  else if (*m < 0)
    *info = -3;
  else if (*n < 0)
    *info = -4;
  else if (*k < 0 || *k > nq)
    *info = -5;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, nq))
    *info = -7;
  else if (*ldc < std::max<EIGEN_BLAS_INT>(1, *m))
    *info = -10;
  else if (*lwork < nw && !query)
    *info = -12;
  if (*info != 0) {
    EIGEN_BLAS_INT e = -*info;
    return xerbla_(name, &e, kBlasNameLength);
  }
  work[0] = Scalar(roundup_lwork(nw));
  if (query) return;
  if (*m == 0 || *n == 0 || *k == 0) {
    work[0] = Scalar(1);
    return;
  }
  const ConstMatrixType V(reinterpret_cast<const Scalar*>(pa), nq, *k, *lda);
  const QrConstVector tau(reinterpret_cast<const Scalar*>(ptau), *k);
  MatrixType C(reinterpret_cast<Scalar*>(pc), *m, *n, *ldc);
  if (left)
    qr_apply_left(V, tau, C, adjoint);
  else
    qr_apply_right(V, tau, C, adjoint);
}

}  // namespace

// GEQRF computes A = Q R of a general M-by-N matrix, R on and above the diagonal, the reflectors below it.
EIGEN_LAPACK_FUNC(geqrf)
(EIGEN_BLAS_INT* m, EIGEN_BLAS_INT* n, RealScalar* pa, EIGEN_BLAS_INT* lda, RealScalar* ptau, RealScalar* pwork,
 EIGEN_BLAS_INT* lwork, EIGEN_BLAS_INT* info) {
  const EIGEN_BLAS_INT k = std::min(*m, *n);
  const bool query = *lwork == -1;
  *info = 0;
  if (*m < 0)
    *info = -1;
  else if (*n < 0)
    *info = -2;
  else if (*lda < std::max<EIGEN_BLAS_INT>(1, *m))
    *info = -4;
  else if (!query && (*lwork <= 0 || (*m > 0 && *lwork < std::max<EIGEN_BLAS_INT>(1, *n))))
    *info = -7;
  if (*info != 0) {
    EIGEN_BLAS_INT e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "GEQRF", &e, kBlasNameLength);
  }

  Scalar* work = reinterpret_cast<Scalar*>(pwork);
  if (!query && k > 0) {
    MatrixType A(reinterpret_cast<Scalar*>(pa), *m, *n, *lda);
    CompactVectorType tau(reinterpret_cast<Scalar*>(ptau), k);
    // The kernel's temp takes N scalars, which LWORK >= N provides.
    Eigen::internal::householder_qr_inplace_blocked<MatrixType, CompactVectorType>::run(
        A, tau, Eigen::internal::householder_qr_panel_width<Scalar>(*m, *n), work);
#if ISCOMPLEX
    tau = tau.conjugate();
#endif
  }
  work[0] = Scalar(roundup_lwork(k == 0 ? 1 : *n));
}

#if ISCOMPLEX

// UNGQR generates the first N columns of the product of K reflectors of GEQRF.
EIGEN_LAPACK_FUNC(ungqr)
(EIGEN_BLAS_INT* m, EIGEN_BLAS_INT* n, EIGEN_BLAS_INT* k, RealScalar* pa, EIGEN_BLAS_INT* lda, RealScalar* ptau,
 RealScalar* pwork, EIGEN_BLAS_INT* lwork, EIGEN_BLAS_INT* info) {
  qr_generate_entry(SCALAR_SUFFIX_UP "UNGQR", m, n, k, pa, lda, ptau, pwork, lwork, info);
}

// UNMQR multiplies C by Q or Q^* from the reflectors of GEQRF.
EIGEN_LAPACK_FUNC(unmqr)
(const char* side, const char* trans, EIGEN_BLAS_INT* m, EIGEN_BLAS_INT* n, EIGEN_BLAS_INT* k, RealScalar* pa,
 EIGEN_BLAS_INT* lda, RealScalar* ptau, RealScalar* pc, EIGEN_BLAS_INT* ldc, RealScalar* pwork, EIGEN_BLAS_INT* lwork,
 EIGEN_BLAS_INT* info) {
  qr_multiply_entry(SCALAR_SUFFIX_UP "UNMQR", side, trans, m, n, k, pa, lda, ptau, pc, ldc, pwork, lwork, info);
}

#else

// ORGQR generates the first N columns of the product of K reflectors of GEQRF.
EIGEN_LAPACK_FUNC(orgqr)
(EIGEN_BLAS_INT* m, EIGEN_BLAS_INT* n, EIGEN_BLAS_INT* k, RealScalar* pa, EIGEN_BLAS_INT* lda, RealScalar* ptau,
 RealScalar* pwork, EIGEN_BLAS_INT* lwork, EIGEN_BLAS_INT* info) {
  qr_generate_entry(SCALAR_SUFFIX_UP "ORGQR", m, n, k, pa, lda, ptau, pwork, lwork, info);
}

// ORMQR multiplies C by Q or Q^T from the reflectors of GEQRF.
EIGEN_LAPACK_FUNC(ormqr)
(const char* side, const char* trans, EIGEN_BLAS_INT* m, EIGEN_BLAS_INT* n, EIGEN_BLAS_INT* k, RealScalar* pa,
 EIGEN_BLAS_INT* lda, RealScalar* ptau, RealScalar* pc, EIGEN_BLAS_INT* ldc, RealScalar* pwork, EIGEN_BLAS_INT* lwork,
 EIGEN_BLAS_INT* info) {
  qr_multiply_entry(SCALAR_SUFFIX_UP "ORMQR", side, trans, m, n, k, pa, lda, ptau, pc, ldc, pwork, lwork, info);
}

#endif
