// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "lapack_common.h"

// Eigen's Householder module (Eigen/src/Householder/) covers mainly left-applied, forward, column-wise reflectors.
// xLARFG, xLARF, xLARFT and xLARFB need every combination of SIDE, TRANS, DIRECT and STOREV, as well as reference
// LAPACK's scaling (SAFMIN) and sign conventions, so they are implemented here. Intermediate results live in the
// caller's arrays: T for xLARFT and WORK for xLARF and xLARFB.

// LARFG generates an elementary reflector H of order n, such that
// H * [alpha, x]^T = [beta, 0]^T, H^* * H = I.
EIGEN_LAPACK_FUNC(larfg)
(EIGEN_BLAS_INT* n, RealScalar* palpha, RealScalar* px, EIGEN_BLAS_INT* incx, RealScalar* ptau) {
  Scalar* alpha = reinterpret_cast<Scalar*>(palpha);
  Scalar* tau = reinterpret_cast<Scalar*>(ptau);

  if (*n <= 0) {
    *tau = Scalar(0);
    return;
  }

  const EIGEN_BLAS_INT m = *n - 1;
  StridedVectorType x = make_vector(reinterpret_cast<Scalar*>(px), m, *incx);
  RealScalar xnorm = m > 0 ? x.stableNorm() : RealScalar(0);

#if !ISCOMPLEX
  if (m == 0 || xnorm == RealScalar(0)) {
    *tau = Scalar(0);
    return;
  }

  RealScalar alphar = *alpha;
  RealScalar beta = -Eigen::numext::copysign(Eigen::internal::xlapy2(alphar, xnorm), alphar);
  const RealScalar safmin = Eigen::internal::xlamch<RealScalar>('S') / Eigen::internal::xlamch<RealScalar>('E');
  EIGEN_BLAS_INT knt = 0;
  if (Eigen::numext::abs(beta) < safmin) {
    const RealScalar rsafmn = RealScalar(1) / safmin;
    // At most 20 passes, as in reference xLARFG.
    do {
      ++knt;
      x *= rsafmn;
      beta *= rsafmn;
      alphar *= rsafmn;
    } while (Eigen::numext::abs(beta) < safmin && knt < 20);

    xnorm = x.stableNorm();
    beta = -Eigen::numext::copysign(Eigen::internal::xlapy2(alphar, xnorm), alphar);
  }

  *tau = (beta - alphar) / beta;
  x *= (RealScalar(1) / (alphar - beta));

  for (EIGEN_BLAS_INT j = 0; j < knt; ++j) beta *= safmin;
  *alpha = beta;

#else
  RealScalar alphr = Eigen::numext::real(*alpha);
  RealScalar alphi = Eigen::numext::imag(*alpha);

  if (Eigen::numext::is_exactly_zero(xnorm) && Eigen::numext::is_exactly_zero(alphi)) {
    *tau = Scalar(0);
    return;
  }

  RealScalar beta = -Eigen::numext::copysign(Eigen::internal::xlapy3(alphr, alphi, xnorm), alphr);
  const RealScalar safmin = Eigen::internal::xlamch<RealScalar>('S') / Eigen::internal::xlamch<RealScalar>('E');
  const RealScalar rsafmn = RealScalar(1) / safmin;
  EIGEN_BLAS_INT knt = 0;

  if (Eigen::numext::abs(beta) < safmin) {
    // At most 20 passes, as in reference xLARFG.
    do {
      ++knt;
      x *= rsafmn;
      beta *= rsafmn;
      alphi *= rsafmn;
      alphr *= rsafmn;
    } while (Eigen::numext::abs(beta) < safmin && knt < 20);

    if (m > 0) xnorm = x.stableNorm();
    *alpha = Scalar(alphr, alphi);
    beta = -Eigen::numext::copysign(Eigen::internal::xlapy3(alphr, alphi, xnorm), alphr);
  }

  *tau = Scalar((beta - alphr) / beta, -alphi / beta);
  // 1 / (alpha - beta) by xLADIV, as in reference xLARFG, avoiding the intermediate overflow of naive division.
  x *= Eigen::internal::xladiv(Scalar(1), *alpha - beta);

  for (EIGEN_BLAS_INT j = 0; j < knt; ++j) beta *= safmin;
  *alpha = Scalar(beta, RealScalar(0));
#endif
}

// LARF applies an elementary reflector H = I - tau * v * v^* to matrix C from either left or right.
EIGEN_LAPACK_FUNC(larf)
(const char* side, EIGEN_BLAS_INT* m, EIGEN_BLAS_INT* n, RealScalar* pv, EIGEN_BLAS_INT* incv, RealScalar* ptau,
 RealScalar* pc, EIGEN_BLAS_INT* ldc, RealScalar* pwork) {
  if (*m <= 0 || *n <= 0) return;

  const Scalar* v = reinterpret_cast<const Scalar*>(pv);
  const Scalar* tau = reinterpret_cast<const Scalar*>(ptau);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar* work = reinterpret_cast<Scalar*>(pwork);

  if (*tau == Scalar(0)) return;

  const bool applyleft = (SIDE(*side) == LEFT);

  // Trim trailing zeros of v; for INCV < 0 its last entry is v[0].
  EIGEN_BLAS_INT lastv = applyleft ? *m : *n;
  EIGEN_BLAS_INT idx = *incv > 0 ? (lastv - 1) * *incv : 0;
  while (lastv > 0 && v[idx] == Scalar(0)) {
    --lastv;
    idx -= *incv;
  }
  if (lastv == 0) return;

  const EIGEN_BLAS_INT lastc =
      applyleft ? Eigen::internal::ilaxlc(lastv, *n, c, *ldc) : Eigen::internal::ilaxlr(*m, lastv, c, *ldc);
  if (lastc == 0) return;

  // C := C - tau v (C^H v)^H from the left, or C := C - tau (C v) v^H from the right, with w = C^H v or C v.
  const auto apply_reflector = [&](const auto& v_sub) {
    CompactVectorType w(work, lastc);
    if (applyleft) {
      MatrixType C_sub(c, lastv, lastc, *ldc);
      w.noalias() = C_sub.adjoint() * v_sub;
      C_sub.noalias() -= (*tau) * (v_sub * w.adjoint());
    } else {
      MatrixType C_sub(c, lastc, lastv, *ldc);
      w.noalias() = C_sub * v_sub;
      C_sub.noalias() -= (*tau) * (w * v_sub.adjoint());
    }
  };
  if (*incv > 0) {
    apply_reflector(make_vector(v, lastv, *incv));
  } else {
    auto v_reversed = make_vector(v, lastv, -*incv);
    apply_reflector(v_reversed.reverse());
  }
}

// Read-only views of V's rows and columns.
using ConstVectorType = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> >;
using ConstStridedVectorType = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<> >;
using ConstStridedRowVectorType = Eigen::Map<const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>, 0, Eigen::InnerStride<> >;

// LARFT forms the triangular factor T of a block reflector H of order n.
EIGEN_LAPACK_FUNC(larft)
(const char* direct, const char* storev, EIGEN_BLAS_INT* n, EIGEN_BLAS_INT* k, RealScalar* pv, EIGEN_BLAS_INT* ldv,
 RealScalar* ptau, RealScalar* pt, EIGEN_BLAS_INT* ldt) {
  if (*n == 0 || *k == 0) return;

  const Scalar* v = reinterpret_cast<const Scalar*>(pv);
  const Scalar* tau = reinterpret_cast<const Scalar*>(ptau);

  const bool forward = (*direct == 'F' || *direct == 'f');
  const bool colwise = (*storev == 'C' || *storev == 'c');

  // V(r, c) is v[r + c * ld]: V is n x k for STOREV = 'C' and k x n for STOREV = 'R'.
  const Eigen::Index ld = *ldv;
  MatrixType T_mat(reinterpret_cast<Scalar*>(pt), *k, *k, *ldt);

  if (forward) {
    EIGEN_BLAS_INT prevlastv = *n;
    for (EIGEN_BLAS_INT i = 0; i < *k; ++i) {
      prevlastv = Eigen::numext::maxi(prevlastv, i + 1);
      if (tau[i] == Scalar(0)) {
        for (EIGEN_BLAS_INT j = 0; j <= i; ++j) {
          T_mat(j, i) = Scalar(0);
        }
      } else {
        EIGEN_BLAS_INT lastv = i;
        if (colwise) {
          for (lastv = *n - 1; lastv >= i + 1; --lastv) {
            if (v[lastv + i * ld] != Scalar(0)) break;
          }
          EIGEN_BLAS_INT j = Eigen::numext::mini(lastv, prevlastv - 1);
          if (i > 0) {
            // T(0:i-1, i) = -tau(i) V(i:j, 0:i-1)^H V(i:j, i), with V(i, i) = 1 implicit and not read.
            ConstStridedRowVectorType V_prev_row(&v[i], i, Eigen::InnerStride<>(ld));
            T_mat.col(i).head(i).noalias() = -tau[i] * V_prev_row.conjugate().transpose();

            if (j > i) {
              EIGEN_BLAS_INT len_sub = j - i;
              ConstMatrixType V_prev_sub(&v[i + 1], len_sub, i, ld);
              ConstVectorType V_cur_sub(&v[(i + 1) + i * ld], len_sub);
              T_mat.col(i).head(i).noalias() += -tau[i] * (V_prev_sub.adjoint() * V_cur_sub);
            }
          }
        } else {
          for (lastv = *n - 1; lastv >= i + 1; --lastv) {
            if (v[i + lastv * ld] != Scalar(0)) break;
          }
          EIGEN_BLAS_INT j = Eigen::numext::mini(lastv, prevlastv - 1);
          if (i > 0) {
            // T(0:i-1, i) = -tau(i) V(0:i-1, i:j) V(i, i:j)^H, with V(i, i) = 1 implicit and not read.
            ConstVectorType V_prev_col(&v[i * ld], i);
            T_mat.col(i).head(i).noalias() = -tau[i] * V_prev_col;

            if (j > i) {
              EIGEN_BLAS_INT len_sub = j - i;
              ConstMatrixType V_prev_sub(&v[(i + 1) * ld], i, len_sub, ld);
              ConstStridedVectorType V_cur_sub(&v[i + (i + 1) * ld], len_sub, Eigen::InnerStride<>(ld));
              T_mat.col(i).head(i).noalias() += -tau[i] * (V_prev_sub * V_cur_sub.conjugate());
            }
          }
        }

        if (i > 0) {
          // T(0:i-1, i) = T(0:i-1, 0:i-1) T(0:i-1, i), in place.
          for (EIGEN_BLAS_INT r = 0; r < i; ++r) {
            Scalar sum = Scalar(0);
            for (EIGEN_BLAS_INT c = r; c < i; ++c) {
              sum += T_mat(r, c) * T_mat(c, i);
            }
            T_mat(r, i) = sum;
          }
        }
        T_mat(i, i) = tau[i];
        if (i > 0) {
          prevlastv = Eigen::numext::maxi(prevlastv, lastv + 1);
        } else {
          prevlastv = lastv + 1;
        }
      }
    }
  } else {
    EIGEN_BLAS_INT prevlastv = 0;
    for (EIGEN_BLAS_INT i = *k - 1; i >= 0; --i) {
      if (tau[i] == Scalar(0)) {
        for (EIGEN_BLAS_INT j = i; j < *k; ++j) {
          T_mat(j, i) = Scalar(0);
        }
      } else {
        if (i < *k - 1) {
          EIGEN_BLAS_INT lastv = 0;
          EIGEN_BLAS_INT num_vecs = *k - 1 - i;
          if (colwise) {
            EIGEN_BLAS_INT diag_row = *n - *k + i;
            for (lastv = 0; lastv < i; ++lastv) {
              if (v[lastv + i * ld] != Scalar(0)) break;
            }
            EIGEN_BLAS_INT j = Eigen::numext::maxi(lastv, prevlastv);

            // T(i+1:k-1, i) = -tau(i) V(j:diag_row, i+1:k-1)^H V(j:diag_row, i), with V(diag_row, i) = 1 implicit
            // and not read.
            ConstStridedRowVectorType V_rest_row(&v[diag_row + (i + 1) * ld], num_vecs, Eigen::InnerStride<>(ld));
            T_mat.col(i).tail(num_vecs).noalias() = -tau[i] * V_rest_row.conjugate().transpose();

            if (diag_row > j) {
              EIGEN_BLAS_INT len_sub = diag_row - j;
              ConstMatrixType V_rest_sub(&v[j + (i + 1) * ld], len_sub, num_vecs, ld);
              ConstVectorType V_cur_sub(&v[j + i * ld], len_sub);
              T_mat.col(i).tail(num_vecs).noalias() += -tau[i] * (V_rest_sub.adjoint() * V_cur_sub);
            }
          } else {
            EIGEN_BLAS_INT diag_col = *n - *k + i;
            for (lastv = 0; lastv < i; ++lastv) {
              if (v[i + lastv * ld] != Scalar(0)) break;
            }
            EIGEN_BLAS_INT j = Eigen::numext::maxi(lastv, prevlastv);

            // T(i+1:k-1, i) = -tau(i) V(i+1:k-1, j:diag_col) V(i, j:diag_col)^H, with V(i, diag_col) = 1 implicit
            // and not read.
            ConstVectorType V_rest_col(&v[(i + 1) + diag_col * ld], num_vecs);
            T_mat.col(i).tail(num_vecs).noalias() = -tau[i] * V_rest_col;

            if (diag_col > j) {
              EIGEN_BLAS_INT len_sub = diag_col - j;
              ConstMatrixType V_rest_sub(&v[(i + 1) + j * ld], num_vecs, len_sub, ld);
              ConstStridedVectorType V_cur_sub(&v[i + j * ld], len_sub, Eigen::InnerStride<>(ld));
              T_mat.col(i).tail(num_vecs).noalias() += -tau[i] * (V_rest_sub * V_cur_sub.conjugate());
            }
          }

          // T(i+1:k-1, i) = T(i+1:k-1, i+1:k-1) T(i+1:k-1, i), in place.
          for (EIGEN_BLAS_INT r = num_vecs - 1; r >= 0; --r) {
            EIGEN_BLAS_INT row = (i + 1) + r;
            Scalar sum = Scalar(0);
            for (EIGEN_BLAS_INT c = 0; c <= r; ++c) {
              sum += T_mat(row, (i + 1) + c) * T_mat((i + 1) + c, i);
            }
            T_mat(row, i) = sum;
          }

          prevlastv = Eigen::numext::mini(prevlastv, lastv);
        }
        T_mat(i, i) = tau[i];
      }
    }
  }
}

namespace {

// W := W * op(Tri) in place, one column at a time, for k x k triangular Tri and W with k columns.

template <bool UnitDiag, typename MatrixW, typename MatrixTri>
void apply_triangular_right_upper(MatrixW& W, const MatrixTri& U) {
  const Eigen::Index k = W.cols();
  for (Eigen::Index j = k - 1; j >= 1; --j) {
    if (!UnitDiag) {
      W.col(j) *= U(j, j);
    }
    W.col(j).noalias() += W.leftCols(j) * U.col(j).head(j);
  }
  if (!UnitDiag && k > 0) {
    W.col(0) *= U(0, 0);
  }
}

template <bool UnitDiag, typename MatrixW, typename MatrixTri>
void apply_triangular_right_lower(MatrixW& W, const MatrixTri& L) {
  const Eigen::Index k = W.cols();
  for (Eigen::Index j = 0; j < k - 1; ++j) {
    const Eigen::Index tail_len = k - 1 - j;
    if (!UnitDiag) {
      W.col(j) *= L(j, j);
    }
    W.col(j).noalias() += W.rightCols(tail_len) * L.col(j).tail(tail_len);
  }
  if (!UnitDiag && k > 0) {
    W.col(k - 1) *= L(k - 1, k - 1);
  }
}

template <bool UnitDiag, typename MatrixW, typename MatrixTri>
void apply_triangular_right_upper_adjoint(MatrixW& W, const MatrixTri& U) {
  // U^H is lower triangular.
  const Eigen::Index k = W.cols();
  for (Eigen::Index j = 0; j < k - 1; ++j) {
    const Eigen::Index tail_len = k - 1 - j;
    if (!UnitDiag) {
      W.col(j) *= Eigen::numext::conj(U(j, j));
    }
    W.col(j).noalias() += W.rightCols(tail_len) * U.row(j).tail(tail_len).adjoint();
  }
  if (!UnitDiag && k > 0) {
    W.col(k - 1) *= Eigen::numext::conj(U(k - 1, k - 1));
  }
}

template <bool UnitDiag, typename MatrixW, typename MatrixTri>
void apply_triangular_right_lower_adjoint(MatrixW& W, const MatrixTri& L) {
  // L^H is upper triangular.
  const Eigen::Index k = W.cols();
  for (Eigen::Index j = k - 1; j >= 1; --j) {
    if (!UnitDiag) {
      W.col(j) *= Eigen::numext::conj(L(j, j));
    }
    W.col(j).noalias() += W.leftCols(j) * L.row(j).head(j).adjoint();
  }
  if (!UnitDiag && k > 0) {
    W.col(0) *= Eigen::numext::conj(L(0, 0));
  }
}

// W := W * op(T) for the triangular factor T of xLARFB, which is upper triangular for DIRECT = 'F' and lower for 'B';
// op(T) = T^H if adjoint, else T.
template <typename MatrixW, typename MatrixT>
void apply_triangular_factor(MatrixW& W, const MatrixT& T, bool upper, bool adjoint) {
  if (upper) {
    if (adjoint)
      apply_triangular_right_upper_adjoint<false>(W, T);
    else
      apply_triangular_right_upper<false>(W, T);
  } else if (adjoint) {
    apply_triangular_right_lower_adjoint<false>(W, T);
  } else {
    apply_triangular_right_lower<false>(W, T);
  }
}

}  // namespace

// LARFB applies a block reflector H or its transpose/adjoint to matrix C.
EIGEN_LAPACK_FUNC(larfb)
(const char* side, const char* trans, const char* direct, const char* storev, EIGEN_BLAS_INT* m, EIGEN_BLAS_INT* n,
 EIGEN_BLAS_INT* k, RealScalar* pv, EIGEN_BLAS_INT* ldv, RealScalar* pt, EIGEN_BLAS_INT* ldt, RealScalar* pc,
 EIGEN_BLAS_INT* ldc, RealScalar* pwork, EIGEN_BLAS_INT* ldwork) {
  if (*m <= 0 || *n <= 0 || *k <= 0) return;

  const Scalar* v = reinterpret_cast<const Scalar*>(pv);
  Scalar* c = reinterpret_cast<Scalar*>(pc);

  const bool applyleft = (SIDE(*side) == LEFT);
  const bool notran = (OP(*trans) == NOTR);
  const bool forward = (*direct == 'F' || *direct == 'f');
  const bool colwise = (*storev == 'C' || *storev == 'c');
  // With W = C^H V (left) or C V (right), H C and C H^H take W T^H, while H^H C and C H take W T.
  const bool adjoint_t = (applyleft == notran);
  // Like reference xLARFB, apply H to all of C without trimming V or C to their last nonzero rows or columns. A trim
  // would skip the NaNs and infinities the reference propagates from the rows it drops, and in the backward direction
  // it would cut off the implicit unit triangle in V's last k rows (columns), whose diagonal the caller need not store.
  const EIGEN_BLAS_INT lastv = applyleft ? *m : *n;
  const EIGEN_BLAS_INT lastc = applyleft ? *n : *m;

  ConstMatrixType T_mat(reinterpret_cast<const Scalar*>(pt), *k, *k, *ldt);
  MatrixType W(reinterpret_cast<Scalar*>(pwork), lastc, *k, *ldwork);

  if (colwise) {
    if (forward) {
      if (applyleft) {
        MatrixType C1(c, *k, lastc, *ldc);
        ConstMatrixType V1(v, *k, *k, *ldv);

        W.noalias() = C1.adjoint() * V1.triangularView<Eigen::UnitLower>();
        if (lastv > *k) {
          MatrixType C2(&c[*k], lastv - *k, lastc, *ldc);
          ConstMatrixType V2(&v[*k], lastv - *k, *k, *ldv);
          W.noalias() += C2.adjoint() * V2;
        }
        apply_triangular_factor(W, T_mat, forward, adjoint_t);
        if (lastv > *k) {
          MatrixType C2(&c[*k], lastv - *k, lastc, *ldc);
          ConstMatrixType V2(&v[*k], lastv - *k, *k, *ldv);
          C2.noalias() -= V2 * W.adjoint();
        }
        apply_triangular_right_lower_adjoint<true>(W, V1);
        C1.noalias() -= W.adjoint();
      } else {
        MatrixType C1(c, lastc, *k, *ldc);
        ConstMatrixType V1(v, *k, *k, *ldv);

        W.noalias() = C1 * V1.triangularView<Eigen::UnitLower>();
        if (lastv > *k) {
          MatrixType C2(&c[*k * Eigen::Index(*ldc)], lastc, lastv - *k, *ldc);
          ConstMatrixType V2(&v[*k], lastv - *k, *k, *ldv);
          W.noalias() += C2 * V2;
        }
        apply_triangular_factor(W, T_mat, forward, adjoint_t);
        if (lastv > *k) {
          MatrixType C2(&c[*k * Eigen::Index(*ldc)], lastc, lastv - *k, *ldc);
          ConstMatrixType V2(&v[*k], lastv - *k, *k, *ldv);
          C2.noalias() -= W * V2.adjoint();
        }
        apply_triangular_right_lower_adjoint<true>(W, V1);
        C1.noalias() -= W;
      }
    } else {
      if (applyleft) {
        MatrixType C2(&c[lastv - *k], *k, lastc, *ldc);
        ConstMatrixType V2(&v[lastv - *k], *k, *k, *ldv);

        W.noalias() = C2.adjoint() * V2.triangularView<Eigen::UnitUpper>();
        if (lastv > *k) {
          MatrixType C1(c, lastv - *k, lastc, *ldc);
          ConstMatrixType V1(v, lastv - *k, *k, *ldv);
          W.noalias() += C1.adjoint() * V1;
        }
        apply_triangular_factor(W, T_mat, forward, adjoint_t);
        if (lastv > *k) {
          MatrixType C1(c, lastv - *k, lastc, *ldc);
          ConstMatrixType V1(v, lastv - *k, *k, *ldv);
          C1.noalias() -= V1 * W.adjoint();
        }
        apply_triangular_right_upper_adjoint<true>(W, V2);
        C2.noalias() -= W.adjoint();
      } else {
        MatrixType C2(&c[(lastv - *k) * Eigen::Index(*ldc)], lastc, *k, *ldc);
        ConstMatrixType V2(&v[lastv - *k], *k, *k, *ldv);

        W.noalias() = C2 * V2.triangularView<Eigen::UnitUpper>();
        if (lastv > *k) {
          MatrixType C1(c, lastc, lastv - *k, *ldc);
          ConstMatrixType V1(v, lastv - *k, *k, *ldv);
          W.noalias() += C1 * V1;
        }
        apply_triangular_factor(W, T_mat, forward, adjoint_t);
        if (lastv > *k) {
          MatrixType C1(c, lastc, lastv - *k, *ldc);
          ConstMatrixType V1(v, lastv - *k, *k, *ldv);
          C1.noalias() -= W * V1.adjoint();
        }
        apply_triangular_right_upper_adjoint<true>(W, V2);
        C2.noalias() -= W;
      }
    }
  } else {
    if (forward) {
      if (applyleft) {
        MatrixType C1(c, *k, lastc, *ldc);
        ConstMatrixType V1(v, *k, *k, *ldv);

        W.noalias() = C1.adjoint() * V1.triangularView<Eigen::UnitUpper>().adjoint();
        if (lastv > *k) {
          MatrixType C2(&c[*k], lastv - *k, lastc, *ldc);
          ConstMatrixType V2(&v[*k * Eigen::Index(*ldv)], *k, lastv - *k, *ldv);
          W.noalias() += C2.adjoint() * V2.adjoint();
        }
        apply_triangular_factor(W, T_mat, forward, adjoint_t);
        if (lastv > *k) {
          MatrixType C2(&c[*k], lastv - *k, lastc, *ldc);
          ConstMatrixType V2(&v[*k * Eigen::Index(*ldv)], *k, lastv - *k, *ldv);
          C2.noalias() -= V2.adjoint() * W.adjoint();
        }
        apply_triangular_right_upper<true>(W, V1);
        C1.noalias() -= W.adjoint();
      } else {
        MatrixType C1(c, lastc, *k, *ldc);
        ConstMatrixType V1(v, *k, *k, *ldv);

        W.noalias() = C1 * V1.triangularView<Eigen::UnitUpper>().adjoint();
        if (lastv > *k) {
          MatrixType C2(&c[*k * Eigen::Index(*ldc)], lastc, lastv - *k, *ldc);
          ConstMatrixType V2(&v[*k * Eigen::Index(*ldv)], *k, lastv - *k, *ldv);
          W.noalias() += C2 * V2.adjoint();
        }
        apply_triangular_factor(W, T_mat, forward, adjoint_t);
        if (lastv > *k) {
          MatrixType C2(&c[*k * Eigen::Index(*ldc)], lastc, lastv - *k, *ldc);
          ConstMatrixType V2(&v[*k * Eigen::Index(*ldv)], *k, lastv - *k, *ldv);
          C2.noalias() -= W * V2;
        }
        apply_triangular_right_upper<true>(W, V1);
        C1.noalias() -= W;
      }
    } else {
      if (applyleft) {
        MatrixType C2(&c[lastv - *k], *k, lastc, *ldc);
        ConstMatrixType V2(&v[(lastv - *k) * Eigen::Index(*ldv)], *k, *k, *ldv);

        W.noalias() = C2.adjoint() * V2.triangularView<Eigen::UnitLower>().adjoint();
        if (lastv > *k) {
          MatrixType C1(c, lastv - *k, lastc, *ldc);
          ConstMatrixType V1(v, *k, lastv - *k, *ldv);
          W.noalias() += C1.adjoint() * V1.adjoint();
        }
        apply_triangular_factor(W, T_mat, forward, adjoint_t);
        if (lastv > *k) {
          MatrixType C1(c, lastv - *k, lastc, *ldc);
          ConstMatrixType V1(v, *k, lastv - *k, *ldv);
          C1.noalias() -= V1.adjoint() * W.adjoint();
        }
        apply_triangular_right_lower<true>(W, V2);
        C2.noalias() -= W.adjoint();
      } else {
        MatrixType C2(&c[(lastv - *k) * Eigen::Index(*ldc)], lastc, *k, *ldc);
        ConstMatrixType V2(&v[(lastv - *k) * Eigen::Index(*ldv)], *k, *k, *ldv);

        W.noalias() = C2 * V2.triangularView<Eigen::UnitLower>().adjoint();
        if (lastv > *k) {
          MatrixType C1(c, lastc, lastv - *k, *ldc);
          ConstMatrixType V1(v, *k, lastv - *k, *ldv);
          W.noalias() += C1 * V1.adjoint();
        }
        apply_triangular_factor(W, T_mat, forward, adjoint_t);
        if (lastv > *k) {
          MatrixType C1(c, lastc, lastv - *k, *ldc);
          ConstMatrixType V1(v, *k, lastv - *k, *ldv);
          C1.noalias() -= W * V1;
        }
        apply_triangular_right_lower<true>(W, V2);
        C2.noalias() -= W;
      }
    }
  }
}
