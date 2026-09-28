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

// Architectural Note:
// Eigen's internal Householder routines (Eigen/src/Householder/) implement a subset of
// operations (primarily Left / Forward / Columnwise reflectors). LAPACK routines (larfg,
// larf, larft, larfb) require the complete Cartesian product of side (Left/Right) x
// trans (NoTrans/ConjTrans) x direct (Forward/Backward) x storev (Columnwise/Rowwise),
// as well as Netlib-compliant scaling (safmin/xlamch) and reflector sign conventions.
// They are implemented natively here with zero dynamic heap allocations.

// LARFG generates an elementary reflector H of order n, such that
// H * [alpha, x]^T = [beta, 0]^T, H^* * H = I.
EIGEN_LAPACK_FUNC(larfg)(int* n, RealScalar* palpha, RealScalar* px, int* incx, RealScalar* ptau) {
  Scalar* alpha = reinterpret_cast<Scalar*>(palpha);
  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* tau = reinterpret_cast<Scalar*>(ptau);

  if (*n <= 0) {
    *tau = Scalar(0);
    return;
  }

  int m = *n - 1;
  RealScalar xnorm = RealScalar(0);
  if (m > 0) {
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic> > xvec(
        x, m, Eigen::InnerStride<Eigen::Dynamic>(*incx));
    xnorm = xvec.stableNorm();
  }

#if !ISCOMPLEX
  if (m == 0 || xnorm == RealScalar(0)) {
    *tau = Scalar(0);
    return;
  }

  RealScalar alphar = *alpha;
  RealScalar beta = -Eigen::numext::copysign(Eigen::internal::xlapy2(alphar, xnorm), alphar);
  RealScalar safmin = Eigen::internal::xlamch<RealScalar>('S') / Eigen::internal::xlamch<RealScalar>('E');
  int knt = 0;
  if (Eigen::numext::abs(beta) < safmin) {
    RealScalar rsafmn = RealScalar(1) / safmin;
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic> > xvec(
        x, m, Eigen::InnerStride<Eigen::Dynamic>(*incx));
    do {
      ++knt;
      xvec *= rsafmn;
      beta *= rsafmn;
      alphar *= rsafmn;
    } while (Eigen::numext::abs(beta) < safmin);

    xnorm = xvec.stableNorm();
    beta = -Eigen::numext::copysign(Eigen::internal::xlapy2(alphar, xnorm), alphar);
  }

  *tau = (beta - alphar) / beta;
  {
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic> > xvec(
        x, m, Eigen::InnerStride<Eigen::Dynamic>(*incx));
    xvec *= (RealScalar(1) / (alphar - beta));
  }

  for (int j = 0; j < knt; ++j) {
    beta *= safmin;
  }
  *alpha = beta;

#else
  RealScalar alphr = Eigen::numext::real(*alpha);
  RealScalar alphi = Eigen::numext::imag(*alpha);

  if (Eigen::numext::is_exactly_zero(xnorm) && Eigen::numext::is_exactly_zero(alphi)) {
    *tau = Scalar(0);
    return;
  }

  RealScalar beta = -Eigen::numext::copysign(Eigen::internal::xlapy3(alphr, alphi, xnorm), alphr);
  RealScalar safmin = Eigen::internal::xlamch<RealScalar>('S') / Eigen::internal::xlamch<RealScalar>('E');
  RealScalar rsafmn = RealScalar(1) / safmin;
  int knt = 0;

  if (Eigen::numext::abs(beta) < safmin) {
    do {
      ++knt;
      if (m > 0) {
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic> > xvec(
            x, m, Eigen::InnerStride<Eigen::Dynamic>(*incx));
        xvec *= rsafmn;
      }
      beta *= rsafmn;
      alphi *= rsafmn;
      alphr *= rsafmn;
    } while (Eigen::numext::abs(beta) < safmin);

    if (m > 0) {
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic> > xvec(
          x, m, Eigen::InnerStride<Eigen::Dynamic>(*incx));
      xnorm = xvec.stableNorm();
    }
    *alpha = Scalar(alphr, alphi);
    beta = -Eigen::numext::copysign(Eigen::internal::xlapy3(alphr, alphi, xnorm), alphr);
  }

  *tau = Scalar((beta - alphr) / beta, -alphi / beta);
  Scalar denom = *alpha - beta;
  // For consistency with LAPACK (CLARFG/ZLARFG), compute 1 / (alpha - beta) via xladiv
  // (Smith's algorithm) to avoid intermediate overflow when scaling the reflector vector.
  Scalar scale = Eigen::internal::xladiv(Scalar(1), denom);
  if (m > 0) {
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic> > xvec(
        x, m, Eigen::InnerStride<Eigen::Dynamic>(*incx));
    xvec *= scale;
  }

  for (int j = 0; j < knt; ++j) {
    beta *= safmin;
  }
  *alpha = Scalar(beta, RealScalar(0));
#endif
}

// LARF applies an elementary reflector H = I - tau * v * v^* to matrix C from either left or right.
EIGEN_LAPACK_FUNC(larf)
(const char* side, int* m, int* n, RealScalar* pv, int* incv, RealScalar* ptau, RealScalar* pc, int* ldc,
 RealScalar* pwork) {
  if (*m <= 0 || *n <= 0) return;

  Scalar* v = reinterpret_cast<Scalar*>(pv);
  Scalar* tau = reinterpret_cast<Scalar*>(ptau);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar* work = reinterpret_cast<Scalar*>(pwork);

  if (*tau == Scalar(0)) return;

  bool applyleft = (SIDE(*side) == LEFT);
  int lastv = applyleft ? *m : *n;

  if (*incv > 0) {
    int idx = (lastv - 1) * *incv;
    while (lastv > 0 && v[idx] == Scalar(0)) {
      --lastv;
      idx -= *incv;
    }
  } else {
    int idx = 0;
    while (lastv > 0 && v[idx] == Scalar(0)) {
      --lastv;
      idx -= *incv;
    }
  }

  if (lastv == 0) return;

  int lastc = 0;
  if (applyleft) {
    lastc = Eigen::internal::ilaxlc(lastv, *n, c, *ldc);
  } else {
    lastc = Eigen::internal::ilaxlr(*m, lastv, c, *ldc);
  }

  if (lastc == 0) return;

  if (applyleft) {
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > C_sub(
        c, lastv, lastc, Eigen::OuterStride<>(*ldc));
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > w(work, lastc);

    if (*incv > 0) {
      Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic> > v_sub(
          v, lastv, Eigen::InnerStride<Eigen::Dynamic>(*incv));
      w.noalias() = C_sub.adjoint() * v_sub;
      C_sub.noalias() -= (*tau) * (v_sub * w.adjoint());
    } else {
      Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic> > v_fwd(
          v, lastv, Eigen::InnerStride<Eigen::Dynamic>(-*incv));
      auto v_sub = v_fwd.reverse();
      w.noalias() = C_sub.adjoint() * v_sub;
      C_sub.noalias() -= (*tau) * (v_sub * w.adjoint());
    }
  } else {
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > C_sub(
        c, lastc, lastv, Eigen::OuterStride<>(*ldc));
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > w(work, lastc);

    if (*incv > 0) {
      Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic> > v_sub(
          v, lastv, Eigen::InnerStride<Eigen::Dynamic>(*incv));
      w.noalias() = C_sub * v_sub;
      C_sub.noalias() -= (*tau) * (w * v_sub.adjoint());
    } else {
      Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic> > v_fwd(
          v, lastv, Eigen::InnerStride<Eigen::Dynamic>(-*incv));
      auto v_sub = v_fwd.reverse();
      w.noalias() = C_sub * v_sub;
      C_sub.noalias() -= (*tau) * (w * v_sub.adjoint());
    }
  }
}

// LARFT forms the triangular factor T of a block reflector H of order n.
EIGEN_LAPACK_FUNC(larft)
(const char* direct, const char* storev, int* n, int* k, RealScalar* pv, int* ldv, RealScalar* ptau, RealScalar* pt,
 int* ldt) {
  if (*n == 0 || *k == 0) return;

  const Scalar* v = reinterpret_cast<const Scalar*>(pv);
  Scalar* tau = reinterpret_cast<Scalar*>(ptau);
  Scalar* t = reinterpret_cast<Scalar*>(pt);

  bool forward = (*direct == 'F' || *direct == 'f');
  bool colwise = (*storev == 'C' || *storev == 'c');

  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > T_mat(
      t, *k, *k, Eigen::OuterStride<>(*ldt));

  if (forward) {
    int prevlastv = *n;
    for (int i = 0; i < *k; ++i) {
      prevlastv = Eigen::numext::maxi(prevlastv, i + 1);
      if (tau[i] == Scalar(0)) {
        for (int j = 0; j <= i; ++j) {
          T_mat(j, i) = Scalar(0);
        }
      } else {
        int lastv = i;
        if (colwise) {
          for (lastv = *n - 1; lastv >= i + 1; --lastv) {
            if (v[lastv + i * (*ldv)] != Scalar(0)) break;
          }
          int j = Eigen::numext::mini(lastv, prevlastv - 1);
          if (i > 0) {
            // Reflectors v(i, 0..i-1) have implicit unit diagonal at row i.
            // Split dot product: V_prev(i, :)^H * 1 + V_prev(i+1:j, :)^H * V_cur(i+1:j)
            // avoiding mutation of v[i + i*ldv].
            Eigen::Map<const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>, 0, Eigen::InnerStride<Eigen::Dynamic> >
                V_prev_row(&v[i], i, Eigen::InnerStride<Eigen::Dynamic>(*ldv));
            T_mat.col(i).head(i).noalias() = -tau[i] * V_prev_row.conjugate().transpose();

            if (j > i) {
              int len_sub = j - i;
              Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                         Eigen::OuterStride<> >
                  V_prev_sub(&v[i + 1], len_sub, i, Eigen::OuterStride<>(*ldv));
              Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > V_cur_sub(&v[(i + 1) + i * (*ldv)], len_sub);
              T_mat.col(i).head(i).noalias() += -tau[i] * (V_prev_sub.adjoint() * V_cur_sub);
            }
          }
        } else {
          for (lastv = *n - 1; lastv >= i + 1; --lastv) {
            if (v[i + lastv * (*ldv)] != Scalar(0)) break;
          }
          int j = Eigen::numext::mini(lastv, prevlastv - 1);
          if (i > 0) {
            // Reflectors v(0..i-1, i) have implicit unit diagonal at col i.
            // Split dot product: V_prev(:, i) * 1 + V_prev(:, i+1:j) * V_cur(i+1:j)^*
            // avoiding mutation of v[i + i*ldv].
            Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > V_prev_col(&v[i * (*ldv)], i);
            T_mat.col(i).head(i).noalias() = -tau[i] * V_prev_col;

            if (j > i) {
              int len_sub = j - i;
              Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                         Eigen::OuterStride<> >
                  V_prev_sub(&v[(i + 1) * (*ldv)], i, len_sub, Eigen::OuterStride<>(*ldv));
              Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic> >
                  V_cur_sub(&v[i + (i + 1) * (*ldv)], len_sub, Eigen::InnerStride<Eigen::Dynamic>(*ldv));
              T_mat.col(i).head(i).noalias() += -tau[i] * (V_prev_sub * V_cur_sub.conjugate());
            }
          }
        }

        if (i > 0) {
          // In-place triangular update: T(0:i-1, i) = T(0:i-1, 0:i-1) * T(0:i-1, i)
          for (int r = 0; r < i; ++r) {
            Scalar sum = Scalar(0);
            for (int c = r; c < i; ++c) {
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
    int prevlastv = 0;
    for (int i = *k - 1; i >= 0; --i) {
      if (tau[i] == Scalar(0)) {
        for (int j = i; j < *k; ++j) {
          T_mat(j, i) = Scalar(0);
        }
      } else {
        if (i < *k - 1) {
          int lastv = 0;
          int num_vecs = *k - 1 - i;
          if (colwise) {
            int diag_row = *n - *k + i;
            for (lastv = 0; lastv < i; ++lastv) {
              if (v[lastv + i * (*ldv)] != Scalar(0)) break;
            }
            int j = Eigen::numext::maxi(lastv, prevlastv);

            // Reflectors v(j..diag_row, i+1..k-1) have implicit unit diagonal at row diag_row.
            // Split dot product: V_rest(diag_row, :)^H * 1 + V_rest(j:diag_row-1, :)^H * V_cur(j:diag_row-1)
            // avoiding mutation of v[diag_row + i*ldv].
            Eigen::Map<const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>, 0, Eigen::InnerStride<Eigen::Dynamic> >
                V_rest_row(&v[diag_row + (i + 1) * (*ldv)], num_vecs, Eigen::InnerStride<Eigen::Dynamic>(*ldv));
            T_mat.col(i).tail(num_vecs).noalias() = -tau[i] * V_rest_row.conjugate().transpose();

            if (diag_row > j) {
              int len_sub = diag_row - j;
              Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                         Eigen::OuterStride<> >
                  V_rest_sub(&v[j + (i + 1) * (*ldv)], len_sub, num_vecs, Eigen::OuterStride<>(*ldv));
              Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > V_cur_sub(&v[j + i * (*ldv)], len_sub);
              T_mat.col(i).tail(num_vecs).noalias() += -tau[i] * (V_rest_sub.adjoint() * V_cur_sub);
            }
          } else {
            int diag_col = *n - *k + i;
            for (lastv = 0; lastv < i; ++lastv) {
              if (v[i + lastv * (*ldv)] != Scalar(0)) break;
            }
            int j = Eigen::numext::maxi(lastv, prevlastv);

            // Reflectors v(i+1..k-1, j..diag_col) have implicit unit diagonal at col diag_col.
            // Split dot product: V_rest(:, diag_col) * 1 + V_rest(:, j:diag_col-1) * V_cur(j:diag_col-1)^*
            // avoiding mutation of v[i + diag_col*ldv].
            Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > V_rest_col(&v[(i + 1) + diag_col * (*ldv)],
                                                                                   num_vecs);
            T_mat.col(i).tail(num_vecs).noalias() = -tau[i] * V_rest_col;

            if (diag_col > j) {
              int len_sub = diag_col - j;
              Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                         Eigen::OuterStride<> >
                  V_rest_sub(&v[(i + 1) + j * (*ldv)], num_vecs, len_sub, Eigen::OuterStride<>(*ldv));
              Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic> >
                  V_cur_sub(&v[i + j * (*ldv)], len_sub, Eigen::InnerStride<Eigen::Dynamic>(*ldv));
              T_mat.col(i).tail(num_vecs).noalias() += -tau[i] * (V_rest_sub * V_cur_sub.conjugate());
            }
          }

          // In-place triangular update: T(i+1:k-1, i) = T(i+1:k-1, i+1:k-1) * T(i+1:k-1, i)
          for (int r = num_vecs - 1; r >= 0; --r) {
            int row = (i + 1) + r;
            Scalar sum = Scalar(0);
            for (int c = 0; c <= r; ++c) {
              sum += T_mat(row, (i + 1) + c) * T_mat((i + 1) + c, i);
            }
            T_mat(row, i) = sum;
          }

          if (i < *k - 1) {
            prevlastv = Eigen::numext::mini(prevlastv, lastv);
          } else {
            prevlastv = lastv;
          }
        }
        T_mat(i, i) = tau[i];
      }
    }
  }
}

namespace {

// In-place triangular matrix multiplications: W = W * Tri
// Updates W column-by-column directly within caller-supplied work buffer with zero heap allocations.

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
  // U is Upper Triangular, so U^H is Lower Triangular.
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
  // L is Lower Triangular, so L^H is Upper Triangular.
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

}  // namespace

// LARFB applies a block reflector H or its transpose/adjoint to matrix C.
EIGEN_LAPACK_FUNC(larfb)
(const char* side, const char* trans, const char* direct, const char* storev, int* m, int* n, int* k, RealScalar* pv,
 int* ldv, RealScalar* pt, int* ldt, RealScalar* pc, int* ldc, RealScalar* pwork, int* ldwork) {
  if (*m <= 0 || *n <= 0 || *k <= 0) return;

  const Scalar* v = reinterpret_cast<const Scalar*>(pv);
  const Scalar* t = reinterpret_cast<const Scalar*>(pt);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar* work = reinterpret_cast<Scalar*>(pwork);

  bool applyleft = (SIDE(*side) == LEFT);
  bool notran = (OP(*trans) == NOTR);
  bool forward = (*direct == 'F' || *direct == 'f');
  bool colwise = (*storev == 'C' || *storev == 'c');

  Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
      T_mat(t, *k, *k, Eigen::OuterStride<>(*ldt));

  int lastv = 0;
  int lastc = 0;

  if (colwise) {
    if (forward) {
      if (applyleft) {
        lastv = Eigen::numext::maxi(*k, Eigen::internal::ilaxlr(*m, *k, v, *ldv));
        lastc = Eigen::internal::ilaxlc(lastv, *n, c, *ldc);
        if (lastc == 0) return;

        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > W(
            work, lastc, *k, Eigen::OuterStride<>(*ldwork));
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > C1(
            c, *k, lastc, Eigen::OuterStride<>(*ldc));
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                   Eigen::OuterStride<> >
            V1(v, *k, *k, Eigen::OuterStride<>(*ldv));

        W.noalias() = C1.adjoint() * V1.template triangularView<Eigen::UnitLower>();
        if (lastv > *k) {
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
              C2(&c[*k], lastv - *k, lastc, Eigen::OuterStride<>(*ldc));
          Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     Eigen::OuterStride<> >
              V2(&v[*k], lastv - *k, *k, Eigen::OuterStride<>(*ldv));
          W.noalias() += C2.adjoint() * V2;
        }

        if (notran) {
          apply_triangular_right_upper_adjoint<false>(W, T_mat);
        } else {
          apply_triangular_right_upper<false>(W, T_mat);
        }

        if (lastv > *k) {
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
              C2(&c[*k], lastv - *k, lastc, Eigen::OuterStride<>(*ldc));
          Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     Eigen::OuterStride<> >
              V2(&v[*k], lastv - *k, *k, Eigen::OuterStride<>(*ldv));
          C2.noalias() -= V2 * W.adjoint();
        }

        apply_triangular_right_lower_adjoint<true>(W, V1);
        C1.noalias() -= W.adjoint();

      } else {
        lastv = Eigen::numext::maxi(*k, Eigen::internal::ilaxlr(*n, *k, v, *ldv));
        lastc = Eigen::internal::ilaxlr(*m, lastv, c, *ldc);
        if (lastc == 0) return;

        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > W(
            work, lastc, *k, Eigen::OuterStride<>(*ldwork));
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > C1(
            c, lastc, *k, Eigen::OuterStride<>(*ldc));
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                   Eigen::OuterStride<> >
            V1(v, *k, *k, Eigen::OuterStride<>(*ldv));

        W.noalias() = C1 * V1.template triangularView<Eigen::UnitLower>();
        if (lastv > *k) {
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
              C2(&c[*k * (*ldc)], lastc, lastv - *k, Eigen::OuterStride<>(*ldc));
          Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     Eigen::OuterStride<> >
              V2(&v[*k], lastv - *k, *k, Eigen::OuterStride<>(*ldv));
          W.noalias() += C2 * V2;
        }

        if (notran) {
          apply_triangular_right_upper<false>(W, T_mat);
        } else {
          apply_triangular_right_upper_adjoint<false>(W, T_mat);
        }

        if (lastv > *k) {
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
              C2(&c[*k * (*ldc)], lastc, lastv - *k, Eigen::OuterStride<>(*ldc));
          Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     Eigen::OuterStride<> >
              V2(&v[*k], lastv - *k, *k, Eigen::OuterStride<>(*ldv));
          C2.noalias() -= W * V2.adjoint();
        }

        apply_triangular_right_lower_adjoint<true>(W, V1);
        C1.noalias() -= W;
      }
    } else {
      // colwise && !forward
      if (applyleft) {
        lastv = Eigen::numext::maxi(*k, Eigen::internal::ilaxlr(*m, *k, v, *ldv));
        lastc = Eigen::internal::ilaxlc(lastv, *n, c, *ldc);
        if (lastc == 0) return;

        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > W(
            work, lastc, *k, Eigen::OuterStride<>(*ldwork));
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > C2(
            &c[lastv - *k], *k, lastc, Eigen::OuterStride<>(*ldc));
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                   Eigen::OuterStride<> >
            V2(&v[lastv - *k], *k, *k, Eigen::OuterStride<>(*ldv));

        W.noalias() = C2.adjoint() * V2.template triangularView<Eigen::UnitUpper>();
        if (lastv > *k) {
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
              C1(c, lastv - *k, lastc, Eigen::OuterStride<>(*ldc));
          Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     Eigen::OuterStride<> >
              V1(v, lastv - *k, *k, Eigen::OuterStride<>(*ldv));
          W.noalias() += C1.adjoint() * V1;
        }

        if (notran) {
          apply_triangular_right_lower_adjoint<false>(W, T_mat);
        } else {
          apply_triangular_right_lower<false>(W, T_mat);
        }

        if (lastv > *k) {
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
              C1(c, lastv - *k, lastc, Eigen::OuterStride<>(*ldc));
          Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     Eigen::OuterStride<> >
              V1(v, lastv - *k, *k, Eigen::OuterStride<>(*ldv));
          C1.noalias() -= V1 * W.adjoint();
        }

        apply_triangular_right_upper_adjoint<true>(W, V2);
        C2.noalias() -= W.adjoint();

      } else {
        lastv = Eigen::numext::maxi(*k, Eigen::internal::ilaxlr(*n, *k, v, *ldv));
        lastc = Eigen::internal::ilaxlr(*m, lastv, c, *ldc);
        if (lastc == 0) return;

        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > W(
            work, lastc, *k, Eigen::OuterStride<>(*ldwork));
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > C2(
            &c[(lastv - *k) * (*ldc)], lastc, *k, Eigen::OuterStride<>(*ldc));
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                   Eigen::OuterStride<> >
            V2(&v[lastv - *k], *k, *k, Eigen::OuterStride<>(*ldv));

        W.noalias() = C2 * V2.template triangularView<Eigen::UnitUpper>();
        if (lastv > *k) {
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
              C1(c, lastc, lastv - *k, Eigen::OuterStride<>(*ldc));
          Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     Eigen::OuterStride<> >
              V1(v, lastv - *k, *k, Eigen::OuterStride<>(*ldv));
          W.noalias() += C1 * V1;
        }

        if (notran) {
          apply_triangular_right_lower<false>(W, T_mat);
        } else {
          apply_triangular_right_lower_adjoint<false>(W, T_mat);
        }

        if (lastv > *k) {
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
              C1(c, lastc, lastv - *k, Eigen::OuterStride<>(*ldc));
          Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     Eigen::OuterStride<> >
              V1(v, lastv - *k, *k, Eigen::OuterStride<>(*ldv));
          C1.noalias() -= W * V1.adjoint();
        }

        apply_triangular_right_upper_adjoint<true>(W, V2);
        C2.noalias() -= W;
      }
    }
  } else {
    // storev == 'R'
    if (forward) {
      if (applyleft) {
        lastv = Eigen::numext::maxi(*k, Eigen::internal::ilaxlc(*k, *m, v, *ldv));
        lastc = Eigen::internal::ilaxlc(lastv, *n, c, *ldc);
        if (lastc == 0) return;

        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > W(
            work, lastc, *k, Eigen::OuterStride<>(*ldwork));
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > C1(
            c, *k, lastc, Eigen::OuterStride<>(*ldc));
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                   Eigen::OuterStride<> >
            V1(v, *k, *k, Eigen::OuterStride<>(*ldv));

        W.noalias() = C1.adjoint() * V1.template triangularView<Eigen::UnitUpper>().adjoint();
        if (lastv > *k) {
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
              C2(&c[*k], lastv - *k, lastc, Eigen::OuterStride<>(*ldc));
          Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     Eigen::OuterStride<> >
              V2(&v[*k * (*ldv)], *k, lastv - *k, Eigen::OuterStride<>(*ldv));
          W.noalias() += C2.adjoint() * V2.adjoint();
        }

        if (notran) {
          apply_triangular_right_upper_adjoint<false>(W, T_mat);
        } else {
          apply_triangular_right_upper<false>(W, T_mat);
        }

        if (lastv > *k) {
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
              C2(&c[*k], lastv - *k, lastc, Eigen::OuterStride<>(*ldc));
          Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     Eigen::OuterStride<> >
              V2(&v[*k * (*ldv)], *k, lastv - *k, Eigen::OuterStride<>(*ldv));
          C2.noalias() -= V2.adjoint() * W.adjoint();
        }

        apply_triangular_right_upper<true>(W, V1);
        C1.noalias() -= W.adjoint();

      } else {
        lastv = Eigen::numext::maxi(*k, Eigen::internal::ilaxlc(*k, *n, v, *ldv));
        lastc = Eigen::internal::ilaxlr(*m, lastv, c, *ldc);
        if (lastc == 0) return;

        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > W(
            work, lastc, *k, Eigen::OuterStride<>(*ldwork));
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > C1(
            c, lastc, *k, Eigen::OuterStride<>(*ldc));
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                   Eigen::OuterStride<> >
            V1(v, *k, *k, Eigen::OuterStride<>(*ldv));

        W.noalias() = C1 * V1.template triangularView<Eigen::UnitUpper>().adjoint();
        if (lastv > *k) {
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
              C2(&c[*k * (*ldc)], lastc, lastv - *k, Eigen::OuterStride<>(*ldc));
          Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     Eigen::OuterStride<> >
              V2(&v[*k * (*ldv)], *k, lastv - *k, Eigen::OuterStride<>(*ldv));
          W.noalias() += C2 * V2.adjoint();
        }

        if (notran) {
          apply_triangular_right_upper<false>(W, T_mat);
        } else {
          apply_triangular_right_upper_adjoint<false>(W, T_mat);
        }

        if (lastv > *k) {
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
              C2(&c[*k * (*ldc)], lastc, lastv - *k, Eigen::OuterStride<>(*ldc));
          Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     Eigen::OuterStride<> >
              V2(&v[*k * (*ldv)], *k, lastv - *k, Eigen::OuterStride<>(*ldv));
          C2.noalias() -= W * V2;
        }

        apply_triangular_right_upper<true>(W, V1);
        C1.noalias() -= W;
      }
    } else {
      // storev == 'R' && !forward
      if (applyleft) {
        lastv = Eigen::numext::maxi(*k, Eigen::internal::ilaxlc(*k, *m, v, *ldv));
        lastc = Eigen::internal::ilaxlc(lastv, *n, c, *ldc);
        if (lastc == 0) return;

        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > W(
            work, lastc, *k, Eigen::OuterStride<>(*ldwork));
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > C2(
            &c[lastv - *k], *k, lastc, Eigen::OuterStride<>(*ldc));
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                   Eigen::OuterStride<> >
            V2(&v[(lastv - *k) * (*ldv)], *k, *k, Eigen::OuterStride<>(*ldv));

        W.noalias() = C2.adjoint() * V2.template triangularView<Eigen::UnitLower>().adjoint();
        if (lastv > *k) {
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
              C1(c, lastv - *k, lastc, Eigen::OuterStride<>(*ldc));
          Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     Eigen::OuterStride<> >
              V1(v, *k, lastv - *k, Eigen::OuterStride<>(*ldv));
          W.noalias() += C1.adjoint() * V1.adjoint();
        }

        if (notran) {
          apply_triangular_right_lower_adjoint<false>(W, T_mat);
        } else {
          apply_triangular_right_lower<false>(W, T_mat);
        }

        if (lastv > *k) {
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
              C1(c, lastv - *k, lastc, Eigen::OuterStride<>(*ldc));
          Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     Eigen::OuterStride<> >
              V1(v, *k, lastv - *k, Eigen::OuterStride<>(*ldv));
          C1.noalias() -= V1.adjoint() * W.adjoint();
        }

        apply_triangular_right_lower<true>(W, V2);
        C2.noalias() -= W.adjoint();

      } else {
        lastv = Eigen::numext::maxi(*k, Eigen::internal::ilaxlc(*k, *n, v, *ldv));
        lastc = Eigen::internal::ilaxlr(*m, lastv, c, *ldc);
        if (lastc == 0) return;

        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > W(
            work, lastc, *k, Eigen::OuterStride<>(*ldwork));
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> > C2(
            &c[(lastv - *k) * (*ldc)], lastc, *k, Eigen::OuterStride<>(*ldc));
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                   Eigen::OuterStride<> >
            V2(&v[(lastv - *k) * (*ldv)], *k, *k, Eigen::OuterStride<>(*ldv));

        W.noalias() = C2 * V2.template triangularView<Eigen::UnitLower>().adjoint();
        if (lastv > *k) {
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
              C1(c, lastc, lastv - *k, Eigen::OuterStride<>(*ldc));
          Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     Eigen::OuterStride<> >
              V1(v, *k, lastv - *k, Eigen::OuterStride<>(*ldv));
          W.noalias() += C1 * V1.adjoint();
        }

        if (notran) {
          apply_triangular_right_lower<false>(W, T_mat);
        } else {
          apply_triangular_right_lower_adjoint<false>(W, T_mat);
        }

        if (lastv > *k) {
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<> >
              C1(c, lastc, lastv - *k, Eigen::OuterStride<>(*ldc));
          Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     Eigen::OuterStride<> >
              V1(v, *k, lastv - *k, Eigen::OuterStride<>(*ldv));
          C1.noalias() -= W * V1;
        }

        apply_triangular_right_lower<true>(W, V2);
        C2.noalias() -= W;
      }
    }
  }
}
