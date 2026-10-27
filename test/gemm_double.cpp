// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Core>

template <int LhsOrder, int RhsOrder, int ResOrder>
void double_gemm_tiles() {
  using Lhs = Matrix<double, Dynamic, Dynamic, LhsOrder>;
  using Rhs = Matrix<double, Dynamic, Dynamic, RhsOrder>;
  using Res = Matrix<double, Dynamic, Dynamic, ResOrder>;
  for (Index m : {2, 3, 4, 6, 7, 13}) {
    for (Index n : {3, 4, 5, 8, 9}) {
      for (Index k : {7, 8, 9, 17, 65}) {
        const Lhs lhs = Lhs::Random(m + 2, k + 3);
        const Rhs rhs = Rhs::Random(k + 2, n + 3);
        const auto a = lhs.block(1, 1, m, k);
        const auto b = rhs.block(1, 1, k, n);
        const Index outer_stride = 2 * (ResOrder == ColMajor ? m : n) + 3;
        const Index size = outer_stride * (ResOrder == ColMajor ? n : m) + 2;
        for (double alpha : {1.0, -0.75}) {
          VectorXd storage = VectorXd::Random(size);
          VectorXd reference = storage;
          Map<Res, Unaligned, Stride<Dynamic, Dynamic>> c(storage.data() + 1, m, n,
                                                          Stride<Dynamic, Dynamic>(outer_stride, 2));
          Map<Res, Unaligned, Stride<Dynamic, Dynamic>> expected(reference.data() + 1, m, n,
                                                                 Stride<Dynamic, Dynamic>(outer_stride, 2));
          VectorXd bounds = VectorXd::Zero(size);
          for (Index j = 0; j < n; ++j) {
            for (Index i = 0; i < m; ++i) {
              double sum = 0;
              double scale = numext::abs(expected(i, j));
              for (Index p = 0; p < k; ++p) {
                sum += a(i, p) * b(p, j);
                scale += numext::abs(alpha) * numext::abs(a(i, p)) * numext::abs(b(p, j));
              }
              expected(i, j) += alpha * sum;
              // Bound both evaluations of C0 + alpha*A*B, including packing and accumulation.
              bounds(&expected(i, j) - reference.data()) = 8 * (k + 1) * NumTraits<double>::epsilon() * scale;
            }
          }
          c.noalias() += alpha * a * b;
          VERIFY(((storage - reference).cwiseAbs().array() <= bounds.array()).all());
        }
      }
    }
  }
}

template <int Order, int UpLo>
void double_syrk_tiles() {
  using Mat = Matrix<double, Dynamic, Dynamic, Order>;
  // mr=3 and nr=4 also change the triangular kernel's least-common-multiple block size to 12.
  for (Index n : {3, 4, 5, 11, 12, 13, 23, 24, 25, 49}) {
    for (Index k : {7, 8, 9, 17, 65}) {
      const Mat a = Mat::Random(n, k);
      Mat c = Mat::Random(n, n);
      Mat expected = c;
      Mat bounds = Mat::Zero(n, n);
      const double alpha = -0.75;
      for (Index j = 0; j < n; ++j) {
        for (Index i = 0; i < n; ++i) {
          if ((UpLo == Lower && i < j) || (UpLo == Upper && i > j)) continue;
          double sum = 0;
          double scale = numext::abs(c(i, j));
          for (Index p = 0; p < k; ++p) {
            sum += a(i, p) * a(j, p);
            scale += numext::abs(alpha) * numext::abs(a(i, p)) * numext::abs(a(j, p));
          }
          expected(i, j) += alpha * sum;
          bounds(i, j) = 8 * (k + 1) * NumTraits<double>::epsilon() * scale;
        }
      }
      c.template selfadjointView<UpLo>().rankUpdate(a, alpha);
      VERIFY(((c - expected).cwiseAbs().array() <= bounds.array()).all());
    }
  }
}

EIGEN_DECLARE_TEST(gemm_double) {
#if EIGEN_ARCH_ARM && defined(EIGEN_VECTORIZE_NEON)
  using Traits = internal::gebp_traits<double, double>;
  STATIC_CHECK(Traits::mr == 3);
  STATIC_CHECK((std::is_same<typename Traits::RhsPacketx4, double>::value));
  using IntTraits = internal::gebp_traits<int, int>;
  using GenericIntTraits = internal::gebp_traits<int, int, false, false, Architecture::Generic>;
  STATIC_CHECK((std::is_same<typename IntTraits::RhsPacketx4, typename GenericIntTraits::RhsPacketx4>::value));
#endif
  CALL_SUBTEST((double_gemm_tiles<ColMajor, ColMajor, ColMajor>()));
  CALL_SUBTEST((double_gemm_tiles<ColMajor, ColMajor, RowMajor>()));
  CALL_SUBTEST((double_gemm_tiles<ColMajor, RowMajor, ColMajor>()));
  CALL_SUBTEST((double_gemm_tiles<ColMajor, RowMajor, RowMajor>()));
  CALL_SUBTEST((double_gemm_tiles<RowMajor, ColMajor, ColMajor>()));
  CALL_SUBTEST((double_gemm_tiles<RowMajor, ColMajor, RowMajor>()));
  CALL_SUBTEST((double_gemm_tiles<RowMajor, RowMajor, ColMajor>()));
  CALL_SUBTEST((double_gemm_tiles<RowMajor, RowMajor, RowMajor>()));
  CALL_SUBTEST((double_syrk_tiles<ColMajor, Lower>()));
  CALL_SUBTEST((double_syrk_tiles<ColMajor, Upper>()));
  CALL_SUBTEST((double_syrk_tiles<RowMajor, Lower>()));
  CALL_SUBTEST((double_syrk_tiles<RowMajor, Upper>()));
}
