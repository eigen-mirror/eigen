// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "product_test_helpers.h"

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
          const VectorXd original = storage;
          Map<Res, Unaligned, Stride<Dynamic, Dynamic>> c(storage.data() + 1, m, n,
                                                          Stride<Dynamic, Dynamic>(outer_stride, 2));
          const Res c0 = c;
          const Res expected = c0 + alpha * a.lazyProduct(b);
          c.noalias() += alpha * a * b;
          VERIFY(verifyProductUpdateCoeffwise(c, expected, alpha, a, b, c0));
          VERIFY(strided_block_gaps_untouched(storage, original, 1, ResOrder == ColMajor ? m : n,
                                              ResOrder == ColMajor ? n : m, 2, outer_stride));
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
      const Mat c0 = c;
      const double alpha = -0.75;
      const Mat expected = c0 + alpha * a.lazyProduct(a.transpose());
      c.template selfadjointView<UpLo>().rankUpdate(a, alpha);
      // Only the UpLo triangle is updated.
      const Mat actual = c.template triangularView<UpLo>();
      const Mat actual_c0 = c0.template triangularView<UpLo>();
      const Mat expected_triangle = expected.template triangularView<UpLo>();
      VERIFY(verifyProductUpdateCoeffwise(actual, expected_triangle, alpha, a, a.transpose(), actual_c0));
      constexpr int Opposite = UpLo == Lower ? StrictlyUpper : StrictlyLower;
      VERIFY(bitwise_equal(Mat(c.template triangularView<Opposite>()), Mat(c0.template triangularView<Opposite>())));
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
