// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Exercise the shared kernel even when an ISA-specific TRSM implementation is available.
#define EIGEN_USE_AVX512_TRSM_KERNELS 0
// Keep the cancelling terms in one diagonal panel; synthetic kc=1 moves them into separate GEMM updates.
#define EIGEN_NO_DEBUG_SMALL_PRODUCT_BLOCKS

#include "main.h"
#include <Eigen/Core>

template <typename Scalar>
void trsm_packet_cancellation() {
  STATIC_CHECK((internal::triangular_solve_packet_traits<Scalar>::UseUnblocked ==
                internal::triangular_solve_packet_traits<Scalar>::Enabled));
  using Triangle = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;
  using Operand = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  std::ptrdiff_t saved_l1, saved_l2, saved_l3, saved_l3_per_cpu;
  internal::manage_caching_sizes(GetAction, &saved_l1, &saved_l2, &saved_l3, &saved_l3_per_cpu);
  // Deterministic panels for the blocked cases, including AVX-512 double precision.
  setCpuCacheSizes(64 * 1024, 2 * 1024 * 1024, 16 * 1024 * 1024);
  const Index packet = internal::packet_traits<Scalar>::size;
  const Scalar half_max = (std::numeric_limits<Scalar>::max)() / Scalar(2);
  const Scalar large = ((std::numeric_limits<Scalar>::max)() / Scalar(4)) * Scalar(3);
  for (Index n : {4, 8, 16, 128, 256}) {
    for (Index offset : {Index(0), n - 4}) {
      Triangle a = Triangle::Identity(n, n);
      a(offset + 3, offset) = -half_max;
      a(offset + 3, offset + 1) = half_max;
      a.template triangularView<StrictlyUpper>().setConstant(NumTraits<Scalar>::quiet_NaN());
      const Index widths[] = {packet, 2 * packet, 2 * packet + 1, 4 * packet + 3};
      for (Index nrhs : widths) {
        Operand b = Operand::Ones(n, nrhs), x(n, nrhs);
        b.row(offset + 3).setConstant(large);
        // Only alternating columns overflow, exercising mixed packet lanes and scalar tails.
        for (Index j = 1; j < nrhs; j += 2) b(offset + 3, j) = Scalar(1);
        x = b;
        a.template triangularView<Lower>().solveInPlace(x);
        VERIFY(x.array().isFinite().all());
        // Four eps covers reciprocal/multiply rounding, including fast-math reciprocals.
        VERIFY(((x - b).cwiseAbs().array() <= Scalar(4) * NumTraits<Scalar>::epsilon() * b.array()).all());

        Triangle upper = a.reverse();
        const Operand reversed = b.colwise().reverse();
        x = reversed;
        upper.template triangularView<Upper>().solveInPlace(x);
        VERIFY(x.array().isFinite().all());
        VERIFY(
            ((x - reversed).cwiseAbs().array() <= Scalar(4) * NumTraits<Scalar>::epsilon() * reversed.array()).all());

        a.diagonal().setConstant(NumTraits<Scalar>::quiet_NaN());
        upper.diagonal().setConstant(NumTraits<Scalar>::quiet_NaN());
        x = b;
        a.template triangularView<UnitLower>().solveInPlace(x);
        VERIFY(x.array().isFinite().all());
        VERIFY_IS_CWISE_EQUAL(x, b);
        x = reversed;
        upper.template triangularView<UnitUpper>().solveInPlace(x);
        VERIFY(x.array().isFinite().all());
        VERIFY_IS_CWISE_EQUAL(x, reversed);

        Matrix<Scalar, Dynamic, Dynamic, RowMajor> right = b.transpose();
        a.transpose().template triangularView<UnitUpper>().template solveInPlace<OnTheRight>(right);
        VERIFY(right.array().isFinite().all());
        VERIFY_IS_CWISE_EQUAL(right, b.transpose());
        a.diagonal().setOnes();
      }
    }
  }
  internal::manage_caching_sizes(SetAction, &saved_l1, &saved_l2, &saved_l3, &saved_l3_per_cpu);
}

EIGEN_DECLARE_TEST(trsm_packet_cancellation) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1(trsm_packet_cancellation<float>());
    CALL_SUBTEST_2(trsm_packet_cancellation<double>());
  }
}
