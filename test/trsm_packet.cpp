// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Core>

template <typename Scalar, int TriOrder, int OtherOrder, int Side, unsigned int Mode, int InnerStride>
void trsm_packet_case(Index n, Index nrhs, Index inner, bool well_conditioned = true, Scalar magnitude = Scalar(1)) {
  using Triangle = Matrix<Scalar, Dynamic, Dynamic, TriOrder>;
  using Operand = Matrix<Scalar, Dynamic, Dynamic, OtherOrder>;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  using Strides = Stride<Dynamic, InnerStride>;
  const Index rows = Side == OnTheLeft ? n : nrhs, cols = Side == OnTheLeft ? nrhs : n;
  const Index outer = inner * (OtherOrder == ColMajor ? rows : cols) + 3;
  Vector storage = Vector::Constant(outer * (OtherOrder == ColMajor ? cols : rows) + 2, Scalar(17));
  const Vector before = storage;
  Map<Operand, Unaligned, Strides> x(storage.data() + 1, rows, cols, Strides(outer, inner));
  Triangle a = Triangle::Random(n, n);
  if (well_conditioned) a /= Scalar(n + 1);
  a.diagonal().setOnes();
  a *= magnitude;
  const Triangle saved = a;
  const Operand b = Operand::Random(rows, cols);
  x = b;
  // Poison every coefficient that the solve must not read, including a unit diagonal.
  a.template triangularView<(Mode & Lower) ? StrictlyUpper : StrictlyLower>().setConstant(
      NumTraits<Scalar>::quiet_NaN());
  if (Mode & UnitDiag) a.diagonal().setConstant(NumTraits<Scalar>::quiet_NaN());
  a.template triangularView<Mode>().template solveInPlace<Side>(x);

  // |A X - B| <= O(n*eps) * (|A|*|X| + |B|). The factor 8 covers blocked updates
  // and the independent residual accumulation in wider precision.
  const long double gamma = 8 * n * static_cast<long double>(NumTraits<Scalar>::epsilon());
  for (Index j = 0; j < cols; ++j) {
    for (Index i = 0; i < rows; ++i) {
      long double sum = 0, scale = numext::abs(static_cast<long double>(b(i, j)));
      for (Index k = 0; k < n; ++k) {
        const Index r = Side == OnTheLeft ? i : k, c = Side == OnTheLeft ? k : j;
        if ((Mode & Lower) ? r < c : r > c) continue;
        const long double t = r == c && (Mode & UnitDiag) ? 1 : static_cast<long double>(saved(r, c));
        const long double v = Side == OnTheLeft ? static_cast<long double>(x(k, j)) : static_cast<long double>(x(i, k));
        sum += t * v;
        scale += numext::abs(t * v);
      }
      VERIFY((numext::isfinite)(scale));
      VERIFY(numext::abs(sum - static_cast<long double>(b(i, j))) <= gamma * scale);
    }
  }
  Vector expected = before;
  Map<Operand, Unaligned, Strides> mapped(expected.data() + 1, rows, cols, Strides(outer, inner));
  mapped = x;
  VERIFY_IS_CWISE_EQUAL(storage, expected);
}

template <typename Scalar, int TriOrder, int OtherOrder, int Side, int InnerStride = 1>
void trsm_packet_modes(Index n, Index nrhs, Index inner = 1) {
  trsm_packet_case<Scalar, TriOrder, OtherOrder, Side, Lower, InnerStride>(n, nrhs, inner);
  trsm_packet_case<Scalar, TriOrder, OtherOrder, Side, Upper, InnerStride>(n, nrhs, inner);
  trsm_packet_case<Scalar, TriOrder, OtherOrder, Side, UnitLower, InnerStride>(n, nrhs, inner);
  trsm_packet_case<Scalar, TriOrder, OtherOrder, Side, UnitUpper, InnerStride>(n, nrhs, inner);
}

template <typename Scalar>
void trsm_packet_cache_sizes() {
  using Traits = internal::triangular_solve_packet_traits<Scalar>;
  std::ptrdiff_t saved_l1, saved_l2, saved_l3, saved_l3_per_cpu;
  internal::manage_caching_sizes(GetAction, &saved_l1, &saved_l2, &saved_l3, &saved_l3_per_cpu);
  const Index packet = Traits::PacketSize;
  const Index widths[] = {packet, Traits::RhsPackets * packet, Traits::RhsPackets * packet + 1};
  const auto cutoff_for = [](Index nrhs) {
    Index cutoff = 0;
    for (Index n = Traits::RegisterRows; n <= Traits::WorkspaceRows; ++n) {
      if (!Traits::use_unblocked(n, nrhs, l1CacheSize())) break;
      cutoff = n;
    }
    return cutoff;
  };
  Index previous = 0;
  for (std::ptrdiff_t l1 : {512, 2048, 8192, 32768}) {
    setCpuCacheSizes(l1, 4 * l1, 16 * l1);
    const Index wide_cutoff = cutoff_for(widths[1]);
    VERIFY(wide_cutoff >= previous);
    previous = wide_cutoff;
    for (Index nrhs : widths) {
      const Index cutoff = cutoff_for(nrhs);
      VERIFY(cutoff >= 0 && cutoff <= Traits::WorkspaceRows);
      if (nrhs == packet) VERIFY(cutoff >= wide_cutoff);
      // Exercise both algorithms at the configured crossover, plus the allocation bound.
      const Index sizes[] = {
          (numext::maxi)(Index(Traits::RegisterRows), cutoff - 1), (numext::maxi)(Index(Traits::RegisterRows), cutoff),
          (numext::maxi)(Index(Traits::RegisterRows), cutoff + 1), Traits::WorkspaceRows, Traits::WorkspaceRows + 1};
      for (Index n : sizes) {
        trsm_packet_modes<Scalar, ColMajor, ColMajor, OnTheLeft>(n, nrhs);
        trsm_packet_modes<Scalar, RowMajor, ColMajor, OnTheLeft>(n, nrhs);
        trsm_packet_modes<Scalar, ColMajor, RowMajor, OnTheRight>(n, nrhs);
        trsm_packet_modes<Scalar, RowMajor, RowMajor, OnTheRight>(n, nrhs);
      }
    }
  }
  VERIFY(!Traits::use_unblocked(Index(Traits::RegisterRows), packet - 1, saved_l1));
  VERIFY(!Traits::use_unblocked(Index(Traits::RegisterRows), packet, std::ptrdiff_t(0)));
  VERIFY(!Traits::use_unblocked(Index(Traits::RegisterRows), packet, (std::numeric_limits<std::ptrdiff_t>::min)()));
  VERIFY(
      !Traits::use_unblocked(Index(Traits::WorkspaceRows + 1), packet, (std::numeric_limits<std::ptrdiff_t>::max)()));
  VERIFY(Traits::use_unblocked(Index(Traits::WorkspaceRows), widths[1], (std::numeric_limits<std::ptrdiff_t>::max)()) ==
         Traits::UseUnblocked);
  VERIFY(!Traits::use_unblocked((std::numeric_limits<Index>::max)(), packet,
                                (std::numeric_limits<std::ptrdiff_t>::max)()));
  internal::manage_caching_sizes(SetAction, &saved_l1, &saved_l2, &saved_l3, &saved_l3_per_cpu);
}

template <typename Scalar>
void trsm_packet() {
  STATIC_CHECK((internal::triangular_solve_packet_traits<Scalar>::Enabled ==
                bool(internal::packet_traits<Scalar>::Vectorizable && std::numeric_limits<Scalar>::is_iec559 &&
                     std::numeric_limits<Scalar>::radix == 2)));
  STATIC_CHECK((!internal::triangular_solve_packet_traits<std::complex<Scalar>>::Enabled));
  const Index packet = internal::packet_traits<Scalar>::size;
  const Index widths[] = {1, 2, packet, 2 * packet - 1, 2 * packet, 2 * packet + 1, 4 * packet + 3};
  const Index sizes[] = {0, 1, 3, 4, 7, 8, 11, 12, 16, 23, 24, 25, 31, 64, 127, 128, 129, 257};
  for (Index n : sizes) {
    for (Index nrhs : widths) {
      trsm_packet_modes<Scalar, ColMajor, ColMajor, OnTheLeft>(n, nrhs);
      trsm_packet_modes<Scalar, RowMajor, ColMajor, OnTheLeft>(n, nrhs);
      trsm_packet_modes<Scalar, ColMajor, RowMajor, OnTheRight>(n, nrhs);
      trsm_packet_modes<Scalar, RowMajor, RowMajor, OnTheRight>(n, nrhs);
    }
  }
  trsm_packet_modes<Scalar, ColMajor, ColMajor, OnTheLeft, Dynamic>(65, 19, 1);
  trsm_packet_modes<Scalar, RowMajor, ColMajor, OnTheLeft, Dynamic>(65, 19, 2);
  trsm_packet_modes<Scalar, ColMajor, RowMajor, OnTheRight, 2>(65, 19, 2);
  // Unscaled off-diagonals allow inverse growth; only backward error is checked.
  trsm_packet_case<Scalar, ColMajor, ColMajor, OnTheLeft, Lower, 1>(33, 19, 1, false);
  trsm_packet_case<Scalar, RowMajor, ColMajor, OnTheLeft, Upper, 1>(33, 19, 1, false);
  trsm_packet_case<Scalar, ColMajor, RowMajor, OnTheRight, UnitLower, 1>(33, 19, 1, false);
  trsm_packet_case<Scalar, RowMajor, RowMajor, OnTheRight, UnitUpper, 1>(33, 19, 1, false);
  for (int exponent : {-40, 40}) {
    const Scalar magnitude = numext::ldexp(Scalar(1), exponent);
    trsm_packet_case<Scalar, ColMajor, ColMajor, OnTheLeft, Lower, 1>(65, 19, 1, true, magnitude);
    trsm_packet_case<Scalar, RowMajor, ColMajor, OnTheLeft, Upper, 1>(65, 19, 1, true, magnitude);
  }
  trsm_packet_cache_sizes<Scalar>();
}

EIGEN_DECLARE_TEST(trsm_packet) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1(trsm_packet<float>());
    CALL_SUBTEST_2(trsm_packet<double>());
  }
}
