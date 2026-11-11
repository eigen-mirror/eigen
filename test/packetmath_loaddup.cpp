// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "guard_page_buffer.h"

#if defined(__GNUC__) && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

template <typename Scalar, typename Packet = typename internal::packet_traits<Scalar>::type,
          int Size = internal::unpacket_traits<Packet>::size>
void loaddup_boundary() {
  constexpr int Count = (Size + 1) / 2;
  const test::GuardPageBuffer buffer(Count * sizeof(Scalar));
  Scalar* const data = buffer.place_at_end<Scalar>(Count);
  for (Index i = 0; i < Count; ++i) data[i] = Scalar(i + 1);
  Scalar actual[Size];
  internal::pstoreu(actual, internal::ploaddup<Packet>(data));
  for (Index i = 0; i < Size; ++i) {
    const Index source = i / 2;
    VERIFY_IS_EQUAL(actual[i], Scalar(source + 1));
  }
}

template <typename Scalar>
void loaddup_packet_boundaries() {
  using Packet = typename internal::packet_traits<Scalar>::type;
  using Half = typename internal::unpacket_traits<Packet>::half;
  using Quarter = typename internal::unpacket_traits<Half>::half;
  loaddup_boundary<Scalar, Packet>();
  EIGEN_IF_CONSTEXPR ((!internal::is_same<Packet, Half>::value)) {
    loaddup_boundary<Scalar, Half>();
  }
  EIGEN_IF_CONSTEXPR ((!internal::is_same<Half, Quarter>::value)) {
    loaddup_boundary<Scalar, Quarter>();
  }
}

template <typename Real, int Rows, int Cols, int Order>
void mixed_product_loaddup_boundary() {
  using Complex = std::complex<Real>;
  using MatrixType = Matrix<Real, Rows, Cols, Order>;
  const test::GuardPageBuffer buffer(Rows * Cols * sizeof(Real));
  Map<MatrixType, Unaligned> lhs(buffer.place_at_end<Real>(Rows * Cols));
  for (Index j = 0; j < Cols; ++j)
    for (Index i = 0; i < Rows; ++i) lhs(i, j) = Real(i + j + 1);
  Matrix<Complex, Cols, 1> rhs;
  for (Index j = 0; j < Cols; ++j) rhs(j) = Complex(Real(j + 1), Real(1 - j));
  const Matrix<Complex, Rows, 1> actual = lhs * rhs;
  const Matrix<Complex, Rows, 1> lazy = lhs.lazyProduct(rhs);
  for (Index i = 0; i < Rows; ++i) {
    Complex expected(0);
    for (Index j = 0; j < Cols; ++j) expected += lhs(i, j) * rhs(j);
    VERIFY_IS_APPROX(actual(i), expected);
    VERIFY_IS_APPROX(lazy(i), expected);
  }
}

template <typename Real>
void mixed_product_loaddup_boundaries() {
  mixed_product_loaddup_boundary<Real, 2, 2, ColMajor>();
  mixed_product_loaddup_boundary<Real, 2, 2, RowMajor>();
  mixed_product_loaddup_boundary<Real, 8, 17, ColMajor>();
  mixed_product_loaddup_boundary<Real, 8, 17, RowMajor>();
}

EIGEN_DECLARE_TEST(packetmath_loaddup) {
  CALL_SUBTEST_1(loaddup_packet_boundaries<float>());
  CALL_SUBTEST_2(loaddup_packet_boundaries<double>());
  CALL_SUBTEST_3(loaddup_packet_boundaries<int>());
  CALL_SUBTEST_1(mixed_product_loaddup_boundaries<float>());
  CALL_SUBTEST_2(mixed_product_loaddup_boundaries<double>());
#if defined(EIGEN_VECTORIZE_LSX) || defined(EIGEN_VECTORIZE_ALTIVEC)
  CALL_SUBTEST_3(loaddup_boundary<numext::int8_t>());
  CALL_SUBTEST_3(loaddup_boundary<numext::uint8_t>());
  CALL_SUBTEST_3(loaddup_boundary<numext::int16_t>());
  CALL_SUBTEST_3(loaddup_boundary<numext::uint16_t>());
#if defined(EIGEN_VECTORIZE_LSX)
  CALL_SUBTEST_3(loaddup_boundary<numext::uint32_t>());
#endif
#if defined(EIGEN_VECTORIZE_ALTIVEC)
  CALL_SUBTEST_3(loaddup_boundary<bfloat16>());
#endif
#endif
#if defined(EIGEN_VECTORIZE_RVV10)
  // Explicit LMULs cover half packets even when the default packet uses a larger register group.
#define EIGEN_TEST_RVV_DUP(Scalar, Packet) \
  loaddup_boundary<Scalar, internal::Packet, internal::unpacket_traits<internal::Packet>::size>()
  CALL_SUBTEST_1((EIGEN_TEST_RVV_DUP(float, Packet1Xf)));
  CALL_SUBTEST_1((EIGEN_TEST_RVV_DUP(float, Packet2Xf)));
  CALL_SUBTEST_1((EIGEN_TEST_RVV_DUP(float, Packet4Xf)));
  CALL_SUBTEST_2((EIGEN_TEST_RVV_DUP(double, Packet1Xd)));
  CALL_SUBTEST_2((EIGEN_TEST_RVV_DUP(double, Packet2Xd)));
  CALL_SUBTEST_2((EIGEN_TEST_RVV_DUP(double, Packet4Xd)));
  CALL_SUBTEST_3((EIGEN_TEST_RVV_DUP(numext::int32_t, Packet1Xi)));
  CALL_SUBTEST_3((EIGEN_TEST_RVV_DUP(numext::int32_t, Packet2Xi)));
  CALL_SUBTEST_3((EIGEN_TEST_RVV_DUP(numext::int32_t, Packet4Xi)));
  CALL_SUBTEST_3((EIGEN_TEST_RVV_DUP(numext::int64_t, Packet1Xl)));
  CALL_SUBTEST_3((EIGEN_TEST_RVV_DUP(numext::int64_t, Packet2Xl)));
  CALL_SUBTEST_3((EIGEN_TEST_RVV_DUP(numext::int64_t, Packet4Xl)));
  CALL_SUBTEST_3((EIGEN_TEST_RVV_DUP(numext::int16_t, Packet1Xs)));
  CALL_SUBTEST_3((EIGEN_TEST_RVV_DUP(numext::int16_t, Packet2Xs)));
#if defined(EIGEN_VECTORIZE_RVV10FP16)
  CALL_SUBTEST_3((EIGEN_TEST_RVV_DUP(half, Packet1Xh)));
  CALL_SUBTEST_3((EIGEN_TEST_RVV_DUP(half, Packet2Xh)));
#endif
#if defined(EIGEN_VECTORIZE_RVV10BF16)
  CALL_SUBTEST_3((EIGEN_TEST_RVV_DUP(bfloat16, Packet1Xbf)));
  CALL_SUBTEST_3((EIGEN_TEST_RVV_DUP(bfloat16, Packet2Xbf)));
#endif
#undef EIGEN_TEST_RVV_DUP
#endif
}
