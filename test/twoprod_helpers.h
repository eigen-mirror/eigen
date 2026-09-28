// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_TWOPROD_HELPERS_H
#define EIGEN_TEST_TWOPROD_HELPERS_H

template <typename Packet>
void check_twoprod_packet() {
  using Scalar = typename internal::unpacket_traits<Packet>::type;
  constexpr Index size = internal::unpacket_traits<Packet>::size;
  EIGEN_ALIGN_MAX Scalar x[size], y[size], hi[size], lo[size], residual[size];
  const Scalar epsilon = NumTraits<Scalar>::epsilon();
  volatile int one = 1;
  for (int sign : {-1, 1}) {
    for (Index i = 0; i < size; ++i) {
      x[i] = Scalar(i % 2 ? -sign : sign) * (Scalar(one) + epsilon);
      y[i] = Scalar(one) - epsilon;
    }
    const Packet px = internal::pload<Packet>(x), py = internal::pload<Packet>(y);
    Packet p_hi, p_lo;
    internal::twoprod(px, py, p_hi, p_lo);
    internal::pstore(hi, p_hi);
    internal::pstore(lo, p_lo);
    internal::pstore(residual, internal::twoprod_low(px, py, p_hi));
    for (Index i = 0; i < size; ++i) {
      const Scalar lane_sign = Scalar(i % 2 ? -sign : sign);
      // (1 + eps)(1 - eps) = 1 - eps^2; an unfused pmsub loses the residual.
      VERIFY_IS_EQUAL(hi[i], lane_sign);
      VERIFY_IS_EQUAL(lo[i], -lane_sign * epsilon * epsilon);
      VERIFY_IS_EQUAL(residual[i], -lane_sign * epsilon * epsilon);
    }
  }
}

// Where a hardware fma exists, GCC's C++ default -ffp-contract=fast may fuse fl(x*y) into the sums that consume it, and
// Dekker's splitting product is then no longer error-free. The double-word product exposes it:
// (1 + eps)(1 - eps) = 1 - eps^2, whose fl(x*y) = 1 feeds the renormalizing sums.
template <typename T>
void check_twoprod_contraction() {
  const T epsilon = NumTraits<T>::epsilon();
  // Runtime operands: constant folding would round every step before contraction could apply.
  volatile T one = T(1);
  for (int sign : {-1, 1}) {
    const T x = T(sign) * (one + epsilon), y = one - epsilon;
    T hi, lo;
    internal::twoprod(x, T(0), y, hi, lo);
    VERIFY_IS_EQUAL(hi, T(sign));
    VERIFY_IS_EQUAL(lo, -T(sign) * epsilon * epsilon);
  }
}

#endif  // EIGEN_TEST_TWOPROD_HELPERS_H
