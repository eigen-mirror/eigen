// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Own small TU: GCC 13 -O3 -ffast-math zeroed the trig sign mask only where IPA-CP could specialize it (#3132).
#include "main.h"

template <typename Scalar, int Func, bool Enabled>
struct packet_trig_sign_check {
  static void run() {}
};

template <typename Scalar, int Func>
struct packet_trig_sign_check<Scalar, Func, true> {
  using Packet = typename internal::packet_traits<Scalar>::type;
  static Packet eval(const Packet& p, std::integral_constant<int, 0>) { return internal::psin(p); }
  static Packet eval(const Packet& p, std::integral_constant<int, 1>) { return internal::pcos(p); }
  static Packet eval(const Packet& p, std::integral_constant<int, 2>) { return internal::ptan(p); }

  static void run() {
    constexpr Index kSize = internal::unpacket_traits<Packet>::size;
    const Index n = numext::maxi<Index>(32, kSize);
    std::vector<Scalar> x(n), y(n);
    // +-(0.3 + 0.7 k) covers all four quadrants with sin, cos and tan bounded away from 0.
    for (Index i = 0; i < n; ++i) x[i] = Scalar((i / 16) % 2 ? 1 : -1) * (Scalar(0.3) + Scalar(0.7) * Scalar(i % 16));
    // The packet kernels are called directly: through the array API they inline and the defect does not appear.
    for (Index i = 0; i < n; i += kSize) {
      const Packet p = internal::ploadu<Packet>(&x[i]);
      internal::pstoreu(&y[i], eval(p, std::integral_constant<int, Func>()));
    }
    for (Index i = 0; i < n; ++i) {
      const Scalar ref = Func == 0 ? std::sin(x[i]) : Func == 1 ? std::cos(x[i]) : std::tan(x[i]);
      VERIFY_IS_APPROX(y[i], ref);
    }
  }
};

// pexp of a complex packet reaches psincos_*<SinCos>, which has its own sign logic.
template <typename Scalar, bool Enabled = internal::packet_traits<std::complex<Scalar>>::Vectorizable &&
                                          internal::packet_traits<std::complex<Scalar>>::HasExp>
struct packet_cexp_sign_check {
  static void run() {}
};

template <typename Scalar>
struct packet_cexp_sign_check<Scalar, true> {
  static void run() {
    using Complex = std::complex<Scalar>;
    using Packet = typename internal::packet_traits<Complex>::type;
    constexpr Index kSize = internal::unpacket_traits<Packet>::size;
    const Index n = numext::maxi<Index>(32, kSize);
    std::vector<Complex> x(n), y(n);
    for (Index i = 0; i < n; ++i)
      x[i] = Complex(Scalar(0.25), Scalar((i / 16) % 2 ? 1 : -1) * (Scalar(0.3) + Scalar(0.7) * Scalar(i % 16)));
    for (Index i = 0; i < n; i += kSize) {
      internal::pstoreu(&y[i], internal::pexp(internal::ploadu<Packet>(&x[i])));
    }
    for (Index i = 0; i < n; ++i) VERIFY_IS_APPROX(y[i], std::exp(x[i]));
  }
};

template <typename Scalar>
void check_packet_trig_signs() {
  using Traits = internal::packet_traits<Scalar>;
  packet_trig_sign_check<Scalar, 0, Traits::Vectorizable && Traits::HasSin>::run();
  packet_trig_sign_check<Scalar, 1, Traits::Vectorizable && Traits::HasCos>::run();
  packet_trig_sign_check<Scalar, 2, Traits::Vectorizable && Traits::HasTan>::run();
  packet_cexp_sign_check<Scalar>::run();
}

EIGEN_DECLARE_TEST(trig_sign_fastmath) {
  CALL_SUBTEST(check_packet_trig_signs<float>());
  CALL_SUBTEST(check_packet_trig_signs<double>());
}
