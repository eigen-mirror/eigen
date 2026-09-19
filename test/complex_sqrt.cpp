// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "CustomComplex.h"
#include "fp_control.h"

template <typename T>
void check_root_component(T actual, T expected) {
  if ((numext::isnan)(expected)) {
    VERIFY((numext::isnan)(actual));
  } else if ((numext::isinf)(expected) || numext::is_exactly_zero(expected)) {
    VERIFY_IS_EQUAL(actual, expected);
    VERIFY_IS_EQUAL(std::signbit(actual), std::signbit(expected));
  } else {
    VERIFY((numext::isfinite)(actual));
    // Two square roots, divisions and scaling: allow a few rounding errors per component.
    const T bound =
        T(4) * NumTraits<T>::epsilon() * numext::abs(expected) + T(2) * std::numeric_limits<T>::denorm_min();
    VERIFY(numext::abs(actual - expected) <= bound);
  }
}

template <typename T>
void check_complex_roots(const std::complex<T>& input, const std::complex<T>& root,
                         const std::complex<T>& reciprocal_root) {
  const auto actual = numext::sqrt(input);
  const auto reciprocal_actual = numext::rsqrt(input);
  check_root_component(actual.real(), root.real());
  check_root_component(actual.imag(), root.imag());
  check_root_component(reciprocal_actual.real(), reciprocal_root.real());
  check_root_component(reciprocal_actual.imag(), reciprocal_root.imag());

  const auto custom = numext::sqrt(CustomComplex<T>(input.real(), input.imag()));
  check_root_component(numext::real(custom), root.real());
  check_root_component(numext::imag(custom), root.imag());
}

template <typename T>
void complex_roots_special_values() {
  using C = std::complex<T>;
  const T zero = T(0);
  const T inf = NumTraits<T>::infinity();
  const T nan = NumTraits<T>::quiet_NaN();
  for (T sign : {T(-1), T(1)}) {
    const T y = sign * zero;
    check_complex_roots(C(zero, y), C(zero, y), C(inf, nan));
    check_complex_roots(C(-zero, y), C(zero, y), C(inf, nan));
    check_complex_roots(C(T(4), y), C(T(2), y), C(T(0.5), -y));
    check_complex_roots(C(T(-4), y), C(zero, sign * T(2)), C(zero, -sign * T(0.5)));
    for (T x : {zero, T(1), T(-1), inf, -inf, nan}) {
      check_complex_roots(C(x, sign * inf), C(inf, sign * inf), C(zero, -y));
    }
    for (T magnitude : {zero, T(1)}) {
      check_complex_roots(C(inf, sign * magnitude), C(inf, y), C(zero, -y));
      check_complex_roots(C(-inf, sign * magnitude), C(zero, sign * inf), C(zero, -y));
    }
  }
  for (T value : {zero, T(1), T(-1), nan}) {
    check_complex_roots(C(value, nan), C(nan, nan), C(nan, nan));
    check_complex_roots(C(nan, value), C(nan, nan), C(nan, nan));
  }
  const auto positive_inf_nan = numext::sqrt(C(inf, nan));
  const auto negative_inf_nan = numext::sqrt(C(-inf, nan));
  check_root_component(positive_inf_nan.real(), inf);
  VERIFY((numext::isnan)(positive_inf_nan.imag()));
  VERIFY((numext::isnan)(negative_inf_nan.real()));
  VERIFY((numext::isinf)(negative_inf_nan.imag()));
}

template <typename T>
void complex_roots_finite() {
  using C = std::complex<T>;
  const bool flushes_inputs = ScopedFlushToZero::hardwareFlushesSubnormalInputs();
  const int minimum_exponent = std::numeric_limits<T>::min_exponent - std::numeric_limits<T>::digits;
  // sqrt((3 + 4i) * 2^(2k)) = (2 + i) * 2^k, exactly.
  for (int exponent = minimum_exponent; exponent <= std::numeric_limits<T>::max_exponent - 3; ++exponent) {
    if (exponent % 2 != 0 || (flushes_inputs && exponent < std::numeric_limits<T>::min_exponent - 1)) continue;
    const T scale = std::ldexp(T(1), exponent);
    const T root_scale = std::ldexp(T(1), exponent / 2);
    for (T sign_x : {T(-1), T(1)}) {
      for (T sign_y : {T(-1), T(1)}) {
        const T u = sign_x > 0 ? T(2) : T(1);
        const T v = sign_x > 0 ? T(1) : T(2);
        check_complex_roots(C(sign_x * T(3) * scale, sign_y * T(4) * scale), C(u * root_scale, sign_y * v * root_scale),
                            C((u / T(5)) / root_scale, (-sign_y * v / T(5)) / root_scale));
      }
    }
  }

  const T minimum = (std::numeric_limits<T>::min)();
  const T maximum = NumTraits<T>::highest();
  const T lower = T(2) * minimum;
  const T upper = maximum / T(4);
  const T values[] = {std::numeric_limits<T>::denorm_min(),
                      minimum,
                      std::nextafter(lower, T(0)),
                      lower,
                      std::nextafter(lower, maximum),
                      std::nextafter(upper, T(0)),
                      upper,
                      std::nextafter(upper, maximum),
                      maximum};
  // MPFR, 256 bits: components of sqrt(1+i) and 1/sqrt(1+i).
  const T a = T(1.0986841134678099660398011952406783785L);
  const T b = T(0.4550898605622273413043577578224685696L);
  const T c = T(0.7768869870150186536720794765315734741L);
  const T d = T(0.3217971264527913123677217187091049045L);
  for (T p : values) {
    if (p == T(0) || (flushes_inputs && p < minimum)) continue;
    const T root_p = numext::sqrt(p);
    for (T sign : {T(-1), T(1)}) {
      check_complex_roots(C(p, sign * p), C(a * root_p, sign * b * root_p), C(c / root_p, -sign * d / root_p));
      check_complex_roots(C(-p, sign * p), C(b * root_p, sign * a * root_p), C(d / root_p, -sign * c / root_p));
      const T half_root = root_p * T(0.7071067811865475244008443621048490393L);
      const T inverse_half_root = T(0.7071067811865475244008443621048490393L) / root_p;
      check_complex_roots(C(T(0), sign * p), C(half_root, sign * half_root),
                          C(inverse_half_root, -sign * inverse_half_root));
    }
  }

  const T large = std::ldexp(T(1), std::numeric_limits<T>::max_exponent - 2);
  const T small = std::ldexp(T(1), -std::numeric_limits<T>::max_exponent / 3);
  const T root_large = numext::sqrt(large);
  // small/large underflows, but the smaller component of sqrt(large + i*small) does not.
  check_complex_roots(C(large, small), C(root_large, small / (T(2) * root_large)), C(T(1) / root_large, -T(0)));
  check_complex_roots(C(-large, -small), C(small / (T(2) * root_large), -root_large), C(T(0), T(1) / root_large));

  // A strided input selects scalar evaluation even on backends with complex packets.
  Array<C, 34, 1> storage;
  Map<Array<C, 17, 1>, Unaligned, InnerStride<2>> input(storage.data());
  for (Index i = 0; i < input.size(); ++i) {
    const int exponent = i % 2 ? std::numeric_limits<T>::max_exponent - 4 : std::numeric_limits<T>::min_exponent;
    const T scale = std::ldexp(T(1), exponent - exponent % 2);
    input(i) = C(T(3) * scale, T(4) * scale);
  }
  const Array<C, 17, 1> roots = input.sqrt();
  const Array<C, 17, 1> reciprocal_roots = input.rsqrt();
  for (Index i = 0; i < input.size(); ++i) {
    const T scale = numext::sqrt(input(i).imag() / T(4));
    check_root_component(roots(i).real(), T(2) * scale);
    check_root_component(roots(i).imag(), scale);
    check_root_component(reciprocal_roots(i).real(), (T(2) / T(5)) / scale);
    check_root_component(reciprocal_roots(i).imag(), (T(-1) / T(5)) / scale);
  }
}

EIGEN_DECLARE_TEST(complex_sqrt) {
  CALL_SUBTEST(complex_roots_special_values<float>());
  CALL_SUBTEST(complex_roots_special_values<double>());
  CALL_SUBTEST(complex_roots_special_values<long double>());
  CALL_SUBTEST(complex_roots_finite<float>());
  CALL_SUBTEST(complex_roots_finite<double>());
  CALL_SUBTEST(complex_roots_finite<long double>());
}
