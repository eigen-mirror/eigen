// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPECIAL_FUNCTIONS_OVERLOADS_H
#define EIGEN_SPECIAL_FUNCTIONS_OVERLOADS_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

// Unqualified lookup of a special function from this namespace sees std::erf, std::erfc, std::lgamma and whatever
// argument-dependent lookup adds. The using-declaration and the deleted declarations hide the ArrayBase overloads in
// namespace Eigen and keep ordinary lookup from reaching anything else. Both the detection and the call below are made
// from here, so they see the same overload set.
namespace special_function_overloads {

#define EIGEN_SPECIAL_FUNCTION_OVERLOAD(NAME)                                                                  \
  struct NAME##_fn {                                                                                           \
    template <typename Scalar, typename... Rest,                                                               \
              typename Result = decltype(NAME(std::declval<const Scalar&>(), std::declval<const Rest&>()...))> \
    static std::is_same<std::decay_t<Result>, Scalar> test(int);                                               \
    template <typename... Args>                                                                                \
    static std::false_type test(...);                                                                          \
    template <typename Scalar, typename... Args>                                                               \
    EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Scalar call(const Args&... args) {                            \
      return NAME(args...);                                                                                    \
    }                                                                                                          \
  };
// A deleted template rather than a deleted function: GCC 10 drops the overloads that argument-dependent lookup finds
// when ordinary lookup finds a deleted non-template function.
#define EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(NAME) \
  template <typename = void>                           \
  void NAME() = delete;                                \
  EIGEN_SPECIAL_FUNCTION_OVERLOAD(NAME)

EIGEN_USING_STD(erf)
EIGEN_SPECIAL_FUNCTION_OVERLOAD(erf)
EIGEN_USING_STD(erfc)
EIGEN_SPECIAL_FUNCTION_OVERLOAD(erfc)
EIGEN_USING_STD(lgamma)
EIGEN_SPECIAL_FUNCTION_OVERLOAD(lgamma)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(digamma)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(zeta)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(polygamma)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(ndtri)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(igamma)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(igammac)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(igamma_der_a)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(gamma_sample_der_alpha)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(betainc)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(bessel_i0)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(bessel_i0e)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(bessel_i1)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(bessel_i1e)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(bessel_j0)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(bessel_j1)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(bessel_y0)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(bessel_y1)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(bessel_k0)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(bessel_k0e)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(bessel_k1)
EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY(bessel_k1e)

#undef EIGEN_SPECIAL_FUNCTION_OVERLOAD_ADL_ONLY
#undef EIGEN_SPECIAL_FUNCTION_OVERLOAD

}  // namespace special_function_overloads

template <typename Fn, bool IsScalar, typename Scalar, typename... Rest>
struct has_special_function_overload_impl : std::false_type {};
template <typename Fn, typename Scalar, typename... Rest>
struct has_special_function_overload_impl<Fn, true, Scalar, Rest...>
    : bool_constant<!NumTraits<Scalar>::IsInteger && decltype(Fn::template test<Scalar, Rest...>(0))::value> {};

// Whether Fn's lookup finds an overload callable with Scalar, Rest... for the non-integer scalar type Scalar that
// returns Scalar. Another result type means that the overload takes another type, such as the double of std::lgamma,
// which the argument reaches through a conversion, and computes at that type's precision. Packets never use one.
template <typename Fn, typename Scalar, typename... Rest>
struct has_special_function_overload
    : has_special_function_overload_impl<Fn, std::is_same<Scalar, typename unpacket_traits<Scalar>::type>::value,
                                         Scalar, Rest...> {};

template <typename Fn, typename Scalar>
struct special_function_overload {
  template <typename... Args>
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Scalar run(const Args&... args) {
    return Fn::template call<Scalar>(args...);
  }
};

// The implementation of a special function for arguments of types Scalar, Rest...: the overload that Fn's lookup finds
// when there is one, so that a custom scalar type keeps its own implementation, and Default otherwise.
template <typename Fn, typename Default, typename Scalar, typename... Rest>
struct special_function_overload_or : std::conditional_t<has_special_function_overload<Fn, Scalar, Rest...>::value,
                                                         special_function_overload<Fn, Scalar>, Default> {};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_SPECIAL_FUNCTIONS_OVERLOADS_H
