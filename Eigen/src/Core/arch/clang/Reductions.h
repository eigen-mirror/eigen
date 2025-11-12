// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Rasmus Munk Larsen
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_REDUCTIONS_CLANG_H
#define EIGEN_REDUCTIONS_CLANG_H

namespace Eigen {
namespace internal {

// --- Reductions ---
#if EIGEN_HAS_BUILTIN(__builtin_reduce_min) && EIGEN_HAS_BUILTIN(__builtin_reduce_max) && \
    EIGEN_HAS_BUILTIN(__builtin_reduce_or)
#define EIGEN_CLANG_PACKET_REDUX_MINMAX(PACKET_TYPE)                                        \
  template <>                                                                               \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type predux_min(const PACKET_TYPE& a) { \
    return __builtin_reduce_min(a);                                                         \
  }                                                                                         \
  template <>                                                                               \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type predux_max(const PACKET_TYPE& a) { \
    return __builtin_reduce_max(a);                                                         \
  }                                                                                         \
  template <>                                                                               \
  EIGEN_STRONG_INLINE bool predux_any(const PACKET_TYPE& a) {                               \
    return __builtin_reduce_or(a != 0) != 0;                                                \
  }

EIGEN_CLANG_PACKET_REDUX_MINMAX(Packet16f)
EIGEN_CLANG_PACKET_REDUX_MINMAX(Packet8d)
EIGEN_CLANG_PACKET_REDUX_MINMAX(Packet16i)
EIGEN_CLANG_PACKET_REDUX_MINMAX(Packet8l)
#undef EIGEN_CLANG_PACKET_REDUX_MINMAX
#endif

#if EIGEN_HAS_BUILTIN(__builtin_reduce_add) && EIGEN_HAS_BUILTIN(__builtin_reduce_mul)
#define EIGEN_CLANG_PACKET_REDUX_INT(PACKET_TYPE)                                                        \
  template <>                                                                                            \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type predux<PACKET_TYPE>(const PACKET_TYPE& a) {     \
    return __builtin_reduce_add(a);                                                                      \
  }                                                                                                      \
  template <>                                                                                            \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type predux_mul<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return __builtin_reduce_mul(a);                                                                      \
  }

// __builtin_reduce_{mul,add} are only defined for integer types.
EIGEN_CLANG_PACKET_REDUX_INT(Packet16i)
EIGEN_CLANG_PACKET_REDUX_INT(Packet8l)
#undef EIGEN_CLANG_PACKET_REDUX_INT
#endif

#if EIGEN_HAS_BUILTIN(__builtin_shufflevector)
namespace detail {
template <typename VectorT>
EIGEN_STRONG_INLINE scalar_type_of_vector_t<VectorT> ReduceAdd16(const VectorT& a) {
  auto t1 = __builtin_shufflevector(a, a, 0, 2, 4, 6, 8, 10, 12, 14) +
            __builtin_shufflevector(a, a, 1, 3, 5, 7, 9, 11, 13, 15);
  auto t2 = __builtin_shufflevector(t1, t1, 0, 2, 4, 6) + __builtin_shufflevector(t1, t1, 1, 3, 5, 7);
  auto t3 = __builtin_shufflevector(t2, t2, 0, 2) + __builtin_shufflevector(t2, t2, 1, 3);
  return t3[0] + t3[1];
}

template <typename VectorT>
EIGEN_STRONG_INLINE scalar_type_of_vector_t<VectorT> ReduceAdd8(const VectorT& a) {
  auto t1 = __builtin_shufflevector(a, a, 0, 2, 4, 6) + __builtin_shufflevector(a, a, 1, 3, 5, 7);
  auto t2 = __builtin_shufflevector(t1, t1, 0, 2) + __builtin_shufflevector(t1, t1, 1, 3);
  return t2[0] + t2[1];
}

template <typename VectorT>
EIGEN_STRONG_INLINE scalar_type_of_vector_t<VectorT> ReduceMul16(const VectorT& a) {
  auto t1 = __builtin_shufflevector(a, a, 0, 2, 4, 6, 8, 10, 12, 14) *
            __builtin_shufflevector(a, a, 1, 3, 5, 7, 9, 11, 13, 15);
  auto t2 = __builtin_shufflevector(t1, t1, 0, 2, 4, 6) * __builtin_shufflevector(t1, t1, 1, 3, 5, 7);
  auto t3 = __builtin_shufflevector(t2, t2, 0, 2) * __builtin_shufflevector(t2, t2, 1, 3);
  return t3[0] * t3[1];
}

template <typename VectorT>
EIGEN_STRONG_INLINE scalar_type_of_vector_t<VectorT> ReduceMul8(const VectorT& a) {
  auto t1 = __builtin_shufflevector(a, a, 0, 2, 4, 6) * __builtin_shufflevector(a, a, 1, 3, 5, 7);
  auto t2 = __builtin_shufflevector(t1, t1, 0, 2) * __builtin_shufflevector(t1, t1, 1, 3);
  return t2[0] * t2[1];
}
}  // namespace detail

template <>
EIGEN_STRONG_INLINE float predux<Packet16f>(const Packet16f& a) {
  return detail::ReduceAdd16(a);
}
template <>
EIGEN_STRONG_INLINE double predux<Packet8d>(const Packet8d& a) {
  return detail::ReduceAdd8(a);
}
template <>
EIGEN_STRONG_INLINE float predux_mul<Packet16f>(const Packet16f& a) {
  return detail::ReduceMul16(a);
}
template <>
EIGEN_STRONG_INLINE double predux_mul<Packet8d>(const Packet8d& a) {
  return detail::ReduceMul8(a);
}
#endif

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_REDUCTIONS_CLANG_H
