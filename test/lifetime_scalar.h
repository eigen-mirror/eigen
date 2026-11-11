// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_LIFETIME_SCALAR_H
#define EIGEN_TEST_LIFETIME_SCALAR_H

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>

#include <Eigen/Core>

namespace lifetime_scalar {

using Index = Eigen::Index;

struct Failure : std::exception {};

// Counts its lifetime events in static counters: live objects, and copy and move constructions and assignments.
// Constructing from an int (or by default) throws Failure if live >= max_live, copy or move construction if
// live >= max_live_on_copy, so at most max_live objects result from the former. Throwing needs EIGEN_EXCEPTIONS.
// Construction checks that the object is aligned to Alignment.
template <std::size_t Alignment = alignof(int)>
struct alignas(Alignment) LifetimeScalar {
  static constexpr Index kUnlimited = (std::numeric_limits<Index>::max)();
  static Index live;
  static Index copies;
  static Index moves;
  static Index max_live;
  static Index max_live_on_copy;

  int value;

  explicit LifetimeScalar(int v = 0) : value(v) { construct(max_live); }
  LifetimeScalar(const LifetimeScalar& other) : value(other.value) {
    construct(max_live_on_copy);
    ++copies;
  }
  LifetimeScalar(LifetimeScalar&& other) : value(other.value) {
    construct(max_live_on_copy);
    ++moves;
  }
  LifetimeScalar& operator=(const LifetimeScalar& other) {
    value = other.value;
    ++copies;
    return *this;
  }
  LifetimeScalar& operator=(LifetimeScalar&& other) {
    value = other.value;
    ++moves;
    return *this;
  }
  ~LifetimeScalar() { --live; }

  // Zeroes the copy and move counts and removes both bounds; live is left alone.
  static void reset() {
    copies = moves = 0;
    max_live = max_live_on_copy = kUnlimited;
  }

 private:
  void construct(Index bound) {
    eigen_assert(std::uintptr_t(this) % Alignment == 0);
#ifdef EIGEN_EXCEPTIONS
    if (live >= bound) throw Failure();
#else
    EIGEN_UNUSED_VARIABLE(bound);
#endif
    ++live;
  }
};

template <std::size_t Alignment>
constexpr Index LifetimeScalar<Alignment>::kUnlimited;
template <std::size_t Alignment>
Index LifetimeScalar<Alignment>::live = 0;
template <std::size_t Alignment>
Index LifetimeScalar<Alignment>::copies = 0;
template <std::size_t Alignment>
Index LifetimeScalar<Alignment>::moves = 0;
template <std::size_t Alignment>
Index LifetimeScalar<Alignment>::max_live = LifetimeScalar<Alignment>::kUnlimited;
template <std::size_t Alignment>
Index LifetimeScalar<Alignment>::max_live_on_copy = LifetimeScalar<Alignment>::kUnlimited;

}  // namespace lifetime_scalar

#endif  // EIGEN_TEST_LIFETIME_SCALAR_H
