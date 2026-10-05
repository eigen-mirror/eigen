// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_NUMERICALDIFF_STENCIL_H
#define EIGEN_NUMERICALDIFF_STENCIL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \class Stencil
 * \ingroup NumericalDiff_Module
 *
 * \brief Finite-difference weights on an arbitrarily spaced grid.
 *
 * Computes the weights of the order-`Derivative` finite-difference formula for the grid `points`,
 * using Fornberg's recursive algorithm:
 *
 * B. Fornberg, "Generation of Finite Difference Formulas on Arbitrarily Spaced Grids",
 * Math. Comp. 51 (1988), 699-706.
 *
 * The implementation follows the equivalent, in-place algorithmic form given in the follow-up:
 *
 * B. Fornberg, "Calculation of Weights in Finite Difference Formulas", SIAM Review 40 (1998), 685-691.
 *
 * The evaluation point is fixed at 0: `points` are read as offsets from wherever the derivative is
 * to be evaluated. This is lossless, since shifting every point (and the evaluation point) by the
 * same amount leaves the weights unchanged, so a caller evaluating at `x` shifts `points` by `-x`
 * first. `Derivative == 0` is supported and corresponds to interpolation: the weights reproduce
 * `f(0)` exactly for every polynomial of degree less than `Size`.
 *
 * `points` are used in exactly the order given: this class does not sort or otherwise reorder them.
 * Fornberg's paper notes that the order nodes are introduced in affects the conditioning of the
 * recursion (not the exact result, by uniqueness of the interpolating polynomial), and suggests
 * introducing nodes in order of ascending distance from the evaluation point for stability. Choosing
 * and applying such an ordering, including e.g. a Leja ordering for complex nodes, is the caller's
 * responsibility.
 *
 * For offsets `h * k` at a fixed relative layout `k` and varying step `h`, the weights scale as
 * `w_i(h * k) == h^(-Derivative) * w_i(k)`: computing weights once on `k` and scaling by `h^(-Derivative)`
 * avoids repeating the `O(Size^2 * Derivative)` recursion for every step. `Stencil` only produces the
 * weights for the points it is given; choosing or adapting `h` itself is the caller's responsibility.
 *
 * \tparam Derivative The order of the derivative the weights approximate.
 * \tparam Scalar The scalar type of the grid points and weights. Must be a non-integer type.
 * \tparam Size The number of grid points.
 */
template <int Derivative, typename Scalar, int Size>
class Stencil {
  EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar)
  static_assert(Size >= 0, "Stencil requires a fixed-size point set");
  static_assert(Derivative >= 0 && Derivative < Size, "Stencil requires at least `Derivative + 1` points");

 public:
  static constexpr int DerivativeOrder = Derivative;
  static constexpr int PointCount = Size;

  /** The type returned by weights() and points(): a plain, vectorizable array, copied out of the object. */
  using ArrayType = Array<Scalar, Size, 1>;

  /** Computes the weights of the order-\a Derivative finite-difference formula for \a points, given as
   * any fixed-size, 1D dense expression of \a Size coefficients convertible to \a Scalar. */
  template <typename Derived>
  explicit constexpr Stencil(const DenseBase<Derived>& points) {
    static_assert(Derived::IsVectorAtCompileTime, "Stencil requires a 1D dense expression");
    static_assert(Derived::SizeAtCompileTime == Size, "Stencil requires exactly `Size` points");
    for (Index i = 0; i < Size; ++i) m_points[i] = points.coeff(i);
    computeWeights();
  }

  /** Constructs from a plain C array of \a Size points. Unlike the DenseBase overload above, this
   * constructor does not route through Eigen::Array, so it and weight()/point() remain usable in an
   * actual constant expression on compilers where Array's own fixed-size constructors are not. */
  explicit constexpr Stencil(const Scalar (&points)[Size]) {
    for (Index i = 0; i < Size; ++i) m_points[i] = points[i];
    computeWeights();
  }

  /** \returns the weights, in the same order as points(). */
  ArrayType weights() const { return Map<const ArrayType>(m_weights); }

  /** \returns the grid points, in the order originally supplied. */
  ArrayType points() const { return Map<const ArrayType>(m_points); }

  /** \returns the weight of points()[i]. Unlike weights()[i], usable in a constant expression: it
   * indexes the underlying storage directly rather than through Map, whose constructor is not
   * constexpr-evaluable on every supported compiler. */
  constexpr const Scalar& weight(Index i) const { return m_weights[i]; }

  /** \returns points()[i]. Unlike points()[i] (through the Map returned by points()), usable in a
   * constant expression, for the same reason as weight(). */
  constexpr const Scalar& point(Index i) const { return m_points[i]; }

 private:
  // Fornberg's recursion (1988), in the in-place algorithmic form given by Fornberg (1998).
  // `delta[nu][m]` is the weight of `points[nu]` in the order-`m` formula built from `points[0..n]`
  // so far; each column `nu` only ever reads/writes its own entries as `n` grows, so a single working
  // table can be updated in place.
  constexpr void computeWeights() {
    Scalar delta[Size][Derivative + 1]{};
    delta[0][0] = Scalar(1);

    for (Index n = 1; n < Size; ++n) {
      const Index mn = numext::mini(n, Index(Derivative));
      const Scalar c4 = m_points[n];
      // c1/c2 = product_{j<n-1} ((x_{n-1}-x_j)/(x_n-x_j)) / (x_n-x_{n-1}).
      // Form the ratio directly to avoid overflow/underflow of the individual products.
      Scalar ratio(1);
      for (Index nu = 0; nu < n; ++nu) {
        const Scalar c3 = m_points[n] - m_points[nu];
        eigen_assert(!(c3 == Scalar(0)) && "Stencil points must be pairwise distinct");
        if (nu == n - 1) {
          ratio /= c3;
          delta[n][0] = -ratio * (m_points[n - 1] * delta[n - 1][0]);
          for (Index m = mn; m >= 1; --m)
            delta[n][m] = ratio * (Scalar(m) * delta[n - 1][m - 1] - m_points[n - 1] * delta[n - 1][m]);
        } else {
          ratio *= (m_points[n - 1] - m_points[nu]) / c3;
        }
        for (Index m = mn; m >= 1; --m) delta[nu][m] = (c4 * delta[nu][m] - Scalar(m) * delta[nu][m - 1]) / c3;
        delta[nu][0] = c4 * delta[nu][0] / c3;
      }
    }

    for (Index nu = 0; nu < Size; ++nu) m_weights[nu] = delta[nu][Derivative];
  }

  // NOTE: C arrays instead of `std::array` (through `Eigen::array`) is intentional since
  // non-`const` `operator[]` is only `constexpr` in C++17, not C++14.
  Scalar m_points[Size]{};
  Scalar m_weights[Size]{};
};

// Redundant out-of-class definitions are required pre-C++17 but deprecated since.
#if EIGEN_COMP_CXXVER < 17
template <int Derivative, typename Scalar, int Size>
constexpr int Stencil<Derivative, Scalar, Size>::DerivativeOrder;
template <int Derivative, typename Scalar, int Size>
constexpr int Stencil<Derivative, Scalar, Size>::PointCount;
#endif

/** \relates Stencil
 * Deduces \a Scalar and \a Size from \a points; \a Derivative must still be supplied explicitly,
 * e.g. `makeStencil<2>(points)`. Exists because C++14 has no class template argument deduction.
 */
template <int Derivative, typename Derived>
constexpr Stencil<Derivative, typename Derived::Scalar, Derived::SizeAtCompileTime> makeStencil(
    const DenseBase<Derived>& points) {
  return Stencil<Derivative, typename Derived::Scalar, Derived::SizeAtCompileTime>(points);
}

/** \relates Stencil
 * As above, deducing \a Scalar and \a Size from a plain C array of points.
 */
template <int Derivative, typename Scalar, int Size>
constexpr Stencil<Derivative, Scalar, Size> makeStencil(const Scalar (&points)[Size]) {
  return Stencil<Derivative, Scalar, Size>(points);
}

}  // namespace Eigen

#endif
