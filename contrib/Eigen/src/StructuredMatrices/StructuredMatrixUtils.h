// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// References:
//  [1] N. J. Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed.,
//      SIAM, 2002, chapter 27. Avoiding spurious overflow by rescaling with
//      powers of two, the technique behind structured_exponent_bound() and the
//      balanced determinant accumulations.
//  [2] P. H. Sterbenz, "Floating-Point Computation", Prentice-Hall, 1974.
//      Scaling by a power of two is exact, so the balanced accumulations
//      introduce no roundoff of their own.

#ifndef EIGEN_STRUCTURED_MATRIX_UTILS_H
#define EIGEN_STRUCTURED_MATRIX_UTILS_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// A distinct shape routes assignment through evalTo/addTo/subTo and lets one
// product specialization cover every product tag without colliding with DenseShape.
struct StructuredShape {};

// Below this dimension the FFT setup costs more than a plain O(n^2) evaluation,
// so the structured operators fall back to a direct segment-based product.
constexpr Index structured_direct_threshold() { return 32; }

// Below this dimension even the segment-based direct product loses to a plain
// scalar loop: the per-segment setup dominates when the average segment holds
// fewer than a couple of packets (measured crossover on AVX2 hardware).
constexpr Index structured_scalar_threshold() { return 16; }

// Numerical scale exponents are independent of Eigen's configurable dimension
// index. A 32-bit Index can overflow while accumulating O(n^2) factor
// exponents for dimensions that are otherwise practical.
using structured_exponent_type = numext::int64_t;

/** \internal max(|re z|, |im z|), the component magnitude the balanced forms
 * key on; from the representation for std::complex<float> and
 * std::complex<double>, where a flush-to-zero comparison would read a subnormal
 * component as zero. */
template <typename Scalar>
typename NumTraits<Scalar>::Real structured_component_magnitude_impl(const Scalar& z, std::true_type) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Binary = binary_floating_point_traits<RealScalar>;
  return numext::bit_cast<RealScalar>(
      larger_magnitude_bits<RealScalar>(Binary::magnitude(numext::real(z)), Binary::magnitude(numext::imag(z))));
}
template <typename Scalar>
typename NumTraits<Scalar>::Real structured_component_magnitude_impl(const Scalar& z, std::false_type) {
  return numext::maxi(numext::abs(numext::real(z)), numext::abs(numext::imag(z)));
}
template <typename Scalar>
typename NumTraits<Scalar>::Real structured_component_magnitude(const Scalar& z) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  return structured_component_magnitude_impl(z,
                                             bool_constant < complex_array_access<Scalar>::value &&
                                                 use_subnormal_preserving_scaling<RealScalar, RealScalar>::value > ());
}

/** \internal Balanced mantissa*2^e arithmetic shared by the structured
 * operators' determinant-style accumulations (the split fraction/exponent
 * convention of LINPACK's xGEDI; see the per-class references).
 * structured_balance() rescales \a z by the power of two that brings
 * \c max(|re|,|im|) (the modulus can overflow where the components do not) --
 * or \c |z| for a real scalar -- into [0.5, 1), accumulating the removed
 * exponent into \a exponent. The rescaling is exact, so no roundoff is
 * introduced; zeros and non-finite values, which must propagate exactly, are
 * returned untouched. The tests and the scalings read the representation for
 * float and double (numext::is_exactly_zero_no_flush(),
 * internal::frexp_exponent_preserving_subnormals(),
 * internal::ldexp_preserving_subnormals()): under flush-to-zero a comparison
 * or a frexp on a subnormal recovered from a flushed reduction would read it as
 * zero again. */
template <typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct structured_balance_impl {
  using RealScalar = typename NumTraits<Scalar>::Real;
  template <typename Exponent>
  static Scalar run(const Scalar& z, Exponent& exponent) {
    const RealScalar mag = structured_component_magnitude(z);
    if (numext::is_exactly_zero_no_flush(mag) || !(numext::isfinite)(mag)) return z;
    const int e = frexp_exponent_preserving_subnormals(mag);
    exponent += e;
    return apply_exponent(z, -e);
  }
  static Scalar apply_exponent(const Scalar& z, int e) {
    return Scalar(ldexp_preserving_subnormals(numext::real(z), e), ldexp_preserving_subnormals(numext::imag(z), e));
  }
};

template <typename Scalar>
struct structured_balance_impl<Scalar, false> {
  template <typename Exponent>
  static Scalar run(const Scalar& x, Exponent& exponent) {
    if (numext::is_exactly_zero_no_flush(x) || !(numext::isfinite)(x)) return x;
    const int e = frexp_exponent_preserving_subnormals(x);
    exponent += e;
    return apply_exponent(x, -e);
  }
  static Scalar apply_exponent(const Scalar& x, int e) { return ldexp_preserving_subnormals(x, e); }
};

template <typename Scalar, typename Exponent>
Scalar structured_balance(const Scalar& z, Exponent& exponent) {
  return structured_balance_impl<Scalar>::run(z, exponent);
}

/** \internal Applies an accumulated power-of-two \a exponent to \a z,
 * component-wise for complex scalars. ldexp saturates cleanly to zero /
 * infinity (preserving signs) once the exponent leaves the representable
 * range; the clamp only guards the narrowing to int. */
template <typename Scalar, typename Exponent>
Scalar structured_ldexp_clamped(const Scalar& z, Exponent exponent) {
  constexpr Exponent kMaxExponent = Exponent(1) << 24;
  const int e = static_cast<int>(numext::mini(numext::maxi(exponent, -kMaxExponent), kMaxExponent));
  return structured_balance_impl<Scalar>::apply_exponent(z, e);
}

/** \internal \returns the indices sorted by decreasing precomputed modulus
 * \a mods (each modulus is computed once, not on every comparison); the shared
 * ordering of the operators' singularValues()/matrixU()/matrixV(). The sort is
 * stable so repeated calls agree even in the presence of ties, and NaN moduli
 * order last (comparing through NaN directly would break the strict weak
 * ordering std::stable_sort requires). */
template <typename RealVectorType>
std::vector<Index> structured_svd_permutation(const RealVectorType& mods) {
  using RealScalar = typename RealVectorType::Scalar;
  std::vector<Index> perm;
  perm.reserve(static_cast<std::size_t>(mods.size()));
  for (Index k = 0; k < mods.size(); ++k) perm.push_back(k);
  std::stable_sort(perm.begin(), perm.end(), [&mods](Index a, Index b) {
    const RealScalar ka = mods[a], kb = mods[b];
    // isgreater is quiet for NaNs; the second clause implements NaN-last ordering.
    return std::isgreater(ka, kb) || (!(numext::isnan)(ka) && (numext::isnan)(kb));
  });
  return perm;
}

/** \internal \returns the per-thread FFT engine shared by all structured
 * operators. The kissfft backend caches its twiddle/plan tables per transform
 * size inside the engine, so reusing one engine amortizes the plan setup that a
 * per-call engine would redo on every product, solve and symbol computation
 * (the tables themselves are identical, so results are bit-for-bit unchanged).
 * One engine per thread keeps concurrent products on the same operator free of
 * data races; the cache grows with the number of distinct transform sizes a
 * thread touches, which mirrors the operators it works with.
 *
 * Under EIGEN_AVOID_THREAD_LOCAL -- the library-wide opt-out for targets
 * without usable `thread_local` (see Core's Memory.h and ThreadPool's
 * ThreadLocal.h) -- each call returns a fresh engine by value instead: the
 * plan tables are rebuilt per call, the results are identical. Callers bind
 * the engine with `auto&&`, which works for both signatures. */
#ifndef EIGEN_AVOID_THREAD_LOCAL
template <typename RealScalar>
FFT<RealScalar>& structured_fft_engine() {
  static thread_local FFT<RealScalar> fft;
  return fft;
}
#else
template <typename RealScalar>
FFT<RealScalar> structured_fft_engine() {
  return FFT<RealScalar>();
}
#endif

/** \internal
 * \returns the smallest integer >= \a n whose only prime factors are 2, 3 and 5.
 *
 * Such "5-smooth" sizes keep the FFT fast and sidestep the default kissfft
 * backend's poor handling of sizes with large prime factors. Used to pad the
 * circulant embedding of a Toeplitz matrix. The {2,3,5}-smooth numbers are dense
 * enough that the linear search returns after only a handful of steps.
 */
inline Index fft_next_good_size(Index n) {
  if (n < 1) return 1;
  for (Index m = n;; ++m) {
    Index r = m;
    while (r % 2 == 0) r /= 2;
    while (r % 3 == 0) r /= 3;
    while (r % 5 == 0) r /= 5;
    if (r == 1) return m;
  }
}

/** \internal View a complex-valued expression as \a Scalar: the expression itself
 * when \a Scalar is complex, its real part when \a Scalar is real (the imaginary
 * part then only holds numerically negligible roundoff). \c run_scalar is the
 * single-coefficient analogue. A single dispatch struct keeps the number of
 * instantiated helpers down to one per scalar type. */
template <typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct structured_scalar_part_impl {
  template <typename Xpr>
  static const Xpr& run(const Xpr& xpr) {
    return xpr;
  }
  static const Scalar& run_scalar(const Scalar& x) { return x; }
};

template <typename Scalar>
struct structured_scalar_part_impl<Scalar, false> {
  template <typename Xpr>
  static typename Xpr::RealReturnType run(const Xpr& xpr) {
    return xpr.real();
  }
  static Scalar run_scalar(const std::complex<Scalar>& x) { return numext::real(x); }
};

/** \internal Computes an exponent bound \c e with \c max_k|x[k]| < 2^e (0 when
 * \a x is zero or reduces to a non-finite maximum) and \returns whether \a x is
 * safe for the transforms, from a single plain (fast-max) reduction pass. The
 * bound is derived from the component-wise magnitudes, never from the modulus:
 * a finite complex value near the overflow threshold has a non-representable
 * modulus, which would silently disable the overflow-protection scaling exactly
 * where it is needed. Bounding the modulus by twice the largest component costs
 * at most one extra bit.
 *
 * The routing predicate deliberately uses the fast max reduction, which is not
 * guaranteed to propagate NaN (the NaN-propagating reduction de-vectorizes to a
 * branchy scalar loop on strided component views). That is sufficient: an Inf
 * in NaN-free data always surfaces in the maximum (every comparison is
 * ordered), and a column containing NaN produces the all-NaN output the dense
 * product semantics require through *either* path -- every dot product picks
 * up a coeff*NaN term, and the transforms propagate NaN just the same -- so
 * missing a NaN here cannot change the result. */
template <typename Xpr>
bool structured_exponent_bound_finite(const Xpr& x, int& e) {
  using ScalarTraits = NumTraits<typename Xpr::Scalar>;
  using RealScalar = typename ScalarTraits::Real;
  // maxCoeff() asserts on an empty input, and a degenerate operand -- a rank-0
  // factor, a solve with no right-hand sides -- reaches here legitimately.
  // An empty operand bounds nothing, so its exponent bound is 0.
  e = 0;
  if (x.size() == 0) return true;
  RealScalar m;
  if (ScalarTraits::IsComplex)
    // realView() reduces over both components in one pass, vectorized for
    // direct-access storage; the strided real()/imag() views never vectorize.
    m = x.realView().cwiseAbs().maxCoeff();
  else
    m = x.cwiseAbs().maxCoeff();
  // A SIMD unit that flushes subnormal inputs (ARMv7 NEON, Arm FZ, DAZ) reads
  // an all-subnormal operand as zero; the rescan recovers the largest component
  // from its representation.
  m = safe_scaling<RealScalar>::recover_flushed_max_coeff(x, m);
  e = 0;
  if (!(numext::isfinite)(m)) return false;
  if (!numext::is_exactly_zero_no_flush(m)) {
    e = frexp_exponent_preserving_subnormals(m);
    if (ScalarTraits::IsComplex) ++e;
  }
  return true;
}

/** \internal The exponent bound alone (see structured_exponent_bound_finite()),
 * for callers that handle non-finite data separately: the bound is 0 there. */
template <typename Xpr>
int structured_exponent_bound(const Xpr& x) {
  int e;
  structured_exponent_bound_finite(x, e);
  return e;
}

// The packet form of M *= 2^e. ldexp saturates entrywise for every e without
// forming a possibly unrepresentable 2^e; for std::complex storage, realView()
// exposes both components to the real ldexp packets.
template <typename Xpr, std::enable_if_t<!NumTraits<typename Xpr::Scalar>::IsComplex, bool> = true>
void structured_ldexp_entries_packet(Xpr& M, int e) {
  M.array() = M.array().ldexp(e);
}

template <typename Xpr, std::enable_if_t<complex_array_access<typename Xpr::Scalar>::value, bool> = true>
void structured_ldexp_entries_packet(Xpr& M, int e) {
  M.realView().array() = M.realView().array().ldexp(e);
}

template <typename Xpr, std::enable_if_t<NumTraits<typename Xpr::Scalar>::IsComplex &&
                                             !complex_array_access<typename Xpr::Scalar>::value,
                                         bool> = true>
void structured_ldexp_entries_packet(Xpr& M, int e) {
  using Scalar = typename Xpr::Scalar;
  M = M.unaryExpr([e](const Scalar& z) { return structured_ldexp_clamped(z, Index(e)); });
}

template <typename Xpr>
void structured_ldexp_entries_impl(Xpr& M, int e, int, std::false_type) {
  structured_ldexp_entries_packet(M, e);
}

template <typename Xpr>
void structured_ldexp_entries_impl(Xpr& M, int e, int bound, std::true_type) {
  using Scalar = typename Xpr::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  // The largest component is at least 2^(componentBound - 1): a complex bound
  // carries one extra bit for the modulus. The input holds no significant
  // subnormal when componentBound - 1 >= recovery, and neither does the result
  // when componentBound - 1 + e >= recovery.
  const int componentBound = bound - (NumTraits<Scalar>::IsComplex ? 1 : 0);
  const int recovery = safe_scaling<RealScalar>::subnormal_recovery_exponent();
  if (componentBound - 1 >= recovery && componentBound - 1 + e >= recovery)
    structured_ldexp_entries_packet(M, e);
  else
    M = M.unaryExpr(scale_by_exponent_op<RealScalar>(e));
}

/** \internal M *= 2^e, exactly wherever the result is representable, with
 * ldexp's saturation beyond. \a bound is the exponent bound of the input,
 * max|M| < 2^bound with 2^(bound - 1) <= max|M| as structured_exponent_bound()
 * returns it, or 0 for zero and non-finite data. Coefficients within a factor
 * 2^(1 - digits) of the largest one are exact: where the input or the result
 * can hold subnormals among them the scaling goes through integer significands
 * (internal::scale_by_exponent_op), which FTZ/DAZ hardware and ARMv7 NEON
 * cannot flush; elsewhere the packet ldexp applies, and coefficients further
 * below the largest may still flush under FTZ/DAZ. See
 * structured_ldexp_entries_exact() for data where every coefficient matters. */
template <typename Xpr>
void structured_ldexp_entries(Xpr& M, int e, int bound) {
  if (e == 0) return;
  using Scalar = typename Xpr::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  structured_ldexp_entries_impl(M, e, bound,
                                bool_constant<use_subnormal_preserving_scaling<RealScalar, Scalar>::value>());
}

/** \internal M *= 2^e with the exponent bound taken from M itself, for data
 * whose bound the caller does not track, such as a result being folded back
 * from a normalized frame. */
template <typename Xpr>
void structured_ldexp_entries(Xpr& M, int e) {
  if (e == 0) return;
  structured_ldexp_entries(M, e, structured_exponent_bound(M));
}

template <typename Xpr>
void structured_ldexp_entries_exact_impl(Xpr& M, int e, std::false_type) {
  structured_ldexp_entries_packet(M, e);
}
template <typename Xpr>
void structured_ldexp_entries_exact_impl(Xpr& M, int e, std::true_type) {
  using RealScalar = typename NumTraits<typename Xpr::Scalar>::Real;
  M = M.unaryExpr(scale_by_exponent_op<RealScalar>(e));
}

/** \internal M *= 2^e through integer significands for float and double
 * whatever the magnitudes: for the factors of an LU, the poles of a secular
 * equation or a spectrum, a coefficient far below the largest one still
 * matters, and the packet path may flush it under FTZ/DAZ. The pass is scalar;
 * use it where it precedes a factorization or covers O(n) data. */
template <typename Xpr>
void structured_ldexp_entries_exact(Xpr& M, int e) {
  if (e == 0) return;
  using Scalar = typename Xpr::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  structured_ldexp_entries_exact_impl(M, e,
                                      bool_constant<use_subnormal_preserving_scaling<RealScalar, Scalar>::value>());
}

/** \internal \returns the index reversal of a DFT \a symbol: result[k] =
 * symbol[(p - k) mod p]. This is the symbol of the transposed operator: reversing
 * the generating sequence in index space reverses the frequencies of its DFT, for
 * both a circulant generator and the circulant embedding of a Toeplitz matrix.
 * An empty symbol (small operator, nothing cached) stays empty. */
template <typename ComplexVectorType>
ComplexVectorType structured_reverse_symbol(const ComplexVectorType& symbol) {
  const Index p = symbol.size();
  ComplexVectorType reversed(p);
  if (p > 0) {
    reversed[0] = symbol[0];
    reversed.tail(p - 1) = symbol.tail(p - 1).reverse();
  }
  return reversed;
}

/** \internal Computes \c dst.col(k) += alpha * ifft( symbol .* fft(rhs.col(k)) ) for
 * every column of \a rhs, i.e. applies the circulant operator whose eigenvalues
 * are \a symbol; the leading \a outSize entries of each back-transform form the
 * output column. Right-hand sides shorter than the transform length are
 * zero-padded into a buffer allocated once outside the column loop.
 *
 * The transforms are evaluated in plain floating-point arithmetic: a column within
 * a factor of about p * max|symbol| of the overflow threshold can overflow in the
 * intermediates even when the result is representable, and a single Inf or NaN
 * makes the whole output column non-finite. */
template <typename Scalar, typename Dest, typename Rhs>
void structured_fft_apply(Dest& dst, const Matrix<std::complex<typename NumTraits<Scalar>::Real>, Dynamic, 1>& symbol,
                          Index outSize, const Rhs& rhs, const Scalar& alpha) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Complex = std::complex<RealScalar>;
  using ComplexVector = Matrix<Complex, Dynamic, 1>;

  const Index p = symbol.size();
  eigen_assert(rhs.rows() <= p && outSize <= p);
  if (p == 1) {
    // The length-one DFT is the identity and is unsupported by kissfft.
    dst.row(0) += alpha * structured_scalar_part_impl<Scalar>::run(Complex(symbol.coeff(0)) *
                                                                   rhs.row(0).template cast<Complex>());
    return;
  }

  auto&& fft = structured_fft_engine<RealScalar>();
  ComplexVector xt = ComplexVector::Zero(p);
  ComplexVector xf(p), yt(p);
  for (Index k = 0; k < rhs.cols(); ++k) {
    xt.head(rhs.rows()) = rhs.col(k).template cast<Complex>();
    fft.fwd(xf, xt, p);
    xf.array() *= symbol.array();
    fft.inv(yt, xf, p);
    dst.col(k) += alpha * structured_scalar_part_impl<Scalar>::run(yt.head(outSize));
  }
}

/** \internal \returns the pseudo-inverse symbol of a circulant-diagonalized
 * operator: the reciprocal of every entry of \a symbol whose modulus is at least
 * \c tol, zero for the others. With \c w = 1/|z| the reciprocal is formed as
 * \c (conj(z) w) w, which stays representable wherever 1/z is and vectorizes as
 * plain products. A NaN entry fails the comparison and stays in the inverted set,
 * so it propagates instead of being silently truncated. */
template <typename SymbolType, typename ModsType, typename RealScalar>
SymbolType structured_pinv_symbol(const SymbolType& symbol, const ModsType& mods, const RealScalar& tol) {
  using Complex = typename SymbolType::Scalar;
  const auto w = (mods.array() < tol).select(RealScalar(0), mods.array().inverse()).template cast<Complex>().eval();
  return (symbol.array().conjugate() * w * w).matrix();
}

/** \internal \returns the rank threshold \c size * epsilon * max|symbol| of [Golub
 * and Van Loan, Matrix Computations, 4th ed., 5.4], clamped from below by the
 * smallest normal number like SVDBase::rank(); \a mods holds |symbol|. A finite
 * entry whose modulus overflows would make the threshold infinite and truncate
 * every mode, so that rare case takes the maximum from the halved symbol. */
template <typename SymbolType, typename ModsType>
typename ModsType::Scalar structured_rank_threshold(const SymbolType& symbol, const ModsType& mods) {
  using RealScalar = typename ModsType::Scalar;
  const RealScalar factor = RealScalar(mods.size()) * NumTraits<RealScalar>::epsilon();
  RealScalar tol = factor * mods.maxCoeff();
  if (!(numext::isfinite)(tol)) tol = (RealScalar(2) * factor) * (symbol * RealScalar(0.5)).cwiseAbs().maxCoeff();
  return numext::maxi(tol, (std::numeric_limits<RealScalar>::min)());
}

/** \internal Shared product implementation for the structured operator types.
 * Forwards to the operator's \c addProduct member, which performs the fast
 * matrix-vector product. The same body serves every dense product dispatch tag.
 *
 * The structured products carry the default product tag, so assignment has the
 * ordinary dense-product semantics: \c x = op * expr first materializes the
 * product into a temporary, which resolves every form of aliasing between the
 * destination and the right-hand side (same object, overlapping views,
 * expressions referencing the destination, destinations resized by the
 * assignment), and \c .noalias() skips the temporary under the usual caller
 * promise that no aliasing exists. */
template <typename Op, typename Rhs>
struct structured_product_impl : generic_product_impl_base<Op, Rhs, structured_product_impl<Op, Rhs>> {
  using Scalar = typename Product<Op, Rhs>::Scalar;

  template <typename Dest>
  static void evalTo(Dest& dst, const Op& lhs, const Rhs& rhs) {
    dst.setZero();
    scaleAndAddTo(dst, lhs, rhs, Scalar(1));
  }

  template <typename Dest>
  static void scaleAndAddTo(Dest& dst, const Op& lhs, const Rhs& rhs, const Scalar& alpha) {
    using RhsNested = typename nested_eval<Rhs, Op::RowsAtCompileTime>::type;
    RhsNested actualRhs(rhs);
    lhs.addProduct(dst, actualRhs, alpha);
  }
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_MATRIX_UTILS_H
