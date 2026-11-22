// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Mark Borgerding mark a borgerding net
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_UNSUPPORTED_TEST_FFT_TEST_SHARED_H
#define EIGEN_UNSUPPORTED_TEST_FFT_TEST_SHARED_H

// Enable runtime malloc tracking so test_inplace_complex<>() can assert the
// in-place scratch comes from the stack.  Allocation defaults to allowed; the
// tracking only fires inside explicit set_is_malloc_allowed(false) windows.
#define EIGEN_RUNTIME_NO_MALLOC

#include "main.h"

#include <thread>
#include <contrib/Eigen/FFT>

template <typename T>
inline std::complex<T> RandomCpx() {
  return std::complex<T>((T)(rand() / (T)RAND_MAX - .5), (T)(rand() / (T)RAND_MAX - .5));
}

using namespace std;
using namespace Eigen;

template <typename T>
inline complex<long double> promote(complex<T> x) {
  return complex<long double>((long double)x.real(), (long double)x.imag());
}

inline complex<long double> promote(float x) { return complex<long double>((long double)x); }
inline complex<long double> promote(double x) { return complex<long double>((long double)x); }
inline complex<long double> promote(long double x) { return complex<long double>((long double)x); }

// Reference DFT X[k0] = sum_k1 x[k1] w^(k0 k1 mod n), w = exp(-2 pi i / n). Reducing the exponent in
// integers indexes n precomputed twiddles with arguments in (-2 pi, 0], so the O(n^2) sum evaluates no
// exponential; evaluating exp at the unreduced phases -2 pi k0 k1 / n dominated the whole test.
template <typename VT1, typename VT2>
long double fft_rmse(const VT1& fftbuf, const VT2& timebuf) {
  long double totalpower = 0;
  long double difpower = 0;
  long double pi = acos((long double)-1);
  const size_t n = (size_t)timebuf.size();
  std::vector<complex<long double>> twiddle(n);
  for (size_t m = 0; m < n; ++m) twiddle[m] = polar((long double)1, (long double)(-2.) * pi * m / n);
  for (size_t k0 = 0; k0 < (size_t)fftbuf.size(); ++k0) {
    complex<long double> acc = 0;
    for (size_t k1 = 0; k1 < n; ++k1) {
      acc += promote(timebuf[k1]) * twiddle[(k0 * k1) % n];
    }
    totalpower += numext::abs2(acc);
    complex<long double> x = promote(fftbuf[k0]);
    complex<long double> dif = acc - x;
    difpower += numext::abs2(dif);
    // cerr << k0 << "\t" << acc << "\t" <<  x << "\t" << sqrt(numext::abs2(dif)) << endl;
  }
  // cerr << "rmse:" << sqrt(difpower/totalpower) << endl;
  return sqrt(difpower / totalpower);
}

template <typename VT1, typename VT2>
long double dif_rmse(const VT1 buf1, const VT2 buf2) {
  long double totalpower = 0;
  long double difpower = 0;
  size_t n = (min)(buf1.size(), buf2.size());
  for (size_t k = 0; k < n; ++k) {
    totalpower += (long double)((numext::abs2(buf1[k]) + numext::abs2(buf2[k])) / 2);
    difpower += (long double)(numext::abs2(buf1[k] - buf2[k]));
  }
  return sqrt(difpower / totalpower);
}

enum {
  StdVectorContainer,
  EigenVectorContainer,
  EigenRowVectorContainer,
  EigenArrayXContainer,
  EigenRowArrayXContainer
};

template <int Container, typename Scalar>
struct VectorType;

template <typename Scalar>
struct VectorType<StdVectorContainer, Scalar> {
  using type = vector<Scalar>;
};

template <typename Scalar>
struct VectorType<EigenVectorContainer, Scalar> {
  using type = Matrix<Scalar, Dynamic, 1>;
};

template <typename Scalar>
struct VectorType<EigenRowVectorContainer, Scalar> {
  using type = Matrix<Scalar, 1, Dynamic>;
};

template <typename Scalar>
struct VectorType<EigenArrayXContainer, Scalar> {
  using type = Array<Scalar, Dynamic, 1>;
};

template <typename Scalar>
struct VectorType<EigenRowArrayXContainer, Scalar> {
  using type = Array<Scalar, 1, Dynamic>;
};

template <int ScalarContainer, int ComplexContainer, typename T>
void test_scalar_generic(int nfft) {
  using Complex = typename FFT<T>::Complex;
  using Scalar = typename FFT<T>::Scalar;
  using ScalarVector = typename VectorType<ScalarContainer, Scalar>::type;
  using ComplexVector = typename VectorType<ComplexContainer, Complex>::type;

  FFT<T> fft;
  ScalarVector tbuf(nfft);
  ComplexVector freqBuf;
  for (int k = 0; k < nfft; ++k) tbuf[k] = (T)(rand() / (double)RAND_MAX - .5);

  // make sure it DOESN'T give the right full spectrum answer
  // if we've asked for half-spectrum
  fft.SetFlag(fft.HalfSpectrum);
  fft.fwd(freqBuf, tbuf);
  VERIFY((size_t)freqBuf.size() == (size_t)((nfft >> 1) + 1));
  VERIFY(T(fft_rmse(freqBuf, tbuf)) < test_precision<T>());  // gross check

  fft.ClearFlag(fft.HalfSpectrum);
  fft.fwd(freqBuf, tbuf);
  VERIFY((size_t)freqBuf.size() == (size_t)nfft);
  VERIFY(T(fft_rmse(freqBuf, tbuf)) < test_precision<T>());  // gross check

  if (nfft & 1) return;  // odd FFTs get the wrong size inverse FFT

  ScalarVector tbuf2;
  fft.inv(tbuf2, freqBuf);
  VERIFY(T(dif_rmse(tbuf, tbuf2)) < test_precision<T>());  // gross check

  // verify that the Unscaled flag takes effect
  ScalarVector tbuf3;
  fft.SetFlag(fft.Unscaled);

  fft.inv(tbuf3, freqBuf);

  for (int k = 0; k < nfft; ++k) tbuf3[k] *= T(1. / nfft);

  // for (size_t i=0;i<(size_t) tbuf.size();++i)
  //     cout << "freqBuf=" << freqBuf[i] << " in2=" << tbuf3[i] << " -  in=" << tbuf[i] << " => " << (tbuf3[i] -
  //     tbuf[i] ) <<  endl;

  VERIFY(T(dif_rmse(tbuf, tbuf3)) < test_precision<T>());  // gross check

  // verify that ClearFlag works
  fft.ClearFlag(fft.Unscaled);
  fft.inv(tbuf2, freqBuf);
  VERIFY(T(dif_rmse(tbuf, tbuf2)) < test_precision<T>());  // gross check
}

template <typename T>
void test_scalar(int nfft) {
  // std:vector is a special case that does not interact with DenseBase types
  test_scalar_generic<StdVectorContainer, StdVectorContainer, T>(nfft);

  // All Dense types as dst and src in various combinations
  test_scalar_generic<EigenVectorContainer, EigenVectorContainer, T>(nfft);
  test_scalar_generic<EigenVectorContainer, EigenRowVectorContainer, T>(nfft);
  test_scalar_generic<EigenVectorContainer, EigenArrayXContainer, T>(nfft);
  test_scalar_generic<EigenVectorContainer, EigenRowArrayXContainer, T>(nfft);

  test_scalar_generic<EigenRowVectorContainer, EigenVectorContainer, T>(nfft);
  test_scalar_generic<EigenRowVectorContainer, EigenRowVectorContainer, T>(nfft);
  test_scalar_generic<EigenRowVectorContainer, EigenArrayXContainer, T>(nfft);
  test_scalar_generic<EigenRowVectorContainer, EigenRowArrayXContainer, T>(nfft);

  test_scalar_generic<EigenArrayXContainer, EigenVectorContainer, T>(nfft);
  test_scalar_generic<EigenArrayXContainer, EigenRowVectorContainer, T>(nfft);
  test_scalar_generic<EigenArrayXContainer, EigenArrayXContainer, T>(nfft);
  test_scalar_generic<EigenArrayXContainer, EigenRowArrayXContainer, T>(nfft);

  test_scalar_generic<EigenRowArrayXContainer, EigenVectorContainer, T>(nfft);
  test_scalar_generic<EigenRowArrayXContainer, EigenRowVectorContainer, T>(nfft);
  test_scalar_generic<EigenRowArrayXContainer, EigenArrayXContainer, T>(nfft);
  test_scalar_generic<EigenRowArrayXContainer, EigenRowArrayXContainer, T>(nfft);
}

template <int ContainerA, int ContainerB, typename T>
void test_complex_generic(int nfft) {
  using Complex = typename FFT<T>::Complex;
  using ComplexVectorA = typename VectorType<ContainerA, Complex>::type;
  using ComplexVectorB = typename VectorType<ContainerB, Complex>::type;

  FFT<T> fft;

  ComplexVectorB inbuf(nfft);
  ComplexVectorA outbuf;
  ComplexVectorB buf3;
  for (int k = 0; k < nfft; ++k)
    inbuf[k] = Complex((T)(rand() / (double)RAND_MAX - .5), (T)(rand() / (double)RAND_MAX - .5));
  fft.fwd(outbuf, inbuf);

  VERIFY(T(fft_rmse(outbuf, inbuf)) < test_precision<T>());  // gross check
  fft.inv(buf3, outbuf);

  VERIFY(T(dif_rmse(inbuf, buf3)) < test_precision<T>());  // gross check

  // verify that the Unscaled flag takes effect
  ComplexVectorA buf4;
  fft.SetFlag(fft.Unscaled);
  fft.inv(buf4, outbuf);
  for (int k = 0; k < nfft; ++k) buf4[k] *= T(1. / nfft);
  VERIFY(T(dif_rmse(inbuf, buf4)) < test_precision<T>());  // gross check

  // verify that ClearFlag works
  fft.ClearFlag(fft.Unscaled);
  fft.inv(buf3, outbuf);
  VERIFY(T(dif_rmse(inbuf, buf3)) < test_precision<T>());  // gross check
}

template <typename T>
void test_complex_strided(int nfft) {
  using Complex = typename FFT<T>::Complex;
  using ComplexVector = typename Eigen::Vector<Complex, Dynamic>;
  constexpr int kInputStride = 3;
  constexpr int kOutputStride = 7;
  constexpr int kInvOutputStride = 13;

  FFT<T> fft;

  ComplexVector inbuf(nfft * kInputStride);
  inbuf.setRandom();
  ComplexVector outbuf(nfft * kOutputStride);
  outbuf.setRandom();
  ComplexVector invoutbuf(nfft * kInvOutputStride);
  invoutbuf.setRandom();

  using StridedComplexVector = Map<ComplexVector, /*MapOptions=*/0, InnerStride<Dynamic>>;
  StridedComplexVector input(inbuf.data(), nfft, InnerStride<Dynamic>(kInputStride));
  StridedComplexVector output(outbuf.data(), nfft, InnerStride<Dynamic>(kOutputStride));
  StridedComplexVector inv_output(invoutbuf.data(), nfft, InnerStride<Dynamic>(kInvOutputStride));

  for (int k = 0; k < nfft; ++k)
    input[k] = Complex((T)(rand() / (double)RAND_MAX - .5), (T)(rand() / (double)RAND_MAX - .5));
  fft.fwd(output, input);

  VERIFY(T(fft_rmse(output, input)) < test_precision<T>());  // gross check
  fft.inv(inv_output, output);
  VERIFY(T(dif_rmse(inv_output, input)) < test_precision<T>());  // gross check
}

template <typename T>
void test_complex(int nfft) {
  // std:vector is a special case that does not interact with DenseBase types
  test_complex_generic<StdVectorContainer, StdVectorContainer, T>(nfft);

  // All Dense types as dst and src in various combinations
  test_complex_generic<EigenVectorContainer, EigenVectorContainer, T>(nfft);
  test_complex_generic<EigenVectorContainer, EigenRowVectorContainer, T>(nfft);
  test_complex_generic<EigenVectorContainer, EigenArrayXContainer, T>(nfft);
  test_complex_generic<EigenVectorContainer, EigenRowArrayXContainer, T>(nfft);

  test_complex_generic<EigenRowVectorContainer, EigenVectorContainer, T>(nfft);
  test_complex_generic<EigenRowVectorContainer, EigenRowVectorContainer, T>(nfft);
  test_complex_generic<EigenRowVectorContainer, EigenArrayXContainer, T>(nfft);
  test_complex_generic<EigenRowVectorContainer, EigenRowArrayXContainer, T>(nfft);

  test_complex_generic<EigenArrayXContainer, EigenVectorContainer, T>(nfft);
  test_complex_generic<EigenArrayXContainer, EigenRowVectorContainer, T>(nfft);
  test_complex_generic<EigenArrayXContainer, EigenArrayXContainer, T>(nfft);
  test_complex_generic<EigenArrayXContainer, EigenRowArrayXContainer, T>(nfft);

  test_complex_generic<EigenRowArrayXContainer, EigenVectorContainer, T>(nfft);
  test_complex_generic<EigenRowArrayXContainer, EigenRowVectorContainer, T>(nfft);
  test_complex_generic<EigenRowArrayXContainer, EigenArrayXContainer, T>(nfft);
  test_complex_generic<EigenRowArrayXContainer, EigenRowArrayXContainer, T>(nfft);

  test_complex_strided<T>(nfft);
}

template <typename T, int nrows, int ncols>
void test_complex2d() {
  using Complex = typename Eigen::FFT<T>::Complex;
  FFT<T> fft;
  Eigen::Matrix<Complex, nrows, ncols> src, src2, dst, dst2;

  src = Eigen::Matrix<Complex, nrows, ncols>::Random();
  // src =  Eigen::Matrix<Complex,nrows,ncols>::Identity();

  for (int k = 0; k < ncols; k++) {
    Eigen::Matrix<Complex, nrows, 1> tmpOut;
    fft.fwd(tmpOut, src.col(k));
    dst2.col(k) = tmpOut;
  }

  for (int k = 0; k < nrows; k++) {
    Eigen::Matrix<Complex, 1, ncols> tmpOut;
    fft.fwd(tmpOut, dst2.row(k));
    dst2.row(k) = tmpOut;
  }

  fft.fwd2(dst.data(), src.data(), ncols, nrows);
  fft.inv2(src2.data(), dst.data(), ncols, nrows);
  VERIFY((src - src2).norm() < test_precision<T>());
  VERIFY((dst - dst2).norm() < test_precision<T>());
}

// Regression for issue #868: fft.fwd(buf, buf) / fft.inv(buf, buf) with the
// same buffer as input and output must produce the out-of-place result.
// Also pins down that the in-place scratch comes from the stack for typical
// sizes so EIGEN_RUNTIME_NO_MALLOC users aren't forced to heap-allocate.
template <typename T>
void test_inplace_complex(int nfft) {
  using Complex = typename FFT<T>::Complex;
  using ComplexVector = Matrix<Complex, Dynamic, 1>;

  ComplexVector in(nfft);
  for (int k = 0; k < nfft; ++k)
    in[k] = Complex((T)(rand() / (double)RAND_MAX - .5), (T)(rand() / (double)RAND_MAX - .5));

  FFT<T> fft;
  ComplexVector out_ref;
  fft.fwd(out_ref, in);
  ComplexVector inv_ref;
  fft.inv(inv_ref, out_ref);

  ComplexVector inout_fwd = in;
  ComplexVector inout_inv = out_ref;

  Eigen::internal::set_is_malloc_allowed(false);
  fft.fwd(inout_fwd, inout_fwd);
  fft.inv(inout_inv, inout_inv);
  Eigen::internal::set_is_malloc_allowed(true);

  VERIFY((out_ref - inout_fwd).cwiseAbs().maxCoeff() < test_precision<T>());
  VERIFY((inv_ref - inout_inv).cwiseAbs().maxCoeff() < test_precision<T>());
}

// Regression for issue #675: zero-padding a fixed-size vector must not assume
// a column block in the temporary contiguous FFT input.
template <typename T>
void test_fwd_padding(int nfft) {
  using Complex = typename FFT<T>::Complex;
  using FixedRealColumn = Matrix<T, 10, 1>;
  using FixedRealRow = Matrix<T, 1, 10>;
  using FixedComplexColumn = Matrix<Complex, 10, 1>;
  using FixedComplexRow = Matrix<Complex, 1, 10>;
  using RealVector = Matrix<T, Dynamic, 1>;
  using ComplexVector = Matrix<Complex, Dynamic, 1>;

  FixedRealColumn real_column;
  FixedComplexColumn complex_column;
  for (int k = 0; k < real_column.size(); ++k) {
    real_column[k] = (T)(rand() / (double)RAND_MAX - .5);
    complex_column[k] = Complex((T)(rand() / (double)RAND_MAX - .5), (T)(rand() / (double)RAND_MAX - .5));
  }
  FixedRealRow real_row = real_column.transpose();
  FixedComplexRow complex_row = complex_column.transpose();

  RealVector real_padded = RealVector::Zero(nfft);
  real_padded.head(real_column.size()) = real_column;
  ComplexVector complex_padded = ComplexVector::Zero(nfft);
  complex_padded.head(complex_column.size()) = complex_column;

  FFT<T> fft;
  ComplexVector expected;
  ComplexVector actual;

  fft.fwd(expected, real_padded);
  fft.fwd(actual, real_column, nfft);
  VERIFY_IS_APPROX(actual, expected);
  fft.fwd(actual, real_row, nfft);
  VERIFY_IS_APPROX(actual, expected);

  fft.fwd(expected, complex_padded);
  fft.fwd(actual, complex_column, nfft);
  VERIFY_IS_APPROX(actual, expected);
  fft.fwd(actual, complex_row, nfft);
  VERIFY_IS_APPROX(actual, expected);

  fft.SetFlag(fft.HalfSpectrum);
  fft.fwd(expected, real_padded);
  fft.fwd(actual, real_column, nfft);
  VERIFY_IS_APPROX(actual, expected);
  fft.fwd(actual, real_row, nfft);
  VERIFY_IS_APPROX(actual, expected);
}

// An empty operand has an empty transform; no backend can plan a transform of length zero.
template <typename T>
void test_empty_operands() {
  using Complex = typename FFT<T>::Complex;
  using RealVector = Matrix<T, Dynamic, 1>;
  using ComplexVector = Matrix<Complex, Dynamic, 1>;
  using RealRowVector = Matrix<T, 1, Dynamic>;
  using ComplexRowVector = Matrix<Complex, 1, Dynamic>;

  for (int half = 0; half < 2; ++half) {
    FFT<T> fft;
    if (half) fft.SetFlag(fft.HalfSpectrum);

    RealVector real_in(0);
    ComplexVector complex_in(0), complex_out(3);
    fft.fwd(complex_out, real_in);
    VERIFY_IS_EQUAL(complex_out.size(), 0);
    complex_out.resize(3);
    fft.fwd(complex_out, complex_in);
    VERIFY_IS_EQUAL(complex_out.size(), 0);
    VERIFY_IS_EQUAL(fft.fwd(real_in).size(), 0);

    RealVector real_out(3);
    fft.inv(real_out, complex_in);
    VERIFY_IS_EQUAL(real_out.size(), 0);
    complex_out.resize(3);
    fft.inv(complex_out, complex_in);
    VERIFY_IS_EQUAL(complex_out.size(), 0);

    RealRowVector real_row_in(0), real_row_out(3);
    ComplexRowVector complex_row_in(0), complex_row_out(3);
    fft.fwd(complex_row_out, real_row_in);
    VERIFY_IS_EQUAL(complex_row_out.size(), 0);
    complex_row_out.resize(3);
    fft.inv(real_row_out, complex_row_in);
    VERIFY_IS_EQUAL(real_row_out.size(), 0);

    std::vector<T> real_vector;
    std::vector<Complex> complex_vector, complex_vector_out(3);
    std::vector<T> real_vector_out(3);
    fft.fwd(complex_vector_out, real_vector);
    VERIFY_IS_EQUAL(complex_vector_out.size(), size_t(0));
    complex_vector_out.resize(3);
    fft.fwd(complex_vector_out, complex_vector);
    VERIFY_IS_EQUAL(complex_vector_out.size(), size_t(0));
    fft.inv(real_vector_out, complex_vector);
    VERIFY_IS_EQUAL(real_vector_out.size(), size_t(0));
    std::vector<Complex> complex_vector_back(3);
    fft.inv(complex_vector_back, complex_vector);
    VERIFY_IS_EQUAL(complex_vector_back.size(), size_t(0));

    // An explicit length pads the empty operand with zeros.
    fft.fwd(complex_out, real_in, 4);
    VERIFY_IS_EQUAL(complex_out.size(), half ? 3 : 4);
    VERIFY(complex_out.isZero());
    fft.inv(real_out, complex_in, 4);
    VERIFY_IS_EQUAL(real_out.size(), 4);
    VERIFY(real_out.isZero());
    fft.inv(complex_out, complex_in, 4);
    VERIFY_IS_EQUAL(complex_out.size(), 4);
    VERIFY(complex_out.isZero());
  }
}

// Reference for inverting a spectrum X of length M at length N != M. The integer frequencies |f| < min(M, N) / 2 keep
// their bins. An even min(M, N) also has a Nyquist bin at N / 2: it is the mean of the bins at +-N / 2 when
// shrinking, and each of the bins at +-M / 2 takes half of the old Nyquist bin when expanding. An odd min(M, N)
// has none, so the spectrum of a real signal stays Hermitian and the inverse stays real.
template <typename T>
Matrix<std::complex<long double>, Dynamic, 1> resampled_inverse_reference(const Matrix<std::complex<T>, Dynamic, 1>& X,
                                                                          Index N) {
  using LongComplex = std::complex<long double>;
  const Index M = X.size();
  Matrix<LongComplex, Dynamic, 1> Y = Matrix<LongComplex, Dynamic, 1>::Zero(N);
  const auto bin = [&](Index f) { return LongComplex(X[((f % M) + M) % M]); };
  const Index m = (std::min)(M, N);
  if (M == N) {
    for (Index k = 0; k < N; ++k) Y[k] = bin(k);
  } else {
    for (Index f = -((m - 1) / 2); f <= (m - 1) / 2; ++f) Y[((f % N) + N) % N] = bin(f);
    if (m % 2 == 0) {
      const Index h = m / 2;
      if (M > N) {
        Y[h] = (bin(h) + bin(-h)) * 0.5L;
      } else {
        Y[h] = bin(h) * 0.5L;
        Y[N - h] = Y[h];
      }
    }
  }
  const long double pi = acos((long double)-1);
  Matrix<LongComplex, Dynamic, 1> y(N);
  for (Index n = 0; n < N; ++n) {
    LongComplex acc = 0;
    for (Index k = 0; k < N; ++k) acc += Y[k] * polar((long double)1, 2 * pi * ((k * n) % N) / N);
    y[n] = acc / (long double)N;
  }
  return y;
}

// inv(X, nfft) with nfft different from the length of X truncates or zero-pads the spectrum. The inverse of the
// spectrum of a real signal must remain real for every parity of the two lengths.
template <typename T>
void test_inv_resizes_spectrum() {
  using Complex = typename FFT<T>::Complex;
  using RealVector = Matrix<T, Dynamic, 1>;
  using ComplexVector = Matrix<Complex, Dynamic, 1>;
  const T eps = NumTraits<T>::epsilon();

  for (Index M = 1; M <= 13; ++M) {
    for (Index N = 1; N <= 15; ++N) {
      FFT<T> fft;
      RealVector x = RealVector::Random(M);
      ComplexVector hermitian;
      fft.fwd(hermitian, x);
      ComplexVector general = ComplexVector::Random(M);
      // Error of an N-point transform of unit-size coefficients, plus the rounding of the forward transform.
      const T tolerance = T(16) * T(M + N) * eps;

      const auto reference = resampled_inverse_reference(hermitian, N);
      RealVector real_result;
      fft.inv(real_result, hermitian, N);
      VERIFY_IS_EQUAL(real_result.size(), N);
      ComplexVector complex_result;
      fft.inv(complex_result, hermitian, N);
      VERIFY_IS_EQUAL(complex_result.size(), N);
      for (Index n = 0; n < N; ++n) {
        VERIFY(numext::abs(T(reference[n].imag())) <= tolerance);
        VERIFY(numext::abs(T(reference[n].real()) - real_result[n]) <= tolerance);
        VERIFY(numext::abs(Complex(reference[n]) - complex_result[n]) <= tolerance);
      }

      const auto general_reference = resampled_inverse_reference(general, N);
      fft.inv(complex_result, general, N);
      for (Index n = 0; n < N; ++n) VERIFY(numext::abs(Complex(general_reference[n]) - complex_result[n]) <= tolerance);
    }
  }
}

// A half spectrum of S bins read as the spectrum of an even-length signal is resampled like its full spectrum. A
// length with nfft / 2 + 1 == S bins is the native half spectrum of that length and is not resampled.
template <typename T>
void test_inv_resizes_half_spectrum() {
  using Complex = typename FFT<T>::Complex;
  using RealVector = Matrix<T, Dynamic, 1>;
  using ComplexVector = Matrix<Complex, Dynamic, 1>;
  const T eps = NumTraits<T>::epsilon();

  for (Index M = 2; M <= 14; ++M) {
    for (Index N = 1; N <= 17; ++N) {
      const Index bins = M / 2 + 1;
      const bool shrinking = N / 2 + 1 < bins;
      // A half spectrum does not record the parity of its signal, and expanding one assumes the even length
      // 2 (bins - 1): for odd M the last bin is an ordinary bin that is wrongly made real and halved. The accuracy
      // check below therefore excludes expanding from an odd M (a documented limitation of inv() with nfft).
      const bool expanding = N / 2 + 1 > bins && M % 2 == 0;
      if (!shrinking && !expanding) continue;
      RealVector x = RealVector::Random(M);
      FFT<T> full_fft, half_fft;
      half_fft.SetFlag(half_fft.HalfSpectrum);
      ComplexVector full, half;
      full_fft.fwd(full, x);
      half_fft.fwd(half, x);
      VERIFY_IS_EQUAL(half.size(), bins);
      RealVector from_full, from_half;
      full_fft.inv(from_full, full, N);
      half_fft.inv(from_half, half, N);
      VERIFY_IS_EQUAL(from_half.size(), N);
      VERIFY((from_full - from_half).cwiseAbs().maxCoeff() <= T(16) * T(M + N) * eps);
    }
  }
}

inline void test_return_by_value(int len) {
  VectorXf in;
  VectorXf in1;
  in.setRandom(len);
  VectorXcf out1, out2;
  FFT<float> fft;

  fft.SetFlag(fft.HalfSpectrum);

  fft.fwd(out1, in);
  out2 = fft.fwd(in);
  VERIFY((out1 - out2).norm() < test_precision<float>());
  in1 = fft.inv(out1);
  VERIFY((in1 - in).norm() < test_precision<float>());
}

// Regression for issue #1537: reusing the same FFT object across real-input
// and complex-input transforms of the same size must produce correct results.
// Before the fix, the FFTW backend's plan cache keyed only on (nfft, inverse,
// inplace, aligned), so an r2c plan could be returned for a c2c call (or
// vice versa).
template <typename T>
void test_reuse_real_and_complex(int nfft) {
  using Complex = typename FFT<T>::Complex;
  using ScalarVector = Matrix<T, Dynamic, 1>;
  using ComplexVector = Matrix<Complex, Dynamic, 1>;

  ScalarVector real_in(nfft);
  ComplexVector complex_in(nfft);
  for (int k = 0; k < nfft; ++k) {
    real_in[k] = (T)(rand() / (double)RAND_MAX - .5);
    complex_in[k] = Complex((T)(rand() / (double)RAND_MAX - .5), (T)(rand() / (double)RAND_MAX - .5));
  }

  FFT<T> fft;
  ComplexVector r2c_out;
  ComplexVector c2c_out;

  fft.fwd(r2c_out, real_in);
  fft.fwd(c2c_out, complex_in);
  VERIFY(T(fft_rmse(r2c_out, real_in)) < test_precision<T>());
  VERIFY(T(fft_rmse(c2c_out, complex_in)) < test_precision<T>());

  // Repeat with the reverse first-call ordering so the cache miss happens on
  // the opposite transform kind; this catches the symmetric c2c-then-r2c case.
  fft.fwd(c2c_out, complex_in);
  fft.fwd(r2c_out, real_in);
  VERIFY(T(fft_rmse(r2c_out, real_in)) < test_precision<T>());
  VERIFY(T(fft_rmse(c2c_out, complex_in)) < test_precision<T>());

  // Round-trip fwd->inv on the same shared FFT object exercises the c2r and
  // c2c inverse plans alongside the forward plans cached above.
  ScalarVector real_round;
  ComplexVector complex_round;
  fft.inv(real_round, r2c_out);
  VERIFY(T(dif_rmse(real_in, real_round)) < test_precision<T>());
  fft.inv(complex_round, c2c_out);
  VERIFY(T(dif_rmse(complex_in, complex_round)) < test_precision<T>());
}

// Regression test for issue #690: a dynamically sized matrix is only known to
// hold a single row or column at run time, so it must be accepted like any
// other one-dimensional operand, with the destination keeping its shape.
template <typename T>
void test_dynamic_matrix_operands(int nfft) {
  using Complex = typename FFT<T>::Complex;
  using DynMatrix = Matrix<T, Dynamic, Dynamic>;
  using DynCMatrix = Matrix<Complex, Dynamic, Dynamic>;

  FFT<T> fft;
  Matrix<T, Dynamic, 1> reference(nfft);
  for (int k = 0; k < nfft; ++k) reference[k] = T(rand() / (double)RAND_MAX - .5);
  Matrix<Complex, Dynamic, 1> reference_freq;
  fft.fwd(reference_freq, reference);

  // A 1-by-n source produces a 1-by-n spectrum with the same values.
  DynMatrix row_src(1, nfft);
  row_src.row(0) = reference.transpose();
  DynCMatrix row_freq;
  fft.fwd(row_freq, row_src);
  VERIFY_IS_EQUAL(row_freq.rows(), 1);
  VERIFY_IS_EQUAL(row_freq.cols(), Index(nfft));
  VERIFY(T(dif_rmse(row_freq.reshaped(), reference_freq)) < test_precision<T>());

  // ... and round trips back through the inverse transform.
  DynMatrix row_back;
  fft.inv(row_back, row_freq);
  VERIFY_IS_EQUAL(row_back.rows(), 1);
  VERIFY_IS_EQUAL(row_back.cols(), Index(nfft));
  VERIFY(T(dif_rmse(row_back.reshaped(), reference)) < test_precision<T>());

  // An n-by-1 source keeps the column shape.
  DynMatrix col_src(nfft, 1);
  col_src.col(0) = reference;
  DynCMatrix col_freq;
  fft.fwd(col_freq, col_src);
  VERIFY_IS_EQUAL(col_freq.rows(), Index(nfft));
  VERIFY_IS_EQUAL(col_freq.cols(), 1);
  VERIFY(T(dif_rmse(col_freq.reshaped(), reference_freq)) < test_precision<T>());

  // A destination whose orientation is fixed at compile time keeps it.
  Matrix<Complex, 1, Dynamic> row_vector_freq;
  fft.fwd(row_vector_freq, reference);
  VERIFY_IS_EQUAL(row_vector_freq.rows(), 1);
  VERIFY(T(dif_rmse(row_vector_freq.reshaped(), reference_freq)) < test_precision<T>());
}

// A matrix with only one dynamic dimension is one-dimensional at run time in exactly one
// orientation: the fixed dimension, when it is not one, has to be the long one. Such an operand
// is as acceptable as a fully dynamic one, and it pins the orientation of a destination.
template <typename T, int N>
void test_partially_dynamic_matrix_operands() {
  using Complex = typename FFT<T>::Complex;

  FFT<T> fft;
  Matrix<T, N, 1> reference;
  for (int k = 0; k < N; ++k) reference[k] = T(rand() / (double)RAND_MAX - .5);
  Matrix<Complex, N, 1> reference_freq;
  fft.fwd(reference_freq, reference);

  // A dynamic row count with N columns can only be a row.
  Matrix<T, Dynamic, N> row_src(1, N);
  row_src.row(0) = reference.transpose();
  Matrix<Complex, Dynamic, N> row_freq;
  fft.fwd(row_freq, row_src);
  VERIFY_IS_EQUAL(row_freq.rows(), 1);
  VERIFY(T(dif_rmse(row_freq.reshaped(), reference_freq)) < test_precision<T>());

  Matrix<T, Dynamic, N> row_back;
  fft.inv(row_back, row_freq);
  VERIFY_IS_EQUAL(row_back.rows(), 1);
  VERIFY(T(dif_rmse(row_back.reshaped(), reference)) < test_precision<T>());

  // A dynamic column count with N rows can only be a column.
  Matrix<T, N, Dynamic> col_src(N, 1);
  col_src.col(0) = reference;
  Matrix<Complex, N, Dynamic> col_freq;
  fft.fwd(col_freq, col_src);
  VERIFY_IS_EQUAL(col_freq.cols(), 1);
  VERIFY(T(dif_rmse(col_freq.reshaped(), reference_freq)) < test_precision<T>());

  Matrix<T, N, Dynamic> col_back;
  fft.inv(col_back, col_freq);
  VERIFY_IS_EQUAL(col_back.cols(), 1);
  VERIFY(T(dif_rmse(col_back.reshaped(), reference)) < test_precision<T>());
}

// Once matrices are accepted, a unit inner stride no longer means the operand is packed: a view
// holding a single row of a wider buffer steps along its outer stride instead.
template <typename T>
void test_strided_row_operands(int nfft) {
  using Complex = typename FFT<T>::Complex;
  using DynMatrix = Matrix<T, Dynamic, Dynamic>;
  using DynCMatrix = Matrix<Complex, Dynamic, Dynamic>;
  const Index stride = 3;

  FFT<T> fft;
  Matrix<T, Dynamic, 1> reference(nfft);
  for (int k = 0; k < nfft; ++k) reference[k] = T(rand() / (double)RAND_MAX - .5);
  Matrix<Complex, Dynamic, 1> reference_freq;
  fft.fwd(reference_freq, reference);

  // Strided source.
  DynMatrix storage = DynMatrix::Zero(stride, nfft);
  Map<DynMatrix, 0, OuterStride<>> src(storage.data(), 1, nfft, OuterStride<>(stride));
  src.row(0) = reference.transpose();
  Matrix<Complex, Dynamic, 1> freq;
  fft.fwd(freq, src);
  VERIFY(T(dif_rmse(freq, reference_freq)) < test_precision<T>());

  // Strided destination, and back again as a strided source.
  DynCMatrix freq_storage = DynCMatrix::Zero(stride, nfft);
  Map<DynCMatrix, 0, OuterStride<>> freq_dst(freq_storage.data(), 1, nfft, OuterStride<>(stride));
  fft.fwd(freq_dst, reference);
  VERIFY(T(dif_rmse(freq_storage.row(0), reference_freq)) < test_precision<T>());

  Matrix<T, Dynamic, 1> back;
  fft.inv(back, freq_dst);
  VERIFY(T(dif_rmse(back, reference)) < test_precision<T>());
}
#if defined EIGEN_FFTW_DEFAULT
// Distinct FFT objects must be usable from distinct threads concurrently.
// FFTW's planner is not thread-safe, so the FFTW backend serializes plan
// creation and destruction internally (issue #1483); each thread here creates
// fresh plans of many sizes while verifying its round trips.
void test_concurrent_transforms() {
  const int num_threads = 8;
  const int num_ffts = 24;
  std::vector<int> failures(num_threads, 0);
  std::vector<std::thread> workers;
  for (int t = 0; t < num_threads; ++t) {
    workers.emplace_back([t, &failures] {
      for (int k = 0; k < num_ffts; ++k) {
        const int nfft = 16 + 3 * ((t + k) % 40);
        Eigen::FFT<double> fft;
        std::vector<std::complex<double>> src(nfft), freq, back;
        for (int i = 0; i < nfft; ++i) src[i] = {std::cos(0.3 * i * (t + 1)), std::sin(0.7 * i + k)};
        fft.fwd(freq, src);
        fft.inv(back, freq);
        double max_err = 0;
        for (int i = 0; i < nfft; ++i) max_err = numext::maxi(max_err, std::abs(back[i] - src[i]));
        // VERIFY is not thread-safe; record and check after joining.
        if (!(max_err < 100 * nfft * NumTraits<double>::epsilon())) ++failures[t];
      }
    });
  }
  for (std::size_t t = 0; t < workers.size(); ++t) workers[t].join();
  for (int t = 0; t < num_threads; ++t) VERIFY_IS_EQUAL(failures[t], 0);
}
#endif  // EIGEN_FFTW_DEFAULT

EIGEN_DECLARE_TEST(FFTW) {
  CALL_SUBTEST(test_dynamic_matrix_operands<float>(32));
  CALL_SUBTEST(test_dynamic_matrix_operands<double>(32));
  CALL_SUBTEST(test_dynamic_matrix_operands<double>(2 * 3 * 4 * 5));
  CALL_SUBTEST((test_partially_dynamic_matrix_operands<float, 32>()));
  CALL_SUBTEST((test_partially_dynamic_matrix_operands<double, 32>()));
  CALL_SUBTEST((test_partially_dynamic_matrix_operands<double, 2 * 3 * 4 * 5>()));
  CALL_SUBTEST(test_strided_row_operands<float>(32));
  CALL_SUBTEST(test_strided_row_operands<double>(32));
  CALL_SUBTEST(test_strided_row_operands<double>(2 * 3 * 4 * 5));
  CALL_SUBTEST(test_return_by_value(32));
  CALL_SUBTEST(test_empty_operands<float>());
  CALL_SUBTEST(test_empty_operands<double>());
  CALL_SUBTEST(test_inv_resizes_spectrum<float>());
  CALL_SUBTEST(test_inv_resizes_spectrum<double>());
  CALL_SUBTEST(test_inv_resizes_half_spectrum<float>());
  CALL_SUBTEST(test_inv_resizes_half_spectrum<double>());
  // Regression test for #1537 -- reuse one FFT object for both real and
  // complex inputs of the same size.
  CALL_SUBTEST(test_reuse_real_and_complex<float>(32));
  CALL_SUBTEST(test_reuse_real_and_complex<double>(32));
  CALL_SUBTEST(test_reuse_real_and_complex<float>(256));
  CALL_SUBTEST(test_reuse_real_and_complex<double>(256));
  CALL_SUBTEST(test_inplace_complex<float>(32));
  CALL_SUBTEST(test_inplace_complex<double>(32));
  CALL_SUBTEST(test_inplace_complex<float>(256));
  CALL_SUBTEST(test_inplace_complex<double>(256));
  CALL_SUBTEST(test_fwd_padding<float>(16));
  CALL_SUBTEST(test_fwd_padding<double>(16));
  CALL_SUBTEST(test_complex<double>(1));
  CALL_SUBTEST(test_scalar<double>(1));
  CALL_SUBTEST(test_complex<float>(32));
  CALL_SUBTEST(test_complex<double>(32));
  CALL_SUBTEST(test_complex<float>(256));
  CALL_SUBTEST(test_complex<double>(256));
  CALL_SUBTEST(test_complex<float>(3 * 8));
  CALL_SUBTEST(test_complex<double>(3 * 8));
  CALL_SUBTEST(test_complex<float>(5 * 32));
  CALL_SUBTEST(test_complex<double>(5 * 32));
  CALL_SUBTEST(test_complex<float>(2 * 3 * 4));
  CALL_SUBTEST(test_complex<double>(2 * 3 * 4));
  CALL_SUBTEST(test_complex<float>(2 * 3 * 4 * 5));
  CALL_SUBTEST(test_complex<double>(2 * 3 * 4 * 5));
  CALL_SUBTEST(test_complex<float>(2 * 3 * 4 * 5 * 7));
  CALL_SUBTEST(test_complex<double>(2 * 3 * 4 * 5 * 7));

  CALL_SUBTEST(test_scalar<float>(32));
  CALL_SUBTEST(test_scalar<double>(32));
  CALL_SUBTEST(test_scalar<float>(45));
  CALL_SUBTEST(test_scalar<double>(45));
  CALL_SUBTEST(test_scalar<float>(50));
  CALL_SUBTEST(test_scalar<double>(50));
  CALL_SUBTEST(test_scalar<float>(256));
  CALL_SUBTEST(test_scalar<double>(256));
  CALL_SUBTEST(test_scalar<float>(2 * 3 * 4 * 5 * 7));
  CALL_SUBTEST(test_scalar<double>(2 * 3 * 4 * 5 * 7));

#if defined EIGEN_HAS_FFTWL || defined EIGEN_POCKETFFT_DEFAULT || defined EIGEN_DUCCFFT_DEFAULT
  CALL_SUBTEST(test_complex<long double>(32));
  CALL_SUBTEST(test_complex<long double>(256));
  CALL_SUBTEST(test_complex<long double>(3 * 8));
  CALL_SUBTEST(test_complex<long double>(5 * 32));
  CALL_SUBTEST(test_complex<long double>(2 * 3 * 4));
  CALL_SUBTEST(test_complex<long double>(2 * 3 * 4 * 5));
  CALL_SUBTEST(test_complex<long double>(2 * 3 * 4 * 5 * 7));

  CALL_SUBTEST(test_scalar<long double>(32));
  CALL_SUBTEST(test_scalar<long double>(45));
  CALL_SUBTEST(test_scalar<long double>(50));
  CALL_SUBTEST(test_scalar<long double>(256));
  CALL_SUBTEST(test_scalar<long double>(2 * 3 * 4 * 5 * 7));

  CALL_SUBTEST((test_complex2d<long double, 2 * 3 * 4, 2 * 3 * 4>()));
  CALL_SUBTEST((test_complex2d<long double, 3 * 4 * 5, 3 * 4 * 5>()));
  CALL_SUBTEST((test_complex2d<long double, 24, 60>()));
  CALL_SUBTEST((test_complex2d<long double, 60, 24>()));
// fail to build since Eigen limit the stack allocation size,too big here.
// CALL_SUBTEST( ( test_complex2d<long double, 256, 256> () ) );
#endif
#if defined EIGEN_FFTW_DEFAULT || defined EIGEN_POCKETFFT_DEFAULT || defined EIGEN_DUCCFFT_DEFAULT || \
    defined EIGEN_MKL_DEFAULT
  CALL_SUBTEST((test_complex2d<float, 24, 24>()));
  CALL_SUBTEST((test_complex2d<float, 60, 60>()));
  CALL_SUBTEST((test_complex2d<float, 24, 60>()));
  CALL_SUBTEST((test_complex2d<float, 60, 24>()));
#endif
#if defined EIGEN_FFTW_DEFAULT || defined EIGEN_POCKETFFT_DEFAULT || defined EIGEN_DUCCFFT_DEFAULT || \
    defined EIGEN_MKL_DEFAULT
  CALL_SUBTEST((test_complex2d<double, 24, 24>()));
  CALL_SUBTEST((test_complex2d<double, 60, 60>()));
  CALL_SUBTEST((test_complex2d<double, 24, 60>()));
  CALL_SUBTEST((test_complex2d<double, 60, 24>()));
#endif

#if defined EIGEN_FFTW_DEFAULT
  CALL_SUBTEST(test_concurrent_transforms());
#endif
}

#endif  // EIGEN_UNSUPPORTED_TEST_FFT_TEST_SHARED_H
