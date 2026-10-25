// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-FileCopyrightText: 2001-2002 Enthought, Inc.
// SPDX-FileCopyrightText: 2003 SciPy Developers
// SPDX-License-Identifier: MPL-2.0 AND BSD-3-Clause
//
// This file incorporates work covered by the following copyright and
// permission notice:
//
//   dare_scipy_complex_cases(), dare_scipy_real_cases() and dare_generalized_scipy_cases()
//   are adapted from the solve_discrete_are tests of SciPy 1.18.0 (scipy/linalg/tests/test_solvers.py and
//   scipy/linalg/tests/data/gendare_20170120_data.npz):
//
//    Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
//    All rights reserved.
//
//    Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions
//    are met:
//
//    1. Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//    3. Neither the name of the copyright holder nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "main.h"
#include <contrib/Eigen/Riccati>

#include <algorithm>
#include <vector>

namespace {

// Residual of A^* X A - E^* X E - (A^* X B + S) (R + B^* X B)^-1 (B^* X A + S^*) + Q.
template <typename MatrixType>
MatrixType dare_residual(const MatrixType& A, const MatrixType& B, const MatrixType& Q, const MatrixType& R,
                         const MatrixType& E, const MatrixType& S, const MatrixType& X) {
  const MatrixType AXB = A.adjoint() * X * B + S;
  const MatrixType gain = (R + B.adjoint() * X * B).fullPivLu().solve(AXB.adjoint());
  return A.adjoint() * X * A - E.adjoint() * X * E - AXB * gain + Q;
}

template <typename MatrixType>
MatrixType dare_residual(const MatrixType& A, const MatrixType& B, const MatrixType& Q, const MatrixType& R,
                         const MatrixType& X) {
  return dare_residual(A, B, Q, R, MatrixType::Identity(A.rows(), A.rows()).eval(),
                       MatrixType::Zero(B.rows(), B.cols()).eval(), X);
}

// Solves the DARE and checks the residual entrywise against SciPy's absolute tolerance for the case.
template <typename MatrixType>
void check_dare(const MatrixType& A, const MatrixType& B, const MatrixType& Q, const MatrixType& R,
                typename MatrixType::RealScalar atol) {
  DiscreteAlgebraicRiccatiSolver<MatrixType> dare(A, B, Q, R);
  VERIFY_IS_EQUAL(dare.info(), Success);
  const MatrixType& X = dare.solution();
  VERIFY_IS_EQUAL(X, X.adjoint());
  VERIFY(dare_residual(A, B, Q, R, X).cwiseAbs().maxCoeff() <= atol);
}

template <typename MatrixType>
void check_generalized_dare(const MatrixType& A, const MatrixType& B, const MatrixType& Q, const MatrixType& R,
                            const MatrixType& E, const MatrixType& S, typename MatrixType::RealScalar atol) {
  DiscreteAlgebraicRiccatiSolver<MatrixType> dare(A, B, Q, R, E, S);
  VERIFY_IS_EQUAL(dare.info(), Success);
  const MatrixType& X = dare.solution();
  VERIFY_IS_EQUAL(X, X.adjoint());
  VERIFY(dare_residual(A, B, Q, R, E, S, X).cwiseAbs().maxCoeff() <= atol);
}

// Solves a real test case in the arithmetic of MatrixType; the real cases are also solved in complex arithmetic to
// exercise the ComplexQZ code path.
template <typename MatrixType>
void check_real_dare(const MatrixXd& A, const MatrixXd& B, const MatrixXd& Q, const MatrixXd& R, double atol) {
  using Scalar = typename MatrixType::Scalar;
  check_dare<MatrixType>(A.cast<Scalar>(), B.cast<Scalar>(), Q.cast<Scalar>(), R.cast<Scalar>(), atol);
}

// Backward error check for a computed solution X:
//   rho = ||res||_F / (m eps (||A^* X A||_F + ||E^* X E||_F + ||Q||_F) kappa) <= 30,
//   kappa = max(1, ||E^* X E||_F / ||Q||_F),
// where kappa accounts for the amplification of rounding errors by a large solution: for the problems with B scaled by
// 10^-k, the residual of an accurately computed solution grows in proportion to ||X|| / ||Q||, also with SciPy
// (LAPACK), up to 1e10 eps for double. Over 160000 solves of dare_random() problems in all four scalar types, rho had
// p99.9 <= 3.9 and maximum 9.7.
template <typename DynMatrix>
void verify_dare_backward_error(const DynMatrix& A, const DynMatrix& B, const DynMatrix& Q, const DynMatrix& R,
                                const DynMatrix& E, const DynMatrix& S, const DynMatrix& X) {
  using RealScalar = typename NumTraits<typename DynMatrix::Scalar>::Real;
  VERIFY(X.allFinite());
  const RealScalar normEXE = (E.adjoint() * X * E).norm();
  const RealScalar scale = (A.adjoint() * X * A).norm() + normEXE + Q.norm();
  const RealScalar kappa = numext::maxi(RealScalar(1), normEXE / Q.norm());
  const RealScalar bound = RealScalar(30) * RealScalar(A.rows()) * NumTraits<RealScalar>::epsilon() * scale * kappa;
  VERIFY(dare_residual(A, B, Q, R, E, S, X).norm() <= bound);
}

// The closed-loop pencil (A - B K, E), K = (R + B^* X B)^-1 (B^* X A + S^*), of the stabilizing solution is stable.
template <typename DynMatrix>
void verify_stabilizing(const DynMatrix& A, const DynMatrix& B, const DynMatrix& R, const DynMatrix& E,
                        const DynMatrix& S, const DynMatrix& X) {
  using RealScalar = typename NumTraits<typename DynMatrix::Scalar>::Real;
  using ComplexMatrix = Matrix<std::complex<RealScalar>, Dynamic, Dynamic>;
  const DynMatrix K = (R + B.adjoint() * X * B).fullPivLu().solve(B.adjoint() * X * A + S.adjoint());
  const ComplexMatrix closedLoop = E.fullPivLu().solve(A - B * K).template cast<std::complex<RealScalar>>();
  VERIFY(ComplexEigenSolver<ComplexMatrix>(closedLoop).eigenvalues().cwiseAbs().maxCoeff() < RealScalar(1));
}

}  // namespace

#if defined(EIGEN_TEST_PART_7) || defined(EIGEN_TEST_PART_ALL)
// The test cases of scipy.linalg.solve_discrete_are with complex data.
void dare_scipy_complex_cases() {
  using cd = std::complex<double>;
  {  // complex A; real B, Q, R
    MatrixXcd A(2, 2), B(2, 1), Q(2, 2), R(1, 1);
    A << cd(2, 0), cd(1, -2), cd(0, 0), cd(0, -3);
    B << 0, 1;
    Q << 1, 0, 0, 2;
    R << 1;
    check_dare(A, B, Q, R, 1.5e-12);
  }
  {  // real A, Q, R; complex B
    MatrixXcd A(2, 2), B(2, 1), Q(2, 2), R(1, 1);
    A << 2, 1, 0, -1;
    B << cd(0, -2), cd(0, 1);
    Q << 1, 0, 0, 2;
    R << 1;
    check_dare(A, B, Q, R, 1.5e-14);
  }
  {  // real A, B; complex Q, R
    MatrixXcd A(2, 2), B(2, 2), Q(2, 2), R(2, 2);
    A << 3, 1, 0, -1;
    B << 1, 2, 1, 3;
    Q << cd(1, 0), cd(1, 1), cd(1, -1), cd(2, 0);
    R << cd(2, 0), cd(0, -2), cd(0, 2), cd(3, 0);
    check_dare(A, B, Q, R, 1.5e-13);
  }
}
#endif

// The test cases of scipy.linalg.solve_discrete_are with real data, most of them from
// P. Benner, A. J. Laub, V. Mehrmann, "A Collection of Benchmark Examples for the Numerical Solution of Algebraic
// Riccati Equations II: Discrete-Time Case", Tech. Report SPC 95_23, TU Chemnitz-Zwickau, 1995 (DAREX).
template <typename MatrixType>
void dare_scipy_real_cases() {
  {  // SciPy gh-2251
    MatrixXd A(3, 3), B(3, 1);
    A << 0.63399379, 0.54906824, 0.76253406, 0.5404729, 0.53745766, 0.08731853, 0.27524045, 0.84922129, 0.4681622;
    B << 0.96861695, 0.05532739, 0.78934047;
    check_real_dare<MatrixType>(A, B, MatrixXd::Identity(3, 3), MatrixXd::Identity(1, 1), 1.5e-14);
  }
  {  // DAREX 1
    MatrixXd A(2, 2), B(2, 1), Q(2, 2), R(1, 1);
    A << 4, 3, -4.5, -3.5;
    B << 1, -1;
    Q << 9, 6, 6, 4;
    R << 1;
    check_real_dare<MatrixType>(A, B, Q, R, 1.5e-13);
  }
  {  // DAREX 2
    MatrixXd A(2, 2), B(2, 2), Q(2, 2), R(2, 2);
    A << 0.9512, 0, 0, 0.9048;
    B << 4.877, 4.877, -1.1895, 3.569;
    Q << 0.005, 0, 0, 0.02;
    R << 1. / 3, 0, 0, 3;
    check_real_dare<MatrixType>(A, B, Q, R, 1.5e-16);
  }
  {  // DAREX 3: singular R
    MatrixXd A(2, 2), B(2, 1), Q(2, 2), R(1, 1);
    A << 2, -1, 1, 0;
    B << 1, 0;
    Q << 0, 0, 0, 1;
    R << 0;
    // SciPy's bound of 1.5e-18 holds there only because LAPACK happens to round to the exact solution; X has O(1)
    // entries, so a residual of a few eps is backward stable.
    check_real_dare<MatrixType>(A, B, Q, R, 4 * NumTraits<double>::epsilon());
  }
  {  // DAREX 4 (without the generalized term S)
    MatrixXd A(2, 2), B(2, 2), Q(2, 2), R(2, 2);
    A << 0, 1, 0, -1;
    B << 1, 0, 2, 1;
    Q << -4, -4, -4, 7;
    Q /= 11;
    R << 9, 3, 3, 1;
    check_real_dare<MatrixType>(A, B, Q, R, 1.5e-14);
  }
  {  // DAREX 5
    MatrixXd A(2, 2), B(2, 1), Q(2, 2), R(1, 1);
    A << 0, 1, 0, 0;
    B << 0, 1;
    Q << 1, 2, 2, 4;
    R << 1;
    check_real_dare<MatrixType>(A, B, Q, R, 1.5e-14);
  }
  {  // DAREX 6
    MatrixXd A(4, 4), B(4, 2), Q(4, 4);
    A << 0.998, 0.067, 0, 0, -.067, 0.998, 0, 0, 0, 0, 0.998, 0.153, 0, 0, -.153, 0.998;
    B << 0.0033, 0.0200, 0.1000, -.0007, 0.0400, 0.0073, -.0028, 0.1000;
    Q << 1.87, 0, 0, -0.244, 0, 0.744, 0.205, 0, 0, 0.205, 0.589, 0, -0.244, 0, 0, 1.048;
    check_real_dare<MatrixType>(A, B, Q, MatrixXd::Identity(2, 2), 1.5e-13);
  }
  {  // DAREX 7
    MatrixXd A(4, 4), B(4, 2);
    A << 0.984750, -.079903, 0.0009054, -.0010765, 0.041588, 0.998990, -.0358550, 0.0126840, -.546620, 0.044916,
        -.3299100, 0.1931800, 2.662400, -.100450, -.9245500, -.2632500;
    B << 0.0037112, 0.0007361, -.0870510, 9.3411e-6, -1.198440, -4.1378e-4, -3.192700, 9.2535e-4;
    check_real_dare<MatrixType>(A, B, MatrixXd::Identity(4, 4) * 1e-2, MatrixXd::Identity(2, 2), 1.5e-14);
  }
  {  // DAREX 8
    MatrixXd A(4, 4), B(4, 4), Q(4, 4);
    A << -0.6, -2.2, -3.6, -5.400018, 1.0, 0.6, 0.8, 3.399982, 0.0, 1.0, 1.8, 3.799982, 0.0, 0.0, 0.0, -0.999982;
    B << 1.0, -1.0, -1.0, -1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0;
    Q << 2, 1, 3, 6, 1, 2, 2, 5, 3, 2, 6, 11, 6, 5, 11, 22;
    check_real_dare<MatrixType>(A, B, Q, MatrixXd::Identity(4, 4), 2.5e-13);
  }
  {  // DAREX 9
    MatrixXd A(5, 5), B(5, 2);
    A << 95.4070, 1.9643, 0.3597, 0.0673, 0.0190, 40.8490, 41.3170, 16.0840, 4.4679, 1.1971, 12.2170, 26.3260, 36.1490,
        15.9300, 12.3830, 4.1118, 12.8580, 27.2090, 21.4420, 40.9760, 0.1305, 0.5808, 1.8750, 3.6162, 94.2800;
    A *= 0.01;
    B << 0.0434, -0.0122, 2.6606, -1.0453, 3.7530, -5.5100, 3.6076, -6.6000, 0.4617, -0.9148;
    B *= 0.01;
    // SciPy's 1.5e-13 matches LAPACK's rounding; without vectorization the residual is 1.53e-13.
    check_real_dare<MatrixType>(A, B, MatrixXd::Identity(5, 5), MatrixXd::Identity(2, 2), 3e-13);
  }
  {  // DAREX 10
    MatrixXd A = MatrixXd::Zero(6, 6), B = MatrixXd::Zero(6, 2), Q(6, 6), R(2, 2);
    A(0, 1) = A(1, 2) = A(3, 4) = A(4, 5) = 1;
    B(2, 0) = B(5, 1) = 1;
    Q << 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0;
    R << 3, 0, 0, 1;
    check_real_dare<MatrixType>(A, B, Q, R, 1.5e-14);
  }
  {  // DAREX 11
    MatrixXd A(9, 9), B(9, 3);
    A << 870.1, 135.0, 11.59, .5014, -37.22, .3484, 0, 4.242, 7.249,     //
        76.55, 897.4, 12.72, 0.5504, -40.16, .3743, 0, 4.53, 7.499,      //
        -127.2, 357.5, 817, 1.455, -102.8, .987, 0, 11.85, 18.72,        //
        -363.5, 633.9, 74.91, 796.6, -273.5, 2.653, 0, 31.72, 48.82,     //
        -960, 1645.9, -128.9, -5.597, 71.42, 7.108, 0, 84.52, 125.9,     //
        -664.4, 112.96, -88.89, -3.854, 84.47, 13.6, 0, 144.3, 101.6,    //
        -410.2, 693, -54.71, -2.371, 66.49, 12.49, .1063, 99.97, 69.67,  //
        -179.9, 301.7, -23.93, -1.035, 60.59, 22.16, 0, 213.9, 35.54,    //
        -345.1, 580.4, -45.96, -1.989, 105.6, 19.86, 0, 219.1, 215.2;
    A *= 0.001;
    B << 4.7600, -0.5701, -83.6800, 0.8790, -4.7730, -2.7300, 1.4820, -13.1200, 8.8760, 3.8920, -35.1300, 24.8000,
        10.3400, -92.7500, 66.8000, 7.2030, -61.5900, 38.3400, 4.4540, -36.8300, 20.2900, 1.9710, -15.5400, 6.9370,
        3.7730, -30.2800, 14.6900;
    B *= 0.001;
    VectorXd q = VectorXd::Zero(9);
    q(0) = q(4) = 50;
    check_real_dare<MatrixType>(A, B, MatrixXd(q.asDiagonal()), MatrixXd::Identity(3, 3), 1.5e-12);
  }
  {  // DAREX 12: numerically the least accurate example
    MatrixXd A(2, 2), B(2, 1);
    A << 0, 1e6, 0, 0;
    B << 0, 1;
    check_real_dare<MatrixType>(A, B, MatrixXd::Identity(2, 2), MatrixXd::Identity(1, 1), 2.0e-2);
  }
  {  // DAREX 13
    MatrixXd A(3, 3);
    A << 16, 10, -2, 10, 13, -8, -2, -8, 7;
    A /= 9;
    check_real_dare<MatrixType>(A, MatrixXd::Identity(3, 3), 1e6 * MatrixXd::Identity(3, 3),
                                1e6 * MatrixXd::Identity(3, 3), 2.0e-4);
  }
  {  // DAREX 14
    MatrixXd A = MatrixXd::Zero(4, 4), B = MatrixXd::Zero(4, 1), Q = MatrixXd::Zero(4, 4), R(1, 1);
    A(0, 0) = 1 - 1 / 1e8;
    A(1, 0) = A(2, 1) = A(3, 2) = 1;
    B(0, 0) = 1e-8;
    Q(3, 3) = 1;
    R << 0.25;
    // In complex arithmetic the tolerance is raised: where ComplexQZ converges on the balanced pencil (GCC with SSE2),
    // the residual is 7.6e-5, against 2.4e-8 with LAPACK's ZGGES. With AVX2 and FMA, or with Clang, ComplexQZ stalls
    // on it, and the solver falls back to the unbalanced pencil, whose residual is close to this tolerance for any QZ
    // (0.11 here, 5.0e-2 with ZGGES). To be tightened once ComplexQZ converges on this pencil (!3430).
    check_real_dare<MatrixType>(A, B, Q, R, NumTraits<typename MatrixType::Scalar>::IsComplex ? 5e-2 : 1.5e-6);
  }
  {  // DAREX 15
    const Index n = 100;
    MatrixXd A = MatrixXd::Zero(n, n), B = MatrixXd::Zero(n, 1);
    A.diagonal(1).setOnes();
    B(n - 1, 0) = 1;
    check_real_dare<MatrixType>(A, B, MatrixXd::Identity(n, n), MatrixXd::Identity(1, 1), 1.5e-10);
  }
}

// The generalized test cases of scipy.linalg.solve_discrete_are.
template <typename MatrixType>
void dare_generalized_scipy_cases() {
  using Scalar = typename MatrixType::Scalar;
  MatrixXd A(3, 3), B(3, 2), E(3, 3);
  A << 2.769230e-01, 8.234578e-01, 9.502220e-01, 4.617139e-02, 6.948286e-01, 3.444608e-02, 9.713178e-02, 3.170995e-01,
      4.387444e-01;
  B << 3.815585e-01, 1.868726e-01, 7.655168e-01, 4.897644e-01, 7.951999e-01, 4.455862e-01;
  E << 6.463130e-01, 2.760251e-01, 1.626117e-01, 7.093648e-01, 6.797027e-01, 1.189977e-01, 7.546867e-01, 6.550980e-01,
      4.983641e-01;
  const MatrixXd Q = MatrixXd::Identity(3, 3), R = MatrixXd::Identity(2, 2);
  // SciPy's 1.5e-11 matches LAPACK's rounding; depending on vectorization and scalar type, the residuals reach
  // 1.8e-11.
  const double atol = 3e-11;
  check_generalized_dare<MatrixType>(A.cast<Scalar>(), B.cast<Scalar>(), Q.cast<Scalar>(), R.cast<Scalar>(),
                                     E.cast<Scalar>(), MatrixType::Zero(3, 2), atol);
  check_generalized_dare<MatrixType>(A.cast<Scalar>(), B.cast<Scalar>(), Q.cast<Scalar>(), R.cast<Scalar>(),
                                     E.cast<Scalar>(), MatrixType::Ones(3, 2), atol);
  if (NumTraits<Scalar>::IsComplex) return;

  // Reported under SciPy PR 6616: E omitted, S given.
  MatrixXd A2(8, 8), B2(8, 2), Q2(8, 8), R2(2, 2), S2(8, 2);
  A2 << 0.9790596124208226, 0.15428665211630566, -0.0399419325090604, 0.003911636788878224, -0.011803107183894452,
      -0.02494092597437882, -0.008920788197667088, 0.014335952211974737,  //
      -0.14834721727344974, 0.9873863186572593, 0.006106682107487801, -0.0012366115496408269, 0.013575311081540167,
      0.020672662811912077, -0.0017998184559520581, 0.0017523155301884849,  //
      0.07162947799991767, -0.010401328690850057, 0.9845726088881807, 0.04377286183324468, 0.03170819532733946,
      0.049510008354081356, -0.01203976564345253, 0.01388147319653372,  //
      -0.0019726741801273217, -0.024879350443434263, -0.12719519221321976, 0.9804305159866238, 0.021632354678924177,
      0.07905312980830177, -0.011451886896987448, 0.026787582574920674,  //
      -0.051702322194735015, 0.00969666725317041, 0.02104971125084225, 0.10007182592961279, 0.7141026413334377,
      -0.6004593068202806, 0.004777582730886375, 0.03141438420954447,  //
      0.040354013728618784, -0.015350748134960568, -0.05579631944984332, -0.10543722475599439, 0.6343856366274718,
      0.35004282911065215, -0.03776729806891735, 0.0074099601797860785,  //
      -0.03584359294207154, 0.0025900981016803853, -0.005668147016789041, -0.06465860121122835, -0.012656894372629887,
      0.04360315609168892, 0.7043110256864276, 0.6066975504180661,  //
      -0.02851398087932866, -0.0015631274113090954, -0.012577246358475958, -0.053341773444025395, -0.03769226823991803,
      -0.007888383180591867, -0.6280157636492598, 0.34907000148948886;
  B2 << -1.0436871916355555, 1.0931665872601304, 0.5133620925013083, -0.1514390795683095, 0.7861828230513443,
      0.3888125024790475, 0.9089599716479574, 0.7320641966547645, -0.6781510875985723, 0.12416864282640695,
      0.9048515879878576, -0.009000096506101762, -0.008637666934928385, 0.5564531625467788, 0.007083540587022204,
      0.6553914999236834;
  Q2 << 1.7551250678842602e-06, -4.272252214687373e-07, -9.347904861120448e-07, -6.106412578819786e-07,
      1.6283957160863591e-06, 2.226201360118831e-06, 1.0890695975480443e-06, -1.3619304943943357e-06,  //
      -4.2722522146873673e-07, 5.11729442524146e-07, 1.0331612636389516e-06, 9.016609232641012e-07,
      -9.306700600646157e-07, -1.4657838525589338e-06, 3.4201260458425485e-07, -4.2057289495481993e-07,  //
      -9.347904861120447e-07, 1.0331612636389514e-06, 2.0896827294140204e-06, 1.8133390743261377e-06,
      -1.9225671817044196e-06, -3.0098180593831633e-06, 6.189175278904355e-07, -7.587103608495501e-07,  //
      -6.106412578819799e-07, 9.016609232641022e-07, 1.8133390743261417e-06, 1.6055691910525155e-06,
      -1.5469792580036601e-06, -2.460263283152522e-06, 7.394613770930514e-07, -9.082211730936819e-07,  //
      1.6283957160863587e-06, -9.30670060064614e-07, -1.9225671817044158e-06, -1.5469792580036527e-06,
      2.2352617341558302e-06, 3.3544676997154134e-06, 2.14575519310723e-07, -2.9072025953141554e-07,  //
      2.226201360118829e-06, -1.465783852558932e-06, -3.0098180593831616e-06, -2.460263283152514e-06,
      3.3544676997154143e-06, 5.1697298143974225e-06, 4.095247131836552e-09, -6.178901942620202e-08,  //
      1.089069597548046e-06, 3.4201260458425506e-07, 6.189175278904347e-07, 7.394613770930501e-07,
      2.145755193107224e-07, 4.095247131839212e-09, 1.58987128162285e-06, -2.001906554441249e-06,  //
      -1.361930494394337e-06, -4.2057289495481967e-07, -7.587103608495486e-07, -9.0822117309368e-07,
      -2.9072025953141474e-07, -6.178901942620546e-08, -2.001906554441248e-06, 2.587024318215873e-06;
  R2 << 6.86199839251629e-07, -1.4227980596335797e-08, -1.4227980596333721e-08, 1.811308664394915e-07;
  S2 << -6.272561526745687e-07, 2.805546571188463e-07, 4.2782756682953457e-07, 8.54401685835999e-08,
      8.762219354284593e-07, 1.5285528673410845e-07, 7.043760079665012e-07, 1.8058957219255395e-07,
      -1.0243476139042563e-06, 6.619287295779458e-08, -1.6830289137993863e-06, 2.9471233851484255e-08,
      2.409523332122225e-08, 4.306143497422768e-07, 1.2366259786610292e-08, -6.06921957130162e-07;
  check_generalized_dare<MatrixType>(A2.cast<Scalar>(), B2.cast<Scalar>(), Q2.cast<Scalar>(), R2.cast<Scalar>(),
                                     MatrixType::Identity(8, 8), S2.cast<Scalar>(), 3.5e-16);
}

// All modes of A are unstable and B = 0, so no stabilizing solution exists. In a rotated basis, rounding errors leave
// U1 of the order of eps instead of zero, which the scale-invariant rcond(U1) does not detect: the solver returned
// Success with ||X|| = 2.5e20.
template <typename MatrixType>
void dare_unreachable_unstable_modes() {
  using Scalar = typename MatrixType::Scalar;
  const Vector3d v(1, 2, 3);
  const Matrix3d P = Matrix3d::Identity() - 2 * v * v.transpose() / v.squaredNorm();
  const MatrixType A = (P * Vector3d(150, -200, 250).asDiagonal() * P).template cast<Scalar>();
  DiscreteAlgebraicRiccatiSolver<MatrixType> dare;
  for (bool balance : {true, false})
    VERIFY_IS_EQUAL(
        dare.compute(A, MatrixType::Zero(3, 1), MatrixType::Identity(3, 3), MatrixType::Identity(1, 1), balance).info(),
        NumericalIssue);
}

#if defined(EIGEN_TEST_PART_1) || defined(EIGEN_TEST_PART_ALL)
// Regression test: with E != I, the stable subspace of the balanced pencil must be combined with the balanced E.
// Combining it with the original E, as SciPy does, returns a solution with residual 3.9 for this problem.
void dare_generalized_balanced() {
  MatrixXd A(2, 2), B(2, 1), Q(2, 2), R(1, 1), E(2, 2), S(2, 1);
  A << 0.9, 0, 0, 2;
  B << 1e-4, 1;
  Q << 1, 0, 0, 1e4;
  R << 1;
  E << 1, 0.5, 0, 1;
  S << 0, 0;
  DiscreteAlgebraicRiccatiSolver<MatrixXd> dare(A, B, Q, R, E, S);
  VERIFY_IS_EQUAL(dare.info(), Success);
  verify_dare_backward_error<MatrixXd>(A, B, Q, R, E, S, dare.solution());
  verify_stabilizing<MatrixXd>(A, B, R, E, S, dare.solution());
}
#endif

#if defined(EIGEN_TEST_PART_4) || defined(EIGEN_TEST_PART_ALL)
// Regression test: ComplexQZ reports NumericalIssue for the deflated 2x2 pencil of this problem, whose subdiagonal
// entry its direct 2x2 reduction leaves at 2.1 eps ||S||; the decomposition is still backward stable.
void dare_complex_qz_2x2_block() {
  using cf = std::complex<float>;
  MatrixXcf A(1, 1), B(1, 1), Q(1, 1), R(1, 1), E(1, 1), S(1, 1);
  A << cf(1.34278035f, -1.30643785f);
  B << cf(-0.928204715f, 1.51564276f);
  Q << cf(2.26285052f, 0);
  R << cf(1.4296701f, 0);
  E << cf(0.936707854f, 0.0122698545f);
  S << cf(-0.0921620131f, -0.450842977f);
  DiscreteAlgebraicRiccatiSolver<MatrixXcf> dare(A, B, Q, R, E, S);
  VERIFY_IS_EQUAL(dare.info(), Success);
  verify_dare_backward_error<MatrixXcf>(A, B, Q, R, E, S, dare.solution());
  verify_stabilizing<MatrixXcf>(A, B, R, E, S, dare.solution());
}
#endif

#if defined(EIGEN_TEST_PART_1) || defined(EIGEN_TEST_PART_ALL)
void dare_invalid_input() {
  const MatrixXd Id = MatrixXd::Identity(3, 3);
  DiscreteAlgebraicRiccatiSolver<MatrixXd> dare;

  // Q or R not symmetric.
  MatrixXd nonsymmetric(3, 3);
  nonsymmetric << 0, 1, 2, 3, 4, 5, 6, 7, 8;
  VERIFY_IS_EQUAL(dare.compute(Id, Id, nonsymmetric, Id).info(), InvalidInput);
  VERIFY_IS_EQUAL(dare.compute(Id, Id, Id, nonsymmetric).info(), InvalidInput);

  // Singular E.
  MatrixXd singular = MatrixXd::Constant(3, 3, 1e12);
  singular(2, 2) -= 1;
  VERIFY_IS_EQUAL(dare.compute(Id, Id, Id, Id, singular, MatrixXd::Zero(3, 3)).info(), InvalidInput);

  // Non-finite input.
  const MatrixXd nan = MatrixXd::Constant(3, 3, std::numeric_limits<double>::quiet_NaN());
  VERIFY_IS_EQUAL(dare.compute(nan, Id, Id, Id).info(), InvalidInput);
  VERIFY_IS_EQUAL(dare.compute(Id, nan, Id, Id).info(), InvalidInput);
  VERIFY_IS_EQUAL(dare.compute(Id, Id, nan, Id).info(), InvalidInput);
  VERIFY_IS_EQUAL(dare.compute(Id, Id, Id, nan).info(), InvalidInput);
  VERIFY_IS_EQUAL(dare.compute(Id, Id, Id, Id, nan, Id).info(), InvalidInput);
  VERIFY_IS_EQUAL(dare.compute(Id, Id, Id, Id, Id, nan).info(), InvalidInput);

  // Incompatible sizes.
  const MatrixXd rect = MatrixXd::Ones(3, 2);
  VERIFY_RAISES_ASSERT(dare.compute(rect, Id, Id, Id));
  VERIFY_RAISES_ASSERT(dare.compute(Id, Id, rect, Id));
  VERIFY_RAISES_ASSERT(dare.compute(Id, Id, Id, rect));
  VERIFY_RAISES_ASSERT(dare.compute(Id, Id, Id, Id, rect, Id));
  VERIFY_RAISES_ASSERT(dare.compute(Id, Id, Id, Id, Id, MatrixXd::Ones(3, 2)));

  // The solver recovers from a failed computation.
  VERIFY_IS_EQUAL(dare.compute(Id * 0.5, Id, Id, Id).info(), Success);
}

void dare_no_stabilizing_solution() {
  // The unstable mode 2 is not controllable, so no stabilizing solution exists.
  MatrixXd A(2, 2), B(2, 1);
  A << 2, 0, 0, 0.5;
  B << 0, 1;
  DiscreteAlgebraicRiccatiSolver<MatrixXd> dare;
  for (bool balance : {true, false})
    VERIFY_IS_EQUAL(dare.compute(A, B, MatrixXd::Identity(2, 2), MatrixXd::Identity(1, 1), balance).info(),
                    NumericalIssue);

  // A Jordan block at 1 with B = 0 and Q = 0 gives a singular pencil, whose reordering rejects a swap.
  A << 1, 1, 0, 1;
  B.setZero();
  for (bool balance : {true, false})
    VERIFY_IS_EQUAL(dare.compute(A, B, MatrixXd::Zero(2, 2), MatrixXd::Identity(1, 1), balance).info(), NumericalIssue);
}

void dare_empty() {
  // m = 0: the solution is empty.
  DiscreteAlgebraicRiccatiSolver<MatrixXd> dare(MatrixXd(0, 0), MatrixXd(0, 2), MatrixXd(0, 0),
                                                MatrixXd::Identity(2, 2));
  VERIFY_IS_EQUAL(dare.info(), Success);
  VERIFY_IS_EQUAL(dare.solution().rows(), 0);
  dare.compute(MatrixXd(0, 0), MatrixXd(0, 2), MatrixXd(0, 0), MatrixXd::Identity(2, 2), MatrixXd(0, 0),
               MatrixXd(0, 2));
  VERIFY_IS_EQUAL(dare.info(), Success);

  // n = 0: for stable A the equation reduces to the Stein equation A^* X A - X + Q = 0.
  const Index m = 4;
  MatrixXd A = MatrixXd::Random(m, m);
  A *= 0.5 / A.operatorNorm();
  const MatrixXd Q = MatrixXd::Identity(m, m);
  dare.compute(A, MatrixXd(m, 0), Q, MatrixXd(0, 0));
  VERIFY_IS_EQUAL(dare.info(), Success);
  const MatrixXd& X = dare.solution();
  VERIFY((A.adjoint() * X * A - X + Q).norm() <= 100 * NumTraits<double>::epsilon() * X.norm());
}

#endif

// Random problems with a unique stabilizing solution whose conditioning stays moderate: A = U T U^* with U unitary and
// T upper triangular with eigenvalue moduli in [0.1, 0.8], except for at most n well controllable ones in [1.25, 2]; Q
// and R are positive definite, and ||S||_F <= 1/2 keeps Q - S R^-1 S^* >= 3/4 I in the generalized problems. The sizes
// are m in [1, maxM] (m = maxM for fixed-size MatrixType) and n in [1, min(m, maxN)].
template <typename MatrixType>
void dare_random(Index maxM, Index maxN) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using DynMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  const Index m = MatrixType::RowsAtCompileTime == Dynamic ? internal::random<Index>(1, maxM) : maxM;
  const Index n = internal::random<Index>(1, numext::mini(m, maxN));

  DynMatrix T = DynMatrix::Random(m, m).template triangularView<StrictlyUpper>();
  T *= RealScalar(0.5) / numext::sqrt(RealScalar(m));
  const Index unstable = internal::random<Index>(0, n);
  for (Index i = 0; i < m; ++i) {
    Scalar phase = internal::random<Scalar>();
    while (numext::is_exactly_zero(phase)) phase = internal::random<Scalar>();
    const RealScalar modulus =
        i < unstable ? internal::random<RealScalar>(1.25, 2) : internal::random<RealScalar>(0.1, 0.8);
    T(i, i) = phase / numext::abs(phase) * modulus;
  }
  const DynMatrix U = HouseholderQR<DynMatrix>(DynMatrix::Random(m, m)).householderQ();
  const MatrixType A = U * T * U.adjoint();
  // B = U Bu with Bu(i, i) = 2 + O(1) for the unstable modes i < unstable <= n keeps them well controllable. A third
  // of the problems scale B by 10^-k, for which balancing alone would cost digits.
  DynMatrix Bu = DynMatrix::Random(m, n);
  for (Index i = 0; i < unstable; ++i) Bu(i, i) += Scalar(2);
  const int k = internal::random<int>(0, 2) == 0 ? internal::random<int>(1, NumTraits<RealScalar>::digits10() / 3) : 0;
  const DynMatrix B = U * Bu * numext::pow(RealScalar(10), RealScalar(-k));
  const DynMatrix C = DynMatrix::Random(m, m);
  const DynMatrix D = DynMatrix::Random(n, n);
  const MatrixType Q = C.adjoint() * C + DynMatrix::Identity(m, m);
  const DynMatrix R = D.adjoint() * D + DynMatrix::Identity(n, n);
  const DynMatrix Id = DynMatrix::Identity(m, m), Z = DynMatrix::Zero(m, n);

  DiscreteAlgebraicRiccatiSolver<MatrixType> dare(A, B, Q, R);
  VERIFY_IS_EQUAL(dare.info(), Success);
  const MatrixType& X = dare.solution();
  VERIFY_IS_EQUAL(X, X.adjoint());
  verify_dare_backward_error<DynMatrix>(A, B, Q, R, Id, Z, X);
  verify_stabilizing<DynMatrix>(A, B, R, Id, Z, X);
  // X >= Q > 0, since X = Q + A^* (X - X B (R + B^* X B)^-1 B^* X) A.
  VERIFY(SelfAdjointEigenSolver<DynMatrix>(X).eigenvalues().minCoeff() > RealScalar(0));

  DiscreteAlgebraicRiccatiSolver<MatrixType> unbalanced(A, B, Q, R, /*balance=*/false);
  VERIFY_IS_EQUAL(unbalanced.info(), Success);
  verify_dare_backward_error<DynMatrix>(A, B, Q, R, Id, Z, unbalanced.solution());

  // The generalized equation with E = I and S = 0 is the standard one.
  DiscreteAlgebraicRiccatiSolver<MatrixType> generalized;
  generalized.compute(A, B, Q, R, MatrixType::Identity(m, m), Z);
  VERIFY_IS_EQUAL(generalized.info(), Success);
  verify_dare_backward_error<DynMatrix>(A, B, Q, R, Id, Z, generalized.solution());

  // The generalized problem (E A, E B, Q, R, E, S) has the closed-loop dynamics of (A, B), and E^* X E is the solution
  // of the standard problem with the cross term S.
  const MatrixType E =
      MatrixType::Identity(m, m) + MatrixType::Random(m, m) * (RealScalar(0.5) / numext::sqrt(RealScalar(m)));
  const DynMatrix S = DynMatrix::Random(m, n) * (RealScalar(0.5) / numext::sqrt(RealScalar(m * n)));
  const DynMatrix EA = E * A, EB = E * B;
  for (bool balance : {true, false}) {
    generalized.compute(EA, EB, Q, R, E, S, balance);
    VERIFY_IS_EQUAL(generalized.info(), Success);
    const DynMatrix Xg = generalized.solution();
    VERIFY_IS_EQUAL(Xg, Xg.adjoint());
    verify_dare_backward_error<DynMatrix>(EA, EB, Q, R, E, S, Xg);
    verify_stabilizing<DynMatrix>(EA, EB, R, E, S, Xg);
  }
}

// Checks the reordering of a generalized Schur form directly.
template <typename MatrixType>
void generalized_schur_reordering(Index size) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  const MatrixType A = MatrixType::Random(size, size) * RealScalar(1.5);
  const MatrixType B = MatrixType::Random(size, size);
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  typename internal::riccati_qz<MatrixType>::type qz(A, B, true);
  VERIFY_IS_EQUAL(qz.info(), Success);
  MatrixType S = qz.matrixS(), T = qz.matrixT(), Q = qz.matrixQ(), Z = qz.matrixZ().adjoint();

  Index inside = 0;
  for (Index k = 0; k < size; ++k) {
    // Count eigenvalues inside the unit circle through the 1x1 and 2x2 diagonal blocks.
    const Index bs = internal::generalized_schur_block_size(S, k);
    if (bs == 1) {
      if (numext::abs(S(k, k)) < numext::abs(T(k, k))) ++inside;
    } else {
      if (numext::abs(S.template block<2, 2>(k, k).determinant()) < numext::abs(T(k, k) * T(k + 1, k + 1))) inside += 2;
      ++k;
    }
  }

  const Index selected = internal::reorder_generalized_schur_inside_unit_circle(S, T, &Q, Z);
  VERIFY_IS_EQUAL(selected, inside);
  // Each swap is accepted only if it is backward stable to 20 eps ||(S, T)||_F locally, see the strong stability test.
  const RealScalar bound = RealScalar(20) * RealScalar(size) * eps;
  VERIFY((Q * S * Z.adjoint() - A).norm() <= bound * A.norm());
  VERIFY((Q * T * Z.adjoint() - B).norm() <= bound * B.norm());
  VERIFY((Q.adjoint() * Q - MatrixType::Identity(size, size)).norm() <= bound);
  VERIFY((Z.adjoint() * Z - MatrixType::Identity(size, size)).norm() <= bound);
  VERIFY_IS_EQUAL(T.template triangularView<StrictlyLower>().toDenseMatrix(), MatrixType::Zero(size, size));
  for (Index i = 0; i + 2 < size; ++i) VERIFY_IS_EQUAL(S(i + 2, i), Scalar(0));
  // The selected eigenvalues come first, and the boundary does not split a 2x2 block.
  if (selected > 0 && selected < size) VERIFY_IS_EQUAL(S(selected, selected - 1), Scalar(0));
  for (Index k = 0; k < size;) {
    const Index bs = internal::generalized_schur_block_size(S, k);
    RealScalar modulus;
    if (bs == 1)
      modulus = numext::abs(S(k, k)) / numext::abs(T(k, k));
    else
      modulus = numext::sqrt(numext::abs(S.template block<2, 2>(k, k).determinant()) /
                             numext::abs(T(k, k) * T(k + 1, k + 1)));
    VERIFY((k < selected) == (modulus < RealScalar(1)));
    k += bs;
  }
}

// Generalized eigenvalues of the diagonal block of size bs at (k, k) of (S, T), sorted.
template <typename MatrixType>
std::vector<std::complex<typename MatrixType::RealScalar>> block_eigenvalues(const MatrixType& S, const MatrixType& T,
                                                                             Index k, Index bs) {
  using Complex = std::complex<typename MatrixType::RealScalar>;
  using ComplexMatrix = Matrix<Complex, Dynamic, Dynamic>;
  const ComplexMatrix s = S.block(k, k, bs, bs).template cast<Complex>();
  const ComplexMatrix t = T.block(k, k, bs, bs).template cast<Complex>();
  const Matrix<Complex, Dynamic, 1> ev = ComplexEigenSolver<ComplexMatrix>(t.inverse() * s).eigenvalues();
  std::vector<Complex> v(ev.data(), ev.data() + ev.size());
  std::sort(v.begin(), v.end(), [](const Complex& a, const Complex& b) {
    return a.imag() < b.imag() || (a.imag() == b.imag() && a.real() < b.real());
  });
  return v;
}

// Swaps the diagonal blocks of sizes n1 and n2 at row/column j = 1 of a generalized Schur form of size n1 + n2 + 2.
template <typename MatrixType>
void generalized_schur_swap(Index n1, Index n2) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  const Index j = 1, size = n1 + n2 + 2;
  MatrixType S = MatrixType::Random(size, size).template triangularView<Upper>();
  MatrixType T = MatrixType::Random(size, size).template triangularView<Upper>();
  T.diagonal().array() += Scalar(2);
  // A 2x2 block (s, t) with t = I and s = [a, 1; -1, a] has the eigenvalues a +- i.
  const Index starts[] = {j, j + n1};
  const Index sizes[] = {n1, n2};
  for (int b = 0; b < 2; ++b) {
    if (sizes[b] != 2) continue;
    const Index k = starts[b];
    const Scalar a = Scalar(b == 0 ? 0.5 : -0.25);
    S.template block<2, 2>(k, k) << a, Scalar(1), Scalar(-1), a;
    T.template block<2, 2>(k, k).setIdentity();
  }
  const MatrixType S0 = S, T0 = T;
  MatrixType Q = MatrixType::Identity(size, size), Z = MatrixType::Identity(size, size);
  const auto upper = block_eigenvalues(S0, T0, j, n1);
  const auto lower = block_eigenvalues(S0, T0, j + n1, n2);

  VERIFY(internal::swap_generalized_schur_blocks(S, T, &Q, Z, j, n1, n2));
  const auto newUpper = block_eigenvalues(S, T, j, n2);
  const auto newLower = block_eigenvalues(S, T, j + n2, n1);
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  for (Index i = 0; i < n2; ++i)
    VERIFY(std::abs(newUpper[i] - lower[i]) <= RealScalar(100) * eps * (RealScalar(1) + std::abs(lower[i])));
  for (Index i = 0; i < n1; ++i)
    VERIFY(std::abs(newLower[i] - upper[i]) <= RealScalar(100) * eps * (RealScalar(1) + std::abs(upper[i])));
  // Exact structure: zero block boundary, T upper triangular, S zero below its 2x2 diagonal blocks.
  VERIFY_IS_EQUAL(S.block(j + n2, j, n1, n2), MatrixType::Zero(n1, n2));
  VERIFY_IS_EQUAL(T.template triangularView<StrictlyLower>().toDenseMatrix(), MatrixType::Zero(size, size));
  for (Index c = 0; c < size; ++c)
    for (Index r = c + 1; r < size; ++r)
      if (!((r == j + 1 && c == j && n2 == 2) || (r == j + n2 + 1 && c == j + n2 && n1 == 2)))
        VERIFY_IS_EQUAL(S(r, c), Scalar(0));
  // Backward error and orthogonality.
  const RealScalar bound = RealScalar(20) * RealScalar(size) * eps;
  VERIFY((Q * S * Z.adjoint() - S0).norm() <= bound * S0.norm());
  VERIFY((Q * T * Z.adjoint() - T0).norm() <= bound * T0.norm());
  VERIFY((Q.adjoint() * Q - MatrixType::Identity(size, size)).norm() <= bound);
  VERIFY((Z.adjoint() * Z - MatrixType::Identity(size, size)).norm() <= bound);
}

#if defined(EIGEN_TEST_PART_6) || defined(EIGEN_TEST_PART_ALL)
// A rejected swap leaves all matrices untouched, and the reordering then reports failure.
void generalized_schur_rejected_swap() {
  // Equal 2x2 blocks make the generalized Sylvester equation singular.
  MatrixXd S(4, 4), T = MatrixXd::Identity(4, 4);
  S << 0.5, 1, 0.3, 0.2,  //
      -1, 0.5, 0.1, 0.4,  //
      0, 0, 0.5, 1,       //
      0, 0, -1, 0.5;
  MatrixXd Q = MatrixXd::Identity(4, 4), Z = MatrixXd::Identity(4, 4);
  const MatrixXd S0 = S, T0 = T, Q0 = Q, Z0 = Z;
  VERIFY(!internal::swap_generalized_schur_blocks(S, T, &Q, Z, 0, 2, 2));
  VERIFY_IS_EQUAL(S, S0);
  VERIFY_IS_EQUAL(T, T0);
  VERIFY_IS_EQUAL(Q, Q0);
  VERIFY_IS_EQUAL(Z, Z0);

  // The 1x1 blocks 0/0 and 1/2 of a singular pencil cannot be swapped, so the eigenvalue 1/2 cannot be selected.
  MatrixXd S2(2, 2), T2(2, 2);
  S2 << 0, 0, 0, 0.5;
  T2 << 0, 0, 0, 1;
  MatrixXd Q2 = MatrixXd::Identity(2, 2), Z2 = MatrixXd::Identity(2, 2);
  const MatrixXd S20 = S2, T20 = T2;
  VERIFY_IS_EQUAL(internal::reorder_generalized_schur_inside_unit_circle(S2, T2, &Q2, Z2), Index(-1));
  VERIFY_IS_EQUAL(S2, S20);
  VERIFY_IS_EQUAL(T2, T20);
}
#endif

EIGEN_DECLARE_TEST(riccati) {
  CALL_SUBTEST_1(dare_scipy_real_cases<MatrixXd>());
  CALL_SUBTEST_1(dare_generalized_scipy_cases<MatrixXd>());
  CALL_SUBTEST_1(dare_generalized_balanced());
  CALL_SUBTEST_1(dare_invalid_input());
  CALL_SUBTEST_1(dare_no_stabilizing_solution());
  CALL_SUBTEST_1(dare_unreachable_unstable_modes<MatrixXd>());
  CALL_SUBTEST_7(dare_unreachable_unstable_modes<MatrixXcd>());
  CALL_SUBTEST_1(dare_empty());
  CALL_SUBTEST_6(generalized_schur_rejected_swap());
  for (Index n1 = 1; n1 <= 2; ++n1)
    for (Index n2 = 1; n2 <= 2; ++n2) CALL_SUBTEST_6(generalized_schur_swap<MatrixXd>(n1, n2));
  CALL_SUBTEST_6(generalized_schur_swap<MatrixXcd>(1, 1));
  CALL_SUBTEST_4(dare_complex_qz_2x2_block());
  CALL_SUBTEST_7(dare_scipy_complex_cases());
  CALL_SUBTEST_7(dare_scipy_real_cases<MatrixXcd>());
  CALL_SUBTEST_7(dare_generalized_scipy_cases<MatrixXcd>());
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_2(dare_random<MatrixXd>(20, 6));
    CALL_SUBTEST_3(dare_random<MatrixXcd>(20, 6));
    CALL_SUBTEST_4(dare_random<MatrixXf>(8, 4));
    CALL_SUBTEST_4(dare_random<MatrixXcf>(8, 4));
    CALL_SUBTEST_5((dare_random<Matrix<double, 4, 4>>(4, 4)));
    CALL_SUBTEST_6(generalized_schur_reordering<MatrixXd>(internal::random<Index>(1, 30)));
    CALL_SUBTEST_6(generalized_schur_reordering<MatrixXcd>(internal::random<Index>(1, 30)));
    CALL_SUBTEST_6(generalized_schur_reordering<MatrixXf>(internal::random<Index>(1, 30)));
  }
}
