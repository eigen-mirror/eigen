// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// A Schroedinger-type problem (-(u_xx + u_yy) + V(x, y) u = f) whose potential
// V is not separable, so the matrix L + diag(V) is no Kronecker sum. The
// separable part is still exploited twice: L.sparseView() assembles the
// sparse matrix in one expression, and the fast Poisson solver for
// L + mean(V) I preconditions conjugate gradients.

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <iostream>

using namespace Eigen;

SparseMatrix<double> secondDifference(int n) {
  const double h = 1.0 / (n + 1);
  SparseMatrix<double> D(n, n);
  D.reserve(VectorXi::Constant(n, 3));  // at most 3 nonzeros per column
  for (int i = 0; i < n; ++i) {
    D.insert(i, i) = 2 / (h * h);
    if (i > 0) D.insert(i, i - 1) = -1 / (h * h);
    if (i + 1 < n) D.insert(i, i + 1) = -1 / (h * h);
  }
  D.makeCompressed();
  return D;
}

using Laplacian = KroneckerSum<SparseMatrix<double>, SparseMatrix<double>>;

// Applies (L + c I)^{-1} through BartelsStewart: the preconditioner interface
// the iterative solvers expect.
class SeparablePreconditioner {
 public:
  void setOperator(const Laplacian& shifted) { m_solver.compute(shifted); }
  template <typename MatrixType>
  SeparablePreconditioner& analyzePattern(const MatrixType&) {
    return *this;
  }
  template <typename MatrixType>
  SeparablePreconditioner& factorize(const MatrixType&) {
    return *this;
  }
  template <typename MatrixType>
  SeparablePreconditioner& compute(const MatrixType&) {
    return *this;
  }
  template <typename Rhs>
  VectorXd solve(const MatrixBase<Rhs>& b) const {
    return m_solver.solve(b);
  }
  ComputationInfo info() const { return m_solver.info(); }

 private:
  BartelsStewart<Laplacian> m_solver;
};

int main() {
  const int n = 128;
  const double h = 1.0 / (n + 1);
  const SparseMatrix<double> D = secondDifference(n);
  const Laplacian L = makeKroneckerSum(D, D);

  // A Gaussian well, centered off the grid's symmetry axes.
  VectorXd V(n * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 1) * h - 0.4, y = (j + 1) * h - 0.6;
      V[j * n + i] = 2000 * std::exp(-(x * x + y * y) / 0.01);
    }
  SparseMatrix<double> Vd(n * n, n * n);
  Vd.setIdentity();
  Vd = Vd * V.asDiagonal();
  const SparseMatrix<double> H = L.sparseView() + Vd;  // assembled without materializing L
  const VectorXd f = VectorXd::Ones(n * n);

  ConjugateGradient<SparseMatrix<double>, Lower | Upper, IdentityPreconditioner> plain(H);
  ConjugateGradient<SparseMatrix<double>, Lower | Upper, DiagonalPreconditioner<double>> jacobi(H);
  // The preconditioner gets its operator before compute(), which reads its info().
  ConjugateGradient<SparseMatrix<double>, Lower | Upper, SeparablePreconditioner> separable;
  SparseMatrix<double> I(n, n);
  I.setIdentity();
  separable.preconditioner().setOperator(makeKroneckerSum(SparseMatrix<double>(D + V.mean() * I), D));
  separable.compute(H);

  const VectorXd u0 = plain.solve(f), u1 = jacobi.solve(f), u2 = separable.solve(f);
  std::cout << n * n << " unknowns, CG iterations to relative residual " << plain.tolerance() << ":\n";
  std::cout << "  no preconditioner:       " << plain.iterations() << "\n";
  std::cout << "  Jacobi:                  " << jacobi.iterations() << "\n";
  std::cout << "  separable (fast Poisson): " << separable.iterations() << "\n";
  std::cout << "max |u_separable - u_Jacobi| / max |u|: " << (u2 - u1).cwiseAbs().maxCoeff() / u1.cwiseAbs().maxCoeff()
            << "\n";
}
