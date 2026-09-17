// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <Eigen/SparseCholesky>
#include "sparse_factorize_benchmark.h"

template <typename Scalar>
using LLT = Eigen::SimplicialLLT<Eigen::SparseMatrix<Scalar>, Eigen::Lower, Eigen::NaturalOrdering<int>>;
template <typename Scalar>
using LDLT = Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>, Eigen::Lower, Eigen::NaturalOrdering<int>>;

BENCHMARK_TEMPLATE(sparseFactorizeBenchmark, LLT<float>)->ArgsProduct({{128, 512}, {3, 17, 63}});
BENCHMARK_TEMPLATE(sparseFactorizeBenchmark, LLT<double>)->ArgsProduct({{128, 512}, {3, 17, 63}});
BENCHMARK_TEMPLATE(sparseFactorizeBenchmark, LLT<std::complex<double>>)->ArgsProduct({{128, 512}, {3, 17, 63}});
BENCHMARK_TEMPLATE(sparseFactorizeBenchmark, LDLT<double>)->ArgsProduct({{128, 512}, {3, 17, 63}});

using FileLLT = Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>;
using FileLDLT = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;
BENCHMARK_TEMPLATE(sparseFactorizeFileBenchmark, FileLLT)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(sparseFactorizeFileBenchmark, FileLDLT)->Unit(benchmark::kMillisecond);
