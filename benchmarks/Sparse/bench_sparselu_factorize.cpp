// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <Eigen/SparseLU>
#include "sparse_factorize_benchmark.h"

template <typename Scalar>
using LU = Eigen::SparseLU<Eigen::SparseMatrix<Scalar>, Eigen::NaturalOrdering<int>>;

BENCHMARK_TEMPLATE(sparseFactorizeBenchmark, LU<float>)->ArgsProduct({{128, 512}, {3, 17, 63}});
BENCHMARK_TEMPLATE(sparseFactorizeBenchmark, LU<double>)->ArgsProduct({{128, 512}, {3, 17, 63}});
BENCHMARK_TEMPLATE(sparseFactorizeBenchmark, LU<std::complex<double>>)->ArgsProduct({{128, 512}, {3, 17, 63}});

using FileLU = Eigen::SparseLU<Eigen::SparseMatrix<double>>;
BENCHMARK_TEMPLATE(sparseFactorizeFileBenchmark, FileLU)->Unit(benchmark::kMillisecond);
