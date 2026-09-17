// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <Eigen/SparseQR>
#include "sparse_factorize_benchmark.h"

template <typename Scalar>
using QR = Eigen::SparseQR<Eigen::SparseMatrix<Scalar>, Eigen::NaturalOrdering<int>>;

BENCHMARK_TEMPLATE(sparseFactorizeBenchmark, QR<float>)->ArgsProduct({{128, 512}, {3, 17, 63}});
BENCHMARK_TEMPLATE(sparseFactorizeBenchmark, QR<double>)->ArgsProduct({{128, 512}, {3, 17, 63}});
BENCHMARK_TEMPLATE(sparseFactorizeBenchmark, QR<std::complex<double>>)->ArgsProduct({{128, 512}, {3, 17, 63}});

using FileQR = Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>;
BENCHMARK_TEMPLATE(sparseFactorizeFileBenchmark, FileQR, false)->Unit(benchmark::kMillisecond);
