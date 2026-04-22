// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_GEMM_THREADPOOL
#include "main.h"

void test_parallelize_gemm() {
  constexpr int n = 1024;
  constexpr int num_threads = 4;
  MatrixXf a = MatrixXf::Random(n, n);
  MatrixXf b = MatrixXf::Random(n, n);
  MatrixXf c = MatrixXf::Random(n, n);
  c.noalias() = a * b;

  ThreadPool pool(num_threads);
  Eigen::setGemmThreadPool(&pool);
  MatrixXf c_threaded(n, n);
  c_threaded.noalias() = a * b;

  VERIFY_IS_APPROX(c, c_threaded);
  Eigen::setGemmThreadPool(nullptr);
}

void test_parallelize_gemm_varied() {
  constexpr int num_threads = 4;
  ThreadPool pool(num_threads);

  // Non-square float
  {
    MatrixXf a = MatrixXf::Random(512, 2048);
    MatrixXf b = MatrixXf::Random(2048, 256);
    MatrixXf c_serial(512, 256);
    c_serial.noalias() = a * b;
    Eigen::setGemmThreadPool(&pool);
    MatrixXf c_threaded(512, 256);
    c_threaded.noalias() = a * b;
    Eigen::setGemmThreadPool(nullptr);
    VERIFY_IS_APPROX(c_serial, c_threaded);
  }

  // Double
  {
    MatrixXd a = MatrixXd::Random(512, 512);
    MatrixXd b = MatrixXd::Random(512, 512);
    MatrixXd c_serial(512, 512);
    c_serial.noalias() = a * b;
    Eigen::setGemmThreadPool(&pool);
    MatrixXd c_threaded(512, 512);
    c_threaded.noalias() = a * b;
    Eigen::setGemmThreadPool(nullptr);
    VERIFY_IS_APPROX(c_serial, c_threaded);
  }

  // Complex double
  {
    MatrixXcd a = MatrixXcd::Random(256, 256);
    MatrixXcd b = MatrixXcd::Random(256, 256);
    MatrixXcd c_serial(256, 256);
    c_serial.noalias() = a * b;
    Eigen::setGemmThreadPool(&pool);
    MatrixXcd c_threaded(256, 256);
    c_threaded.noalias() = a * b;
    Eigen::setGemmThreadPool(nullptr);
    VERIFY_IS_APPROX(c_serial, c_threaded);
  }
}

EIGEN_DECLARE_TEST(product_threaded) {
  CALL_SUBTEST_1(test_parallelize_gemm());
  CALL_SUBTEST_2(test_parallelize_gemm_varied());
}
