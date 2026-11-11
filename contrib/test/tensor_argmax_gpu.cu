// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_TEST_NO_LONGDOUBLE

#define EIGEN_USE_GPU

#include "main.h"
#include <contrib/Eigen/Tensor>
#include "tensor_gpu_test_helpers.h"

using Eigen::Tensor;

template <int Layout>
void test_gpu_simple_argmax() {
  Tensor<double, 3, Layout> in(Eigen::array<DenseIndex, 3>{72, 53, 97});
  in.setRandom();
  in *= in.constant(100.0);
  in(0, 0, 0) = -1000.0;
  in(71, 52, 96) = 1000.0;

  tensor_test::GpuTensorTestFixture fixture;
  auto gpu_in = fixture.fromHost(in);
  auto gpu_out_max = fixture.tensor<DenseIndex, 0, Aligned, Layout>();
  auto gpu_out_min = fixture.tensor<DenseIndex, 0, Aligned, Layout>();

  gpu_out_max.device(fixture.device()) = gpu_in.argmax();
  gpu_out_min.device(fixture.device()) = gpu_in.argmin();

  const auto out_max = fixture.toHost(gpu_out_max);
  const auto out_min = fixture.toHost(gpu_out_min);

  VERIFY_IS_EQUAL(out_max(), 72 * 53 * 97 - 1);
  VERIFY_IS_EQUAL(out_min(), 0);
}

template <int DataLayout>
void test_gpu_argmax_dim() {
  Tensor<float, 4, DataLayout> tensor(2, 3, 5, 7);
  std::vector<int> dims;
  dims.push_back(2);
  dims.push_back(3);
  dims.push_back(5);
  dims.push_back(7);

  for (int dim = 0; dim < 4; ++dim) {
    tensor.setRandom();
    tensor = (tensor + tensor.constant(0.5)).log();

    array<DenseIndex, 3> out_shape;
    for (int d = 0; d < 3; ++d) out_shape[d] = (d < dim) ? dims[d] : dims[d + 1];

    Tensor<DenseIndex, 3, DataLayout> tensor_arg(out_shape);

    array<DenseIndex, 4> ix;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 5; ++k) {
          for (int l = 0; l < 7; ++l) {
            ix[0] = i;
            ix[1] = j;
            ix[2] = k;
            ix[3] = l;
            if (ix[dim] != 0) continue;
            // suppose dim == 1, then for all i, k, l, set tensor(i, 0, k, l) = 10.0
            tensor(ix) = 10.0;
          }
        }
      }
    }

    tensor_test::GpuTensorTestFixture fixture;
    auto gpu_in = fixture.fromHost(tensor);
    auto gpu_out = fixture.tensor<DenseIndex, 3, Aligned, DataLayout>(tensor_arg.dimensions());

    gpu_out.device(fixture.device()) = gpu_in.argmax(dim);

    tensor_arg = fixture.toHost(gpu_out);

    VERIFY_IS_EQUAL(tensor_arg.size(), size_t(2 * 3 * 5 * 7 / tensor.dimension(dim)));

    for (DenseIndex n = 0; n < tensor_arg.size(); ++n) {
      // Expect max to be in the first index of the reduced dimension
      VERIFY_IS_EQUAL(tensor_arg.data()[n], 0);
    }

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 5; ++k) {
          for (int l = 0; l < 7; ++l) {
            ix[0] = i;
            ix[1] = j;
            ix[2] = k;
            ix[3] = l;
            if (ix[dim] != tensor.dimension(dim) - 1) continue;
            // suppose dim == 1, then for all i, k, l, set tensor(i, 2, k, l) = 20.0
            tensor(ix) = 20.0;
          }
        }
      }
    }

    fixture.device().memcpyHostToDevice(gpu_in.data(), tensor.data(), tensor.size() * sizeof(float));

    gpu_out.device(fixture.device()) = gpu_in.argmax(dim);

    tensor_arg = fixture.toHost(gpu_out);

    for (DenseIndex n = 0; n < tensor_arg.size(); ++n) {
      // Expect max to be in the last index of the reduced dimension
      VERIFY_IS_EQUAL(tensor_arg.data()[n], tensor.dimension(dim) - 1);
    }
  }
}

template <int DataLayout>
void test_gpu_argmin_dim() {
  Tensor<float, 4, DataLayout> tensor(2, 3, 5, 7);
  std::vector<int> dims;
  dims.push_back(2);
  dims.push_back(3);
  dims.push_back(5);
  dims.push_back(7);

  for (int dim = 0; dim < 4; ++dim) {
    tensor.setRandom();
    tensor = (tensor + tensor.constant(0.5)).log();

    array<DenseIndex, 3> out_shape;
    for (int d = 0; d < 3; ++d) out_shape[d] = (d < dim) ? dims[d] : dims[d + 1];

    Tensor<DenseIndex, 3, DataLayout> tensor_arg(out_shape);

    array<DenseIndex, 4> ix;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 5; ++k) {
          for (int l = 0; l < 7; ++l) {
            ix[0] = i;
            ix[1] = j;
            ix[2] = k;
            ix[3] = l;
            if (ix[dim] != 0) continue;
            // suppose dim == 1, then for all i, k, l, set tensor(i, 0, k, l) = 10.0
            tensor(ix) = -10.0;
          }
        }
      }
    }

    tensor_test::GpuTensorTestFixture fixture;
    auto gpu_in = fixture.fromHost(tensor);
    auto gpu_out = fixture.tensor<DenseIndex, 3, Aligned, DataLayout>(tensor_arg.dimensions());

    gpu_out.device(fixture.device()) = gpu_in.argmin(dim);

    tensor_arg = fixture.toHost(gpu_out);

    VERIFY_IS_EQUAL(tensor_arg.size(), 2 * 3 * 5 * 7 / tensor.dimension(dim));

    for (DenseIndex n = 0; n < tensor_arg.size(); ++n) {
      // Expect min to be in the first index of the reduced dimension
      VERIFY_IS_EQUAL(tensor_arg.data()[n], 0);
    }

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 5; ++k) {
          for (int l = 0; l < 7; ++l) {
            ix[0] = i;
            ix[1] = j;
            ix[2] = k;
            ix[3] = l;
            if (ix[dim] != tensor.dimension(dim) - 1) continue;
            // suppose dim == 1, then for all i, k, l, set tensor(i, 2, k, l) = 20.0
            tensor(ix) = -20.0;
          }
        }
      }
    }

    fixture.device().memcpyHostToDevice(gpu_in.data(), tensor.data(), tensor.size() * sizeof(float));

    gpu_out.device(fixture.device()) = gpu_in.argmin(dim);

    tensor_arg = fixture.toHost(gpu_out);

    for (DenseIndex n = 0; n < tensor_arg.size(); ++n) {
      // Expect max to be in the last index of the reduced dimension
      VERIFY_IS_EQUAL(tensor_arg.data()[n], tensor.dimension(dim) - 1);
    }
  }
}

EIGEN_DECLARE_TEST(tensor_argmax_gpu) {
  CALL_SUBTEST_1(test_gpu_simple_argmax<RowMajor>());
  CALL_SUBTEST_1(test_gpu_simple_argmax<ColMajor>());
  CALL_SUBTEST_2(test_gpu_argmax_dim<RowMajor>());
  CALL_SUBTEST_2(test_gpu_argmax_dim<ColMajor>());
  CALL_SUBTEST_3(test_gpu_argmin_dim<RowMajor>());
  CALL_SUBTEST_3(test_gpu_argmin_dim<ColMajor>());
}
