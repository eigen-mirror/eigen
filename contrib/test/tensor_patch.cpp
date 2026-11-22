// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <Eigen/Tensor>

using Eigen::Tensor;

template <int DataLayout>
static void test_simple_patch() {
  Tensor<float, 4, DataLayout> tensor(2, 3, 5, 7);
  tensor.setRandom();
  array<ptrdiff_t, 4> patch_dims;

  patch_dims[0] = 1;
  patch_dims[1] = 1;
  patch_dims[2] = 1;
  patch_dims[3] = 1;

  Tensor<float, 5, DataLayout> no_patch;
  no_patch = tensor.extract_patches(patch_dims);

  if (DataLayout == ColMajor) {
    VERIFY_IS_EQUAL(no_patch.dimension(0), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(1), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(2), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(3), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(4), tensor.size());
  } else {
    VERIFY_IS_EQUAL(no_patch.dimension(0), tensor.size());
    VERIFY_IS_EQUAL(no_patch.dimension(1), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(2), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(3), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(4), 1);
  }

  for (int i = 0; i < tensor.size(); ++i) {
    VERIFY_IS_EQUAL(tensor.data()[i], no_patch.data()[i]);
  }

  patch_dims[0] = 2;
  patch_dims[1] = 3;
  patch_dims[2] = 5;
  patch_dims[3] = 7;
  Tensor<float, 5, DataLayout> single_patch;
  single_patch = tensor.extract_patches(patch_dims);

  if (DataLayout == ColMajor) {
    VERIFY_IS_EQUAL(single_patch.dimension(0), 2);
    VERIFY_IS_EQUAL(single_patch.dimension(1), 3);
    VERIFY_IS_EQUAL(single_patch.dimension(2), 5);
    VERIFY_IS_EQUAL(single_patch.dimension(3), 7);
    VERIFY_IS_EQUAL(single_patch.dimension(4), 1);
  } else {
    VERIFY_IS_EQUAL(single_patch.dimension(0), 1);
    VERIFY_IS_EQUAL(single_patch.dimension(1), 2);
    VERIFY_IS_EQUAL(single_patch.dimension(2), 3);
    VERIFY_IS_EQUAL(single_patch.dimension(3), 5);
    VERIFY_IS_EQUAL(single_patch.dimension(4), 7);
  }

  for (int i = 0; i < tensor.size(); ++i) {
    VERIFY_IS_EQUAL(tensor.data()[i], single_patch.data()[i]);
  }

  patch_dims[0] = 1;
  patch_dims[1] = 2;
  patch_dims[2] = 2;
  patch_dims[3] = 1;
  Tensor<float, 5, DataLayout> twod_patch;
  twod_patch = tensor.extract_patches(patch_dims);

  if (DataLayout == ColMajor) {
    VERIFY_IS_EQUAL(twod_patch.dimension(0), 1);
    VERIFY_IS_EQUAL(twod_patch.dimension(1), 2);
    VERIFY_IS_EQUAL(twod_patch.dimension(2), 2);
    VERIFY_IS_EQUAL(twod_patch.dimension(3), 1);
    VERIFY_IS_EQUAL(twod_patch.dimension(4), 2 * 2 * 4 * 7);
  } else {
    VERIFY_IS_EQUAL(twod_patch.dimension(0), 2 * 2 * 4 * 7);
    VERIFY_IS_EQUAL(twod_patch.dimension(1), 1);
    VERIFY_IS_EQUAL(twod_patch.dimension(2), 2);
    VERIFY_IS_EQUAL(twod_patch.dimension(3), 2);
    VERIFY_IS_EQUAL(twod_patch.dimension(4), 1);
  }

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 4; ++k) {
        for (int l = 0; l < 7; ++l) {
          int patch_loc;
          if (DataLayout == ColMajor) {
            patch_loc = i + 2 * (j + 2 * (k + 4 * l));
          } else {
            patch_loc = l + 7 * (k + 4 * (j + 2 * i));
          }
          for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 2; ++y) {
              if (DataLayout == ColMajor) {
                VERIFY_IS_EQUAL(tensor(i, j + x, k + y, l), twod_patch(0, x, y, 0, patch_loc));
              } else {
                VERIFY_IS_EQUAL(tensor(i, j + x, k + y, l), twod_patch(patch_loc, 0, x, y, 0));
              }
            }
          }
        }
      }
    }
  }

  patch_dims[0] = 1;
  patch_dims[1] = 2;
  patch_dims[2] = 3;
  patch_dims[3] = 5;
  Tensor<float, 5, DataLayout> threed_patch;
  threed_patch = tensor.extract_patches(patch_dims);

  if (DataLayout == ColMajor) {
    VERIFY_IS_EQUAL(threed_patch.dimension(0), 1);
    VERIFY_IS_EQUAL(threed_patch.dimension(1), 2);
    VERIFY_IS_EQUAL(threed_patch.dimension(2), 3);
    VERIFY_IS_EQUAL(threed_patch.dimension(3), 5);
    VERIFY_IS_EQUAL(threed_patch.dimension(4), 2 * 2 * 3 * 3);
  } else {
    VERIFY_IS_EQUAL(threed_patch.dimension(0), 2 * 2 * 3 * 3);
    VERIFY_IS_EQUAL(threed_patch.dimension(1), 1);
    VERIFY_IS_EQUAL(threed_patch.dimension(2), 2);
    VERIFY_IS_EQUAL(threed_patch.dimension(3), 3);
    VERIFY_IS_EQUAL(threed_patch.dimension(4), 5);
  }

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          int patch_loc;
          if (DataLayout == ColMajor) {
            patch_loc = i + 2 * (j + 2 * (k + 3 * l));
          } else {
            patch_loc = l + 3 * (k + 3 * (j + 2 * i));
          }
          for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 3; ++y) {
              for (int z = 0; z < 5; ++z) {
                if (DataLayout == ColMajor) {
                  VERIFY_IS_EQUAL(tensor(i, j + x, k + y, l + z), threed_patch(0, x, y, z, patch_loc));
                } else {
                  VERIFY_IS_EQUAL(tensor(i, j + x, k + y, l + z), threed_patch(patch_loc, 0, x, y, z));
                }
              }
            }
          }
        }
      }
    }
  }
}

// A reduction operand has no block access, so adding a (zero) reduction to the patches takes the vectorized
// coefficient/packet path instead of block(). A packet that straddles a patch row is not contiguous in the input.
template <int DataLayout>
static void test_patch_packet_path(Index rows, Index cols, Index patch_rows, Index patch_cols) {
  Tensor<float, 2, DataLayout> input(rows, cols);
  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) input(i, j) = static_cast<float>(i + rows * j);
  }
  const Index patches_per_col = rows - patch_rows + 1;
  const Index num_patches = patches_per_col * (cols - patch_cols + 1);

  Eigen::array<Index, 2> patch_dims{{patch_rows, patch_cols}};
  Tensor<float, 3, DataLayout> result;
  if (DataLayout == ColMajor) {
    Tensor<float, 4, DataLayout> zeros(patch_rows, patch_cols, num_patches, 2);
    zeros.setZero();
    result = input.extract_patches(patch_dims) + zeros.sum(Eigen::array<int, 1>{{3}});
  } else {
    Tensor<float, 4, DataLayout> zeros(num_patches, patch_rows, patch_cols, 2);
    zeros.setZero();
    result = input.extract_patches(patch_dims) + zeros.sum(Eigen::array<int, 1>{{3}});
  }

  for (Index p = 0; p < num_patches; ++p) {
    // Patches are enumerated with the first input dimension varying fastest (ColMajor) or slowest (RowMajor).
    const Index patch_row = DataLayout == ColMajor ? p % patches_per_col : p / (cols - patch_cols + 1);
    const Index patch_col = DataLayout == ColMajor ? p / patches_per_col : p % (cols - patch_cols + 1);
    for (Index c = 0; c < patch_cols; ++c) {
      for (Index r = 0; r < patch_rows; ++r) {
        const float expected = input(patch_row + r, patch_col + c);
        VERIFY_IS_EQUAL(DataLayout == ColMajor ? result(r, c, p) : result(p, r, c), expected);
      }
    }
  }
}

template <int DataLayout>
static void test_patch_packet_path() {
  // Packets of 2, 4, 8 and 16 coefficients straddle the patch rows of these shapes.
  test_patch_packet_path<DataLayout>(4, 4, 1, 2);
  test_patch_packet_path<DataLayout>(2, 4, 1, 2);
  test_patch_packet_path<DataLayout>(4, 4, 2, 2);
  test_patch_packet_path<DataLayout>(7, 9, 3, 2);
  test_patch_packet_path<DataLayout>(8, 8, 1, 4);
  test_patch_packet_path<DataLayout>(9, 7, 4, 3);
  test_patch_packet_path<DataLayout>(16, 5, 2, 5);
  for (int i = 0; i < 10; ++i) {
    const Index rows = internal::random<Index>(1, 20);
    const Index cols = internal::random<Index>(1, 20);
    test_patch_packet_path<DataLayout>(rows, cols, internal::random<Index>(1, rows), internal::random<Index>(1, cols));
  }
}

// Three-dimensional input, so that the index decomposition in packet() runs more than one reduction step.
template <int DataLayout>
static void test_patch_packet_path_3d() {
  const Index d0 = 5, d1 = 6, d2 = 4;
  const Index p0 = 2, p1 = 3, p2 = 2;
  Tensor<float, 3, DataLayout> input(d0, d1, d2);
  for (Index i = 0; i < input.size(); ++i) input.data()[i] = static_cast<float>(i);
  Eigen::array<Index, 3> patch_dims{{p0, p1, p2}};
  const Index n0 = d0 - p0 + 1, n1 = d1 - p1 + 1, n2 = d2 - p2 + 1;
  const Index num_patches = n0 * n1 * n2;
  Tensor<float, 4, DataLayout> result;
  if (DataLayout == ColMajor) {
    Tensor<float, 5, DataLayout> zeros(p0, p1, p2, num_patches, 2);
    zeros.setZero();
    result = input.extract_patches(patch_dims) + zeros.sum(Eigen::array<int, 1>{{4}});
  } else {
    Tensor<float, 5, DataLayout> zeros(num_patches, p0, p1, p2, 2);
    zeros.setZero();
    result = input.extract_patches(patch_dims) + zeros.sum(Eigen::array<int, 1>{{4}});
  }
  for (Index p = 0; p < num_patches; ++p) {
    const Index a = DataLayout == ColMajor ? p % n0 : p / (n1 * n2);
    const Index b = DataLayout == ColMajor ? (p / n0) % n1 : (p / n2) % n1;
    const Index c = DataLayout == ColMajor ? p / (n0 * n1) : p % n2;
    for (Index k = 0; k < p2; ++k)
      for (Index j = 0; j < p1; ++j)
        for (Index i = 0; i < p0; ++i) {
          const float expected = input(a + i, b + j, c + k);
          VERIFY_IS_EQUAL(DataLayout == ColMajor ? result(i, j, k, p) : result(p, i, j, k), expected);
        }
  }
}

EIGEN_DECLARE_TEST(tensor_patch) {
  CALL_SUBTEST(test_simple_patch<ColMajor>());
  CALL_SUBTEST(test_simple_patch<RowMajor>());
  CALL_SUBTEST(test_patch_packet_path<ColMajor>());
  CALL_SUBTEST(test_patch_packet_path<RowMajor>());
  CALL_SUBTEST(test_patch_packet_path_3d<ColMajor>());
  CALL_SUBTEST(test_patch_packet_path_3d<RowMajor>());
  //   CALL_SUBTEST(test_expr_shuffling());
}
