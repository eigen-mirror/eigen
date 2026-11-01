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

#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

using Eigen::Tensor;

template <int DataLayout>
static void test_simple_chip() {
  Tensor<float, 5, DataLayout> tensor(2, 3, 5, 7, 11);
  tensor.setRandom();

  Tensor<float, 4, DataLayout> chip1;
  chip1 = tensor.template chip<0>(1);

  VERIFY_IS_EQUAL(chip1.dimension(0), 3);
  VERIFY_IS_EQUAL(chip1.dimension(1), 5);
  VERIFY_IS_EQUAL(chip1.dimension(2), 7);
  VERIFY_IS_EQUAL(chip1.dimension(3), 11);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 11; ++l) {
          VERIFY_IS_EQUAL(chip1(i, j, k, l), tensor(1, i, j, k, l));
        }
      }
    }
  }

  Tensor<float, 4, DataLayout> chip2 = tensor.template chip<1>(1);
  VERIFY_IS_EQUAL(chip2.dimension(0), 2);
  VERIFY_IS_EQUAL(chip2.dimension(1), 5);
  VERIFY_IS_EQUAL(chip2.dimension(2), 7);
  VERIFY_IS_EQUAL(chip2.dimension(3), 11);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 11; ++l) {
          VERIFY_IS_EQUAL(chip2(i, j, k, l), tensor(i, 1, j, k, l));
        }
      }
    }
  }

  Tensor<float, 4, DataLayout> chip3 = tensor.template chip<2>(2);
  VERIFY_IS_EQUAL(chip3.dimension(0), 2);
  VERIFY_IS_EQUAL(chip3.dimension(1), 3);
  VERIFY_IS_EQUAL(chip3.dimension(2), 7);
  VERIFY_IS_EQUAL(chip3.dimension(3), 11);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 11; ++l) {
          VERIFY_IS_EQUAL(chip3(i, j, k, l), tensor(i, j, 2, k, l));
        }
      }
    }
  }

  Tensor<float, 4, DataLayout> chip4(tensor.template chip<3>(5));
  VERIFY_IS_EQUAL(chip4.dimension(0), 2);
  VERIFY_IS_EQUAL(chip4.dimension(1), 3);
  VERIFY_IS_EQUAL(chip4.dimension(2), 5);
  VERIFY_IS_EQUAL(chip4.dimension(3), 11);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 11; ++l) {
          VERIFY_IS_EQUAL(chip4(i, j, k, l), tensor(i, j, k, 5, l));
        }
      }
    }
  }

  Tensor<float, 4, DataLayout> chip5(tensor.template chip<4>(7));
  VERIFY_IS_EQUAL(chip5.dimension(0), 2);
  VERIFY_IS_EQUAL(chip5.dimension(1), 3);
  VERIFY_IS_EQUAL(chip5.dimension(2), 5);
  VERIFY_IS_EQUAL(chip5.dimension(3), 7);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(chip5(i, j, k, l), tensor(i, j, k, l, 7));
        }
      }
    }
  }
}

template <int DataLayout>
static void test_dynamic_chip() {
  Tensor<float, 5, DataLayout> tensor(2, 3, 5, 7, 11);
  tensor.setRandom();

  Tensor<float, 4, DataLayout> chip1;
  chip1 = tensor.chip(1, 0);
  VERIFY_IS_EQUAL(chip1.dimension(0), 3);
  VERIFY_IS_EQUAL(chip1.dimension(1), 5);
  VERIFY_IS_EQUAL(chip1.dimension(2), 7);
  VERIFY_IS_EQUAL(chip1.dimension(3), 11);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 11; ++l) {
          VERIFY_IS_EQUAL(chip1(i, j, k, l), tensor(1, i, j, k, l));
        }
      }
    }
  }

  Tensor<float, 4, DataLayout> chip2 = tensor.chip(1, 1);
  VERIFY_IS_EQUAL(chip2.dimension(0), 2);
  VERIFY_IS_EQUAL(chip2.dimension(1), 5);
  VERIFY_IS_EQUAL(chip2.dimension(2), 7);
  VERIFY_IS_EQUAL(chip2.dimension(3), 11);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 11; ++l) {
          VERIFY_IS_EQUAL(chip2(i, j, k, l), tensor(i, 1, j, k, l));
        }
      }
    }
  }

  Tensor<float, 4, DataLayout> chip3 = tensor.chip(2, 2);
  VERIFY_IS_EQUAL(chip3.dimension(0), 2);
  VERIFY_IS_EQUAL(chip3.dimension(1), 3);
  VERIFY_IS_EQUAL(chip3.dimension(2), 7);
  VERIFY_IS_EQUAL(chip3.dimension(3), 11);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 11; ++l) {
          VERIFY_IS_EQUAL(chip3(i, j, k, l), tensor(i, j, 2, k, l));
        }
      }
    }
  }

  Tensor<float, 4, DataLayout> chip4(tensor.chip(5, 3));
  VERIFY_IS_EQUAL(chip4.dimension(0), 2);
  VERIFY_IS_EQUAL(chip4.dimension(1), 3);
  VERIFY_IS_EQUAL(chip4.dimension(2), 5);
  VERIFY_IS_EQUAL(chip4.dimension(3), 11);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 11; ++l) {
          VERIFY_IS_EQUAL(chip4(i, j, k, l), tensor(i, j, k, 5, l));
        }
      }
    }
  }

  Tensor<float, 4, DataLayout> chip5(tensor.chip(7, 4));
  VERIFY_IS_EQUAL(chip5.dimension(0), 2);
  VERIFY_IS_EQUAL(chip5.dimension(1), 3);
  VERIFY_IS_EQUAL(chip5.dimension(2), 5);
  VERIFY_IS_EQUAL(chip5.dimension(3), 7);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(chip5(i, j, k, l), tensor(i, j, k, l, 7));
        }
      }
    }
  }
}

template <int DataLayout>
static void test_chip_in_expr() {
  Tensor<float, 5, DataLayout> input1(2, 3, 5, 7, 11);
  input1.setRandom();
  Tensor<float, 4, DataLayout> input2(3, 5, 7, 11);
  input2.setRandom();

  Tensor<float, 4, DataLayout> result = input1.template chip<0>(0) + input2;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 11; ++l) {
          float expected = input1(0, i, j, k, l) + input2(i, j, k, l);
          VERIFY_IS_EQUAL(result(i, j, k, l), expected);
        }
      }
    }
  }

  Tensor<float, 3, DataLayout> input3(3, 7, 11);
  input3.setRandom();
  Tensor<float, 3, DataLayout> result2 = input1.template chip<0>(0).template chip<1>(2) + input3;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 7; ++j) {
      for (int k = 0; k < 11; ++k) {
        float expected = input1(0, i, 2, j, k) + input3(i, j, k);
        VERIFY_IS_EQUAL(result2(i, j, k), expected);
      }
    }
  }
}

template <int DataLayout>
static void test_chip_as_lvalue() {
  Tensor<float, 5, DataLayout> input1(2, 3, 5, 7, 11);
  input1.setRandom();

  Tensor<float, 4, DataLayout> input2(3, 5, 7, 11);
  input2.setRandom();
  Tensor<float, 5, DataLayout> tensor = input1;
  tensor.template chip<0>(1) = input2;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          for (int m = 0; m < 11; ++m) {
            if (i != 1) {
              VERIFY_IS_EQUAL(tensor(i, j, k, l, m), input1(i, j, k, l, m));
            } else {
              VERIFY_IS_EQUAL(tensor(i, j, k, l, m), input2(j, k, l, m));
            }
          }
        }
      }
    }
  }

  Tensor<float, 4, DataLayout> input3(2, 5, 7, 11);
  input3.setRandom();
  tensor = input1;
  tensor.template chip<1>(1) = input3;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          for (int m = 0; m < 11; ++m) {
            if (j != 1) {
              VERIFY_IS_EQUAL(tensor(i, j, k, l, m), input1(i, j, k, l, m));
            } else {
              VERIFY_IS_EQUAL(tensor(i, j, k, l, m), input3(i, k, l, m));
            }
          }
        }
      }
    }
  }

  Tensor<float, 4, DataLayout> input4(2, 3, 7, 11);
  input4.setRandom();
  tensor = input1;
  tensor.template chip<2>(3) = input4;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          for (int m = 0; m < 11; ++m) {
            if (k != 3) {
              VERIFY_IS_EQUAL(tensor(i, j, k, l, m), input1(i, j, k, l, m));
            } else {
              VERIFY_IS_EQUAL(tensor(i, j, k, l, m), input4(i, j, l, m));
            }
          }
        }
      }
    }
  }

  Tensor<float, 4, DataLayout> input5(2, 3, 5, 11);
  input5.setRandom();
  tensor = input1;
  tensor.template chip<3>(4) = input5;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          for (int m = 0; m < 11; ++m) {
            if (l != 4) {
              VERIFY_IS_EQUAL(tensor(i, j, k, l, m), input1(i, j, k, l, m));
            } else {
              VERIFY_IS_EQUAL(tensor(i, j, k, l, m), input5(i, j, k, m));
            }
          }
        }
      }
    }
  }

  Tensor<float, 4, DataLayout> input6(2, 3, 5, 7);
  input6.setRandom();
  tensor = input1;
  tensor.template chip<4>(5) = input6;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          for (int m = 0; m < 11; ++m) {
            if (m != 5) {
              VERIFY_IS_EQUAL(tensor(i, j, k, l, m), input1(i, j, k, l, m));
            } else {
              VERIFY_IS_EQUAL(tensor(i, j, k, l, m), input6(i, j, k, l));
            }
          }
        }
      }
    }
  }

  Tensor<float, 5, DataLayout> input7(2, 3, 5, 7, 11);
  input7.setRandom();
  tensor = input1;
  tensor.chip(0, 0) = input7.chip(0, 0);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          for (int m = 0; m < 11; ++m) {
            if (i != 0) {
              VERIFY_IS_EQUAL(tensor(i, j, k, l, m), input1(i, j, k, l, m));
            } else {
              VERIFY_IS_EQUAL(tensor(i, j, k, l, m), input7(i, j, k, l, m));
            }
          }
        }
      }
    }
  }
}

static void test_chip_raw_data_col_major() {
  Tensor<float, 5, ColMajor> tensor(2, 3, 5, 7, 11);
  tensor.setRandom();

  typedef TensorEvaluator<decltype(tensor.chip<4>(3)), DefaultDevice> Evaluator4;
  auto chip = Evaluator4(tensor.chip<4>(3), DefaultDevice());
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          int chip_index = i + 2 * (j + 3 * (k + 5 * l));
          VERIFY_IS_EQUAL(chip.data()[chip_index], tensor(i, j, k, l, 3));
        }
      }
    }
  }

  typedef TensorEvaluator<decltype(tensor.chip<0>(0)), DefaultDevice> Evaluator0;
  auto chip0 = Evaluator0(tensor.chip<0>(0), DefaultDevice());
  VERIFY_IS_EQUAL(chip0.data(), static_cast<float*>(0));

  typedef TensorEvaluator<decltype(tensor.chip<1>(0)), DefaultDevice> Evaluator1;
  auto chip1 = Evaluator1(tensor.chip<1>(0), DefaultDevice());
  VERIFY_IS_EQUAL(chip1.data(), static_cast<float*>(0));

  typedef TensorEvaluator<decltype(tensor.chip<2>(0)), DefaultDevice> Evaluator2;
  auto chip2 = Evaluator2(tensor.chip<2>(0), DefaultDevice());
  VERIFY_IS_EQUAL(chip2.data(), static_cast<float*>(0));

  typedef TensorEvaluator<decltype(tensor.chip<3>(0)), DefaultDevice> Evaluator3;
  auto chip3 = Evaluator3(tensor.chip<3>(0), DefaultDevice());
  VERIFY_IS_EQUAL(chip3.data(), static_cast<float*>(0));
}

static void test_chip_raw_data_row_major() {
  Tensor<float, 5, RowMajor> tensor(11, 7, 5, 3, 2);
  tensor.setRandom();

  typedef TensorEvaluator<decltype(tensor.chip<0>(3)), DefaultDevice> Evaluator0;
  auto chip = Evaluator0(tensor.chip<0>(3), DefaultDevice());
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 2; ++l) {
          int chip_index = l + 2 * (k + 3 * (j + 5 * i));
          VERIFY_IS_EQUAL(chip.data()[chip_index], tensor(3, i, j, k, l));
        }
      }
    }
  }

  typedef TensorEvaluator<decltype(tensor.chip<1>(0)), DefaultDevice> Evaluator1;
  auto chip1 = Evaluator1(tensor.chip<1>(0), DefaultDevice());
  VERIFY_IS_EQUAL(chip1.data(), static_cast<float*>(0));

  typedef TensorEvaluator<decltype(tensor.chip<2>(0)), DefaultDevice> Evaluator2;
  auto chip2 = Evaluator2(tensor.chip<2>(0), DefaultDevice());
  VERIFY_IS_EQUAL(chip2.data(), static_cast<float*>(0));

  typedef TensorEvaluator<decltype(tensor.chip<3>(0)), DefaultDevice> Evaluator3;
  auto chip3 = Evaluator3(tensor.chip<3>(0), DefaultDevice());
  VERIFY_IS_EQUAL(chip3.data(), static_cast<float*>(0));

  typedef TensorEvaluator<decltype(tensor.chip<4>(0)), DefaultDevice> Evaluator4;
  auto chip4 = Evaluator4(tensor.chip<4>(0), DefaultDevice());
  VERIFY_IS_EQUAL(chip4.data(), static_cast<float*>(0));
}

template <int Rank, int Layout>
EIGEN_DONT_INLINE Tensor<double, Rank - 1, Layout> runtime_axis_chip(const Tensor<double, Rank, Layout>& input,
                                                                     Index offset, Index axis) {
  return input.chip(offset, axis);
}

template <int Rank, int Layout, int Axis>
void test_runtime_axis_block_chip() {
  DSizes<Index, Rank> dimensions;
  for (int i = 0; i < Rank; ++i) dimensions[i] = 3 + i;
  Tensor<double, Rank, Layout> input(dimensions);
  for (Index i = 0; i < input.size(); ++i) input.data()[i] = double(i);
  for (Index offset : {Index(0), dimensions[Axis] - 1}) {
    const Tensor<double, Rank - 1, Layout> result = runtime_axis_chip(input, offset, Axis);
    const Tensor<double, Rank - 1, Layout> expected = input.template chip<Axis>(offset);
    VERIFY_IS_EQUAL(result.size(), expected.size());
    for (Index i = 0; i < result.size(); ++i) VERIFY_IS_EQUAL(result.data()[i], expected.data()[i]);

    Tensor<double, Rank, Layout> actual = input;
    Tensor<double, Rank, Layout> reference = input;
    actual.chip(offset, Axis) = result + result;
    reference.template chip<Axis>(offset) = expected + expected;
    for (Index i = 0; i < actual.size(); ++i) VERIFY_IS_EQUAL(actual.data()[i], reference.data()[i]);
  }
}

template <int Layout>
void test_runtime_axis_block_chips() {
  test_runtime_axis_block_chip<3, Layout, 0>();
  test_runtime_axis_block_chip<3, Layout, 1>();
  test_runtime_axis_block_chip<3, Layout, 2>();
  test_runtime_axis_block_chip<4, Layout, 0>();
  test_runtime_axis_block_chip<4, Layout, 1>();
  test_runtime_axis_block_chip<4, Layout, 2>();
  test_runtime_axis_block_chip<4, Layout, 3>();
}

template <typename Scalar, int Layout>
void test_chip_packet_boundary_for_stride(Index inner_dim) {
  const Index chip_axis = (Layout == ColMajor) ? 1 : 2;
  DSizes<Index, 4> dims;
  if (Layout == ColMajor) {
    dims = DSizes<Index, 4>(inner_dim, 3, 5, 7);
  } else {
    dims = DSizes<Index, 4>(7, 5, 3, inner_dim);
  }

  Tensor<Scalar, 4, Layout> tensor(dims);
  for (Index i = 0; i < tensor.size(); ++i) {
    tensor.data()[i] = static_cast<Scalar>((i % 29) - 14);
  }

  DefaultDevice device;
  for (Index chip_offset = 0; chip_offset < 3; ++chip_offset) {
    // 1. Raw storage evaluator (m_impl.data() != nullptr).
    auto raw_chip_op = tensor.chip(chip_offset, chip_axis);
    TensorEvaluator<const decltype(raw_chip_op), DefaultDevice> raw_eval(raw_chip_op, device);
    const int packet_size = TensorEvaluator<const decltype(raw_chip_op), DefaultDevice>::PacketSize;
    const Index total_size = raw_eval.dimensions().TotalSize();

    EIGEN_ALIGN_MAX Scalar unpacked[64];
    for (Index idx = 0; idx + packet_size <= total_size; ++idx) {
      auto pkt = raw_eval.template packet<Unaligned>(idx);
      internal::pstore<Scalar>(unpacked, pkt);
      for (int lane = 0; lane < packet_size; ++lane) {
        VERIFY_IS_EQUAL(unpacked[lane], raw_eval.coeff(idx + lane));
      }
    }

    // 2. Non-raw nested expression evaluator (m_impl.data() == nullptr) - exercises division-free stepping.
    auto expr_chip_op = (tensor + tensor).chip(chip_offset, chip_axis);
    TensorEvaluator<const decltype(expr_chip_op), DefaultDevice> expr_eval(expr_chip_op, device);
    expr_eval.evalSubExprsIfNeeded(nullptr);
    for (Index idx = 0; idx + packet_size <= total_size; ++idx) {
      auto pkt = expr_eval.template packet<Unaligned>(idx);
      internal::pstore<Scalar>(unpacked, pkt);
      for (int lane = 0; lane < packet_size; ++lane) {
        VERIFY_IS_EQUAL(unpacked[lane], expr_eval.coeff(idx + lane));
      }
    }
    expr_eval.cleanup();

    // 3. Lvalue writePacket across stride boundaries.
    for (Index start_phase = 0; start_phase < numext::mini<Index>(packet_size, total_size); ++start_phase) {
      Tensor<Scalar, 4, Layout> dst = tensor;
      Tensor<Scalar, 4, Layout> expected = tensor;
      auto lval_op = dst.chip(chip_offset, chip_axis);
      TensorEvaluator<decltype(lval_op), DefaultDevice> lval_eval(lval_op, device);
      auto ref_op = expected.chip(chip_offset, chip_axis);
      TensorEvaluator<decltype(ref_op), DefaultDevice> ref_eval(ref_op, device);

      for (Index idx = start_phase; idx + packet_size <= total_size; idx += packet_size) {
        for (int lane = 0; lane < packet_size; ++lane) {
          unpacked[lane] = static_cast<Scalar>(((idx + lane) % 19) + 3);
          ref_eval.coeffRef(idx + lane) = unpacked[lane];
        }
        auto pkt =
            internal::pload<typename TensorEvaluator<decltype(lval_op), DefaultDevice>::PacketReturnType>(unpacked);
        lval_eval.template writePacket<Unaligned>(idx, pkt);
      }
      for (Index i = 0; i < dst.size(); ++i) {
        VERIFY_IS_EQUAL(dst.data()[i], expected.data()[i]);
      }
    }
  }
}

template <int Layout>
void test_chip_packet_boundaries() {
  for (Index stride : {Index(2), Index(3), Index(4), Index(6), Index(8), Index(12), Index(18), Index(24)}) {
    test_chip_packet_boundary_for_stride<float, Layout>(stride);
    test_chip_packet_boundary_for_stride<double, Layout>(stride);
    test_chip_packet_boundary_for_stride<Eigen::half, Layout>(stride);
    test_chip_packet_boundary_for_stride<Eigen::bfloat16, Layout>(stride);
    test_chip_packet_boundary_for_stride<std::complex<float>, Layout>(stride);
  }
}

template <typename EvalT, typename PacketT,
          std::enable_if_t<internal::unpacket_traits<PacketT>::masked_load_available, int> = 0>
void verify_partial_packets_for_eval(const EvalT& eval) {
  using Scalar = typename internal::unpacket_traits<PacketT>::type;
  using MaskT = typename internal::unpacket_traits<PacketT>::mask_t;
  constexpr int kPacketSize = internal::unpacket_traits<PacketT>::size;
  const Index total_size = eval.dimensions().TotalSize();
  EIGEN_ALIGN_MAX Scalar unpacked[64];

  const MaskT full_mask = kPacketSize < static_cast<int>(CHAR_BIT * sizeof(MaskT))
                              ? static_cast<MaskT>((MaskT(1) << kPacketSize) - 1)
                              : (std::numeric_limits<MaskT>::max)();
  const MaskT sparse_mask = static_cast<MaskT>(0xA5A5u) & full_mask;
  const MaskT upper_half_mask = static_cast<MaskT>(full_mask ^ ((MaskT(1) << (kPacketSize / 2)) - 1));

  // Zero mask never touches underlying memory, even at boundary indices.
  {
    PacketT zero_pkt = eval.template partialPacket<PacketT>(0, MaskT(0));
    internal::pstore<Scalar>(unpacked, zero_pkt);
    for (int lane = 0; lane < kPacketSize; ++lane) {
      VERIFY_IS_EQUAL(unpacked[lane], Scalar(0));
    }
  }

  // Negative starting index with upper lanes active (exercised by spatial convolution mappers).
  if (total_size >= kPacketSize / 2) {
    const Index neg_idx = -(kPacketSize / 2);
    PacketT pkt = eval.template partialPacket<PacketT>(neg_idx, upper_half_mask);
    internal::pstore<Scalar>(unpacked, pkt);
    for (int lane = 0; lane < kPacketSize; ++lane) {
      const Scalar expected = ((upper_half_mask >> lane) & MaskT(1)) ? eval.coeff(neg_idx + lane) : Scalar(0);
      VERIFY_IS_EQUAL(unpacked[lane], expected);
    }
  }

  // Sweep contiguous prefix/suffix masks and non-contiguous masks across the chipped tensor.
  for (Index idx = 0; idx < total_size; ++idx) {
    const int avail = static_cast<int>(numext::mini<Index>(kPacketSize, total_size - idx));
    const MaskT prefix_mask =
        avail == static_cast<int>(CHAR_BIT * sizeof(MaskT)) ? full_mask : static_cast<MaskT>((MaskT(1) << avail) - 1);
    const MaskT masks_to_check[2] = {prefix_mask, static_cast<MaskT>(sparse_mask & prefix_mask)};
    for (MaskT m : masks_to_check) {
      PacketT pkt = eval.template partialPacket<PacketT>(idx, m);
      internal::pstore<Scalar>(unpacked, pkt);
      for (int lane = 0; lane < kPacketSize; ++lane) {
        const Scalar expected = ((m >> lane) & MaskT(1)) ? eval.coeff(idx + lane) : Scalar(0);
        VERIFY_IS_EQUAL(unpacked[lane], expected);
      }
    }
  }
}

template <typename EvalT, typename PacketT,
          std::enable_if_t<!internal::unpacket_traits<PacketT>::masked_load_available, int> = 0>
void verify_partial_packets_for_eval(const EvalT&) {}

template <typename Scalar, int Layout>
void test_chip_partial_packets_for_type() {
  using Packet = typename internal::packet_traits<Scalar>::type;
  using HalfPacket = typename internal::unpacket_traits<Packet>::half;
  DefaultDevice device;

  for (Index inner_dim : {Index(3), Index(4), Index(8), Index(12)}) {
    DSizes<Index, 4> dims =
        (Layout == ColMajor) ? DSizes<Index, 4>(inner_dim, 3, 5, 2) : DSizes<Index, 4>(2, 5, 3, inner_dim);
    Tensor<Scalar, 4, Layout> tensor(dims);
    for (Index i = 0; i < tensor.size(); ++i) {
      tensor.data()[i] = static_cast<Scalar>((i % 31) + 1);
    }

    for (int axis = 0; axis < 4; ++axis) {
      const Index extent = tensor.dimension(axis);
      for (Index offset = 0; offset < numext::mini<Index>(extent, 2); ++offset) {
        auto raw_op = tensor.chip(offset, axis);
        TensorEvaluator<const decltype(raw_op), DefaultDevice> raw_eval(raw_op, device);
        verify_partial_packets_for_eval<decltype(raw_eval), Packet>(raw_eval);
        if (!std::is_same<HalfPacket, Packet>::value) {
          verify_partial_packets_for_eval<decltype(raw_eval), HalfPacket>(raw_eval);
        }

        auto expr_op = (tensor + tensor).chip(offset, axis);
        TensorEvaluator<const decltype(expr_op), DefaultDevice> expr_eval(expr_op, device);
        expr_eval.evalSubExprsIfNeeded(nullptr);
        verify_partial_packets_for_eval<decltype(expr_eval), Packet>(expr_eval);
        expr_eval.cleanup();
      }
    }

    // Nested middle-dimension chipping.
    auto nested_op = tensor.chip(1, Layout == ColMajor ? 2 : 1).chip(1, 1);
    TensorEvaluator<const decltype(nested_op), DefaultDevice> nested_eval(nested_op, device);
    verify_partial_packets_for_eval<decltype(nested_eval), Packet>(nested_eval);
  }
}

template <int Layout>
void test_chip_partial_packets() {
  test_chip_partial_packets_for_type<float, Layout>();
  test_chip_partial_packets_for_type<double, Layout>();
}

template <int Layout>
void test_chip_trivial_size_one_dim() {
  DefaultDevice device;
  for (int singleton_axis = 0; singleton_axis < 4; ++singleton_axis) {
    DSizes<Index, 4> dims(5, 3, 7, 4);
    dims[singleton_axis] = 1;
    Tensor<float, 4, Layout> src(dims);
    for (Index i = 0; i < src.size(); ++i) {
      src.data()[i] = static_cast<float>(i + 1);
    }

    auto chip_op = src.chip(0, singleton_axis);
    TensorEvaluator<const decltype(chip_op), DefaultDevice> eval(chip_op, device);
    VERIFY(eval.data() == src.data());

    Tensor<float, 3, Layout> out = chip_op;
    VERIFY_IS_EQUAL(out.size(), src.size());
    for (Index i = 0; i < out.size(); ++i) {
      VERIFY_IS_EQUAL(out.data()[i], src.data()[i]);
    }

    Tensor<float, 4, Layout> dst(dims);
    dst.setZero();
    dst.chip(0, singleton_axis) = out * out.constant(2.0f);
    for (Index i = 0; i < dst.size(); ++i) {
      VERIFY_IS_EQUAL(dst.data()[i], 2.0f * src.data()[i]);
    }
  }

  // Static inner axis with runtime extent 1 is also contiguous.
  DSizes<Index, 3> static_dims = (Layout == ColMajor) ? DSizes<Index, 3>(1, 9, 5) : DSizes<Index, 3>(5, 9, 1);
  Tensor<float, 3, Layout> static_src(static_dims);
  static_src.setRandom();
  constexpr int kInnerAxis = (Layout == ColMajor) ? 0 : 2;
  auto static_chip = static_src.template chip<kInnerAxis>(0);
  TensorEvaluator<const decltype(static_chip), DefaultDevice> static_eval(static_chip, device);
  VERIFY(static_eval.data() == static_src.data());
  Tensor<float, 2, Layout> static_out = static_chip + static_chip;
  for (Index i = 0; i < static_src.size(); ++i) {
    VERIFY_IS_APPROX(static_out.data()[i], 2.0f * static_src.data()[i]);
  }
}

template <typename EvalT, typename SubPacket>
void verify_subpackets_for_eval(const EvalT& eval) {
  using Scalar = std::remove_const_t<typename EvalT::CoeffReturnType>;
  constexpr int kSubSize = internal::unpacket_traits<SubPacket>::size;
  const Index total_size = eval.dimensions().TotalSize();
  if (total_size < kSubSize) return;

  EIGEN_ALIGN_TO_BOUNDARY(internal::unpacket_traits<SubPacket>::alignment) Scalar unpacked[kSubSize];
  for (Index idx = 0; idx + kSubSize <= total_size; ++idx) {
    SubPacket pkt = eval.template packet<Unaligned, SubPacket>(idx);
    internal::pstore<Scalar>(unpacked, pkt);
    for (int lane = 0; lane < kSubSize; ++lane) {
      VERIFY_IS_EQUAL(unpacked[lane], eval.coeff(idx + lane));
    }
  }
}

template <typename Scalar, int Layout>
void test_chip_subpackets_and_empty_for_type() {
  using Packet = typename internal::packet_traits<Scalar>::type;
  using HalfPacket = typename internal::unpacket_traits<Packet>::half;
  using QuarterPacket = typename internal::unpacket_traits<HalfPacket>::half;
  DefaultDevice device;

  // Zero-sized unchipped dimension exercises m_stride == 0 initialization.
  DSizes<Index, 3> empty_dims = (Layout == ColMajor) ? DSizes<Index, 3>(0, 4, 5) : DSizes<Index, 3>(5, 4, 0);
  Tensor<Scalar, 3, Layout> empty_src(empty_dims);
  Tensor<Scalar, 2, Layout> empty_out = empty_src.chip(1, 1);
  VERIFY_IS_EQUAL(empty_out.size(), Index(0));

  for (Index inner_dim : {Index(3), Index(4), Index(8), Index(12)}) {
    DSizes<Index, 4> dims =
        (Layout == ColMajor) ? DSizes<Index, 4>(inner_dim, 3, 5, 2) : DSizes<Index, 4>(2, 5, 3, inner_dim);
    Tensor<Scalar, 4, Layout> tensor(dims);
    for (Index i = 0; i < tensor.size(); ++i) {
      tensor.data()[i] = static_cast<Scalar>((i % 29) + 1);
    }

    for (int axis = 0; axis < 4; ++axis) {
      auto raw_op = tensor.chip(0, axis);
      TensorEvaluator<const decltype(raw_op), DefaultDevice> raw_eval(raw_op, device);
      verify_subpackets_for_eval<decltype(raw_eval), Packet>(raw_eval);
      if (!std::is_same<HalfPacket, Packet>::value) {
        verify_subpackets_for_eval<decltype(raw_eval), HalfPacket>(raw_eval);
      }
      if (!std::is_same<QuarterPacket, HalfPacket>::value) {
        verify_subpackets_for_eval<decltype(raw_eval), QuarterPacket>(raw_eval);
      }

      auto expr_op = (tensor + tensor).chip(0, axis);
      TensorEvaluator<const decltype(expr_op), DefaultDevice> expr_eval(expr_op, device);
      expr_eval.evalSubExprsIfNeeded(nullptr);
      if (!std::is_same<HalfPacket, Packet>::value) {
        verify_subpackets_for_eval<decltype(expr_eval), HalfPacket>(expr_eval);
      }
      expr_eval.cleanup();
    }

    auto nested_op = tensor.chip(1, Layout == ColMajor ? 2 : 1).chip(1, 1);
    TensorEvaluator<const decltype(nested_op), DefaultDevice> nested_eval(nested_op, device);
    if (!std::is_same<HalfPacket, Packet>::value) {
      verify_subpackets_for_eval<decltype(nested_eval), HalfPacket>(nested_eval);
    }
  }
}

template <int Layout>
void test_chip_subpackets_and_empty() {
  test_chip_subpackets_and_empty_for_type<float, Layout>();
  test_chip_subpackets_and_empty_for_type<double, Layout>();
  test_chip_subpackets_and_empty_for_type<int, Layout>();
}

EIGEN_DECLARE_TEST(tensor_chipping) {
  CALL_SUBTEST(test_runtime_axis_block_chips<ColMajor>());
  CALL_SUBTEST(test_runtime_axis_block_chips<RowMajor>());
  CALL_SUBTEST(test_chip_packet_boundaries<ColMajor>());
  CALL_SUBTEST(test_chip_packet_boundaries<RowMajor>());
  CALL_SUBTEST(test_chip_partial_packets<ColMajor>());
  CALL_SUBTEST(test_chip_partial_packets<RowMajor>());
  CALL_SUBTEST(test_chip_subpackets_and_empty<ColMajor>());
  CALL_SUBTEST(test_chip_subpackets_and_empty<RowMajor>());
  CALL_SUBTEST(test_chip_trivial_size_one_dim<ColMajor>());
  CALL_SUBTEST(test_chip_trivial_size_one_dim<RowMajor>());
  CALL_SUBTEST(test_simple_chip<ColMajor>());
  CALL_SUBTEST(test_simple_chip<RowMajor>());
  CALL_SUBTEST(test_dynamic_chip<ColMajor>());
  CALL_SUBTEST(test_dynamic_chip<RowMajor>());
  CALL_SUBTEST(test_chip_in_expr<ColMajor>());
  CALL_SUBTEST(test_chip_in_expr<RowMajor>());
  CALL_SUBTEST(test_chip_as_lvalue<ColMajor>());
  CALL_SUBTEST(test_chip_as_lvalue<RowMajor>());
  CALL_SUBTEST(test_chip_raw_data_col_major());
  CALL_SUBTEST(test_chip_raw_data_row_major());
}
