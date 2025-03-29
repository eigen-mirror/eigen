// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/Core>

template <typename Packet, bool Run = internal::has_packet_segment<Packet>::value>
struct packet_segment_test_impl {
  using Scalar = typename internal::unpacket_traits<Packet>::type;
  static constexpr int PacketSize = internal::unpacket_traits<Packet>::size;
  static void test_unaligned() {
    // test an unaligned array that is shorter than PacketSize
    {
      alignas(Packet) Scalar data_in[PacketSize], data_out[PacketSize];

      Index begin = 0;
      Index count = PacketSize - 1;

      Scalar* unaligned_data_in = data_in + 1;
      Scalar* unaligned_data_out = data_out + 1;

      // unaligned_data_in[PacketSize - 1] is unallocated memory

      for (Index i = begin; i < begin + count; i++) {
        unaligned_data_in[i] = internal::random<Scalar>();
        unaligned_data_out[i] = internal::random<Scalar>();
      }

      Packet a = internal::ploaduSegment<Packet>(unaligned_data_in, begin, count);
      internal::pstoreuSegment<Scalar, Packet>(unaligned_data_out, a, begin, count);

      for (Index i = begin; i < begin + count; i++) {
        VERIFY_IS_EQUAL(unaligned_data_in[i], unaligned_data_out[i]);
      }

      begin = 1;

      unaligned_data_in = data_in - 1;
      unaligned_data_out = data_out - 1;

      // unaligned_data_in[0] is unallocated memory

      for (Index i = begin; i < begin + count; i++) {
        unaligned_data_in[i] = internal::random<Scalar>();
        unaligned_data_out[i] = internal::random<Scalar>();
      }

      Packet b = internal::ploaduSegment<Packet>(unaligned_data_in, begin, count);
      internal::pstoreuSegment<Scalar, Packet>(unaligned_data_out, b, begin, count);

      for (Index i = begin; i < begin + count; i++) {
        VERIFY_IS_EQUAL(unaligned_data_in[i], unaligned_data_out[i]);
      }
    }
  }
  static void test_aligned() {
    // test an unaligned array that is shorter than PacketSize
    {
      alignas(Packet) Scalar aligned_data_in[PacketSize - 1], aligned_data_out[PacketSize - 1];

      Index begin = 0;
      Index count = PacketSize - 1;

      // unaligned_data_in[PacketSize - 1] is unallocated memory

      for (Index i = begin; i < begin + count; i++) {
        aligned_data_in[i] = internal::random<Scalar>();
        aligned_data_out[i] = internal::random<Scalar>();
      }

      Packet a = internal::ploadSegment<Packet>(aligned_data_in, begin, count);
      internal::pstoreSegment<Scalar, Packet>(aligned_data_out, a, begin, count);

      for (Index i = begin; i < begin + count; i++) {
        VERIFY_IS_EQUAL(aligned_data_in[i], aligned_data_out[i]);
      }
    }
  }
  static void run() {
    test_unaligned();
    test_aligned();
  }
};

template <typename Packet>
struct packet_segment_test_impl<Packet, false> {
  static void run() {}
};

template <typename Packet, typename HalfPacket = typename internal::unpacket_traits<Packet>::half>
struct packet_segment_test_driver {
  static void run() {
    packet_segment_test_impl<Packet>::run();
    packet_segment_test_driver<HalfPacket>::run();
  }
};

template <typename Packet>
struct packet_segment_test_driver<Packet, Packet> : packet_segment_test_impl<Packet> {};

template <typename Scalar>
void test_packet_segment() {
  using Packet = typename internal::packet_traits<Scalar>::type;
  packet_segment_test_driver<Packet>::run();
}

// most of these will not test any code, but are included for future proofing
EIGEN_DECLARE_TEST(packet_segment) {
  test_packet_segment<bool>();
  test_packet_segment<int8_t>();
  test_packet_segment<uint8_t>();
  test_packet_segment<int16_t>();
  test_packet_segment<uint16_t>();
  test_packet_segment<int32_t>();
  test_packet_segment<uint32_t>();
  test_packet_segment<int64_t>();
  test_packet_segment<uint64_t>();
  test_packet_segment<bfloat16>();
  test_packet_segment<half>();
  test_packet_segment<float>();
  test_packet_segment<double>();
  test_packet_segment<std::complex<float>>();
  test_packet_segment<std::complex<double>>();
}
