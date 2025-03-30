// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template <typename Packet>
struct packet_segment_test_impl {
  using Scalar = typename internal::unpacket_traits<Packet>::type;
  static constexpr int PacketSize = internal::unpacket_traits<Packet>::size;
  static void run() {
    // 2 * PacketSize - 1 is used to avoid zero-size allocations when PacketSize == 1
    constexpr int Size = 2 * PacketSize - 1;
    alignas(Packet) Scalar data_in[Size];
    alignas(Packet) Scalar data_out[Size];

    {
      // test loading a packet segment from aligned memory that includes unallocated memory
      //    X X X X|* * * *|* * * *|X X X X
      //           begin ->|* * * X| <- begin + count

      Index begin = 0;
      Index count = PacketSize - 1;

      Scalar* aligned_data_in = data_in + PacketSize;
      Scalar* aligned_data_out = data_out + PacketSize;

      for (Index i = 0; i < Size; i++) {
        data_in[i] = internal::random<Scalar>();
        data_out[i] = internal::random<Scalar>();
      }

      Packet a = internal::ploadSegment<Packet>(aligned_data_in, begin, count);
      internal::pstoreSegment<Scalar, Packet>(aligned_data_out, a, begin, count);

      for (Index i = begin; i < begin + count; i++) {
        VERIFY_IS_EQUAL(aligned_data_in[i], aligned_data_out[i]);
      }
    }

    {
      // test loading a packet segment from unaligned memory that includes unallocated memory
      //    X X X X|* * * *|* * * *|X X X X
      // begin ->|X * * *| <- begin + count

      Index begin = 1;
      Index count = PacketSize - 1;

      Scalar* unaligned_data_in = data_in - 1;
      Scalar* unaligned_data_out = data_out - 1;

      for (Index i = 0; i < Size; i++) {
        data_in[i] = internal::random<Scalar>();
        data_out[i] = internal::random<Scalar>();
      }

      Packet b = internal::ploadSegment<Packet>(unaligned_data_in, begin, count);
      internal::pstoreSegment<Scalar, Packet>(unaligned_data_out, b, begin, count);

      for (Index i = begin; i < begin + count; i++) {
        VERIFY_IS_EQUAL(unaligned_data_in[i], unaligned_data_out[i]);
      }
    }

    {
      // test loading an empty packet segment from a null pointer
      // begin can be anything, but offsetting a null pointer is undefined behavior
      Index begin = 0;
      Index count = 0;

      Scalar* null_data_in = nullptr;
      Scalar* null_data_out = nullptr;

      Packet c = internal::ploadSegment<Packet>(null_data_in, begin, count);
      internal::pstoreSegment<Scalar, Packet>(null_data_out, c, begin, count);
    }
  }
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
