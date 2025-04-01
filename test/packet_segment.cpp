// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template <typename Packet, bool Run = internal::has_packet_segment<Packet>::value>
struct packet_segment_test_impl {
  using Scalar = typename internal::unpacket_traits<Packet>::type;
  static constexpr int PacketSize = internal::unpacket_traits<Packet>::size;
  static constexpr int Alignment = internal::unpacket_traits<Packet>::alignment;
  static void test_unaligned() {
    // test loading a packet segment from unaligned memory that includes unallocated memory

    // | X   X   X   X | *   *   *   X | X   X   X   X |
    //    begin -> { X | *   *   *   } <- begin + count

    VectorX<Scalar> data_in(PacketSize), data_out(PacketSize);
    data_in.setRandom();
    data_out.setRandom();

    Scalar* unaligned_data_in = data_in.data() - 1;
    Scalar* unaligned_data_out = data_out.data() - 1;

    Index begin = 1;
    Index count = PacketSize - 1;

    Packet b = internal::ploaduSegment<Packet>(unaligned_data_in, begin, count);
    internal::pstoreuSegment<Scalar, Packet>(unaligned_data_out, b, begin, count);

    for (Index i = begin; i < begin + count; i++) {
      VERIFY_IS_EQUAL(unaligned_data_in[i], unaligned_data_out[i]);
    }

    // test loading an empty packet segment in unallocated memory

    data_in.setRandom();
    data_out = data_in;

    unaligned_data_in = data_in.data() + 100 * data_in.size();
    unaligned_data_out = data_out.data() + 100 * data_out.size();

    count = 0;

    for (begin = 0; begin < PacketSize; begin++) {
      Packet c = internal::ploaduSegment<Packet>(unaligned_data_in, begin, count);
      internal::pstoreuSegment<Scalar, Packet>(unaligned_data_out, c, begin, count);
    }

    // verify that ploaduSegment / pstoreuSegment resulted in a no-op
    VERIFY_IS_CWISE_EQUAL(data_in, data_out);
  }
  static void test_aligned() {
    // test loading a packet segment from aligned memory that includes unallocated memory

    // | X   X   X   X | *   *   *   X | X   X   X   X |
    //        begin -> { *   *   *   X } <- begin + count

    VectorX<Scalar> data_in(PacketSize - 1), data_out(PacketSize - 1);
    data_in.setRandom();
    data_out.setRandom();

    Scalar* aligned_data_in = data_in.data();
    Scalar* aligned_data_out = data_out.data();

    Index begin = 0;
    Index count = PacketSize - 1;

    Packet b = internal::ploadSegment<Packet>(aligned_data_in, begin, count);
    internal::pstoreSegment<Scalar, Packet>(aligned_data_out, b, begin, count);

    for (Index i = begin; i < begin + count; i++) {
      VERIFY_IS_EQUAL(aligned_data_in[i], aligned_data_out[i]);
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
