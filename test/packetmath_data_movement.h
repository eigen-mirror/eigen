// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_PACKETMATH_DATA_MOVEMENT_H
#define EIGEN_TEST_PACKETMATH_DATA_MOVEMENT_H

#include "packetmath_test_shared.h"

namespace Eigen {
namespace test {

using internal::unpacket_traits;

template <typename Scalar>
using Buffer = Array<Scalar, Dynamic, 1>;

// Loads and stores with their own addressing. `offset` misaligns the unaligned forms.
template <typename Packet>
struct ploadu_kernel {
  using Scalar = typename unpacket_traits<Packet>::type;
  static constexpr int kSize = unpacket_traits<Packet>::size;
  int offset;
  EIGEN_PACKET_TEST_FUNC void operator()(int i, const Scalar* in, Scalar* out) const {
    internal::pstore(out + i * kSize, internal::ploadu<Packet>(in + i * kSize + offset));
  }
};
template <typename Packet>
struct pstoreu_kernel {
  using Scalar = typename unpacket_traits<Packet>::type;
  static constexpr int kSize = unpacket_traits<Packet>::size;
  int offset;
  EIGEN_PACKET_TEST_FUNC void operator()(int i, const Scalar* in, Scalar* out) const {
    internal::pstoreu(out + i * kSize + offset, internal::pload<Packet>(in + i * kSize));
  }
};
template <typename Packet, int Alignment>
struct ploadt_ro_kernel {
  using Scalar = typename unpacket_traits<Packet>::type;
  static constexpr int kSize = unpacket_traits<Packet>::size;
  int offset;
  EIGEN_PACKET_TEST_FUNC void operator()(int i, const Scalar* in, Scalar* out) const {
    internal::pstore(out + i * kSize, internal::ploadt_ro<Packet, Alignment>(in + i * kSize + offset));
  }
};
template <typename Packet>
struct ploaddup_kernel {
  using Scalar = typename unpacket_traits<Packet>::type;
  static constexpr int kSize = unpacket_traits<Packet>::size;
  int offset;
  EIGEN_PACKET_TEST_FUNC void operator()(int i, const Scalar* in, Scalar* out) const {
    internal::pstore(out + i * kSize, internal::ploaddup<Packet>(in + i * (kSize / 2) + offset));
  }
};
template <typename Packet>
struct pset1_kernel {
  using Scalar = typename unpacket_traits<Packet>::type;
  static constexpr int kSize = unpacket_traits<Packet>::size;
  EIGEN_PACKET_TEST_FUNC void operator()(int i, const Scalar* in, Scalar* out) const {
    internal::pstore(out + i * kSize, internal::pset1<Packet>(in[i]));
  }
};
template <typename Packet>
struct pgather_kernel {
  using Scalar = typename unpacket_traits<Packet>::type;
  static constexpr int kSize = unpacket_traits<Packet>::size;
  int stride;
  EIGEN_PACKET_TEST_FUNC void operator()(int i, const Scalar* in, Scalar* out) const {
    internal::pstore(out + i * kSize, internal::pgather<Scalar, Packet>(in + i * kSize * stride, stride));
  }
};
template <typename Packet>
struct pscatter_kernel {
  using Scalar = typename unpacket_traits<Packet>::type;
  static constexpr int kSize = unpacket_traits<Packet>::size;
  int stride;
  EIGEN_PACKET_TEST_FUNC void operator()(int i, const Scalar* in, Scalar* out) const {
    internal::pscatter<Scalar, Packet>(out + i * kSize * stride, internal::pload<Packet>(in + i * kSize), stride);
  }
};
template <typename Packet>
struct ptranspose_kernel {
  using Scalar = typename unpacket_traits<Packet>::type;
  static constexpr int kSize = unpacket_traits<Packet>::size;
  EIGEN_PACKET_TEST_FUNC void operator()(int i, const Scalar* in, Scalar* out) const {
    internal::PacketBlock<Packet, kSize> block;
    for (int r = 0; r < kSize; ++r) block.packet[r] = internal::pload<Packet>(in + (i * kSize + r) * kSize);
    internal::ptranspose(block);
    for (int r = 0; r < kSize; ++r) internal::pstore(out + (i * kSize + r) * kSize, block.packet[r]);
  }
};
// Thread i copies its first (i mod (kSize + 1)) lanes; the rest of the output keeps its sentinel.
template <typename Packet>
struct partial_kernel {
  using Scalar = typename unpacket_traits<Packet>::type;
  static constexpr int kSize = unpacket_traits<Packet>::size;
  EIGEN_PACKET_TEST_FUNC void operator()(int i, const Scalar* in, Scalar* out) const {
    const Index n = i % (kSize + 1);
    internal::pstore_partial(out + i * kSize, internal::pload_partial<Packet>(in + i * kSize, n), n);
  }
};
template <typename Packet>
struct preverse_kernel {
  using Scalar = typename unpacket_traits<Packet>::type;
  static constexpr int kSize = unpacket_traits<Packet>::size;
  EIGEN_PACKET_TEST_FUNC void operator()(int i, const Scalar* in, Scalar* out) const {
    internal::pstore(out + i * kSize, internal::preverse(internal::pload<Packet>(in + i * kSize)));
  }
};

// The input owns aligned storage and contains at least max(4, packet size) complete packets.
// run(kernel, count, input, output) executes one kernel invocation per packet (or transpose block).
template <typename Packet, typename Run>
void packetmath_data_movement(const Buffer<typename unpacket_traits<Packet>::type>& in, const Run& run) {
  using Scalar = typename unpacket_traits<Packet>::type;
  constexpr int kSize = unpacket_traits<Packet>::size;
  VERIFY(in.size() >= kSize * (kSize > 4 ? kSize : 4) && in.size() % kSize == 0);
  const int n = int(in.size()) / kSize;
  {
    Buffer<Scalar> out(in.size()), padded(in.size() + kSize), out_padded(in.size() + kSize),
        expected_padded(in.size() + kSize);
    padded << in, Buffer<Scalar>::Constant(kSize, Scalar(1));
    for (int offset = 0; offset < kSize; ++offset) {
      out.setConstant(Scalar(-7));
      run(ploadu_kernel<Packet>{offset}, n, padded, out);
      VERIFY(areEqualBits(padded.data() + offset, out.data(), int(in.size()), false) && "ploadu");
      run(ploadt_ro_kernel<Packet, Unaligned>{offset}, n, padded, out);
      VERIFY(areEqualBits(padded.data() + offset, out.data(), int(in.size()), false) && "ploadt_ro<Unaligned>");
      out_padded.setConstant(Scalar(-7));
      expected_padded.setConstant(Scalar(-7));
      expected_padded.segment(offset, in.size()) = in;
      run(pstoreu_kernel<Packet>{offset}, n, in, out_padded);
      VERIFY(areEqualBits(expected_padded.data(), out_padded.data(), int(out_padded.size()), false) && "pstoreu");
    }
  }
  {
    Buffer<Scalar> out(in.size());
    out.setConstant(Scalar(-7));
    run(ploadt_ro_kernel<Packet, Aligned>{0}, n, in, out);
    VERIFY(areEqualBits(in.data(), out.data(), int(in.size()), false) && "ploadt_ro<Aligned>");
    Buffer<Scalar> expected(in.size());
    if (kSize > 1) {
      for (int offset = 0; offset < 4; ++offset) {
        run(ploaddup_kernel<Packet>{offset}, n, in, out);
        for (Index k = 0; k < in.size(); ++k) expected[k] = in[(k / kSize) * (kSize / 2) + (k % kSize) / 2 + offset];
        VERIFY(areEqualBits(expected.data(), out.data(), int(in.size()), false) && "ploaddup");
      }
    }
    run(pset1_kernel<Packet>(), n, in, out);
    for (Index k = 0; k < in.size(); ++k) expected[k] = in[k / kSize];
    VERIFY(areEqualBits(expected.data(), out.data(), int(in.size()), false) && "pset1");
    out.setConstant(Scalar(-7));
    run(partial_kernel<Packet>(), n, in, out);
    for (Index k = 0; k < in.size(); ++k) {
      const Index lanes = (k / kSize) % (kSize + 1);
      expected[k] = (k % kSize) < lanes ? in[k] : Scalar(-7);
    }
    VERIFY(areEqualBits(expected.data(), out.data(), int(in.size()), false) && "pload_partial/pstore_partial");
  }
  {
    const int stride = 3;
    Buffer<Scalar> strided(in.size() * stride);
    strided.setConstant(Scalar(-7));
    for (Index k = 0; k < in.size(); ++k) strided[k * stride] = in[k];
    Buffer<Scalar> out(in.size());
    out.setConstant(Scalar(-7));
    run(pgather_kernel<Packet>{stride}, n, strided, out);
    VERIFY(areEqualBits(in.data(), out.data(), int(in.size()), false) && "pgather");
    Buffer<Scalar> scattered(in.size() * stride);
    scattered.setConstant(Scalar(-7));
    run(pscatter_kernel<Packet>{stride}, n, in, scattered);
    VERIFY(areEqualBits(strided.data(), scattered.data(), int(strided.size()), false) && "pscatter");
  }
  {
    // A kSize x kSize block per thread.
    const int blocks = n / kSize;
    const int block_elements = kSize * kSize;
    Buffer<Scalar> out(blocks * block_elements);
    out.setConstant(Scalar(-7));
    run(ptranspose_kernel<Packet>(), blocks, in, out);
    Buffer<Scalar> expected(blocks * block_elements);
    for (int block = 0; block < blocks; ++block) {
      for (int r = 0; r < kSize; ++r) {
        for (int c = 0; c < kSize; ++c) {
          expected[block * block_elements + r * kSize + c] = in[block * block_elements + c * kSize + r];
        }
      }
    }
    VERIFY(areEqualBits(expected.data(), out.data(), int(out.size()), false) && "ptranspose");
  }
  {
    Buffer<Scalar> out(in.size());
    out.setConstant(Scalar(-7));
    run(preverse_kernel<Packet>(), n, in, out);
    Buffer<Scalar> expected(in.size());
    for (Index k = 0; k < in.size(); ++k) expected[k] = in[(k / kSize) * kSize + (kSize - 1 - k % kSize)];
    VERIFY(areEqualBits(expected.data(), out.data(), int(in.size()), false) && "preverse");
  }
}

}  // namespace test
}  // namespace Eigen

#endif  // EIGEN_TEST_PACKETMATH_DATA_MOVEMENT_H
