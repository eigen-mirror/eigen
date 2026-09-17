// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "packetmath_test_shared.h"
#include <memory>

namespace Eigen {
namespace test {

template <typename Scalar, typename Packet>
struct runall<Scalar, Packet, false, false> {
  static void run() {
    constexpr int PacketSize = internal::unpacket_traits<Packet>::size;
    auto samples = special_values<Scalar>();
    using Bits = typename numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type;
    const Bits nan_payload = numext::bit_cast<Bits>(NumTraits<Scalar>::quiet_NaN()) | Bits(0x123);
    samples.push_back(numext::bit_cast<Scalar>(nan_payload));
    Scalar actual[PacketSize], expected[PacketSize];
    for (Index stride : {Index(-7), Index(-2), Index(-1), Index(0), Index(1), Index(2), Index(3), Index(7)}) {
      const Index span = (PacketSize - 1) * numext::abs(stride) + 1;
      for (Index offset = 0; offset < PacketSize; ++offset) {
        // Exact-size new[] lets sanitizers detect reads past the last gathered coefficient.
        std::unique_ptr<Scalar[]> storage(new Scalar[span + offset]);
        Scalar* data = storage.get() + offset;
        const Scalar* from = stride < 0 ? data + span - 1 : data;
        for (Index rotation = 0; rotation < Index(samples.size()); ++rotation) {
          for (Index i = 0; i < span; ++i) data[i] = samples[(i + rotation) % samples.size()];
          for (Index i = 0; i < PacketSize; ++i) expected[i] = from[i * stride];
          volatile Index runtime_stride = stride;
          internal::pstoreu(actual, internal::pgather<Scalar, Packet>(from, runtime_stride));
          VERIFY(areEqualBits(expected, actual, PacketSize, false));
          if (stride == 2) {
            internal::pstoreu(actual, internal::pgather<Scalar, Packet>(from, 2));
            VERIFY(areEqualBits(expected, actual, PacketSize, false));
          }
        }
      }
    }
  }
};

}  // namespace test
}  // namespace Eigen

EIGEN_DECLARE_TEST(packetmath_gather) {
  CALL_SUBTEST_1(test::runner<float>::run());
  CALL_SUBTEST_2(test::runner<double>::run());
}
