// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <Eigen/Core>

struct TestPacket {
  explicit TestPacket(float x) : value(x) {}
  float value;
};

namespace Eigen {
namespace internal {
template <>
struct unpacket_traits<TestPacket> {
  using type = float;
  static constexpr int size = 4;
};
template <>
TestPacket pset1<TestPacket>(const float& x) {
  return TestPacket(x);
}
}  // namespace internal
}  // namespace Eigen

int main() {
  float values[] = {1, 2};
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  return Eigen::internal::ploaddup<TestPacket>(values).value != values[0];
#else
  return Eigen::internal::ploaddup<float>(values) != values[0];
#endif
}
