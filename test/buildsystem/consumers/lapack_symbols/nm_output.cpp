// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <fstream>
#include <iostream>

// Emit a stored nm listing so each host can check decorated and undecorated symbols.
int main(int argc, char** argv) {
  if (argc < 2) return 1;
  std::ifstream listing(argv[argc - 1]);
  if (!listing) return 1;
  std::cout << listing.rdbuf();
  return std::cout ? 0 : 1;
}
