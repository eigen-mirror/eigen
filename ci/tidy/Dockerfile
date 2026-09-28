# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# CI image for checkformat:clangtidy.
#
# Kept outside ci/docker/ on purpose: everything there is hashed into the test
# pass-cache fingerprint and forces the full affected-test suite, and nothing
# in this image can change a test's outcome.
#
# The job used to install these packages on every run; baking them in keeps
# its clang-tidy the same 18.1.3 build.  clang is needed alongside clang-tidy
# to configure the compilation database and to probe drivers for missing
# third-party includes.  Alpine's clang18 was tried: its LLVM libraries are
# larger than Ubuntu's, clang-tidy ran 23-45% slower on musl, and musl's
# NULL is nullptr, which hides NULL from modernize-use-nullptr.
#
# Any change to this file bumps the tag of EIGEN_CI_IMAGE_LINUX_TIDY in
# .gitlab-ci.yml: runners pull if-not-present, so one that holds a tag never
# sees a rebuild pushed under it.  Then build the new tag with the manual
# build:docker:ubuntu-24.04-tidy job in the merge request's pipeline, or:
#   docker buildx build --platform linux/amd64 \
#     -t registry.gitlab.com/libeigen/eigen/ubuntu-24.04-tidy:<tag> \
#     --push ci/tidy/

FROM ubuntu:24.04

# After installing, drop what no job reaches:
#  - every LLVM tool but clang, clang-tidy and clang-scan-deps (which CMake
#    records when configuring), and libclang's C API;
#  - GCC's sanitizer runtimes and the static C and C++ libraries: CMake's
#    checks link dynamically, which needs only libc_nonshared.a and libgcc.a;
#  - binutils' gold and gprof, the Polly plugin, documentation, CMake's help,
#    cpack and ctest.
RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git cmake ninja-build clang-tidy clang python3 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /var/log/* \
    && find /usr/lib/llvm-18/bin -mindepth 1 \
      ! -name clang ! -name clang++ ! -name clang-18 ! -name clang-tidy \
      ! -name clang-scan-deps -delete \
    && find /usr/bin -xtype l -delete \
    && gcc=/usr/lib/gcc/x86_64-linux-gnu/13 && lib=/usr/lib/x86_64-linux-gnu \
    && rm -rf $lib/libclang-18.so* /usr/lib/llvm-18/lib/libclang.so* \
      /usr/lib/llvm-18/lib/LLVMPolly.so \
      $gcc/*san* $gcc/libstdc++.a $gcc/libstdc++fs.a $gcc/libsupc++.a $lib/lib*san.so* \
      $lib/libc.a $lib/libm.a $lib/libm-*.a $lib/libmvec.a $lib/libcrypt.a \
      $lib/libgprofng.so* /usr/bin/*gold /usr/bin/*gprof* /usr/bin/*-gp-* \
      /usr/share/doc /usr/share/man /usr/share/info /usr/share/lintian \
      /usr/share/cmake-*/Help /usr/bin/cpack /usr/bin/ctest

# Fail the build, not every merge request's job, if the pruning above removed
# something a tool needs.
RUN clang-tidy --version | grep -q 'LLVM version 18\.1\.3' \
    && git --version && cmake --version \
    && for f in /usr/bin/* /usr/lib/llvm-18/bin/* /usr/lib/git-core/*; do \
         if ldd "$f" 2>/dev/null | grep -q 'not found'; then \
           echo "missing shared library: $f"; exit 1; fi; \
       done \
    && printf '#include <immintrin.h>\n#include <iostream>\nint main() { std::cout << "ok\\n"; }\n' > /tmp/t.cpp \
    && clang++ -std=c++14 -o /tmp/t /tmp/t.cpp && /tmp/t \
    && clang-tidy -checks='-*,bugprone-*' /tmp/t.cpp -- -std=c++14 \
    && rm /tmp/t /tmp/t.cpp
