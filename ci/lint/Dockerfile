# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# CI image for the checkformat lint jobs and select:tests.
#
# Kept outside ci/docker/ on purpose: everything there is hashed into the test
# pass-cache fingerprint and forces the full affected-test suite, and nothing
# in this image can change a test's outcome.
#
# clang-format is built from the LLVM release rather than taken from a package:
# Alpine dropped clang17 after 3.22, and the PyPI musllinux wheel's
# patchelf-rewritten binary does not load under musl.  The static build is
# ~3 MB against ~300 MB for Alpine's clang17-extra-tools.
#
# Rebuild and push when this file changes:
#   docker buildx build --platform linux/amd64 \
#     -t registry.gitlab.com/libeigen/eigen/alpine-lint:latest \
#     --push ci/lint/

FROM alpine:3.24.2 AS clang-format

ARG LLVM_VERSION=17.0.6
# SHA-256 of the release tarball; its SHA-512 matches Alpine's llvm17 APKBUILD.
ARG LLVM_SHA256=58a8818c60e6627064f312dbf46c02d9949956558340938b71cf731ad8bc0813

RUN apk add --no-cache build-base cmake samurai python3 curl

WORKDIR /src
RUN curl -fsSLo llvm.tar.xz \
      https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/llvm-project-${LLVM_VERSION}.src.tar.xz \
    && echo "${LLVM_SHA256}  llvm.tar.xz" | sha256sum -c - \
    && tar -xJf llvm.tar.xz --strip-components=1 \
    && rm llvm.tar.xz

# LLVM 17 predates GCC 15, whose libstdc++ no longer includes <cstdint>
# transitively.
RUN cmake -S llvm -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=MinSizeRel \
      -DCMAKE_CXX_FLAGS="-include cstdint" \
      -DCMAKE_EXE_LINKER_FLAGS="-static -s" \
      -DLLVM_ENABLE_PROJECTS=clang \
      -DLLVM_TARGETS_TO_BUILD=Native \
      -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
      -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_DOCS=OFF \
      -DCLANG_INCLUDE_TESTS=OFF -DCLANG_INCLUDE_DOCS=OFF \
      -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_TERMINFO=OFF \
      -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_LIBEDIT=OFF \
    && samu -C build clang-format \
    && install -D -m 755 build/bin/clang-format /out/clang-format \
    && install -D -m 755 clang/tools/clang-format/git-clang-format /out/git-clang-format

FROM alpine:3.24.2

COPY requirements.txt /tmp/

# After populating the venv, drop pip, ensurepip, and the stdlib's GUI,
# REPL-help, test and build parts.  Pruning git's HTTP transport, OpenSSL and
# apk as well saves 5.6 MB compressed, not worth an image nobody can extend or
# debug with apk.
RUN apk add --no-cache python3 git \
    && python3 -m venv /opt/lint \
    && /opt/lint/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    && /opt/lint/bin/pip uninstall -q -y pip \
    && py=$(echo /usr/lib/python3.*) && dyn=$py/lib-dynload \
    && rm -rf $py/ensurepip $py/idlelib $py/tkinter $py/turtledemo $py/turtle.py \
      $py/pydoc_data $py/test $py/config-3.* $py/__phello__ \
      $dyn/_test*.so $dyn/_ctypes_test*.so $dyn/xxlimited*.so $dyn/_xxtestfuzz*.so \
      /tmp/requirements.txt

COPY --from=clang-format /out/ /usr/local/bin/

ENV PATH=/opt/lint/bin:$PATH

# Fail the build, not every merge request's lint job, if the pruning above
# removed something a tool needs.  The clang-format-17 name lets the commands
# in .agents/ci.md run unchanged inside the image.
RUN ln -s clang-format /usr/local/bin/clang-format-17 \
    && clang-format --version && git --version && codespell --version && reuse --version \
    && vermin --version && apk --version \
    && python3 -c 'import hashlib, json, subprocess' \
    && for f in /usr/bin/* /usr/libexec/git-core/* /usr/lib/python3.*/lib-dynload/*.so; do \
         if /lib/ld-musl-x86_64.so.1 --list "$f" 2>&1 | grep -q 'Error loading'; then \
           echo "missing shared library: $f"; exit 1; fi; \
       done
