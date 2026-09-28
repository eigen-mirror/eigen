#!/bin/bash
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# Provision and verify compiler cache tools (sccache, ccache).
# Pinned release binaries are verified against strict SHA-256 checksums
# on both download and cache restoration to protect against supply-chain tampering.

root_dir="${rootdir:-$(pwd)}"
arch="$(uname -m)"

download_file() {
  local url="$1"
  local dest="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --connect-timeout 5 -m 30 "${url}" -o "${dest}" && return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -q --timeout=30 -O "${dest}" "${url}" && return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "${url}" "${dest}" && return 0
  fi
  echo "Notice: Neither curl, wget, nor python3 is available to download $(basename "${dest}")." >&2
  return 1
}

# 1. Provision sccache
sccache_bin="$(command -v sccache 2>/dev/null || true)"

if [[ -z "${sccache_bin}" ]]; then
  sccache_ver="0.18.0"
  sccache_bindir="${root_dir}/.sccache-bin/${arch}"
  sccache_tar_sha256=""
  sccache_bin_sha256=""

  case "${arch}" in
    x86_64)
      sccache_tar_sha256="45f1447fbe231e3037bde351ef70677dd212216c8d62ae7ca409fecc4d6acc89"
      sccache_bin_sha256="973cb15f6a986d84ca334bbed3bbe2eb8f1ee8fd81bf9e115b8539a293bf8d59"
      ;;
    aarch64)
      sccache_tar_sha256="2b3284d5da3b46a47dc4229e75bb7b88ac4aa99c8d754fb7d2f84997e5a4354a"
      sccache_bin_sha256="0fd82ece4469b791e30fffdd43d58d2e6cae303950c0d6c8faad932d93e35cf4"
      ;;
  esac

  if [[ -n "${sccache_bin_sha256}" ]]; then
    mkdir -p "${sccache_bindir}"
    cached_sccache="${sccache_bindir}/sccache"
    if [[ -x "${cached_sccache}" ]] && echo "${sccache_bin_sha256}  ${cached_sccache}" | sha256sum -c - >/dev/null 2>&1; then
      sccache_bin="${cached_sccache}"
    else
      rm -f "${cached_sccache}"
      archive="${sccache_bindir}/sccache.tar.gz"
      if download_file \
        "https://github.com/mozilla/sccache/releases/download/v${sccache_ver}/sccache-v${sccache_ver}-${arch}-unknown-linux-musl.tar.gz" \
        "${archive}"; then
        if echo "${sccache_tar_sha256}  ${archive}" | sha256sum -c - >/dev/null 2>&1; then
          tar -xzf "${archive}" -C "${sccache_bindir}" --strip-components=1 2>/dev/null || true
          chmod +x "${cached_sccache}" 2>/dev/null || true
          if echo "${sccache_bin_sha256}  ${cached_sccache}" | sha256sum -c - >/dev/null 2>&1; then
            sccache_bin="${cached_sccache}"
          else
            echo "Warning: Extracted sccache failed SHA-256 verification; discarding." >&2
            rm -f "${cached_sccache}"
          fi
        else
          echo "Warning: Downloaded sccache archive failed SHA-256 verification; discarding." >&2
        fi
        rm -f "${archive}"
      fi
    fi
  fi
fi

# 2. Provision ccache
ccache_bin="$(command -v ccache 2>/dev/null || true)"

if [[ -z "${ccache_bin}" ]]; then
  ccache_ver="4.13.6"
  ccache_bindir="${root_dir}/.ccache-bin/${arch}"
  ccache_tar_sha256=""
  ccache_bin_sha256=""

  case "${arch}" in
    x86_64)
      ccache_tar_sha256="09e0547a0c3b250a76675c33130366f1399f3580842fb360c052520d56214ead"
      ccache_bin_sha256="c4ca67395c175798c588ee5425c135e76bee5701add4d0631d8b4be0b161b322"
      ;;
    aarch64)
      ccache_tar_sha256="bff0e0c19165db8627c85c36b0885b3b180659eda67a028298d26565aad52f56"
      ccache_bin_sha256="e678bcfe84a0fea0865307dca3f8ac38f4c881125afea91e3a9397ed3a087666"
      ;;
  esac

  if [[ -n "${ccache_bin_sha256}" ]]; then
    mkdir -p "${ccache_bindir}"
    cached_ccache="${ccache_bindir}/ccache"
    if [[ -x "${cached_ccache}" ]] && echo "${ccache_bin_sha256}  ${cached_ccache}" | sha256sum -c - >/dev/null 2>&1; then
      ccache_bin="${cached_ccache}"
    else
      rm -f "${cached_ccache}"
      archive="${ccache_bindir}/ccache.tar.gz"
      if download_file \
        "https://github.com/ccache/ccache/releases/download/v${ccache_ver}/ccache-${ccache_ver}-linux-${arch}-musl-static.tar.gz" \
        "${archive}"; then
        if echo "${ccache_tar_sha256}  ${archive}" | sha256sum -c - >/dev/null 2>&1; then
          tar -xzf "${archive}" -C "${ccache_bindir}" --strip-components=1 2>/dev/null || true
          chmod +x "${cached_ccache}" 2>/dev/null || true
          if echo "${ccache_bin_sha256}  ${cached_ccache}" | sha256sum -c - >/dev/null 2>&1; then
            ccache_bin="${cached_ccache}"
          else
            echo "Warning: Extracted ccache failed SHA-256 verification; discarding." >&2
            rm -f "${cached_ccache}"
          fi
        else
          echo "Warning: Downloaded ccache archive failed SHA-256 verification; discarding." >&2
        fi
        rm -f "${archive}"
      fi
    fi
  fi
fi
