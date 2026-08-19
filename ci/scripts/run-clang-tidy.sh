#!/bin/bash
#
# Run clang-tidy on files changed in the current MR.
#
# Usage: run-clang-tidy.sh <base_sha> <build_dir>
#
# <base_sha>   The merge-base commit to diff against.
# <build_dir>  Path to a CMake build directory containing compile_commands.json.
#
# For header files under Eigen/src/<Module>/, the script generates a driver
# that includes the parent module header first, so InternalHeaderCheck.h does
# not #error out, and then the changed header itself. The explicit second
# include covers new headers that are not exported by the umbrella yet. The
# umbrella name is read from the header's own `#error "Please include <X>"`
# directive (or a sibling InternalHeaderCheck.h), with a fallback to the
# heuristic <root>/<Module> for deeply-nested files (e.g. arch-specific
# backends) that don't carry their own directive.
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

BASE_SHA="${1:?Usage: run-clang-tidy.sh <base_sha> <build_dir>}"
BUILD_DIR="${2:?Usage: run-clang-tidy.sh <base_sha> <build_dir>}"

if [ ! -f "${BUILD_DIR}/compile_commands.json" ]; then
  echo "ERROR: ${BUILD_DIR}/compile_commands.json not found."
  echo "Run cmake with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON first."
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"

# External-dependency modules that require third-party headers we don't
# install in the clang-tidy CI image. The umbrella exists, but `#include`-ing
# it would fail at preprocessor time (e.g. cholmod.h not found).
EXTERNAL_DEP_MODULES="AccelerateSupport|CholmodSupport|KLUSupport|MetisSupport|PaStiXSupport|PardisoSupport|SPQRSupport|SuperLUSupport|UmfPackSupport"

# Get changed files (Added, Modified, Renamed) without losing whitespace in
# repository paths.
mapfile -d '' -t CHANGED_FILES < <(git diff --name-only -z --diff-filter=AMR "${BASE_SHA}" HEAD)

if [ "${#CHANGED_FILES[@]}" -eq 0 ]; then
  echo "No changed files to check."
  exit 0
fi

TIDY_TMPDIR=$(mktemp -d)
trap 'rm -rf "${TIDY_TMPDIR}"' EXIT

ERRORS=0
# Generated drivers live outside the checkout, where clang-tidy cannot discover
# Eigen's configuration.  The job is advisory (`allow_failure`), so promoting
# warnings makes GitLab mark findings without blocking the pipeline.
TIDY_ARGS=(
  "--config-file=${REPO_ROOT}/.clang-tidy"
  "--warnings-as-errors=*"
)

# Determine which umbrella header to include when linting a given source-tree
# header. The source of truth is the `#error "Please include <X>"` directive
# carried either by the header itself (e.g. Eigen/src/StlSupport/StdDeque.h
# -> Eigen/StdDeque) or by its sibling InternalHeaderCheck.h (the common
# case for Eigen/src/<Module>/*.h).
module_include_for_header() {
  local header="$1"
  local module
  local hint
  local candidate

  # Restrict to header files inside the src trees.
  if [[ "${header}" =~ ^Eigen/src/([^/]+)/ ]]; then
    module="${BASH_REMATCH[1]}"
  elif [[ "${header}" =~ ^unsupported/Eigen/src/([^/]+)/ ]]; then
    module="${BASH_REMATCH[1]}"
  else
    return 1
  fi

  # Modules whose umbrella requires a third-party library we don't install.
  if [[ "${module}" =~ ^(${EXTERNAL_DEP_MODULES})$ ]]; then
    return 1
  fi

  # Parse `#error "Please include <X>"` from the header or its sibling
  # InternalHeaderCheck.h.
  for candidate in "${REPO_ROOT}/${header}" \
                   "${REPO_ROOT}/$(dirname "${header}")/InternalHeaderCheck.h"; do
    if [ -f "${candidate}" ]; then
      hint=$(grep "Please include" "${candidate}" 2>/dev/null \
             | sed -nE 's/.*"Please include ([^ "]+).*/\1/p' \
             | head -n1)
      if [ -n "${hint}" ] && [ -f "${REPO_ROOT}/${hint}" ]; then
        echo "${hint}"
        return 0
      fi
    fi
  done

  # Fallback: route through <root>/<Module> if it exists. This catches files
  # nested deeper than the module's top-level src/ (e.g. arch-specific
  # backends under Eigen/src/Core/arch/<ISA>/) that don't carry their own
  # `#error` directive.
  if [[ "${header}" =~ ^unsupported/ ]]; then
    hint="unsupported/Eigen/${module}"
  else
    hint="Eigen/${module}"
  fi
  if [ -f "${REPO_ROOT}/${hint}" ]; then
    echo "${hint}"
    return 0
  fi

  # No parseable directive and no matching umbrella file — likely a
  # utility/details file shared across umbrellas (e.g. StlSupport/details.h).
  # Skip silently.
  return 1
}

# A split test contributes one compilation-database entry per EIGEN_TEST_PART,
# and clang-tidy parses the file once for every entry that names it: 17 for
# test/eigensolver_selfadjoint.cpp, 41 for test/array_cwise.cpp, at roughly a
# minute and 3 GB each. That exhausts the job timeout on a single file and
# leaves every later file unchecked, silently. Narrow the database to the
# entries the changed lines need, which scripts/tidy_compile_db.py selects:
# one per distinct compiler configuration, and within a configuration split
# into parts, the parts that compile the added lines.
#
# Writes the reduced database to <outdir>/compile_commands.json, reports on
# stdout what it left out, and succeeds only when the file is present in the
# full database. CMake normally records absolute source paths, so establishing
# membership takes path resolution rather than a textual search.
reduced_database() {
  python3 "${REPO_ROOT}/scripts/tidy_compile_db.py" \
          "${BUILD_DIR}/compile_commands.json" "${REPO_ROOT}/$1" "$2" "$3"
}

# Restrict diagnostics to the lines this merge request adds. Without it the
# style checks in .clang-tidy (modernize-use-nullptr, modernize-use-using)
# would report every pre-existing occurrence in a touched file — Eigen/src
# holds ~4100 typedefs — rather than the ones under review. An empty filter
# means "no filtering" to clang-tidy, so a file with no added lines is skipped
# rather than linted whole.
line_filter_for() {
  python3 "${REPO_ROOT}/scripts/style_common.py" --line-filter "${BASE_SHA}" "$1"
}

echo "Checking changed files with clang-tidy..."
echo "Base SHA: ${BASE_SHA}"
echo ""

for file in "${CHANGED_FILES[@]}"; do
  LINE_FILTER=$(line_filter_for "${file}")
  if [ -z "${LINE_FILTER}" ] || [ "${LINE_FILTER}" = "[]" ]; then
    # Renamed or mode-only change: nothing added to report on.
    continue
  fi

  # Only check C++ source and header files.
  case "${file}" in
    failtest/*.cpp)
      # The compilation database carries both the successful and intentionally
      # broken variants.  Parse the ordinary variant directly so the _ko
      # command cannot turn every changed failtest into a clang diagnostic.
      echo "=== ${file} ==="
      if ! clang-tidy \
            "${TIDY_ARGS[@]}" \
            --line-filter="${LINE_FILTER}" \
            "${file}" \
            -- -std=c++14 -I"${REPO_ROOT}" 2>&1; then
        ERRORS=$((ERRORS + 1))
      fi
      ;;
    *.cpp|*.cc|*.cxx)
      # Source file: run clang-tidy directly if it's in the compilation database.
      FILE_DB="${TIDY_TMPDIR}/db_${file//\//_}"
      if SELECTION=$(reduced_database "${file}" "${FILE_DB}" "${LINE_FILTER}"); then
        echo "=== ${file} ===${SELECTION:+ ${SELECTION}}"
        if ! clang-tidy \
              -p "${FILE_DB}" \
              "${TIDY_ARGS[@]}" \
              --line-filter="${LINE_FILTER}" \
              "${file}" 2>&1; then
          ERRORS=$((ERRORS + 1))
        fi
      else
        STATUS=$?
        if [ "${STATUS}" -gt 1 ]; then
          exit "${STATUS}"
        fi
      fi
      ;;
    *.h|*.hpp)
      # Header file: include the right module first, then force the changed
      # header into the translation unit even if the umbrella omits it.
      MODULE_INCLUDE=$(module_include_for_header "${file}" || true)
      if [ -z "${MODULE_INCLUDE}" ]; then
        # Not a recognized module header or in skip list.
        continue
      fi

      DRIVER="${TIDY_TMPDIR}/tidy_driver_${file//\//_}.cpp"
      cat > "${DRIVER}" <<EOF
#include <${MODULE_INCLUDE}>
#include <${file}>
EOF

      echo "=== ${file} (via ${MODULE_INCLUDE}) ==="
      if ! clang-tidy \
            "${TIDY_ARGS[@]}" \
            --header-filter="$(echo "${file}" | sed 's/[.[\*^$()+?{|]/\\&/g')" \
            --line-filter="${LINE_FILTER}" \
            "${DRIVER}" \
            -- -std=c++14 -I"${REPO_ROOT}" 2>&1; then
        ERRORS=$((ERRORS + 1))
      fi
      ;;
  esac
done

if [ ${ERRORS} -gt 0 ]; then
  echo ""
  echo "clang-tidy reported issues in ${ERRORS} file(s)."
  exit 1
else
  echo ""
  echo "clang-tidy: all clean."
  exit 0
fi
