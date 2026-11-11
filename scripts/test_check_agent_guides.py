#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for scripts/check_agent_guides.py, on a synthetic tree.

Usage: python3 scripts/test_check_agent_guides.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_agent_guides import (ROOT_GUIDE_MAX_BYTES, check_guide, expand_braces, heading_slugs,  # noqa: E402
                                with_directories)

FILES = {
    "AGENTS.md": "# AGENTS.md\n\n## Scope and precedence\n",
    ".agents/ci.md": "# CI\n\n## Test Tiers On Merge Requests\n\n### `all-tests`, the full tier\n",
    "Eigen/Core": "",
    "Eigen/src/Core/Dot.h": "",
    "ci/build.linux.gitlab-ci.yml": "",
    "test/main.h": "",
    "test/sve_vector_length.cpp": "",
}
PATHS = with_directories(FILES)


def problems(text, rel_path=".agents/testing.md"):
    return [message for _, message in check_guide(rel_path, text, PATHS, FILES.get)]


def test_tree():
    assert {"Eigen", "Eigen/src", "Eigen/src/Core", ".agents"} <= PATHS


def test_links():
    assert not problems("See [`ci.md`](ci.md), [root](../AGENTS.md#scope-and-precedence) and [jobs](../ci).\n")
    assert not problems("[tiers](ci.md#test-tiers-on-merge-requests), [docs](https://example.org/x#y)\n")
    assert problems("See [`gone.md`](gone.md).\n") == ["link target does not exist: gone.md"]
    assert problems("[x](../../outside.md)\n") == ["link target does not exist: ../../outside.md"]
    assert problems("[x](ci.md#no-such-heading)\n") == ["no heading #no-such-heading in .agents/ci.md"]
    # A fragment alone names a heading of the guide itself.
    assert not problems("# T\n\n## Scope\n\n[x](#scope)\n", rel_path="AGENTS.md")
    assert problems("# T\n\n[x](#scope)\n", rel_path="AGENTS.md") == ["no heading #scope in AGENTS.md"]
    # A link quoted as code, or inside a fenced block, is an example.
    assert not problems("Write `[text](#heading)`.\n\n```\n[x](gone.md) `Eigen/Gone`\n```\n")


def test_heading_slugs():
    assert heading_slugs(FILES[".agents/ci.md"]) == {"ci", "test-tiers-on-merge-requests", "all-tests-the-full-tier"}


def test_paths():
    assert not problems("`Eigen/Core`, `Eigen/src/Core/`, `ci/*.gitlab-ci.yml`, `test/main.h`\n")
    # A test target is named for its source file.
    assert not problems("`test/sve_vector_length` fails the run.\n")
    assert problems("`Eigen/src/Core/Gone.h`\n") == ["path does not exist: Eigen/src/Core/Gone.h"]
    assert problems("`ci/*.gone.yml`\n") == ["path does not exist: ci/*.gone.yml"]
    # Not paths in this tree: a placeholder, an elision, a command, a branch, a relative mention.
    assert not problems("`test/<name>.cpp` `unsupported/Eigen/...` `ctest -R x` `origin/master` `arch/AVX/Gone.h`\n")


def test_rule_paths():
    def rule(*globs):
        front = "---\npaths:\n" + "".join('  - "%s"\n' % glob for glob in globs) + "---\n\n"
        return problems(front + "Read [`ci.md`](../../.agents/ci.md).\n", rel_path=".claude/rules/ci.md")

    assert not rule("Eigen/src/Core/**", "ci/*.yml", "**/*.h", "AGENTS.md")
    # As in Claude Code, * stays within one path component; ** crosses them and also matches none.
    assert rule("Eigen/*.h") == ["paths: glob matches no file: Eigen/*.h"]
    assert not rule("Eigen/**/*.h", "**/Core", "Eigen/src/Cor[ef]/Dot.h", "test/main.?")
    assert problems("`Eigen/*/Dot.h`\n") == ["path does not exist: Eigen/*/Dot.h"]
    # Each alternative of a brace list is tried; one match is enough.
    assert expand_braces("a/{b,c}/*{d,e}.h") == ["a/b/*d.h", "a/b/*e.h", "a/c/*d.h", "a/c/*e.h"]
    assert not rule("Eigen/src/{Gone,Core}/**")
    assert rule("Eigen/src/Gone/**", "ci/**") == ["paths: glob matches no file: Eigen/src/Gone/**"]
    # Only a rule file has a paths list.
    assert not problems("---\npaths:\n  - \"Eigen/src/Gone/**\"\n---\n")


def test_root_guide_size():
    text = "# AGENTS.md\n" + "x" * ROOT_GUIDE_MAX_BYTES
    assert any("keep it under" in message for message in problems(text, rel_path="AGENTS.md"))
    assert not problems(text, rel_path=".agents/ci.md")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("PASS %s" % test.__name__)
    print("%d tests passed" % len(tests))
    return 0


if __name__ == "__main__":
    sys.exit(main())
