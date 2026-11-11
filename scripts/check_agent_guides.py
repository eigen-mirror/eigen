#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Check that the agent guidance still describes the tree.

``AGENTS.md``, the guides under ``.agents/`` and the per-tool pointers under
``.claude/`` name files by path.  A guide that names a moved or deleted file
sends every reader to the wrong place, and nothing else notices.  This script
checks, for each guidance file:

  - that the target of every relative Markdown link exists, and that a
    ``#fragment`` into a Markdown file names one of its headings;
  - that every inline code span which is a repository path exists.  A span is
    taken as a path when its first component is a top-level entry of the tree;
    wildcards are allowed, a placeholder (``<name>``, ``...``) exempts the span,
    and ``test/foo`` may name ``test/foo.cpp``;
  - that ``AGENTS.md`` stays under the size at which a harness stops loading
    project instructions;
  - that every glob in the ``paths:`` list of a ``.claude/rules/`` pointer
    matches a file, so that a pointer does not outlive the files it is for.

Fenced code blocks are not checked.  Usage:

    python3 scripts/check_agent_guides.py [ROOT]

Exit status 1 if anything is unresolved.
"""

import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUIDE_PATTERNS = ("AGENTS.md", ".agents/**/*.md", ".claude/rules/*.md", ".claude/agents/*.md")
RULES_TREE = ".claude/rules/"
ROOT_GUIDE = "AGENTS.md"
# Codex stops adding project instruction files at 32 KiB by default
# (project_doc_max_bytes), and a user-level file counts toward the same limit.
ROOT_GUIDE_MAX_BYTES = 24 * 1024

URL_SCHEME = re.compile(r"[a-z][a-z0-9+.-]*:")
LINK = re.compile(r"\]\(([^)\s]+)\)")
CODE_SPAN = re.compile(r"`([^`\n]+)`")
PATH_SPAN = re.compile(r"^[\w.-]+(?:/[\w.*\[\]-]+)+/?$")
HEADING = re.compile(r"^#+\s+(.*?)\s*$")
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
RULE_PATH = re.compile(r"^\s+-\s+\"([^\"]+)\"\s*$", re.M)
BRACES = re.compile(r"\{([^{}]*)\}")


def tree_paths(root):
    """Every file git tracks or would track under ``root``, plus each ancestor directory.

    Untracked files count, so that a new guide and the new files it names are checked before ``git add``; a guide
    that names a file never added therefore passes here and fails in CI.
    """
    listing = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root,
                             capture_output=True, text=True, check=True).stdout.splitlines()
    return with_directories(path for path in listing if os.path.exists(os.path.join(root, path)))


def with_directories(files):
    paths = set()
    for path in files:
        while path and path not in paths:
            paths.add(path)
            path = os.path.dirname(path)
    return paths


def heading_slugs(text):
    """The anchors GitLab gives the headings of a Markdown file."""
    slugs = set()
    for line in unfenced(text):
        match = HEADING.match(line)
        if match:
            slug = re.sub(r"[^\w\- ]", "", match.group(1).lower())
            slugs.add(re.sub(r"-+", "-", slug.replace(" ", "-")))
    return slugs


def unfenced(text):
    """The lines of ``text`` outside fenced code blocks; fenced lines become empty to keep the numbering."""
    lines, fenced = [], False
    for line in text.splitlines():
        fence = line.lstrip().startswith(("```", "~~~"))
        fenced = fenced != fence
        lines.append("" if fenced or fence else line)
    return lines


def glob_filter(paths, pattern):
    """The ``paths`` that ``pattern`` matches as Claude Code and Git match a
    glob: ``*``, ``?`` and ``[...]`` stay within one path component, and
    ``**`` crosses components (``**/`` also matches none)."""
    rx, at = "", 0
    while at < len(pattern):
        if pattern.startswith("**/", at):
            rx, at = rx + "(?:.*/)?", at + 3
        elif pattern.startswith("**", at):
            rx, at = rx + ".*", at + 2
        elif pattern[at] == "*":
            rx, at = rx + "[^/]*", at + 1
        elif pattern[at] == "?":
            rx, at = rx + "[^/]", at + 1
        elif pattern[at] == "[" and "]" in pattern[at + 2:]:
            end = pattern.index("]", at + 2)
            body = pattern[at + 1:end]
            rx, at = rx + "[" + ("^" + body[1:] if body.startswith("!") else body) + "]", end + 1
        else:
            rx, at = rx + re.escape(pattern[at]), at + 1
    compiled = re.compile(rx + r"\Z")
    return [path for path in paths if compiled.match(path)]


def exists(path, paths):
    path = path.rstrip("/")
    if path in paths:
        return True
    if any(c in path for c in "*?["):
        return bool(glob_filter(paths, path))
    return any(other.startswith(path + ".") for other in paths)  # a target named for its source file


def expand_braces(pattern):
    """The patterns that ``pattern`` stands for once each ``{a,b}`` is expanded."""
    match = BRACES.search(pattern)
    if not match:
        return [pattern]
    return [expanded for choice in match.group(1).split(",")
            for expanded in expand_braces(pattern[:match.start()] + choice + pattern[match.end():])]


def rule_paths(text):
    """The globs in the ``paths:`` list of a rule file's front matter."""
    front = FRONTMATTER.match(text)
    return RULE_PATH.findall(front.group(1)) if front else []


def check_guide(rel_path, text, paths, read):
    """Return [(line_no, message)] for one guidance file.

    ``paths`` is the tree as ``with_directories`` returns it, and ``read(rel_path)`` returns a file's text.
    """
    problems = []
    top_level = {path.split("/")[0] for path in paths}
    here = os.path.dirname(rel_path)
    for line_no, line in enumerate(unfenced(text), 1):
        for target in LINK.findall(CODE_SPAN.sub("", line)):
            if URL_SCHEME.match(target):
                continue
            path, _, fragment = target.partition("#")
            resolved = os.path.normpath(os.path.join(here, path)) if path else rel_path
            if resolved.startswith("..") or not exists(resolved, paths):
                problems.append((line_no, "link target does not exist: %s" % target))
            elif fragment and resolved.endswith(".md") and fragment not in heading_slugs(
                    text if resolved == rel_path else read(resolved)):
                problems.append((line_no, "no heading #%s in %s" % (fragment, resolved)))
        for span in CODE_SPAN.findall(line):
            if (PATH_SPAN.match(span) and ".." not in span and span.split("/")[0] in top_level
                    and not exists(span, paths)):
                problems.append((line_no, "path does not exist: %s" % span))
    if rel_path.startswith(RULES_TREE):
        for pattern in rule_paths(text):
            if not any(glob_filter(paths, glob) for glob in expand_braces(pattern)):
                problems.append((1, "paths: glob matches no file: %s" % pattern))
    size = len(text.encode("utf-8"))
    if rel_path == ROOT_GUIDE and size > ROOT_GUIDE_MAX_BYTES:
        problems.append((1, "%d bytes; keep it under %d so that every harness loads all of it, and move detail "
                            "to a guide under .agents/" % (size, ROOT_GUIDE_MAX_BYTES)))
    return problems


def main():
    root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT
    paths = tree_paths(root)

    def read(rel_path):
        with open(os.path.join(root, rel_path), encoding="utf-8") as handle:
            return handle.read()

    guides = sorted(set().union(*(glob_filter(paths, pattern) for pattern in GUIDE_PATTERNS)))
    count = 0
    for rel_path in guides:
        for line_no, message in check_guide(rel_path, read(rel_path), paths, read):
            print("%s:%d: %s" % (rel_path, line_no, message))
            count += 1
    if count:
        print("\n%d unresolved reference(s) in the agent guidance. Update the guide to match the tree." % count)
    else:
        print("%d guidance files checked." % len(guides))
    return 1 if count else 0


if __name__ == "__main__":
    sys.exit(main())
