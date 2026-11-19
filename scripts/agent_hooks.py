#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Adapt the repository's PostToolUse hooks to each agent harness that runs them.

``check_style.py`` and ``clang_tidy_hook.py`` check tool calls in Claude Code's
payload form.  Each takes ``--hook HARNESS``; this module turns that harness's
payload into Claude Code tool calls and the findings into the reply the harness
passes to the model:

  claude   .claude/settings.json   the scripts' native form.  Exit 2 with the
                                   findings on stderr feeds them to the model.
  codex    .codex/hooks.json       the same payload fields, but an edit is one
                                   ``apply_patch`` call whose ``command`` is a
                                   patch that may touch several files.  A
                                   PostToolUse exit 2 replaces the tool's result,
                                   so findings go in ``additionalContext``
                                   beside it and the hook exits 0.

Every hook fails open: with nothing to report, or a payload it cannot read, it
says nothing and exits 0.
"""

import argparse
import json
import os
import re
import sys

HARNESSES = ("claude", "codex")

_PATCH_FILE = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+?)\s*$")
_PATCH_MOVE = "*** Move to: "


def add_hook_arguments(group):
    """Add ``--hook`` and its older spelling ``--claude-hook`` to an argparse group; both set ``args.hook``."""
    group.add_argument("--hook", choices=HARNESSES, help="run as a PostToolUse hook of this agent harness, reading "
                       "the tool call from stdin")
    group.add_argument("--claude-hook", dest="hook", action="store_const", const="claude", help=argparse.SUPPRESS)


def _apply_patch_calls(patch, cwd):
    """Claude Code Write and MultiEdit calls for the files an ``apply_patch`` body adds or updates.

    An update also carries its hunks, as lists of ``' '``/``'+'``/``'-'`` lines, under the ``hunks`` key: they
    locate the added lines in the post-image more reliably than the added text alone.
    """
    files, current = [], None
    for line in patch.splitlines():
        match = _PATCH_FILE.match(line)
        if match:
            action, path = match.groups()
            current = None
            if action != "Delete":
                current = {"path": path, "new": action == "Add", "hunks": [[]]}
                files.append(current)
        elif current is None or line.startswith("*** End"):
            continue
        elif line.startswith(_PATCH_MOVE):
            current["path"] = line[len(_PATCH_MOVE):].strip()
        elif line.startswith("@@"):
            current["hunks"].append([])
        else:
            current["hunks"][-1].append(line if line.startswith(("+", "-", " ")) else " " + line)
    calls = []
    for entry in files:
        path = os.path.join(cwd, entry["path"])
        hunks = [hunk for hunk in entry["hunks"] if hunk]
        if entry["new"]:
            content = "".join(line[1:] + "\n" for hunk in hunks for line in hunk if line.startswith("+"))
            calls.append({"tool_name": "Write", "tool_input": {"file_path": path, "content": content}})
            continue
        runs = []
        for hunk in hunks:
            run = []
            for line in hunk + [" "]:
                if line.startswith("+"):
                    run.append(line[1:])
                elif run:
                    runs.append("\n".join(run))
                    run = []
        calls.append({"tool_name": "MultiEdit", "tool_input": {
            "file_path": path, "edits": [{"new_string": text} for text in runs], "hunks": hunks}})
    return calls


def tool_calls(harness, payload):
    """The Claude Code tool calls ({tool_name, tool_input[, tool_response, cwd]}) that a hook payload describes."""
    if not isinstance(payload, dict):
        return []
    # Codex's matcher aliases Edit and Write name the same apply_patch call.
    if harness == "codex" and payload.get("tool_name") in ("apply_patch", "Edit", "Write"):
        patch = (payload.get("tool_input") or {}).get("command") or ""
        return _apply_patch_calls(patch, payload.get("cwd") or os.getcwd())
    return [payload]


def answer(harness, feedback="", notice=""):
    """Answer a PostToolUse hook: ``feedback`` goes to the model, ``notice`` only to the user.

    Returns the exit status: 2 when Claude Code must pass ``feedback`` on.
    """
    reply = {"systemMessage": notice} if notice else {}
    if harness == "codex" and feedback:
        reply["hookSpecificOutput"] = {"hookEventName": "PostToolUse", "additionalContext": feedback}
        feedback = ""
    if reply:
        sys.stdout.write(json.dumps(reply) + "\n")
    if feedback:
        sys.stderr.write(feedback)
        return 2
    return 0
