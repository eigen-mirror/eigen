#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for scripts/agent_hooks.py and the hook paths built on it.

The Codex payloads and replies follow the forms its hook documentation gives:
``tool_input.command`` carries the patch of an ``apply_patch`` call, and a
PostToolUse hook adds to the tool result through ``additionalContext``.

Usage: python3 scripts/test_agent_hooks.py
"""

import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_hooks import HARNESSES, answer, tool_calls  # noqa: E402
from style_common import added_from_hunks  # noqa: E402
from test_check_style import commit_repository, run_script  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS)


def codex(tool, command, cwd=REPO_ROOT):
    return {"hook_event_name": "PostToolUse", "cwd": cwd, "tool_name": tool, "tool_input": {"command": command}}


def replies(call):
    """(exit status, stdout, stderr) of ``call()``."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = call()
    return code, out.getvalue(), err.getvalue()


def test_codex_tool_calls():
    shell = codex("Bash", "ls")
    assert tool_calls("codex", shell) == [shell]
    patch = "\n".join([
        "*** Begin Patch",
        "*** Add File: test/probe.cpp",
        "+int a;",
        "+int b;",
        "*** Update File: Eigen/src/Core/Dot.h",
        "*** Move to: Eigen/src/Core/Dot2.h",
        "@@ namespace Eigen {",
        " int keep;",
        "-int old;",
        "+int new_one;",
        "+int new_two;",
        " int also_keep;",
        "",
        "+int tail;",
        "*** Delete File: Eigen/src/Core/Gone.h",
        "*** End Patch",
    ])
    added, updated = tool_calls("codex", codex("apply_patch", patch, cwd="/repo"))
    assert added == {"tool_name": "Write", "tool_input": {"file_path": "/repo/test/probe.cpp",
                                                          "content": "int a;\nint b;\n"}}, added
    assert updated["tool_input"]["file_path"] == "/repo/Eigen/src/Core/Dot2.h", updated
    # A blank line inside a hunk is empty context.
    assert updated["tool_input"]["hunks"] == [[" int keep;", "-int old;", "+int new_one;", "+int new_two;",
                                               " int also_keep;", " ", "+int tail;"]], updated
    assert updated["tool_input"]["edits"] == [{"new_string": "int new_one;\nint new_two;"},
                                              {"new_string": "int tail;"}], updated
    # Codex's matcher aliases name the same patch.
    assert len(tool_calls("codex", codex("Write", patch, cwd="/repo"))) == 2
    # Claude Code's own Write call is passed through, not parsed as a patch.
    write = {"tool_name": "Write", "tool_input": {"file_path": "/repo/a.h", "content": "int a;\n"}}
    assert tool_calls("claude", write) == [write]
    assert tool_calls("codex", "not a payload") == []


def test_added_from_hunks():
    lines = ["int keep;", "int x;", "int keep;", "int new_one;", "int also_keep;"]
    assert added_from_hunks(lines, [[" int keep;", "-int old;", "+int new_one;", " int also_keep;"]]) == {4}
    assert added_from_hunks(lines, [["+int x;"]]) == {2}
    # Context that recurs locates nothing; the caller falls back to the added text.
    assert added_from_hunks(lines, [[" int keep;"]]) is None
    assert added_from_hunks(lines, [["+int absent;"]]) is None
    # A deletion-only hunk adds nothing and does not fail the others.
    assert added_from_hunks(lines, [["-int gone;"], ["+int new_one;"]]) == {4}


def test_answers():
    assert replies(lambda: answer("claude", "finding\n")) == (2, "", "finding\n")
    # A Codex PostToolUse that exits 2 replaces the tool's result; added context keeps it.
    code, out, err = replies(lambda: answer("codex", "finding\n"))
    assert code == 0 and not err and json.loads(out) == {"hookSpecificOutput": {
        "hookEventName": "PostToolUse", "additionalContext": "finding\n"}}, out
    for harness in HARNESSES:
        code, out, err = replies(lambda: answer(harness, notice="not checked"))
        assert code == 0 and json.loads(out) == {"systemMessage": "not checked"} and not err
        assert replies(lambda: answer(harness)) == (0, "", "")


def test_scripts_answer_each_harness():
    def commit_hook(args):
        # A fresh repository each time: a hook reports a given commit only once.
        with tempfile.TemporaryDirectory(prefix="agent_hooks_test_") as tmp:
            commit_repository(tmp)
            payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "git commit -m change"}, "cwd": tmp})
            return run_script("check_style.py", args, payload)

    code, out, err = commit_hook(["--hook", "claude"])
    assert code == 2 and not out and "bool_constant" in err and "doc/Page.dox:2" in err, (code, out, err)
    code, out, err = commit_hook(["--hook", "codex"])
    context = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert code == 0 and not err and "bool_constant" in context and "\\ref target" in context, (code, out, err)
    # The older spelling still selects Claude Code.
    assert commit_hook(["--claude-hook"])[0] == 2
    # With nothing to say, including a payload that cannot be read, each hook is silent and exits 0.
    for script in ("check_style.py", "clang_tidy_hook.py"):
        for harness in HARNESSES:
            assert run_script(script, ["--hook", harness], "not json") == (0, "", ""), script
            assert run_script(script, ["--hook", harness], json.dumps(codex("Bash", "ls"))) == (0, "", ""), script


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("PASS %s" % test.__name__)
    print("%d tests passed" % len(tests))
    return 0


if __name__ == "__main__":
    sys.exit(main())
