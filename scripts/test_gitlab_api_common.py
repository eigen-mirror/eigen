#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the paginated GET behind scripts/gitlab_api_{mrs,issues}.py.

`requests` is replaced by a scripted fake before the scripts are imported, so
nothing here contacts gitlab.com.  The fake has no `head`: a request that
needs one fails the test.

Usage: python3 scripts/test_gitlab_api_common.py
"""

import contextlib
import io
import os
import sys
import types

# Replace requests before the scripts import it: the real one would be used for
# real network traffic.
exceptions = types.SimpleNamespace(RequestException=type("RequestException", (Exception,), {}))
fake_requests = types.ModuleType("requests")
fake_requests.exceptions = exceptions
sys.modules["requests"] = fake_requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gitlab_api_issues
import gitlab_api_mrs

FAILURES = []
MAX_CALLS = 200  # a retry loop with no bound runs into this


def check(condition, message):
    if condition:
        return
    FAILURES.append(message)
    print("FAIL: %s" % message)


class Response(object):
    def __init__(self, status, body=None, next_page="", retry_after=None):
        self.status_code = status
        self._body = body if body is not None else []
        self.headers = {"x-next-page": next_page}
        if retry_after is not None:
            self.headers["retry-after"] = str(retry_after)

    def json(self):
        return self._body


class Script(object):
    """Hands out scripted responses to requests.get, in order; the last one repeats."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.pages = []

    def __call__(self, url, params, headers):
        self.pages.append(params["page"])
        if len(self.pages) > MAX_CALLS:
            print("FAIL: unbounded retries (%d requests)" % len(self.pages))
            os._exit(2)  # a bare `except:` in the code under test would swallow anything else
        step = self.steps[min(len(self.pages), len(self.steps)) - 1]
        if isinstance(step, BaseException):
            raise step
        return step


def query(get_api_query, script, **kwargs):
    """Returns (result or exception, sleeps)."""
    fake_requests.get = script
    sleeps = []
    try:
        result = get_api_query("token", "https://example.invalid/api", {"state": "opened"},
                               sleep=sleeps.append, **kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        result = exc
    return result, sleeps


def test_pagination(get_api_query):
    script = Script(Response(200, [1, 2], next_page="2"), Response(200, [3], next_page=""))
    result, sleeps = query(get_api_query, script)
    check(result == [1, 2, 3], "pages not concatenated: %r" % (result,))
    check(script.pages == ["1", "2"], "unexpected pages requested: %r" % script.pages)
    check(not sleeps, "slept without a failure")


def test_transient_failures_are_retried(get_api_query):
    script = Script(Response(503), Response(429, retry_after=7), fake_requests.exceptions.RequestException("reset"),
                    Response(200, [1]))
    result, sleeps = query(get_api_query, script)
    check(result == [1], "a transient failure was not retried: %r" % (result,))
    check(len(sleeps) == 3 and 7 in sleeps, "no backoff, or Retry-After ignored: %r" % sleeps)


def test_retries_are_bounded(get_api_query):
    script = Script(Response(500))
    result, sleeps = query(get_api_query, script, max_attempts=3)
    check(isinstance(result, gitlab_api_mrs.GitLabApiError), "no error after persistent 500: %r" % (result,))
    check(len(script.pages) == 3, "%d attempts, expected 3" % len(script.pages))
    check(len(sleeps) == 2, "slept %d times for 3 attempts" % len(sleeps))
    check("500" in str(result), "the error does not name the status: %s" % result)


def test_permanent_errors_are_not_retried(get_api_query):
    script = Script(Response(404))
    result, _ = query(get_api_query, script)
    check(isinstance(result, gitlab_api_mrs.GitLabApiError), "no error after 404: %r" % (result,))
    check(len(script.pages) == 1, "a 404 was retried %d times" % len(script.pages))


def test_later_page_failure_is_not_a_partial_result(get_api_query):
    script = Script(Response(200, [1], next_page="2"), Response(404))
    result, _ = query(get_api_query, script)
    check(isinstance(result, gitlab_api_mrs.GitLabApiError),
          "a failed second page returned a partial list: %r" % (result,))


def test_interrupt_is_not_swallowed(get_api_query):
    script = Script(KeyboardInterrupt())
    fake_requests.get = script
    try:
        get_api_query("token", "https://example.invalid/api", sleep=lambda _: None)
    except KeyboardInterrupt:
        return
    except Exception as exc:  # pylint: disable=broad-except
        check(False, "KeyboardInterrupt became %r" % (exc,))
        return
    check(False, "KeyboardInterrupt was swallowed")


def test_main_exits_nonzero(module):
    fake_requests.get = Script(Response(404))
    saved = sys.argv
    sys.argv = ["x", "--gitlab_private_token", "token"]
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            module.main(sys.argv)
        status = 0
    except SystemExit as exit_status:
        status = exit_status.code
    finally:
        sys.argv = saved
    check(status not in (0, None), "%s.main reported success after a failed query" % module.__name__)
    check("404" in str(status) + err.getvalue(), "no message naming the failure from %s" % module.__name__)
    check("token" not in str(status) + err.getvalue(), "the token leaked into the message")


def main():
    for module in (gitlab_api_mrs, gitlab_api_issues):
        get_api_query = module._get_api_query  # pylint: disable=protected-access
        for case in (test_pagination, test_transient_failures_are_retried, test_retries_are_bounded,
                     test_permanent_errors_are_not_retried,
                     test_later_page_failure_is_not_a_partial_result, test_interrupt_is_not_swallowed):
            case(get_api_query)
        test_main_exits_nonzero(module)
    if FAILURES:
        print("\n%d check(s) failed" % len(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
