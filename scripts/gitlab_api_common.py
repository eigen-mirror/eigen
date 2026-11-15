# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Paginated GET shared by the gitlab_api_*.py helpers."""

import time

import requests

# Statuses worth retrying: rate limiting and transient server errors.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class GitLabApiError(RuntimeError):
    """A page could not be fetched; the result would be incomplete."""


def get_api_query(
    gitlab_private_token: str,
    url: str,
    params: dict[str, str] | None = None,
    max_attempts: int = 5,
    sleep=time.sleep,
):
    """Return the concatenated JSON lists of every page of a GitLab API GET.

    Retries a page up to `max_attempts` times in all when the request raises or
    returns a retryable status, waiting Retry-After seconds (default 2**attempt).
    Raises GitLabApiError rather than returning a partial list.
    """
    params = dict(params or {})
    params["per_page"] = "100"
    headers = {"PRIVATE-TOKEN": gitlab_private_token}
    out = []
    next_page = "1"
    while next_page:
        params["page"] = next_page
        for attempt in range(1, max_attempts + 1):
            delay = 2**attempt
            try:
                resp = requests.get(url=url, params=params, headers=headers)
            except requests.exceptions.RequestException as exc:
                failure = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    break
                failure = f"HTTP {resp.status_code}"
                if resp.status_code not in RETRYABLE_STATUSES:
                    raise GitLabApiError(f"GET {url} page {next_page}: {failure}")
                retry_after = resp.headers.get("retry-after", "")
                if retry_after.isdigit():
                    delay = int(retry_after)
            if attempt == max_attempts:
                raise GitLabApiError(
                    f"GET {url} page {next_page}: {failure} after {max_attempts} attempts"
                )
            sleep(delay)
        out.extend(resp.json())
        next_page = resp.headers.get("x-next-page", "")
    return out
