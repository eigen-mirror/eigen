#!/usr/bin/env bash
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# Diff the compiler warnings in a merge request pipeline's build logs against a
# pipeline on the target branch.  No job fails on a warning, so this is how a
# new one is found; see "Compiler Warnings" in .agents/ci.md.
#
#   ci_warning_diff.sh <mr-iid>
#   ci_warning_diff.sh <base-pipeline-id> <mr-pipeline-id>
#
# With a merge request, the head pipeline is compared with a scheduled or web
# pipeline at the merge request's base commit.  Those pipelines run every build
# job, while a default-branch push runs only a few.  If none ran at the base
# commit, the newest scheduled pipeline on the target branch is used and named:
# it can be days older, and the changes merged in between then show up in the
# diff.  Lines marked ">" are warnings only the merge request pipeline has, and
# the exit status is 1 when the two pipelines differ.
#
# Needs python3 and an authenticated glab; GITLAB_TOKEN works where
# `glab auth login` has not been run.  EIGEN_CI_PROJECT selects another project
# (URL-encoded path or numeric id; default libeigen%2Feigen).

set -euo pipefail

project=${EIGEN_CI_PROJECT:-libeigen%2Feigen}

api() { glab api "projects/$project/$1"; }

# field <python expression over d>: evaluate it on the JSON document on stdin.
field() { python3 -c 'import json, sys; d = json.load(sys.stdin); print('"$1"')'; }

# warnings <pipeline-id>: one line per build job and warning site.  The
# normalization strips what differs between two runs of the same warning: the
# :affected suffix of a merge request's build jobs, the runners' per-line
# timestamps, line numbers in the file:12:3: and the MSVC/nvcc file(12): forms,
# and the checkout prefix, which on Windows is a backslash path with a per-run
# runner directory.  Without it, every warning reads as new.
warnings() {
  api "pipelines/$1/jobs?per_page=100" --paginate | python3 -c '
import json, sys
text, decoder, at = sys.stdin.read(), json.JSONDecoder(), 0
while at < len(text):
    if text[at].isspace():
        at += 1
        continue
    page, at = decoder.raw_decode(text, at)
    for job in page:
        if job["stage"] == "build" and job["status"] in ("success", "failed"):
            print(job["id"], job["name"])' |
    while read -r id name; do
      name=${name%:affected}
      api "jobs/$id/trace" |
        sed -E 's/\x1b\[[0-9;]*[mK]//g; s/\r//; s/^[0-9-]+T[0-9:.]+Z [0-9A-Z]+ //' |
        { grep -E ': warning[: #]|warning C[0-9]{4}' || true; } |
        sed -E "s|\\\\|/|g; s|^.*/eigen/||; s|:[0-9]+:[0-9]+:|:|; s|\([0-9]+\)||; s|^|$name: |"
    done | sort -u
}

case $# in
  1)
    mr=$(api "merge_requests/$1")
    head=$(field '(d.get("head_pipeline") or {}).get("id") or ""' <<<"$mr")
    base_sha=$(field 'd["diff_refs"]["base_sha"]' <<<"$mr")
    target=$(field 'd["target_branch"]' <<<"$mr")
    [ -n "$head" ] || { echo "merge request !$1 has no pipeline" >&2; exit 2; }
    base=$(api "pipelines?sha=$base_sha&per_page=100" |
      field '*[p["id"] for p in d if p["source"] in ("schedule", "web")][:1]')
    if [ -z "$base" ]; then
      base=$(api "pipelines?ref=$target&source=schedule&per_page=1" | field '*[p["id"] for p in d][:1]')
      [ -n "$base" ] || { echo "no scheduled pipeline on $target to compare with" >&2; exit 2; }
      echo "No scheduled or web pipeline ran at base commit $base_sha; comparing with the newest scheduled" \
           "pipeline on $target, so changes merged since then also appear." >&2
    fi
    ;;
  2)
    base=$1 head=$2
    ;;
  *)
    sed -n '5,22s/^# \{0,1\}//p' "$0" >&2
    exit 2
    ;;
esac

echo "base pipeline $base, merge request pipeline $head" >&2
diff <(warnings "$base") <(warnings "$head") && echo "no difference in build warnings" >&2
