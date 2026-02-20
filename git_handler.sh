#!/bin/bash
set -euo pipefail

branch="${1:?Usage: $0 <branch>}"

# refuse to run with uncommitted work
git diff --quiet && git diff --cached --quiet || {
  echo "Working tree not clean. Commit/stash first." >&2
  exit 1
}

git fetch origin

git switch main
git pull --ff-only origin main   # guarantees local main matches origin/main

git switch "$branch"
git rebase main

git switch main
git merge --ff-only "$branch"
git push origin main
