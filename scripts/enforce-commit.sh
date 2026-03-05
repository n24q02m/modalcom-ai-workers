#!/usr/bin/env bash
MSG=$(head -1 "$1")
if [[ "$MSG" =~ ^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?(!)?\:\ .+ ]]; then
  exit 0
fi
echo "ERROR: Commit message must follow Conventional Commits format."
echo "  <type>[optional scope][!]: <description>"
echo "  Allowed types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert"
echo "Got: $MSG"
exit 1
