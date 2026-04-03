#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_BUNDLE_PATH="${HOME}/.bundle-jekyll-gsh"
DEFAULT_JEKYLL_HOST="127.0.0.1"
DEFAULT_JEKYLL_PORT="4001"

if [[ -n "${BUNDLER_USER_BIN:-}" ]]; then
  export PATH="${BUNDLER_USER_BIN}:${PATH}"
fi
export BUNDLE_PATH="${BUNDLE_PATH:-${DEFAULT_BUNDLE_PATH}}"
export JEKYLL_HOST="${JEKYLL_HOST:-${DEFAULT_JEKYLL_HOST}}"
export JEKYLL_PORT="${JEKYLL_PORT:-${DEFAULT_JEKYLL_PORT}}"

jekyll_exec() {
  if [[ -n "${JEKYLL_BIN:-}" ]]; then
    "${JEKYLL_BIN}" "$@"
    return
  fi

  if [[ -n "${JEKYLL_BUNDLER_VERSION:-}" ]]; then
    bundle "_${JEKYLL_BUNDLER_VERSION}_" exec jekyll "$@"
    return
  fi

  bundle exec jekyll "$@"
}
