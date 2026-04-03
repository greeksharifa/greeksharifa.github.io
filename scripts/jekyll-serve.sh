#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/jekyll-env.sh"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/jekyll-serve.sh [jekyll serve args...]

Runs:
  bundle exec jekyll serve --host 127.0.0.1 --port 4001

Environment overrides:
  BUNDLER_USER_BIN         Optional bin path prepended before execution
  BUNDLE_PATH              Defaults to ~/.bundle-jekyll-gsh
  JEKYLL_BUNDLER_VERSION   Optional exact Bundler version override
  JEKYLL_HOST              Defaults to 127.0.0.1
  JEKYLL_PORT              Defaults to 4001
  JEKYLL_BIN               Optional direct command override for testing

Examples:
  scripts/jekyll-serve.sh
  JEKYLL_PORT=4002 scripts/jekyll-serve.sh --livereload
EOF
  exit 0
fi

cd "${REPO_ROOT}"
jekyll_exec serve --host "${JEKYLL_HOST}" --port "${JEKYLL_PORT}" "$@"
