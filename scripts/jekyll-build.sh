#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/jekyll-env.sh"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/jekyll-build.sh [jekyll build args...]

Runs:
  bundle exec jekyll build

Environment overrides:
  BUNDLER_USER_BIN         Optional bin path prepended before execution
  BUNDLE_PATH              Defaults to ~/.bundle-jekyll-gsh
  JEKYLL_BUNDLER_VERSION   Optional exact Bundler version override
  JEKYLL_BIN               Optional direct command override for testing
EOF
  exit 0
fi

cd "${REPO_ROOT}"
jekyll_exec build "$@"
