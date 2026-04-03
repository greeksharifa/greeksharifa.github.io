# Repository Guidelines

## Project Structure & Module Organization
This repository is a Jekyll-based research blog. Site configuration lives in `_config.yml` and `Gemfile`. Post content is stored in `_posts/` using `YYYY-MM-DD-title.md` filenames. Shared templates and partials are in `_layouts/` and `_includes/`. Styles are organized under `_scss/`, with component styles in `_scss/component/` and base styles in `_scss/base/`. Static assets live in `public/`. Test coverage for repo-specific regressions is in `tests/`.

## Build, Test, and Development Commands
Use the repository scripts instead of raw Bundler commands so Ruby and Bundler paths stay consistent.

- `scripts/jekyll-build.sh`: builds the site into `_site/`.
- `scripts/jekyll-serve.sh`: starts the local server at `http://127.0.0.1:4001/`.
- `JEKYLL_PORT=4002 scripts/jekyll-serve.sh --livereload`: runs the dev server on a custom port.
- `python3 -m unittest tests.test_jekyll_scripts`: verifies the Jekyll helper scripts.
- `python3 -m unittest`: runs the full Python regression suite in `tests/`.

## Coding Style & Naming Conventions
Preserve existing Jekyll and theme conventions. Keep Markdown front matter and Liquid usage aligned with nearby posts and templates. Use descriptive, date-prefixed post filenames such as `_posts/2025-01-17-AdaptLLM.md`. Match indentation to the file you edit: 4 spaces in Python tests, compact indentation in HTML, Markdown, and SCSS. No formatter or linter is configured here, so keep diffs localized and readable.

## Testing Guidelines
Tests use Python `unittest`. Add or update a focused regression test whenever changing reusable templates, layout markup, or helper scripts. Name tests as `tests/test_<area>.py`, and keep assertions tied to user-visible behavior or required script output. Before opening a PR, run the narrow test you changed and then `python3 -m unittest`.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects such as `Refine about page typography` or `Refresh homepage and sidebar links`. Keep commit messages specific to one change. PRs should include a brief summary, affected paths, verification commands run, and screenshots for any visual or layout change. Link the related issue when one exists.

## Configuration Notes
Prefer the checked-in helper scripts over manual environment setup. If local Jekyll output changes, verify both `scripts/jekyll-build.sh` and the rendered page before merging.
