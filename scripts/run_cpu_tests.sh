#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd)"

cd "${PROJECT_ROOT}"
python -m pytest -q
ruff check src tests ./*.py
