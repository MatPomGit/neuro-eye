#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="eye-tracking-research"

mkdir -p "$ROOT_DIR/data/sessions" "$ROOT_DIR/data/calibrations"

exec conda run --no-capture-output -n "$ENV_NAME" python "$ROOT_DIR/main.py" "$@"
