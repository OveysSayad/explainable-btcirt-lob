#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
mkdir -p .libs
export DYLD_LIBRARY_PATH="$ROOT/.libs:/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib:${DYLD_LIBRARY_PATH:-}"
exec python -m src.pipeline --config "${1:-configs/project_config.yaml}"
