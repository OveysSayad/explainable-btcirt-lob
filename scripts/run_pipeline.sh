#!/usr/bin/env bash
# Run the full research pipeline with macOS OpenMP resolution for XGBoost.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

mkdir -p .libs
if [[ ! -e .libs/libomp.dylib ]]; then
  if [[ -f /opt/homebrew/opt/libomp/lib/libomp.dylib ]]; then
    ln -sf /opt/homebrew/opt/libomp/lib/libomp.dylib .libs/libomp.dylib
  elif [[ -f /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libomp.dylib ]]; then
    ln -sf /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libomp.dylib .libs/libomp.dylib
  else
    echo "WARNING: libomp.dylib not found. On macOS install with: brew install libomp" >&2
  fi
fi

export DYLD_LIBRARY_PATH="$ROOT/.libs:/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib:${DYLD_LIBRARY_PATH:-}"
exec python -m src.pipeline --config "${1:-configs/project_config.yaml}"
