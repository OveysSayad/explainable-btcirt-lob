#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/scripts/run_pipeline.sh" configs/study_c_strict_horizons.yaml
