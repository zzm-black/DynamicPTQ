#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
DynamicPTQ launcher dispatcher

Usage:
  bash scripts/DynamicPTQ.sh [--backend flatquant|spinquant|quarot] [extra args]

This script only dispatches to backend-specific scripts:
  - scripts/run_flatquant.sh
  - scripts/run_spinquant.sh
  - scripts/run_quarot.sh

Examples:
  BACKEND=flatquant METHOD=gptq DEEP_LAYERS=0,1,2,31 MODEL_PATH=meta-llama/Meta-Llama-3-8B \
    bash scripts/DynamicPTQ.sh

  BACKEND=spinquant METHOD=rtn MODEL_PATH=meta-llama/Meta-Llama-3-8B OPTIMIZED_ROTATION_PATH=/path/to/R.bin \
    bash scripts/DynamicPTQ.sh
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND="${BACKEND:-flatquant}"
if [[ "${1:-}" == "--backend" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "[ERROR] --backend requires a value." >&2
    exit 1
  fi
  BACKEND="${2}"
  shift 2
fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

case "${BACKEND}" in
  flatquant|spinquant|quarot) ;;
  *)
    echo "[ERROR] Unsupported BACKEND=${BACKEND}. Use flatquant|spinquant|quarot." >&2
    exit 1
    ;;
esac

exec "${SCRIPT_DIR}/run_${BACKEND}.sh" "$@"
