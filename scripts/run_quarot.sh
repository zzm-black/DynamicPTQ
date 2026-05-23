#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-DynamicPTQ}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda is not available. Set up conda first." >&2
  exit 1
fi
conda activate "${CONDA_ENV_NAME}"

METHOD="${METHOD:-rtn}"
MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B}"
HF_TOKEN="${HF_TOKEN:-}"
CALI_DATASET="${CALI_DATASET:-wikitext2}"
EVAL_DATASETS="${EVAL_DATASETS:-wikitext2 c4}"
RUN_LM_EVAL="${RUN_LM_EVAL:-true}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-piqa arc_easy arc_challenge winogrande lambada_openai}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-16}"
W_BITS="${W_BITS:-4}"
A_BITS="${A_BITS:-4}"
K_BITS="${K_BITS:-4}"
V_BITS="${V_BITS:-4}"
NSAMPLES="${NSAMPLES:-128}"
SEED="${SEED:-0}"
ENABLE_DEEP="${ENABLE_DEEP:-true}"
DEEP_ACT_BITS="${DEEP_ACT_BITS:-8}"
DEEP_LAYERS="${DEEP_LAYERS:-}"
DISTRIBUTE_MODEL="${DISTRIBUTE_MODEL:-false}"
MIN_FREE_MEM_RATIO="${MIN_FREE_MEM_RATIO:-0.5}"
export MIN_FREE_MEM_RATIO

if [[ "${ENABLE_DEEP}" == "true" && -z "${DEEP_LAYERS}" ]]; then
  echo "[ERROR] DEEP_LAYERS must be set when ENABLE_DEEP=true. Example: 0,1,2,31" >&2
  exit 1
fi

check_gpu_memory_headroom() {
  python - <<'PY'
import os, subprocess, sys
required_ratio = float(os.environ["MIN_FREE_MEM_RATIO"])
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
target = [idx.strip() for idx in visible.split(",") if idx.strip()] if visible else ["0"]
records = {}
for line in subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
    text=True,
).splitlines():
    idx, free_mb, total_mb = [x.strip() for x in line.split(",")]
    records[idx] = (int(free_mb), int(total_mb))
for idx in target:
    free_mb, total_mb = records.get(idx, (0, 1))
    if free_mb / max(total_mb, 1) < required_ratio:
        print(f"[ERROR] GPU {idx} free ratio below threshold.", file=sys.stderr)
        sys.exit(1)
print("[INFO] GPU memory headroom check passed.")
PY
}

check_gpu_memory_headroom

HF_TOKEN_FLAGS=()
[[ -n "${HF_TOKEN}" ]] && HF_TOKEN_FLAGS=(--hf_token "${HF_TOKEN}")

read -r -a EVAL_DATASET_ARRAY <<< "${EVAL_DATASETS}"
read -r -a LM_EVAL_TASK_ARRAY <<< "${LM_EVAL_TASKS}"

COMMON_FLAGS=(
  --model "${MODEL_PATH}"
  --cal_dataset "${CALI_DATASET}"
  --eval_datasets "${EVAL_DATASET_ARRAY[@]}"
  --rotate
  --w_bits "${W_BITS}" --a_bits "${A_BITS}" --k_bits "${K_BITS}" --v_bits "${V_BITS}"
  --w_clip --a_asym --k_asym --v_asym
  --k_groupsize 128 --v_groupsize 128
  --nsamples "${NSAMPLES}"
  --seed "${SEED}"
  "${HF_TOKEN_FLAGS[@]}"
)

if [[ "${METHOD}" == "rtn" ]]; then
  COMMON_FLAGS+=(--w_rtn)
fi
if [[ "${ENABLE_DEEP}" == "true" ]]; then
  COMMON_FLAGS+=(--enable_deep --deep_act_bits "${DEEP_ACT_BITS}" --deep_layers "${DEEP_LAYERS}")
else
  COMMON_FLAGS+=(--no-enable_deep)
fi
if [[ "${RUN_LM_EVAL}" == "true" ]]; then
  COMMON_FLAGS+=(--lm_eval --tasks "${LM_EVAL_TASK_ARRAY[@]}" --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}")
fi
if [[ "${DISTRIBUTE_MODEL}" == "true" ]]; then
  COMMON_FLAGS+=(--distribute)
fi

echo "[INFO] backend=quarot method=${METHOD}"
echo "[INFO] model=${MODEL_PATH}"

cd "${WORKSPACE_ROOT}/QuaRot/fake_quant"
PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}" python main.py "${COMMON_FLAGS[@]}"
