#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-DynmicPTQ}"
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
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/outputs/deep_only}"
CALI_DATASET="${CALI_DATASET:-wikitext2}"
EVAL_DATASETS="${EVAL_DATASETS:-wikitext2 c4}"
RUN_LM_EVAL="${RUN_LM_EVAL:-true}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-piqa hellaswag arc_easy arc_challenge winogrande lambada_openai}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-16}"
W_BITS="${W_BITS:-4}"
A_BITS="${A_BITS:-4}"
K_BITS="${K_BITS:-4}"
V_BITS="${V_BITS:-4}"
NSAMPLES="${NSAMPLES:-128}"
CALI_BSZ="${CALI_BSZ:-4}"
EPOCHS="${EPOCHS:-15}"
SEED="${SEED:-0}"
ENABLE_DEEP="${ENABLE_DEEP:-true}"
DEEP_ACT_BITS="${DEEP_ACT_BITS:-8}"
DEEP_LAYERS="${DEEP_LAYERS:-}"

if [[ "${ENABLE_DEEP}" == "true" && -z "${DEEP_LAYERS}" ]]; then
  echo "[ERROR] DEEP_LAYERS must be set when ENABLE_DEEP=true. Example: 0,1,2,31" >&2
  exit 1
fi

read -r -a EVAL_DATASET_ARRAY <<< "${EVAL_DATASETS}"
read -r -a LM_EVAL_TASK_ARRAY <<< "${LM_EVAL_TASKS}"

HF_TOKEN_FLAGS=()
if [[ -n "${HF_TOKEN}" ]]; then
  HF_TOKEN_FLAGS=(--hf_token "${HF_TOKEN}")
fi

DEEP_FLAGS=(--no-enable_deep)
if [[ "${ENABLE_DEEP}" == "true" ]]; then
  DEEP_FLAGS=(--enable_deep --deep_act_bits "${DEEP_ACT_BITS}" --deep_layers "${DEEP_LAYERS}")
fi

LM_EVAL_FLAGS=()
if [[ "${RUN_LM_EVAL}" == "true" ]]; then
  LM_EVAL_FLAGS=(--lm_eval --lm_eval_tasks "${LM_EVAL_TASK_ARRAY[@]}" --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}")
fi

METHOD_FLAGS=(--method rtn)
if [[ "${METHOD}" == "gptq" ]]; then
  GPTQ_MSE="${GPTQ_MSE:-false}"
  ACT_ORDER="${ACT_ORDER:-false}"
  PERCDAMP="${PERCDAMP:-0.01}"
  METHOD_FLAGS=(--method gptq --percdamp "${PERCDAMP}")
  [[ "${GPTQ_MSE}" == "true" ]] && METHOD_FLAGS+=(--gptq_mse)
  [[ "${ACT_ORDER}" == "true" ]] && METHOD_FLAGS+=(--act_order)
fi

MODEL_TAG="${MODEL_PATH//\//_}"
EXP_NAME="${EXP_NAME:-flatquant_${METHOD}_w${W_BITS}a${A_BITS}k${K_BITS}v${V_BITS}_${MODEL_TAG}}"
RUN_OUTPUT_DIR="${OUTPUT_DIR}/${EXP_NAME}"

echo "[INFO] backend=flatquant method=${METHOD}"
echo "[INFO] model=${MODEL_PATH}"
echo "[INFO] output_dir=${RUN_OUTPUT_DIR}"

cd "${WORKSPACE_ROOT}/FlatQuant"
PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}" python -m tools.ptq_toolkit.run_unified_ptq \
  --workspace_root "${WORKSPACE_ROOT}" \
  --backend flatquant \
  --model "${MODEL_PATH}" \
  --cali_dataset "${CALI_DATASET}" \
  --eval_datasets "${EVAL_DATASET_ARRAY[@]}" \
  --w_bits "${W_BITS}" --a_bits "${A_BITS}" --k_bits "${K_BITS}" --v_bits "${V_BITS}" \
  --nsamples "${NSAMPLES}" --cali_bsz "${CALI_BSZ}" --epochs "${EPOCHS}" \
  --output_dir "${RUN_OUTPUT_DIR}" \
  --exp_name "${EXP_NAME}" \
  --seed "${SEED}" \
  "${HF_TOKEN_FLAGS[@]}" \
  "${METHOD_FLAGS[@]}" \
  "${LM_EVAL_FLAGS[@]}" \
  "${DEEP_FLAGS[@]}"
