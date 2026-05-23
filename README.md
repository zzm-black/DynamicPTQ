# DynamicPTQ

DynamicPTQ is a unified PTQ project for running DEEP mixed-precision quantization experiments
across three backends with a single launcher and aligned runtime options.

- `FlatQuant`
- `SpinQuant`
- `QuaRot`

## Repository Layout

- `scripts/DynamicPTQ.sh`: backend dispatcher
- `scripts/run_flatquant.sh`: FlatQuant runner
- `scripts/run_spinquant.sh`: SpinQuant runner
- `scripts/run_quarot.sh`: QuaRot runner
- `tools/ptq_toolkit/`: unified FlatQuant PTQ runner
- `FlatQuant/`: FlatQuant backend code
- `SpinQuant/`: SpinQuant backend code
- `QuaRot/`: QuaRot backend code

## Project Goal

- Run PTQ with `METHOD=rtn|gptq` on all supported backends
- Configure DEEP layers manually with explicit 0-based layer indices
- Keep calibration and evaluation setup consistent across experiments

## Quick Start

```bash
cd DynamicPTQ
```

### Create and activate the `DynamicPTQ` environment

```bash
conda create -n DynamicPTQ python=3.10 -y
conda activate DynamicPTQ
pip install -r requirements.txt
```

### Optional for gated models

```bash
export HF_TOKEN="your_hf_token"
```

### Run with defaults

```bash
bash scripts/DynamicPTQ.sh
```

### Direct backend scripts (recommended for clarity)

```bash
bash scripts/run_flatquant.sh
bash scripts/run_spinquant.sh
bash scripts/run_quarot.sh
```

## Repro Commands

### Backend-only runs (DEEP disabled)

FlatQuant + RTN:

```bash
BACKEND=flatquant METHOD=rtn \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
bash scripts/run_flatquant.sh
```

FlatQuant + GPTQ:

```bash
BACKEND=flatquant METHOD=gptq \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
bash scripts/run_flatquant.sh
```

SpinQuant + RTN:

```bash
BACKEND=spinquant METHOD=rtn \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
OPTIMIZED_ROTATION_PATH=/path/to/R.bin \
bash scripts/run_spinquant.sh
```

SpinQuant + GPTQ:

```bash
BACKEND=spinquant METHOD=gptq \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
OPTIMIZED_ROTATION_PATH=/path/to/R.bin \
bash scripts/run_spinquant.sh
```

QuaRot + RTN:

```bash
BACKEND=quarot METHOD=rtn \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
bash scripts/run_quarot.sh
```

QuaRot + GPTQ:

```bash
BACKEND=quarot METHOD=gptq \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
bash scripts/run_quarot.sh
```

### DynamicPTQ runs with DEEP enabled (`METHOD=rtn|gptq`)

Use the unified launcher to enable DEEP. Both RTN and GPTQ support DEEP when
`ENABLE_DEEP=true`.

```bash
# Example: RTN + DEEP
BACKEND=flatquant METHOD=rtn \
ENABLE_DEEP=true DEEP_LAYERS=0,1,2,31 DEEP_ACT_BITS=8 \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
bash scripts/DynamicPTQ.sh
```

```bash
# Example: GPTQ + DEEP
BACKEND=flatquant METHOD=gptq \
ENABLE_DEEP=true DEEP_LAYERS=0,1,2,31 DEEP_ACT_BITS=8 \
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
bash scripts/DynamicPTQ.sh
```

## Calibration And Evaluation Data Guide

### Calibration data controls PTQ fitting quality

- `CALI_DATASET`: calibration dataset name, default `wikitext2`
- `NSAMPLES`: number of calibration samples, default `128`
- `CALI_BSZ`: calibration batch size (FlatQuant path), default `4`

### How to download calibration and evaluation datasets

Prepare datasets under `FlatQuant/datasets` (or your own mirrored directory):

- Calibration / PPL evaluation:
  - WikiText2: download from [wikitext](https://huggingface.co/datasets/wikitext), place under `FlatQuant/datasets/wikitext`.
  - C4: download from [allenai/c4](https://huggingface.co/datasets/allenai/c4), place under `FlatQuant/datasets/allenai/c4`.

- Commonsense QA evaluation (`lm-eval` tasks):
  - ARC-E / ARC-C: [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) -> `FlatQuant/datasets/ai2_arc`
  - HellaSwag: [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) -> `FlatQuant/datasets/hellaswag`
  - LAMBADA: [EleutherAI/lambada_openai](https://huggingface.co/datasets/EleutherAI/lambada_openai) -> `FlatQuant/datasets/lambada_openai`
  - PIQA: [ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa) -> `FlatQuant/datasets/piqa`
  - WinoGrande: [winogrande](https://huggingface.co/datasets/winogrande) -> `FlatQuant/datasets/winogrande`

### Perplexity evaluation data

- `EVAL_DATASETS`: space-separated list, default `wikitext2 c4`

### Task evaluation data (`lm-eval`)

- `RUN_LM_EVAL=true|false`
- `LM_EVAL_TASKS`: space-separated task list
- `LM_EVAL_BATCH_SIZE`: batch size for task evaluation

### Recommended baseline setup

```bash
CALI_DATASET=wikitext2 \
NSAMPLES=128 \
EVAL_DATASETS="wikitext2 c4" \
RUN_LM_EVAL=true \
LM_EVAL_TASKS="piqa hellaswag arc_easy arc_challenge winogrande lambada_openai" \
bash scripts/DynamicPTQ.sh
```

## Backbone-Specific Notes

The following practical settings are summarized from the upstream backend docs and
from the current integrated runtime behavior in this repository.

### FlatQuant

- Python environment should use `python=3.10`.
- Recommended calibration defaults: `NSAMPLES=128`, `CALI_BSZ=4`, `EPOCHS=15`.
- Typical low-bit setup is `W_BITS=4 A_BITS=4 K_BITS=4 V_BITS=4`.
- For gated models (for example Llama family), export `HF_TOKEN` before running.

### SpinQuant

- `OPTIMIZED_ROTATION_PATH` is mandatory and must point to an existing rotation file.
- For large models (for example 70B), set multiple visible GPUs and keep
  `DISTRIBUTE_MODEL=true`.

### QuaRot

- Native fake-quant path assumes rotation is enabled (`--rotate` in launcher path).
- Upstream fake-quant README emphasizes LLaMA-family support first; test on LLaMA
  before expanding to other families.

### Model-family hints

- Llama-2 / Llama-3 / Llama-3.1 / Qwen2.5 are the most verified paths in the
  upstream FlatQuant docs and scripts.
- For 70B-class checkpoints, prefer multi-GPU execution and keep
  `DISTRIBUTE_MODEL=true`.
- Start from `DEEP_LAYERS=0,1,2,<last_layer_id>` and tune only after a stable
  baseline run.

## Output Layout

### Run output directory

- `${OUTPUT_DIR}/${EXP_NAME}/`

### Default base output dir

- `OUTPUT_DIR=${WORKSPACE_ROOT}/outputs/deep_only`

### Example

```bash
OUTPUT_DIR=./outputs \
EXP_NAME=flatquant_gptq_llama3_8b \
bash scripts/DynamicPTQ.sh
```

Files generated for FlatQuant unified runs are placed directly under that run
directory (for example `flat_parameters.pth` and log files), without additional
model/bit-width subfolders.

## Main Configuration

- `BACKEND`: `flatquant|spinquant|quarot`
- `METHOD`: `rtn|gptq`
- `MODEL_PATH`: model id or local model directory
- `ENABLE_DEEP`: `true|false`
- `DEEP_LAYERS`: comma-separated 0-based layer ids (for example: `0,1,2,31`)
- `DEEP_ACT_BITS`: activation bits for DEEP-selected layers
- `W_BITS`, `A_BITS`, `K_BITS`, `V_BITS`: PTQ precision setup
- `CALI_DATASET`, `NSAMPLES`, `CALI_BSZ`: calibration setup
- `EVAL_DATASETS`: PPL evaluation datasets
- `RUN_LM_EVAL`: `true|false`
- `LM_EVAL_TASKS`: space-separated lm-eval tasks

## Notes

- The launcher auto-detects `WORKSPACE_ROOT` from script location.
- For SpinQuant backend, `OPTIMIZED_ROTATION_PATH` is required.
- No hardcoded personal paths or tokens are required.

## Acknowledgements

This project builds on and integrates ideas and implementations from:

- `FlatQuant`
- `SpinQuant`
- `QuaRot`

We thank the original authors and maintainers of these projects for their
valuable open-source contributions.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See LICENSE for details.
