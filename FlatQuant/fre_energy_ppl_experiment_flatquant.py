import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.fftpack import dct, idct


THIS_DIR = Path(__file__).resolve().parent
SMOOTHQUANT_ROOT = THIS_DIR.parent
EMNLP_ROOT = SMOOTHQUANT_ROOT.parent
SMOOTHQUANT_EXAMPLES_DIR = SMOOTHQUANT_ROOT / "smoothquant" / "examples"
for candidate in (THIS_DIR, SMOOTHQUANT_ROOT, EMNLP_ROOT, SMOOTHQUANT_EXAMPLES_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from wide_experiment import (
    EvalConfig,
    WikiTextPplEvaluator,
    collect_act_scales_wikitext2,
    load_model,
    smooth_lm,
)


def quantize_weight_per_channel_absmax(weight: torch.Tensor, n_bits: int) -> torch.Tensor:
    if n_bits >= 16:
        return weight
    qmax = (1 << (n_bits - 1)) - 1
    scales = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / qmax
    return torch.round(weight / scales) * scales


def quantize_activation_per_token_absmax(activation: torch.Tensor, n_bits: int) -> torch.Tensor:
    if n_bits >= 16:
        return activation
    qmax = (1 << (n_bits - 1)) - 1
    x_fp32 = activation.float()
    scales = x_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7) / qmax
    qx = torch.round(x_fp32 / scales) * scales
    return qx.to(activation.dtype)


def dct2d(matrix: np.ndarray) -> np.ndarray:
    return dct(dct(matrix.T, norm="ortho").T, norm="ortho")


def idct2d(matrix: np.ndarray) -> np.ndarray:
    return idct(idct(matrix.T, norm="ortho").T, norm="ortho")


class FreqHybridLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_low: torch.Tensor,
        weight_residual_q: torch.Tensor,
        bias: Optional[torch.Tensor],
        act_bits: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_bits = act_bits
        self.register_buffer("weight_low", weight_low.detach())
        self.register_buffer("weight_residual_q", weight_residual_q.detach())
        self.register_buffer("bias", bias.detach() if bias is not None else None)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_input = x.to(self.weight_low.dtype)
        low_bias = self.bias.to(self.weight_low.dtype) if self.bias is not None else None
        low_out = torch.functional.F.linear(low_input, self.weight_low, low_bias)

        qx = quantize_activation_per_token_absmax(x, self.act_bits).to(self.weight_residual_q.dtype)
        residual_out = torch.functional.F.linear(qx, self.weight_residual_q, None)
        return low_out.to(x.dtype) + residual_out.to(x.dtype)


def _get_parent_module(model: nn.Module, path: str) -> Tuple[nn.Module, str]:
    fields = path.split(".")
    parent = model
    for field in fields[:-1]:
        parent = getattr(parent, field)
    return parent, fields[-1]


def find_all_linear_paths(model: nn.Module, exclude_lm_head: bool) -> List[str]:
    paths: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if exclude_lm_head and name == "lm_head":
            continue
        paths.append(name)
    return paths


def parse_target_modules(raw_modules: str) -> List[str]:
    return [name.strip() for name in raw_modules.split(",") if name.strip()]


def find_target_linear_paths(
    model: nn.Module,
    exclude_lm_head: bool,
    target_module_names: Sequence[str],
) -> List[str]:
    target_set = set(target_module_names)
    paths: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if exclude_lm_head and name == "lm_head":
            continue
        if target_set:
            name_parts = name.split(".")
            if not any(part in target_set for part in name_parts):
                continue
        paths.append(name)
    return paths


def build_frequency_hybrid_weight(
    weight: torch.Tensor,
    weight_bits: int,
    ratio: float,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    weight_fp32 = weight.detach().float().cpu()
    h, w = weight_fp32.shape
    h_pad = (h // block_size) * block_size
    w_pad = (w // block_size) * block_size
    if h_pad == 0 or w_pad == 0:
        zero = torch.zeros_like(weight_fp32)
        return zero, quantize_weight_per_channel_absmax(weight_fp32, n_bits=weight_bits)

    block_area = block_size * block_size
    keep_count = min(max(int(round(block_area * ratio)), 1), block_area)
    k = int(np.ceil(np.sqrt(keep_count)))
    # Precompute the fixed top-left low-frequency mask once per matrix.
    low_mask_template = np.zeros((block_size, block_size), dtype=bool)
    selected = 0
    for row_idx in range(k):
        for col_idx in range(k):
            if selected >= keep_count:
                break
            low_mask_template[row_idx, col_idx] = True
            selected += 1
        if selected >= keep_count:
            break

    weight_np = weight_fp32.numpy()
    low_np = np.zeros_like(weight_np, dtype=np.float32)
    residual_q_np = np.zeros_like(weight_np, dtype=np.float32)
    qmax = (1 << (weight_bits - 1)) - 1

    for i in range(0, h_pad, block_size):
        for j in range(0, w_pad, block_size):
            block = weight_np[i : i + block_size, j : j + block_size]
            coeff = dct2d(block)

            low_coeff = np.zeros_like(coeff, dtype=np.float32)
            low_coeff[low_mask_template] = coeff[low_mask_template].astype(np.float32)

            residual = np.zeros_like(coeff, dtype=np.float32)
            residual[~low_mask_template] = coeff[~low_mask_template].astype(np.float32)
            max_abs = float(np.max(np.abs(residual)))
            scale = max_abs / float(qmax) if max_abs > 0 else 1.0
            quant_residual = np.round(residual / scale) * scale
            low_np[i : i + block_size, j : j + block_size] = idct2d(low_coeff).astype(np.float32)
            residual_q_np[i : i + block_size, j : j + block_size] = idct2d(quant_residual).astype(
                np.float32
            )

    weight_low = torch.from_numpy(low_np).to(weight_fp32.dtype)
    weight_residual_q = torch.from_numpy(residual_q_np).to(weight_fp32.dtype)

    if h_pad < h:
        tail = weight_fp32[h_pad:, :]
        weight_low[h_pad:, :] = 0.0
        weight_residual_q[h_pad:, :] = quantize_weight_per_channel_absmax(tail, n_bits=weight_bits)
    if w_pad < w:
        tail = weight_fp32[:, w_pad:]
        weight_low[:, w_pad:] = 0.0
        weight_residual_q[:, w_pad:] = quantize_weight_per_channel_absmax(tail, n_bits=weight_bits)
    return weight_low, weight_residual_q


@torch.no_grad()
def apply_frequency_hybrid_linears(
    model: nn.Module,
    target_paths: Sequence[str],
    act_bits: int,
    weight_bits: int,
    ratio: float,
    block_size: int,
    low_dtype: torch.dtype,
) -> None:
    for path in target_paths:
        parent, attr_name = _get_parent_module(model, path)
        linear = getattr(parent, attr_name)
        if not isinstance(linear, nn.Linear):
            continue

        weight = linear.weight.detach()
        bias = linear.bias.detach() if linear.bias is not None else None
        weight_low, weight_residual_q = build_frequency_hybrid_weight(
            weight=weight,
            weight_bits=weight_bits,
            ratio=ratio,
            block_size=block_size,
        )
        quant_linear = FreqHybridLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            weight_low=weight_low.to(weight.device, dtype=low_dtype),
            weight_residual_q=weight_residual_q.to(weight.device, dtype=low_dtype),
            bias=bias,
            act_bits=act_bits,
        ).to(weight.device)
        setattr(parent, attr_name, quant_linear)


def to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(x) for x in obj]
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("PPL test for frequency-only hybrid scheme")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--act_scales_path", type=str, default="")
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--output_dir", type=str, default="outputs/fre_energy_ppl")
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--act_bits", type=int, default=4)
    parser.add_argument("--hybrid_ratio", type=float, default=0.01)
    parser.add_argument("--freq_block_size", type=int, default=32)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj",
        help="Comma-separated target linear module names for WIDE insertion.",
    )
    parser.add_argument(
        "--low_precision",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Precision for kept low-frequency branch (the 1%% branch).",
    )
    parser.add_argument(
        "--include_lm_head",
        action="store_true",
        help="Include lm_head in linear replacement (default: excluded).",
    )
    parser.add_argument("--ppl_split", type=str, default="test")
    parser.add_argument("--ppl_seq_len", type=int, default=2048)
    parser.add_argument("--ppl_samples", type=int, default=None)

    parser.add_argument("--auto_generate_act_scales", action="store_true")
    parser.add_argument("--act_scales_split", type=str, default="train")
    parser.add_argument("--act_scales_samples", type=int, default=128)
    parser.add_argument("--act_scales_seq_len", type=int, default=512)
    parser.add_argument(
        "--flat_parameters_path",
        type=str,
        default="",
        help="Path to FlatQuant calibration file (flat_parameters.pth) or its parent directory.",
    )
    parser.add_argument("--flatquant_q_bits", type=int, default=4)
    parser.add_argument("--flatquant_k_bits", type=int, default=4)
    parser.add_argument("--flatquant_v_bits", type=int, default=4)
    parser.add_argument("--flatquant_lwc", dest="flatquant_lwc", action="store_true")
    parser.add_argument("--no_flatquant_lwc", dest="flatquant_lwc", action="store_false")
    parser.add_argument("--flatquant_lac", dest="flatquant_lac", action="store_true")
    parser.add_argument("--no_flatquant_lac", dest="flatquant_lac", action="store_false")
    parser.add_argument("--flatquant_add_diag", dest="flatquant_add_diag", action="store_true")
    parser.add_argument("--no_flatquant_add_diag", dest="flatquant_add_diag", action="store_false")
    parser.add_argument("--flatquant_direct_inv", action="store_true", default=False)
    parser.add_argument("--flatquant_separate_vtrans", action="store_true", default=False)
    parser.add_argument(
        "--flatquant_diag_init",
        type=str,
        default="sq_style",
        choices=["sq_style", "one_style"],
    )
    parser.set_defaults(flatquant_lwc=True, flatquant_lac=True, flatquant_add_diag=True)
    return parser.parse_args()


def _resolve_flat_parameters_path(path_or_dir: str) -> str:
    if not path_or_dir:
        raise ValueError("flat_parameters_path is empty.")
    if os.path.isdir(path_or_dir):
        resolved = os.path.join(path_or_dir, "flat_parameters.pth")
    else:
        resolved = path_or_dir
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"FlatQuant parameters not found: {resolved}")
    return resolved


def _build_flatquant_runtime_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        w_bits=args.weight_bits,
        a_bits=args.act_bits,
        q_bits=args.flatquant_q_bits,
        k_bits=args.flatquant_k_bits,
        v_bits=args.flatquant_v_bits,
        w_asym=False,
        a_asym=False,
        q_asym=False,
        k_asym=False,
        v_asym=False,
        a_groupsize=-1,
        lwc=args.flatquant_lwc,
        lac=args.flatquant_lac,
        add_diag=args.flatquant_add_diag,
        direct_inv=args.flatquant_direct_inv,
        separate_vtrans=args.flatquant_separate_vtrans,
        diag_init=args.flatquant_diag_init,
    )


@torch.no_grad()
def _apply_flatquant_parameters(model: nn.Module, args: argparse.Namespace) -> str:
    from flatquant.flat_utils import reparameterize_model
    from flatquant.model_tools.llama_utils import apply_flatquant_to_llama
    from flatquant.model_tools.llama31_utils import apply_flatquant_to_llama_31

    runtime_args = _build_flatquant_runtime_args(args)
    if "llama-3.1" in args.model_path.lower():
        model = apply_flatquant_to_llama_31(runtime_args, model)
    else:
        model = apply_flatquant_to_llama(runtime_args, model)

    resolved_path = _resolve_flat_parameters_path(args.flat_parameters_path)
    flat_parameters = torch.load(resolved_path, map_location="cpu")
    layers = model.model.layers
    loaded_layers = 0
    for key, flat_param in flat_parameters.items():
        layer_idx = int(key)
        layers[layer_idx].load_state_dict(flat_param, strict=False)
        loaded_layers += 1
    if loaded_layers <= 0:
        raise ValueError(f"No layer parameters loaded from {resolved_path}")

    reparameterize_model(model)
    print(f"[FlatQuant] Loaded and reparameterized {loaded_layers} layers from: {resolved_path}")
    return resolved_path


def _load_or_generate_act_scales(
    args: argparse.Namespace,
    tokenizer,
    dtype_map: Dict[str, torch.dtype],
) -> Tuple[Dict[str, torch.Tensor], str]:
    raw_act_scales_path = args.act_scales_path.strip()
    is_placeholder = raw_act_scales_path in {"", "/path/to/act_scales.pt", "path/to/act_scales.pt"}
    act_scales_path = raw_act_scales_path

    should_auto_generate = (
        args.auto_generate_act_scales
        or is_placeholder
        or (act_scales_path and not os.path.exists(act_scales_path))
    )
    if should_auto_generate:
        generated_path = os.path.join(args.output_dir, "generated_act_scales.wikitext2.pt")
        if not act_scales_path or is_placeholder:
            act_scales_path = generated_path

        if os.path.exists(act_scales_path):
            print(f"[ActScales] Found existing file: {act_scales_path}")
            act_scales = torch.load(act_scales_path, map_location="cpu")
        else:
            print(
                "[ActScales] Missing scales file, auto-generating from wikitext-2 "
                f"(split={args.act_scales_split}, samples={args.act_scales_samples}, seq_len={args.act_scales_seq_len})."
            )
            scale_model = load_model(args.model_path, dtype_map[args.dtype])
            act_scales = collect_act_scales_wikitext2(
                model=scale_model,
                tokenizer=tokenizer,
                split=args.act_scales_split,
                num_samples=args.act_scales_samples,
                seq_len=args.act_scales_seq_len,
            )
            torch.save(act_scales, act_scales_path)
            print(f"[ActScales] Saved generated scales to: {act_scales_path}")
            del scale_model
            torch.cuda.empty_cache()
    else:
        if not os.path.exists(act_scales_path):
            raise FileNotFoundError(
                f"act_scales file not found: {act_scales_path}. "
                "Use a valid file or enable --auto_generate_act_scales."
            )
        act_scales = torch.load(act_scales_path, map_location="cpu")
    return act_scales, act_scales_path


@torch.no_grad()
def run_single_experiment(
    *,
    args: argparse.Namespace,
    model_dtype: torch.dtype,
    low_dtype: torch.dtype,
    act_scales: Optional[Dict[str, torch.Tensor]],
    evaluator: WikiTextPplEvaluator,
    eval_config: EvalConfig,
) -> Dict[str, object]:
    model = load_model(args.model_path, model_dtype)
    flat_parameters_used = ""
    if args.flat_parameters_path:
        flat_parameters_used = _apply_flatquant_parameters(model, args)
    else:
        if act_scales is None:
            raise ValueError("act_scales is required when flat_parameters_path is not provided.")
        smooth_lm(model, act_scales, alpha=args.alpha)

    exclude_lm_head = not args.include_lm_head
    target_names = parse_target_modules(args.target_modules)
    target_paths = find_target_linear_paths(
        model,
        exclude_lm_head=exclude_lm_head,
        target_module_names=target_names,
    )
    if not target_paths:
        raise ValueError("No target linear layers found for replacement.")

    apply_frequency_hybrid_linears(
        model=model,
        target_paths=target_paths,
        act_bits=args.act_bits,
        weight_bits=args.weight_bits,
        ratio=args.hybrid_ratio,
        block_size=args.freq_block_size,
        low_dtype=low_dtype,
    )

    ppl = evaluator.evaluate(model, eval_config)
    del model
    torch.cuda.empty_cache()
    return {
        "scheme": "freq_only",
        "ppl": float(ppl),
        "num_linear_layers_applied": int(len(target_paths)),
        "flat_parameters_path": flat_parameters_used,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if not 0.0 < args.hybrid_ratio < 1.0:
        raise ValueError(f"hybrid_ratio must be in (0, 1), got {args.hybrid_ratio}")
    if args.weight_bits < 2:
        raise ValueError(f"weight_bits must be >= 2, got {args.weight_bits}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    low_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    act_scales: Optional[Dict[str, torch.Tensor]] = None
    act_scales_path: str = ""
    if not args.flat_parameters_path:
        act_scales, act_scales_path = _load_or_generate_act_scales(args, tokenizer, dtype_map)

    evaluator = WikiTextPplEvaluator(tokenizer=tokenizer, split=args.ppl_split)
    eval_config = EvalConfig(seq_len=args.ppl_seq_len, n_samples=args.ppl_samples)

    print("[Eval] Running frequency-only hybrid quantization.")
    result = run_single_experiment(
        args=args,
        model_dtype=dtype_map[args.dtype],
        low_dtype=low_dtype_map[args.low_precision],
        act_scales=act_scales,
        evaluator=evaluator,
        eval_config=eval_config,
    )
    print(f"[Eval] freq_only ppl={result['ppl']:.6f}")

    output = {
        "config": {
            "model_path": args.model_path,
            "act_scales_path": act_scales_path,
            "alpha": args.alpha,
            "dtype": args.dtype,
            "weight_bits": args.weight_bits,
            "act_bits": args.act_bits,
            "hybrid_ratio": args.hybrid_ratio,
            "freq_block_size": args.freq_block_size,
            "target_modules": args.target_modules,
            "low_precision": args.low_precision,
            "include_lm_head": args.include_lm_head,
            "ppl_split": args.ppl_split,
            "ppl_seq_len": args.ppl_seq_len,
            "ppl_samples": args.ppl_samples,
            "flat_parameters_path": args.flat_parameters_path,
            "flatquant_q_bits": args.flatquant_q_bits,
            "flatquant_k_bits": args.flatquant_k_bits,
            "flatquant_v_bits": args.flatquant_v_bits,
        },
        "result": result,
    }
    out_path = os.path.join(args.output_dir, "fre_energy_ppl_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(output), f, indent=2, ensure_ascii=False)
    print(f"[Eval] Saved result to: {out_path}")


if __name__ == "__main__":
    main()
