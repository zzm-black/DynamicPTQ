import argparse
import json
import logging
from pathlib import Path

from tools.ptq_toolkit.unified_runner import UnifiedPTQRunner, UnifiedRunConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified PTQ runner with FlatQuant calibration/evaluation.")
    parser.add_argument("--workspace_root", type=str, default="/home/zmzhao6/EMNLP")
    parser.add_argument("--backend", type=str, choices=["flatquant", "quarot", "spinquant"], required=True)
    parser.add_argument("--model", type=str, required=True, help="HF model path or local model path.")
    parser.add_argument("--cali_dataset", type=str, default="wikitext2")
    parser.add_argument("--eval_datasets", nargs="+", default=["wikitext2", "c4"])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--cali_bsz", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--k_bits", type=int, default=4)
    parser.add_argument("--v_bits", type=int, default=4)
    parser.add_argument("--method", type=str, choices=["rtn", "gptq"], default="rtn")
    parser.add_argument("--gptq_mse", action="store_true")
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--act_order", action="store_true")
    parser.add_argument("--lm_eval", action="store_true")
    parser.add_argument(
        "--lm_eval_tasks",
        nargs="+",
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai"],
    )
    parser.add_argument("--lm_eval_batch_size", type=int, default=16)
    parser.add_argument("--distribute_model", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--flat_parameters_path",
        type=str,
        default=None,
        help="Directory containing flat_parameters.pth for eval-only runs.",
    )
    parser.add_argument("--enable_deep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deep_act_bits", type=int, default=8)
    parser.add_argument("--deep_layers", type=str, default="0,1,2,31", help="Comma-separated 0-based layer ids.")
    parser.add_argument("--output_dir", type=str, default="./outputs_unified")
    parser.add_argument("--exp_name", type=str, default="unified")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf_token", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    cfg = UnifiedRunConfig(
        workspace_root=Path(args.workspace_root),
        backend=args.backend,
        model_path=args.model,
        cali_dataset=args.cali_dataset,
        eval_datasets=args.eval_datasets,
        nsamples=args.nsamples,
        cali_bsz=args.cali_bsz,
        epochs=args.epochs,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        method=args.method,
        gptq_mse=args.gptq_mse,
        percdamp=args.percdamp,
        act_order=args.act_order,
        lm_eval=args.lm_eval,
        lm_eval_tasks=args.lm_eval_tasks,
        lm_eval_batch_size=args.lm_eval_batch_size,
        distribute_model=args.distribute_model,
        eval_only=args.eval_only,
        flat_parameters_path=args.flat_parameters_path,
        enable_deep=args.enable_deep,
        deep_act_bits=args.deep_act_bits,
        deep_layers=args.deep_layers,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        seed=args.seed,
        hf_token=args.hf_token,
    )
    runner = UnifiedPTQRunner()
    metrics = runner.run(cfg)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

