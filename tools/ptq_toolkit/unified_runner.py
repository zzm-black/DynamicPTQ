import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import torch
import transformers

from .adapters import FlatQuantAdapter, QuaRotAdapter, SpinQuantAdapter
from .config import DeepBitConfig, StageAConfig
from .pipeline import MethodToolkit

LOGGER = logging.getLogger(__name__)


@dataclass
class UnifiedRunConfig:
    workspace_root: Path
    backend: str
    model_path: str
    cali_dataset: str = "wikitext2"
    eval_datasets: List[str] = None
    nsamples: int = 128
    cali_bsz: int = 4
    epochs: int = 15
    w_bits: int = 4
    a_bits: int = 4
    k_bits: int = 4
    v_bits: int = 4
    method: str = "rtn"
    gptq_mse: bool = False
    percdamp: float = 0.01
    act_order: bool = False
    lm_eval: bool = False
    lm_eval_tasks: List[str] = None
    lm_eval_batch_size: int = 16
    distribute_model: bool = False
    eval_only: bool = False
    flat_parameters_path: str = None
    output_dir: str = "./outputs_unified"
    exp_name: str = "unified"
    seed: int = 0
    hf_token: str = None
    enable_deep: bool = True
    deep_act_bits: int = 8
    deep_layers: str = "0,1,2,31"

    def __post_init__(self) -> None:
        if self.eval_datasets is None:
            self.eval_datasets = ["wikitext2", "c4"]
        if self.lm_eval_tasks is None:
            self.lm_eval_tasks = ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai"]


class UnifiedPTQRunner:
    """
    Unified flow:
      1) Optional backend preprocess (QuaRot/SpinQuant)
      2) FlatQuant calibration and reparameterization
      3) FlatQuant evaluation (PPL / lm-eval)
    """

    def __init__(self) -> None:
        pass

    def run(self, cfg: UnifiedRunConfig) -> Dict[str, float]:
        self._prepare_import_path(cfg.workspace_root / "FlatQuant")
        import flatquant.data_utils as fq_data_utils
        import flatquant.model_utils as fq_model_utils
        import flatquant.train_utils as fq_train_utils
        import flatquant.eval_utils as fq_eval_utils
        import flatquant.flat_utils as fq_flat_utils
        import flatquant.utils as fq_utils

        transformers.set_seed(cfg.seed)
        fq_utils.seed_everything(seed=cfg.seed)

        model, apply_flatquant = fq_model_utils.get_model(cfg.model_path, cfg.hf_token)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.model_path,
            use_fast=False,
            use_auth_token=cfg.hf_token,
        )

        adapter = self._build_adapter(cfg)
        if hasattr(adapter, "apply_backend_preprocess") and cfg.backend in {"quarot", "spinquant"}:
            LOGGER.info("Applying %s preprocess before FlatQuant calibration.", cfg.backend)
            adapter.apply_backend_preprocess(model)

        fq_args = self._build_flatquant_args(cfg)
        model = apply_flatquant(fq_args, model)
        self._apply_deep_activation_bits(model, cfg)

        toolkit = self._build_toolkit(cfg, len(model.model.layers))
        toolkit.stage_a_log()
        if cfg.eval_only:
            fq_flat_utils.load_flat_parameters(fq_args, model, path=cfg.flat_parameters_path)
            LOGGER.info(
                "Loaded flat parameters from %s",
                cfg.flat_parameters_path or fq_args.exp_dir,
            )
        else:
            trainloader = fq_data_utils.get_loaders(
                fq_args,
                cfg.cali_dataset,
                tokenizer,
                nsamples=cfg.nsamples,
                seqlen=model.seqlen,
                eval_mode=False,
            )
            fq_train_utils.cali_flat_quant(fq_args, model, trainloader, fq_utils.DEV, logger=LOGGER)

        toolkit.run_post_calib(model, adapter)

        if cfg.distribute_model:
            fq_utils.distribute_model(model)
        else:
            model.to(fq_utils.DEV)
        metrics: Dict[str, float] = {}
        for eval_dataset in cfg.eval_datasets:
            testloader = fq_data_utils.get_loaders(
                fq_args,
                eval_dataset,
                tokenizer,
                seqlen=model.seqlen,
                eval_mode=True,
            )
            ppl = fq_eval_utils.ppl_eval(model, testloader)
            metrics[f"ppl/{eval_dataset}"] = float(ppl)
            LOGGER.info("ppl/%s = %.4f", eval_dataset, ppl)

        if cfg.lm_eval:
            metrics.update(self._run_lm_eval(model, tokenizer, cfg))

        return metrics

    def _build_adapter(self, cfg: UnifiedRunConfig):
        if cfg.backend == "flatquant":
            return FlatQuantAdapter()
        if cfg.backend == "quarot":
            return QuaRotAdapter(cfg.workspace_root / "QuaRot")
        if cfg.backend == "spinquant":
            return SpinQuantAdapter(cfg.workspace_root / "SpinQuant")
        raise ValueError(f"Unsupported backend: {cfg.backend}")

    @staticmethod
    def _build_toolkit(cfg: UnifiedRunConfig, num_layers: int) -> MethodToolkit:
        deep_layers = _parse_deep_layers(cfg.deep_layers, num_layers)
        if not cfg.enable_deep:
            deep_layers = tuple()
            deep_act_bits = cfg.a_bits
        else:
            deep_act_bits = cfg.deep_act_bits
        deep_layers_1based = tuple(layer_id + 1 for layer_id in deep_layers)
        deep_cfg = DeepBitConfig(
            default_act_bits=cfg.a_bits,
            late_layer_act_bits=deep_act_bits,
            late_layer_ids_1based=deep_layers_1based,
            kv_bits=cfg.k_bits,
        )
        stage_a_cfg = StageAConfig()
        return MethodToolkit(
            deep_cfg=deep_cfg,
            stage_a_cfg=stage_a_cfg,
        )

    @staticmethod
    def _prepare_import_path(repo_root: Path) -> None:
        import sys

        repo = str(repo_root.resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)

    @staticmethod
    def _build_flatquant_args(cfg: UnifiedRunConfig):
        # Keep outputs flat: caller controls final run directory via output_dir.
        exp_dir = Path(cfg.output_dir).resolve()
        exp_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            model=cfg.model_path,
            hf_token=cfg.hf_token,
            seed=cfg.seed,
            cali_dataset=cfg.cali_dataset,
            nsamples=cfg.nsamples,
            cali_bsz=cfg.cali_bsz,
            epochs=cfg.epochs,
            flat_lr=5e-3,
            cali_trans=True,
            add_diag=True,
            lwc=True,
            lac=True,
            diag_init="sq_style",
            diag_alpha=0.3,
            warmup=False,
            deactive_amp=False,
            direct_inv=False,
            separate_vtrans=False,
            resume=False,
            save_matrix=False,
            reload_matrix=False,
            matrix_path=None,
            quantize=True,
            w_bits=cfg.w_bits,
            a_bits=cfg.a_bits,
            k_bits=cfg.k_bits,
            v_bits=cfg.v_bits,
            q_bits=16,
            w_groupsize=-1,
            a_groupsize=-1,
            k_groupsize=128,
            v_groupsize=128,
            q_groupsize=-1,
            w_asym=False,
            a_asym=False,
            k_asym=True,
            v_asym=True,
            q_asym=False,
            gptq=(cfg.method == "gptq"),
            gptq_mse=cfg.gptq_mse,
            percdamp=cfg.percdamp,
            act_order=cfg.act_order,
            output_dir=cfg.output_dir,
            exp_name=cfg.exp_name,
            exp_dir=str(exp_dir),
            lm_eval=cfg.lm_eval,
            tasks=cfg.lm_eval_tasks,
            lm_eval_batch_size=cfg.lm_eval_batch_size,
            distribute_model=False,
            quantized_save=False,
        )

    @staticmethod
    def _apply_deep_activation_bits(model, cfg: UnifiedRunConfig) -> None:
        num_layers = len(model.model.layers)
        deep_layers = set(_parse_deep_layers(cfg.deep_layers, num_layers))
        for layer_idx, layer in enumerate(model.model.layers):
            target_bits = cfg.deep_act_bits if (cfg.enable_deep and layer_idx in deep_layers) else cfg.a_bits
            for module in layer.modules():
                act_q = getattr(module, "act_quantizer", None)
                if act_q is not None and hasattr(act_q, "bits"):
                    act_q.bits = int(target_bits)
                    if hasattr(act_q, "q_max") and hasattr(act_q, "q_min"):
                        qmax, qmin = _get_qmin_qmax(target_bits, bool(getattr(act_q, "sym", True)))
                        act_q.q_max = qmax.to(act_q.q_max.device)
                        act_q.q_min = qmin.to(act_q.q_min.device)

    @staticmethod
    def _run_lm_eval(model, tokenizer, cfg: UnifiedRunConfig) -> Dict[str, float]:
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=cfg.lm_eval_batch_size)
        results = {}
        for task_name in cfg.lm_eval_tasks:
            result = lm_eval.simple_evaluate(
                hflm,
                tasks=[task_name],
                batch_size=cfg.lm_eval_batch_size,
            )["results"][task_name]
            acc = round(result.get("acc_norm,none", result.get("acc,none", 0.0)) * 100, 2)
            results[f"lm_eval/{task_name}"] = acc
        if results:
            results["lm_eval/acc_avg"] = round(sum(results.values()) / len(results), 2)
        return results


def _parse_deep_layers(deep_layers: str, num_layers: int) -> Tuple[int, ...]:
    value = deep_layers.strip().lower()
    if value == "last5":
        start = max(0, num_layers - 5)
        return tuple(range(start, num_layers))
    layers: List[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        layer_id = int(token)
        if layer_id < 0:
            raise ValueError(f"deep layer id must be 0-based non-negative int, got: {layer_id}")
        if layer_id >= num_layers:
            raise ValueError(f"deep layer id out of range [0, {num_layers - 1}], got: {layer_id}")
        layers.append(layer_id)
    if not layers:
        raise ValueError("deep_layers must be 'last5' or comma-separated 0-based layer ids.")
    return tuple(sorted(set(layers)))


def _get_qmin_qmax(bits: int, sym: bool):
    if sym:
        q_max = torch.tensor(2 ** (bits - 1) - 1)
        q_min = -q_max - 1
    else:
        q_max = torch.tensor(2**bits - 1)
        q_min = torch.tensor(0)
    return q_max, q_min

