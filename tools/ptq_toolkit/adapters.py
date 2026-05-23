from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable
import importlib
import sys

import torch
import torch.nn as nn


@dataclass
class LinearHandle:
    name: str
    module: nn.Module
    layer_id_1based: int


class PTQAdapter(ABC):
    """
    Adapter contract for integrating the method with different PTQ repos.

    Each PTQ implementation only needs to implement this interface.
    """

    @abstractmethod
    def iter_target_linears(self, model: nn.Module) -> Iterable[LinearHandle]:
        pass

    @abstractmethod
    def reparameterize_model(self, model: nn.Module) -> None:
        pass

    @abstractmethod
    def apply_deploy_weight(self, handle: LinearHandle, new_weight: torch.Tensor) -> None:
        pass


class FlatQuantAdapter(PTQAdapter):
    def iter_target_linears(self, model: nn.Module) -> Iterable[LinearHandle]:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_id = _guess_layer_id_1based(name)
                yield LinearHandle(name=name, module=module, layer_id_1based=layer_id)

    def reparameterize_model(self, model: nn.Module) -> None:
        from flatquant.flat_utils import reparameterize_model

        reparameterize_model(model)

    def apply_deploy_weight(self, handle: LinearHandle, new_weight: torch.Tensor) -> None:
        handle.module.weight.data.copy_(new_weight.to(handle.module.weight.dtype))


class QuaRotAdapter(FlatQuantAdapter):
    """
    Keep QuaRot-specific preprocessing and reuse FlatQuant calibration/evaluation.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)

    def apply_backend_preprocess(self, model: nn.Module) -> None:
        repo_path = str(self.repo_root / "fake_quant")
        with _sys_path(repo_path):
            rotation_utils = importlib.import_module("rotation_utils")
            quant_utils = importlib.import_module("quant_utils")
            hadamard_utils = importlib.import_module("hadamard_utils")

            args = SimpleNamespace(rotate_mode="hadamard", fp32_had=False)
            rotation_utils.fuse_layer_norms(model)
            rotation_utils.rotate_model(model, args)
            quant_utils.add_actquant(model)
            qlayers = quant_utils.find_qlayers(model)
            for name in qlayers:
                if "down_proj" in name:
                    had_k, k = hadamard_utils.get_hadK(model.config.intermediate_size)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_k
                    qlayers[name].K = k
                    qlayers[name].fp32_had = False
                if "o_proj" in name:
                    had_k, k = hadamard_utils.get_hadK(model.config.num_attention_heads)
                    qlayers[name].online_partial_had = True
                    qlayers[name].had_K = had_k
                    qlayers[name].K = k
                    qlayers[name].had_dim = model.config.hidden_size // model.config.num_attention_heads
                    qlayers[name].fp32_had = False


class SpinQuantAdapter(FlatQuantAdapter):
    """
    Keep SpinQuant-specific preprocessing and reuse FlatQuant calibration/evaluation.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)

    def apply_backend_preprocess(self, model: nn.Module) -> None:
        repo_path = str(self.repo_root)
        with _sys_path(repo_path):
            fuse_norm_utils = importlib.import_module("utils.fuse_norm_utils")
            quant_utils = importlib.import_module("utils.quant_utils")
            hadamard_utils = importlib.import_module("utils.hadamard_utils")
            rotation_utils = importlib.import_module("eval_utils.rotation_utils")

            args = SimpleNamespace(
                rotate_mode="hadamard",
                fp32_had=False,
                optimized_rotation_path=None,
            )
            fuse_norm_utils.fuse_layer_norms(model)
            rotation_utils.rotate_model(model, args)
            quant_utils.add_actquant(model)
            qlayers = quant_utils.find_qlayers(model)
            for name in qlayers:
                if "down_proj" in name:
                    had_k, k = hadamard_utils.get_hadK(model.config.intermediate_size)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_k
                    qlayers[name].K = k
                    qlayers[name].fp32_had = False


def _guess_layer_id_1based(module_name: str) -> int:
    parts = module_name.split(".")
    for idx, token in enumerate(parts[:-1]):
        if token == "layers" and parts[idx + 1].isdigit():
            return int(parts[idx + 1]) + 1
    return 1


class _sys_path:
    def __init__(self, prepend: str) -> None:
        self.prepend = prepend

    def __enter__(self):
        self._old_path = list(sys.path)
        sys.path.insert(0, self.prepend)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.path[:] = self._old_path
        return False
