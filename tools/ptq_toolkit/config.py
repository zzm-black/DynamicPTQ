from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DeepBitConfig:
    """Static activation bit schedule for DEEP."""

    default_act_bits: int = 4
    late_layer_act_bits: int = 8
    late_layer_ids_1based: Tuple[int, ...] = (27, 28, 29, 30, 31)
    kv_bits: int = 4


@dataclass(frozen=True)
class StageAConfig:
    """Calibration stage config (FlatQuant-only optimization)."""

    enabled: bool = True
    optimize_flatquant_only: bool = True
    loss_type: str = "layer_mse"

