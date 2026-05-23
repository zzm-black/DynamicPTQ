from typing import Iterable, Set

from .config import DeepBitConfig


class DeepBitScheduler:
    """DEEP activation bit rule with optional 0/1-based layer id input."""

    def __init__(self, config: DeepBitConfig) -> None:
        self.config = config
        self._late_layers: Set[int] = set(config.late_layer_ids_1based)

    def activation_bits(self, layer_id: int, *, one_based: bool = True) -> int:
        layer_1based = layer_id if one_based else layer_id + 1
        if layer_1based in self._late_layers:
            return self.config.late_layer_act_bits
        return self.config.default_act_bits

    def kv_bits(self) -> int:
        return self.config.kv_bits

    def as_log_string(self) -> str:
        ids = sorted(self._late_layers)
        return (
            f"DEEP bits: {ids} -> {self.config.late_layer_act_bits}, "
            f"others -> {self.config.default_act_bits}, kv -> {self.config.kv_bits}"
        )

    def bit_vector(self, total_layers: int, *, one_based: bool = True) -> Iterable[int]:
        for layer_id in range(1 if one_based else 0, total_layers + (1 if one_based else 0)):
            yield self.activation_bits(layer_id, one_based=one_based)

