import logging
from dataclasses import dataclass

import torch.nn as nn

from .adapters import PTQAdapter
from .config import DeepBitConfig, StageAConfig
from .deep import DeepBitScheduler

LOGGER = logging.getLogger(__name__)


@dataclass
class MethodToolkit:
    """
    Method driver:
    - Stage A: FlatQuant calibration with DEEP bit scheduling
    - Stage B: Reparameterize for deployment
    """

    deep_cfg: DeepBitConfig
    stage_a_cfg: StageAConfig

    def __post_init__(self) -> None:
        self.deep_scheduler = DeepBitScheduler(self.deep_cfg)

    def stage_a_log(self) -> None:
        LOGGER.info(self.deep_scheduler.as_log_string())

    def stage_b_log(self) -> None:
        LOGGER.info("FlatQuant reparameterize done: True")
        LOGGER.info(self.deep_scheduler.as_log_string())

    def run_post_calib(self, model: nn.Module, adapter: PTQAdapter) -> None:
        """
        Stage B core: reparameterize model by adapter.
        """
        adapter.reparameterize_model(model)
        self.stage_b_log()

