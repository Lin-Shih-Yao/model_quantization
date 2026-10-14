"""
Calibration Profiler Module (Backward-Compatibility Adapter)
Delegates extreme value profiling to the unified LayerObserver and calibration engine.
"""

from typing import Optional
import torch
import torch.nn as nn
from src.quantization.calibration import (
    collect_calibration_stats,
    save_calibration_stats_json as save_stats_to_json,
)


class CalibrationProfiler:
    """
    輕量適配器：底層完全呼叫統一的 LayerObserver 校準引擎，消除重複的前向 Hook 邏輯。
    """

    def __init__(self, clip_quantile: float = 0.9999):
        self.clip_quantile = clip_quantile
        self.layer_stats: dict[str, dict] = {}

    @torch.no_grad()
    def collect_stats(
        self,
        model: nn.Module,
        calib_tensor: torch.Tensor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> dict[str, dict]:
        """呼叫統一校準引擎收集全層統計資料。"""
        self.layer_stats = collect_calibration_stats(
            model=model,
            samples=calib_tensor,
            device=device,
            clip_quantile=self.clip_quantile,
        )
        return self.layer_stats


__all__ = ["CalibrationProfiler", "save_stats_to_json"]
