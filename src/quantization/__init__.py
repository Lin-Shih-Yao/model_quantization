"""
Quantization Core Modules (Calibration, Observers, Datasets, W8A16, W8A8, Utilities)
"""

from src.quantization.dataset import get_calib_dataset
from src.quantization.observer import LayerObserver
from src.quantization.calibration import calibrate_model
from src.quantization.w8a16 import W8A16Linear, quantize_model_w8a16
from src.quantization.w8a8 import W8A8Linear, quantize_model_w8a8
from src.quantization.utils import (
    quantize_weight_int8,
    replace_linear_with_factory,
    load_calib_layers,
    find_calib_scale_and_clip,
)

__all__ = [
    "get_calib_dataset",
    "LayerObserver",
    "calibrate_model",
    "W8A16Linear",
    "quantize_model_w8a16",
    "W8A8Linear",
    "quantize_model_w8a8",
    "quantize_weight_int8",
    "replace_linear_with_factory",
    "load_calib_layers",
    "find_calib_scale_and_clip",
]
