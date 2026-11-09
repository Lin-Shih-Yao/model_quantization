"""
W8A16 Quantization Module (Weight INT8, Activation FP16/BF16)
Provides custom W8A16Linear layer and recursive model-level quantizer.
Every function is kept strictly concise (< 20 lines) for readability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.quantization.utils import (
    TARGET_MODULE_NAMES,
    quantize_weight_int8,
    replace_linear_with_factory,
)


class W8A16Linear(nn.Module):
    """Custom W8A16 Linear layer: stores weights in INT8 and dequantizes on-the-fly during forward pass."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.empty((out_features, 1), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "W8A16Linear":
        """Converts native nn.Linear into W8A16Linear, aligning device/dtype and quantizing weights to INT8."""
        qlayer = cls(linear.in_features, linear.out_features, bias=(linear.bias is not None))
        qlayer.to(device=linear.weight.device, dtype=linear.weight.dtype)
        w_int8, scales = quantize_weight_int8(linear.weight.data)
        qlayer.weight_int8.copy_(w_int8)
        qlayer.scales.copy_(scales)
        if linear.bias is not None:
            qlayer.bias.data.copy_(linear.bias.data)
        return qlayer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: dequantizes INT8 weights to activation precision and executes linear projection."""
        w_dequant = self.weight_int8.to(x.dtype) * self.scales.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_dequant, bias)


def replace_linear_modules(module: nn.Module, target_names: tuple = TARGET_MODULE_NAMES) -> int:
    """Recursively replaces targeted Linear modules with W8A16Linear."""
    return replace_linear_with_factory(
        module,
        factory_fn=lambda linear, _name: W8A16Linear.from_linear(linear),
        target_names=target_names,
    )


def quantize_model_w8a16(model: nn.Module, target_names: tuple = TARGET_MODULE_NAMES) -> nn.Module:
    """Seamlessly replaces core Linear projections with physical W8A16 layers while preserving Embeddings & Head."""
    replaced_count = replace_linear_modules(model, target_names)
    print(f"✅ Successfully replaced {replaced_count} core Linear modules with physical W8A16 (INT8 weights)!")
    return model
