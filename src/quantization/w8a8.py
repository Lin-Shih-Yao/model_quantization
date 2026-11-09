"""
W8A8 Quantization Module (Weight INT8, Activation INT8)
Simulates INT8 weights and INT8 activations with 99.99% quantile clipping and RTN rounding.
Every function is kept strictly concise (< 20 lines) for readability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.quantization.utils import (
    TARGET_MODULE_NAMES,
    quantize_weight_int8,
    replace_linear_with_factory,
    load_calib_layers,
    find_calib_scale_and_clip,
)


class W8A8Linear(nn.Module):
    """Custom W8A8 Linear layer: performs matrix operations with INT8 weights and INT8 activations."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        act_scale: float = 1.0,
        act_clip_max: float | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scales", torch.empty((out_features, 1), dtype=torch.float32))
        self.register_buffer("act_scale", torch.tensor(act_scale, dtype=torch.float32))
        self.act_clip_max = act_clip_max
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        act_scale: float = 1.0,
        act_clip_max: float | None = None,
    ) -> "W8A8Linear":
        """Converts native nn.Linear into W8A8Linear, assigning INT8 weights and activation scale/clip parameters."""
        qlayer = cls(
            linear.in_features,
            linear.out_features,
            bias=(linear.bias is not None),
            act_scale=act_scale,
            act_clip_max=act_clip_max,
        )
        qlayer.to(device=linear.weight.device, dtype=linear.weight.dtype)
        w_int8, w_scales = quantize_weight_int8(linear.weight.data)
        qlayer.weight_int8.copy_(w_int8)
        qlayer.weight_scales.copy_(w_scales)
        if linear.bias is not None:
            qlayer.bias.data.copy_(linear.bias.data)
        return qlayer

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Applies 99.99% quantile clipping and symmetric INT8 rounding to activation feature X."""
        s_act = self.act_scale.to(device=x.device, dtype=x.dtype)
        if self.act_clip_max is not None:
            clip_bound = torch.tensor(self.act_clip_max, device=x.device, dtype=x.dtype)
            x_val = torch.clamp(x, -clip_bound, clip_bound)
        else:
            x_val = x
        x_q = torch.clamp(torch.round(x_val / s_act), -127, 127)
        return x_q * s_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: quantizes activations, dequantizes weights, and executes linear projection."""
        x_dequant = self.quantize_activation(x)
        w_dequant = self.weight_int8.to(x.dtype) * self.weight_scales.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x_dequant, w_dequant, bias)


def replace_linear_modules_w8a8(
    module: nn.Module,
    calib_layers: dict,
    target_names: tuple = TARGET_MODULE_NAMES,
) -> int:
    """Recursively replaces targeted Linear modules with W8A8Linear matching calibration stats."""
    def factory(linear: nn.Linear, full_name: str) -> W8A8Linear:
        scale, clip_max = find_calib_scale_and_clip(calib_layers, full_name)
        return W8A8Linear.from_linear(linear, act_scale=scale, act_clip_max=clip_max)

    return replace_linear_with_factory(module, factory, target_names=target_names)


def quantize_model_w8a8(
    model: nn.Module,
    calib_file: str | None = None,
    target_names: tuple = TARGET_MODULE_NAMES,
) -> nn.Module:
    """Unified entrypoint: replaces core linear projections with W8A8 while preserving Embeddings & Head."""
    calib_layers = load_calib_layers(calib_file)
    replaced_count = replace_linear_modules_w8a8(model, calib_layers, target_names=target_names)
    print(f"✅ Successfully replaced {replaced_count} core Linear modules with W8A8 (INT8 weights + INT8 activations)!")
    return model
