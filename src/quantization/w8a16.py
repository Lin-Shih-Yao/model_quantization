"""
W8A16 Quantization Module (Weight INT8, Activation FP16/BF16)
Provides custom W8A16Linear layer and recursive model-level quantizer.
Every function is kept strictly concise (< 20 lines) for readability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_MODULE_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
)


class W8A16Linear(nn.Module):
    """自訂 W8A16 線性層：權重以 INT8 儲存，推論時即時反量化執行矩陣乘法。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.empty((out_features, 1), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "W8A16Linear":
        """將原生 nn.Linear 轉換為 W8A16Linear，對齊硬體裝置與精度並轉為 INT8。"""
        qlayer = cls(linear.in_features, linear.out_features, bias=(linear.bias is not None))
        qlayer.to(device=linear.weight.device, dtype=linear.weight.dtype)
        w = linear.weight.data.float()

        # Per-channel 對稱 Scale: S = max(|W|, dim=1) / 127
        scales = torch.clamp(w.abs().amax(dim=1, keepdim=True) / 127.0, min=1e-8)
        w_int8 = torch.clamp(torch.round(w / scales), -127, 127).to(torch.int8)

        qlayer.weight_int8.copy_(w_int8)
        qlayer.scales.copy_(scales)
        if linear.bias is not None:
            qlayer.bias.data.copy_(linear.bias.data)
        return qlayer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向推論：反量化回輸入精度並對齊 Bias 精度執行線性投影。"""
        w_dequant = self.weight_int8.to(x.dtype) * self.scales.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_dequant, bias)


def replace_linear_modules(module: nn.Module, target_names: tuple = TARGET_MODULE_NAMES) -> int:
    """遞迴走訪神經網路子層，遇到目標名稱的 Linear 即替換為 W8A16Linear。"""
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(t in name for t in target_names):
            setattr(module, name, W8A16Linear.from_linear(child))
            count += 1
        else:
            count += replace_linear_modules(child, target_names)
    return count


def quantize_model_w8a16(model: nn.Module, target_names: tuple = TARGET_MODULE_NAMES) -> nn.Module:
    """將模型的核心 Linear 層無縫替換為 W8A16 INT8 權重，保留 Embedding 與 lm_head。"""
    replaced_count = replace_linear_modules(model, target_names)
    print(f"✅ 成功將 {replaced_count} 個核心 Linear 模組替換為 W8A16 (INT8 權重)！")
    return model
