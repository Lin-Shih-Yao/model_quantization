"""
Fake Quantizer Module (Activation Asymmetric Fake Quantization)
Simulates INT8 / INT4 activations using Forward Pre-Hooks and layer min/max statistics.
Every function is kept strictly concise (< 20 lines) for readability.
"""

import json
import torch
import torch.nn as nn
from src.quantization.utils import TARGET_MODULE_NAMES


def asymmetric_fake_quantize(
    x: torch.Tensor,
    min_val: float,
    max_val: float,
    bits: int = 8,
) -> torch.Tensor:
    """對輸入張量 X 執行非對稱偽量化 (Quantize -> Clamp -> Dequantize)。"""
    q_max = (1 << bits) - 1
    val_range = max(max_val - min_val, 1e-8)
    scale = val_range / float(q_max)
    zero_point = round(-min_val / scale)
    zero_point = max(0, min(q_max, zero_point))

    s = torch.tensor(scale, device=x.device, dtype=x.dtype)
    zp = torch.tensor(zero_point, device=x.device, dtype=x.dtype)

    x_int = torch.clamp(torch.round(x / s) + zp, 0, q_max)
    return (x_int - zp) * s


class FakeQuantizer:
    """純激活值偽量化管理器：使用 Forward Pre-Hook 動態注入與安全卸載。"""

    def __init__(self, target_names: tuple = TARGET_MODULE_NAMES):
        self.target_names = target_names
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self.current_bits: int | None = None

    def _create_pre_hook(self, min_val: float, max_val: float, bits: int):
        """建立 Pre-Hook 回呼：在輸入進入 Linear 前原地替換 Activation。"""
        def _pre_hook(module: nn.Module, inputs: tuple):
            if not inputs or inputs[0] is None:
                return inputs
            x_q = asymmetric_fake_quantize(inputs[0], min_val, max_val, bits=bits)
            return (x_q,) + inputs[1:]

        return _pre_hook

    def attach(self, model: nn.Module, stats_data: dict | str, bits: int = 8) -> int:
        """讀取模組 2 的極值統計，並為模型掛載 Activation Pre-Hook。"""
        self.detach()
        self.current_bits = bits

        if isinstance(stats_data, str):
            with open(stats_data, "r", encoding="utf-8") as f:
                stats = json.load(f)
        else:
            stats = stats_data

        count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(t in name for t in self.target_names):
                if name not in stats:
                    continue
                min_v = stats[name]["min"]
                max_v = stats[name]["max"]
                handle = module.register_forward_pre_hook(
                    self._create_pre_hook(min_v, max_v, bits=bits)
                )
                self.hooks.append(handle)
                count += 1

        print(f"🎯 [模組 3] 成功為 {count} 個核心層掛載 {bits}-bit Activation 偽量化器！")
        return count

    def detach(self):
        """乾淨卸載所有 Hooks，模型秒速還原為原始無量化狀態。"""
        if self.hooks:
            for handle in self.hooks:
                handle.remove()
            self.hooks.clear()
            print("🧹 [模組 3] 已乾淨卸載所有偽量化 Hooks，模型回到原始乾淨狀態。")
        self.current_bits = None
