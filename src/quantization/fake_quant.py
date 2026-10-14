"""
Fake Quantizer Module (Activation Fake Quantization)
Simulates INT8 / INT4 activations using Forward Pre-Hooks and layer calibration statistics.
Supports both symmetric (aligned with physical W8A8) and asymmetric quantization.
Every function is kept strictly concise (< 20 lines) for readability.
"""

import json
import torch
import torch.nn as nn
from src.quantization.utils import TARGET_MODULE_NAMES


def symmetric_fake_quantize(
    x: torch.Tensor,
    scale: float | torch.Tensor,
    clip_max: float | None = None,
    bits: int = 8,
) -> torch.Tensor:
    """對輸入張量 X 執行對稱偽量化 (Clip -> Quantize -> Clamp -> Dequantize)。"""
    q_max = (1 << (bits - 1)) - 1
    s = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale, device=x.device, dtype=x.dtype)
    s = s.to(device=x.device, dtype=x.dtype)

    if clip_max is not None:
        clip_bound = torch.tensor(clip_max, device=x.device, dtype=x.dtype)
        x_val = torch.clamp(x, -clip_bound, clip_bound)
    else:
        x_val = x

    x_q = torch.clamp(torch.round(x_val / s), -q_max, q_max)
    return x_q * s


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


def extract_layer_quant_params(
    layer_info: dict,
    bits: int = 8,
    mode: str = "symmetric",
) -> tuple[float | torch.Tensor, float | None, float, float]:
    """從層統計字典萃取對稱/非對稱量化參數 (scale, clip_max, min_val, max_val)。"""
    min_v = float(layer_info.get("min_val", layer_info.get("min", -1.0)))
    max_v = float(layer_info.get("max_val", layer_info.get("max", 1.0)))

    scales = layer_info.get("scales", {})
    scale_key = f"int{bits}_sym_scale"
    if scale_key in scales:
        scale = scales[scale_key]
    else:
        q_max = (1 << (bits - 1)) - 1
        abs_max = max(abs(min_v), abs(max_v), layer_info.get("abs_max", 1.0))
        scale = max(abs_max, 1e-8) / float(q_max)

    clip_max = layer_info.get("clip_max_9999", None)
    return scale, clip_max, min_v, max_v


def load_stats_data(stats_data: dict | str) -> dict:
    """統一載入統計數據，支援 .pt、.json 檔案路徑與記憶體 dict。"""
    if isinstance(stats_data, str):
        if stats_data.endswith(".pt"):
            loaded = torch.load(stats_data, map_location="cpu")
        else:
            with open(stats_data, "r", encoding="utf-8") as f:
                loaded = json.load(f)
    else:
        loaded = stats_data
    return loaded.get("layers", loaded)


class FakeQuantizer:
    """純激活值偽量化管理器：使用 Forward Pre-Hook 動態注入與安全卸載。"""

    def __init__(self, target_names: tuple = TARGET_MODULE_NAMES, mode: str = "symmetric"):
        self.target_names = target_names
        self.mode = mode
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self.current_bits: int | None = None

    def _create_pre_hook(self, layer_info: dict, bits: int):
        """建立 Pre-Hook 回呼：在輸入進入 Linear 前原地替換 Activation。"""
        scale, clip_max, min_v, max_v = extract_layer_quant_params(layer_info, bits=bits, mode=self.mode)

        def _pre_hook(module: nn.Module, inputs: tuple):
            if not inputs or inputs[0] is None:
                return inputs
            if self.mode == "symmetric":
                x_q = symmetric_fake_quantize(inputs[0], scale=scale, clip_max=clip_max, bits=bits)
            else:
                x_q = asymmetric_fake_quantize(inputs[0], min_v, max_v, bits=bits)
            return (x_q,) + inputs[1:]

        return _pre_hook

    def attach(self, model: nn.Module, stats_data: dict | str, bits: int = 8) -> int:
        """讀取校準統計，並為模型目標層掛載 Activation Pre-Hook。"""
        self.detach()
        self.current_bits = bits
        stats = load_stats_data(stats_data)

        count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(t in name for t in self.target_names):
                matched = next((stats[k] for k in stats if k in name or name in k), None)
                if matched is None:
                    continue
                handle = module.register_forward_pre_hook(self._create_pre_hook(matched, bits=bits))
                self.hooks.append(handle)
                count += 1

        print(f"🎯 [模組 3] 成功為 {count} 個核心層掛載 {bits}-bit ({self.mode}) Activation 偽量化器！")
        return count

    def detach(self):
        """乾淨卸載所有 Hooks，模型秒速還原為原始無量化狀態。"""
        if self.hooks:
            for handle in self.hooks:
                handle.remove()
            self.hooks.clear()
            print("🧹 [模組 3] 已乾淨卸載所有偽量化 Hooks，模型回到原始乾淨狀態。")
        self.current_bits = None
