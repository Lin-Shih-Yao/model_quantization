"""
Quantization Utility Functions & Shared Helpers
Provides common weight quantization math, calibration lookup, and module tree replacements.
Every function is kept strictly concise (< 20 lines) for readability.
"""

import os
import torch
import torch.nn as nn

TARGET_MODULE_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
)


def quantize_weight_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """計算 Per-channel 對稱 Scale 並將權重張量四捨五入為 INT8 整數。"""
    w = weight.float()
    scales = torch.clamp(w.abs().amax(dim=1, keepdim=True) / 127.0, min=1e-8)
    w_int8 = torch.clamp(torch.round(w / scales), -127, 127).to(torch.int8)
    return w_int8, scales


def replace_linear_with_factory(
    module: nn.Module,
    factory_fn,
    target_names: tuple = TARGET_MODULE_NAMES,
    prefix: str = "",
) -> int:
    """通用遞迴走訪器：遇到目標 Linear 即以工廠函式產出的量化層替換。"""
    count = 0
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and any(t in name for t in target_names):
            setattr(module, name, factory_fn(child, full_name))
            count += 1
        else:
            count += replace_linear_with_factory(child, factory_fn, target_names, prefix=full_name)
    return count


def load_calib_layers(calib_file: str | None = None) -> dict:
    """讀取校準快取 .pt 檔案中的 layers 統計字典。"""
    if calib_file and os.path.exists(calib_file):
        data = torch.load(calib_file, map_location="cpu")
        print(f"📖 成功載入校準快取數據: {calib_file} (包含 {len(data.get('layers', {}))} 層)")
        return data.get("layers", {})
    return {}


def find_calib_scale_and_clip(calib_layers: dict, module_full_name: str) -> tuple[float, float | None]:
    """從校準字典中尋找該層匹配的 Activation Scale 與 99.99% 截斷門檻。"""
    for key, info in calib_layers.items():
        if key in module_full_name or module_full_name in key:
            scale = info["scales"].get("int8_sym_scale", 1.0)
            clip_max = info.get("clip_max_9999", None)
            return scale, clip_max
    return 1.0, None
