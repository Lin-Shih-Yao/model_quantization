"""
Calibration Profiler Module (Extreme Values Profiler)
Collects scalar min and max activation statistics across target linear layers.
Leverages existing target layer matching, hook removal, and memory cleanup from calibration.py.
"""

import json
import os
import torch
import torch.nn as nn
from typing import Optional

from src.quantization.calibration import (
    find_target_linear_modules,
    remove_all_hooks,
    clean_memory,
)


class CalibrationProfiler:
    """
    輕量化特徵極值收集器：
    僅記錄各層 Activation 之全域最小值 (min) 與最大值 (max)，專為非對稱量化服務。
    """

    def __init__(self):
        self.layer_stats: dict[str, dict[str, float]] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _create_hook_fn(self, name: str):
        """建立指定層的 forward hook 攔截回呼。"""
        def _hook(module: nn.Module, input_tensor: tuple, output_tensor: torch.Tensor):
            if not input_tensor or input_tensor[0] is None:
                return

            x = input_tensor[0]
            with torch.no_grad():
                cur_min = float(x.detach().min().cpu().item())
                cur_max = float(x.detach().max().cpu().item())

                stats = self.layer_stats[name]
                stats["min"] = min(stats["min"], cur_min)
                stats["max"] = max(stats["max"], cur_max)

        return _hook

    def register_hooks(self, model: nn.Module) -> int:
        """遍歷模型並僅針對核心 Linear 投影層掛載 Forward Hook。"""
        self.remove_hooks()
        self.layer_stats.clear()

        target_modules = find_target_linear_modules(model)
        for name, module in target_modules.items():
            self.layer_stats[name] = {
                "min": float("inf"),
                "max": float("-inf"),
            }
            handle = module.register_forward_hook(self._create_hook_fn(name))
            self.hooks.append(handle)

        print(f"🪝 [模組 2] 成功掛載 {len(self.hooks)} 個核心 Linear 模組之極值監視器！")
        return len(self.hooks)

    def remove_hooks(self):
        """乾淨卸載所有 Forward Hooks。"""
        if self.hooks:
            remove_all_hooks(self.hooks)
            print("🧹 [模組 2] 已乾淨卸載所有 Forward Hooks。")

    @torch.no_grad()
    def collect_stats(
        self,
        model: nn.Module,
        calib_tensor: torch.Tensor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> dict[str, dict[str, float]]:
        """執行單次前向傳播收集全量樣本極值，並確保自動卸載 Hooks。"""
        self.register_hooks(model)
        model.eval()

        try:
            n_samples = calib_tensor.shape[0]
            for i in range(n_samples):
                batch = calib_tensor[i : i + 1].to(device)
                model(batch)
                del batch
                clean_memory(device)
        finally:
            self.remove_hooks()

        print(f"✅ [模組 2] 極值收集完畢，共統計 {len(self.layer_stats)} 個核心層！")
        return self.layer_stats


def save_stats_to_json(
    stats: dict[str, dict[str, float]],
    model_name: str,
    dataset_name: str,
    n_samples: int,
    node_name: str = "Node_A",
    custom_dir: Optional[str] = None,
) -> str:
    """將極值字典儲存為 JSON 檔，檔名嵌入 node_name 以支援雙開 Colab 存檔防撞。"""
    safe_model = model_name.split("/")[-1].replace(" ", "_")
    if custom_dir:
        save_dir = custom_dir
    elif os.path.exists("/content/drive/MyDrive"):
        save_dir = "/content/drive/MyDrive/calibration_stats"
    else:
        save_dir = "./calibration_results"

    os.makedirs(save_dir, exist_ok=True)
    filename = f"stats_{node_name}_{safe_model}_{dataset_name}_N{n_samples}.json"
    full_path = os.path.join(save_dir, filename)

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"💾 [模組 2] 極值統計已儲存至: {full_path}")
    return full_path
