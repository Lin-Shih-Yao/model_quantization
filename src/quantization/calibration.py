"""
Calibration Orchestration Module
Manages layer hooks, executes single forward passes over calibration tokens,
and serializes layer statistics and scales into a reusable .pt checkpoint.
"""

import gc
import os
import time
import torch
from tqdm import tqdm

from src.quantization.dataset import get_calib_dataset
from src.quantization.observer import LayerObserver

TARGET_PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
)


def find_target_linear_modules(model: torch.nn.Module) -> dict[str, torch.nn.Linear]:
    """掃描模型內部所有屬於注意力機制與 MLP 的核心 Linear 層。"""
    targets = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(proj in name for proj in TARGET_PROJECTIONS):
                targets[name] = module
    return targets


def attach_observers(
    target_modules: dict[str, torch.nn.Linear],
    clip_quantile: float = 0.9999,
) -> tuple[dict[str, LayerObserver], list]:
    """為所有目標 Linear 層掛載 LayerObserver 與 Forward Hook。"""
    observers = {}
    hooks = []
    for name, module in target_modules.items():
        obs = LayerObserver(name=name, clip_quantile=clip_quantile)
        hook = module.register_forward_hook(obs.hook_fn)
        observers[name] = obs
        hooks.append(hook)
    return observers, hooks


def remove_all_hooks(hooks: list):
    """清理並卸載所有 Forward Hooks，還原模型乾淨狀態。"""
    for hook in hooks:
        hook.remove()
    hooks.clear()


def clean_memory(device: str):
    """即時釋放當前硬體顯存與執行垃圾回收。"""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    gc.collect()


def run_forward_batches(model: torch.nn.Module, samples: list[torch.Tensor], device: str):
    """遍歷所有校準樣本進行單次前向傳播，激發並收集特徵張量。"""
    progress = tqdm(samples, desc="🚀 執行激活值校準前向傳播")
    for chunk in progress:
        input_ids = chunk.to(device)
        with torch.no_grad():
            model(input_ids=input_ids)
        del input_ids
        clean_memory(device)


def collect_and_save_stats(
    observers: dict[str, LayerObserver],
    model_id: str,
    output_path: str,
    elapsed_time: float,
) -> dict:
    """彙整所有層之統計量與 Scale，並持久化保存至 .pt 檔案。"""
    all_layers = {name: obs.to_dict() for name, obs in observers.items()}
    result_data = {
        "model_id": model_id,
        "num_layers": len(all_layers),
        "calibration_time_sec": elapsed_time,
        "layers": all_layers,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(result_data, output_path)
    print(f"✅ 校準完成！已成功儲存 {len(all_layers)} 層統計資料至: {output_path}")
    return result_data


def calibrate_model(
    model: torch.nn.Module,
    tokenizer,
    device: str,
    model_id: str = "custom_model",
    n_samples: int = 128,
    seq_len: int = 2048,
    output_path: str | None = None,
    clip_quantile: float = 0.9999,
) -> dict:
    """
    統一校準主函式：串接資料集準備、Hook 掛載、前向傳播與結果保存。
    """
    output_file = output_path or f"./calibration_results/calib_{model_id.replace('/', '_')}.pt"

    # 1. 準備校準資料
    samples = get_calib_dataset(tokenizer, n_samples=n_samples, seq_len=seq_len)

    # 2. 探測目標層並掛載 Hook
    target_modules = find_target_linear_modules(model)
    print(f"🎯 成功定位 {len(target_modules)} 個核心 Linear 模組 (Q/K/V/O/Gate/Up/Down)")
    observers, hooks = attach_observers(target_modules, clip_quantile=clip_quantile)

    # 3. 執行前向傳播
    start_time = time.time()
    try:
        run_forward_batches(model, samples, device)
    finally:
        remove_all_hooks(hooks)

    elapsed_time = time.time() - start_time

    # 4. 打包儲存結果
    return collect_and_save_stats(observers, model_id, output_file, elapsed_time)
