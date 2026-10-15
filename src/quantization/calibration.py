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

from src.data import get_calib_dataset
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
        batch = chunk.to(device)
        with torch.no_grad():
            try:
                model(input_ids=batch)
            except TypeError:
                model(batch)
        del batch
        clean_memory(device)


def collect_calibration_stats(
    model: torch.nn.Module,
    samples: torch.Tensor | list[torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    clip_quantile: float = 0.9999,
) -> dict[str, dict]:
    """通用前向校準統計採集器：掛載 LayerObserver、遍歷輸入樣本並回傳統計字典。"""
    target_modules = find_target_linear_modules(model)
    observers, hooks = attach_observers(target_modules, clip_quantile=clip_quantile)
    model.eval()

    try:
        if isinstance(samples, torch.Tensor):
            batches = [samples[i : i + 1] for i in range(samples.shape[0])]
        else:
            batches = samples
        run_forward_batches(model, batches, device=device)
    finally:
        remove_all_hooks(hooks)

    return {name: obs.to_dict() for name, obs in observers.items()}


def _to_json_serializable(obj):
    """將 Tensor 遞迴轉為純 Python 資料結構以利 JSON 序列化。"""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_serializable(x) for x in obj]
    return obj


def save_calibration_stats_json(
    stats: dict,
    model_name: str,
    dataset_name: str,
    n_samples: int,
    node_name: str = "Node_A",
    custom_dir: str | None = None,
) -> str:
    """將極值與量化統計儲存為 JSON 檔，支援 Colab 雙節點存檔防撞。"""
    import json
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

    serializable_stats = _to_json_serializable(stats)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)

    print(f"💾 [校準模組] 極值統計已儲存至: {full_path}")
    return full_path


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
    dataset_name: str = "wikitext2",
    n_samples: int = 128,
    seq_len: int = 2048,
    output_path: str | None = None,
    clip_quantile: float = 0.9999,
) -> dict:
    """
    統一校準主函式：串接資料集準備、Hook 掛載、前向傳播與結果保存。
    """
    clean_model_name = model_id.replace('/', '_')
    output_file = output_path or f"./calibration_results/calib_{clean_model_name}_{dataset_name}.pt"

    # 1. 準備校準資料
    samples = get_calib_dataset(tokenizer, dataset_name=dataset_name, n_samples=n_samples, seq_len=seq_len)

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
