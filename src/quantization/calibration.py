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
    """Scans and extracts target Linear layers belonging to Attention and MLP blocks."""
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
    """Attaches LayerObserver forward hooks onto target Linear layers."""
    observers = {}
    hooks = []
    for name, module in target_modules.items():
        obs = LayerObserver(name=name, clip_quantile=clip_quantile)
        hook = module.register_forward_hook(obs.hook_fn)
        observers[name] = obs
        hooks.append(hook)
    return observers, hooks


def remove_all_hooks(hooks: list):
    """Safely removes all registered forward hooks, restoring clean model state."""
    for hook in hooks:
        hook.remove()
    hooks.clear()


def clean_memory(device: str):
    """Immediately reclaims accelerator memory and executes garbage collection."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    gc.collect()


def run_forward_batches(model: torch.nn.Module, samples: list[torch.Tensor], device: str):
    """Iterates through calibration samples for a single forward pass, recording activations."""
    progress = tqdm(samples, desc="🚀 Executing Calibration Forward Pass")
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
    """Unified forward calibration harvester: attaches observers, processes samples, and returns stats dict."""
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
    """Recursively converts PyTorch Tensors into native Python data structures for JSON serialization."""
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
    """Serializes layer extrema and scale statistics to JSON with multi-node crash resilience."""
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

    print(f"💾 [Calibration Engine] Layer statistics saved to: {full_path}")
    return full_path


def collect_and_save_stats(
    observers: dict[str, LayerObserver],
    model_id: str,
    output_path: str,
    elapsed_time: float,
) -> dict:
    """Aggregates layer statistics and scales into a persistent PyTorch .pt file."""
    all_layers = {name: obs.to_dict() for name, obs in observers.items()}
    result_data = {
        "model_id": model_id,
        "num_layers": len(all_layers),
        "calibration_time_sec": elapsed_time,
        "layers": all_layers,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(result_data, output_path)
    print(f"✅ Calibration complete! Saved {len(all_layers)} layer statistics to: {output_path}")
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
    Unified calibration orchestrator: data preparation, hook registration, forward execution, and artifact saving.
    """
    clean_model_name = model_id.replace('/', '_')
    output_file = output_path or f"./calibration_results/calib_{clean_model_name}_{dataset_name}.pt"

    # 1. Prepare calibration samples
    samples = get_calib_dataset(tokenizer, dataset_name=dataset_name, n_samples=n_samples, seq_len=seq_len)

    # 2. Probe target layers and attach observers
    target_modules = find_target_linear_modules(model)
    print(f"🎯 Successfully localized {len(target_modules)} core Linear modules (Q/K/V/O/Gate/Up/Down)")
    observers, hooks = attach_observers(target_modules, clip_quantile=clip_quantile)

    # 3. Execute forward pass
    start_time = time.time()
    try:
        run_forward_batches(model, samples, device)
    finally:
        remove_all_hooks(hooks)

    elapsed_time = time.time() - start_time

    # 4. Save serialized artifact
    return collect_and_save_stats(observers, model_id, output_file, elapsed_time)
