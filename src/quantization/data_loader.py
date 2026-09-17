"""
Data Loader & Global Configuration Module (Module 1)
Provides CONFIG and multi-domain calibration data loader (WikiText-2, C4, HumanEval, GSM8K).
"""

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

# ==============================================================================
# 全域配置字典 (Global Config)
# ==============================================================================
CONFIG = {
    "node_name": "Node_A",
    "models": [
        "meta-llama/Llama-3.2-1B-Instruct",
    ],
    "datasets": ["wikitext2", "c4", "humaneval", "gsm8k"],
    "N_samples": [16, 128, 512],
    "bits": [8, 4],
    "seq_len": 2048,
}


def get_calibration_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str,
    n_samples: int,
    seq_len: int = 2048,
) -> torch.Tensor:
    """從指定領域資料集抽樣並編碼為固定形狀 (n_samples, seq_len) 的校準張量。"""
    print(f"\n📦 [模組 1] 載入資料集: {dataset_name.upper()} (目標 {n_samples} 筆, seq_len={seq_len}) ...")
    if dataset_name == "wikitext2":
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        except Exception:
            dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    elif dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    elif dataset_name == "humaneval":
        try:
            dataset = load_dataset("openai_humaneval", split="test")
        except Exception:
            dataset = load_dataset("openai/openai_humaneval", split="test")
    elif dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="train")
    else:
        raise ValueError(f"不支援的資料集: {dataset_name}")

    samples: list[torch.Tensor] = []
    min_len = seq_len // 2
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    for item in dataset:
        if dataset_name in ("wikitext2", "c4"):
            text = item.get("text", "")
        elif dataset_name == "humaneval":
            text = item.get("prompt", "")
        elif dataset_name == "gsm8k":
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            text = f"{q}\n{a}" if (q or a) else ""
        else:
            text = ""

        if not text or not text.strip():
            continue

        tokens = tokenizer(text, truncation=True, max_length=seq_len, return_tensors="pt")
        cur_len = tokens.input_ids.shape[1]
        if cur_len < min_len:
            continue

        input_ids = tokens.input_ids
        if cur_len < seq_len:
            pad = torch.full((1, seq_len - cur_len), pad_id, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, pad], dim=1)

        samples.append(input_ids)
        if len(samples) >= n_samples:
            break

    if len(samples) < n_samples:
        raise RuntimeError(f"資料集 '{dataset_name}' 樣本不足，僅收集到 {len(samples)}/{n_samples} 筆。")

    calib_tensor = torch.cat(samples, dim=0)
    return calib_tensor
