"""
Data Loader & Calibration Dataset Utilities (Module 1)
Extracts multi-domain calibration tokens (WikiText-2, C4, HumanEval, GSM8K).
"""

import random
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def get_calibration_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str = "wikitext2",
    n_samples: int = 128,
    seq_len: int = 2048,
    as_tensor: bool = True,
    seed: int = 42,
) -> torch.Tensor | list[torch.Tensor]:
    """
    從指定領域資料集抽樣並編碼為固定形狀 (n_samples, seq_len) 的校準張量或張量列表。

    Args:
        tokenizer: 預訓練分詞器
        dataset_name: 資料集名稱 ("wikitext2", "c4", "humaneval", "gsm8k")
        n_samples: 取樣樣本數量 (e.g. 16, 128, 512)
        seq_len: 每個樣本之序列長度 (預設 2048)
        as_tensor: 若為 True 則回傳形狀為 [n_samples, seq_len] 之單一張量，
                   若為 False 則回傳包含 n_samples 個 [1, seq_len] 張量之列表。
        seed: 隨機種子

    Returns:
        torch.Tensor | list[torch.Tensor]
    """
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n📦 [數據模組] 載入資料集: {dataset_name.upper()} (目標 {n_samples} 筆, seq_len={seq_len}) ...")
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
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

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
        if cur_len < seq_len and pad_id is not None:
            pad = torch.full((1, seq_len - cur_len), pad_id, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, pad], dim=1)
        elif cur_len < seq_len:
            continue  # 若無可用 pad_id 且未達 seq_len 則跳過

        samples.append(input_ids)
        if len(samples) >= n_samples:
            break

    if len(samples) < n_samples:
        raise RuntimeError(f"資料集 '{dataset_name}' 樣本不足，僅收集到 {len(samples)}/{n_samples} 筆。")

    if as_tensor:
        calib_tensor = torch.cat(samples, dim=0)
        return calib_tensor
    return samples


def get_calib_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str = "wikitext2",
    n_samples: int = 128,
    seq_len: int = 2048,
    seed: int = 42,
    **kwargs,
) -> list[torch.Tensor]:
    """
    取得校準樣本列表 (向後相容介面，回傳 list[torch.Tensor]，每個為 [1, seq_len])。
    """
    return get_calibration_dataloader(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        n_samples=n_samples,
        seq_len=seq_len,
        as_tensor=False,
        seed=seed,
    )
