"""
Calibration Dataset Module
Prepares standard calibration samples (128 samples x 2048 tokens) from WikiText-2 / C4.
"""

import random
import torch
from datasets import load_dataset


def get_calib_dataset(
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    n_samples: int = 128,
    seq_len: int = 2048,
    seed: int = 42,
) -> list[torch.Tensor]:
    """
    從資料集抓取無重疊的 n_samples 條長度為 seq_len 的 token IDs。
    遵循學術標準 (GPTQ / AWQ / QuaRot)：
    預設使用 wikitext-2-raw-v1 的 train 分割集，與 test 評估集嚴格隔離。

    Returns:
        list[torch.Tensor]: 包含 n_samples 個形狀為 [1, seq_len] 的 Tensor
    """
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"📖 正在載入校準資料集: {dataset_name}/{dataset_config} ({split}) ...")
    try:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    except Exception:
        # Fallback 嘗試 Salesforce/wikitext
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)

    full_text = "\n\n".join([t for t in dataset["text"] if t.strip()])

    print(f"  • Tokenizing 原始文本長度 (總字元數: {len(full_text):,}) ...")
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids[0]
    total_tokens = len(input_ids)
    print(f"  • 文本累計總 Token 數 : {total_tokens:,}")

    max_start = total_tokens - seq_len
    if max_start <= 0:
        raise ValueError(f"資料集過短 (僅 {total_tokens} tokens)，無法切出長度為 {seq_len} 的樣本")

    # 等距切塊，確保樣本覆蓋整個文本分佈且彼此不重疊
    samples = []
    step = max_start // n_samples
    for i in range(n_samples):
        start = i * step
        chunk = input_ids[start : start + seq_len].unsqueeze(0)  # [1, seq_len]
        samples.append(chunk)

    print(f"✅ 校準資料集切片完成！共生成 {len(samples)} 條樣本，每條 {seq_len} tokens (累計 {len(samples)*seq_len:,} tokens)")
    return samples
