"""
Data Loader & Calibration Dataset Utilities (Module 1)
Extracts multi-domain calibration tokens (WikiText-2, C4, HumanEval, GSM8K).
"""

import random
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


# ==============================================================================
# Multi-domain Text Extractor Registry (Strategy Pattern)
# ==============================================================================
TEXT_EXTRACTORS = {
    "wikitext2": lambda x: x.get("text", ""),
    "c4": lambda x: x.get("text", ""),
    "humaneval": lambda x: x.get("prompt", ""),
    "gsm8k": lambda x: f"{x.get('question', '').strip()}\n{x.get('answer', '').strip()}" if (x.get("question") or x.get("answer")) else "",
}


def get_calibration_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str = "wikitext2",
    n_samples: int = 128,
    seq_len: int = 2048,
    as_tensor: bool = True,
    seed: int = 42,
) -> torch.Tensor | list[torch.Tensor]:
    """
    Samples and tokenizes multi-domain calibration tokens into uniform tensors of shape [n_samples, seq_len].

    Args:
        tokenizer: Pretrained Hugging Face tokenizer.
        dataset_name: Target domain dataset ("wikitext2", "c4", "humaneval", "gsm8k").
        n_samples: Number of sample windows to harvest (e.g., 16, 128, 512).
        seq_len: Sequence window length for each sample (default: 2048).
        as_tensor: If True, returns a concatenated 2D tensor [n_samples, seq_len].
                   If False, returns a list of 2D tensors each of shape [1, seq_len].
        seed: Random generator seed for reproducibility.

    Returns:
        torch.Tensor | list[torch.Tensor]: Harvested calibration token sequences.
    """
    if dataset_name not in TEXT_EXTRACTORS:
        raise ValueError(f"Unsupported dataset: {dataset_name} (Supported: {list(TEXT_EXTRACTORS.keys())})")

    random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n📦 [Data Engine] Loading dataset: {dataset_name.upper()} (Target: {n_samples} samples, seq_len={seq_len}) ...")
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
        try:
            dataset = load_dataset("openai/gsm8k", "main", split="train")
        except Exception:
            dataset = load_dataset("gsm8k", "main", split="train")

    samples: list[torch.Tensor] = []
    token_buffer: list[int] = []
    extractor = TEXT_EXTRACTORS[dataset_name]
    dataset_iter = iter(dataset)
    max_retries = n_samples * 250
    retries = 0

    while len(samples) < n_samples and retries < max_retries:
        try:
            item = next(dataset_iter)
        except StopIteration:
            dataset_iter = iter(dataset)
            item = next(dataset_iter)
        retries += 1

        text = extractor(item)
        if not text or not text.strip():
            continue

        item_tokens = tokenizer.encode(text, add_special_tokens=False)
        token_buffer.extend(item_tokens)

        while len(token_buffer) >= seq_len and len(samples) < n_samples:
            chunk = torch.tensor([token_buffer[:seq_len]], dtype=torch.long)
            samples.append(chunk)
            token_buffer = token_buffer[seq_len:]

    if len(samples) < n_samples:
        raise RuntimeError(f"Insufficient samples harvested from '{dataset_name}': got {len(samples)}/{n_samples}.")

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
    Retrieves calibration sample list (backward compatibility wrapper returning list[torch.Tensor], each [1, seq_len]).
    """
    return get_calibration_dataloader(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        n_samples=n_samples,
        seq_len=seq_len,
        as_tensor=False,
        seed=seed,
    )
