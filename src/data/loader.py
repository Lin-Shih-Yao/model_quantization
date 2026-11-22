"""
Data Loader & Calibration Dataset Utilities (Module 1)
Extracts multi-domain calibration tokens (WikiText-2, C4, HumanEval, GSM8K).
"""

import random
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


# ==============================================================================
# Multi-domain Dataset Registry (Table-Driven Configuration)
# ==============================================================================
DATASET_REGISTRY = {
    "wikitext2": {
        "sources": [
            ("wikitext", "wikitext-2-raw-v1", "train", {}),
            ("Salesforce/wikitext", "wikitext-2-raw-v1", "train", {}),
        ],
        "extractor": lambda x: x.get("text", ""),
    },
    "c4": {
        "sources": [
            ("allenai/c4", "en", "train", {"streaming": True}),
        ],
        "extractor": lambda x: x.get("text", ""),
    },
    "humaneval": {
        "sources": [
            ("openai_humaneval", None, "test", {}),
            ("openai/openai_humaneval", None, "test", {}),
        ],
        "extractor": lambda x: x.get("prompt", ""),
    },
    "gsm8k": {
        "sources": [
            ("openai/gsm8k", "main", "train", {}),
            ("gsm8k", "main", "train", {}),
        ],
        "extractor": lambda x: f"{x.get('question', '').strip()}\n{x.get('answer', '').strip()}" if (x.get("question") or x.get("answer")) else "",
    },
}


def _fetch_dataset(dataset_name: str):
    """Fetches dataset by walking through configured primary and fallback sources."""
    for path, name, split, kwargs in DATASET_REGISTRY[dataset_name]["sources"]:
        try:
            return load_dataset(path, name, split=split, **kwargs)
        except Exception:
            continue
    raise RuntimeError(f"Failed to load dataset '{dataset_name}' from all configured sources.")


def _harvest_token_chunks(
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    extractor,
    n_samples: int,
    seq_len: int,
) -> list[torch.Tensor]:
    """Continuous token streaming buffer that yields uniform [1, seq_len] chunks without zero-padding."""
    samples: list[torch.Tensor] = []
    token_buffer: list[int] = []
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

        token_buffer.extend(tokenizer.encode(text, add_special_tokens=False))

        while len(token_buffer) >= seq_len and len(samples) < n_samples:
            chunk = torch.tensor([token_buffer[:seq_len]], dtype=torch.long)
            samples.append(chunk)
            token_buffer = token_buffer[seq_len:]

    if len(samples) < n_samples:
        raise RuntimeError(f"Insufficient samples harvested: got {len(samples)}/{n_samples}.")

    return samples


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
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name} (Supported: {list(DATASET_REGISTRY.keys())})")

    random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n📦 [Data Engine] Loading dataset: {dataset_name.upper()} (Target: {n_samples} samples, seq_len={seq_len}) ...")
    dataset = _fetch_dataset(dataset_name)
    samples = _harvest_token_chunks(
        dataset=dataset,
        tokenizer=tokenizer,
        extractor=DATASET_REGISTRY[dataset_name]["extractor"],
        n_samples=n_samples,
        seq_len=seq_len,
    )

    return torch.cat(samples, dim=0) if as_tensor else samples
