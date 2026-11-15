"""
GSM8K Math Reasoning Task (Loading & Multi-Metric Scoring).
"""

import random
import re
from datasets import load_dataset


def load_samples(limit: int | None = 200, seed: int = 42) -> list[dict]:
    """Loads GSM8K test questions with optional deterministic subsampling."""
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception:
        ds = load_dataset("gsm8k", "main", split="test")

    indices = list(range(len(ds)))
    if limit and limit < len(indices):
        rng = random.Random(seed)
        indices = sorted(rng.sample(indices, limit))

    return [
        {
            "id": f"gsm8k_{idx}",
            "prompt": (
                "Solve the following math problem step by step. "
                "At the end, output your final numerical answer after '#### '.\n\n"
                f"Question: {ds[idx]['question'].strip()}"
            ),
            "ground_truth": ds[idx]["answer"].strip(),
        }
        for idx in indices
    ]


def _extract_number(text: str, strict: bool) -> str | None:
    """Extracts final numerical answer (strict '####' format or last number)."""
    if strict:
        match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
        return match.group(1).replace(",", "") if match else None
    matches = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return matches[-1].replace(",", "") if matches else None


def score_records(records: list[dict]) -> dict:
    """Computes strict_match and flexible_extract accuracy over GSM8K records."""
    strict_hits = 0
    flex_hits = 0
    for r in records:
        gold = _extract_number(r["ground_truth"], strict=True)
        if gold is None:
            continue
        if _extract_number(r["prediction"], strict=True) == gold:
            strict_hits += 1
        if _extract_number(r["prediction"], strict=False) == gold:
            flex_hits += 1

    total = len(records)
    return {
        "primary_metric": "flexible_extract",
        "primary_value": round(flex_hits / total, 4),
        "secondary_metrics": {"strict_match": round(strict_hits / total, 4)},
    }

