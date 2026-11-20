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


def analyze_records(records: list[dict]) -> dict:
    """Categorizes GSM8K predictions into success and failure modes."""
    counts = {
        "correct": 0,
        "strict_correct": 0,
        "wrong_answer": 0,
        "format_lost": 0,
        "no_number": 0,
    }
    for r in records:
        gold = _extract_number(r["ground_truth"], strict=True)
        if gold is None:
            continue
        strict_pred = _extract_number(r["prediction"], strict=True)
        flex_pred = _extract_number(r["prediction"], strict=False)

        if strict_pred == gold:
            counts["strict_correct"] += 1

        if flex_pred == gold:
            counts["correct"] += 1
        elif flex_pred is None:
            counts["no_number"] += 1
        elif strict_pred is None:
            counts["format_lost"] += 1
        else:
            counts["wrong_answer"] += 1

    total = max(len(records), 1)
    rates = {k: round(v / total, 4) for k, v in counts.items()}
    return {"total": len(records), "counts": counts, "rates": rates}


def score_records(records: list[dict]) -> dict:
    """Computes strict_match, flexible_extract, and failure breakdown over GSM8K records."""
    analysis = analyze_records(records)
    rates = analysis["rates"]
    return {
        "primary_metric": "flexible_extract",
        "primary_value": rates["correct"],
        "secondary_metrics": {
            "strict_match": rates["strict_correct"],
            "wrong_answer_rate": rates["wrong_answer"],
            "format_lost_rate": rates["format_lost"],
            "no_number_rate": rates["no_number"],
        },
    }

