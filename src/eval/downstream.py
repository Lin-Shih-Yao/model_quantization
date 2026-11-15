"""
Downstream Task Evaluation & Raw Inference Persistence Engine.

Loads one (model, quant_method) at a time, runs registered downstream tasks
(src/eval/tasks/), saves raw outputs to raw/*.jsonl with automatic resume,
and logs aggregated metrics to summary.csv.
Supports CPU-only re-scoring (--score_only).
"""

import argparse
import csv
import gc
import json
import os
import time
import torch
from tqdm import tqdm

from src.eval.tasks import get_task_module
from src.models import (
    generate_response,
    get_optimal_device,
    load_pure_text_model_and_tokenizer,
    resolve_model_path,
)
from src.quantization import FakeQuantizer, quantize_model_w8a8, quantize_model_w8a16


def _model_tag(model_id: str) -> str:
    return os.path.basename(model_id.rstrip("/")).replace("/", "_")


def _reclaim_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    gc.collect()


def apply_quantization(model, quant_method: str, calib_file: str | None = None):
    """Applies the requested quantization method to a loaded model."""
    method = quant_method.lower()
    if method in ("none", "bf16", "fp16"):
        return model
    if method == "w8a16":
        return quantize_model_w8a16(model)
    if method == "w8a8":
        return quantize_model_w8a8(model, calib_file=calib_file)
    if method in ("fake_w16a8", "fake_w16a4"):
        if not calib_file or not os.path.exists(calib_file):
            raise ValueError(f"'{method}' requires a valid --calib_file (.json or .pt).")
        bits = 8 if method == "fake_w16a8" else 4
        FakeQuantizer().attach(model, calib_file, bits=bits)
        return model
    raise ValueError(f"Unsupported quant_method: {quant_method}")


def _load_completed_ids(jsonl_path: str) -> set[str]:
    if not os.path.exists(jsonl_path):
        return set()
    completed: set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    completed.add(str(json.loads(line)["id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def run_and_save_inference(
    model,
    tokenizer,
    task_name: str,
    samples: list[dict],
    out_jsonl: str,
    device: str,
    max_new_tokens: int = 512,
) -> None:
    """Runs generation per pending sample and flushes immediately to JSONL."""
    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
    completed_ids = _load_completed_ids(out_jsonl)
    pending = [s for s in samples if str(s["id"]) not in completed_ids]
    if not pending:
        print(f"⏭️  [Skip] All {len(samples)} samples already saved in {out_jsonl}")
        return

    print(f"🚀 [Inference] {task_name.upper()} | {len(pending)}/{len(samples)} pending -> {out_jsonl}")
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for sample in tqdm(pending, desc=task_name):
            res = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=sample["prompt"],
                device=device,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            record = {
                "id": sample["id"],
                "task": task_name,
                "prompt": sample["prompt"],
                "raw_prompt": sample.get("raw_prompt"),
                "ground_truth": sample["ground_truth"],
                "entry_point": sample.get("entry_point"),
                "prediction": res["response"],
                "input_tokens": res["input_tokens"],
                "output_tokens": res["output_tokens"],
                "latency_sec": round(res["latency_sec"], 4),
                "tokens_per_sec": round(res["tokens_per_sec"], 2),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def _repetition_rate(text: str, n: int = 4) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def score_jsonl(jsonl_path: str, task_name: str) -> dict:
    """Scores saved raw JSONL predictions on CPU using the registered task scorer."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        return {"num_samples": 0}

    task_mod = get_task_module(task_name)
    task_scores = task_mod.score_records(records)
    return {
        **task_scores,
        "num_samples": len(records),
        "avg_output_tokens": round(sum(r.get("output_tokens", 0) for r in records) / len(records), 2),
        "avg_tokens_per_sec": round(sum(r.get("tokens_per_sec", 0.0) for r in records) / len(records), 2),
        "repetition_rate": round(sum(_repetition_rate(r.get("prediction", "")) for r in records) / len(records), 4),
    }


def _append_summary_row(summary_csv: str, row: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(summary_csv)), exist_ok=True)
    fieldnames = [
        "timestamp", "model", "quant_method", "task", "primary_metric",
        "primary_value", "secondary_metrics", "num_samples",
        "avg_output_tokens", "avg_tokens_per_sec", "repetition_rate",
    ]
    is_empty = not os.path.exists(summary_csv) or os.path.getsize(summary_csv) == 0
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_empty:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_evaluation_suite(
    model_id: str,
    quant_methods: list[str],
    tasks: list[str],
    output_dir: str = "results",
    calib_file: str | None = None,
    limit: int | None = 200,
    max_new_tokens: int = 512,
    dtype: str = "bfloat16",
) -> None:
    """Executes downstream generation tasks per (model, quant_method)."""
    device = get_optimal_device()
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    resolved_path = resolve_model_path(model_id)
    model_tag = _model_tag(model_id)
    summary_csv = os.path.join(output_dir, "summary.csv")
    raw_dir = os.path.join(output_dir, "raw")

    for quant_method in quant_methods:
        print("\n" + "=" * 70)
        print(f"🧪 Evaluating: {model_tag} | Quant: {quant_method} | Device: {device}")
        print("=" * 70)

        model, tokenizer = load_pure_text_model_and_tokenizer(resolved_path, device, torch_dtype)
        model = apply_quantization(model, quant_method, calib_file=calib_file)

        for task in tasks:
            samples = get_task_module(task).load_samples(limit=limit)
            out_jsonl = os.path.join(raw_dir, f"{model_tag}__{quant_method}__{task}.jsonl")
            run_and_save_inference(model, tokenizer, task, samples, out_jsonl, device, max_new_tokens)
            metrics = score_jsonl(out_jsonl, task)
            _log_task_summary(summary_csv, model_tag, quant_method, task, metrics)
            print(f"📊 [{task.upper()}] {metrics}")

        del model
        _reclaim_memory(device)


def _log_task_summary(summary_csv: str, model_tag: str, quant_method: str, task: str, metrics: dict) -> None:
    """Logs scored generation task metrics to summary.csv."""
    _append_summary_row(summary_csv, {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_tag,
        "quant_method": quant_method,
        "task": task,
        "primary_metric": metrics.get("primary_metric", ""),
        "primary_value": metrics.get("primary_value", ""),
        "secondary_metrics": json.dumps(metrics.get("secondary_metrics", {})),
        "num_samples": metrics.get("num_samples", 0),
        "avg_output_tokens": metrics.get("avg_output_tokens", ""),
        "avg_tokens_per_sec": metrics.get("avg_tokens_per_sec", ""),
        "repetition_rate": metrics.get("repetition_rate", ""),
    })


def rescore_all_jsonl(output_dir: str = "results") -> None:
    """Re-scores all saved raw/*.jsonl files on CPU and writes summary_rescored.csv."""
    raw_dir = os.path.join(output_dir, "raw")
    if not os.path.isdir(raw_dir):
        print(f"⚠️ No raw directory found at: {raw_dir}")
        return

    rescored_csv = os.path.join(output_dir, "summary_rescored.csv")
    if os.path.exists(rescored_csv):
        os.remove(rescored_csv)

    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".jsonl") or len(fname[:-6].split("__")) != 3:
            continue
        model_tag, quant_method, task = fname[:-6].split("__")
        metrics = score_jsonl(os.path.join(raw_dir, fname), task)
        _log_task_summary(rescored_csv, model_tag, quant_method, task, metrics)
        print(f"✅ Rescored {fname}: {metrics}")


def main():
    parser = argparse.ArgumentParser(description="Downstream Task Evaluation & Raw Persistence")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--quant_methods", nargs="+", default=["none", "w8a16"])
    parser.add_argument("--calib_file", type=str, default=None)
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "humaneval"])
    parser.add_argument("--limit", type=int, default=200, help="Max samples for generation tasks (0 = all)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--score_only", action="store_true", help="CPU-only re-scoring of raw/*.jsonl")
    args = parser.parse_args()

    if args.score_only:
        rescore_all_jsonl(args.output_dir)
    else:
        run_evaluation_suite(
            model_id=args.model_id,
            quant_methods=args.quant_methods,
            tasks=args.tasks,
            output_dir=args.output_dir,
            calib_file=args.calib_file,
            limit=None if args.limit == 0 else args.limit,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
        )


if __name__ == "__main__":
    main()
