"""
Downstream Evaluation Analyzer & Visualizer.

Aggregates raw JSONL predictions from `results/raw/`, invokes task-specific
`analyze_records` handlers (`gsm8k`, `humaneval`), computes token/latency/repetition
telemetry, exports `results/downstream_analysis.csv`, and generates comparison plots
under `results/figures/`.
"""

import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.eval.downstream import _repetition_rate
from src.eval.tasks import get_task_module

QUANT_ORDER = ["none", "w8a16", "fake_w16a8", "fake_w16a4"]
QUANT_LABELS = {
    "none": "BF16",
    "w8a16": "W8A16",
    "fake_w16a8": "W16A8 (Fake)",
    "fake_w16a4": "W16A4 (Fake)",
}


def _quant_sort_key(quant: str) -> tuple[int, str]:
    idx = QUANT_ORDER.index(quant) if quant in QUANT_ORDER else len(QUANT_ORDER)
    return (idx, quant)


def analyze_jsonl_file(jsonl_path: str, task_name: str, max_tokens_cap: int = 512) -> dict:
    """Computes task-specific failure modes and generation telemetry for one JSONL file."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        return {"num_samples": 0}

    task_mod = get_task_module(task_name)
    score_info = task_mod.score_records(records)
    analysis = task_mod.analyze_records(records)

    n = len(records)
    avg_out = round(sum(r.get("output_tokens", 0) for r in records) / n, 2)
    max_hit = round(sum(1 for r in records if r.get("output_tokens", 0) >= max_tokens_cap) / n, 4)
    avg_lat = round(sum(r.get("latency_sec", 0.0) for r in records) / n, 3)
    avg_tps = round(sum(r.get("tokens_per_sec", 0.0) for r in records) / n, 2)
    rep_rate = round(sum(_repetition_rate(r.get("prediction", "")) for r in records) / n, 4)

    return {
        "num_samples": n,
        "primary_metric": score_info["primary_metric"],
        "primary_value": score_info["primary_value"],
        "avg_output_tokens": avg_out,
        "max_token_hit_rate": max_hit,
        "avg_latency_sec": avg_lat,
        "avg_tokens_per_sec": avg_tps,
        "repetition_rate": rep_rate,
        "rates": analysis["rates"],
        "counts": analysis["counts"],
    }


def collect_results(raw_dir: str, max_tokens_cap: int = 512) -> list[dict]:
    """Scans raw_dir for `<model>__<quant>__<task>.jsonl` files and analyzes each."""
    rows = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".jsonl") or len(fname[:-6].split("__")) != 3:
            continue
        model_tag, quant_method, task = fname[:-6].split("__")
        metrics = analyze_jsonl_file(os.path.join(raw_dir, fname), task, max_tokens_cap)
        rows.append(
            {
                "model": model_tag,
                "quant_method": quant_method,
                "task": task,
                **metrics,
            }
        )

    rows.sort(key=lambda r: (r["task"], r["model"], _quant_sort_key(r["quant_method"])))
    return rows


def export_csv(rows: list[dict], csv_path: str) -> None:
    """Writes aggregated downstream analysis rows to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = [
        "model",
        "quant_method",
        "task",
        "num_samples",
        "primary_metric",
        "primary_value",
        "avg_output_tokens",
        "max_token_hit_rate",
        "avg_latency_sec",
        "avg_tokens_per_sec",
        "repetition_rate",
        "failure_rates_json",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames if k != "failure_rates_json"}
            out["failure_rates_json"] = json.dumps(r.get("rates", {}), ensure_ascii=False)
            writer.writerow(out)


def plot_accuracy_degradation(rows: list[dict], fig_dir: str) -> str:
    """Plots GSM8K accuracy and HumanEval Pass@1 across quantization configurations."""
    tasks = [("gsm8k", "GSM8K (Flexible Extract Accuracy)"), ("humaneval", "HumanEval (Pass@1)")]
    models = sorted({r["model"] for r in rows})
    quants = [q for q in QUANT_ORDER if any(r["quant_method"] == q for r in rows)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (task, title) in zip(axes, tasks):
        for model in models:
            lookup = {
                r["quant_method"]: r["primary_value"] * 100.0
                for r in rows
                if r["task"] == task and r["model"] == model
            }
            xs = list(range(len(quants)))
            ys = [lookup.get(q, float("nan")) for q in quants]
            ax.plot(xs, ys, marker="o", linewidth=2.2, label=model)
            for x, y in zip(xs, ys):
                if y == y:
                    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(quants)))
        ax.set_xticklabels([QUANT_LABELS.get(q, q) for q in quants])
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Score (%)")
    fig.suptitle("Downstream Task Degradation Across Quantization Precisions", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(fig_dir, "downstream_accuracy_degradation.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_failure_modes(rows: list[dict], fig_dir: str) -> str:
    """Plots stacked failure mode distributions per task, model, and precision."""
    task_specs = [
        (
            "gsm8k",
            "GSM8K Outcome Breakdown",
            [
                ("correct", "Correct", "#2ca02c"),
                ("wrong_answer", "Wrong Answer", "#ff7f0e"),
                ("format_lost", "Format Lost", "#9467bd"),
                ("no_number", "No Number", "#d62728"),
            ],
        ),
        (
            "humaneval",
            "HumanEval Execution Breakdown",
            [
                ("pass", "Pass", "#2ca02c"),
                ("assertion_error", "AssertionError", "#ff7f0e"),
                ("runtime_error", "RuntimeError", "#1f77b4"),
                ("syntax_error", "SyntaxError", "#d62728"),
                ("timeout", "Timeout", "#7f7f7f"),
            ],
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, (task, title, categories) in zip(axes, task_specs):
        sub = [r for r in rows if r["task"] == task]
        sub.sort(key=lambda r: (r["model"], _quant_sort_key(r["quant_method"])))
        labels = [
            f"{r['model'].replace('Llama-3.2-', '').replace('-Instruct', '')}\n{QUANT_LABELS.get(r['quant_method'], r['quant_method'])}"
            for r in sub
        ]
        bottoms = [0.0] * len(sub)
        for key, label, color in categories:
            vals = [r["rates"].get(key, 0.0) * 100.0 for r in sub]
            ax.bar(range(len(sub)), vals, bottom=bottoms, label=label, color=color, width=0.62)
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("Share of Samples (%)")
    fig.suptitle("Downstream Failure Mode Taxonomy by Task & Quantization", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(fig_dir, "downstream_failure_modes.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_tokens_and_throughput(rows: list[dict], fig_dir: str) -> str:
    """Plots average generated tokens and decoding throughput across configurations."""
    models = sorted({r["model"] for r in rows})
    quants = [q for q in QUANT_ORDER if any(r["quant_method"] == q for r in rows)]
    tasks = ["gsm8k", "humaneval"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for model in models:
        for task in tasks:
            sub = {
                r["quant_method"]: r
                for r in rows
                if r["model"] == model and r["task"] == task
            }
            xs = list(range(len(quants)))
            out_toks = [sub[q]["avg_output_tokens"] if q in sub else float("nan") for q in quants]
            tps = [sub[q]["avg_tokens_per_sec"] if q in sub else float("nan") for q in quants]
            short_model = model.replace("Llama-3.2-", "Llama-").replace("-Instruct", "")
            style = "-" if task == "gsm8k" else "--"
            axes[0].plot(xs, out_toks, marker="o", linestyle=style, label=f"{short_model} ({task})")
            axes[1].plot(xs, tps, marker="s", linestyle=style, label=f"{short_model} ({task})")

    axes[0].set_title("Average Output Tokens", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Tokens / Sample")
    axes[1].set_title("Generation Throughput", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Throughput (tokens/s)")
    for ax in axes:
        ax.set_xticks(range(len(quants)))
        ax.set_xticklabels([QUANT_LABELS.get(q, q) for q in quants])
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle("Generation Length & Decoding Throughput Telemetry", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(fig_dir, "downstream_tokens_throughput.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def print_summary(rows: list[dict]) -> None:
    """Prints a compact summary table to stdout."""
    header = (
        f"{'Model':<24} {'Quant':<11} {'Task':<10} {'Score':>7} "
        f"{'AvgTok':>7} {'CapHit':>7} {'TPS':>7} {'RepRate':>8}  Failure Rates"
    )
    print(header)
    print("-" * len(header) + "-" * 30)
    for r in rows:
        print(
            f"{r['model']:<24} {r['quant_method']:<11} {r['task']:<10} "
            f"{r['primary_value']*100:>6.2f}% {r['avg_output_tokens']:>7.1f} "
            f"{r['max_token_hit_rate']*100:>6.1f}% {r['avg_tokens_per_sec']:>7.2f} "
            f"{r['repetition_rate']:>8.4f}  {r['rates']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze & visualize raw downstream JSONL results")
    parser.add_argument("--input_dir", type=str, default="results/raw", help="Directory containing raw JSONL files")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to write CSV and figures")
    parser.add_argument("--max_tokens_cap", type=int, default=512, help="Max new tokens cap used during generation")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"⚠️ Input directory not found: {args.input_dir}")
        return

    rows = collect_results(args.input_dir, max_tokens_cap=args.max_tokens_cap)
    if not rows:
        print(f"⚠️ No valid JSONL files found in: {args.input_dir}")
        return

    csv_path = os.path.join(args.output_dir, "downstream_analysis.csv")
    export_csv(rows, csv_path)

    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    figs = [
        plot_accuracy_degradation(rows, fig_dir),
        plot_failure_modes(rows, fig_dir),
        plot_tokens_and_throughput(rows, fig_dir),
    ]

    print_summary(rows)
    print(f"\n✅ Saved analysis CSV: {csv_path}")
    for fp in figs:
        print(f"📊 Saved figure: {fp}")


if __name__ == "__main__":
    main()

