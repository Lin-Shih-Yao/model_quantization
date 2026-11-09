"""
Perplexity (PPL) Evaluation Module for Causal Language Models.
Directly implements Hugging Face's official Strided Sliding Window benchmark:
Reference: https://huggingface.co/docs/transformers/perplexity

Evaluates models on standard WikiText-2 test set with hardware acceleration (CUDA / MPS),
strict Context Label Masking (-100), and immediate per-step cache memory cleanup.
"""
import argparse
import csv
import gc
import math
import os
import time
import psutil
import torch
from datasets import load_dataset
from tqdm import tqdm

# Prevent Apple Silicon MPS from hoarding cached unified memory
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from src.models import (
    load_pure_text_model_and_tokenizer,
    get_optimal_device,
    resolve_model_path,
)

def evaluate_ppl(
    model,
    tokenizer,
    device: str = "mps",
    max_length: int = 2048,
    stride: int = 512,
    max_steps: int | None = None,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    dataset_split: str = "test",
) -> dict:
    """
    Computes Perplexity (PPL) for Causal LMs following Hugging Face's official strided sliding window algorithm.

    Operational Mechanism:
      1. Window length is max_length (e.g., 2048), sliding by stride (e.g., 512).
      2. Historical prefix context (first max_length - stride tokens) is masked with -100 in labels.
      3. Only newly shifted tokens (trg_len, typically 512) incur CrossEntropyLoss computation.
      4. Eliminates non-overlapping chunk boundary distortion and context starvation.

    Returns:
        dict: {
            "ppl": float,
            "total_tokens": int,
            "elapsed_sec": float,
            "tokens_per_sec": float,
            "total_steps": int,
        }
    """
    # 1. Load benchmark dataset
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    full_text = "\n\n".join([t for t in dataset["text"] if t.strip()])

    # 2. Tokenize full corpus
    encodings = tokenizer(full_text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    print(f"📖 Dataset: {dataset_name}/{dataset_config} ({dataset_split})")
    print(f"  • Total Corpus Tokens : {seq_len:,}")
    print(f"  • Context Window      : {max_length}")
    print(f"  • Sliding Stride      : {stride}")

    nlls = []
    prev_end_loc = 0
    step_count = 0
    start_time = time.time()

    # Calculate total steps for progress bar
    total_possible_steps = (seq_len + stride - 1) // stride
    total_steps = min(total_possible_steps, max_steps) if max_steps else total_possible_steps

    progress_bar = tqdm(total=total_steps, desc="Strided Sliding-Window PPL")

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # Newly scored target tokens

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()

        # Context label masking: history tokens masked to -100
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=target_ids)
            # outputs.loss averages over labels != -100
            neg_log_likelihood = outputs.loss.item() * trg_len
            nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        step_count += 1
        progress_bar.update(1)

        # Proactive memory reclamation per step
        del outputs, input_ids, target_ids
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # Safety Circuit Breaker: prevent host OOM panic if RAM usage exceeds 85%
        mem_percent = psutil.virtual_memory().percent
        if mem_percent > 85.0:
            print(f"\n⚠️ [Safety Circuit Breaker] System RAM reached {mem_percent:.1f}% (>85%). Aborting safely to protect host!")
            break

        if end_loc == seq_len or (max_steps and step_count >= max_steps):
            break

    progress_bar.close()

    total_evaluated_tokens = prev_end_loc
    total_nll = sum(nlls)
    ppl = math.exp(total_nll / total_evaluated_tokens) if total_evaluated_tokens > 0 else float("nan")
    elapsed_sec = time.time() - start_time

    return {
        "ppl": ppl,
        "total_tokens": total_evaluated_tokens,
        "elapsed_sec": elapsed_sec,
        "tokens_per_sec": total_evaluated_tokens / elapsed_sec if elapsed_sec > 0 else 0,
        "total_steps": step_count,
    }


def log_result_to_csv(
    log_path: str,
    node_name: str,
    model_id: str,
    calib_dataset: str,
    n_samples: int | str,
    bit_str: str,
    ppl_score: float,
):
    """Safely appends single experiment result to CSV file with automatic header generation."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    is_empty = not os.path.exists(log_path) or os.path.getsize(log_path) == 0

    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_empty:
            writer.writerow(["Node", "Model", "Calib_Dataset", "N_Samples", "A_Bit", "PPL"])
        writer.writerow([node_name, model_id, calib_dataset, n_samples, bit_str, f"{ppl_score:.4f}"])
    print(f"📝 [Result Logged] {node_name} | {model_id} | {calib_dataset} | N={n_samples} | {bit_str} -> PPL: {ppl_score:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Official Hugging Face Strided Sliding Window Perplexity (PPL) Benchmark"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3.5-2B",
        help="HuggingFace model ID or local directory path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext-2-raw-v1",
        choices=["wikitext-2-raw-v1"],
        help="Benchmark dataset to evaluate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Context window length (default: 2048)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Sliding window stride size (default: 512, official standard)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum sliding steps to evaluate (default: None for full dataset, or e.g. 40/100 for fast screening)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model precision dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--quant_method",
        type=str,
        default="none",
        choices=["none", "w8a16", "w8a8"],
        help="Quantization method to apply before evaluation (default: none for unquantized model)",
    )
    parser.add_argument(
        "--calib_file",
        type=str,
        default=None,
        help="Path to calibration .pt file (used by W8A8 for activation clipping)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_optimal_device()
    resolved_path = resolve_model_path(args.model_id)

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print("=" * 68)
    print("🧪 Official Hugging Face Strided Sliding Window Perplexity Benchmark")
    print(f"  • Target Model : {resolved_path}")
    print(f"  • Quant Method : {args.quant_method.upper()}")
    if args.calib_file:
        print(f"  • Calib File   : {args.calib_file}")
    print(f"  • Device GPU   : {device.upper()} (Apple Silicon MPS / CUDA)")
    print(f"  • Precision    : {args.dtype}")
    print(f"  • Dataset      : {args.dataset}")
    print(f"  • Context Len  : {args.max_length} tokens")
    print(f"  • Stride Step  : {args.stride} tokens (with -100 Context Masking)")
    if args.max_steps:
        print(f"  • Max Steps    : {args.max_steps} steps (Fast Mode)")
    else:
        print(f"  • Max Steps    : Full Dataset Evaluation")
    print("=" * 68)

    print("\n⏳ Loading model architecture and weights to hardware device...")
    load_start = time.time()
    model, tokenizer = load_pure_text_model_and_tokenizer(resolved_path, device, torch_dtype)
    load_time = time.time() - load_start

    ram_gb = psutil.Process().memory_info().rss / (1024 ** 3)
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"✅ Model and Tokenizer successfully loaded!")
    print(f"  • Loading Latency  : {load_time:.2f} s")
    print(f"  • Text Parameters  : {num_params:.2f} B ({num_params * 1000:.0f} M)")
    print(f"  • Physical Memory  : ~{ram_gb:.2f} GB RAM")

    # Seamless quantization layer replacement if specified
    if args.quant_method == "w8a16":
        from src.quantization import quantize_model_w8a16
        print("\n⚙️ Applying physical W8A16 INT8 linear replacement...")
        model = quantize_model_w8a16(model)
    elif args.quant_method == "w8a8":
        from src.quantization import quantize_model_w8a8
        print("\n⚙️ Applying physical W8A8 INT8 linear replacement (with calibrated clipping)...")
        model = quantize_model_w8a8(model, calib_file=args.calib_file)

    # Execute official standard Strided Sliding Window PPL evaluation
    results = evaluate_ppl(
        model,
        tokenizer,
        device=device,
        max_length=args.max_length,
        stride=args.stride,
        max_steps=args.max_steps,
    )

    print("\n" + "=" * 68)
    if args.quant_method != "none":
        print(f"📊 Official WikiText-2 Benchmark Results [{args.quant_method.upper()} Quantization]:")
    else:
        print("📊 Official WikiText-2 Benchmark Results (FP16/BF16 Baseline):")
    print("-" * 68)
    print(f"  🏆 WikiText-2 PPL (Perplexity) : {results['ppl']:.4f} (Lower is better)")
    print(f"  ⚡ Total Elapsed Time          : {results['elapsed_sec']:.2f} s")
    print(f"  🚀 Processing Throughput       : {results['tokens_per_sec']:.2f} tokens/s")
    print(f"  📝 Evaluated Corpus Tokens     : {results['total_tokens']:,} tokens")
    print(f"  🔄 Sliding Steps Executed      : {results['total_steps']} steps")
    print("=" * 68)


if __name__ == "__main__":
    main()
