"""
Perplexity (PPL) Evaluation Module for Causal Language Models.
Directly implements Hugging Face's official Strided Sliding Window benchmark:
Reference: https://huggingface.co/docs/transformers/perplexity

Evaluates models on standard WikiText-2 test set with Apple Silicon MPS GPU acceleration,
strict Context Label Masking (-100), and immediate per-step MPS memory cleanup.
"""

import argparse
import math
import time
import psutil
import torch
from datasets import load_dataset
from tqdm import tqdm

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
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    dataset_split: str = "test",
) -> dict:
    """
    計算 Causal LM 的 Perplexity (PPL) - 遵循 Hugging Face 官方標準 Strided Sliding Window 演算法。

    運作原理：
      1. 窗口長度為 max_length (如 2048)，滑動步長為 stride (如 512)。
      2. 前綴歷史上下文 (前 max_length - stride 個 tokens) 在 labels 中被設為 -100 (遮罩忽略)。
      3. 只有最新跨進窗口的 trg_len (通常為 512) 個 tokens 會計算 CrossEntropyLoss。
      4. 消除非重疊切塊的「上下文飢餓效應 (Context Starvation)」，獲得具國際公信力的黃金基準分。

    Returns:
        dict: {
            "ppl": float,
            "total_tokens": int,
            "elapsed_sec": float,
            "tokens_per_sec": float,
            "total_steps": int,
        }
    """
    # 1. 載入標準測試集
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    full_text = "\n\n".join([t for t in dataset["text"] if t.strip()])

    # 2. 轉為 Token ID
    encodings = tokenizer(full_text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    print(f"📖 資料集: {dataset_name}/{dataset_config} ({dataset_split})")
    print(f"  • 資料集總 Token 數 : {seq_len:,}")
    print(f"  • 最大窗口 (max_len) : {max_length}")
    print(f"  • 滑動步長 (stride)  : {stride}")

    nlls = []
    prev_end_loc = 0
    step_count = 0
    start_time = time.time()

    # 計算總步數供 tqdm 顯示
    total_possible_steps = (seq_len + stride - 1) // stride
    total_steps = min(total_possible_steps, max_steps) if max_steps else total_possible_steps

    progress_bar = tqdm(total=total_steps, desc="官方標準 Strided PPL 評估")

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # 本次步長實際新增計分的 Token 數

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()

        # 關鍵核心：將歷史上下文 tokens 遮罩為 -100，只有最新的 trg_len 個 tokens 計算損失
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=target_ids)
            # outputs.loss 是針對 labels != -100 的 token 的平均損失
            neg_log_likelihood = outputs.loss.item() * trg_len
            nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        step_count += 1
        progress_bar.update(1)

        # 關鍵優化：每一步立即釋放暫存並清理 GPU 快取，杜絕記憶體堆積
        del outputs, input_ids, target_ids
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

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

    print("\n⏳ 正在安全載入模型結構至 GPU (MPS)...")
    load_start = time.time()
    model, tokenizer = load_pure_text_model_and_tokenizer(resolved_path, device, torch_dtype)
    load_time = time.time() - load_start

    ram_gb = psutil.Process().memory_info().rss / (1024 ** 3)
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"✅ 模型與 Tokenizer 載入完成！")
    print(f"  • 載入耗時 : {load_time:.2f} 秒")
    print(f"  • 文字參數量 : {num_params:.2f} B ({num_params * 1000:.0f} M)")
    print(f"  • 實體記憶體佔用 : ~{ram_gb:.2f} GB RAM")

    # 執行官方標準 Strided Sliding Window PPL 評估
    results = evaluate_ppl(
        model,
        tokenizer,
        device=device,
        max_length=args.max_length,
        stride=args.stride,
        max_steps=args.max_steps,
    )

    print("\n" + "=" * 68)
    print("📊 官方 WikiText-2 基準評估結果 (Official Benchmark Results)：")
    print("-" * 68)
    print(f"  🏆 WikiText-2 PPL (困惑度) : {results['ppl']:.4f} (越低越好)")
    print(f"  ⚡ 評估總耗時              : {results['elapsed_sec']:.2f} 秒")
    print(f"  🚀 處理吞吐量              : {results['tokens_per_sec']:.2f} tokens/秒")
    print(f"  📝 累計評估 Token 數量     : {results['total_tokens']:,} tokens")
    print(f"  🔄 執行滑動步數 (Steps)    : {results['total_steps']} 步")
    print("=" * 68)


if __name__ == "__main__":
    main()
