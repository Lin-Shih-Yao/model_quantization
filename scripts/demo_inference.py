"""
Demo Inference Script & Function for Target LLMs (Qwen 3.5 2B/4B, Gemma 2B, LLaMA 3.2 1B/3B)
Can be imported directly as a function `demo_inference(...)` for downstream evaluation and calibration,
or executed from the command line.
"""

import argparse
import os
import sys
import time
import psutil
import torch

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import (
    get_optimal_device,
    resolve_model_path,
    load_pure_text_model_and_tokenizer,
)
from src.models.inference import generate_response


def demo_inference(
    model_id: str = "Qwen/Qwen3.5-2B",
    prompt: str = "Explain what Post-Training Quantization (PTQ) means in one concise sentence.",
    model=None,
    tokenizer=None,
    max_new_tokens: int = 50,
    dtype: str = "bfloat16",
    device: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Universal inference entrypoint.
    Supports two invocation patterns:
      1. Pass model_id: loads model, runs generation, and returns performance metrics.
      2. Pass existing model and tokenizer: reuses in-memory quantized model.

    Returns:
        dict: {
            "response": str,
            "input_tokens": int,
            "output_tokens": int,
            "latency_sec": float,
            "tokens_per_sec": float,
            "ram_gb": float,
            "model": model,
            "tokenizer": tokenizer,
        }
    """
    if device is None:
        device = get_optimal_device()

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)

    resolved_path = resolve_model_path(model_id)

    if verbose:
        print("=" * 68)
        print("🚀 LLM Text Inference Benchmark")
        print(f"  • Target Model : {resolved_path}")
        print(f"  • Device GPU   : {device.upper()} (MPS / CUDA)")
        print(f"  • Precision    : {dtype}")
        print(f"  • Prompt       : {prompt}")
        print("=" * 68)

    # 1. Load model if not provided
    if model is None or tokenizer is None:
        if verbose:
            print("\n⏳ Loading model architecture and weights to hardware device...")
        load_start = time.time()
        model, tokenizer = load_pure_text_model_and_tokenizer(resolved_path, device=device, dtype=torch_dtype)
        load_time = time.time() - load_start

        ram_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        num_params = sum(p.numel() for p in model.parameters()) / 1e9
        if verbose:
            print(f"✅ Model successfully loaded!")
            print(f"  • Loading Time    : {load_time:.2f} s")
            print(f"  • Parameter Count : {num_params:.2f} B ({num_params * 1000:.0f} M)")
            print(f"  • Host RAM Usage  : ~{ram_gb:.2f} GB RAM")
    else:
        ram_gb = psutil.Process().memory_info().rss / (1024 ** 3)

    # 2. Execute inference generation
    if verbose:
        print("\n⚡ Executing autoregressive text generation...")
    gen_result = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
    )

    if verbose:
        print("\n" + "=" * 68)
        print("📝 Model Generation Output:")
        print("-" * 68)
        print(gen_result["response"])
        print("=" * 68)
        print(f"📊 Inference Profiling Metrics:")
        print(f"  • Prompt Tokens  : {gen_result['input_tokens']} tokens")
        print(f"  • Output Tokens  : {gen_result['output_tokens']} tokens")
        print(f"  • Generation Time: {gen_result['latency_sec']:.2f} s")
        print(f"  • Throughput     : {gen_result['tokens_per_sec']:.2f} tokens/s")
        print("=" * 68)

    return {
        "response": gen_result["response"],
        "input_tokens": gen_result["input_tokens"],
        "output_tokens": gen_result["output_tokens"],
        "latency_sec": gen_result["latency_sec"],
        "tokens_per_sec": gen_result["tokens_per_sec"],
        "ram_gb": ram_gb,
        "model": model,
        "tokenizer": tokenizer,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo Text Inference for LLMs (Qwen 3.5 2B/4B, Gemma 2B, LLaMA 3.2)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3.5-2B",
        help="HuggingFace model ID or local directory (e.g. Qwen/Qwen3.5-2B, google/gemma-2-2b, meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain what Post-Training Quantization (PTQ) means in one concise sentence.",
        help="User query or prompt to test generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model weight precision (bfloat16 recommended)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    demo_inference(
        model_id=args.model_id,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
