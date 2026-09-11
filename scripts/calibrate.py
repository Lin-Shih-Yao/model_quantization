"""
CLI Shortcut for LLM Activation Calibration.
Slices WikiText-2 (train split), hooks Linear layers, and saves .pt scales for PTQ.

Usage:
    python scripts/calibrate.py --model_id Qwen/Qwen3.5-2B
    python scripts/calibrate.py --model_id meta-llama/Llama-3.2-1B --output_dir /content/drive/MyDrive/calib_results
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import (
    load_pure_text_model_and_tokenizer,
    get_optimal_device,
    resolve_model_path,
)
from src.quantization import calibrate_model


def parse_args():
    """解析校準命令行參數。"""
    parser = argparse.ArgumentParser(description="LLM Activation Calibration Pipeline")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3.5-2B",
        help="HuggingFace model ID or local directory path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./calibration_results",
        help="Directory to save calibration .pt checkpoint",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Token sequence length per sample (default: 2048)",
    )
    parser.add_argument(
        "--clip_quantile",
        type=float,
        default=0.9999,
        help="Quantile clipping ratio to suppress outliers (default: 0.9999)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model load precision",
    )
    return parser.parse_args()


def build_output_path(output_dir: str, model_id: str) -> str:
    """產出結構化之 .pt 快取存檔路徑。"""
    clean_name = model_id.strip("/").replace("/", "_")
    return os.path.join(output_dir, f"calib_{clean_name}.pt")


def main():
    """校準 CLI 主進入點。"""
    args = parse_args()
    device = get_optimal_device()
    resolved_path = resolve_model_path(args.model_id)
    output_path = build_output_path(args.output_dir, args.model_id)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    print("=" * 68)
    print("🎯 大語言模型激活值校準系統 (Activation Calibration Pipeline)")
    print(f"  • Target Model   : {resolved_path}")
    print(f"  • Device         : {device.upper()}")
    print(f"  • Sample Config  : {args.n_samples} samples × {args.seq_len} tokens")
    print(f"  • Clip Quantile  : {args.clip_quantile * 100:.2f}% (Outlier Suppression)")
    print(f"  • Output Path    : {output_path}")
    print("=" * 68)

    print("\n⏳ 正在載入模型與 Tokenizer...")
    model, tokenizer = load_pure_text_model_and_tokenizer(
        resolved_path,
        device=device,
        dtype=dtype_map[args.dtype],
    )

    print("\n⚙️ 開始執行激活值前向傳播與統計採集...")
    calibrate_model(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_id=args.model_id,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        output_path=output_path,
        clip_quantile=args.clip_quantile,
    )


if __name__ == "__main__":
    main()
