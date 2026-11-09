"""
Unified Model Downloader for LLM Benchmarks
Supports downloading single models, custom model lists, or all benchmark models.

Usage:
    # 1. Download single model
    python scripts/download_model.py --model_id meta-llama/Llama-3.2-1B-Instruct

    # 2. Batch download all benchmark models
    python scripts/download_model.py --all

    # 3. Download custom list of models
    python scripts/download_model.py --models Qwen/Qwen3.5-2B google/gemma-2-2b-it
"""

import argparse
import os
import sys
from huggingface_hub import snapshot_download

TARGET_MODELS = [
    # 1. LLaMA 3.2 Series (Gated, requires HF authorization)
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    # 2. Qwen 3.5 Series
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    # 3. Gemma Series
    "google/gemma-2-2b-it",
    "google/gemma-4-E2B-it",
]


def download_single_model(model_id: str, base_dir: str = "./models", token: str | None = None) -> bool:
    """Downloads single Hugging Face model weights to target directory."""
    short_name = model_id.replace("/", "_")
    target_dir = os.path.abspath(os.path.join(base_dir, short_name))

    print("\n" + "=" * 65)
    print(f"📦 Preparing download: {model_id}")
    print(f"📁 Local target path : {target_dir}")
    print(f"🔑 Auth Token        : {'Provided via argument/env' if token else 'Using local cached token'}")
    print("=" * 65)

    if os.path.exists(target_dir) and any(os.scandir(target_dir)):
        print(f"💡 Target directory exists. Verifying file integrity and resuming...")

    os.makedirs(target_dir, exist_ok=True)
    print(f"⏳ Downloading weights, please wait...")

    try:
        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,  # Download physical weight files directly
            token=token,
        )
        print(f"✅ {model_id} successfully downloaded to: {local_path}")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download {model_id}: {e}")
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            print("\n👉 Note: Gated models (e.g. LLaMA 3.2) require accepted licenses on Hugging Face:")
            print("   1. Accept license on official Hugging Face model card.")
            print("   2. Run `huggingface-cli login` or supply token via `--token hf_xxxx`:")
            print(f"      python scripts/download_model.py --model_id {model_id} --token hf_xxxx")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified LLM weight downloader supporting single, custom list, or all benchmark models."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Single model ID (e.g., meta-llama/Llama-3.2-1B-Instruct, Qwen/Qwen3.5-2B)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Custom list of model IDs (e.g., --models Qwen/Qwen3.5-2B google/gemma-2-2b-it)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Batch download all benchmark models (Llama 3.2, Qwen 3.5, Gemma)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models",
        help="Local target directory (default: ./models)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face Access Token (optional, can also use HF_TOKEN env var)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    if args.all:
        models_to_download = TARGET_MODELS
    elif args.models:
        models_to_download = args.models
    elif args.model_id:
        models_to_download = [args.model_id]
    else:
        default_model = "meta-llama/Llama-3.2-1B-Instruct"
        print(f"ℹ️ No arguments specified. Defaulting to: {default_model}")
        print("💡 Use --all to download all benchmark models, or --help for options.\n")
        models_to_download = [default_model]

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 65)
    print(f"🚀 Starting model download pipeline ({len(models_to_download)} models total)")
    for idx, m in enumerate(models_to_download, 1):
        print(f"   {idx}. {m}")
    print("=" * 65)

    success_count = 0
    failed_models = []

    for model_id in models_to_download:
        success = download_single_model(model_id, base_dir=args.save_dir, token=token)
        if success:
            success_count += 1
        else:
            failed_models.append(model_id)

    print("\n" + "=" * 65)
    print(f"🎉 Download pipeline completed!")
    print(f"  • Successful : {success_count}/{len(models_to_download)}")
    if failed_models:
        print(f"  • Failed/Gated: {failed_models}")
    print("=" * 65)


if __name__ == "__main__":
    main()
