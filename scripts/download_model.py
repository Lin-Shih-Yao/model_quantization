"""
Download Hugging Face Models to a fixed local directory.
"""

import argparse
import os
from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download LLM weights to a specified local folder"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/gemma-4-E2B-it",
        help="HuggingFace model ID (e.g. google/gemma-4-E2B-it, Qwen/Qwen3.5-2B, meta-llama/Llama-3.2-1B-Instruct)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Target local directory (default: ./models/<model_short_name>)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 若未指定 save_dir，預設放到 ./models/<model_short_name>
    if args.save_dir is None:
        short_name = args.model_id.replace("/", "_")
        target_dir = os.path.abspath(f"./models/{short_name}")
    else:
        target_dir = os.path.abspath(args.save_dir)

    print("=" * 65)
    print(f"📦 Downloading Model to Local Directory")
    print(f"  • Source Model ID : {args.model_id}")
    print(f"  • Target Location : {target_dir}")
    print("=" * 65)

    os.makedirs(target_dir, exist_ok=True)

    print(f"\n⏳ 下載中，請稍候...")
    local_path = snapshot_download(
        repo_id=args.model_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,  # 直接存放實體檔案，而非快取軟連結
    )

    print("\n" + "=" * 65)
    print(f"✅ 下載完成！實體檔案已存放在：")
    print(f"   {local_path}")
    print("\n💡 後續在程式中即可直接使用此路徑：")
    print(f"   python scripts/demo_inference.py --model_id {local_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
