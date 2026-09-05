"""
Batch Download Script for Target LLM Benchmarks
Downloads Gemma, Qwen, and LLaMA models to ./models/
"""

import argparse
import os
import sys
from huggingface_hub import snapshot_download

TARGET_MODELS = [
    # 1. Qwen 3.5 系列
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    # 2. Gemma 4 系列
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    # 3. LLaMA 3.2 系列 (Gated, 需 HF 授權)
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def download_single_model(model_id: str, base_dir: str = "./models", token: str | None = None):
    short_name = model_id.replace("/", "_")
    target_dir = os.path.abspath(os.path.join(base_dir, short_name))

    print("\n" + "=" * 65)
    print(f"📦 準備下載: {model_id}")
    print(f"📁 儲存路徑: {target_dir}")
    print("=" * 65)

    if os.path.exists(target_dir) and any(os.scandir(target_dir)):
        print(f"💡 本地已存在該模型目錄，將檢查完整性並更新...")

    try:
        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"✅ {model_id} 下載完成！")
        return True
    except Exception as e:
        print(f"❌ {model_id} 下載失敗: {e}")
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            print("   👉 提示：此模型需要 Hugging Face 存取權限。")
            print("   請先至 Hugging Face 網頁同意授權，並在終端機執行 `huggingface-cli login` 或加上 `--token hf_xxx`。")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch download target LLMs")
    parser.add_argument(
        "--models",
        nargs="+",
        default=TARGET_MODELS,
        help="List of model IDs to download",
    )
    parser.add_argument(
        "--save_base_dir",
        type=str,
        default="./models",
        help="Base directory to save models",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace Access Token (optional if already logged in)",
    )
    args = parser.parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    os.makedirs(args.save_base_dir, exist_ok=True)

    print("=" * 65)
    print("🚀 開始批次下載目標模型")
    print(f"📋 待下載清單 ({len(args.models)} 個模型):")
    for idx, m in enumerate(args.models, 1):
        print(f"   {idx}. {m}")
    print(f"🔑 Token 提供狀態 : {'已提供' if token else '未提供 (使用本機登入快取)'}")
    print("=" * 65)

    success_count = 0
    failed_models = []

    for model_id in args.models:
        success = download_single_model(model_id, args.save_base_dir, token=token)
        if success:
            success_count += 1
        else:
            failed_models.append(model_id)

    print("\n" + "=" * 65)
    print(f"🎉 批次下載流程結束！")
    print(f"  • 成功: {success_count}/{len(args.models)}")
    if failed_models:
        print(f"  • 失敗或需授權的模型: {failed_models}")
    print("=" * 65)


if __name__ == "__main__":
    main()
