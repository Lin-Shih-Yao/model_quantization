"""
Unified Model Downloader for LLM Benchmarks
Supports downloading single models, custom model lists, or all benchmark models.

Usage:
    # 1. 下載單一模型
    python scripts/download_model.py --model_id meta-llama/Llama-3.2-1B-Instruct

    # 2. 批次下載專案所有基準模型
    python scripts/download_model.py --all

    # 3. 下載自訂多個模型清單
    python scripts/download_model.py --models Qwen/Qwen3.5-2B google/gemma-2-2b-it
"""

import argparse
import os
import sys
from huggingface_hub import snapshot_download

TARGET_MODELS = [
    # 1. LLaMA 3.2 系列 (Gated, 需 HF 授權)
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    # 2. Qwen 3.5 系列
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    # 3. Gemma 系列
    "google/gemma-2-2b-it",
    "google/gemma-4-E2B-it",
]


def download_single_model(model_id: str, base_dir: str = "./models", token: str | None = None) -> bool:
    """下載單一 Hugging Face 模型權重至指定本地目錄。"""
    short_name = model_id.replace("/", "_")
    target_dir = os.path.abspath(os.path.join(base_dir, short_name))

    print("\n" + "=" * 65)
    print(f"📦 準備下載: {model_id}")
    print(f"📁 儲存路徑: {target_dir}")
    print(f"🔑 授權狀態: {'已提供 Token' if token else '未提供 (使用本機登入快取)'}")
    print("=" * 65)

    if os.path.exists(target_dir) and any(os.scandir(target_dir)):
        print(f"💡 本地已存在該模型目錄，將檢查完整性並接續下載...")

    os.makedirs(target_dir, exist_ok=True)
    print(f"⏳ 下載中，請稍候...")

    try:
        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,  # 直接存放實體權重檔案
            token=token,
        )
        print(f"✅ {model_id} 下載完成！實體路徑: {local_path}")
        return True
    except Exception as e:
        print(f"\n❌ {model_id} 下載失敗: {e}")
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            print("\n👉 提示：此模型 (如 Llama 3.2) 為 Gated Model，需要 Hugging Face 存取權限：")
            print("   1. 請至 Hugging Face 模型官方頁面確認已點擊同意授權 (Agree to License)")
            print("   2. 執行 `huggingface-cli login` 輸入 Token，或在指令後加上 `--token hf_xxxx`：")
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
        help="單一模型 ID (例如: meta-llama/Llama-3.2-1B-Instruct, Qwen/Qwen3.5-2B)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="自訂多個模型 ID 清單 (例如: --models Qwen/Qwen3.5-2B google/gemma-2-2b-it)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="批次下載專案所有基準模型 (Llama 3.2, Qwen 3.5, Gemma)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models",
        help="本地存放根目錄 (預設: ./models)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace Access Token (可選，亦可透過環境變數 HF_TOKEN 或 huggingface-cli login 提供)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    # 決定待下載的模型清單
    if args.all:
        models_to_download = TARGET_MODELS
    elif args.models:
        models_to_download = args.models
    elif args.model_id:
        models_to_download = [args.model_id]
    else:
        # 若未指定任何參數，預設下載 Llama-3.2-1B-Instruct
        default_model = "meta-llama/Llama-3.2-1B-Instruct"
        print(f"ℹ️ 未指定參數，預設下載基準模型: {default_model}")
        print("💡 亦可使用 --all 批次下載所有模型，或使用 --help 查看選項。\n")
        models_to_download = [default_model]

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 65)
    print(f"🚀 開始模型下載流程 (共 {len(models_to_download)} 款模型)")
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
    print(f"🎉 下載流程結束！")
    print(f"  • 成功: {success_count}/{len(models_to_download)}")
    if failed_models:
        print(f"  • 失敗或需授權: {failed_models}")
    print("=" * 65)


if __name__ == "__main__":
    main()
