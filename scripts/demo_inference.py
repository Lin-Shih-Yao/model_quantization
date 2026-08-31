"""
Demo Inference Script for LLMs (Gemma 4, Qwen 3.5, LLaMA 3.2)
Pure-Text Mode: Automatically strips non-text towers (vision/audio) to minimize memory and compute,
supports automatic device detection (MPS on Apple Silicon / CUDA / CPU),
Chat Template formatting, and throughput measurement.
"""

import argparse
import os
import time
import psutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)


def get_optimal_device():
    """Determine the fastest available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_model_path(model_id: str) -> str:
    """If model exists locally in ./models/<model_short_name>, use the local path."""
    if os.path.exists(model_id):
        return os.path.abspath(model_id)

    short_name = model_id.replace("/", "_")
    local_dir = os.path.abspath(f"./models/{short_name}")
    if os.path.exists(local_dir):
        print(f"💡 偵測到本地已存在下載權重，自動切換至本機路徑: {local_dir}")
        return local_dir

    return model_id


def prune_non_text_modules(model):
    """Prune vision and audio encoders to free memory and prevent CPU/GPU heating."""
    non_text_attrs = [
        "vision_tower",
        "audio_tower",
        "visual",
        "vision_model",
        "audio_model",
        "multi_modal_projector",
        "image_encoder",
    ]
    pruned = []
    for attr in non_text_attrs:
        if hasattr(model, attr):
            try:
                delattr(model, attr)
                pruned.append(attr)
            except Exception:
                pass

    if pruned:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"🧹 已自動剝離非文字模組 ({', '.join(pruned)})，顯存已立即釋放！")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo Text Inference for Target LLMs (Gemma 4, Qwen 3.5, LLaMA 3.2)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/gemma-4-E2B-it",
        help="HuggingFace model ID or local directory (e.g. google/gemma-4-E2B-it, Qwen/Qwen3.5-2B, meta-llama/Llama-3.2-1B-Instruct)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="請用一句話解釋什麼是大語言模型量化（Model Quantization）？",
        help="User query or prompt to test generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model weight precision (bfloat16 recommended for Mac M3)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_optimal_device()
    resolved_path = resolve_model_path(args.model_id)

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    print("=" * 65)
    print(f"🚀 LLM Pure-Text Inference Demo")
    print(f"  • Target Model : {resolved_path}")
    print(f"  • Device       : {device.upper()} (Apple Silicon MPS / CUDA / CPU)")
    print(f"  • Precision    : {args.dtype}")
    print(f"  • Prompt       : {args.prompt}")
    print("=" * 65)

    # 1. 載入 Model 與 Processor/Tokenizer
    print("\n⏳ 正在載入純文字模型結構...")
    load_start = time.time()

    is_multimodal_wrapper = False
    processor = None
    tokenizer = None
    model = None

    # 優先嘗試以 AutoProcessor / AutoModelForImageTextToText 載入 (Gemma 4 / Qwen 3.5)
    try:
        processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            resolved_path,
            dtype=torch_dtype,
            device_map="auto" if device != "mps" else None,
            trust_remote_code=True,
        ).to(device)
        is_multimodal_wrapper = True
    except Exception:
        # Fallback 到純文字 AutoTokenizer / AutoModelForCausalLM (Llama 3.2 等)
        tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_path,
            dtype=torch_dtype,
            device_map="auto" if device != "mps" else None,
            trust_remote_code=True,
        ).to(device)
        is_multimodal_wrapper = False

    # 2. 自動剝離所有視覺與語音編碼器，只保留文字主幹
    prune_non_text_modules(model)

    model.eval()
    load_time = time.time() - load_start

    # 統計記憶體與純文字參數量
    ram_gb = psutil.Process().memory_info().rss / (1024 ** 3)
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"✅ 模型載入完成！")
    print(f"  • 載入耗時 : {load_time:.2f} 秒")
    print(f"  • 文字參數量 : {num_params:.2f} B ({num_params * 1000:.0f} M)")
    print(f"  • 實體記憶體佔用 : ~{ram_gb:.2f} GB RAM (極為輕量且安全)")

    # 3. 構建純文字對話與 Chat Template
    messages = [
        {"role": "user", "content": [{"type": "text", "text": args.prompt}] if is_multimodal_wrapper else args.prompt}
    ]

    print("\n⚡ 開始純文字推論生成中...")
    gen_start = time.time()

    if is_multimodal_wrapper:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

        gen_time = time.time() - gen_start
        generated_tokens = outputs[0][input_len:]
        num_generated = len(generated_tokens)
        response_text = processor.decode(generated_tokens, skip_special_tokens=True)
    else:
        if tokenizer.chat_template is not None:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = args.prompt

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id or 0,
            )

        gen_time = time.time() - gen_start
        generated_tokens = outputs[0][input_len:]
        num_generated = len(generated_tokens)
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    throughput = num_generated / gen_time if gen_time > 0 else 0

    # 4. 輸出結果與性能指標
    print("\n" + "=" * 65)
    print("📝 純文字回覆結果：")
    print("-" * 65)
    print(str(response_text).strip())
    print("=" * 65)
    print(f"📊 推論性能指標：")
    print(f"  • 輸入長度 : {input_len} tokens")
    print(f"  • 生成長度 : {num_generated} tokens")
    print(f"  • 生成耗時 : {gen_time:.2f} 秒")
    print(f"  • 生成速度 : {throughput:.2f} tokens/秒")
    print("=" * 65)


if __name__ == "__main__":
    main()
