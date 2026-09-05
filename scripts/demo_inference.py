"""
Demo Inference Script & Function for Target LLMs (Qwen 3.5 2B/4B, Gemma 4 E2B, LLaMA 3.2 1B/3B)
Can be imported directly as a function `demo_inference(...)` for downstream evaluation and calibration,
or executed from the command line.
"""

import argparse
import os
import sys
import time
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_optimal_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def resolve_model_path(model_id: str) -> str:
    if os.path.exists(model_id):
        return model_id
    drive_base = "/content/drive/MyDrive/models"
    clean_name = model_id.replace("/", "_")
    candidates = [
        os.path.join(drive_base, model_id.split("/")[-1]),
        os.path.join(drive_base, clean_name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return model_id

def load_pure_text_model_and_tokenizer(model_path: str, device: str, torch_dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda" and torch_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda" or model.device.type == "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 50) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_tokens = inputs["input_ids"].shape[1]

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    if device == "cuda":
        torch.cuda.synchronize()

    latency_sec = time.time() - start_time
    output_tokens = output_ids.shape[1] - input_tokens
    response_text = tokenizer.decode(output_ids[0][input_tokens:], skip_special_tokens=True)
    tokens_per_sec = output_tokens / latency_sec if latency_sec > 0 else 0.0

    return {
        "response": response_text.strip(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_sec": latency_sec,
        "tokens_per_sec": tokens_per_sec,
    }

def demo_inference(
    model_id: str = "Qwen/Qwen3.5-2B",
    prompt: str = "請用一句話解釋什麼是大語言模型量化（Model Quantization）？",
    model=None,
    tokenizer=None,
    max_new_tokens: int = 50,
    dtype: str = "bfloat16",
    device: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    通用推論入口函式（可直接在其他 evaluation、校正或測試腳本中引用）：
    
    支援兩種呼叫模式：
      1. 傳入 model_id：自動載入模型、自動推論並回傳結果。
      2. 傳入已存在的 model 與 tokenizer（如量化後模型）：直接復用記憶體中的模型執行推論。

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
        print("🚀 LLM Pure-Text Inference Demo (MPS GPU Acceleration)")
        print(f"  • Target Model : {resolved_path}")
        print(f"  • Device GPU   : {device.upper()} (Apple Silicon MPS / CUDA)")
        print(f"  • Precision    : {dtype}")
        print(f"  • Prompt       : {prompt}")
        print("=" * 68)

    # 1. 若外部未提供已載入的模型，則自動安全載入純文字模型
    if model is None or tokenizer is None:
        if verbose:
            print("\n⏳ 正在載入純文字模型結構至 GPU (MPS)...")
        load_start = time.time()
        model, tokenizer = load_pure_text_model_and_tokenizer(resolved_path, device, torch_dtype)
        load_time = time.time() - load_start

        ram_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        num_params = sum(p.numel() for p in model.parameters()) / 1e9
        if verbose:
            print(f"✅ 模型載入完成！")
            print(f"  • 載入耗時 : {load_time:.2f} 秒")
            print(f"  • 文字參數量 : {num_params:.2f} B ({num_params * 1000:.0f} M)")
            print(f"  • 實體記憶體佔用 : ~{ram_gb:.2f} GB RAM (安全且輕量)")
    else:
        ram_gb = psutil.Process().memory_info().rss / (1024 ** 3)

    # 2. 執行推論生成
    if verbose:
        print("\n⚡ 開始純文字推論生成中 (MPS GPU)...")
    gen_result = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
    )

    if verbose:
        print("\n" + "=" * 68)
        print("📝 純文字回覆結果：")
        print("-" * 68)
        print(gen_result["response"])
        print("=" * 68)
        print(f"📊 推論性能指標 (MPS GPU)：")
        print(f"  • 輸入長度 : {gen_result['input_tokens']} tokens")
        print(f"  • 生成長度 : {gen_result['output_tokens']} tokens")
        print(f"  • 生成耗時 : {gen_result['latency_sec']:.2f} 秒")
        print(f"  • 生成速度 : {gen_result['tokens_per_sec']:.2f} tokens/秒")
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
        description="Demo Pure-Text Inference for LLMs (Qwen 3.5 2B/4B, Gemma 4 E2B, LLaMA 3.2)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3.5-2B",
        help="HuggingFace model ID or local directory (e.g. Qwen/Qwen3.5-2B, google/gemma-4-E2B-it, Qwen/Qwen3.5-4B)",
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
        default=50,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model weight precision (bfloat16 recommended for Mac M3 MPS)",
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
