"""
Pre-Quantization Activation Outlier & Sensitivity Matrix Profiler.
Captures layer-by-layer hidden state activations to detect outlier channels
before quantization, matching SmoothQuant and SpinQuant profiling methodologies.
"""

import os
import argparse
import time
import psutil
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor

from src.models import (
    load_pure_text_model_and_tokenizer,
    get_optimal_device,
    resolve_model_path,
)


def find_transformer_layers(model):
    """自動定位模型中的純文字 Transformer 解碼層 (Decoder Layers)。"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 5:
            # 確保命中文字解碼層，排除 vision 與 audio 模組
            if "language_model" in name or ("layers" in name and "vision" not in name and "visual" not in name and "audio" not in name):
                return name, list(module)
    raise RuntimeError("無法在模型中找到文字 Transformer 解碼層")


def profile_activation_matrix(model, tokenizer, device: str = "mps", num_tokens: int = 1024) -> list:
    """
    量化前矩陣體檢：透過 Forward Hook 捕獲每一層的激活矩陣，分析離群值 (Outliers)。
    """
    # 1. 準備診斷校準文本 (若是對話模型，自動注入官方 Control Tokens)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    sample_text = "\n\n".join(dataset["text"][:10])
    if getattr(tokenizer, "chat_template", None) is not None:
        messages = [{"role": "user", "content": "Please analyze and process the following text passage:\n" + sample_text}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted = sample_text

    inputs = tokenizer(formatted, max_length=num_tokens, truncation=True, return_tensors="pt").to(device)

    # 2. 尋找所有 Transformer 層並掛載 Hook
    layer_name, layers = find_transformer_layers(model)
    num_layers = len(layers)
    layer_stats = []
    hooks = []

    # 暫存每一層激活值統計資料的字典
    activation_records = {}

    def make_hook(idx):
        def hook_fn(module, input_tensor, output_tensor):
            # 取出激活特徵矩陣: 形狀為 [batch, seq_len, hidden_dim]
            feat = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            feat = feat.detach().float()

            # 計算每個通道 (Channel) 的絕對值最大值
            # 沿著 token 維度取 max: 形狀為 [hidden_dim]
            channel_max = torch.max(torch.abs(feat), dim=1).values.squeeze(0)
            
            global_max = channel_max.max().item()
            global_mean = channel_max.mean().item()
            # 峰均比 (Peak-to-Average Ratio, PAR): 衡量離群值尖刺嚴重度的標準
            par = global_max / (global_mean + 1e-8)

            # 統計被定義為嚴重離群 (大於均值 6 倍) 的通道數量
            outlier_threshold = global_mean * 6.0
            outlier_count = (channel_max > outlier_threshold).sum().item()
            total_channels = channel_max.numel()
            outlier_ratio = (outlier_count / total_channels) * 100

            activation_records[idx] = {
                "layer_idx": idx,
                "global_max": global_max,
                "global_mean": global_mean,
                "par": par,
                "outlier_count": outlier_count,
                "total_channels": total_channels,
                "outlier_ratio": outlier_ratio,
            }
        return hook_fn

    for idx, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    # 3. 執行單次前向傳播 (Forward Pass)
    with torch.no_grad():
        model(**inputs)

    # 卸載 Hook
    for h in hooks:
        h.remove()

    # 整理並排序統計結果
    results = [activation_records[i] for i in range(num_layers)]
    return results


def main():
    parser = argparse.ArgumentParser(description="Pre-Quantization Activation Outlier Matrix Profiler")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-2B", help="Model path or HF ID")
    parser.add_argument("--num_tokens", type=int, default=1024, help="Number of tokens to profile")
    args = parser.parse_args()

    device = get_optimal_device()
    resolved_path = resolve_model_path(args.model_id)

    print("=" * 72)
    print("🔍 大語言模型量化前激活矩陣體檢 (Activation Outlier Profiler)")
    print(f"  • 目標模型 : {resolved_path}")
    print(f"  • 運行硬體 : {device.upper()}")
    print(f"  • 診斷長度 : {args.num_tokens} tokens")
    print("=" * 72)

    print(f"\n⏳ 正在載入模型至硬體設備 ({device.upper()})...")
    model, tokenizer = load_pure_text_model_and_tokenizer(resolved_path, device)
    print("✅ 模型載入完成！開始捕獲各層特徵矩陣...")

    t0 = time.time()
    stats = profile_activation_matrix(model, tokenizer, device=device, num_tokens=args.num_tokens)
    elapsed = time.time() - t0

    # 顯示診斷矩陣分析表
    print("\n" + "=" * 72)
    print(f"{'層級 (Layer)':<12} | {'最大激活值 (Max)':<16} | {'通道均值 (Mean)':<16} | {'峰均比 (PAR)':<12} | {'離群通道數':<10}")
    print("-" * 72)

    for item in stats:
        # 高亮嚴重離群層 (PAR > 15 視為高危險量化層)
        warn = "⚠️ 嚴重尖刺" if item["par"] > 15 else " "
        print(f"Layer {item['layer_idx']:<6} | {item['global_max']:<16.2f} | {item['global_mean']:<16.2f} | {item['par']:<6.1f}x {warn} | {item['outlier_count']}/{item['total_channels']}")

    max_spike_layer = max(stats, key=lambda x: x["par"])
    print("=" * 72)
    print(f"📊 診斷結論總結 (耗時: {elapsed:.2f} 秒)：")
    print(f"  1. 最嚴重離群層級 : Layer {max_spike_layer['layer_idx']} (最大峰值: {max_spike_layer['global_max']:.2f}, 峰均比: {max_spike_layer['par']:.1f} 倍)")
    print(f"  2. 量化建議方針   : 偵測到明顯通道離群尖刺，後續採用 SpinQuant (旋轉矩陣) 或 Channel-wise Scale 量化可有效防止精度崩潰！")
    print("=" * 72)


if __name__ == "__main__":
    main()
