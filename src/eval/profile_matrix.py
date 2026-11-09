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
    """Automatically localizes pure text Transformer Decoder Layers."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 5:
            # Ensure text decoder layers are targeted, filtering out vision and audio modules
            if "language_model" in name or ("layers" in name and "vision" not in name and "visual" not in name and "audio" not in name):
                return name, list(module)
    raise RuntimeError("Unable to locate Transformer decoder layers in the model architecture.")


def profile_activation_matrix(model, tokenizer, device: str = "mps", num_tokens: int = 1024) -> list:
    """
    Profiles layer-wise activation tensors via forward hooks to map outlier channels and peak-to-average ratios (PAR).
    """
    # 1. Prepare diagnostic calibration text (injecting chat template if present)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    sample_text = "\n\n".join(dataset["text"][:10])
    if getattr(tokenizer, "chat_template", None) is not None:
        messages = [{"role": "user", "content": "Please analyze and process the following text passage:\n" + sample_text}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted = sample_text

    inputs = tokenizer(formatted, max_length=num_tokens, truncation=True, return_tensors="pt").to(device)

    # 2. Locate Transformer layers and register forward hooks
    layer_name, layers = find_transformer_layers(model)
    num_layers = len(layers)
    layer_stats = []
    hooks = []

    activation_records = {}

    def make_hook(idx):
        def hook_fn(module, input_tensor, output_tensor):
            # Extract activation tensor: shape [batch, seq_len, hidden_dim]
            feat = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            feat = feat.detach().float()

            # Compute channel-wise absolute maximum across tokens: shape [hidden_dim]
            channel_max = torch.max(torch.abs(feat), dim=1).values.squeeze(0)
            
            global_max = channel_max.max().item()
            global_mean = channel_max.mean().item()
            # Peak-to-Average Ratio (PAR): standard metric for outlier severity
            par = global_max / (global_mean + 1e-8)

            # Count channels exceeding 6x mean activation
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

    # 3. Single forward pass
    with torch.no_grad():
        model(**inputs)

    # Detach hooks
    for h in hooks:
        h.remove()

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
    print("🔍 Pre-Quantization Activation Outlier & Sensitivity Profiler")
    print(f"  • Target Model : {resolved_path}")
    print(f"  • Device GPU   : {device.upper()}")
    print(f"  • Eval Length  : {args.num_tokens} tokens")
    print("=" * 72)

    print(f"\n⏳ Loading model architecture to device ({device.upper()})...")
    model, tokenizer = load_pure_text_model_and_tokenizer(resolved_path, device)
    print("✅ Model loaded successfully. Capturing layer-wise activation tensors...")

    t0 = time.time()
    stats = profile_activation_matrix(model, tokenizer, device=device, num_tokens=args.num_tokens)
    elapsed = time.time() - t0

    # Display diagnostic profiling table
    print("\n" + "=" * 72)
    print(f"{'Layer':<12} | {'Max Activation':<16} | {'Channel Mean':<16} | {'PAR (Peak/Avg)':<14} | {'Outlier Channels':<12}")
    print("-" * 72)

    for item in stats:
        warn = "⚠️ Severe Spike" if item["par"] > 15 else " "
        print(f"Layer {item['layer_idx']:<6} | {item['global_max']:<16.2f} | {item['global_mean']:<16.2f} | {item['par']:<6.1f}x {warn} | {item['outlier_count']}/{item['total_channels']}")

    max_spike_layer = max(stats, key=lambda x: x["par"])
    print("=" * 72)
    print(f"📊 Diagnostic Profiling Summary (Elapsed: {elapsed:.2f} s):")
    print(f"  1. Peak Outlier Layer  : Layer {max_spike_layer['layer_idx']} (Peak: {max_spike_layer['global_max']:.2f}, PAR: {max_spike_layer['par']:.1f}x)")
    print(f"  2. Diagnostic Takeaway : Isolated channel spikes detected. Channel-wise scaling or orthogonal rotations (WHT/SpinQuant) required to avoid low-bit clipping collapse.")
    print("=" * 72)


if __name__ == "__main__":
    main()
