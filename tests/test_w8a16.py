"""
Unit Tests for W8A16 Quantization Module
Verifies numerical precision, INT8 storage compression, and selective layer replacement.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from src.quantization.w8a16 import W8A16Linear, quantize_model_w8a16


class MockLLM(nn.Module):
    """用於測試結構替換的輕量模擬 LLM 骨幹結構。"""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.down_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, 100)


def test_w8a16_linear_precision():
    """驗證 W8A16Linear 的數值誤差與 INT8 權重儲存。"""
    torch.manual_seed(42)
    orig_linear = nn.Linear(128, 256, bias=True)
    q_linear = W8A16Linear.from_linear(orig_linear)

    assert q_linear.weight_int8.dtype == torch.int8
    assert q_linear.scales.shape == (256, 1)

    x = torch.randn(2, 16, 128)
    y_orig = orig_linear(x)
    y_quant = q_linear(x)

    # 驗證量化與原始浮點輸出的平均誤差小於 0.05
    mae = torch.mean(torch.abs(y_orig - y_quant)).item()
    assert mae < 0.05, f"W8A16 數值誤差過大: {mae:.4f}"


def test_model_selective_quantization():
    """驗證模型量化時只替換核心 7 層，保留 Embedding 與 lm_head。"""
    model = MockLLM(dim=32)
    quantize_model_w8a16(model)

    # 核心層被替換為 W8A16Linear
    assert isinstance(model.q_proj, W8A16Linear)
    assert isinstance(model.down_proj, W8A16Linear)

    # 非目標敏感層保持原樣 (不可被量化)
    assert isinstance(model.embed_tokens, nn.Embedding)
    assert isinstance(model.norm, nn.LayerNorm)
    assert isinstance(model.lm_head, nn.Linear)


if __name__ == "__main__":
    test_w8a16_linear_precision()
    test_model_selective_quantization()
    print("🎉 All W8A16 unit tests passed successfully!")
