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
    """Lightweight mock LLM backbone for testing structural layer replacement."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.down_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, 100)


def test_w8a16_linear_precision():
    """Verifies W8A16Linear numerical precision and INT8 buffer storage."""
    torch.manual_seed(42)
    orig_linear = nn.Linear(128, 256, bias=True)
    q_linear = W8A16Linear.from_linear(orig_linear)

    assert q_linear.weight_int8.dtype == torch.int8
    assert q_linear.scales.shape == (256, 1)

    x = torch.randn(2, 16, 128)
    y_orig = orig_linear(x)
    y_quant = q_linear(x)

    # Verify mean absolute error between quantized and float output is small (< 0.05)
    mae = torch.mean(torch.abs(y_orig - y_quant)).item()
    assert mae < 0.05, f"W8A16 MAE too large: {mae:.4f}"


def test_model_selective_quantization():
    """Verifies that only targeted core projections are replaced, leaving Embeddings & Head untouched."""
    model = MockLLM(dim=32)
    quantize_model_w8a16(model)

    # Core projections replaced by W8A16Linear
    assert isinstance(model.q_proj, W8A16Linear)
    assert isinstance(model.down_proj, W8A16Linear)

    # Sensitive non-target layers remain untouched
    assert isinstance(model.embed_tokens, nn.Embedding)
    assert isinstance(model.norm, nn.LayerNorm)
    assert isinstance(model.lm_head, nn.Linear)


if __name__ == "__main__":
    test_w8a16_linear_precision()
    test_model_selective_quantization()
    print("🎉 All W8A16 unit tests passed successfully!")
