"""
Unit Tests for W8A8 Quantization Module
Verifies INT8 weight and activation quantization, clipping protection, and layer replacement.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from src.quantization.w8a8 import W8A8Linear, quantize_model_w8a8


class MockLLM(nn.Module):
    """Lightweight neural network for module replacement testing."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.embed_tokens = nn.Embedding(50, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.down_proj = nn.Linear(dim * 2, dim)
        self.lm_head = nn.Linear(dim, 50)


def test_w8a8_clipping_and_rounding():
    """Verifies W8A8Linear clipping and rounding behaviors under extreme activation outliers."""
    torch.manual_seed(42)
    linear = nn.Linear(16, 32)
    # Set activation clipping to 5.0 and scale to 5.0 / 127
    qlayer = W8A8Linear.from_linear(linear, act_scale=5.0 / 127.0, act_clip_max=5.0)

    # Simulate input tensor with an extreme outlier spike (e.g., 500.0)
    x = torch.tensor([[[0.5, -2.0, 500.0] + [0.1] * 13]])
    x_q = qlayer.quantize_activation(x)

    # Verify spike at 500.0 is clipped and capped near 5.0
    spike_val = x_q[0, 0, 2].item()
    assert abs(spike_val) <= 5.05, f"Spike clipping failed: {spike_val}"

    # Verify forward pass executes cleanly without NaNs
    out = qlayer(x)
    assert not torch.isnan(out).any()
    assert out.shape == (1, 1, 32)


def test_w8a8_selective_replacement():
    """Verifies that W8A8 replacement preserves sensitive Embedding and lm_head layers."""
    model = MockLLM(dim=32)
    quantize_model_w8a8(model)

    assert isinstance(model.q_proj, W8A8Linear)
    assert isinstance(model.down_proj, W8A8Linear)
    assert isinstance(model.embed_tokens, nn.Embedding)
    assert isinstance(model.lm_head, nn.Linear)


if __name__ == "__main__":
    test_w8a8_clipping_and_rounding()
    test_w8a8_selective_replacement()
    print("🎉 All W8A8 unit tests passed successfully!")
