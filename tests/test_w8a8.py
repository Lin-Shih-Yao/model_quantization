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
    """測試替換用的輕量神經網路。"""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.embed_tokens = nn.Embedding(50, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.down_proj = nn.Linear(dim * 2, dim)
        self.lm_head = nn.Linear(dim, 50)


def test_w8a8_clipping_and_rounding():
    """驗證 W8A8Linear 在輸入極端尖刺下的截斷與四捨五入行為。"""
    torch.manual_seed(42)
    linear = nn.Linear(16, 32)
    # 設定激活截斷為 5.0，Scale 為 5.0 / 127
    qlayer = W8A8Linear.from_linear(linear, act_scale=5.0 / 127.0, act_clip_max=5.0)

    # 模擬含有極端尖刺 (如 500.0) 的輸入張量
    x = torch.tensor([[[0.5, -2.0, 500.0] + [0.1] * 13]])
    x_q = qlayer.quantize_activation(x)

    # 驗證原本 500.0 的尖刺被截斷並限制在 5.0 附近 (不超過截斷上限)
    spike_val = x_q[0, 0, 2].item()
    assert abs(spike_val) <= 5.05, f"尖刺截斷失敗: {spike_val}"

    # 驗證前向運算輸出正常且無 NaN
    out = qlayer(x)
    assert not torch.isnan(out).any()
    assert out.shape == (1, 1, 32)


def test_w8a8_selective_replacement():
    """驗證 W8A8 替換時完整保留敏感的 Embedding 與 lm_head。"""
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
