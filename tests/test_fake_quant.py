"""
Unit Tests for Fake Quantizer Module
Verifies symmetric and asymmetric activation fake quantization, hook injection, and multi-format support.
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.quantization.fake_quant import (
    FakeQuantizer,
    symmetric_fake_quantize,
    asymmetric_fake_quantize,
)


class SimpleNetwork(nn.Module):
    def __init__(self, in_features=16, out_features=16):
        super().__init__()
        self.q_proj = nn.Linear(in_features, out_features, bias=False)
        self.out_proj = nn.Linear(out_features, 8, bias=False)

    def forward(self, x):
        return self.out_proj(self.q_proj(x))


def test_symmetric_fake_quantize_math():
    """驗證對稱偽量化的數值計算、截斷與縮放正確性。"""
    x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
    scale = 2.0 / 127.0
    # 截斷在 [-2.0, 2.0]
    q_x = symmetric_fake_quantize(x, scale=scale, clip_max=2.0, bits=8)
    assert abs(q_x[0].item() - (-2.0)) < 1e-5
    assert abs(q_x[-1].item() - 2.0) < 1e-5
    assert abs(q_x[2].item()) < 1e-5


def test_asymmetric_fake_quantize_math():
    """驗證非對稱偽量化的數值計算。"""
    x = torch.tensor([0.0, 5.0, 10.0])
    q_x = asymmetric_fake_quantize(x, min_val=0.0, max_val=10.0, bits=8)
    assert q_x.shape == x.shape
    assert torch.allclose(q_x, x, atol=0.1)


def test_fake_quantizer_attach_and_detach():
    """驗證 FakeQuantizer 在模型上的掛載、推論攔截與乾淨卸載。"""
    model = SimpleNetwork()
    quantizer = FakeQuantizer(target_names=("q_proj",), mode="symmetric")

    # 模擬 LayerObserver 的統計字典
    stats_data = {
        "layers": {
            "q_proj": {
                "clip_max_9999": 2.5,
                "scales": {
                    "int8_sym_scale": 2.5 / 127.0,
                },
                "min_val": -2.5,
                "max_val": 2.5,
            }
        }
    }

    count = quantizer.attach(model, stats_data, bits=8)
    assert count == 1
    assert len(quantizer.hooks) == 1

    x = torch.randn(2, 16)
    out_quant = model(x)
    assert out_quant.shape == (2, 8)

    quantizer.detach()
    assert len(quantizer.hooks) == 0
    out_clean = model(x)
    assert out_clean.shape == (2, 8)


def test_fake_quantizer_multi_format():
    """驗證 FakeQuantizer 支援傳入 .pt 檔案路徑與 .json 檔案路徑。"""
    model = SimpleNetwork()
    quantizer = FakeQuantizer(target_names=("q_proj",))

    dummy_stats = {
        "q_proj": {
            "clip_max_9999": 3.0,
            "scales": {"int8_sym_scale": 3.0 / 127.0},
        }
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 測試 .pt
        pt_path = os.path.join(tmp_dir, "test_calib.pt")
        torch.save({"layers": dummy_stats}, pt_path)
        assert quantizer.attach(model, pt_path, bits=8) == 1
        quantizer.detach()

        # 測試 .json (傳統格式)
        import json
        json_path = os.path.join(tmp_dir, "test_stats.json")
        with open(json_path, "w") as f:
            json.dump({"q_proj": {"min": -3.0, "max": 3.0}}, f)
        assert quantizer.attach(model, json_path, bits=8) == 1
        quantizer.detach()


if __name__ == "__main__":
    test_symmetric_fake_quantize_math()
    test_asymmetric_fake_quantize_math()
    test_fake_quantizer_attach_and_detach()
    test_fake_quantizer_multi_format()
    print("🎉 All fake_quant unit tests passed successfully!")
