"""
Unit Tests for Quantization Calibration Pipeline
Verifies LayerObserver statistical correctness and calibration hook workflows.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from src.quantization.observer import LayerObserver
from src.quantization.calibration import (
    find_target_linear_modules,
    attach_observers,
    remove_all_hooks,
)


class DummyDecoderLayer(nn.Module):
    """用於單元測試的輕量模擬 Transformer Decoder 層。"""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 2)

    def forward(self, x):
        q = self.q_proj(x)
        g = self.gate_proj(q)
        return g


def test_layer_observer_basic():
    """測試 LayerObserver 的極值與 Scale 計算正確性。"""
    obs = LayerObserver("test_layer", clip_quantile=0.99)
    dummy_input = torch.tensor([[[-2.0, 1.0, 4.0], [0.5, -3.0, 1.5]]])

    obs.update(dummy_input)

    assert obs.min_val == -3.0
    assert obs.max_val == 4.0
    assert obs.abs_max == 4.0
    assert obs.channel_abs_max.shape == (3,)
    assert obs.channel_abs_max.tolist() == [2.0, 3.0, 4.0]

    sym_s = obs.compute_sym_scale(bits=8)
    assert sym_s > 0.0

    asym_s, asym_zp = obs.compute_asym_scale_zp(bits=8)
    assert asym_s > 0.0
    assert 0 <= asym_zp <= 255


def test_hook_lifecycle():
    """測試 Hook 註冊、前向攔截與安全卸載流程。"""
    model = DummyDecoderLayer(hidden_dim=32)
    targets = find_target_linear_modules(model)

    assert "q_proj" in targets
    assert "gate_proj" in targets

    observers, hooks = attach_observers(targets)
    assert len(hooks) == 2

    # 執行一次假輸入
    dummy_x = torch.randn(1, 16, 32)
    model(dummy_x)

    # 驗證統計有被記錄
    assert observers["q_proj"].sample_count == 1
    assert observers["gate_proj"].sample_count == 1

    # 驗證卸載後不再更新
    remove_all_hooks(hooks)
    model(dummy_x)
    assert observers["q_proj"].sample_count == 1


def test_calibration_profiler():
    """測試輕量 CalibrationProfiler 收集純量極值與存檔。"""
    from src.quantization.profiler import CalibrationProfiler, save_stats_to_json
    import tempfile

    model = DummyDecoderLayer(hidden_dim=16)
    profiler = CalibrationProfiler()
    calib_tensor = torch.randn(4, 8, 16)

    stats = profiler.collect_stats(model, calib_tensor, device="cpu")
    assert "q_proj" in stats
    assert "gate_proj" in stats
    assert stats["q_proj"]["min"] < stats["q_proj"]["max"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = save_stats_to_json(
            stats, "TestModel", "wikitext2", 4, node_name="Node_A", custom_dir=tmp_dir
        )
        assert os.path.exists(json_path)


if __name__ == "__main__":
    test_layer_observer_basic()
    test_hook_lifecycle()
    test_calibration_profiler()
    print("🎉 All calibration unit tests passed successfully!")
