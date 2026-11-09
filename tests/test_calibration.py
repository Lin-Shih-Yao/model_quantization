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
    """Lightweight dummy Transformer Decoder layer for testing."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 2)

    def forward(self, x):
        q = self.q_proj(x)
        g = self.gate_proj(q)
        return g


def test_layer_observer_basic():
    """Tests LayerObserver extrema and scale computation correctness."""
    obs = LayerObserver("test_layer", clip_quantile=0.99)
    dummy_input = torch.tensor([[[-2.0, 1.0, 4.0], [0.5, -3.0, 1.5]]])

    # Guard check before update
    uninit_obs = LayerObserver("uninit")
    assert uninit_obs.compute_channel_sym_scale(bits=8) is None

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

    # Test per-channel symmetric scale computation
    ch_sym_s = obs.compute_channel_sym_scale(bits=8)
    assert ch_sym_s is not None
    assert ch_sym_s.shape == (3,)
    expected_ch_scale = torch.tensor([2.0 / 127.0, 3.0 / 127.0, 4.0 / 127.0])
    assert torch.allclose(ch_sym_s, expected_ch_scale, atol=1e-5)

    # Verify to_dict includes channel scale
    dict_output = obs.to_dict()
    assert "int8_channel_sym_scale" in dict_output["scales"]
    assert torch.equal(dict_output["scales"]["int8_channel_sym_scale"], ch_sym_s)


def test_hook_lifecycle():
    """Tests forward hook registration, activation interception, and removal."""
    model = DummyDecoderLayer(hidden_dim=32)
    targets = find_target_linear_modules(model)

    assert "q_proj" in targets
    assert "gate_proj" in targets

    observers, hooks = attach_observers(targets)
    assert len(hooks) == 2

    dummy_x = torch.randn(1, 16, 32)
    model(dummy_x)

    assert observers["q_proj"].sample_count == 1
    assert observers["gate_proj"].sample_count == 1

    # Verify hooks are cleanly removed
    remove_all_hooks(hooks)
    model(dummy_x)
    assert observers["q_proj"].sample_count == 1


def test_calibration_profiler():
    """Tests CalibrationProfiler adapter collecting statistics and saving JSON."""
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


def test_unified_calibration_stats():
    """Tests unified collect_calibration_stats and save_calibration_stats_json."""
    from src.quantization.calibration import (
        collect_calibration_stats,
        save_calibration_stats_json,
    )
    import tempfile

    model = DummyDecoderLayer(hidden_dim=16)
    calib_tensor = torch.randn(2, 4, 16)

    stats = collect_calibration_stats(model, calib_tensor, device="cpu")
    assert "q_proj" in stats
    assert "scales" in stats["q_proj"]
    assert "int8_sym_scale" in stats["q_proj"]["scales"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = save_calibration_stats_json(
            stats, "TestModel", "c4", 2, node_name="Node_B", custom_dir=tmp_dir
        )
        assert os.path.exists(json_path)


if __name__ == "__main__":
    test_layer_observer_basic()
    test_hook_lifecycle()
    test_calibration_profiler()
    test_unified_calibration_stats()
    print("🎉 All calibration unit tests passed successfully!")
