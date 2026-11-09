"""
Layer Observer Module
Hooks linear layers during forward pass to track activation min/max, 99.99% quantiles,
channel-wise maximums, and calculates INT8/INT4 scales and zero points.
"""

import torch


class LayerObserver:
    """
    Dedicated observer monitoring input activation tensors of a single neural network layer.
    Maintains clean, single-responsibility methods under 20 lines each.
    """

    def __init__(self, name: str, clip_quantile: float = 0.9999):
        self.name = name
        self.clip_quantile = clip_quantile
        self.min_val: float = float("inf")
        self.max_val: float = float("-inf")
        self.abs_max: float = 0.0
        self.clip_max: float = 0.0
        self.channel_abs_max: torch.Tensor | None = None
        self.sample_count: int = 0

    def hook_fn(self, module, inputs, outputs):
        """Forward Hook entrypoint: intercepts input tensor X and triggers statistics update."""
        if not inputs or inputs[0] is None:
            return
        x = inputs[0].detach()
        self.update(x)

    def update(self, tensor: torch.Tensor):
        """Updates extrema, clipping quantiles, and channel maximums for the current feature batch."""
        feat = tensor.float()
        self.min_val = min(self.min_val, feat.min().item())
        self.max_val = max(self.max_val, feat.max().item())

        cur_abs_max = feat.abs().max().item()
        self.abs_max = max(self.abs_max, cur_abs_max)

        # 99.99% quantile clipping to isolate extreme deep-layer activation spikes
        q_val = torch.quantile(feat.abs(), self.clip_quantile).item()
        self.clip_max = max(self.clip_max, q_val)

        # Track channel maximums along the feature dimension (last dimension)
        ch_dims = list(range(feat.ndim - 1))
        batch_ch_max = feat.abs().amax(dim=ch_dims).cpu()
        if self.channel_abs_max is None:
            self.channel_abs_max = batch_ch_max
        else:
            self.channel_abs_max = torch.maximum(self.channel_abs_max, batch_ch_max)

        self.sample_count += 1

    def compute_sym_scale(self, bits: int = 8) -> float:
        """Computes symmetric scale: S = clip_max / (2^(b-1) - 1)."""
        q_max = (1 << (bits - 1)) - 1
        safe_max = max(self.clip_max, 1e-8)
        return safe_max / q_max

    def compute_asym_scale_zp(self, bits: int = 8) -> tuple[float, int]:
        """Computes asymmetric scale and zero point."""
        q_max = (1 << bits) - 1
        val_range = max(self.max_val - self.min_val, 1e-8)
        scale = val_range / q_max
        zero_point = int(round(-self.min_val / scale))
        zero_point = max(0, min(q_max, zero_point))
        return scale, zero_point

    def compute_channel_sym_scale(self, bits: int = 8) -> torch.Tensor | None:
        """Computes per-channel symmetric scale: S_c = channel_abs_max / (2^(b-1) - 1)."""
        if self.channel_abs_max is None:
            return None
        q_max = (1 << (bits - 1)) - 1
        return torch.clamp(self.channel_abs_max, min=1e-8) / q_max

    def to_dict(self) -> dict:
        """Exports layer statistics and precision scales into a standardized dictionary."""
        int8_s_scale = self.compute_sym_scale(bits=8)
        int4_s_scale = self.compute_sym_scale(bits=4)
        int8_a_scale, int8_a_zp = self.compute_asym_scale_zp(bits=8)
        int4_a_scale, int4_a_zp = self.compute_asym_scale_zp(bits=4)
        int8_ch_s_scale = self.compute_channel_sym_scale(bits=8)

        return {
            "layer_name": self.name,
            "sample_count": self.sample_count,
            "min": self.min_val,
            "max": self.max_val,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "abs_max": self.abs_max,
            "clip_max_9999": self.clip_max,
            "channel_abs_max": self.channel_abs_max,
            "scales": {
                "int8_sym_scale": int8_s_scale,
                "int4_sym_scale": int4_s_scale,
                "int8_asym_scale": int8_a_scale,
                "int8_asym_zp": int8_a_zp,
                "int4_asym_scale": int4_a_scale,
                "int4_asym_zp": int4_a_zp,
                "int8_channel_sym_scale": int8_ch_s_scale,
            },
        }
