"""
W8A8 Quantization Module for LLaDA Models

This module provides custom INT8 weight and INT8 activation quantization:
- Weights: Per-channel quantization (scale per output channel)
- Activations: Dynamic per-token quantization (scale computed at runtime)
- SmoothQuant: Integrated activation smoothing during quantization
"""

from .quantization_utils import (
    quantize_per_channel,
    quantize_per_token,
    dequantize,
)
from .quantized_linear import W8A8Linear, replace_linear_with_w8a8
from .smoothquant import (
    ActivationCollector,
    SmoothQuantConfig,
    compute_smooth_scales,
    compute_smooth_scales_multi,
    compute_all_smooth_scales,
    run_calibration,
)

__all__ = [
    "quantize_per_channel",
    "quantize_per_token", 
    "dequantize",
    "W8A8Linear",
    "replace_linear_with_w8a8",
    "ActivationCollector",
    "SmoothQuantConfig",
    "compute_smooth_scales",
    "compute_smooth_scales_multi",
    "compute_all_smooth_scales",
    "run_calibration",
]
