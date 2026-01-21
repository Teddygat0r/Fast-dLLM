"""
Minimal SmoothQuant Implementation for LLaDA Models

This module provides a clean, modular implementation of SmoothQuant for LLaDA models.
It consists of 4 core components:

1. calibrate.py - Collect activation scales during calibration
2. smooth.py - Apply smooth transformations to LayerNorm/Linear pairs
3. quantize.py - Quantized Linear layer with weight/activation quantization
4. smoothquant_llada.py - Main pipeline orchestrating the full process

Usage:
    from smoothquant import apply_smoothquant_pipeline
    
    model, act_scales = apply_smoothquant_pipeline(
        model_path="GSAI-ML/LLaDA-8B-Instruct",
        calibration_samples=128,
        alpha=0.5,
        w_bits=8,
        a_bits=8,
    )
"""

from .calibrate import collect_act_scales
from .smooth import compute_smooth_scale, compute_smooth_scale_multi, smooth_ln_fc, smooth_fc_fc
from .quantize import quantize_tensor, QuantLinear
from .smoothquant_llada import (
    apply_smoothquant_llada,
    replace_with_quant_linear,
    apply_smoothquant_pipeline,
)

__all__ = [
    # Calibration
    "collect_act_scales",
    # Smoothing
    "compute_smooth_scale",
    "compute_smooth_scale_multi",
    "smooth_ln_fc", 
    "smooth_fc_fc",
    # Quantization
    "quantize_tensor",
    "QuantLinear",
    # Pipeline
    "apply_smoothquant_llada",
    "replace_with_quant_linear",
    "apply_smoothquant_pipeline",
]
