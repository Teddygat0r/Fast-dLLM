"""
DuQuant Implementation for LLaDA Models

This module provides a clean, modular implementation of DuQuant for LLaDA models.
DuQuant uses rotation and permutation transforms to smooth outliers before quantization.

The implementation consists of 3 core components:

1. simple_quantizer.py - Core DuQuant algorithm (rotation, permutation, quantization)
2. quant_linear.py - Quantized Linear layer wrapper
3. duquant_llada.py - Main pipeline orchestrating the full process

Usage:
    from duquant import apply_duquant_pipeline
    
    model, info = apply_duquant_pipeline(
        model_path="GSAI-ML/LLaDA-8B-Instruct",
        calibration_samples=128,
        n_bits=8,
        block_size=128,
    )

Algorithm Overview:
    1. Block-wise Rotation: Apply orthogonal rotation to smooth outliers
    2. Zigzag Permutation: Reorder channels to distribute outliers across blocks
    3. Quantization: Standard uniform affine quantization

Reference:
    DuQuant paper - https://arxiv.org/abs/2406.01721
"""

from .simple_quantizer import SimpleQuantizer, get_hadamard_matrix
from .quant_linear import DuQuantLinear, quantize_tensor_simple
from .duquant_llada import (
    apply_duquant_llada,
    apply_duquant_pipeline,
    get_llada_blocks,
    capture_layer_inputs,
)

__all__ = [
    # Core quantizer
    "SimpleQuantizer",
    "get_hadamard_matrix",
    # Quantized linear layer
    "DuQuantLinear",
    "quantize_tensor_simple",
    # Pipeline functions
    "apply_duquant_llada",
    "apply_duquant_pipeline",
    "get_llada_blocks",
    "capture_layer_inputs",
]
