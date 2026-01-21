"""
Quantization Utilities for W8A8 Quantization

Provides core functions for:
- Per-channel weight quantization (INT8)
- Dynamic per-token activation quantization (INT8)
- Dequantization back to floating point
"""

import torch
from typing import Tuple


def quantize_per_channel(
    weight: torch.Tensor,
    bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weight tensor using per-channel (per output channel) symmetric quantization.
    
    For a weight matrix of shape [out_features, in_features], we compute one scale
    per output channel (row), so the scale tensor has shape [out_features].
    
    Args:
        weight: Weight tensor of shape [out_features, in_features]
        bits: Number of bits for quantization (default: 8)
        
    Returns:
        weight_int8: Quantized weight tensor of shape [out_features, in_features] as int8
        scale: Per-channel scale tensor of shape [out_features]
    """
    assert bits == 8, "Only 8-bit quantization is currently supported"
    
    # Symmetric quantization: range is [-127, 127] for signed int8
    qmax = 127
    
    # Compute per-channel (per-row) max absolute value
    # weight shape: [out_features, in_features]
    # max over in_features (dim=1) -> scale shape: [out_features]
    max_abs = weight.abs().max(dim=1, keepdim=True).values
    
    # Compute scale: scale = max_abs / qmax
    # Clamp to avoid division by zero
    scale = (max_abs / qmax).clamp(min=1e-8)
    
    # Quantize: q = round(w / scale)
    weight_scaled = weight / scale
    weight_int8 = weight_scaled.round().clamp(-128, 127).to(torch.int8)
    
    # Remove the keepdim dimension from scale
    scale = scale.squeeze(1)
    
    return weight_int8, scale


def quantize_per_token(
    x: torch.Tensor,
    bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize activation tensor using dynamic per-token symmetric quantization.
    
    For an activation tensor of shape [batch, seq_len, hidden_dim] or 
    [num_tokens, hidden_dim], we compute one scale per token.
    
    Args:
        x: Activation tensor of shape [..., hidden_dim]
        bits: Number of bits for quantization (default: 8)
        
    Returns:
        x_int8: Quantized activation tensor as int8, same shape as input
        scale: Per-token scale tensor of shape [..., 1] or [num_tokens, 1]
    """
    assert bits == 8, "Only 8-bit quantization is currently supported"
    
    # Symmetric quantization range
    qmax = 127
    
    # Store original shape for reshaping later
    original_shape = x.shape
    
    # Flatten to [num_tokens, hidden_dim]
    x_flat = x.view(-1, x.shape[-1])
    
    # Compute per-token (per-row) max absolute value
    max_abs = x_flat.abs().max(dim=1, keepdim=True).values
    
    # Compute scale and clamp to avoid division by zero
    scale = (max_abs / qmax).clamp(min=1e-8)
    
    # Quantize
    x_scaled = x_flat / scale
    x_int8 = x_scaled.round().clamp(-128, 127).to(torch.int8)
    
    # Reshape back to original shape
    x_int8 = x_int8.view(original_shape)
    
    # Reshape scale to match the leading dimensions
    # e.g., for [batch, seq_len, hidden_dim] input, scale becomes [batch, seq_len, 1]
    scale_shape = list(original_shape[:-1]) + [1]
    scale = scale.view(scale_shape)
    
    return x_int8, scale


def dequantize(
    q_tensor: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Dequantize an INT8 tensor back to floating point.
    
    Args:
        q_tensor: Quantized INT8 tensor
        scale: Scale tensor (must be broadcastable to q_tensor shape)
        dtype: Target dtype for dequantized tensor
        
    Returns:
        Dequantized tensor in the specified dtype
    """
    return q_tensor.to(dtype) * scale.to(dtype)


def quantize_per_channel_asymmetric(
    weight: torch.Tensor,
    bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize weight tensor using per-channel asymmetric quantization.
    
    This is an alternative to symmetric quantization that can provide
    better accuracy when weights are not centered around zero.
    
    Args:
        weight: Weight tensor of shape [out_features, in_features]
        bits: Number of bits for quantization (default: 8)
        
    Returns:
        weight_int8: Quantized weight tensor as int8
        scale: Per-channel scale tensor
        zero_point: Per-channel zero point tensor
    """
    assert bits == 8, "Only 8-bit quantization is currently supported"
    
    qmin, qmax = -128, 127
    
    # Compute per-channel min and max
    w_min = weight.min(dim=1, keepdim=True).values
    w_max = weight.max(dim=1, keepdim=True).values
    
    # Compute scale and zero point
    scale = (w_max - w_min) / (qmax - qmin)
    scale = scale.clamp(min=1e-8)
    zero_point = (qmin - w_min / scale).round().clamp(qmin, qmax)
    
    # Quantize
    weight_int8 = (weight / scale + zero_point).round().clamp(qmin, qmax).to(torch.int8)
    
    return weight_int8, scale.squeeze(1), zero_point.squeeze(1).to(torch.int8)


def compute_quantization_error(
    original: torch.Tensor,
    quantized: torch.Tensor,
    scale: torch.Tensor,
) -> dict:
    """
    Compute quantization error metrics.
    
    Args:
        original: Original floating point tensor
        quantized: Quantized INT8 tensor
        scale: Scale used for quantization
        
    Returns:
        Dictionary with error metrics (MSE, max error, relative error)
    """
    # Dequantize for comparison
    reconstructed = dequantize(quantized, scale, dtype=original.dtype)
    
    # Compute errors
    diff = original - reconstructed
    mse = (diff ** 2).mean().item()
    max_error = diff.abs().max().item()
    
    # Relative error (avoid division by zero)
    orig_norm = original.abs().mean().item()
    relative_error = mse / (orig_norm ** 2 + 1e-8)
    
    return {
        "mse": mse,
        "max_error": max_error,
        "relative_error": relative_error,
    }
