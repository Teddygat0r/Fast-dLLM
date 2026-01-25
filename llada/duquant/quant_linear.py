"""
Quantized Linear Layer for DuQuant

This module provides a quantized Linear layer wrapper that uses DuQuant
transformations (rotation + permutation) before quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .simple_quantizer import SimpleQuantizer


class DuQuantLinear(nn.Module):
    """
    Quantized Linear layer using DuQuant transformations.
    
    This layer wraps an existing nn.Linear and applies:
    1. Optional activation quantization
    2. DuQuant weight transformation (rotation + permutation)
    3. Weight quantization in the transformed space
    4. Inverse transformation back to original space
    
    Args:
        original_linear: The nn.Linear module to replace.
        weight_quantizer: SimpleQuantizer for weight quantization.
        act_quantizer: Optional SimpleQuantizer for activation quantization.
        w_bits: Weight quantization bits (used if weight_quantizer not provided).
        a_bits: Activation quantization bits (if > 0 and act_quantizer not provided).
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        weight_quantizer: Optional[SimpleQuantizer] = None,
        act_quantizer: Optional[SimpleQuantizer] = None,
        w_bits: int = 8,
        a_bits: int = 8,
    ):
        super().__init__()
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.w_bits = w_bits
        self.a_bits = a_bits
        
        # Copy weights as buffer (frozen after quantization setup)
        self.register_buffer('weight', original_linear.weight.clone())
        
        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.clone())
        else:
            self.register_buffer('bias', None)
        
        # Set up quantizers
        self.weight_quantizer = weight_quantizer
        self.act_quantizer = act_quantizer
        
        # Cache for quantized weight (optional optimization)
        self.register_buffer('_cached_quant_weight', None)
        self._use_cache = False
    
    def quantize_weight(self, use_cache: bool = True) -> None:
        """
        Pre-quantize the weight tensor and optionally cache it.
        
        Call this after calibration to avoid repeated quantization during inference.
        
        Args:
            use_cache: If True, cache the quantized weight for faster inference.
        """
        self._use_cache = use_cache
        if use_cache and self.weight_quantizer is not None:
            with torch.no_grad():
                self._cached_quant_weight = self.weight_quantizer(self.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with activation and weight quantization.
        
        Args:
            x: Input tensor of shape (..., in_features).
        
        Returns:
            Output tensor of shape (..., out_features).
        """
        # Quantize activations if configured
        if self.act_quantizer is not None and self.a_bits < 16:
            x = self._quantize_activation(x)
        
        # Get quantized weight
        if self._use_cache and self._cached_quant_weight is not None:
            weight = self._cached_quant_weight
        elif self.weight_quantizer is not None:
            weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight
        
        return F.linear(x, weight, self.bias)
    
    def _quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize activation tensor.
        
        For activations, we use simpler per-tensor quantization without
        the full DuQuant rotation/permutation transforms.
        
        Args:
            x: Activation tensor.
        
        Returns:
            Quantized activation tensor.
        """
        if self.act_quantizer is not None:
            # If we have a calibrated quantizer, use it
            # For activations, we typically use a simpler approach
            return self._simple_quant_activation(x)
        return x
    
    def _simple_quant_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple per-tensor asymmetric quantization for activations.
        
        Args:
            x: Activation tensor.
        
        Returns:
            Quantized activation tensor.
        """
        if self.a_bits >= 16:
            return x
        
        qmin = 0
        qmax = 2 ** self.a_bits - 1
        
        x_min = x.min()
        x_max = x.max()
        
        scale = (x_max - x_min) / (qmax - qmin)
        scale = scale.clamp(min=1e-8)
        zero_point = qmin - x_min / scale
        
        x_int = (x / scale + zero_point).round().clamp(qmin, qmax)
        x_dequant = (x_int - zero_point) * scale
        
        return x_dequant
    
    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, w_bits={self.w_bits}, a_bits={self.a_bits}, '
            f'cached={self._use_cache and self._cached_quant_weight is not None}'
        )


def quantize_tensor_simple(
    x: torch.Tensor,
    n_bits: int = 8,
    per_channel: bool = False,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    Simple quantization helper function (without DuQuant transforms).
    
    This is useful for quick quantization without calibration.
    
    Args:
        x: Input tensor to quantize.
        n_bits: Number of bits for quantization.
        per_channel: If True, compute separate scale per output channel.
        symmetric: If True, use symmetric quantization.
    
    Returns:
        Fake-quantized tensor.
    """
    if n_bits >= 16:
        return x
    
    if per_channel:
        reduce_dims = tuple(range(1, x.dim()))
    else:
        reduce_dims = None
    
    if reduce_dims:
        x_min = x.amin(dim=reduce_dims, keepdim=True)
        x_max = x.amax(dim=reduce_dims, keepdim=True)
    else:
        x_min = x.min()
        x_max = x.max()
    
    if symmetric:
        abs_max = torch.max(x_max.abs(), x_min.abs())
        qmin = -(2 ** (n_bits - 1))
        qmax = 2 ** (n_bits - 1) - 1
        scale = abs_max / qmax
        zero_point = torch.zeros_like(scale)
    else:
        qmin = 0
        qmax = 2 ** n_bits - 1
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = qmin - x_min / scale
    
    scale = scale.clamp(min=1e-8)
    x_int = (x / scale + zero_point).round().clamp(qmin, qmax)
    x_dequant = (x_int - zero_point) * scale
    
    return x_dequant
