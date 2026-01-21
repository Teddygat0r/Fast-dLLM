"""
Quantization Module for SmoothQuant

This module provides the quantized Linear layer implementation that performs
weight and activation quantization. It uses fake quantization (quantize then
dequantize) for simplicity and compatibility with PyTorch operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def quantize_tensor(
    x: torch.Tensor,
    n_bits: int = 8,
    per_channel: bool = False,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    Quantize tensor to n_bits with optional per-channel scaling.
    
    This implements fake quantization: the tensor is quantized to integer
    values and then dequantized back to floating point. This allows the
    quantized model to run with standard PyTorch operations.
    
    Args:
        x: Input tensor to quantize.
        n_bits: Number of bits for quantization (typically 8 or 4).
        per_channel: If True, compute separate scale/zero-point per output
                    channel (dim 0). If False, use per-tensor quantization.
        symmetric: If True, use symmetric quantization (zero_point = 0).
                  If False, use asymmetric quantization.
    
    Returns:
        Dequantized tensor (same dtype as input, but with quantization error).
    
    Example:
        >>> weight = torch.randn(512, 768)
        >>> weight_q = quantize_tensor(weight, n_bits=8, per_channel=True)
        >>> print(weight_q.shape)
        torch.Size([512, 768])
    """
    if n_bits >= 16:
        # No quantization needed
        return x
    
    # Determine reduction dimensions for computing min/max
    if per_channel:
        # Per-channel: reduce all dims except dim 0
        reduce_dims = tuple(range(1, x.dim()))
    else:
        # Per-tensor: reduce all dims
        reduce_dims = None
    
    # Compute min/max values
    if reduce_dims:
        x_min = x.amin(dim=reduce_dims, keepdim=True)
        x_max = x.amax(dim=reduce_dims, keepdim=True)
    else:
        x_min = x.min()
        x_max = x.max()
    
    # Compute scale and zero point
    if symmetric:
        # Symmetric quantization: range is [-max_abs, max_abs]
        abs_max = torch.max(x_max.abs(), x_min.abs())
        qmin = -(2 ** (n_bits - 1))
        qmax = 2 ** (n_bits - 1) - 1
        scale = abs_max / qmax
        zero_point = torch.zeros_like(scale)
    else:
        # Asymmetric quantization: range is [min, max]
        qmin = 0
        qmax = 2 ** n_bits - 1
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = qmin - x_min / scale
    
    # Clamp scale to avoid division by zero
    scale = scale.clamp(min=1e-8)
    
    # Quantize: x_int = round(x / scale + zero_point)
    x_int = (x / scale + zero_point).round().clamp(qmin, qmax)
    
    # Dequantize: x_dequant = (x_int - zero_point) * scale
    x_dequant = (x_int - zero_point) * scale
    
    return x_dequant


class QuantLinear(nn.Module):
    """
    Quantized Linear layer with weight and activation quantization.
    
    This layer replaces nn.Linear and performs:
    1. Weight quantization (done once, stored as quantized values)
    2. Activation quantization (done at each forward pass)
    
    Attributes:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        w_bits: Number of bits for weight quantization.
        a_bits: Number of bits for activation quantization.
        weight: Quantized weight tensor (stored as buffer).
        bias: Bias tensor (if present).
        weight_quantized: Flag indicating if weights have been quantized.
    
    Example:
        >>> linear = nn.Linear(768, 512)
        >>> quant_linear = QuantLinear(linear, w_bits=8, a_bits=8)
        >>> quant_linear.quantize_weight()
        >>> output = quant_linear(torch.randn(2, 10, 768))
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        w_bits: int = 8,
        a_bits: int = 8,
    ):
        """
        Initialize QuantLinear from an existing nn.Linear.
        
        Args:
            original_linear: The nn.Linear module to replace.
            w_bits: Number of bits for weight quantization.
            a_bits: Number of bits for activation quantization.
        """
        super().__init__()
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.w_bits = w_bits
        self.a_bits = a_bits
        
        # Copy weights as buffer (not parameter, since they're frozen after quantization)
        self.register_buffer('weight', original_linear.weight.clone())
        
        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.clone())
        else:
            self.register_buffer('bias', None)
        
        self.weight_quantized = False
    
    def quantize_weight(self) -> None:
        """
        Pre-quantize weights. Call this once after smoothing.
        
        Weights are quantized per-channel (each output channel has its own scale)
        using symmetric quantization for better hardware efficiency.
        """
        if self.weight_quantized:
            return
        
        self.weight.data = quantize_tensor(
            self.weight,
            n_bits=self.w_bits,
            per_channel=True,  # Per-channel for weights (better accuracy)
            symmetric=True,     # Symmetric for weights (better efficiency)
        )
        self.weight_quantized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with activation quantization.
        
        Args:
            x: Input tensor of shape (..., in_features).
        
        Returns:
            Output tensor of shape (..., out_features).
        """
        # Quantize activations (per-tensor, asymmetric for better accuracy)
        if self.a_bits < 16:
            x = quantize_tensor(
                x,
                n_bits=self.a_bits,
                per_channel=False,  # Per-tensor for activations
                symmetric=False,    # Asymmetric for activations
            )
        
        # Use pre-quantized weights
        return F.linear(x, self.weight, self.bias)
    
    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, w_bits={self.w_bits}, a_bits={self.a_bits}, '
            f'quantized={self.weight_quantized}'
        )


def replace_linear_with_quant(
    model: nn.Module,
    w_bits: int = 8,
    a_bits: int = 8,
    skip_layers: Optional[list] = None,
) -> int:
    """
    Replace all nn.Linear modules in a model with QuantLinear.
    
    Args:
        model: The model to modify.
        w_bits: Number of bits for weight quantization.
        a_bits: Number of bits for activation quantization.
        skip_layers: List of layer name patterns to skip (e.g., ["lm_head"]).
    
    Returns:
        Number of layers replaced.
    
    Note:
        This function modifies the model in-place.
    """
    skip_layers = skip_layers or []
    replaced = 0
    
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        
        # Check if this layer should be skipped
        should_skip = any(skip in name for skip in skip_layers)
        if should_skip:
            continue
        
        # Get parent module and child name
        parts = name.rsplit('.', 1)
        if len(parts) == 1:
            parent = model
            child_name = parts[0]
        else:
            parent_name, child_name = parts
            parent = model.get_submodule(parent_name)
        
        # Replace with QuantLinear
        quant_linear = QuantLinear(module, w_bits=w_bits, a_bits=a_bits)
        quant_linear.quantize_weight()
        setattr(parent, child_name, quant_linear)
        replaced += 1
    
    return replaced
