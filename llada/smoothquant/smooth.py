"""
Smooth Transformations Module for SmoothQuant

This module applies smooth transformations to LayerNorm + Linear pairs.
The key insight of SmoothQuant is that we can mathematically equivalently
transform the computation to make activations easier to quantize:

    Y = LN(X) @ W^T = (LN(X) / s) @ (s * W)^T

Where s is the smooth scale that balances quantization difficulty between
activations and weights.
"""

import torch
import torch.nn as nn
from typing import List, Optional


@torch.no_grad()
def compute_smooth_scale(
    act_scale: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 0.5,
    min_scale: float = 1e-5,
    max_scale: float = 1e5,
) -> torch.Tensor:
    """
    Compute the smooth scale that balances quantization difficulty.
    
    The scale is computed as:
        scale = act_max^alpha / weight_max^(1-alpha)
    
    Where:
        - act_max is the per-channel maximum activation value
        - weight_max is the per-input-channel maximum weight value
        - alpha controls the migration strength (0.5 = balanced)
    
    Args:
        act_scale: Per-channel activation maximum, shape (in_features,).
        weight: Weight tensor, shape (out_features, in_features).
        alpha: Migration strength. Lower values (e.g., 0.3) are more conservative,
               higher values push more difficulty to weights. Default 0.5.
        min_scale: Minimum allowed scale to avoid division by zero.
        max_scale: Maximum allowed scale to avoid numerical instability.
    
    Returns:
        Per-channel smooth scale, shape (in_features,).
    """
    # Clamp activation scale to avoid numerical issues
    act_scale = act_scale.clamp(min=min_scale)
    
    # Compute per-input-channel weight maximum
    # weight shape: (out_features, in_features)
    # We want max over output features (dim 0) -> shape (in_features,)
    weight_max = weight.abs().max(dim=0)[0].clamp(min=min_scale)
    
    # Compute scale: act^alpha / weight^(1-alpha)
    scale = act_scale.pow(alpha) / weight_max.pow(1 - alpha)
    
    # Clamp to reasonable range
    scale = scale.clamp(min=min_scale, max=max_scale)
    
    return scale


@torch.no_grad()
def compute_smooth_scale_multi(
    act_scale: torch.Tensor,
    weights: List[torch.Tensor],
    alpha: float = 0.5,
    min_scale: float = 1e-5,
    max_scale: float = 1e5,
) -> torch.Tensor:
    """
    Compute smooth scale for multiple Linear layers that share the same input.
    
    This is used when multiple layers receive the same LayerNorm output:
        - Q, K, V projections share input from attn_norm
        - ff_proj, up_proj share input from ff_norm
    
    The weight_max is computed as the element-wise maximum across all weights.
    
    Args:
        act_scale: Per-channel activation maximum, shape (in_features,).
        weights: List of weight tensors, each shape (out_features, in_features).
        alpha: Migration strength. Default 0.5.
        min_scale: Minimum allowed scale.
        max_scale: Maximum allowed scale.
    
    Returns:
        Per-channel smooth scale, shape (in_features,).
    """
    act_scale = act_scale.clamp(min=min_scale)
    
    # Compute combined weight max across all layers
    weight_maxes = [w.abs().max(dim=0)[0] for w in weights]
    weight_max = weight_maxes[0]
    for wm in weight_maxes[1:]:
        weight_max = torch.max(weight_max, wm)
    weight_max = weight_max.clamp(min=min_scale)
    
    # Compute scale
    scale = act_scale.pow(alpha) / weight_max.pow(1 - alpha)
    scale = scale.clamp(min=min_scale, max=max_scale)
    
    return scale


@torch.no_grad()
def smooth_ln_fc(
    ln: nn.Module,
    fcs: List[nn.Linear],
    scale: torch.Tensor,
) -> None:
    """
    Apply smoothing between LayerNorm and following Linear layers.
    
    The transformation is:
        Y = LN(X) @ W^T = (LN(X) / s) @ (s * W)^T
    
    This modifies:
        - LN weights: divided by scale
        - LN bias (if present): divided by scale  
        - FC weights: multiplied by scale
    
    Args:
        ln: LayerNorm module (must have 'weight' attribute).
        fcs: List of Linear modules that receive LN output.
        scale: Per-channel smooth scale, shape (normalized_shape,).
    
    Note:
        This function modifies the modules in-place.
    """
    device = ln.weight.device
    dtype = ln.weight.dtype
    scale = scale.to(device=device, dtype=dtype)
    
    # Modify LayerNorm: divide by scale
    ln.weight.div_(scale)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scale)
    
    # Modify Linear layers: multiply by scale
    # Weight shape: (out_features, in_features)
    # We multiply along the in_features dimension
    scale_view = scale.view(1, -1)
    for fc in fcs:
        fc.weight.mul_(scale_view.to(device=fc.weight.device, dtype=fc.weight.dtype))


@torch.no_grad()
def smooth_fc_fc(
    fc1: nn.Linear,
    fc2: nn.Linear,
    scale: torch.Tensor,
) -> None:
    """
    Apply smoothing between two consecutive Linear layers.
    
    This is used for:
        - V_proj -> attn_out (attention output)
        - up_proj -> ff_out (MLP output after SwiGLU)
    
    The transformation is:
        Y = (X @ W1^T) @ W2^T = (X @ W1^T / s) @ (s * W2)^T
    
    This modifies:
        - fc1 weights: divided by scale (along output dimension)
        - fc1 bias (if present): divided by scale
        - fc2 weights: multiplied by scale (along input dimension)
    
    Args:
        fc1: First Linear module (its output is divided by scale).
        fc2: Second Linear module (its input is multiplied by scale).
        scale: Per-channel smooth scale, shape (fc1.out_features,).
    
    Note:
        This function modifies the modules in-place.
    """
    device = fc1.weight.device
    dtype = fc1.weight.dtype
    scale = scale.to(device=device, dtype=dtype)
    
    # Modify fc1: divide output by scale
    # fc1.weight shape: (out_features, in_features)
    # We divide along the out_features dimension (dim 0)
    fc1.weight.div_(scale.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scale)
    
    # Modify fc2: multiply input by scale
    # fc2.weight shape: (out_features, in_features)
    # We multiply along the in_features dimension (dim 1)
    scale_view = scale.view(1, -1)
    fc2.weight.mul_(scale_view.to(device=fc2.weight.device, dtype=fc2.weight.dtype))
