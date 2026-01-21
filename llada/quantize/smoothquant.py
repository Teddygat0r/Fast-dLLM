"""
SmoothQuant Integration for W8A8 Quantization

This module provides integrated SmoothQuant functionality that applies
activation smoothing directly during quantization, rather than as a 
separate preprocessing step.

SmoothQuant migrates quantization difficulty from activations to weights
by scaling: activations are divided by scales, weights are multiplied by scales.
This is mathematically equivalent but makes activations easier to quantize.

Key insight: Instead of modifying LayerNorm weights, we store the smooth scales
in the quantized linear layer and apply them to inputs at runtime.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SmoothQuantConfig:
    """Configuration for SmoothQuant."""
    alpha: float = 0.5  # Migration strength: 0 = all to weights, 1 = all to activations
    min_scale: float = 1e-5  # Only clamp minimum to avoid division by zero


class ActivationCollector:
    """
    Collects per-channel activation maximum statistics for SmoothQuant calibration.
    
    For each Linear layer, records the maximum absolute value per input channel
    across all calibration samples.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.stats: Dict[str, torch.Tensor] = {}
    
    def _create_hook(self, name: str):
        """Create a forward hook that captures input activation statistics."""
        def hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            x = input[0].detach()
            # Compute max absolute value per channel (last dimension)
            # x shape: [batch, seq_len, hidden_dim] or similar
            max_val = x.abs().view(-1, x.shape[-1]).max(dim=0).values.cpu()
            
            if name not in self.stats:
                self.stats[name] = max_val
            else:
                # Take element-wise max across batches
                self.stats[name] = torch.max(self.stats[name], max_val)
        
        return hook
    
    def register_hooks(self) -> int:
        """
        Register forward hooks on all Linear layers.
        
        Returns:
            Number of hooks registered
        """
        self.clear_hooks()
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)
        
        return len(self.hooks)
    
    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear_stats(self) -> None:
        """Clear collected statistics."""
        self.stats = {}
    
    def get_stats(self) -> Dict[str, torch.Tensor]:
        """Get the collected activation statistics."""
        return self.stats


def compute_smooth_scales(
    act_max: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 0.5,
    min_scale: float = 1e-5,
) -> torch.Tensor:
    """
    Compute SmoothQuant scales for a single Linear layer.
    
    The scale balances quantization difficulty between activations and weights:
    - scale = (act_max ** alpha) / (weight_max ** (1 - alpha))
    
    After applying:
    - Activations are divided by scale (easier to quantize)
    - Weights are multiplied by scale (absorbs the difficulty)
    
    Args:
        act_max: Per-channel activation maximum [in_features]
        weight: Weight tensor [out_features, in_features]
        alpha: Migration strength (0.5 = balanced, lower = more aggressive)
        min_scale: Minimum allowed scale value (no maximum clamping)
        
    Returns:
        scales: Per-channel scales [in_features]
    """
    # Compute per-input-channel weight maximum
    # weight shape: [out_features, in_features]
    # We want max over out_features (dim 0) -> [in_features]
    weight_max = weight.abs().max(dim=0).values
    
    # Clamp to avoid numerical issues
    act_max = act_max.clamp(min=1e-5)
    weight_max = weight_max.clamp(min=1e-5)
    
    # Compute scales
    scales = (act_max ** alpha) / (weight_max ** (1 - alpha))
    
    # Clamp scales to avoid numerical issues (only minimum, no maximum)
    scales = scales.clamp(min=min_scale)
    
    return scales


def compute_smooth_scales_multi(
    act_max: torch.Tensor,
    weights: List[torch.Tensor],
    alpha: float = 0.5,
    min_scale: float = 1e-5,
) -> torch.Tensor:
    """
    Compute SmoothQuant scales for multiple Linear layers that share the same input.
    
    This is used for Q, K, V projections (share attn_norm output) and 
    ff_proj, up_proj (share ff_norm output).
    
    Args:
        act_max: Per-channel activation maximum [in_features]
        weights: List of weight tensors, each [out_features, in_features]
        alpha: Migration strength
        min_scale: Minimum allowed scale value (no maximum clamping)
        
    Returns:
        scales: Per-channel scales [in_features]
    """
    # Compute combined weight max across all layers
    weight_maxes = [w.abs().max(dim=0).values for w in weights]
    weight_max = weight_maxes[0]
    for wm in weight_maxes[1:]:
        weight_max = torch.max(weight_max, wm)
    
    # Clamp to avoid numerical issues
    act_max = act_max.clamp(min=1e-5)
    weight_max = weight_max.clamp(min=1e-5)
    
    # Compute scales
    scales = (act_max ** alpha) / (weight_max ** (1 - alpha))
    
    # Clamp scales to avoid numerical issues (only minimum, no maximum)
    scales = scales.clamp(min=min_scale)
    
    return scales


def compute_all_smooth_scales(
    model: nn.Module,
    activation_stats: Dict[str, torch.Tensor],
    alpha: float = 0.5,
    min_scale: float = 1e-5,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Compute SmoothQuant scales for Linear layers that receive LayerNorm output.
    
    IMPORTANT: SmoothQuant should ONLY be applied to layers whose inputs come
    directly from LayerNorm. This includes:
    - q_proj, k_proj, v_proj (input from attn_norm)
    - ff_proj, up_proj (input from ff_norm)
    
    It should NOT be applied to:
    - attn_out (input from attention computation)
    - ff_out (input from activation * up_proj)
    - Other layers not directly after LayerNorm
    
    Handles the special cases where multiple Linear layers share the same input:
    - q_proj, k_proj, v_proj share input from attn_norm
    - ff_proj, up_proj share input from ff_norm
    
    For these shared-input layers, we compute a single scale based on the
    combined weight statistics.
    
    Args:
        model: The model to compute scales for
        activation_stats: Per-layer activation statistics from ActivationCollector
        alpha: SmoothQuant migration strength
        min_scale: Minimum allowed scale value (no maximum clamping)
        device: Device to compute on
        
    Returns:
        Dictionary mapping layer names to their smooth scales
    """
    smooth_scales: Dict[str, torch.Tensor] = {}
    
    # Build a mapping of module name -> module
    name_to_module = {name: module for name, module in model.named_modules()}
    
    # Track which layers we've already processed (for shared-input groups)
    processed = set()
    
    # Only apply SmoothQuant to specific layer types that receive LayerNorm output
    # These are the layers where activation outliers cause quantization issues
    SMOOTHQUANT_LAYERS = ['q_proj', 'k_proj', 'v_proj', 'ff_proj', 'up_proj']
    
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name in processed:
            continue
        
        # Check if this layer should receive SmoothQuant
        layer_suffix = name.split('.')[-1] if '.' in name else name
        if layer_suffix not in SMOOTHQUANT_LAYERS:
            # Skip layers that shouldn't have SmoothQuant (attn_out, ff_out, etc.)
            continue
            
        # Check if this is part of a shared-input group
        # Pattern: block.q_proj, block.k_proj, block.v_proj
        # Pattern: block.ff_proj, block.up_proj
        
        if name.endswith('.q_proj'):
            # Attention group: q_proj, k_proj, v_proj
            base_name = name[:-7]  # Remove '.q_proj'
            k_name = f"{base_name}.k_proj"
            v_name = f"{base_name}.v_proj"
            
            if k_name in name_to_module and v_name in name_to_module:
                # Get activation stats (all three have same input)
                if name in activation_stats:
                    act_max = activation_stats[name].to(device)
                    
                    q_weight = name_to_module[name].weight
                    k_weight = name_to_module[k_name].weight
                    v_weight = name_to_module[v_name].weight
                    
                    scales = compute_smooth_scales_multi(
                        act_max,
                        [q_weight, k_weight, v_weight],
                        alpha=alpha,
                        min_scale=min_scale,
                    )
                    
                    # All three layers get the same scale
                    smooth_scales[name] = scales
                    smooth_scales[k_name] = scales
                    smooth_scales[v_name] = scales
                    
                    processed.add(name)
                    processed.add(k_name)
                    processed.add(v_name)
                    continue
        
        elif name.endswith('.ff_proj'):
            # MLP group: ff_proj, up_proj
            base_name = name[:-8]  # Remove '.ff_proj'
            up_name = f"{base_name}.up_proj"
            
            if up_name in name_to_module:
                # Get activation stats (both have same input)
                if name in activation_stats:
                    act_max = activation_stats[name].to(device)
                    
                    ff_weight = name_to_module[name].weight
                    up_weight = name_to_module[up_name].weight
                    
                    scales = compute_smooth_scales_multi(
                        act_max,
                        [ff_weight, up_weight],
                        alpha=alpha,
                        min_scale=min_scale,
                    )
                    
                    # Both layers get the same scale
                    smooth_scales[name] = scales
                    smooth_scales[up_name] = scales
                    
                    processed.add(name)
                    processed.add(up_name)
                    continue
        
        # Individual layer (k_proj, v_proj, up_proj may reach here if processed separately)
        # Only process if it's a valid SmoothQuant target and not already processed
        if name in activation_stats and name not in processed:
            act_max = activation_stats[name].to(device)
            scales = compute_smooth_scales(
                act_max,
                module.weight,
                alpha=alpha,
                min_scale=min_scale,
            )
            smooth_scales[name] = scales
            processed.add(name)
    
    return smooth_scales


def run_calibration(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run calibration to collect activation statistics.
    
    Args:
        model: The model to calibrate
        dataloader: DataLoader providing calibration samples
        device: Device to run on
        max_batches: Maximum number of batches to process (None = all)
        
    Returns:
        Dictionary of activation statistics per layer
    """
    collector = ActivationCollector(model)
    num_hooks = collector.register_hooks()
    print(f"  Registered {num_hooks} activation collection hooks")
    
    model.eval()
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(device)
                model(input_ids=input_ids)
                
                if (batch_idx + 1) % 10 == 0 or batch_idx == len(dataloader) - 1:
                    print(f"  Calibration: processed {batch_idx + 1} batches")
    finally:
        collector.clear_hooks()
    
    print(f"  Collected statistics for {len(collector.stats)} layers")
    return collector.stats
