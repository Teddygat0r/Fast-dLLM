"""
Calibration Module for SmoothQuant

This module collects activation statistics during a calibration pass.
For each Linear layer, it tracks the maximum absolute activation value
per input channel across all calibration samples.

These statistics are then used to compute smooth scales that balance
quantization difficulty between activations and weights.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm


@torch.no_grad()
def collect_act_scales(
    model: nn.Module,
    dataloader,
    num_samples: int = 128,
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Collect maximum absolute activation value per channel for each Linear layer.
    
    This function registers forward hooks on all Linear layers to capture
    input activation statistics. For each layer, it tracks the maximum
    absolute value per input channel (the last dimension of the input tensor).
    
    Args:
        model: The model to collect activation scales from.
        dataloader: DataLoader providing calibration samples. Each batch should
                   have an 'input_ids' key or be a tuple where the first element
                   is input_ids.
        num_samples: Maximum number of samples to process for calibration.
        device: Device to run calibration on. If None, uses model's device.
    
    Returns:
        Dictionary mapping layer names (e.g., "model.transformer.blocks.0.q_proj")
        to torch.Tensor of shape (in_features,) containing the per-channel
        maximum activation values.
    
    Example:
        >>> from torch.utils.data import DataLoader
        >>> act_scales = collect_act_scales(model, calib_loader, num_samples=128)
        >>> print(act_scales["model.transformer.blocks.0.q_proj"].shape)
        torch.Size([4096])
    """
    model.eval()
    
    # Determine device
    if device is None:
        device = next(model.parameters()).device
    
    act_scales: Dict[str, torch.Tensor] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []
    
    def create_hook(name: str):
        """Create a forward hook that captures input activation statistics."""
        def hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            x = input[0] if isinstance(input, tuple) else input
            
            # Handle different input shapes
            # Typical shapes: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
            x = x.detach()
            
            # Flatten all dimensions except the last (channel dimension)
            # Shape: (batch * seq_len, hidden_dim) or similar
            x_flat = x.view(-1, x.shape[-1])
            
            # Compute max absolute value per channel
            channel_max = x_flat.abs().max(dim=0)[0].float().cpu()
            
            # Update running maximum
            if name in act_scales:
                act_scales[name] = torch.max(act_scales[name], channel_max)
            else:
                act_scales[name] = channel_max
        
        return hook
    
    # Register hooks on all Linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = module.register_forward_hook(create_hook(name))
            hooks.append(hook)
    
    print(f"  Registered hooks on {len(hooks)} Linear layers")
    
    # Run calibration
    print(f"Running calibration on batch size {dataloader.batch_size}")
    try:
        sample_count = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_samples, desc="  Calibration")):
            if sample_count >= num_samples:
                break
            
            # Handle different batch formats
            if isinstance(batch, dict):
                input_ids = batch['input_ids']
            elif isinstance(batch, (tuple, list)):
                input_ids = batch[0]
            else:
                input_ids = batch
            
            # Move to device and run forward pass
            input_ids = input_ids.to(device)
            model(input_ids=input_ids)
            
            sample_count += input_ids.shape[0]
    
    finally:
        # Always cleanup hooks
        for hook in hooks:
            hook.remove()
    
    print(f"  Collected activation scales for {len(act_scales)} layers")
    
    return act_scales


def save_act_scales(act_scales: Dict[str, torch.Tensor], path: str) -> None:
    """
    Save activation scales to a file.
    
    Args:
        act_scales: Dictionary of activation scales from collect_act_scales.
        path: Path to save the scales (typically .pt file).
    """
    torch.save(act_scales, path)
    print(f"  Saved activation scales to: {path}")


def load_act_scales(path: str) -> Dict[str, torch.Tensor]:
    """
    Load activation scales from a file.
    
    Args:
        path: Path to the saved scales file.
    
    Returns:
        Dictionary of activation scales.
    """
    act_scales = torch.load(path, weights_only=True)
    print(f"  Loaded activation scales from: {path} ({len(act_scales)} layers)")
    return act_scales
