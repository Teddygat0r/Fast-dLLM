"""
Main SmoothQuant Pipeline for LLaDA Models

This module orchestrates the complete SmoothQuant process:
1. Load model and tokenizer
2. Collect activation scales during calibration
3. Apply smooth transformations to LayerNorm/Linear pairs
4. Replace Linear layers with quantized versions

The pipeline is specifically designed for LLaDA model architecture.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from torch.utils.data import DataLoader

from .calibrate import collect_act_scales, save_act_scales, load_act_scales
from .smooth import compute_smooth_scale, compute_smooth_scale_multi, smooth_ln_fc, smooth_fc_fc
from .quantize import QuantLinear, replace_linear_with_quant


def get_llada_blocks(model: nn.Module):
    """
    Get the list of transformer blocks from a LLaDA model.
    
    Handles both 'blocks' and 'block_groups' model structures.
    
    Args:
        model: The LLaDA model.
    
    Returns:
        List of transformer blocks.
    """
    transformer = model.model.transformer
    
    if hasattr(transformer, 'blocks'):
        return list(transformer.blocks)
    elif hasattr(transformer, 'block_groups'):
        # Flatten block groups into a list of blocks
        blocks = []
        for group in transformer.block_groups:
            blocks.extend(list(group))
        return blocks
    else:
        raise ValueError("Could not find 'blocks' or 'block_groups' in model structure")


@torch.no_grad()
def apply_smoothquant_llada(
    model: nn.Module,
    act_scales: Dict[str, torch.Tensor],
    alpha: float = 0.5,
    device: Optional[str] = None,
) -> nn.Module:
    """
    Apply SmoothQuant transformations to all layers of a LLaDA model.
    
    For each transformer block, this applies smoothing to:
    1. attn_norm -> (q_proj, k_proj, v_proj): LN to multiple FCs
    2. ff_norm -> (ff_proj, up_proj): LN to multiple FCs
    3. v_proj -> attn_out: FC to FC
    4. up_proj -> ff_out: FC to FC
    
    Args:
        model: The LLaDA model to modify.
        act_scales: Dictionary of activation scales from calibration.
        alpha: SmoothQuant migration strength (0.5 = balanced).
        device: Device to perform computations on.
    
    Returns:
        The modified model (same object, modified in-place).
    
    Note:
        The model is modified in-place. Make a copy if you need the original.
    """
    if device is None:
        device = next(model.parameters()).device
    
    blocks = get_llada_blocks(model)
    
    # Build module name mapping
    module_to_name = {m: n for n, m in model.named_modules()}
    
    print(f"  Applying SmoothQuant with alpha={alpha} to {len(blocks)} blocks...")
    
    for i, block in enumerate(blocks):
        block_name = module_to_name.get(block, f"block_{i}")
        
        # Check if this block has the expected attributes (LLaDALlamaBlock structure)
        if not all(hasattr(block, attr) for attr in ['attn_norm', 'q_proj', 'k_proj', 'v_proj',
                                                       'ff_norm', 'ff_proj', 'up_proj',
                                                       'attn_out', 'ff_out']):
            print(f"    Skipping block {i}: missing expected attributes")
            continue
        
        # 1. Smooth: attn_norm -> (q_proj, k_proj, v_proj)
        q_key = f"{block_name}.q_proj"
        if q_key in act_scales:
            act_max = act_scales[q_key].to(device)
            
            # Compute scale using combined weight statistics from Q, K, V
            qkv_scale = compute_smooth_scale_multi(
                act_max,
                [block.q_proj.weight, block.k_proj.weight, block.v_proj.weight],
                alpha=alpha,
            )
            
            # Apply smoothing
            smooth_ln_fc(block.attn_norm, [block.q_proj, block.k_proj, block.v_proj], qkv_scale)
        
        # 2. Smooth: ff_norm -> (ff_proj, up_proj)
        ff_key = f"{block_name}.ff_proj"
        up_key = f"{block_name}.up_proj"
        
        # Use whichever key is available (they share the same input)
        mlp_act_key = ff_key if ff_key in act_scales else (up_key if up_key in act_scales else None)
        
        if mlp_act_key:
            act_max = act_scales[mlp_act_key].to(device)
            
            # If both keys exist, take max of both
            if ff_key in act_scales and up_key in act_scales:
                act_max = torch.max(act_max, act_scales[up_key].to(device))
            
            # Compute scale using combined weight statistics
            mlp_scale = compute_smooth_scale_multi(
                act_max,
                [block.ff_proj.weight, block.up_proj.weight],
                alpha=alpha,
            )
            
            # Apply smoothing
            smooth_ln_fc(block.ff_norm, [block.ff_proj, block.up_proj], mlp_scale)
        
        # 3. Smooth: v_proj -> attn_out
        attn_out_key = f"{block_name}.attn_out"
        if attn_out_key in act_scales:
            act_max = act_scales[attn_out_key].to(device)
            
            out_scale = compute_smooth_scale(
                act_max,
                block.attn_out.weight,
                alpha=alpha,
            )
            
            smooth_fc_fc(block.v_proj, block.attn_out, out_scale)
        
        # 4. Smooth: up_proj -> ff_out
        ff_out_key = f"{block_name}.ff_out"
        if ff_out_key in act_scales:
            act_max = act_scales[ff_out_key].to(device)
            
            down_scale = compute_smooth_scale(
                act_max,
                block.ff_out.weight,
                alpha=alpha,
            )
            
            smooth_fc_fc(block.up_proj, block.ff_out, down_scale)
        
        if (i + 1) % 10 == 0 or i == len(blocks) - 1:
            print(f"    Processed {i + 1}/{len(blocks)} blocks")
    
    return model


def replace_with_quant_linear(
    model: nn.Module,
    w_bits: int = 8,
    a_bits: int = 8,
    skip_layers: Optional[list] = None,
) -> int:
    """
    Replace all nn.Linear layers with QuantLinear.
    
    Args:
        model: The model to modify.
        w_bits: Number of bits for weight quantization.
        a_bits: Number of bits for activation quantization.
        skip_layers: List of layer name patterns to skip.
    
    Returns:
        Number of layers replaced.
    """
    if skip_layers is None:
        skip_layers = []
    
    replaced = replace_linear_with_quant(model, w_bits, a_bits, skip_layers)
    print(f"  Replaced {replaced} Linear layers with QuantLinear (W{w_bits}A{a_bits})")
    
    return replaced


# Default layers to skip during quantization (sensitive to quantization error)
DEFAULT_SKIP_LAYERS = ["lm_head"]


def apply_smoothquant_pipeline(
    model_path: str,
    calibration_samples: int = 128,
    alpha: float = 0.5,
    w_bits: int = 8,
    a_bits: int = 8,
    seq_len: int = 512,
    batch_size: int = 1,
    device: str = "cuda",
    save_scales_path: Optional[str] = None,
    load_scales_path: Optional[str] = None,
    skip_quantization: bool = False,
    skip_layers: Optional[list] = None,
) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    """
    Apply the complete SmoothQuant pipeline to a LLaDA model.
    
    This function orchestrates the entire process:
    1. Load the model and tokenizer
    2. Collect or load activation scales
    3. Apply smooth transformations
    4. Replace with quantized layers (optional)
    
    Args:
        model_path: HuggingFace model path or local path.
        calibration_samples: Number of samples for calibration.
        alpha: SmoothQuant migration strength (0.5 = balanced).
        w_bits: Weight quantization bits.
        a_bits: Activation quantization bits.
        seq_len: Sequence length for calibration.
        batch_size: Batch size for calibration.
        device: Device to run on.
        save_scales_path: Path to save activation scales (optional).
        load_scales_path: Path to load pre-computed activation scales (optional).
        skip_quantization: If True, skip the quantization step (only smooth).
        skip_layers: List of layer name patterns to skip during quantization.
                    If None, defaults to DEFAULT_SKIP_LAYERS (["lm_head"]).
                    The lm_head is typically kept in full precision as it is
                    particularly sensitive to quantization error.
    
    Returns:
        Tuple of (model, act_scales).
    
    Example:
        >>> model, scales = apply_smoothquant_pipeline(
        ...     model_path="GSAI-ML/LLaDA-8B-Instruct",
        ...     calibration_samples=128,
        ...     alpha=0.5,
        ...     w_bits=8,
        ...     a_bits=8,
        ... )
    """
    from transformers import AutoTokenizer
    
    # Import LLaDA model (try local import first)
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model.modeling_llada import LLaDAModelLM
    except ImportError:
        from transformers import AutoModelForCausalLM as LLaDAModelLM
    
    print(f"\n{'='*60}")
    print("SmoothQuant Pipeline for LLaDA")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Alpha: {alpha}")
    print(f"Quantization: W{w_bits}A{a_bits}")
    print(f"Calibration samples: {calibration_samples}")
    print(f"Device: {device}")
    
    # Step 1: Load model
    print(f"\n[Step 1/4] Loading model...")
    model = LLaDAModelLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded successfully")
    
    # Step 2: Collect or load activation scales
    print(f"\n[Step 2/4] Collecting activation scales...")
    
    if load_scales_path and os.path.exists(load_scales_path):
        act_scales = load_act_scales(load_scales_path)
    else:
        # Load tokenizer and create calibration dataloader
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Import calibration dataset
        try:
            from quantization_calibration_dataset import LLaDACalibrationDataset
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from quantization_calibration_dataset import LLaDACalibrationDataset
        
        dataset = LLaDACalibrationDataset(
            tokenizer=tokenizer,
            seq_len=seq_len,
            samples=calibration_samples,
            block_size=32,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Collect scales
        act_scales = collect_act_scales(model, dataloader, num_samples=calibration_samples, device=device)
        
        # Optionally save scales
        if save_scales_path:
            save_act_scales(act_scales, save_scales_path)
    
    # Step 3: Apply smoothing
    print(f"\n[Step 3/4] Applying SmoothQuant transformations...")
    model = apply_smoothquant_llada(model, act_scales, alpha=alpha, device=device)
    
    # Step 4: Replace with quantized layers
    if not skip_quantization:
        print(f"\n[Step 4/4] Replacing with quantized layers...")
        # Use default skip layers if none provided
        layers_to_skip = skip_layers if skip_layers is not None else DEFAULT_SKIP_LAYERS
        if layers_to_skip:
            print(f"  Skipping layers matching: {layers_to_skip}")
        replace_with_quant_linear(model, w_bits=w_bits, a_bits=a_bits, skip_layers=layers_to_skip)
    else:
        print(f"\n[Step 4/4] Skipping quantization (skip_quantization=True)")
    
    print(f"\n{'='*60}")
    print("SmoothQuant Pipeline Complete")
    print(f"{'='*60}\n")
    
    return model, act_scales


# Main script entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SmoothQuant for LLaDA models")
    parser.add_argument("--model-path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="HuggingFace model path")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="SmoothQuant migration strength")
    parser.add_argument("--w-bits", type=int, default=8,
                        help="Weight quantization bits")
    parser.add_argument("--a-bits", type=int, default=8,
                        help="Activation quantization bits")
    parser.add_argument("--calibration-samples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for calibration")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for calibration")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--save-scales", type=str, default=None,
                        help="Path to save activation scales")
    parser.add_argument("--load-scales", type=str, default=None,
                        help="Path to load pre-computed activation scales")
    parser.add_argument("--save-model", type=str, default=None,
                        help="Path to save the quantized model state dict")
    parser.add_argument("--skip-quantization", action="store_true",
                        help="Skip quantization step (only apply smoothing)")
    parser.add_argument("--skip-layers", type=str, nargs="*", default=None,
                        help="Layer name patterns to skip during quantization. "
                             "If not specified, defaults to ['lm_head']. "
                             "Use --skip-layers (with no args) to skip nothing.")
    
    args = parser.parse_args()
    
    model, act_scales = apply_smoothquant_pipeline(
        model_path=args.model_path,
        calibration_samples=args.calibration_samples,
        alpha=args.alpha,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
        save_scales_path=args.save_scales,
        load_scales_path=args.load_scales,
        skip_quantization=args.skip_quantization,
        skip_layers=args.skip_layers,
    )
    
    if args.save_model:
        print(f"Saving model state dict to: {args.save_model}")
        os.makedirs(os.path.dirname(args.save_model) if os.path.dirname(args.save_model) else ".", exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print("Model saved successfully")
