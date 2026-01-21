"""
W8A8 Quantization Script for LLaDA Models

This script applies custom INT8 weight and INT8 activation quantization to LLaDA models.
- Weights: Per-channel quantization (scale per output channel)
- Activations: Dynamic per-token quantization (scale computed at runtime)
- SmoothQuant: Integrated activation smoothing (optional)

Usage:
    # Quantize base model
    python -m llada.quantize.quantize_model --model-path GSAI-ML/LLaDA-8B-Instruct
    
    # Quantize with integrated SmoothQuant
    python -m llada.quantize.quantize_model --model-path GSAI-ML/LLaDA-8B-Instruct --enable-smoothquant --alpha 0.5
    
    # With custom save directory
    python -m llada.quantize.quantize_model --model-path GSAI-ML/LLaDA-8B-Instruct --save-dir models/llada_custom_w8a8
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM
from quantize.quantized_linear import W8A8Linear, replace_linear_with_w8a8
from quantize.quantization_utils import compute_quantization_error
from quantize.smoothquant import (
    ActivationCollector,
    compute_all_smooth_scales,
    run_calibration,
)

# Configuration
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default SmoothQuant settings
DEFAULT_ALPHA = 0.5
DEFAULT_CALIBRATION_SAMPLES = 128
DEFAULT_BATCH_SIZE = 1

# Layers to exclude from quantization (keep in higher precision)
QUANTIZATION_EXCLUDE = ["lm_head"]


def print_memory_usage(step_name: str) -> None:
    """Print GPU memory usage statistics."""
    if not torch.cuda.is_available():
        print(f"\n--- Memory Stats: {step_name} ---")
        print("CUDA not available, skipping memory stats")
        return
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    
    print(f"\n--- Memory Stats: {step_name} ---")
    print(f"1. Actual Model Weights (Allocated): {allocated:.2f} GB")
    print(f"2. Total Reserved (nvidia-smi):      {reserved:.2f} GB")
    
    overhead = reserved - allocated
    print(f"3. Buffer/Overhead:                  {overhead:.2f} GB")
    print("-" * 40)


def fix_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters by type."""
    total_params = 0
    quantized_params = 0
    fp_params = 0
    smooth_layers = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        fp_params += param.numel()
    
    for name, buffer in model.named_buffers():
        if 'weight_int8' in name:
            quantized_params += buffer.numel()
    
    for name, module in model.named_modules():
        if isinstance(module, W8A8Linear) and module.has_smooth_scale:
            smooth_layers += 1
    
    return {
        "total_fp_params": fp_params,
        "quantized_weights": quantized_params,
        "smooth_layers": smooth_layers,
        "estimated_size_fp16_gb": fp_params * 2 / (1024**3),
        "estimated_size_int8_gb": quantized_params * 1 / (1024**3),
    }


def print_model_structure(model: nn.Module, max_depth: int = 3) -> None:
    """Print model structure with quantization status."""
    print("\n=== Model Structure (Quantization Status) ===")
    
    def _print_module(module: nn.Module, prefix: str = "", depth: int = 0):
        if depth >= max_depth:
            return
            
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, W8A8Linear):
                smooth_info = " [SMOOTH]" if child.has_smooth_scale else ""
                print(f"  [QUANTIZED{smooth_info}] {full_name}: W8A8Linear({child.in_features}, {child.out_features})")
            elif isinstance(child, nn.Linear):
                print(f"  [FP] {full_name}: Linear({child.in_features}, {child.out_features})")
            elif isinstance(child, nn.Embedding):
                print(f"  [FP] {full_name}: Embedding({child.num_embeddings}, {child.embedding_dim})")
            else:
                _print_module(child, full_name, depth + 1)
    
    _print_module(model)
    print("=" * 50)


def quantize_model(
    model: nn.Module,
    exclude_patterns: list,
    compute_dtype: torch.dtype = torch.bfloat16,
    smooth_scales: dict = None,
) -> nn.Module:
    """
    Quantize all Linear layers in the model to W8A8.
    
    Args:
        model: The model to quantize
        exclude_patterns: List of layer name patterns to exclude
        compute_dtype: Data type for computation
        smooth_scales: Optional dictionary of SmoothQuant scales per layer
        
    Returns:
        The quantized model
    """
    print(f"\nQuantizing model with W8A8...")
    print(f"  Exclude patterns: {exclude_patterns}")
    print(f"  Compute dtype: {compute_dtype}")
    if smooth_scales:
        print(f"  SmoothQuant scales: {len(smooth_scales)} layers")
    
    # Count layers before quantization
    linear_count = 0
    excluded_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            should_exclude = any(excl in name for excl in exclude_patterns)
            if should_exclude:
                excluded_count += 1
            else:
                linear_count += 1
    
    print(f"  Linear layers to quantize: {linear_count}")
    print(f"  Linear layers excluded: {excluded_count}")
    
    # Perform quantization
    model = replace_linear_with_w8a8(
        model,
        exclude_names=exclude_patterns,
        compute_dtype=compute_dtype,
        smooth_scales=smooth_scales,
    )
    
    # Verify quantization
    quantized_count = 0
    smooth_count = 0
    for name, module in model.named_modules():
        if isinstance(module, W8A8Linear):
            quantized_count += 1
            if module.has_smooth_scale:
                smooth_count += 1
    
    print(f"  Successfully quantized: {quantized_count} layers")
    if smooth_scales:
        print(f"  Layers with SmoothQuant: {smooth_count}")
    
    return model


def run_smoothquant_calibration(
    model: nn.Module,
    tokenizer,
    calibration_samples: int,
    batch_size: int,
    alpha: float,
    device: str,
) -> dict:
    """
    Run SmoothQuant calibration to compute scales.
    
    Args:
        model: The model to calibrate
        tokenizer: Tokenizer for the model
        calibration_samples: Number of calibration samples
        batch_size: Batch size for calibration
        alpha: SmoothQuant migration strength
        device: Device to run on
        
    Returns:
        Dictionary mapping layer names to smooth scales
    """
    print(f"\n  Running SmoothQuant calibration...")
    print(f"    Alpha: {alpha}")
    print(f"    Calibration samples: {calibration_samples}")
    print(f"    Batch size: {batch_size}")
    
    # Import calibration dataset
    from quantization_calibration_dataset import LLaDACalibrationDataset
    
    # Create calibration dataset
    dataset = LLaDACalibrationDataset(
        tokenizer=tokenizer,
        seq_len=512,
        samples=calibration_samples,
        block_size=32,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Collect activation statistics
    activation_stats = run_calibration(model, dataloader, device=device)
    
    # Compute smooth scales for all layers
    print(f"\n  Computing SmoothQuant scales...")
    smooth_scales = compute_all_smooth_scales(
        model,
        activation_stats,
        alpha=alpha,
        device=device,
    )
    
    print(f"    Computed scales for {len(smooth_scales)} layers")
    
    # Print some statistics about the scales
    if smooth_scales:
        all_scales = torch.cat([s.flatten() for s in smooth_scales.values()])
        print(f"    Scale statistics:")
        print(f"      Mean: {all_scales.mean().item():.4f}")
        print(f"      Std:  {all_scales.std().item():.4f}")
        print(f"      Min:  {all_scales.min().item():.4f}")
        print(f"      Max:  {all_scales.max().item():.4f}")
    
    return smooth_scales


def verify_quantization(model: nn.Module, tokenizer, device: str) -> None:
    """
    Verify that quantized model produces reasonable outputs.
    
    Args:
        model: The quantized model
        tokenizer: Tokenizer for the model
        device: Device to run verification on
    """
    print("\n=== Verifying Quantization ===")
    
    # Simple test input
    test_text = "The quick brown fox"
    inputs = tokenizer(test_text, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(device)
    
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            
            # Check for NaN or Inf
            has_nan = torch.isnan(logits).any().item()
            has_inf = torch.isinf(logits).any().item()
            
            if has_nan or has_inf:
                print(f"  WARNING: Output contains NaN={has_nan}, Inf={has_inf}")
            else:
                print(f"  Output shape: {logits.shape}")
                print(f"  Output range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
                print(f"  Verification PASSED")
                
        except Exception as e:
            print(f"  Verification FAILED: {e}")


def save_quantized_model(
    model: nn.Module,
    tokenizer,
    save_dir: str,
) -> None:
    """
    Save the quantized model.
    
    Args:
        model: The quantized model to save
        tokenizer: Tokenizer to save alongside
        save_dir: Directory to save to
    """
    print(f"\nSaving quantized model to: {save_dir}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(save_dir, safe_serialization=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    
    print(f"  Model saved successfully")
    
    # Print size info
    total_size = 0
    for f in os.listdir(save_dir):
        fpath = os.path.join(save_dir, f)
        if os.path.isfile(fpath):
            total_size += os.path.getsize(fpath)
    
    print(f"  Total size: {total_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Quantize LLaDA model to W8A8 (custom implementation)")
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=MODEL_NAME,
        help=f"HuggingFace model path (default: {MODEL_NAME})"
    )
    
    # SmoothQuant options (new integrated approach)
    parser.add_argument(
        "--enable-smoothquant",
        action="store_true",
        help="Enable integrated SmoothQuant (runs calibration and applies scaling during quantization)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"SmoothQuant migration strength (default: {DEFAULT_ALPHA}). Lower = more aggressive smoothing."
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help=f"Number of calibration samples for SmoothQuant (default: {DEFAULT_CALIBRATION_SAMPLES})"
    )
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for calibration (default: {DEFAULT_BATCH_SIZE})"
    )
    
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save quantized model (default: models/llada_custom_w8a8)",
    )
    parser.add_argument(
        "--exclude-layers", 
        type=str, 
        nargs="+", 
        default=QUANTIZATION_EXCLUDE,
        help=f"Layer name patterns to exclude from quantization (default: {QUANTIZATION_EXCLUDE})"
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Compute dtype for scales and bias (default: bfloat16)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification after quantization"
    )
    parser.add_argument(
        "--print-structure",
        action="store_true", 
        help="Print model structure after quantization"
    )
    args = parser.parse_args()
    
    # Parse compute dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map[args.compute_dtype]
    
    # Set save directory
    if args.save_dir is None:
        base_name = "llada_custom_w8a8"
        if args.enable_smoothquant:
            base_name += "_smoothquant"
        save_dir = os.path.join("models", base_name)
    else:
        save_dir = args.save_dir
    
    print(f"\n{'='*60}")
    print("W8A8 Quantization for LLaDA (Custom Implementation)")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Device: {DEVICE}")
    print(f"Compute dtype: {args.compute_dtype}")
    if args.enable_smoothquant:
        print(f"SmoothQuant: ENABLED (alpha={args.alpha})")
        print(f"  Calibration samples: {args.calibration_samples}")
        print(f"  Calibration batch size: {args.calibration_batch_size}")
    else:
        print(f"SmoothQuant: DISABLED")
    print(f"Exclude patterns: {args.exclude_layers}")
    print(f"Save directory: {save_dir}")
    
    fix_seed(42)
    
    # Determine total steps
    if args.enable_smoothquant:
        total_steps = 5  # Load, Calibrate, Quantize, Verify, Save
    else:
        total_steps = 4  # Load, Quantize, Verify, Save
    step = 1
    
    # Step 1: Load model
    print(f"\n[Step {step}/{total_steps}] Loading model: {args.model_path}...")
    torch.cuda.empty_cache()
    
    model = LLaDAModelLM.from_pretrained(
        args.model_path,
        torch_dtype=compute_dtype,
        device_map=DEVICE,
        trust_remote_code=True
    )
    model.eval()
    print("  Model loaded")
    
    print_memory_usage("After Model Load")
    step += 1
    
    # Load tokenizer (needed for both calibration and verification)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Step 2 (optional): SmoothQuant calibration
    smooth_scales = None
    if args.enable_smoothquant:
        print(f"\n[Step {step}/{total_steps}] Running SmoothQuant calibration...")
        smooth_scales = run_smoothquant_calibration(
            model=model,
            tokenizer=tokenizer,
            calibration_samples=args.calibration_samples,
            batch_size=args.calibration_batch_size,
            alpha=args.alpha,
            device=DEVICE,
        )
        step += 1
    
    # Step 3: Quantize model
    print(f"\n[Step {step}/{total_steps}] Quantizing model to W8A8...")
    
    model = quantize_model(
        model,
        exclude_patterns=args.exclude_layers,
        compute_dtype=compute_dtype,
        smooth_scales=smooth_scales,
    )
    
    print_memory_usage("After Quantization")
    step += 1
    
    # Print parameter counts
    param_info = count_parameters(model)
    print(f"\n  Parameter info:")
    print(f"    FP parameters: {param_info['total_fp_params']:,}")
    print(f"    Quantized weights: {param_info['quantized_weights']:,}")
    if args.enable_smoothquant:
        print(f"    Layers with SmoothQuant: {param_info['smooth_layers']}")
    
    if args.print_structure:
        print_model_structure(model)
    
    # Step 4: Verify (optional)
    if args.verify:
        print(f"\n[Step {step}/{total_steps}] Verifying quantized model...")
        verify_quantization(model, tokenizer, DEVICE)
    else:
        print(f"\n[Step {step}/{total_steps}] Skipping verification (use --verify to enable)")
    step += 1
    
    # Step 5: Save model
    print(f"\n[Step {step}/{total_steps}] Saving quantized model...")
    save_quantized_model(model, tokenizer, save_dir)
    
    print_memory_usage("After Save")
    
    print(f"\n{'='*60}")
    print("W8A8 Quantization Complete")
    print(f"{'='*60}")
    print(f"\nQuantized model saved to: {save_dir}")
    print("\nQuantization Details:")
    print("  - Weights: INT8 per-channel quantization")
    print("  - Activations: Dynamic INT8 per-token quantization")
    if args.enable_smoothquant:
        print(f"  - SmoothQuant: Integrated (alpha={args.alpha})")
    print("\nTo load the quantized model:")
    print("  from transformers import AutoTokenizer")
    print("  from llada.model.modeling_llada import LLaDAModelLM")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{save_dir}', trust_remote_code=True)")
    print(f"  model = LLaDAModelLM.from_pretrained('{save_dir}', trust_remote_code=True, device_map='{DEVICE}')")


if __name__ == "__main__":
    main()
