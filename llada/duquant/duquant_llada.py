"""
Main DuQuant Pipeline for LLaDA Models

This module orchestrates the complete DuQuant quantization process:
1. Load model and tokenizer
2. Capture layer inputs for calibration
3. Apply DuQuant transformations layer-by-layer
4. Replace Linear layers with quantized versions

The pipeline is specifically designed for LLaDA model architecture.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import DataLoader
from tqdm import tqdm

from .simple_quantizer import SimpleQuantizer
from .quant_linear import DuQuantLinear


def get_llada_blocks(model: nn.Module) -> List[nn.Module]:
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


def get_layer_linear_names(block: nn.Module) -> List[str]:
    """
    Get names of Linear layers to quantize in a LLaDA block.
    
    Args:
        block: A LLaDA transformer block.
    
    Returns:
        List of linear layer attribute names.
    """
    # LLaDALlamaBlock has these linear layers:
    # - q_proj, k_proj, v_proj: attention projections
    # - attn_out: attention output
    # - ff_proj, up_proj: MLP projections (SwiGLU)
    # - ff_out: MLP output
    
    linear_names = []
    candidate_names = ['q_proj', 'k_proj', 'v_proj', 'attn_out', 
                       'ff_proj', 'up_proj', 'ff_out']
    
    for name in candidate_names:
        if hasattr(block, name):
            module = getattr(block, name)
            if isinstance(module, nn.Linear):
                linear_names.append(name)
    
    return linear_names


@torch.no_grad()
def capture_layer_inputs(
    model: nn.Module,
    dataloader: DataLoader,
    num_samples: int = 128,
    device: Optional[str] = None,
) -> List[torch.Tensor]:
    """
    Capture inputs to the first transformer block for calibration.
    
    Args:
        model: The LLaDA model.
        dataloader: DataLoader providing calibration samples.
        num_samples: Number of samples to capture.
        device: Device to run on.
    
    Returns:
        List of input tensors to the first block.
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    inputs = []
    captured = [False]  # Use list to allow modification in closure
    
    def hook(module, inp, out):
        if not captured[0]:
            x = inp[0] if isinstance(inp, tuple) else inp
            inputs.append(x.detach().clone())
            if len(inputs) >= num_samples:
                captured[0] = True
    
    # Get blocks and register hook on first block
    blocks = get_llada_blocks(model)
    handle = blocks[0].register_forward_hook(hook)
    
    try:
        sample_count = 0
        for batch in tqdm(dataloader, desc="  Capturing inputs"):
            if sample_count >= num_samples:
                break
            
            if isinstance(batch, dict):
                input_ids = batch['input_ids']
            elif isinstance(batch, (tuple, list)):
                input_ids = batch[0]
            else:
                input_ids = batch
            
            input_ids = input_ids.to(device)
            model(input_ids=input_ids)
            sample_count += input_ids.shape[0]
    finally:
        handle.remove()
    
    return inputs


@torch.no_grad()
def apply_duquant_to_block(
    block: nn.Module,
    inputs: List[torch.Tensor],
    n_bits: int = 8,
    block_size: int = 128,
    max_rotation_step: int = 256,
    permutation_times: int = 1,
    a_bits: int = 8,
    skip_layers: Optional[List[str]] = None,
) -> List[torch.Tensor]:
    """
    Apply DuQuant quantization to a single transformer block.
    
    Args:
        block: Transformer block to quantize.
        inputs: List of input tensors for calibration.
        n_bits: Weight quantization bits.
        block_size: DuQuant block size.
        max_rotation_step: Maximum rotation iterations.
        permutation_times: Number of permutation iterations.
        a_bits: Activation quantization bits.
        skip_layers: Layer names to skip.
    
    Returns:
        List of output tensors (inputs to next block).
    """
    skip_layers = skip_layers or []
    linear_names = get_layer_linear_names(block)
    
    # Quantize each linear layer
    for name in linear_names:
        if name in skip_layers:
            continue
        
        orig_linear = getattr(block, name)
        
        # Create and calibrate quantizer
        quantizer = SimpleQuantizer(
            n_bits=n_bits,
            symmetric=False,
            block_size=block_size,
            max_rotation_step=max_rotation_step,
            permutation_times=permutation_times,
        )
        quantizer.online_duquant_cali(orig_linear.weight)
        
        # Create quantized linear layer
        quant_linear = DuQuantLinear(
            orig_linear,
            weight_quantizer=quantizer,
            w_bits=n_bits,
            a_bits=a_bits,
        )
        
        # Pre-quantize weights for faster inference
        quant_linear.quantize_weight(use_cache=True)
        
        # Replace the layer
        setattr(block, name, quant_linear)
    
    # Forward inputs through the quantized block
    outputs = []
    for inp in inputs:
        out, _ = block(inp)
        outputs.append(out.detach())
    
    return outputs


@torch.no_grad()
def apply_duquant_llada(
    model: nn.Module,
    dataloader: DataLoader,
    n_bits: int = 8,
    block_size: int = 128,
    max_rotation_step: int = 256,
    permutation_times: int = 1,
    a_bits: int = 8,
    num_samples: int = 128,
    skip_layers: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> nn.Module:
    """
    Apply DuQuant quantization to all layers of a LLaDA model.
    
    This performs layer-by-layer quantization, calibrating each layer
    using the outputs from the previous layer.
    
    Args:
        model: The LLaDA model to quantize.
        dataloader: DataLoader providing calibration samples.
        n_bits: Weight quantization bits.
        block_size: DuQuant block size for rotation/permutation.
        max_rotation_step: Maximum rotation iterations per block.
        permutation_times: Number of rotation+permutation iterations.
        a_bits: Activation quantization bits.
        num_samples: Number of calibration samples.
        skip_layers: Layer names to skip (e.g., ['lm_head']).
        device: Device to run on.
    
    Returns:
        The quantized model (modified in-place).
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    blocks = get_llada_blocks(model)
    
    print(f"  Capturing inputs for calibration...")
    inputs = capture_layer_inputs(model, dataloader, num_samples, device)
    
    print(f"  Applying DuQuant to {len(blocks)} blocks...")
    
    for i, block in enumerate(tqdm(blocks, desc="  Quantizing blocks")):
        # Apply DuQuant to this block
        inputs = apply_duquant_to_block(
            block,
            inputs,
            n_bits=n_bits,
            block_size=block_size,
            max_rotation_step=max_rotation_step,
            permutation_times=permutation_times,
            a_bits=a_bits,
            skip_layers=skip_layers,
        )
    
    return model


# Default layers to skip during quantization
DEFAULT_SKIP_LAYERS = ["lm_head"]


def apply_duquant_pipeline(
    model_path: str,
    calibration_samples: int = 128,
    n_bits: int = 8,
    block_size: int = 128,
    max_rotation_step: int = 256,
    permutation_times: int = 1,
    a_bits: int = 8,
    seq_len: int = 512,
    batch_size: int = 1,
    device: str = "cuda",
    skip_layers: Optional[List[str]] = None,
    save_model_path: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Apply the complete DuQuant pipeline to a LLaDA model.
    
    This function orchestrates the entire process:
    1. Load the model and tokenizer
    2. Create calibration dataloader
    3. Apply DuQuant quantization layer-by-layer
    
    Args:
        model_path: HuggingFace model path or local path.
        calibration_samples: Number of samples for calibration.
        n_bits: Weight quantization bits.
        block_size: DuQuant block size.
        max_rotation_step: Maximum rotation iterations.
        permutation_times: Number of rotation+permutation iterations.
        a_bits: Activation quantization bits.
        seq_len: Sequence length for calibration.
        batch_size: Batch size for calibration.
        device: Device to run on.
        skip_layers: Layer names to skip during quantization.
        save_model_path: Path to save the quantized model state dict.
    
    Returns:
        Tuple of (quantized_model, info_dict).
    
    Example:
        >>> model, info = apply_duquant_pipeline(
        ...     model_path="GSAI-ML/LLaDA-8B-Instruct",
        ...     calibration_samples=128,
        ...     n_bits=8,
        ...     block_size=128,
        ... )
    """
    from transformers import AutoTokenizer
    
    # Import LLaDA model
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model.modeling_llada import LLaDAModelLM
    except ImportError:
        from transformers import AutoModelForCausalLM as LLaDAModelLM
    
    print(f"\n{'='*60}")
    print("DuQuant Pipeline for LLaDA")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Weight bits: {n_bits}")
    print(f"Activation bits: {a_bits}")
    print(f"Block size: {block_size}")
    print(f"Max rotation steps: {max_rotation_step}")
    print(f"Permutation times: {permutation_times}")
    print(f"Calibration samples: {calibration_samples}")
    print(f"Device: {device}")
    
    # Step 1: Load model
    print(f"\n[Step 1/3] Loading model...")
    model = LLaDAModelLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded successfully")
    
    # Step 2: Create calibration dataloader
    print(f"\n[Step 2/3] Preparing calibration data...")
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
    print(f"  Calibration dataset ready: {len(dataset)} samples")
    
    # Step 3: Apply DuQuant
    print(f"\n[Step 3/3] Applying DuQuant quantization...")
    layers_to_skip = skip_layers if skip_layers is not None else DEFAULT_SKIP_LAYERS
    if layers_to_skip:
        print(f"  Skipping layers matching: {layers_to_skip}")
    
    model = apply_duquant_llada(
        model,
        dataloader,
        n_bits=n_bits,
        block_size=block_size,
        max_rotation_step=max_rotation_step,
        permutation_times=permutation_times,
        a_bits=a_bits,
        num_samples=calibration_samples,
        skip_layers=layers_to_skip,
        device=device,
    )
    
    # Optionally save model
    if save_model_path:
        print(f"\n  Saving model to: {save_model_path}")
        os.makedirs(os.path.dirname(save_model_path) if os.path.dirname(save_model_path) else ".", exist_ok=True)
        torch.save(model.state_dict(), save_model_path)
    
    print(f"\n{'='*60}")
    print("DuQuant Pipeline Complete")
    print(f"{'='*60}\n")
    
    info = {
        'n_bits': n_bits,
        'a_bits': a_bits,
        'block_size': block_size,
        'max_rotation_step': max_rotation_step,
        'permutation_times': permutation_times,
        'calibration_samples': calibration_samples,
    }
    
    return model, info


# Main script entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DuQuant for LLaDA models")
    parser.add_argument("--model-path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="HuggingFace model path")
    parser.add_argument("--n-bits", type=int, default=8,
                        help="Weight quantization bits")
    parser.add_argument("--a-bits", type=int, default=8,
                        help="Activation quantization bits")
    parser.add_argument("--block-size", type=int, default=128,
                        help="DuQuant block size")
    parser.add_argument("--max-rotation-step", type=int, default=256,
                        help="Maximum rotation iterations")
    parser.add_argument("--permutation-times", type=int, default=1,
                        help="Number of rotation+permutation iterations")
    parser.add_argument("--calibration-samples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for calibration")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for calibration")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--save-model", type=str, default=None,
                        help="Path to save the quantized model state dict")
    parser.add_argument("--skip-layers", type=str, nargs="*", default=None,
                        help="Layer name patterns to skip during quantization")
    
    args = parser.parse_args()
    
    model, info = apply_duquant_pipeline(
        model_path=args.model_path,
        calibration_samples=args.calibration_samples,
        n_bits=args.n_bits,
        a_bits=args.a_bits,
        block_size=args.block_size,
        max_rotation_step=args.max_rotation_step,
        permutation_times=args.permutation_times,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
        skip_layers=args.skip_layers,
        save_model_path=args.save_model,
    )
    
    print("Quantization info:", info)
