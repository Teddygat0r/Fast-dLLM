"""
W8A8 Quantization script for LLaDA Models

This script applies INT8 weight and INT8 activation quantization to LLaDA models.
It can work with either the base model or a SmoothQuant-preprocessed model.

W8A8 provides higher accuracy than W4A8 at the cost of larger model size.
Best used when accuracy is critical and memory constraints allow for 8-bit weights.

Usage:
    # Quantize base model
    python quantize_llada_w8a8.py --model-path GSAI-ML/LLaDA-8B-Instruct
    
    # Quantize SmoothQuant-preprocessed model
    python quantize_llada_w8a8.py --model-path GSAI-ML/LLaDA-8B-Instruct --load-smoothed --smoothed-model-path models/llada_smoothquant.pt
"""

from transformers import AutoTokenizer, AutoConfig
import torch
import numpy as np
import random
import argparse
import os
from torch.utils.data import DataLoader
from quantization_calibration_dataset import LLaDACalibrationDataset
from model.modeling_llada import LLaDAModelLM
from optimum.quanto import quantize, Calibration, qint8, freeze

# Configuration
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOOTHED_WEIGHTS_PATH = "models/llada_smoothquant.pt"

# Layers to exclude from quantization (keep in higher precision)
# wte: input embeddings, often kept in FP16
# ff_out: output layer (if not using weight tying), critical for generation quality
# ln_f: final layer norm, typically kept in higher precision
QUANTIZATION_EXCLUDE = ["wte", "ff_out", "ln_f", "norm"]

def print_memory_usage(step_name):
    """Print GPU memory usage statistics."""
    if not torch.cuda.is_available():
        print(f"\n--- Memory Stats: {step_name} ---")
        print("CUDA not available, skipping memory stats")
        return
    
    # Convert to GB
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    
    print(f"\n--- Memory Stats: {step_name} ---")
    print(f"1. Actual Model Weights (Allocated): {allocated:.2f} GB")
    print(f"2. Total Reserved (nvidia-smi):      {reserved:.2f} GB")
    
    # Calculate the 'waste' or buffer
    overhead = reserved - allocated
    print(f"3. Buffer/Overhead:                  {overhead:.2f} GB")
    print("-" * 40)

def fix_seed(seed=42):
    """Fix random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize LLaDA model to W8A8")
    parser.add_argument("--model-path", type=str, default=MODEL_NAME,
                        help=f"HuggingFace model path (default: {MODEL_NAME})")
    parser.add_argument("--load-smoothed", action="store_true",
                        help="Load SmoothQuant-preprocessed weights before quantization")
    parser.add_argument("--smoothed-model-path", type=str, default=SMOOTHED_WEIGHTS_PATH,
                        help=f"Path to smoothed model weights (default: {SMOOTHED_WEIGHTS_PATH})")
    parser.add_argument("--calibration-samples", type=int, default=128,
                        help="Number of calibration samples (default: 128)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for calibration (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save quantized model with Quanto (default: auto-generated)",
    )
    parser.add_argument("--exclude-layers", type=str, nargs="+", default=QUANTIZATION_EXCLUDE,
                        help=f"Layers to exclude from quantization (default: {QUANTIZATION_EXCLUDE})")
    args = parser.parse_args()
    
    MODEL_NAME = args.model_path
    BATCH_SIZE = args.batch_size
    CALIBRATION_SAMPLES = args.calibration_samples
    
    print(f"\n=== W8A8 Quantization for LLaDA ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    if args.load_smoothed:
        print(f"SmoothQuant model: {args.smoothed_model_path}")
    print(f"Calibration samples: {CALIBRATION_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Excluding layers: {args.exclude_layers}")
    
    # Step 1: Load model
    print(f"\n[Step 1/5] Loading model: {MODEL_NAME}...")
    torch.cuda.empty_cache()
    
    model = LLaDAModelLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    model.eval()
    print("  Model loaded")
    
    # Step 2: Load smoothed weights if requested
    if args.load_smoothed:
        print(f"\n[Step 2/5] Loading SmoothQuant-preprocessed weights...")
        if os.path.exists(args.smoothed_model_path):
            print(f"  Loading from {args.smoothed_model_path}...")
            state_dict = torch.load(args.smoothed_model_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print("  Smoothed weights loaded")
        else:
            print(f"  WARNING: Smoothed weights not found at {args.smoothed_model_path}")
            print("  Proceeding without SmoothQuant preprocessing.")
            print("  For better results, run: python smooth_quant_llada.py first")
    else:
        print(f"\n[Step 2/5] Skipping SmoothQuant (using base model)")
    
    print_memory_usage("After Model Load")
    
    # Step 3: Quantize model
    print(f"\n[Step 3/5] Quantizing model: weights=qint8, activations=qint8")
    print(f"  Excluding layers: {args.exclude_layers}")
    
    # Build exclude list - need to match actual module names in the model
    exclude_list = []
    for name, module in model.named_modules():
        for exclude_pattern in args.exclude_layers:
            if exclude_pattern in name:
                exclude_list.append(name)
                break
    
    if exclude_list:
        print(f"  Found {len(exclude_list)} modules to exclude")
        # Use the first few as examples
        for name in exclude_list[:5]:
            print(f"    - {name}")
        if len(exclude_list) > 5:
            print(f"    ... and {len(exclude_list) - 5} more")
    
    # W8A8: Both weights and activations are INT8
    quantize(model, weights=qint8, activations=qint8, exclude=args.exclude_layers)
    print("  Model quantized (W8A8)")
    
    print_memory_usage("After Quantization")
    
    # Step 4: Calibration pass
    print(f"\n[Step 4/5] Running calibration pass...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    fix_seed(42)
    
    dataset = LLaDACalibrationDataset(
        tokenizer=tokenizer,
        seq_len=512,
        samples=CALIBRATION_SAMPLES,
        block_size=32
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("  Starting calibration (forward pass only)...")
    with Calibration(momentum=0.9):
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(DEVICE)
            
            with torch.no_grad():
                model(input_ids=input_ids)
            
            if (batch_idx + 1) % 5 == 0 or batch_idx == len(dataloader) - 1:
                print(f"  Processed batch {batch_idx + 1}/{len(dataloader)}")
    
    print("  Calibration complete")
    
    # Step 5: Freeze and save (using Quanto utilities)
    print(f"\n[Step 5/5] Freezing model to integer representation...")
    freeze(model)
    print("  Model frozen")
    
    # Determine save directory for Quanto
    if args.save_dir:
        save_dir = args.save_dir
    else:
        base_name = "llada_quantized_w8a8"
        if args.load_smoothed:
            base_name += "_smoothquant"
        save_dir = os.path.join("models", base_name)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSaving quantized model with Quanto...")
    print(f"  Save directory: {save_dir}")
    # Save both model and tokenizer so the directory is self-contained
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    print("  Quantized model and tokenizer saved with Quanto-compatible format")
    
    print_memory_usage("After Save")
    
    print("\n=== W8A8 Quantization Complete ===")
    print("Quantized model saved with Quanto utilities to:")
    print(f"  - Directory: {save_dir}")
    print("\nW8A8 vs W4A8 comparison:")
    print("  - W8A8: Higher accuracy, ~8GB for 8B model")
    print("  - W4A8: Lower memory, ~4GB for 8B model")
    print("\nTo load the quantized model, use:")
    print("  from transformers import AutoTokenizer")
    print("  from model.modeling_llada import LLaDAModelLM")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{save_dir}', trust_remote_code=True)")
    print(f"  model = LLaDAModelLM.from_pretrained('{save_dir}', trust_remote_code=True, device_map='{DEVICE}')")
