"""
SmoothQuant Preprocessing for LLaDA Models

This script applies SmoothQuant weight preprocessing to LLaDA models.
It collects activation statistics during a calibration pass, then migrates
quantization difficulty from activations to weights by scaling.

The smoothed model is saved for subsequent quantization.
"""

from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
from model.modeling_llada import LLaDAModelLM, LLaDALlamaBlock
from quantization_calibration_dataset import LLaDACalibrationDataset

# Configuration
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
BATCH_SIZE = 1
ALPHA = 0.5  # SmoothQuant migration strength (lower = less aggressive, 0.3 is more conservative)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "models"
SAVE_PATH = os.path.join(SAVE_DIR, "llada_smoothquant.pt")
CALIBRATION_SAMPLES = 128  # Number of calibration samples


class ActivationCapture:
    """Captures per-channel activation maximums for SmoothQuant calibration."""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.stats = {}
    
    def _hook_fn(self, name):
        def hook(module, input, output):
            x = input[0].detach()
            max_step = x.abs().view(-1, x.shape[-1]).max(dim=0).values.cpu()
            if name not in self.stats:
                self.stats[name] = max_step
            else:
                self.stats[name] = torch.max(self.stats[name], max_step)
        return hook
    
    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __str__(self):
        return str(self.stats)


def get_layers(model):
    """
    Get list of transformer blocks, handling both 'blocks' and 'block_groups' structures.
    """
    transformer = model.model.transformer
    if hasattr(transformer, 'blocks'):
        return transformer.blocks
    elif hasattr(transformer, 'block_groups'):
        # Flatten block groups into a list of blocks
        layers = []
        for group in transformer.block_groups:
            layers.extend(list(group))
        return layers
    else:
        raise ValueError("Could not find blocks or block_groups in model structure")


def fuse_smoothquant_llada(model, capture_stats, alpha=0.5, device="cuda"):
    # alpha=0.5 is usually preferred for SmoothQuant on LLaMA to balance migration
    
    layers = get_layers(model) # Ensure this retrieves the block list
    module_to_name = {m: n for n, m in model.named_modules()}
    
    print(f"Applying SmoothQuant with alpha={alpha}...")
    
    for i, block in enumerate(layers):
        if not isinstance(block, LLaDALlamaBlock):
            continue
            
        block_name = module_to_name[block]
        
        # =====================================================
        # 1. Attention Block (Q, K, V share input from attn_norm)
        # =====================================================
        q_name = f"{block_name}.q_proj"
        
        # Check if we have stats
        if q_name in capture_stats:
            # Load stats (activation max per channel)
            act_max = capture_stats[q_name].to(device)
            
            # Combine weights to find max scale requirement across Q, K, V
            # We calculate scale per input-channel (dim 1 of weights, dim 0 here if transposed)
            # Standard Linear weight shape: [out_features, in_features]
            # We want max over out_features (dim 0) -> resulting shape [in_features]
            w_q = block.q_proj.weight.abs().max(dim=0).values
            w_k = block.k_proj.weight.abs().max(dim=0).values
            w_v = block.v_proj.weight.abs().max(dim=0).values
            w_max = torch.max(torch.max(w_q, w_k), w_v)
            
            # STRICT SHAPE CHECK: Do not use slicing here. 
            # If these don't match, your stats collection or model def is wrong.
            assert act_max.shape == w_max.shape, \
                f"Shape mismatch in Attn Layer {i}: Act {act_max.shape} vs W {w_max.shape}"

            # Calculate Scale
            act_max = torch.clamp(act_max, min=1e-5)
            w_max = torch.clamp(w_max, min=1e-5)
            
            scales = act_max.pow(alpha) / w_max.pow(1 - alpha)
            scales = torch.clamp(scales, min=1e-3, max=6.0) # Widen clamp slightly or stick to 0.1-10

            # Apply Smoothing
            with torch.no_grad():
                # 1. Scale Input (Norm) DOWN
                block.attn_norm.weight.div_(scales)
                if hasattr(block.attn_norm, 'bias') and block.attn_norm.bias is not None:
                    block.attn_norm.bias.div_(scales)
                
                # 2. Scale Weights (Q, K, V) UP
                # Reshape for broadcasting: [1, in_features]
                scales_view = scales.view(1, -1)
                
                block.q_proj.weight.mul_(scales_view)
                block.k_proj.weight.mul_(scales_view)
                block.v_proj.weight.mul_(scales_view)

        ff_proj_key = f"{block_name}.ff_proj"
        up_proj_key = f"{block_name}.up_proj"
        
        print(" ------------------------------------------------------------")
        print(f"  Layer {i}: Attention Block Scales Mean: {scales.mean():.4f}, Std: {scales.std():.4f}, Max: {scales.max():.4f}, Min: {scales.min():.4f}")
        print(f"  Layer {i}: Act Max: {act_max.mean():.4f}, W Max: {w_max.mean():.4f}")
        # We only care about stats for the *inputs* to the MLP, which creates keys
        # for ff_proj or up_proj. They share the same input (ff_norm output).
        if ff_proj_key in capture_stats or up_proj_key in capture_stats:
            
            # 1. Get Activation Max (Input to MLP)
            if ff_proj_key in capture_stats:
                act_max = capture_stats[ff_proj_key].to(device)
            else:
                act_max = capture_stats[up_proj_key].to(device)
            
            # Robustness: if both exist, take the max of both to be safe
            if ff_proj_key in capture_stats and up_proj_key in capture_stats:
                act_max = torch.max(act_max, capture_stats[up_proj_key].to(device))
            
            # 2. Get Weight Max (Combine BOTH expansion layers)
            # We must look at the "difficulty" of quantizing both matrices
            w_ff = block.ff_proj.weight.abs().max(dim=0).values
            w_up = block.up_proj.weight.abs().max(dim=0).values
            w_max = torch.max(w_ff, w_up)
            
            # 3. Calculate Scale
            # Shape check: act_max and w_max must both be size [4096]
            if act_max.shape != w_max.shape:
                # This should theoretically not happen given your logs, 
                # but good to keep for safety against bad stats collection.
                print(f"WARNING: Shape mismatch in layer {i}. Skipping MLP smoothing.")
                continue

            act_max = torch.clamp(act_max, min=1e-5)
            w_max = torch.clamp(w_max, min=1e-5)
            
            s_mlp = act_max.pow(alpha) / w_max.pow(1 - alpha)
            s_mlp = torch.clamp(s_mlp, min=1e-3, max=6.0)

            # 4. Apply Transformation
            with torch.no_grad():
                # A. Scale the Norm DOWN
                block.ff_norm.weight.div_(s_mlp)
                if hasattr(block.ff_norm, 'bias') and block.ff_norm.bias is not None:
                    block.ff_norm.bias.div_(s_mlp)

                # B. Scale BOTH Projections UP
                s_broadcast = s_mlp.view(1, -1)
                
                block.ff_proj.weight.mul_(s_broadcast)
                block.up_proj.weight.mul_(s_broadcast)
            print(f"  Layer {i}: Smoothed MLP (Mean Scale: {s_mlp.mean():.4f})")
            print(f"  Layer {i}: STDEV smoothed scale {s_mlp.std():.4f}")
            print(f"  Layer {i}: Max smoothed scale {s_mlp.max():.4f}, Min smoothed scale {s_mlp.min():.4f}")
            print(f"  Layer {i}: Act Max: {act_max.mean():.4f}, W Max: {w_max.mean():.4f}")
                        
    return model

# === Main execution ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmoothQuant preprocessing for LLaDA models")
    parser.add_argument("--model-path", type=str, default=MODEL_NAME,
                        help=f"HuggingFace model path (default: {MODEL_NAME})")
    parser.add_argument("--alpha", type=float, default=ALPHA, 
                        help=f"SmoothQuant migration strength (default: {ALPHA})")
    parser.add_argument("--calibration-samples", type=int, default=CALIBRATION_SAMPLES,
                        help=f"Number of calibration samples (default: {CALIBRATION_SAMPLES})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for calibration (default: {BATCH_SIZE})")
    parser.add_argument("--save-path", type=str, default=SAVE_PATH,
                        help=f"Path to save smoothed model (default: {SAVE_PATH})")
    args = parser.parse_args()
    
    MODEL_NAME = args.model_path
    ALPHA = args.alpha
    CALIBRATION_SAMPLES = args.calibration_samples
    BATCH_SIZE = args.batch_size
    SAVE_PATH = args.save_path
    
    print(f"\n=== SmoothQuant Preprocessing for LLaDA ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Alpha: {ALPHA}")
    print(f"Calibration samples: {CALIBRATION_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    
    # Step 1: Load model
    print(f"\n[Step 1/4] Loading model: {MODEL_NAME}...")
    model = LLaDAModelLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    model.eval()
    print("  ✓ Model loaded")
    
    # Step 2: Register hooks for activation capture
    print("\n[Step 2/4] Registering activation capture hooks...")
    capture = ActivationCapture(model)
    capture.register_hooks()
    print(f"  Registered hooks on {len(capture.hooks)} Linear layers")

    # Step 3: Calibration pass to collect activation statistics
    print("\n[Step 3/4] Running calibration pass with LLaDA calibration dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    dataset = LLaDACalibrationDataset(
        tokenizer=tokenizer,
        seq_len=512,
        samples=CALIBRATION_SAMPLES,
        block_size=32
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    batch_count = len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(DEVICE)
            model(input_ids=input_ids)
            if (batch_idx + 1) % 5 == 0 or batch_idx == batch_count - 1:
                print(f"  Processed {batch_idx + 1}/{batch_count} batches")

    # Step 4: Clear hooks and apply SmoothQuant
    print("\n[Step 4/4] Applying SmoothQuant weight transformations...")
    capture_stats = capture.stats
    capture.clear_hooks()
    print(f"  Collected stats for {len(capture_stats)} layers")
    
    model = fuse_smoothquant_llada(model, capture_stats, alpha=ALPHA, device=DEVICE)

    # Step 5: Save the smoothed model
    print(f"\n[Step 5/5] Saving smoothed model...")
    os.makedirs(os.path.dirname(SAVE_PATH) if os.path.dirname(SAVE_PATH) else ".", exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"  ✓ Saved smoothed model state_dict to: {SAVE_PATH}")

    print("\n=== SmoothQuant Preprocessing Complete ===")
    print(f"The smoothed model can now be loaded and quantized.")
    print(f"Next step: Load the state_dict and apply INT8/INT4 quantization.")
