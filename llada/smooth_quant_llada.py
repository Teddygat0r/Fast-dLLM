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
from model.modeling_llada import LLaDAModelLM, LLaDASequentialBlock, LLaDALlamaBlock
from quantization_calibration_dataset import LLaDACalibrationDataset

# Configuration
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
BATCH_SIZE = 32
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


def fuse_smoothquant_llada(model, capture_stats, alpha=0.3, device="cuda"):
    """
    Apply SmoothQuant weight preprocessing to the LLaDA model.
    
    Migrates quantization difficulty from activations to weights by:
    1. Dividing LayerNorm weights by scale s
    2. Multiplying subsequent Linear weights by scale s
    
    Scale s = act_max^alpha / weight_max^(1-alpha)
    
    Uses per-tensor scaling (mean of per-channel scales) for stability.
    Lower alpha (default 0.3) is more conservative and preserves model quality better.
    """
    layers = get_layers(model)
    scale_stats = {
        'attn_scales': [],
        'mlp_scales': []
    }
    
    # Build a mapping from layer index to actual module path names
    layer_name_map = {}
    for name, module in model.named_modules():
        if isinstance(module, (LLaDASequentialBlock, LLaDALlamaBlock)):
            # Extract layer index from name
            # Names can be like "model.transformer.blocks.0" or "model.transformer.block_groups.0.0"
            parts = name.split('.')
            if 'blocks' in parts:
                idx = parts.index('blocks')
                if idx + 1 < len(parts):
                    try:
                        layer_idx = int(parts[idx + 1])
                        layer_name_map[layer_idx] = name
                    except ValueError:
                        pass
            elif 'block_groups' in parts:
                idx = parts.index('block_groups')
                if idx + 2 < len(parts):
                    try:
                        group_idx = int(parts[idx + 1])
                        block_idx = int(parts[idx + 2])
                        # Calculate global layer index
                        # Access config through model.config (LLaDAConfig) which has block_group_size
                        # We need to get it from the actual model config
                        transformer = model.model.transformer
                        if hasattr(transformer, 'block_groups') and len(transformer.block_groups) > 0:
                            # Get block_group_size from the first group's config
                            block_group_size = transformer.block_groups[0].config.block_group_size
                        else:
                            block_group_size = 1  # Default
                        layer_idx = group_idx * block_group_size + block_idx
                        layer_name_map[layer_idx] = name
                    except (ValueError, AttributeError):
                        pass
    
    for i, block in enumerate(layers):
        # Detect block type
        is_sequential = isinstance(block, LLaDASequentialBlock)
        is_llama = isinstance(block, LLaDALlamaBlock)
        
        if not (is_sequential or is_llama):
            print(f"WARNING: Unknown block type at layer {i}, skipping SmoothQuant")
            continue
        
        # Get base name for this layer
        base_name = layer_name_map.get(i, f"model.transformer.blocks.{i}")
        
        # === Attention block smoothing ===
        if is_sequential:
            # LLaDASequentialBlock uses fused att_proj
            att_proj_name = f"{base_name}.att_proj"
            
            if att_proj_name not in capture_stats:
                print(f"WARNING: Could not find activation stats for att_proj in layer {i} (tried {att_proj_name})")
                continue
            
            # Get activation max from fused att_proj
            attn_activation_max = capture_stats[att_proj_name].to(device)
            
            # Get weight max from att_proj (fused QKV)
            # att_proj shape: [out_features, in_features] where out_features = Q_dim + K_dim + V_dim
            att_proj_weight = block.att_proj.weight
            # Split into Q, K, V dimensions
            # Access config from the block's config
            config = block.config
            head_dim = config.d_model // config.n_heads
            q_dim = config.d_model
            k_dim = config.effective_n_kv_heads * head_dim
            v_dim = config.effective_n_kv_heads * head_dim
            
            w_q_max = att_proj_weight[:q_dim].abs().max(dim=0).values
            w_k_max = att_proj_weight[q_dim:q_dim+k_dim].abs().max(dim=0).values
            w_v_max = att_proj_weight[q_dim+k_dim:q_dim+k_dim+v_dim].abs().max(dim=0).values
            w_max = torch.max(torch.max(w_q_max, w_k_max), w_v_max)
            
        else:  # is_llama
            # LLaDALlamaBlock uses separate q_proj, k_proj, v_proj
            q_proj_name = f"{base_name}.q_proj"
            k_proj_name = f"{base_name}.k_proj"
            v_proj_name = f"{base_name}.v_proj"
            
            if q_proj_name not in capture_stats or k_proj_name not in capture_stats or v_proj_name not in capture_stats:
                print(f"WARNING: Could not find activation stats for Q/K/V in layer {i} (tried {q_proj_name}, {k_proj_name}, {v_proj_name})")
                continue
            
            # Get combined activation max across Q, K, V
            attn_activation_max = capture_stats[q_proj_name].to(device)
            attn_activation_max = torch.max(attn_activation_max, capture_stats[k_proj_name].to(device))
            attn_activation_max = torch.max(attn_activation_max, capture_stats[v_proj_name].to(device))

            # Get combined weight max across Q, K, V
            w_q_max = block.q_proj.weight.abs().max(dim=0).values
            w_k_max = block.k_proj.weight.abs().max(dim=0).values
            w_v_max = block.v_proj.weight.abs().max(dim=0).values
            w_max = torch.max(torch.max(w_q_max, w_k_max), w_v_max)
        
        # Ensure shapes match
        if attn_activation_max.shape != w_max.shape:
            print(f"WARNING: Shape mismatch in layer {i}: activation_max {attn_activation_max.shape} != weight_max {w_max.shape}")
            min_len = min(attn_activation_max.shape[0], w_max.shape[0])
            attn_activation_max = attn_activation_max[:min_len]
            w_max = w_max[:min_len]
        
        # Clamp to avoid division by zero
        attn_activation_max = torch.clamp(attn_activation_max, min=1e-5)
        w_max = torch.clamp(w_max, min=1e-5)

        # Compute smoothing scale (per-channel)
        s_attn_per_channel = attn_activation_max.pow(alpha) / w_max.pow(1 - alpha)
        # Clip per-channel scales to prevent extreme values
        s_attn_per_channel = torch.clamp(s_attn_per_channel, min=0.1, max=10.0)
        
        # Use per-tensor scale (mean of per-channel) for more stable behavior
        s_attn_scalar = s_attn_per_channel.mean().item()
        s_attn = torch.full_like(s_attn_per_channel, s_attn_scalar)
        scale_stats['attn_scales'].append(s_attn.cpu())

        # Apply smoothing: divide LayerNorm, multiply Linear weights
        with torch.no_grad():
            # Scale LayerNorm weights and bias down by s
            block.attn_norm.weight.div_(s_attn)
            if hasattr(block.attn_norm, 'bias') and block.attn_norm.bias is not None:
                block.attn_norm.bias.div_(s_attn)
            
            # Scale attention projection weights up by s to compensate for scaled LayerNorm output
            s_attn_broadcast = s_attn.view(1, -1)
            if is_sequential:
                if s_attn_broadcast.shape[1] != block.att_proj.weight.shape[1]:
                    print(f"WARNING: Using scalar scale for att_proj in layer {i} due to shape mismatch")
                    s_attn_broadcast = s_attn_scalar
                block.att_proj.weight.mul_(s_attn_broadcast)
            else:  # is_llama
                if s_attn_broadcast.shape[1] != block.q_proj.weight.shape[1]:
                    print(f"WARNING: Using scalar scale for Q/K/V in layer {i} due to shape mismatch")
                    s_attn_broadcast = s_attn_scalar
                block.q_proj.weight.mul_(s_attn_broadcast)
                block.k_proj.weight.mul_(s_attn_broadcast)
                block.v_proj.weight.mul_(s_attn_broadcast)
        
        # DON'T scale attn_out - the attention computation is already at original scale
        # after the Q/K/V scaling cancels out the LayerNorm scaling
        # The residual connection will work correctly: x_new = x + attn_out (both at original scale)
        
        # === MLP block smoothing ===
        ff_proj_name = f"{base_name}.ff_proj"
        up_proj_name = f"{base_name}.up_proj"
        
        if ff_proj_name not in capture_stats or up_proj_name not in capture_stats:
            print(f"WARNING: Could not find activation stats for ff_proj/up_proj in layer {i} (tried {ff_proj_name}, {up_proj_name})")
            continue
        
        # Get combined activation max across ff_proj and up_proj
        mlp_activation_max = capture_stats[ff_proj_name].to(device)
        mlp_activation_max = torch.max(mlp_activation_max, capture_stats[up_proj_name].to(device))
        
        # Get combined weight max
        w_ff_max = block.ff_proj.weight.abs().max(dim=0).values
        w_up_max = block.up_proj.weight.abs().max(dim=0).values
        w_max = torch.max(w_ff_max, w_up_max)

        # Ensure shapes match
        if mlp_activation_max.shape != w_max.shape:
            print(f"WARNING: Shape mismatch in MLP layer {i}: activation_max {mlp_activation_max.shape} != weight_max {w_max.shape}")
            min_len = min(mlp_activation_max.shape[0], w_max.shape[0])
            mlp_activation_max = mlp_activation_max[:min_len]
            w_max = w_max[:min_len]

        # Clamp to avoid division by zero
        mlp_activation_max = torch.clamp(mlp_activation_max, min=1e-5)
        w_max = torch.clamp(w_max, min=1e-5)

        # Compute smoothing scale (per-channel)
        s_mlp_per_channel = mlp_activation_max.pow(alpha) / w_max.pow(1 - alpha)
        # Clip per-channel scales to prevent extreme values
        s_mlp_per_channel = torch.clamp(s_mlp_per_channel, min=0.1, max=10.0)
        
        # Use per-tensor scale (mean of per-channel) for more stable behavior
        s_mlp_scalar = s_mlp_per_channel.mean().item()
        s_mlp = torch.full_like(s_mlp_per_channel, s_mlp_scalar)
        scale_stats['mlp_scales'].append(s_mlp.cpu())

        # Apply smoothing: same logic as attention block
        with torch.no_grad():
            # Scale LayerNorm weights and bias down by s
            block.ff_norm.weight.div_(s_mlp)
            if hasattr(block.ff_norm, 'bias') and block.ff_norm.bias is not None:
                block.ff_norm.bias.div_(s_mlp)
            
            # Scale ff_proj/up_proj weights up by s to compensate for scaled LayerNorm output
            s_mlp_broadcast = s_mlp.view(1, -1)
            if s_mlp_broadcast.shape[1] != block.ff_proj.weight.shape[1]:
                print(f"WARNING: Using scalar scale for ff_proj/up_proj in layer {i} due to shape mismatch")
                s_mlp_broadcast = s_mlp_scalar
            block.ff_proj.weight.mul_(s_mlp_broadcast)
            block.up_proj.weight.mul_(s_mlp_broadcast)
        
        # DON'T scale ff_out - the MLP computation is already at original scale
        # after the ff_proj/up_proj scaling cancels out the LayerNorm scaling
        # The residual connection will work correctly: x_new = x + mlp_out (both at original scale)
        
        if (i + 1) % 4 == 0 or i == len(layers) - 1:
            print(f"  Smoothed layers 1-{i+1}/{len(layers)}")
    
    # Log scale statistics
    if scale_stats['attn_scales']:
        all_attn_scales = torch.cat([s.flatten() for s in scale_stats['attn_scales']])
        all_mlp_scales = torch.cat([s.flatten() for s in scale_stats['mlp_scales']])
        print(f"\n  Scale Statistics:")
        print(f"    Attention scales - Mean: {all_attn_scales.mean():.4f}, Std: {all_attn_scales.std():.4f}, "
              f"Min: {all_attn_scales.min():.4f}, Max: {all_attn_scales.max():.4f}")
        print(f"    MLP scales - Mean: {all_mlp_scales.mean():.4f}, Std: {all_mlp_scales.std():.4f}, "
              f"Min: {all_mlp_scales.min():.4f}, Max: {all_mlp_scales.max():.4f}")
    
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
