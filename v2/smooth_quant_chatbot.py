"""
SmoothQuant Preprocessing for Fast-dLLM

This script applies SmoothQuant weight preprocessing to the Fast-dLLM model.
It collects activation statistics during a calibration pass, then migrates
quantization difficulty from activations to weights by scaling.

The smoothed model is saved for subsequent quantization.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from data.quantization_calibration_dataset import FastDLLMCalibrationDataset
import os
import argparse

# Configuration
model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"
BATCH_SIZE = 32
ALPHA = 0.3  # SmoothQuant migration strength (lower = less aggressive, 0.3 is more conservative)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "models"
SAVE_PATH = os.path.join(SAVE_DIR, "fast_dllm_smoothquant.pt")
CALIBRATION_SAMPLES = 128  # Number of calibration samples

print(f"Loading model: {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map=DEVICE,
    trust_remote_code=True
)


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


def fuse_smoothquant(model, capture_stats, alpha=0.3, device="cuda"):
    """
    Apply SmoothQuant weight preprocessing to the model.
    
    Migrates quantization difficulty from activations to weights by:
    1. Dividing LayerNorm weights by scale s
    2. Multiplying subsequent Linear weights by scale s
    3. Dividing output projection weights by scale s to maintain residual connections
    
    Scale s = act_max^alpha / weight_max^(1-alpha)
    
    Uses per-tensor scaling (mean of per-channel scales) for stability.
    Lower alpha (default 0.3) is more conservative and preserves model quality better.
    """
    layers = model.model.layers
    scale_stats = {
        'attn_scales': [],
        'mlp_scales': []
    }
    
    for i, layer in enumerate(layers):
        # === Attention block smoothing ===
        q_layer_name = f"model.layers.{i}.self_attn.q_proj"
        k_layer_name = f"model.layers.{i}.self_attn.k_proj"
        v_layer_name = f"model.layers.{i}.self_attn.v_proj"
        o_layer_name = f"model.layers.{i}.self_attn.o_proj"
        
        # Get combined activation max across Q, K, V
        attn_activation_max = capture_stats[q_layer_name].to(device)
        attn_activation_max = torch.max(attn_activation_max, capture_stats[k_layer_name].to(device))
        attn_activation_max = torch.max(attn_activation_max, capture_stats[v_layer_name].to(device))

        # Get combined weight max across Q, K, V
        # For weights [out_features, in_features], max(dim=0) gives max per input channel
        w_q_max = layer.self_attn.q_proj.weight.abs().max(dim=0).values
        w_k_max = layer.self_attn.k_proj.weight.abs().max(dim=0).values
        w_v_max = layer.self_attn.v_proj.weight.abs().max(dim=0).values
        w_max = torch.max(torch.max(w_q_max, w_k_max), w_v_max)
        
        # Ensure shapes match
        if attn_activation_max.shape != w_max.shape:
            print(f"WARNING: Shape mismatch in layer {i}: activation_max {attn_activation_max.shape} != weight_max {w_max.shape}")
            # Try to align shapes - take min length
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
        # This is more conservative and prevents extreme per-channel variations
        s_attn_scalar = s_attn_per_channel.mean().item()
        s_attn = torch.full_like(s_attn_per_channel, s_attn_scalar)
        scale_stats['attn_scales'].append(s_attn.cpu())

        # Apply smoothing: divide LayerNorm, multiply Linear weights
        # Math: After scaling, h = LN(x)/s, then Q/K/V(h*s) = Q/K/V(LN(x)) (original scale)
        # So attention output is at original scale, and residual connection works correctly
        with torch.no_grad():
            # Scale LayerNorm weights and bias down by s
            layer.input_layernorm.weight.div_(s_attn)
            if hasattr(layer.input_layernorm, 'bias') and layer.input_layernorm.bias is not None:
                layer.input_layernorm.bias.div_(s_attn)
            
            # Scale Q/K/V weights up by s to compensate for scaled LayerNorm output
            # Weight shape: [out_features, in_features], scale shape: [in_features]
            # Broadcast: [1, in_features] * [out_features, in_features]
            s_attn_broadcast = s_attn.view(1, -1)
            if s_attn_broadcast.shape[1] != layer.self_attn.q_proj.weight.shape[1]:
                # Fallback: use scalar if shape mismatch
                print(f"WARNING: Using scalar scale for Q/K/V in layer {i} due to shape mismatch")
                s_attn_broadcast = s_attn_scalar
            layer.self_attn.q_proj.weight.mul_(s_attn_broadcast)
            layer.self_attn.k_proj.weight.mul_(s_attn_broadcast)
            layer.self_attn.v_proj.weight.mul_(s_attn_broadcast)
        
        # DON'T scale o_proj - the attention computation is already at original scale
        # after the Q/K/V scaling cancels out the LayerNorm scaling
        # The residual connection will work correctly: x_new = x + attn_out (both at original scale)
        
        # === MLP block smoothing ===
        gate_layer_name = f"model.layers.{i}.mlp.gate_proj"
        up_layer_name = f"model.layers.{i}.mlp.up_proj"
        down_layer_name = f"model.layers.{i}.mlp.down_proj"
        
        # Get combined activation max across gate and up projections
        mlp_activation_max = capture_stats[gate_layer_name].to(device)
        mlp_activation_max = torch.max(mlp_activation_max, capture_stats[up_layer_name].to(device))
        
        # Get combined weight max
        w_gate_max = layer.mlp.gate_proj.weight.abs().max(dim=0).values
        w_up_max = layer.mlp.up_proj.weight.abs().max(dim=0).values
        w_max = torch.max(w_gate_max, w_up_max)

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
            layer.post_attention_layernorm.weight.div_(s_mlp)
            if hasattr(layer.post_attention_layernorm, 'bias') and layer.post_attention_layernorm.bias is not None:
                layer.post_attention_layernorm.bias.div_(s_mlp)
            
            # Scale gate/up weights up by s to compensate for scaled LayerNorm output
            s_mlp_broadcast = s_mlp.view(1, -1)
            if s_mlp_broadcast.shape[1] != layer.mlp.gate_proj.weight.shape[1]:
                print(f"WARNING: Using scalar scale for gate/up in layer {i} due to shape mismatch")
                s_mlp_broadcast = s_mlp_scalar
            layer.mlp.gate_proj.weight.mul_(s_mlp_broadcast)
            layer.mlp.up_proj.weight.mul_(s_mlp_broadcast)
        
        # DON'T scale down_proj - the MLP computation is already at original scale
        # after the gate/up scaling cancels out the LayerNorm scaling
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
    parser = argparse.ArgumentParser(description="SmoothQuant preprocessing for Fast-dLLM")
    parser.add_argument("--alpha", type=float, default=ALPHA, 
                        help=f"SmoothQuant migration strength (default: {ALPHA})")
    parser.add_argument("--calibration-samples", type=int, default=CALIBRATION_SAMPLES,
                        help=f"Number of calibration samples (default: {CALIBRATION_SAMPLES})")
    args = parser.parse_args()
    
    ALPHA = args.alpha
    CALIBRATION_SAMPLES = args.calibration_samples
    
    print("\n=== SmoothQuant Preprocessing ===")
    print(f"Alpha: {ALPHA}")
    print(f"Calibration samples: {CALIBRATION_SAMPLES}")
    print(f"Device: {DEVICE}")
    
    # Step 1: Register hooks for activation capture
    print("\n[Step 1/4] Registering activation capture hooks...")
    capture = ActivationCapture(model)
    capture.register_hooks()
    print(f"  Registered hooks on {len(capture.hooks)} Linear layers")

    # Step 2: Calibration pass to collect activation statistics with timesteps
    print("\n[Step 2/4] Running calibration pass with Fast-dLLM calibration dataset (with timesteps)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset = FastDLLMCalibrationDataset(
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
            timesteps = batch['timestep'].to(DEVICE)
            model(input_ids=input_ids, timesteps=timesteps)
            if (batch_idx + 1) % 5 == 0 or batch_idx == batch_count - 1:
                print(f"  Processed {batch_idx + 1}/{batch_count} batches")

    # Step 3: Clear hooks and apply SmoothQuant
    print("\n[Step 3/4] Applying SmoothQuant weight transformations...")
    capture_stats = capture.stats
    capture.clear_hooks()
    print(f"  Collected stats for {len(capture_stats)} layers")
    
    model = fuse_smoothquant(model, capture_stats, alpha=ALPHA, device=DEVICE)

    # Step 4: Save the smoothed model
    print(f"\n[Step 4/4] Saving smoothed model...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"  ✓ Saved smoothed model state_dict to: {SAVE_PATH}")

    print("\n=== SmoothQuant Preprocessing Complete ===")
    print(f"The smoothed model can now be loaded and quantized.")
    print(f"Next step: Load the state_dict and apply INT8/INT4 quantization.")