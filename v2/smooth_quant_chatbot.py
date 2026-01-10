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
import os

# Configuration
model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"
BATCH_SIZE = 32
ALPHA = 0.5  # SmoothQuant migration strength (0.5 is typical)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "models"
SAVE_PATH = os.path.join(SAVE_DIR, "fast_dllm_smoothquant.pt")

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


def fuse_smoothquant(model, capture_stats, alpha=0.5, device="cuda"):
    """
    Apply SmoothQuant weight preprocessing to the model.
    
    Migrates quantization difficulty from activations to weights by:
    1. Dividing LayerNorm weights by scale s
    2. Multiplying subsequent Linear weights by scale s
    
    Scale s = act_max^alpha / weight_max^(1-alpha)
    """
    layers = model.model.layers
    
    for i, layer in enumerate(layers):
        # === Attention block smoothing ===
        q_layer_name = f"model.layers.{i}.self_attn.q_proj"
        k_layer_name = f"model.layers.{i}.self_attn.k_proj"
        v_layer_name = f"model.layers.{i}.self_attn.v_proj"
        
        # Get combined activation max across Q, K, V
        attn_activation_max = capture_stats[q_layer_name].to(device)
        attn_activation_max = torch.max(attn_activation_max, capture_stats[k_layer_name].to(device))
        attn_activation_max = torch.max(attn_activation_max, capture_stats[v_layer_name].to(device))

        # Get combined weight max across Q, K, V
        w_q_max = layer.self_attn.q_proj.weight.abs().max(dim=0).values
        w_k_max = layer.self_attn.k_proj.weight.abs().max(dim=0).values
        w_v_max = layer.self_attn.v_proj.weight.abs().max(dim=0).values
        w_max = torch.max(torch.max(w_q_max, w_k_max), w_v_max)
        
        # Clamp to avoid division by zero
        attn_activation_max = torch.clamp(attn_activation_max, min=1e-5)
        w_max = torch.clamp(w_max, min=1e-5)

        # Compute smoothing scale
        s_attn = attn_activation_max.pow(alpha) / w_max.pow(1 - alpha)

        # Apply smoothing: divide LayerNorm, multiply Linear weights
        with torch.no_grad():
            layer.input_layernorm.weight.div_(s_attn)
            if hasattr(layer.input_layernorm, 'bias') and layer.input_layernorm.bias is not None:
                layer.input_layernorm.bias.div_(s_attn)
            
            layer.self_attn.q_proj.weight.mul_(s_attn.view(1, -1))
            layer.self_attn.k_proj.weight.mul_(s_attn.view(1, -1))
            layer.self_attn.v_proj.weight.mul_(s_attn.view(1, -1))
        
        # === MLP block smoothing ===
        gate_layer_name = f"model.layers.{i}.mlp.gate_proj"
        up_layer_name = f"model.layers.{i}.mlp.up_proj"
        
        # Get combined activation max across gate and up projections
        mlp_activation_max = capture_stats[gate_layer_name].to(device)
        mlp_activation_max = torch.max(mlp_activation_max, capture_stats[up_layer_name].to(device))
        
        # Get combined weight max
        w_gate_max = layer.mlp.gate_proj.weight.abs().max(dim=0).values
        w_up_max = layer.mlp.up_proj.weight.abs().max(dim=0).values
        w_max = torch.max(w_gate_max, w_up_max)

        # Clamp to avoid division by zero
        mlp_activation_max = torch.clamp(mlp_activation_max, min=1e-5)
        w_max = torch.clamp(w_max, min=1e-5)

        # Compute smoothing scale
        s_mlp = mlp_activation_max.pow(alpha) / w_max.pow(1 - alpha)

        # Apply smoothing
        with torch.no_grad():
            layer.post_attention_layernorm.weight.div_(s_mlp)
            if hasattr(layer.post_attention_layernorm, 'bias') and layer.post_attention_layernorm.bias is not None:
                layer.post_attention_layernorm.bias.div_(s_mlp)
            
            layer.mlp.gate_proj.weight.mul_(s_mlp.view(1, -1))
            layer.mlp.up_proj.weight.mul_(s_mlp.view(1, -1))
        
        if (i + 1) % 4 == 0 or i == len(layers) - 1:
            print(f"  Smoothed layers 1-{i+1}/{len(layers)}")
    
    return model


# === Main execution ===
if __name__ == "__main__":
    print("\n=== SmoothQuant Preprocessing ===")
    print(f"Alpha: {ALPHA}")
    print(f"Device: {DEVICE}")
    
    # Step 1: Register hooks for activation capture
    print("\n[Step 1/4] Registering activation capture hooks...")
    capture = ActivationCapture(model)
    capture.register_hooks()
    print(f"  Registered hooks on {len(capture.hooks)} Linear layers")

    # Step 2: Calibration pass to collect activation statistics
    print("\n[Step 2/4] Running calibration pass on WikiText-2...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    batch_count = (len(dataset) + (BATCH_SIZE - 1)) // BATCH_SIZE

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            tokenized_text = tokenizer(
                batch["text"], 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(DEVICE)
            model(**tokenized_text)
            if (batch_idx + 1) % 20 == 0 or batch_idx == batch_count - 1:
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