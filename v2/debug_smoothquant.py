"""
Diagnostic script for SmoothQuant debugging.

This script compares model behavior before and after SmoothQuant:
- Activation statistics
- Logit distributions
- Per-layer scale visualization
- Performance metrics
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import numpy as np
import os
from data.quantization_calibration_dataset import FastDLLMCalibrationDataset
from torch.utils.data import DataLoader

# Configuration
model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"
smoothed_weights_path = "models/fast_dllm_smoothquant.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_TEST_SAMPLES = 16


class ActivationCollector:
    """Collects activation statistics from model layers."""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activations = {}
        
    def _hook_fn(self, name):
        def hook(module, input, output):
            if input[0] is not None:
                x = input[0].detach().cpu()
                if name not in self.activations:
                    self.activations[name] = []
                self.activations[name].append(x)
        return hook
    
    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_statistics(self):
        """Compute statistics for collected activations."""
        stats = {}
        for name, acts in self.activations.items():
            if acts:
                all_acts = torch.cat([a.flatten() for a in acts])
                stats[name] = {
                    'mean': all_acts.mean().item(),
                    'std': all_acts.std().item(),
                    'min': all_acts.min().item(),
                    'max': all_acts.max().item(),
                    'abs_max': all_acts.abs().max().item(),
                }
        return stats
    
    def clear(self):
        self.activations = {}


def compare_activations(model_before, model_after, tokenizer, device):
    """Compare activation statistics between two models."""
    print("\n" + "=" * 60)
    print("Activation Statistics Comparison")
    print("=" * 60)
    
    # Create test dataset
    dataset = FastDLLMCalibrationDataset(
        tokenizer=tokenizer,
        seq_len=512,
        samples=NUM_TEST_SAMPLES,
        block_size=32
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Collect activations from original model
    print("\n[1/2] Collecting activations from original model...")
    collector_before = ActivationCollector(model_before)
    collector_before.register_hooks()
    
    model_before.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            timesteps = batch['timestep'].to(device)
            model_before(input_ids=input_ids, timesteps=timesteps)
            if batch_idx >= len(dataloader) - 1:
                break
    
    stats_before = collector_before.get_statistics()
    collector_before.clear_hooks()
    
    # Collect activations from smoothed model
    print("[2/2] Collecting activations from smoothed model...")
    collector_after = ActivationCollector(model_after)
    collector_after.register_hooks()
    
    model_after.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            timesteps = batch['timestep'].to(device)
            model_after(input_ids=input_ids, timesteps=timesteps)
            if batch_idx >= len(dataloader) - 1:
                break
    
    stats_after = collector_after.get_statistics()
    collector_after.clear_hooks()
    
    # Compare statistics
    print("\n" + "-" * 60)
    print("Layer-wise Activation Comparison (sample layers):")
    print("-" * 60)
    
    common_layers = set(stats_before.keys()) & set(stats_after.keys())
    sample_layers = list(common_layers)[:10]  # Show first 10 layers
    
    for layer_name in sample_layers:
        before = stats_before[layer_name]
        after = stats_after[layer_name]
        
        mean_diff = abs(after['mean'] - before['mean']) / (abs(before['mean']) + 1e-8)
        std_diff = abs(after['std'] - before['std']) / (abs(before['std']) + 1e-8)
        max_diff = abs(after['abs_max'] - before['abs_max']) / (abs(before['abs_max']) + 1e-8)
        
        print(f"\n{layer_name}:")
        print(f"  Mean: {before['mean']:.4f} -> {after['mean']:.4f} (diff: {mean_diff*100:.2f}%)")
        print(f"  Std:  {before['std']:.4f} -> {after['std']:.4f} (diff: {std_diff*100:.2f}%)")
        print(f"  Max:  {before['abs_max']:.4f} -> {after['abs_max']:.4f} (diff: {max_diff*100:.2f}%)")
    
    return stats_before, stats_after


def compare_logits(model_before, model_after, tokenizer, device):
    """Compare logit distributions between two models."""
    print("\n" + "=" * 60)
    print("Logit Distribution Comparison")
    print("=" * 60)
    
    # Create test dataset
    dataset = FastDLLMCalibrationDataset(
        tokenizer=tokenizer,
        seq_len=128,
        samples=NUM_TEST_SAMPLES,
        block_size=32
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_logits_before = []
    all_logits_after = []
    
    model_before.eval()
    model_after.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            timesteps = batch['timestep'].to(device)
            
            # Get logits from both models
            output_before = model_before(input_ids=input_ids, timesteps=timesteps)
            output_after = model_after(input_ids=input_ids, timesteps=timesteps)
            
            logits_before = output_before.logits
            logits_after = output_after.logits
            
            all_logits_before.append(logits_before.cpu())
            all_logits_after.append(logits_after.cpu())
            
            if batch_idx >= len(dataloader) - 1:
                break
    
    all_logits_before = torch.cat([l.flatten() for l in all_logits_before])
    all_logits_after = torch.cat([l.flatten() for l in all_logits_after])
    
    # Compute statistics
    print(f"\nLogit Statistics:")
    print(f"  Before - Mean: {all_logits_before.mean():.4f}, Std: {all_logits_before.std():.4f}, "
          f"Min: {all_logits_before.min():.4f}, Max: {all_logits_before.max():.4f}")
    print(f"  After  - Mean: {all_logits_after.mean():.4f}, Std: {all_logits_after.std():.4f}, "
          f"Min: {all_logits_after.min():.4f}, Max: {all_logits_after.max():.4f}")
    
    # Compute KL divergence (approximate)
    mean_diff = abs(all_logits_after.mean() - all_logits_before.mean())
    std_diff = abs(all_logits_after.std() - all_logits_before.std())
    print(f"\n  Difference - Mean: {mean_diff:.4f}, Std: {std_diff:.4f}")
    
    # Check for NaN or Inf
    has_nan_before = torch.isnan(all_logits_before).any().item()
    has_nan_after = torch.isnan(all_logits_after).any().item()
    has_inf_before = torch.isinf(all_logits_before).any().item()
    has_inf_after = torch.isinf(all_logits_after).any().item()
    
    print(f"\n  NaN check - Before: {has_nan_before}, After: {has_nan_after}")
    print(f"  Inf check - Before: {has_inf_before}, After: {has_inf_after}")
    
    return all_logits_before, all_logits_after


def test_generation_quality(model_before, model_after, tokenizer, device):
    """Test generation quality on sample prompts."""
    print("\n" + "=" * 60)
    print("Generation Quality Test")
    print("=" * 60)
    
    test_prompts = [
        "The capital of France is",
        "In machine learning, a neural network is",
        "The quick brown fox",
    ]
    
    print("\nGenerating with original model:")
    model_before.eval()
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]
        print(f"  Input length: {input_length}")
        with torch.no_grad():
            outputs = model_before.generate(
                inputs["input_ids"],
                tokenizer=tokenizer,
                max_new_tokens=50,  # Match test_smoothquant_model.py
                block_size=32,
                small_block_size=8,
                threshold=0.9,
            )
        print(f"  Output length: {outputs.shape[1]}, Generated: {outputs.shape[1] - input_length} tokens")
        # Decode full output first to see what we got
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Decode only the newly generated tokens (exclude the input prompt)
        generated_tokens = outputs[0][input_length:]
        generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"  Prompt: '{prompt}'")
        print(f"  Full output: '{full_output}'")
        print(f"  Generated only: '{generated}'\n")
    
    print("\nGenerating with smoothed model:")
    model_after.eval()
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]
        print(f"  Input length: {input_length}")
        with torch.no_grad():
            outputs = model_after.generate(
                inputs["input_ids"],
                tokenizer=tokenizer,
                max_new_tokens=50,  # Match test_smoothquant_model.py
                block_size=32,
                small_block_size=8,
                threshold=0.9,
            )
        print(f"  Output length: {outputs.shape[1]}, Generated: {outputs.shape[1] - input_length} tokens")
        # Decode full output first to see what we got
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Decode only the newly generated tokens (exclude the input prompt)
        generated_tokens = outputs[0][input_length:]
        generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"  Prompt: '{prompt}'")
        print(f"  Full output: '{full_output}'")
        print(f"  Generated only: '{generated}'\n")


def main():
    print("=" * 60)
    print("SmoothQuant Diagnostic Tool")
    print("=" * 60)
    
    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✓ Tokenizer loaded")
    
    # Load original model
    print("\n[2/4] Loading original model...")
    model_before = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=DEVICE,
        trust_remote_code=True
    )
    model_before.eval()
    print("✓ Original model loaded")
    
    # Load smoothed model
    print("\n[3/4] Loading smoothed model...")
    if not os.path.exists(smoothed_weights_path):
        print(f"ERROR: Smoothed weights not found at {smoothed_weights_path}")
        print("Please run smooth_quant_chatbot.py first.")
        return
    
    model_after = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=DEVICE,
        trust_remote_code=True
    )
    state_dict = torch.load(smoothed_weights_path, map_location=DEVICE, weights_only=True)
    model_after.load_state_dict(state_dict)
    model_after.eval()
    print("✓ Smoothed model loaded")
    
    # Run diagnostics
    print("\n[4/4] Running diagnostics...")
    
    # Compare activations
    stats_before, stats_after = compare_activations(model_before, model_after, tokenizer, DEVICE)
    
    # Compare logits
    logits_before, logits_after = compare_logits(model_before, model_after, tokenizer, DEVICE)
    
    # Test generation
    test_generation_quality(model_before, model_after, tokenizer, DEVICE)
    
    print("\n" + "=" * 60)
    print("Diagnostics Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
