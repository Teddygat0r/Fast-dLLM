"""
Quantization script for Fast-dLLM with SmoothQuant preprocessing.

This script applies INT4 weight and INT8 activation quantization to the model.
It is recommended to run smooth_quant_chatbot.py first to apply SmoothQuant
weight preprocessing, which improves quantization quality.

Usage:
    # First apply SmoothQuant (recommended)
    python smooth_quant_chatbot.py
    
    # Then load smoothed weights and quantize
    python quantize_chatbot_activations.py --load-smoothed
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import numpy as np
import random
import argparse
import os
from data.quantization_calibration_dataset import FastDLLMCalibrationDataset
from optimum.quanto import quantize, Calibration, qint4, qint8, freeze
from torch.utils.data import DataLoader

model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOOTHED_WEIGHTS_PATH = "models/fast_dllm_smoothquant.pt"

# Layers to exclude from quantization (keep in higher precision)
# lm_head: output layer, critical for generation quality
# embed_tokens: input embeddings, often kept in FP16
QUANTIZATION_EXCLUDE = ["lm_head", "embed_tokens"]

print(f"Loading model: {model_name}...")

torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map=DEVICE,
    trust_remote_code=True
)

# Check if smoothed weights should be loaded
parser = argparse.ArgumentParser(description="Quantize Fast-dLLM model")
parser.add_argument("--load-smoothed", action="store_true",
                    help="Load SmoothQuant-preprocessed weights before quantization")
args = parser.parse_args()

if args.load_smoothed:
    if os.path.exists(SMOOTHED_WEIGHTS_PATH):
        print(f"\nLoading SmoothQuant-preprocessed weights from {SMOOTHED_WEIGHTS_PATH}...")
        state_dict = torch.load(SMOOTHED_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print("✓ Smoothed weights loaded")
    else:
        print(f"\nWARNING: Smoothed weights not found at {SMOOTHED_WEIGHTS_PATH}")
        print("Proceeding without SmoothQuant preprocessing.")
        print("For better results, run: python smooth_quant_chatbot.py first")

print(f"\nQuantizing model: weights=qint4, activations=qint8")
print(f"Excluding layers from quantization: {QUANTIZATION_EXCLUDE}")
quantize(model, weights=qint4, activations=qint8, exclude=QUANTIZATION_EXCLUDE)

# --- MEMORY DIAGNOSTICS START ---
def print_memory_usage(step_name):
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

# Print GPU stats
print_memory_usage("After Model Load")

# Print Hugging Face's internal calculation of footprint
footprint = model.get_memory_footprint() / (1024 ** 3)
print(f"HF Model Footprint:                  {footprint:.2f} GB\n")
# --- MEMORY DIAGNOSTICS END ---

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Initialize conversation
messages = []

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

fix_seed(42)

dataset = FastDLLMCalibrationDataset(
    tokenizer=tokenizer,
    seq_len=512,
    samples=128,
    block_size=32
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Starting Calibration Pass (Forward Prop Only)...")
with Calibration(momentum=0.9):
    for batch_idx, batch in enumerate(dataloader):
        # Move inputs to GPU
        input_ids = batch['input_ids'].to(DEVICE)
        timesteps = batch['timestep'].to(DEVICE)
        
        with torch.no_grad():
            model(input_ids=input_ids, timesteps=timesteps)
            
        if (batch_idx + 1) % 5 == 0:
            print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

print("Calibration complete.")

print("Freezing model to integer representation...")
freeze(model)

save_path = "models/fast_dllm_quantized_w4a8_more_full.pt"
print(f"Saving complete quantized model to {save_path}...")
torch.save(model, save_path)
print(f"✓ Complete quantized model saved to {save_path}")

# Also save state_dict for backwards compatibility
save_path_state_dict = "models/fast_dllm_quantized_w4a8_more.pt"
print(f"\nSaving state_dict for backwards compatibility to {save_path_state_dict}...")
torch.save(model.state_dict(), save_path_state_dict)
print(f"✓ State dict saved to {save_path_state_dict}")

print("\nSummary:")
print("- Full model: fast_dllm_quantized_w4a8_more_full.pt (recommended for fast loading)")
print("- State dict: fast_dllm_quantized_w4a8_more.pt (backwards compatible)")