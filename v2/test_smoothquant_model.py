"""
Test script for the SmoothQuant-preprocessed Fast-dLLM model.

This script loads the smoothed model weights and runs test generations
to verify the model still works correctly after weight transformations.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import random
import os

# Configuration
model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"
smoothed_weights_path = "models/fast_dllm_smoothquant.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fix_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def print_memory_usage(step_name):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"[Memory - {step_name}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def main():
    print("=" * 60)
    print("SmoothQuant Model Test")
    print("=" * 60)
    
    fix_seed(42)
    torch.cuda.empty_cache()
    
    # Check if smoothed weights exist
    if not os.path.exists(smoothed_weights_path):
        print(f"ERROR: Smoothed weights not found at {smoothed_weights_path}")
        print("Please run smooth_quant_chatbot.py first to generate the weights.")
        return
    
    # Step 1: Load base model
    print(f"\n[1/3] Loading base model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=DEVICE,
        trust_remote_code=True
    )
    print("✓ Base model loaded")
    
    # Step 2: Load smoothed weights
    print(f"\n[2/3] Loading smoothed weights from {smoothed_weights_path}...")
    state_dict = torch.load(smoothed_weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    print("✓ Smoothed weights loaded")
    print_memory_usage("After loading")
    
    # Step 3: Load tokenizer
    print(f"\n[3/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✓ Tokenizer loaded")
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "In machine learning, a neural network is",
        "The quick brown fox",
    ]
    
    print("\n" + "=" * 60)
    print("Running Generation Tests")
    print("=" * 60)
    
    model.eval()
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        print(f"Prompt: \"{prompt}\"")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                tokenizer=tokenizer,
                max_new_tokens=50,
                block_size=32,
                small_block_size=8,
                threshold=0.9,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output: \"{generated_text}\"")
    
    # Interactive test
    print("\n" + "=" * 60)
    print("Interactive Chat Test")
    print("=" * 60)
    print("Type a message to test, or 'quit' to exit.\n")
    
    messages = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Test complete!")
            break
        
        if not user_input:
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs["input_ids"],
                tokenizer=tokenizer,
                block_size=32,
                max_new_tokens=256,
                small_block_size=8,
                threshold=0.9,
            )
        
        response = tokenizer.decode(
            generated_ids[0][model_inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        print(f"AI: {response}\n")
        
        messages.append({"role": "assistant", "content": response})
    
    print_memory_usage("Final")
    print("\n✓ All tests completed successfully!")


if __name__ == "__main__":
    main()
