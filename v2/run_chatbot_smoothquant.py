from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import random
import os

model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"
smoothed_weights_path = "models/fast_dllm_smoothquant.pt"

print(f"Loading base model: {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda:0",
    trust_remote_code=True
)

# Load SmoothQuant-preprocessed weights
if os.path.exists(smoothed_weights_path):
    print(f"\nLoading SmoothQuant-preprocessed weights from {smoothed_weights_path}...")
    state_dict = torch.load(smoothed_weights_path, map_location="cuda:0", weights_only=True)
    model.load_state_dict(state_dict)
    print("✓ Smoothed weights loaded successfully!")
else:
    print(f"\nWARNING: Smoothed weights not found at {smoothed_weights_path}")
    print("Proceeding with original (non-smoothed) model.")
    print("To generate smoothed weights, run: python smooth_quant_chatbot.py")

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
print("\n" + "=" * 50)
print("SmoothQuant Chatbot started!")
print("(Model with SmoothQuant preprocessing, before quantization)")
print("Type 'exit' to quit the conversation.")
print("Type 'clear' to clear conversation history.")
print("=" * 50)

while True:
    # Get user input
    user_input = input("\nUser: ").strip()
    
    # Check if exit
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    
    # Check if clear conversation history
    if user_input.lower() == "clear":
        messages = messages[:1] if messages and messages[0].get("role") == "system" else []
        print("Conversation history cleared!")
        continue
    
    if not user_input:
        continue
    
    messages.append({"role": "user", "content": user_input})
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        model_inputs["input_ids"],
        tokenizer=tokenizer,
        block_size=32,
        max_new_tokens=2048,
        small_block_size=8,
        threshold=0.9,
    )
    response = tokenizer.decode(generated_ids[0][model_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"AI: {response}")
    
    # Add AI response to conversation history
    messages.append({"role": "assistant", "content": response})
    
    print("-" * 50)
