from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import numpy as np
import random

model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"

bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
)

print(f"Loading model: {model_name}...")

torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda:0",
    trust_remote_code=True
    # quantization_config=bnb_config_8bit,
)

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
print("Chatbot started! Type 'exit' to quit the conversation.")
print("Type 'clear' to clear conversation history.")
print("-" * 50)

while True:
    # Get user input
    user_input = input("User: ").strip()
    
    # Check if exit
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    
    # Check if clear conversation history
    if user_input.lower() == "clear":
        messages = messages[:1] if messages[0]["role"] == "system" else messages[:0]
        print("Conversation history cleared!")
        print_memory_usage("After Clear") # Optional: Check memory after clear
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
    
    # Optional: Check memory after a generation turn to see how much context is consuming
    # print_memory_usage("After Generation")
    
    print("-" * 50)