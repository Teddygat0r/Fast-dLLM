"""
Simple test script to verify text generation works with the regular model.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map=DEVICE,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model.eval()

# Check if model has generate method and what it expects
print(f"Model type: {type(model)}")
print(f"Has generate method: {hasattr(model, 'generate')}")
if hasattr(model, 'generate'):
    import inspect
    sig = inspect.signature(model.generate)
    print(f"Generate signature: {sig}")

print("Model loaded. Testing generation...")
print("-" * 60)

test_prompts = [
    "The capital of France is",
    "In machine learning, a neural network is",
]

for prompt in test_prompts:
    print(f"\nPrompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt")
    # Check device
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Input device before move: {inputs['input_ids'].device}")
    inputs = inputs.to(model.device)
    print(f"Input device after move: {inputs['input_ids'].device}")
    input_length = inputs["input_ids"].shape[1]
    
    print(f"Input length: {input_length}")
    print("Generating...")
    
    with torch.no_grad():
        # Try with threshold=1.0 (default) and tokenizer
        print("Trying with threshold=1.0 and tokenizer...")
        try:
            outputs = model.generate(
                inputs["input_ids"],
                tokenizer=tokenizer,
                max_new_tokens=30,
                block_size=32,
                small_block_size=8,
                threshold=1.0,  # Use default threshold
            )
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print(f"Output shape: {outputs.shape}")
    print(f"Output length: {outputs.shape[1]}")
    print(f"Generated tokens: {outputs.shape[1] - input_length}")
    
    # Decode full output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Full output: '{full_output}'")
    
    # Decode only generated tokens
    generated_tokens = outputs[0][input_length:]
    generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Generated only: '{generated}'")
    print("-" * 60)

print("\nTest complete!")
