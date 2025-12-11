import torch
import numpy as np
import gradio as gr
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.quanto import quantize, qint4, qint8, freeze
import time
import random
import types
import os
from generation_functions import setup_model_with_custom_generation


# Check available GPU
device_accelerated = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f"Accelerated model using device: {device_accelerated}")

# Set random seed
def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

fix_seed(42)

model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"

print(f"Loading model: {model_name}...")

torch.cuda.empty_cache()

# Check if complete quantized model exists
import os
full_model_path = "models/fast_dllm_quantized_w4a8_full.pt"
state_dict_path = "models/fast_dllm_quantized_w4a8.pt"

if os.path.exists(full_model_path):
    # Fast loading: Load complete quantized model directly
    print(f"Loading complete quantized model from {full_model_path}...")
    print("This method is much faster as it avoids reconstructing the quantization structure.")
    
    # Pre-load model class definitions so transformers_modules is populated
    print("Pre-loading model class definitions...")
    _ = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="meta",  # Use 'meta' device to avoid loading weights
        trust_remote_code=True
    )
    print("✓ Model class loaded")
    
    # Now load the quantized checkpoint
    # PyTorch 2.6+ requires weights_only=False for complete models with custom classes
    print("Loading quantized checkpoint...")
    model = torch.load(full_model_path, map_location=device_accelerated, weights_only=False)
    print("✓ Model loaded successfully!")
    
else:
    # Fallback: Traditional method - load base model, quantize, then load state_dict
    print(f"Complete model not found. Using fallback method (slower)...")
    print(f"Tip: Run quantize_chatbot_activations.py to generate {full_model_path} for faster loading.")
    
    # Load model on CPU first for quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True
    )

    # Quantize model
    print("Quantizing model...")
    quantize(model, weights=qint4, activations=qint8, exclude=["lm_head"])

    print("Freezing model to integer representation...")
    freeze(model)

    # Load quantized weights
    if os.path.exists(state_dict_path):
        print(f"Loading quantized model weights from {state_dict_path}...")
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
        print("✓ Weights loaded successfully!")
    else:
        print(f"WARNING: Neither {full_model_path} nor {state_dict_path} found!")
        print("The model will run but may not be properly quantized.")

    print("Moving model to CUDA...")
    model.to(device_accelerated)


# Set up custom generation functions for visualization
model = setup_model_with_custom_generation(model)

# Memory diagnostics function
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
    
    return f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Overhead: {overhead:.2f} GB"

# Print GPU stats after model load
memory_stats = print_memory_usage("After Model Load")

# Print Hugging Face's internal calculation of footprint
footprint = model.get_memory_footprint() / (1024 ** 3)
print(f"HF Model Footprint:                  {footprint:.2f} GB\n")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Constants
MASK_TOKEN = "[MASK]"
MASK_ID = 151665  # mask_id for Fast_dLLM model
question_ai = '''Write a piece of code to implement quick sort.'''
question_math = '''A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?'''
question_gsm8k = '''Question: Skyler has 100 hats on his hand with the colors red, blue, and white. Half of the hats are red, 3/5 of the remaining hats are blue, and the rest are white. How many white hats does Skyler have?'''


def format_chat_history(history):
    """
    Format chat history for the model
    
    Args:
        history: List of [user_message, assistant_message] pairs
        
    Returns:
        Formatted conversation for the model
    """
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:  # Skip if None (for the latest user message)
            messages.append({"role": "assistant", "content": assistant_msg})
    
    return messages


@torch.no_grad()
def generate_response_with_visualization(model, tokenizer, device, messages, max_new_tokens=1024, 
                                         temperature=0.0, block_length=32,
                                         threshold=0.9, top_p=0.9):
    """
    Generate text with quantized Fast_dLLM model with visualization
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        block_length: Block size for generation
        threshold: Threshold for generation
        top_p: Top-p sampling parameter
        
    Yields:
        Visualization states showing the progression and final text
    """
    
    # Prepare the prompt using chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    # Use custom mdm_sample_with_visualization method
    generator = model.mdm_sample_with_visualization(
        model_inputs["input_ids"],
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        small_block_size=block_length,
        temperature=temperature,
        threshold=threshold,
        top_p=top_p,
    )
    
    # Collect all states and final text
    states = []
    for item in generator:
        if isinstance(item, list):  # Visualization state
            states.append(item)
            yield item
        else:  # Final text
            final_text = item
            break
    
    # Return final text
    yield final_text


css = '''
.category-legend{display:none}
.message, .bubble, .chatbot .message, .chatbot .bubble {
    max-width: 80% !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    box-sizing: border-box !important;
}
/* HighlightedText allows auto line wrapping and sets fixed height */
.highlighted-text-container {
    white-space: pre-wrap !important;
    word-break: break-word !important;
    height: 200px !important;
    overflow-y: auto !important;
}
.generating {
    border: none;
}
#input-row {
    align-items: center !important;
}
'''

def create_chatbot_demo():
    with gr.Blocks(css=css) as demo:
        gr.Markdown("# Fast-dLLM Quantized Chatbot with Visualization")
        gr.Markdown("**Quantized Model (4-bit weights, 8-bit activations)** - [code](https://github.com/NVlabs/Fast-dLLM), [project page](https://nvlabs.github.io/Fast-dLLM/)")
        
        # STATE MANAGEMENT
        chat_history_cache = gr.State([])
        
        # UI COMPONENTS
        
        # Display memory stats at the top
        with gr.Row():
            memory_display = gr.Textbox(
                label="GPU Memory Usage",
                value=memory_stats,
                interactive=False
            )
        
        # Input area
        with gr.Group():
            with gr.Row(elem_id="input-row"):
                user_input = gr.Textbox(
                    label="Your Message", 
                    placeholder="Type your message here...",
                    show_label=False,
                    scale=8
                )
                send_btn = gr.Button("Send", scale=1)
                clear_btn = gr.Button("Clear Conversation", scale=1)
        
        # Quantized model conversation interface
        gr.Markdown("## Quantized Fast-dLLM Model (7B Parameters, 4-bit/8-bit)")
        with gr.Row():
            with gr.Column(scale=2):
                chatbot_ui = gr.Chatbot(label="Conversation (Quantized Model)", height=520)
            with gr.Column(scale=2):
                with gr.Row():
                    generation_time = gr.Textbox(
                        label="Generation Time",
                        value="wait for generation",
                        interactive=False
                    )
                    throughput = gr.Textbox(
                        label="Generation Speed",
                        value="wait for generation",
                        interactive=False
                    )
                output_vis = gr.HighlightedText(
                    label="Denoising Process Visualization (Real-time)",
                    combine_adjacent=False,
                    show_legend=True,
                    elem_classes=["highlighted-text-container"]
                )
                output_vis_slow = gr.HighlightedText(
                    label="Denoising Process Visualization (Slow Motion)",
                    combine_adjacent=False,
                    show_legend=True,
                    elem_classes=["highlighted-text-container"]
                )
        
        # Examples
        gr.Examples(
            examples=[
                [question_ai],
                [question_gsm8k],
                [question_math],
            ],
            inputs=user_input,
            label="Example Inputs"
        )
        
        # Advanced generation settings
        with gr.Accordion("Generation Settings", open=True):
            with gr.Row():
                max_new_tokens = gr.Slider(
                    minimum=64, maximum=2048, value=1024, step=64,
                    label="Max New Tokens"
                )
                block_length = gr.Slider(
                    minimum=4, maximum=32, value=16, step=4,
                    label="Block Size"
                )
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.0, step=0.1,
                    label="Temperature"
                )
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                    label="Top-p"
                )
            with gr.Row():
                threshold = gr.Slider(
                    minimum=0.5, maximum=1.0, value=0.95, step=0.05,
                    label="Threshold"
                )
                visualization_delay = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.1, step=0.1,
                    label="Visualization Delay (seconds)"
                )

        
        # Current response text box (hidden)
        current_response = gr.Textbox(
            label="Current Response",
            placeholder="The assistant's response will appear here...",
            lines=3,
            visible=False
        )
        
        # HELPER FUNCTIONS
        def add_message(history, message, response):
            """Add a message pair to the history and return the updated history"""
            history = history.copy()
            history.append({"role": "user", "content": message})
            if response:
                history.append({"role": "assistant", "content": response})
            return history
            
        def user_message_submitted(message, history_cache, max_new_tokens):
            """Process a submitted user message"""
            # Skip empty messages
            if not message.strip():
                # Return current state unchanged
                history_cache_for_display = history_cache.copy()
                return history_cache, history_cache_for_display, "", [], [], "wait for generation", "wait for generation", memory_stats
                
            # Add user message to history (without response yet)
            history_cache = history_cache.copy()
            history_cache.append({"role": "user", "content": message})
            
            # Format for display - temporarily show user message with empty response
            history_cache_for_display = history_cache.copy()
            
            # Clear the input
            message_out = ""
            
            # Return immediately to update UI with user message
            return history_cache, history_cache_for_display, message_out, [], [], "processing...", "processing...", memory_stats
            

        
        def generate_response(history_cache, max_new_tokens, temperature, top_p, block_length, threshold, visualization_delay):
            """Generate model response with visualization"""
            if not history_cache:
                return history_cache, [], [], "", "wait for generation", "wait for generation", memory_stats
                
            # Get the last user message
            last_user_message = history_cache[-1]["content"]
            
            try:
                # Use history_cache directly as it's already in the right format
                # But we need to convert it to the format expected by the model
                messages = []
                for msg in history_cache:
                    if msg["role"] in ["user", "assistant"]:
                        messages.append(msg)
                
                # Start timing
                start_time = time.time()
                
                # Generate with quantized model and yield states in real-time
                with torch.no_grad():
                    generator = generate_response_with_visualization(
                        model, tokenizer, device_accelerated,
                        messages, max_new_tokens, temperature, block_length, threshold, top_p
                    )
                    
                    # Collect all states and get final text
                    states = []
                    for item in generator:
                        if isinstance(item, list):  # Visualization state
                            states.append(item)
                            yield history_cache, item, [], "", "processing...", "processing...", memory_stats
                        else:  # Final text
                            response_text = item
                            break
                
                complete_time = time.time() - start_time
                generation_time_str = f"{complete_time:.2f}s"
                
                # Calculate throughput
                response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
                num_tokens = len(response_tokens)
                tokens_per_sec = num_tokens / complete_time if complete_time > 0 else 0
                throughput_str = f"{tokens_per_sec:.2f} tokens/s"
                
                # Add assistant response to history
                history_cache.append({"role": "assistant", "content": response_text})
                
                # Get updated memory stats
                current_memory_stats = print_memory_usage("After Generation")
                
                # Final yield with complete information and start slow motion visualization
                if states:
                    # First, yield the final real-time state
                    yield history_cache, states[-1], states[0], response_text, generation_time_str, throughput_str, current_memory_stats
                    
                    # Then animate through slow motion visualization
                    for state in states[1:]:
                        time.sleep(visualization_delay)
                        yield history_cache, states[-1], state, response_text, generation_time_str, throughput_str, current_memory_stats
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                error_vis = [(error_msg, "red")]
                yield history_cache, error_vis, error_vis, error_msg, "Error", "Error", memory_stats
        
        def clear_conversation():
            """Clear the conversation history"""
            empty_history = []
            empty_response = ""
            empty_vis = []
            time_str = "wait for generation"
            throughput_str = "wait for generation"
            
            # Update memory stats after clearing
            current_memory_stats = print_memory_usage("After Clear")
            
            return (
                empty_history,  # chat_history_cache
                empty_history,  # chatbot_ui
                empty_response,  # current_response
                empty_vis,      # output_vis
                empty_vis,      # output_vis_slow
                time_str,       # generation_time
                throughput_str,  # throughput
                current_memory_stats  # memory_display
            )
        
        # EVENT HANDLERS
        
        # Clear button handler
        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[chat_history_cache, chatbot_ui, current_response, output_vis, output_vis_slow, generation_time, throughput, memory_display]
        )
        
        # User message submission flow (2-step process)
        # Step 1: Add user message to history and update UI
        msg_submit = user_input.submit(
            fn=user_message_submitted,
            inputs=[user_input, chat_history_cache, max_new_tokens],
            outputs=[chat_history_cache, chatbot_ui, user_input, output_vis, output_vis_slow, generation_time, throughput, memory_display]
        )
        
        # Also connect the send button
        send_click = send_btn.click(
            fn=user_message_submitted,
            inputs=[user_input, chat_history_cache, max_new_tokens],
            outputs=[chat_history_cache, chatbot_ui, user_input, output_vis, output_vis_slow, generation_time, throughput, memory_display]
        )
        
        # Step 2: Generate model response
        msg_submit.then(
            fn=generate_response,
            inputs=[
                chat_history_cache, max_new_tokens, 
                temperature, top_p, block_length, threshold, visualization_delay
            ],
            outputs=[chatbot_ui, output_vis, output_vis_slow, current_response, generation_time, throughput, memory_display]
        )
        
        send_click.then(
            fn=generate_response,
            inputs=[
                chat_history_cache, max_new_tokens, 
                temperature, top_p, block_length, threshold, visualization_delay
            ],
            outputs=[chatbot_ui, output_vis, output_vis_slow, current_response, generation_time, throughput, memory_display]
        )
        
    return demo

# Launch the demo
if __name__ == "__main__":
    demo = create_chatbot_demo()
    demo.queue().launch(server_port=10087, share=True)
