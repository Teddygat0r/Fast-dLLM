# Chat interface for DuQuant-quantized LLaDA model
#
# This provides the same chat functionality as chat.py but with
# W8A8 DuQuant quantization applied to the model.

import torch
import argparse
import os

from typing import Dict, List, Optional, Tuple, Any

from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
from transformers import AutoTokenizer
from duquant import apply_duquant_pipeline


def load_duquant_model(
    model_path: str = "GSAI-ML/LLaDA-8B-Instruct",
    calibration_samples: int = 128,
    n_bits: int = 8,
    block_size: int = 128,
    max_rotation_step: int = 256,
    permutation_times: int = 1,
    a_bits: int = 8,
    seq_len: int = 512,
    batch_size: int = 1,
    device: str = "cuda",
    skip_layers: Optional[List[str]] = None,
    save_model_path: Optional[str] = None,
):
    """
    Load a LLaDA model with DuQuant applied.
    
    Args:
        model_path: HuggingFace model path
        n_bits: Weight quantization bits
        block_size: DuQuant block size
        max_rotation_step: Maximum rotation iterations
        permutation_times: Number of rotation+permutation iterations
        w_bits: Weight quantization bits
        a_bits: Activation quantization bits
        seq_len: Sequence length for calibration
        batch_size: Batch size for calibration
        device: Device to run on
        skip_layers: Layer names to skip during quantization
        save_model_path: Path to save the quantized model state dict
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model, info_dict = apply_duquant_pipeline(
        model_path=model_path,
        calibration_samples=calibration_samples,
        n_bits=n_bits,
        block_size=block_size,
        max_rotation_step=max_rotation_step,
        permutation_times=permutation_times,
        a_bits=a_bits,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
        skip_layers=skip_layers,
        save_model_path=save_model_path,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    return model, tokenizer


def chat(args):
    device = 'cuda'
    
    print("\n" + "=" * 66)
    print("Loading DuQuant LLaDA Model...")
    print("=" * 66)
    
    model, tokenizer = load_duquant_model(
        model_path=args.model_path,
        calibration_samples=args.calibration_samples,
        n_bits=args.n_bits,
        block_size=args.block_size,
        max_rotation_step=args.max_rotation_step,
        permutation_times=args.permutation_times,
        a_bits=args.a_bits,
        seq_len=512,
        batch_size=1,
        device=device,
        skip_layers=["lm_head"]
    )
    
    gen_length = args.gen_length
    steps = args.steps
    
    print("\n" + "*" * 66)
    print(f"**  DuQuant W{args.n_bits}A{args.a_bits} Chat Interface  **")
    print(f"**  Answer Length: {gen_length}  |  Sampling Steps: {steps}  **")
    print("*" * 66)
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("Type 'clear' to start a new conversation.\n")

    conversation_num = 0
    prompt = None
    
    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        
        # Handle special commands
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break
        elif user_input.lower() == 'clear':
            conversation_num = 0
            prompt = None
            print("\n[Conversation cleared]\n")
            continue
        elif not user_input.strip():
            continue

        # Format the message using chat template
        m = [{"role": "user", "content": user_input}]
        formatted_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(formatted_input)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        # Build prompt (accumulate for multi-turn)
        if conversation_num == 0:
            prompt = input_ids
        else:
            prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)
        
        # Generate response
        print(f"\n[Generating with cache={args.use_cache}, steps={steps}, block_size={args.block_size}]")
        
        with torch.no_grad():
            if args.use_cache:
                if args.if_cache_position:
                    out, nfe = generate_with_dual_cache(
                        model, prompt, 
                        steps=steps, 
                        gen_length=gen_length, 
                        block_length=args.block_size, 
                        temperature=args.temperature, 
                        remasking='low_confidence', 
                        threshold=args.threshold
                    )
                else:
                    out, nfe = generate_with_prefix_cache(
                        model, prompt, 
                        steps=steps, 
                        gen_length=gen_length, 
                        block_length=args.block_size, 
                        temperature=args.temperature, 
                        remasking='low_confidence', 
                        threshold=args.threshold
                    )
            else:
                out, nfe = generate(
                    model, prompt, 
                    steps=steps, 
                    gen_length=gen_length, 
                    block_length=args.block_size, 
                    temperature=args.temperature, 
                    remasking='low_confidence', 
                    threshold=args.threshold
                )

        # Decode and print response
        answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
        print(f"\nBot: {answer}")
        print(f"[Forward passes: {nfe}]")

        # Update prompt for next turn (remove EOS token)
        prompt = out[out != 126081].unsqueeze(0)
        conversation_num += 1
        print("-" * 66 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with DuQuant-quantized LLaDA model")
    
    # Generation parameters
    parser.add_argument("--gen_length", type=int, default=128,
                        help="Maximum generation length")
    parser.add_argument("--steps", type=int, default=128,
                        help="Number of diffusion sampling steps")
    parser.add_argument("--block_size", type=int, default=32,
                        help="Block size for semi-autoregressive generation")
    parser.add_argument("--temperature", type=float, default=0.,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Confidence threshold for token transfer")
    
    # Cache options
    parser.add_argument("--use_cache", action="store_true",
                        help="Use KV-cache for faster generation")
    parser.add_argument("--if_cache_position", action="store_true",
                        help="Use dual cache with position tracking")
    
    # Model and quantization parameters
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="HuggingFace model path")
    parser.add_argument("--calibration_samples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--max_rotation_step", type=int, default=256,
                        help="Maximum rotation iterations")
    parser.add_argument("--permutation_times", type=int, default=1,
                        help="Number of rotation+permutation iterations")
    parser.add_argument("--n_bits", type=int, default=8,
                        help="Weight quantization bits")
    parser.add_argument("--a_bits", type=int, default=8,
                        help="Activation quantization bits")

    args = parser.parse_args()
    chat(args)
