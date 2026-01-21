# Chat interface for SmoothQuant-quantized LLaDA model
#
# This provides the same chat functionality as chat.py but with
# W8A8 SmoothQuant quantization applied to the model.

import torch
import argparse
import os

from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
from transformers import AutoTokenizer
from smoothquant import apply_smoothquant_pipeline


def load_smoothquant_model(
    model_path: str = "GSAI-ML/LLaDA-8B-Instruct",
    alpha: float = 0.5,
    w_bits: int = 8,
    a_bits: int = 8,
    calibration_samples: int = 64,
    load_scales_path: str = None,
    save_scales_path: str = None,
    skip_quantization: bool = False,
):
    """
    Load a LLaDA model with SmoothQuant applied.
    
    Args:
        model_path: HuggingFace model path
        alpha: SmoothQuant migration strength
        w_bits: Weight quantization bits
        a_bits: Activation quantization bits
        calibration_samples: Number of calibration samples
        load_scales_path: Path to load pre-computed activation scales
        save_scales_path: Path to save activation scales
        skip_quantization: If True, only apply smoothing without quantization
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model, act_scales = apply_smoothquant_pipeline(
        model_path=model_path,
        calibration_samples=calibration_samples,
        alpha=alpha,
        w_bits=w_bits,
        a_bits=a_bits,
        seq_len=512,
        batch_size=1,
        device="cuda",
        save_scales_path=save_scales_path,
        load_scales_path=load_scales_path,
        skip_quantization=skip_quantization,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    return model, tokenizer


def chat(args):
    device = 'cuda'
    
    print("\n" + "=" * 66)
    print("Loading SmoothQuant LLaDA Model...")
    print("=" * 66)
    
    # Check for pre-computed scales
    scales_path = args.scales_path
    if scales_path and os.path.exists(scales_path):
        print(f"Loading pre-computed activation scales from: {scales_path}")
    
    model, tokenizer = load_smoothquant_model(
        model_path=args.model_path,
        alpha=args.alpha,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        calibration_samples=args.calibration_samples,
        load_scales_path=scales_path if scales_path and os.path.exists(scales_path) else None,
        save_scales_path=scales_path if scales_path and not os.path.exists(scales_path) else None,
        skip_quantization=args.skip_quantization,
    )
    
    gen_length = args.gen_length
    steps = args.steps
    
    print("\n" + "*" * 66)
    print(f"**  SmoothQuant W{args.w_bits}A{args.a_bits} Chat Interface  **")
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
    parser = argparse.ArgumentParser(description="Chat with SmoothQuant-quantized LLaDA model")
    
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
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="SmoothQuant migration strength")
    parser.add_argument("--w_bits", type=int, default=8,
                        help="Weight quantization bits")
    parser.add_argument("--a_bits", type=int, default=8,
                        help="Activation quantization bits")
    parser.add_argument("--calibration_samples", type=int, default=64,
                        help="Number of calibration samples")
    parser.add_argument("--scales_path", type=str, default="models/act_scales_chat.pt",
                        help="Path to save/load activation scales")
    parser.add_argument("--skip_quantization", action="store_true",
                        help="Only apply smoothing, skip quantization")

    args = parser.parse_args()
    chat(args)
