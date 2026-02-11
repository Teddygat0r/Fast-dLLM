import argparse
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from duquant_utils import compile_linear, create_quant_args, replace_linear_layers, replace_llada_blocks
import json
from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
import gc

DEVICE = "cuda"

def load_model(
    model_path: str,
    weight_path: str,
    args,
):
    model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu", dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    quant_config = json.load(open('model/quantize/quant_args.json'))

    quant_config["wbits"] = args.wbits
    quant_config["abits"] = args.abits
    quant_args = create_quant_args(quant_config)

    weights = torch.load(weight_path, map_location="cpu")

    # Replace LLaDA blocks with Quantized LLaDA blocks
    replace_llada_blocks(model, quant_args, device="cpu")

    replace_linear_layers(model, quant_args, weights)

    print("Loading Quantized Model...")
    # we expect to have missing keys: (act quantizer scales and zeros)
    # we expect to have unexpected keys: (ori layers, biases)
    missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    # print("\n\n\n")
    # print("Model Keys: ")
    # print(model.state_dict().keys())
    # print("\n\n\n")
    print(gc.get_stats())
    compile_linear(model)
    model.to(DEVICE)
    model.eval()
    # model = torch.compile(model, mode="reduce-overhead")
    return model, tokenizer

def chat(args):
    device = 'cuda'
    
    print("\n" + "=" * 66)
    print("Loading DuQuant LLaDA Model...")
    print("=" * 66)
    
    model, tokenizer = load_model(
        model_path=args.model_path,
        weight_path=args.weight_path,
        args=args
    )
    
    gen_length = args.gen_length
    steps = args.steps
    
    print("\n" + "*" * 66)
    print(f"**  DuQuant W8A8 Chat Interface  **")
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
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="HuggingFace model path")
    parser.add_argument("--weight_path", type=str, default="models/quantized_model.pth",
                        help="Weight Loading Path")

    parser.add_argument("--wbits", type=int, default=4,
                        help="Weight quantization bits")
    parser.add_argument("--abits", type=int, default=4,
                        help="Activation quantization bits")

    # Cache options
    parser.add_argument("--use_cache", action="store_true",
                        help="Use KV-cache for faster generation")
    parser.add_argument("--if_cache_position", action="store_true",
                        help="Use dual cache with position tracking")
    
    parser.add_argument("--gen_length", type=int, default=256,
                        help="Maximum generation length")
    parser.add_argument("--steps", type=int, default=128,
                        help="Number of diffusion sampling steps")
    parser.add_argument("--block_size", type=int, default=32,
                        help="Block size for semi-autoregressive generation")
    parser.add_argument("--temperature", type=float, default=0.,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Confidence threshold for token transfer")

    args = parser.parse_args()
    chat(args)