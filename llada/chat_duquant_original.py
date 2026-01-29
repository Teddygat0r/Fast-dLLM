import argparse
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from model.quantize.int_linear import QuantLinear
import copy
import json
from generate import generate, generate_with_prefix_cache, generate_with_dual_cache

DEVICE = "cuda"

def create_quant_args(quant_config):
    """Create an args-like object from saved quant config."""
    class Args:
        pass
    
    args = Args()
    for key, value in quant_config.items():
        setattr(args, key, value)
    
    # Set up weight quant params
    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": "per_channel",
        "group_size": args.group_size,
        "lwc": args.lwc,
        "swc": args.swc,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    args.act_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "lac": args.lac,
        "act_group_size": args.act_group_size,
        "dynamic_method": "per_token",
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": "per_token",
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": "per_token",
        "quant_method": args.quant_method,
        "block_size": args.block_size,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": "per_token",
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }
    
    return args

def replace_linear_layers(model, quant_args):
    print("Starting replacement...")
    for name, module in dict(model.named_modules()).items():
        if isinstance(module, nn.Linear):
            print(f"Replacing {name}...")
            weight_quant_params = quant_args.weight_quant_params
            act_quant_params = quant_args.act_quant_params
            quant_linear = QuantLinear(
                module,
                weight_quant_params=weight_quant_params, 
                act_quant_params=act_quant_params
            )
            parent = model
            path = name.split('.')
            for s in path[:-1]:
                if s.isdigit():
                    parent = parent[int(s)]
                else:
                    parent = getattr(parent, s)
            
            leaf_name = path[-1]
            if leaf_name.isdigit():
                parent[int(leaf_name)] = quant_linear
            else:
                print(f"Setting {leaf_name} to {quant_linear}...")
                setattr(parent, leaf_name, quant_linear)


def load_model(
    model_path: str,
    weight_path: str,
    args,
):
    model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, device_map=DEVICE, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    quant_args = json.load(open('model/quantize/quant_args.json'))
    quant_args = create_quant_args(quant_args)

    replace_linear_layers(model, quant_args)

    print("Original Model State Dict: ")
    print(model.state_dict().keys())

    print("Loading Quantized Model...")
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weight_path), strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    # finished loading model
    model.eval()
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