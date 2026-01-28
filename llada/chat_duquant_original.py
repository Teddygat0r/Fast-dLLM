import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from model.int_llada_layer import QuantLLadaDecoderLayer
import copy


DEVICE = "cuda"

def load_model(
    model_path: str,
    weight_path: str,
    args,
):
    model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, device_map=DEVICE, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    blocks = model.model.transformer.blocks

    #load duquant parameters

    for i in range(len(blocks)):
        for name in ['q', 'k', 'v', 'o', 'gate', 'up', 'down']:
            exec(f"args.{name}_weight_quant_params = copy.copy(args.weight_quant_params)")
            exec(f"args.{name}_act_quant_params = copy.copy(args._act_quant_params)")
        args.q_quant_params = copy.copy(args.act_quant_params)
        args.k_quant_params = copy.copy(args.act_quant_params)

        block = blocks[i]
        qlayer = QuantLLadaDecoderLayer(model.config, block, args)
        qlayer = qlayer.to(DEVICE)

        


        # finished loading model
        model.eval()
        return model, tokenizer
def chat(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with SmoothQuant-quantized LLaDA model")
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="HuggingFace model path")
    parser.add_argument("--weight_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="Weight Loading Path")