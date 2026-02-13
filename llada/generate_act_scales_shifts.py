import torch
import os

from transformers import AutoTokenizer
import argparse
import torch.nn as nn

import functools
from tqdm import tqdm

from datautils import get_wikitext2
from model.modeling_llada import LLaDAModelLM
from quantization_calibration_dataset import LLaDACalibrationDataset


def get_act_scales(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        input_ids = dataloader[i].to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales

def get_act_shifts(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_shifts = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        comming_min = torch.min(tensor, dim=0)[0].float().cpu()
        if name in act_shifts:
            act_shifts[name] = 0.99*act_shifts[name] + 0.01 *((comming_max+comming_min)/2)
        else:
            act_shifts[name] = (comming_max+comming_min)/2

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    for i in tqdm(range(num_samples)):
        input_ids = dataloader[i]["input_ids"].to(device).unsqueeze(0)
        model(input_ids)


    for h in hooks:
        h.remove()

    return act_shifts


def build_model_and_tokenizer(model_name):
    device = 'cuda'
    model = LLaDAModelLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def parse_args():
    FILE_PATH = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(FILE_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='GSAI-ML/LLaDA-8B-Instruct', help='model name')
    parser.add_argument('--scales-output-path', type=str, default=f'{BASE_DIR}/act_scales/',
                        help='where to save the act scales')
    parser.add_argument('--shifts-output-path', type=str, default=f'{BASE_DIR}/act_shifts/',
                        help='where to save the act shifts')
    parser.add_argument('--num-samples', type=int, default=128)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--block-size', type=int, default=32)
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model)
    
    # Create calibration dataset using LLaDACalibrationDataset
    # dataloader = LLaDACalibrationDataset(
    #     tokenizer=tokenizer,
    #     seq_len=args.seq_len,
    #     samples=args.num_samples,
    #     block_size=args.block_size,
    #     seed=args.seed
    # )
    dataloader, _ = get_wikitext2(nsamples=args.num_samples, seed=args.seed, seqlen=args.seq_len, model=args.model)
    
    args.net = args.model.split('/')[-1]
    act_scales = get_act_scales(model, dataloader, args.num_samples)
    save_path = os.path.join(args.scales_output_path, f'{args.net}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_scales, save_path)

    # act_shifts = get_act_shifts(model, dataloader, args.num_samples)
    # save_path = os.path.join(args.shifts_output_path, f'{args.net}.pt')
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # torch.save(act_shifts, save_path)


if __name__ == '__main__':
    main()