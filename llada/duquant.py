import argparse
import copy
import gc
import torch
from torch import nn
from torch.utils.data import DataLoader
from duquant_utils import create_quant_args, set_init_duquant_params_state, set_quant_state, smooth_and_let_inplace
from model.quantize.int_linear import QuantLinear
from model.int_llada_layer import LLaDaQuantLayer

CLIPMIN = 1e-5

def duquant(model: nn.Module, act_scales: dict, dataloader, args):
    layers = model.model.transformer.blocks
    use_cache = model.config.use_cache
    model.config.use_cache = False
    dtype = torch.bfloat16
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seqlen = args.seqlen
    pairs = {
        "q_proj":"qkv",
        "attn_out":"out",
        "up_proj":"fc1",
        "ff_out":"down",
    }

    inps = torch.zeros(
        (args.nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            if len(inp.shape) == 3:
                 inps[cache["i"]] = inp[0] 
            else:
                 inps[cache["i"]] = inp
            cache["i"] += 1
            return self.module(inp, **kwargs)
    
    layers[0] = Catcher(layers[0])

    input_ids = []

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                input_ids.append(batch['input_ids'])
                model(batch['input_ids'].to(dev))

            except ValueError:
                pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    print(layers[0])
    duquant_parameters = {}

    torch.cuda.empty_cache()
    quant_inps = inps
    rotate_inps = copy.copy(inps).mean(dim=0)

    fp_inps = copy.deepcopy(inps)
    
    for i in range(len(layers)):
        print("Starting Layer, " + str(i))
        args.q_quant_params = copy.copy(args.act_quant_params)
        args.k_quant_params = copy.copy(args.act_quant_params)
        layer = layers[i]
        qlayer = LLaDaQuantLayer(layer, args)
        qlayer.set_quant_state(weight_quant=False, act_quant=True)

        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                weight_quant = QuantLinear(module, weight_quant_params=copy.copy(args.weight_quant_params), act_quant_params=copy.copy(args.act_quant_params))
                setattr(qlayer, name, weight_quant)

        qlayer.load_state_dict(layer.state_dict())
        qlayer.to(dev)

        set_init_duquant_params_state(qlayer, True)
        set_quant_state(qlayer, weight_quant=False, act_quant=True)

        qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(qlayer.q_proj.out_features,device=dev, dtype=dtype), requires_grad=False))
        for name, module in qlayer.named_modules():
            if isinstance(module, QuantLinear):
                for key in pairs.keys():
                    if key in name:
                        act = act_scales[f"model.transformer.blocks.{i}.{key}"].to(device=dev, dtype=dtype).clamp(min=CLIPMIN)
                        weight = module.weight.abs().max(dim=0)[0].clamp(min=CLIPMIN)
                        scale = (act.pow(args.alpha)/weight.to(act.device).pow(1-args.alpha)).clamp(min=CLIPMIN)

                        qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale, requires_grad=False))

        qlayer.to(dtype=torch.bfloat16)

        try:
            with torch.no_grad():
                qlayer.qkt_smooth_scale.clamp_(min=0.5)
        except:
            pass
        smooth_and_let_inplace(qlayer, args)

        # perform duquant process
        set_init_duquant_params_state(qlayer, False)
        set_quant_state(qlayer, weight_quant=True, act_quant=True)
        with torch.no_grad():
            with torch.amp.autocast(device_type=dev):
                rotate_inps = qlayer(rotate_inps.unsqueeze(0))[0][0]
            qlayer.register_duquant_params()
            set_init_duquant_params_state(qlayer, True)

        qlayer.to(dtype=torch.bfloat16)
        with torch.no_grad():
            for name, module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    module.weight = module.weight_quantizer(module.weight, return_no_quant=False)

        set_quant_state(qlayer, weight_quant=False, act_quant=True)
        layers[i] = qlayer.to("cpu")
        # i dont think this is necessary for loading
        # duquant_parameters[i] = duquant_state_dict(qlayer)

        del layer
        torch.cuda.empty_cache()

    model.model.transformer.embed_tokens = model.model.transformer.wte.to('cpu')
    del inps
    del quant_inps
    del fp_inps
    del rotate_inps
    
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    
    return model


def main(args):
    from transformers import AutoTokenizer
    from model.modeling_llada import LLaDAModelLM
    from quantization_calibration_dataset import LLaDACalibrationDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build quant args from defaults (CLI overrides already applied)
    user_args = {
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "wbits": args.wbits,
        "abits": args.abits,
        "alpha": args.alpha,
        "act_group_size": args.act_group_size,
        "smooth": args.smooth,
        "quant_method": args.quant_method,
        "symmetric": args.symmetric,
        "group_size": args.group_size,
        "swc": args.swc,
        "lac": args.lac,
        "lwc": args.lwc,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
        "batch_size": args.batch_size,
    }

    print(user_args)
    quant_args = create_quant_args(user_args)

    # Load model & tokenizer
    print(f"Loading model from {args.model_path} ...")
    model = LLaDAModelLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load activation scales
    print(f"Loading act scales from {args.act_scales_path} ...")
    act_scales = torch.load(args.act_scales_path)

    # Build calibration dataloader
    dataset = LLaDACalibrationDataset(
        tokenizer=tokenizer,
        seq_len=quant_args.seqlen,
        samples=quant_args.nsamples,
        block_size=quant_args.block_size,
    )
    dataloader = DataLoader(dataset, batch_size=quant_args.batch_size, shuffle=True)

    # Run DuQuant
    print("Running DuQuant calibration ...")
    model = duquant(model, act_scales, dataloader, quant_args)

    # Optionally save the quantized model
    if args.save_path:
        print(f"Saving quantized model state dict to {args.save_path} ...")
        torch.save(model.state_dict(), args.save_path)
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DuQuant calibration on LLaDA")

    # Model / data paths
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="HuggingFace model path or local directory")
    parser.add_argument("--act_scales_path", type=str, default="act_scales/LLaDA-8B-Instruct.pt",
                        help="Path to pre-computed activation scales")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save quantized model state dict (e.g. models/quantized_model.pth)")

    # Quantization hyperparameters
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--wbits", type=int, default=8, help="Weight bit-width")
    parser.add_argument("--abits", type=int, default=8, help="Activation bit-width")
    parser.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant alpha")
    parser.add_argument("--act_group_size", type=int, default=None)
    parser.add_argument("--no_smooth", dest="smooth", action="store_false",
                        help="Disable SmoothQuant")
    parser.set_defaults(smooth=True)
    parser.add_argument("--quant_method", type=str, default="duquant")
    parser.add_argument("--no_symmetric", dest="symmetric", action="store_false")
    parser.set_defaults(symmetric=True)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--swc", type=float, default=0.8, help="Weight clipping ratio")
    parser.add_argument("--lac", type=float, default=0.9, help="Activation clipping ratio")
    parser.add_argument("--lwc", action="store_true", default=False,
                        help="Enable learnable weight clipping")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_rotation_step", type=int, default=256)
    parser.add_argument("--permutation_times", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)

    cli_args = parser.parse_args()
    main(cli_args)
