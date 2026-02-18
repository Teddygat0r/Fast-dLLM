"""
DuQuant utilities for loading pre-saved DuQuant models.
Shared by chat_duquant_original.py and eval_llada.py.
"""
import torch
import torch.nn as nn

from model.int_llada_layer import LLaDaQuantLayer
from model.modeling_llada import LLaDALlamaBlock
import gc

def create_quant_args(quant_config):
    """
    Create an args-like object from saved quant config.
    Used for loading pre-saved DuQuant models.
    
    Args:
        quant_config: Dictionary containing quantization configuration
                     (typically loaded from quant_args.json)
    
    Returns:
        Args object with weight_quant_params, act_quant_params, etc.
    """
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

def replace_llada_blocks(model, quant_args, device="cuda"):
    blocks = []
    for name, module in model.named_modules():
        if isinstance(module, LLaDALlamaBlock):
            blocks.append((name, module))
    
    for name, module in blocks:
        print("Replacing block", name)
        original_dtype = next(module.parameters()).dtype
        new_block = LLaDaQuantLayer(module, quant_args).to(device=device, dtype=original_dtype)
        new_block.load_state_dict(module.state_dict())
        new_block.set_quant_state(weight_quant=False, act_quant=True)
        
        parent = model
        path = name.split('.')
        
        for s in path[:-1]:
            if s.isdigit():
                parent = parent[int(s)]
            else:
                parent = getattr(parent, s)
        leaf = path[-1]
        if leaf.isdigit():
            parent[int(leaf)] = new_block
        else:
            setattr(parent, leaf, new_block)
        
        del module
    gc.collect()
        

def replace_linear_layers(model, quant_args, weights):
    """
    Replace Linear layers with QuantLinear layers for DuQuant.
    This function loads the saved DuQuant parameters and applies them
    to the quantizers.
    
    Args:
        model: The LLaDA model to modify
        quant_args: Args object from create_quant_args()
        weight_path: Path to the saved DuQuant parameters (.pth file)
        device: Device to load weights to (default: "cuda")
    """
    # Import here to avoid circular imports
    from model.quantize.int_linear import QuantLinear
    
    for name, module in dict(model.named_modules()).items():
        if name == "model.transformer.ff_out":
            continue
        if isinstance(module, nn.Linear):
            if name.find('block') > -1:
                layer_index = int(name.split('.')[3])
                layer_name = name.split('.')[4]
            else:
                continue  # Skip layers that don't match expected pattern
            
            # Filter layer params, ensuring keys are strings before using .find()
            layer_params = {k:v for k,v in weights.items() 
                           if isinstance(k, str) and k.find(layer_name) > -1 and k.find(f".{layer_index}.") > -1}
            
            weight_quant_params = quant_args.weight_quant_params
            act_quant_params = quant_args.act_quant_params
            quant_linear = QuantLinear(
                module,
                weight_quant_params=weight_quant_params, 
                act_quant_params=act_quant_params
            )

            quant_linear.weight_quantizer.load_scales_and_zeros(layer_params, layer_name)
            # unnecessary in loading
            # quant_linear.act_quantizer.load_scales_and_zeros(layer_params, layer_name)

            quant_linear.weight_quantizer.load_duquant_params(layer_params, layer_name)
            quant_linear.act_quantizer.load_duquant_params(layer_params, layer_name)
            quant_linear = quant_linear.to(dtype=module.weight.dtype)

            quant_linear.set_quant_state(weight_quant=False, act_quant=True)
            
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
                setattr(parent, leaf_name, quant_linear)
            
            del module
    gc.collect()

def replace_act_quantizer_with_inference(quant_linear):
    """Replace a QuantLinear's act_quantizer with an InferenceQuantizer.

    Copies n_bits, lac, block_size, permutation_times, and DuQuant buffers
    (R, permutation_list) from the existing UniformAffineQuantizer.

    Args:
        quant_linear: A QuantLinear layer with an existing act_quantizer.
    """
    from model.quantize.inference_quantizer import InferenceQuantizer

    old = quant_linear.act_quantizer
    if old is None:
        return

    infer = InferenceQuantizer(
        n_bits=old.n_bits,
        lac=old.lac,
        block_size=old.block_size,
        permutation_times=old.permutation_times,
    )

    # Copy pre-calibrated DuQuant buffers
    # R must be float32 (rotations are done in FP32 for precision)
    if hasattr(old, 'R') and old.R is not None:
        infer.R = old.R.clone().detach().float()
    if hasattr(old, 'permutation_list') and old.permutation_list is not None:
        try:
            infer.permutation_list = old.permutation_list.clone().detach()
        except Exception:
            infer.permutation_list = old.permutation_list

    infer._resolve_permutation_type()
    quant_linear.act_quantizer = infer

def replace_act_quantizers_with_inference(model):
    from model.quantize.int_linear import QuantLinear
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            replace_act_quantizer_with_inference(module)

@torch.no_grad()
def set_init_duquant_params_state(model, mode):
    if isinstance(mode, bool):
        mode = torch.tensor(mode)
    for name, module in model.named_modules():
        if hasattr(module, "init_duquant_params"):
            module.init_duquant_params = mode

def set_quant_state(model, weight_quant: bool = False, act_quant: bool = False):
    from model.quantize.int_linear import QuantLinear
    from model.quantize.int_matmul import QuantMatMul
    for name, module in model.named_modules():
        if isinstance(module, (QuantLinear, QuantMatMul)):
            module.set_quant_state(weight_quant, act_quant)

# llada doesn't have biases. 
def smooth_ln_to_fcs(ln, fcs, scales):
    if hasattr(ln, 'bias') and ln.bias is not None:
        # this should not happen
        print(f"Bias found for {ln.__class__.__name__}")
        ln.bias.div_(scales.to(ln.bias.device))

    ln.weight.div_(scales.to(ln.weight.device))
    for fc in fcs:
        fc.weight.mul_(scales.to(fc.weight.device).view(1, -1))

def smooth_fc_to_fc(fc1, fc2, scales):
    fc1.weight.div_(scales.to(fc1.weight.device).view(-1, 1))
    fc2.weight.mul_(scales.to(fc2.weight.device).view(1, -1))

def smooth_q_k(q, k, scales):
    q.weight.div_(scales.to(q.weight.device).view(-1, 1))
    k.weight.mul_(scales.to(k.weight.device).view(1, -1))

@torch.no_grad()
def smooth_and_let_inplace(block, args):
    smooth_ln_to_fcs(block.attn_norm, [block.q_proj, block.k_proj, block.v_proj], block.qkv_smooth_scale)
    smooth_ln_to_fcs(block.ff_norm,[block.up_proj,block.ff_proj], block.fc1_smooth_scale)
    smooth_fc_to_fc(block.up_proj,block.ff_out, block.down_smooth_scale)
    smooth_fc_to_fc(block.v_proj,block.attn_out, block.out_smooth_scale)
    smooth_q_k(block.q_proj, block.k_proj, block.qkt_smooth_scale)

@torch.no_grad()
def set_init_duquant_params_state(block, mode):
    if isinstance(mode, bool):
        mode = torch.tensor(mode)
    for name, module in block.named_modules():
        if hasattr(module, "init_duquant_params"):
            module.init_duquant_params = mode

@torch.no_grad()
def compile_linear(model):
    from model.quantize.int_linear import QuantLinear
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module = torch.compile(module, mode="reduce-overhead")