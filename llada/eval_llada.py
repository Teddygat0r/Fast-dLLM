# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import torch
import torch.nn as nn
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
from model.modeling_llada import LLaDAModelLM
import json
import time

# Import smoothquant pipeline (optional - only needed if using smoothquant)
try:
    from smoothquant import apply_smoothquant_pipeline
    SMOOTHQUANT_AVAILABLE = True
except ImportError:
    SMOOTHQUANT_AVAILABLE = False

from duquant_utils import compile_linear, create_quant_args, replace_linear_layers, replace_llada_blocks

# #region agent log
DEBUG_LOG_PATH = "/home/joshuaz/dllm/Fast-dLLM/.cursor/debug.log"
def _debug_log(session_id, run_id, hypothesis_id, location, message, data):
    try:
        with open(DEBUG_LOG_PATH, 'a') as f:
            f.write(json.dumps({"sessionId": session_id, "runId": run_id, "hypothesisId": hypothesis_id, "location": location, "message": message, "data": data, "timestamp": time.time() * 1000}) + "\n")
    except: pass
# #endregion
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        wbits=8,
        abits=8,
        symmetric=False,
        remasking='low_confidence',
        device="cuda",
        use_cache=False,
        threshold=None,
        factor=None,
        save_dir=None,
        show_speed=False,
        dual_cache=False,
        smoothed_model_path=None,
        quantized_model_path=None,
        # SmoothQuant pipeline parameters (from smoothquant folder)
        use_smoothquant_pipeline=False,
        smoothquant_alpha=0.5,
        smoothquant_w_bits=8,
        smoothquant_a_bits=8,
        smoothquant_calibration_samples=64,
        smoothquant_scales_path=None,
        smoothquant_skip_quantization=False,
        use_duquant_pipeline=False,
        duquant_seq_len=512,
        duquant_batch_size=1,
        duquant_weight_path=None,  # Path to saved DuQuant parameters (.pth file)
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
            
            SmoothQuant Pipeline Parameters (uses smoothquant folder with QuantLinear layers):
            use_smoothquant_pipeline: If True, use the full SmoothQuant pipeline with QuantLinear layers.
            smoothquant_alpha: SmoothQuant migration strength (0.0-1.0, default 0.5).
            smoothquant_w_bits: Weight quantization bits (default 8).
            smoothquant_a_bits: Activation quantization bits (default 8).
            smoothquant_calibration_samples: Number of calibration samples (default 64).
            smoothquant_scales_path: Path to load/save pre-computed activation scales.
            smoothquant_skip_quantization: If True, only apply smoothing without quantization.
        '''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        
        # Track which path should be used for the tokenizer (may differ for quantized models)
        tokenizer_model_path = model_path
        
        # Convert string 'True'/'False' to boolean for use_smoothquant_pipeline
        if isinstance(use_smoothquant_pipeline, str):
            use_smoothquant_pipeline = use_smoothquant_pipeline.lower() == 'true'
        if isinstance(smoothquant_skip_quantization, str):
            smoothquant_skip_quantization = smoothquant_skip_quantization.lower() == 'true'

        # Use SmoothQuant pipeline from smoothquant folder (with QuantLinear layers)
        if use_smoothquant_pipeline:
            if not SMOOTHQUANT_AVAILABLE:
                raise ImportError(
                    "SmoothQuant pipeline requested but smoothquant module not found. "
                    "Make sure the smoothquant folder is in the llada directory."
                )
            
            print(f"\n{'='*60}")
            print("Loading model with SmoothQuant Pipeline (QuantLinear layers)")
            print(f"{'='*60}")
            print(f"  Alpha: {smoothquant_alpha}")
            print(f"  Quantization: W{smoothquant_w_bits}A{smoothquant_a_bits}")
            print(f"  Calibration samples: {smoothquant_calibration_samples}")
            print(f"  Scales path: {smoothquant_scales_path}")
            print(f"  Skip quantization: {smoothquant_skip_quantization}")
            
            # Determine device for smoothquant pipeline
            sq_device = f'{self.accelerator.device}' if self.accelerator is not None else device
            
            # Check if scales file exists for loading
            load_scales = None
            save_scales = None
            if smoothquant_scales_path:
                if os.path.exists(smoothquant_scales_path):
                    load_scales = smoothquant_scales_path
                    print(f"  Loading pre-computed scales from: {smoothquant_scales_path}")
                else:
                    save_scales = smoothquant_scales_path
                    print(f"  Will save scales to: {smoothquant_scales_path}")
            
            self.model, _ = apply_smoothquant_pipeline(
                model_path=model_path,
                calibration_samples=int(smoothquant_calibration_samples),
                alpha=float(smoothquant_alpha),
                w_bits=int(smoothquant_w_bits),
                a_bits=int(smoothquant_a_bits),
                seq_len=512,
                batch_size=1,
                device=sq_device,
                save_scales_path=save_scales,
                load_scales_path=load_scales,
                skip_quantization=smoothquant_skip_quantization,
            )
            
            # Skip the other loading paths
            quantized_model_path = None
            smoothed_model_path = None
            print("✓ SmoothQuant pipeline model loaded with QuantLinear layers")

        if use_duquant_pipeline:
            # Convert string 'True'/'False' to boolean for use_duquant_pipeline
            if isinstance(use_duquant_pipeline, str):
                use_duquant_pipeline = use_duquant_pipeline.lower() == 'true'
            
            # Determine device for duquant
            dq_device = f'{self.accelerator.device}' if self.accelerator is not None else device
            
            # Check if we have a saved DuQuant model to load
            if duquant_weight_path is not None and duquant_weight_path != '' and os.path.exists(duquant_weight_path):
                # Load saved DuQuant model using the pattern from chat_duquant_original.py
                print(f"\n{'='*60}")
                print("Loading pre-saved DuQuant model")
                print(f"{'='*60}")
                print(f"  Weight path: {duquant_weight_path}")
                
                # Load base model first
                self.model = LLaDAModelLM.from_pretrained(
                    model_path, 
                    trust_remote_code=True, 
                    device_map="cpu",
                    dtype=torch.bfloat16
                )
                
                # Load quant args from config file
                quant_args_path = os.path.join(os.path.dirname(__file__), 'model/quantize/quant_args.json')
                quant_config = json.load(open(quant_args_path))

                # Ensure int (model_args from lm_eval may pass strings)
                quant_config["wbits"] = int(wbits)
                quant_config["abits"] = int(abits)
                quant_config["symmetric"] = symmetric
                quant_args = create_quant_args(quant_config)
                weights = torch.load(duquant_weight_path, map_location="cpu")

                print("Replacing LLaDA blocks")
                replace_llada_blocks(self.model, quant_args, device="cpu")
                
                # Replace linear layers with QuantLinear layers
                print("Replacing Linear layers with QuantLinear...")
                replace_linear_layers(self.model, quant_args, weights)
                
                # Load the quantized model weights
                print("Loading DuQuant parameters...")
                missing_keys, unexpected_keys = self.model.load_state_dict(weights, strict=False)
                if missing_keys:
                    print(f"  Missing keys: {len(missing_keys)} (expected for QuantLinear params)")
                if unexpected_keys:
                    print(f"  Unexpected keys: {len(unexpected_keys)}")
                compile_linear(self.model)
                
                self.model.to(dq_device)
                print("✓ Pre-saved DuQuant model loaded successfully")

            # Skip the other loading paths
            quantized_model_path = None
            smoothed_model_path = None

        # Load quantized model if provided (takes precedence, unless smoothquant pipeline was used)
        if not use_smoothquant_pipeline and not use_duquant_pipeline and quantized_model_path is not None and quantized_model_path != '':
            if os.path.exists(quantized_model_path):
                # New path: quantized model saved via `save_pretrained` (directory with config + weights)
                if os.path.isdir(quantized_model_path):
                    print(f"Loading quantized model (HF/Quanto format) from {quantized_model_path}...")
                    load_kwargs = {
                        "trust_remote_code": True,
                    }
                    load_kwargs.update(model_kwargs)
                    # Follow the recommended loading pattern printed in quantize_llada_w4a8.py
                    self.model = LLaDAModelLM.from_pretrained(
                        quantized_model_path,
                        **load_kwargs,
                    )
                    tokenizer_model_path = quantized_model_path
                    print("✓ Quantized model loaded from directory")
                else:
                    # Backwards-compatible path: quantized model saved as a torch .pt file
                    print(f"Loading quantized model from {quantized_model_path} (torch .pt)...")
                    # Pre-load model class to ensure transformers_modules is populated
                    _ = LLaDAModelLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        device_map="meta",  # Use 'meta' to avoid loading weights
                        trust_remote_code=True,
                    )
                    # Load the quantized model object
                    self.model = torch.load(quantized_model_path, map_location=device, weights_only=False)
                    # Ensure model is on the correct device
                    if self.accelerator is None:
                        self.model = self.model.to(device)
                    print("✓ Quantized model loaded from .pt file")
            else:
                print(f"WARNING: Quantized model path not found at {quantized_model_path}")
                print("Falling back to loading base model...")
                quantized_model_path = None  # Fall through to normal loading
        
        # Load base model if not using quantized model or smoothquant pipeline
        if not use_smoothquant_pipeline and not use_duquant_pipeline and (quantized_model_path is None or quantized_model_path == ''):
            self.model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs)
            
            # Load smoothed model weights if provided
            if smoothed_model_path is not None and smoothed_model_path != '':
                if os.path.exists(smoothed_model_path):
                    print(f"Loading SmoothQuant-preprocessed weights from {smoothed_model_path}...")
                    state_dict = torch.load(smoothed_model_path, map_location=device, weights_only=True)
                    self.model.load_state_dict(state_dict, strict=False)
                    print("✓ Smoothed weights loaded")
                else:
                    print(f"WARNING: Smoothed model path not found at {smoothed_model_path}")
                    print("Proceeding with base model weights.")
        
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            # For quantized models, we still need to prepare with accelerator
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)

        self.mask_id = mask_id
        # Use the same path as the model for tokenizer when loading quantized models
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.use_cache = use_cache
        self.threshold = threshold
        self.factor = factor
        self.is_instruct = True if 'instruct' in model_path.lower() else False
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.dual_cache = dual_cache
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    
    
    def generate_until(self, requests):
        output = []
        num_tokens = 0
        num_nfe = 0
        processed_count = 0
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f'rank_{rank}.jsonl')
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, 'r', encoding='utf-8') as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")
        
        batched_requests = [[]]
        for i, req in enumerate(tqdm(requests, desc="Batching...")):
            if i < processed_count:
                continue
            batched_requests[-1].append(req)
            if len(batched_requests[-1]) == self.batch_size:
                batched_requests.append([])
        
        if len(batched_requests[-1]) == 0:
            batched_requests.pop()

        start_time = time.time()

        for batch in tqdm(batched_requests, desc="Generating..."):
            # #region agent log
            batch_start_time = time.time()
            _debug_log("debug-session", "run1", "H3", "eval_llada.py:320", "Batch processing start", {"batch_size": len(batch), "batch_idx": len(output) // self.batch_size if self.batch_size > 0 else 0})
            # #endregion
            batched_input_ids = []
            max_len = 0
            pad_len = []
            seq_lens = []
            for req in batch:
                question = req.args[0]
                if self.is_instruct:
                    m = [{"role": "user", "content": question}]
                    user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                    input_ids = self.tokenizer(user_input)['input_ids']
                else:
                    user_input = question
                    input_ids = self.tokenizer(user_input)['input_ids']
                batched_input_ids.append(input_ids)
                seq_lens.append(len(input_ids))
                max_len = max(max_len, len(input_ids))
                pad_len.append(max_len - len(input_ids))
            
            # #region agent log
            _debug_log("debug-session", "run1", "H2", "eval_llada.py:338", "Before padding", {"batch_size": len(batch), "seq_lens": seq_lens, "max_len": max_len, "total_padding": sum(pad_len), "padding_ratio": sum(pad_len) / (max_len * len(batch)) if max_len > 0 else 0})
            padding_start = time.time()
            # #endregion
            
            # pad batched_input_ids to the same length
            batched_input_ids = [torch.cat([torch.full((1, max_len - len(input_ids)), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device), torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)], dim=1) for input_ids in batched_input_ids]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            batched_input_ids = batched_input_ids.to(self.device)
            
            # #region agent log
            padding_time = time.time() - padding_start
            _debug_log("debug-session", "run1", "H2", "eval_llada.py:340", "After padding", {"padding_time_ms": padding_time * 1000})
            # #endregion
            
            # #region agent log
            _debug_log("debug-session", "run1", "H1", "eval_llada.py:366", "Attention mask skipped (not used by generate functions)", {})
            # #endregion
            # Note: Attention mask creation removed - generate() functions don't accept attention_mask parameter,
            # so creating it was wasting ~1ms + 14MB memory per batch with no benefit


            stop_tokens = req.args[1]['until']
            input_ids = batched_input_ids
            # #region agent log
            gen_start = time.time()
            _debug_log("debug-session", "run1", "H4", "eval_llada.py:352", "Generation start", {"batch_size": input_ids.shape[0], "input_seq_len": input_ids.shape[1], "gen_length": self.gen_length, "use_cache": self.use_cache})
            # #endregion
            if self.use_cache:
                if self.dual_cache:
                    generated_answer, nfe = generate_with_dual_cache(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor)
                else:
                    generated_answer, nfe = generate_with_prefix_cache(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor)
            else:
                generated_answer, nfe = generate(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor)
            # #region agent log
            gen_time = time.time() - gen_start
            batch_time = time.time() - batch_start_time
            _debug_log("debug-session", "run1", "H3", "eval_llada.py:361", "Generation complete", {"gen_time_ms": gen_time * 1000, "batch_time_ms": batch_time * 1000, "nfe": nfe, "nfe_per_sample": nfe / input_ids.shape[0] if input_ids.shape[0] > 0 else 0})
            # #endregion

            if self.is_instruct and 'task_id' in req.doc and str(req.doc['task_id']).lower().startswith('humaneval'):
                generated_answer_ids = generated_answer[:, input_ids.shape[1]:]
                if self.show_speed:
                    num_tokens += (generated_answer_ids != 126081).sum()
                    num_nfe += nfe
                batched_generated_answer = [self.tokenizer.decode(generated_answer_ids[i], skip_special_tokens=True) for i in range(len(generated_answer_ids))]
            else:
                batched_generated_answer = []
                for i in range(len(generated_answer)):
                    generated_answer_i = self.tokenizer.decode(generated_answer[i][input_ids.shape[1]:], skip_special_tokens=False)
                    for stop_seq in stop_tokens:
                        if stop_seq in generated_answer_i:
                            generated_answer_i = generated_answer_i.split(stop_seq)[0]
                    generated_answer_ids = torch.tensor(self.tokenizer(generated_answer_i)["input_ids"])
                    if self.show_speed:
                        num_tokens += (generated_answer_ids != 126081).sum()
                        num_nfe += nfe
                    generated_answer_i = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
                    batched_generated_answer.append(generated_answer_i)

            # output.append(generated_answer)
            output.extend(batched_generated_answer)

            if self.save_dir is not None:
                # Incrementally save newly generated answers
                with open(save_path, 'a', encoding='utf-8') as f:
                    for generated_answer in batched_generated_answer:
                        f.write(json.dumps(generated_answer, ensure_ascii=False) + '\n')

            for i in range(len(batched_generated_answer)):
                print('=' * 20)
                # print('question: ', question)
                print('answer: ', batched_generated_answer[i])
                print('nfe: ', nfe)
                print('avg nfe: ', num_nfe / len(output))
                print('=' * 20, end='\n\n')
            # self.accelerator.wait_for_everyone()
        end_time = time.time()
        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
            print(f"Total NFE is {num_nfe}")
            
        return output


if __name__ == "__main__":
    cli_evaluate()
    
