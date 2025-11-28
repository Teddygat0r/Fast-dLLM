import torch
import random
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

class FastDLLMCalibrationDataset(Dataset):
    def __init__(self, 
                 tokenizer, 
                 seq_len=512, 
                 samples=128, 
                 block_size=32,
                 model_timesteps=1000):
        """
        Args:
            tokenizer: The loaded model tokenizer.
            seq_len: Max sequence length.
            samples: Total number of calibration samples to generate.
            block_size: Fast-dLLM block size (usually 32).
            model_timesteps: Total diffusion steps (usually 1000 or 100).
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.block_size = block_size
        self.model_timesteps = model_timesteps
        
        raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        self.buffer = []
        
        target_ratios = [1.0, 0.9, 0.7, 0.4, 0.1]
        
        if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
            self.mask_token_id = tokenizer.mask_token_id
        else:
            self.mask_token_id = 151665 
            
        print(f"Building Calibration Buffer with Mask ID: {self.mask_token_id}...")

        # Fill buffer
        # We pick (samples / 5) items for each ratio to keep total count exact
        samples_per_ratio = samples // len(target_ratios)
        
        iter_data = iter(raw_data)
        
        for ratio in target_ratios:
            count = 0
            while count < samples_per_ratio:
                try:
                    entry = next(iter_data)
                except StopIteration:
                    iter_data = iter(raw_data) # Restart dataset if needed
                    entry = next(iter_data)

                text = entry['text']
                if len(text) < 50: continue

                # Tokenize
                tokens = tokenizer(text, max_length=seq_len, truncation=True, 
                                 padding="max_length", return_tensors="pt")
                input_ids = tokens.input_ids[0] # remove batch dim

                # Apply Block Masking
                masked_ids = self.apply_block_masking(input_ids, ratio)
                
                # Calculate Timestep (t)
                # If ratio is 1.0 (noise), t is high. If ratio is 0.0, t is 0.
                t = int(ratio * self.model_timesteps)
                
                self.buffer.append({
                    "input_ids": masked_ids,
                    "timestep": torch.tensor(t, dtype=torch.long)
                })
                count += 1
        
        # Shuffle so the quantizer doesn't adapt to one noise level at a time
        random.shuffle(self.buffer)
        print(f"Calibration Dataset Ready: {len(self.buffer)} tensors.")

    def apply_block_masking(self, input_ids, ratio):
        """Standard Fast-dLLM Block Masking"""
        seq_len = len(input_ids)
        masked_ids = input_ids.clone()
        
        # Create a boolean mask of the whole sequence first
        # 1 = MASK, 0 = KEEP
        mask = torch.zeros(seq_len, dtype=torch.bool)
        
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        for i in range(num_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, seq_len)
            block_len = end - start
            
            # Number of tokens to mask in this block
            n_mask = int(block_len * ratio)
            
            # Randomly select indices in this block to mask
            if n_mask > 0:
                indices = torch.randperm(block_len)[:n_mask] + start
                mask[indices] = True
                
        masked_ids[mask] = self.mask_token_id
        return masked_ids

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

# --- Usage in your PTQ Script ---
# from torch.utils.data import DataLoader
# dataset = FastDLLMCalibrationDataset(tokenizer)
# dataloader = DataLoader(dataset, batch_size=1)
# 
# for batch in dataloader:
#     # Feed this directly to model
#     model(input_ids=batch['input_ids'], timesteps=batch['timestep'])