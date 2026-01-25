"""
Simple DuQuant Quantizer for LLaDa

This module implements the core DuQuant algorithm:
1. Block-wise Rotation: Smooth outliers by distributing magnitude across channels
2. Zigzag Permutation: Reorder channels so outliers are distributed evenly across blocks
3. Quantization: Standard uniform affine quantization

Reference: DuQuant paper - https://arxiv.org/abs/2406.01721
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def get_hadamard_matrix(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Generate a normalized Hadamard matrix of the given size.
    The Hadamard matrix is orthogonal and used as the base rotation matrix.
    
    Args:
        size: Size of the matrix (must be a power of 2).
        device: Device to create the tensor on.
        dtype: Data type of the tensor.
    
    Returns:
        Normalized Hadamard matrix of shape (size, size).
    """
    # For sizes that are powers of 2, use recursive construction
    if size == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)
    
    # Check if size is power of 2
    if size & (size - 1) != 0:
        # Pad to next power of 2
        next_pow2 = 2 ** math.ceil(math.log2(size))
        H = get_hadamard_matrix(next_pow2, device, dtype)
        return H[:size, :size]
    
    # Recursive construction: H_2n = [[H_n, H_n], [H_n, -H_n]]
    H_half = get_hadamard_matrix(size // 2, device, dtype)
    H = torch.cat([
        torch.cat([H_half, H_half], dim=1),
        torch.cat([H_half, -H_half], dim=1)
    ], dim=0)
    
    # Normalize to make it orthogonal
    return H / math.sqrt(2)


def exchange_row_col(matrix: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """
    Exchange rows i and j, and columns i and j in the matrix.
    This creates a permutation that swaps two indices.
    
    Args:
        matrix: Input matrix.
        i: First index.
        j: Second index.
    
    Returns:
        Matrix with swapped rows and columns.
    """
    if i == j:
        return matrix.clone()
    
    result = matrix.clone()
    # Swap rows
    result[[i, j], :] = result[[j, i], :]
    # Swap columns
    result[:, [i, j]] = result[:, [j, i]]
    return result


class SimpleQuantizer(nn.Module):
    """
    Simple DuQuant quantizer implementing rotation, permutation, and quantization.
    
    This quantizer transforms weights/activations using:
    1. Greedy rotation to smooth outliers
    2. Zigzag permutation to distribute outliers across blocks
    3. Standard uniform quantization
    
    Args:
        n_bits: Number of bits for quantization (default: 8).
        symmetric: Use symmetric quantization (default: False).
        block_size: Size of quantization blocks (default: 128).
        max_rotation_step: Maximum rotation iterations (default: 256).
        permutation_times: Number of permutation iterations (default: 1).
    """
    
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        block_size: int = 128,
        max_rotation_step: int = 256,
        permutation_times: int = 1,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.block_size = block_size
        self.max_rotation_step = max_rotation_step
        self.permutation_times = permutation_times
        
        # Calibrated parameters (populated during online_duquant_cali)
        self.register_buffer('R', None)  # Final rotation matrix
        self.register_buffer('R_inv', None)  # Inverse rotation matrix
        self.register_buffer('perm', None)  # Permutation indices
        self.register_buffer('perm_inv', None)  # Inverse permutation indices
        self.register_buffer('scale', None)  # Quantization scale
        self.register_buffer('zero_point', None)  # Quantization zero point
        
        self.calibrated = False
    
    @torch.no_grad()
    def rotation(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply greedy rotation to smooth outliers.
        
        The algorithm iteratively finds the channel with the largest outlier
        and applies a rotation to distribute its magnitude.
        
        Args:
            weight: Weight tensor of shape (out_features, in_features).
        
        Returns:
            Tuple of (rotated_weight, rotation_matrix).
        """
        device = weight.device
        dtype = weight.dtype
        out_features, in_features = weight.shape
        
        # Process in blocks
        num_blocks = (in_features + self.block_size - 1) // self.block_size
        
        # Initialize full rotation matrix as identity
        R_full = torch.eye(in_features, device=device, dtype=dtype)
        weight_rotated = weight.clone()
        
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, in_features)
            block_size = end - start
            
            if block_size < 2:
                continue
            
            # Get the block
            weight_block = weight_rotated[:, start:end].clone()
            
            # Get rotation base (Hadamard matrix)
            Rot = get_hadamard_matrix(block_size, device, dtype)
            
            # Initialize block rotation as identity
            R_block = torch.eye(block_size, device=device, dtype=dtype)
            
            for step in range(min(self.max_rotation_step, block_size)):
                # Find the column with the largest peak value
                col_max = weight_block.abs().max(dim=0)[0]
                peak_idx = col_max.argmax().item()
                
                # If the peak is small enough, stop
                if col_max[peak_idx] < 1e-6:
                    break
                
                # Create rotation that mixes the peak channel with others
                # Swap the peak column to position 0, apply Hadamard, swap back
                R_step = exchange_row_col(Rot, 0, peak_idx)
                
                # Apply rotation to weight block
                weight_block = weight_block @ R_step
                
                # Accumulate rotation
                R_block = R_block @ R_step
            
            # Update weight and full rotation matrix
            weight_rotated[:, start:end] = weight_block
            R_full[start:end, start:end] = R_block
        
        return weight_rotated, R_full
    
    @torch.no_grad()
    def permutation_zigzag(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply zigzag permutation to distribute outliers evenly across blocks.
        
        Channels are sorted by magnitude and assigned in a zigzag pattern
        across quantization blocks to ensure each block gets a mix of 
        large and small channels.
        
        Args:
            weight: Weight tensor of shape (out_features, in_features).
        
        Returns:
            Tuple of (permuted_weight, permutation_indices).
        """
        device = weight.device
        out_features, in_features = weight.shape
        
        # Compute per-channel magnitude (max across output dimension)
        channel_max = weight.abs().max(dim=0)[0]
        
        # Sort channels by magnitude (descending)
        sorted_indices = torch.argsort(channel_max, descending=True)
        
        # Number of blocks
        num_blocks = (in_features + self.block_size - 1) // self.block_size
        
        # Zigzag assignment: distribute channels across blocks
        # Block 0 gets indices 0, 2*num_blocks-1, 2*num_blocks, 4*num_blocks-1, ...
        # Block 1 gets indices 1, 2*num_blocks-2, 2*num_blocks+1, 4*num_blocks-2, ...
        
        perm = torch.zeros(in_features, dtype=torch.long, device=device)
        
        # Create block assignments using zigzag
        block_positions = [[] for _ in range(num_blocks)]
        direction = 1  # 1 = forward, -1 = backward
        current_block = 0
        
        for i, sorted_idx in enumerate(sorted_indices):
            block_positions[current_block].append(sorted_idx.item())
            
            # Move to next block in zigzag pattern
            if direction == 1:
                if current_block == num_blocks - 1:
                    direction = -1
                else:
                    current_block += 1
            else:
                if current_block == 0:
                    direction = 1
                else:
                    current_block -= 1
        
        # Flatten block assignments to create permutation
        idx = 0
        for block_idx in range(num_blocks):
            for channel_idx in block_positions[block_idx]:
                perm[idx] = channel_idx
                idx += 1
        
        # Apply permutation
        weight_permuted = weight[:, perm]
        
        return weight_permuted, perm
    
    @torch.no_grad()
    def online_duquant_cali(self, weight: torch.Tensor) -> None:
        """
        Main calibration loop for DuQuant.
        
        Performs:
        1. For each permutation iteration:
           a. Apply rotation
           b. Apply zigzag permutation
        2. Final rotation
        3. Compute quantization parameters
        
        Args:
            weight: Weight tensor to calibrate on.
        """
        device = weight.device
        dtype = weight.dtype
        out_features, in_features = weight.shape
        
        # Initialize
        R_total = torch.eye(in_features, device=device, dtype=dtype)
        perm_total = torch.arange(in_features, device=device, dtype=torch.long)
        
        current_weight = weight.clone()
        
        # Iterative rotation + permutation
        for _ in range(self.permutation_times):
            # Rotation step
            current_weight, R = self.rotation(current_weight)
            R_total = R_total @ R
            
            # Permutation step
            current_weight, perm = self.permutation_zigzag(current_weight)
            perm_total = perm_total[perm]
        
        # Final rotation
        current_weight, R_final = self.rotation(current_weight)
        R_total = R_total @ R_final
        
        # Store calibrated parameters
        self.R = R_total
        self.R_inv = R_total.T  # Orthogonal matrix: inverse = transpose
        self.perm = perm_total
        
        # Compute inverse permutation
        perm_inv = torch.zeros_like(perm_total)
        perm_inv[perm_total] = torch.arange(in_features, device=device, dtype=torch.long)
        self.perm_inv = perm_inv
        
        # Compute quantization parameters for the transformed weight
        self._compute_quant_params(current_weight)
        
        self.calibrated = True
    
    @torch.no_grad()
    def _compute_quant_params(self, weight: torch.Tensor) -> None:
        """
        Compute quantization scale and zero point for the weight.
        
        Args:
            weight: Transformed weight tensor.
        """
        if self.symmetric:
            # Symmetric quantization
            qmin = -(2 ** (self.n_bits - 1))
            qmax = 2 ** (self.n_bits - 1) - 1
            
            # Per-channel quantization (per output channel)
            abs_max = weight.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
            scale = abs_max / qmax
            zero_point = torch.zeros_like(scale)
        else:
            # Asymmetric quantization
            qmin = 0
            qmax = 2 ** self.n_bits - 1
            
            # Per-channel quantization
            w_min = weight.min(dim=1, keepdim=True)[0]
            w_max = weight.max(dim=1, keepdim=True)[0]
            
            scale = (w_max - w_min) / (qmax - qmin)
            scale = scale.clamp(min=1e-8)
            zero_point = qmin - w_min / scale
        
        self.scale = scale
        self.zero_point = zero_point
    
    @torch.no_grad()
    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fake quantization (quantize + dequantize).
        
        Args:
            x: Input tensor.
        
        Returns:
            Fake-quantized tensor.
        """
        if self.n_bits >= 16:
            return x
        
        if self.symmetric:
            qmin = -(2 ** (self.n_bits - 1))
            qmax = 2 ** (self.n_bits - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** self.n_bits - 1
        
        # Quantize
        x_int = (x / self.scale + self.zero_point).round().clamp(qmin, qmax)
        
        # Dequantize
        x_dequant = (x_int - self.zero_point) * self.scale
        
        return x_dequant
    
    def transform_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Apply the calibrated transformations to a weight tensor.
        
        Args:
            weight: Original weight tensor.
        
        Returns:
            Transformed weight tensor.
        """
        if not self.calibrated:
            raise RuntimeError("Quantizer not calibrated. Call online_duquant_cali first.")
        
        # Apply rotation
        weight_transformed = weight @ self.R
        
        # Apply permutation
        weight_transformed = weight_transformed[:, self.perm]
        
        return weight_transformed
    
    def inverse_transform(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse transformations to recover original weight space.
        
        Args:
            weight: Transformed weight tensor.
        
        Returns:
            Weight tensor in original space.
        """
        if not self.calibrated:
            raise RuntimeError("Quantizer not calibrated. Call online_duquant_cali first.")
        
        # Inverse permutation
        weight_recovered = weight[:, self.perm_inv]
        
        # Inverse rotation
        weight_recovered = weight_recovered @ self.R_inv
        
        return weight_recovered
    
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Apply DuQuant transformation and quantization.
        
        Args:
            weight: Original weight tensor.
        
        Returns:
            Quantized weight tensor (in original space).
        """
        if not self.calibrated:
            # If not calibrated, just do basic quantization
            return self._basic_quant(weight)
        
        # Transform to DuQuant space
        weight_transformed = self.transform_weight(weight)
        
        # Quantize in transformed space
        weight_quant = self.fake_quant(weight_transformed)
        
        # Transform back to original space
        weight_recovered = self.inverse_transform(weight_quant)
        
        return weight_recovered
    
    def _basic_quant(self, x: torch.Tensor) -> torch.Tensor:
        """Basic quantization without DuQuant transforms."""
        if self.n_bits >= 16:
            return x
        
        if self.symmetric:
            qmin = -(2 ** (self.n_bits - 1))
            qmax = 2 ** (self.n_bits - 1) - 1
            abs_max = x.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
            scale = abs_max / qmax
            zero_point = torch.zeros_like(scale)
        else:
            qmin = 0
            qmax = 2 ** self.n_bits - 1
            x_min = x.min(dim=1, keepdim=True)[0]
            x_max = x.max(dim=1, keepdim=True)[0]
            scale = (x_max - x_min) / (qmax - qmin)
            scale = scale.clamp(min=1e-8)
            zero_point = qmin - x_min / scale
        
        x_int = (x / scale + zero_point).round().clamp(qmin, qmax)
        x_dequant = (x_int - zero_point) * scale
        
        return x_dequant
    
    def extra_repr(self) -> str:
        return (
            f'n_bits={self.n_bits}, symmetric={self.symmetric}, '
            f'block_size={self.block_size}, max_rotation_step={self.max_rotation_step}, '
            f'permutation_times={self.permutation_times}, calibrated={self.calibrated}'
        )
