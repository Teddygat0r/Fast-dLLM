"""
W8A8 Quantized Linear Layer

Implements a linear layer with:
- INT8 per-channel weight quantization (static, computed once)
- INT8 dynamic per-token activation quantization (computed at runtime)
- Optional SmoothQuant input scaling (integrated, not fused into LayerNorm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from .quantization_utils import quantize_per_channel, quantize_per_token


class W8A8Linear(nn.Module):
    """
    INT8 Weight, INT8 Activation Linear Layer with optional SmoothQuant.
    
    Weights are quantized per output channel (row-wise) and stored as INT8.
    Activations are dynamically quantized per token at runtime.
    
    When SmoothQuant is enabled (smooth_scale is set):
    - Input is multiplied by smooth_scale before quantization
    - Weights were pre-scaled by smooth_scale during quantization
    - This migrates quantization difficulty from activations to weights
    
    The forward pass performs:
    1. (Optional) Apply SmoothQuant input scaling
    2. Dynamic per-token quantization of input activations
    3. INT8 matrix multiplication (simulated via dequantization for compatibility)
    4. Rescaling with combined activation and weight scales
    5. Optional bias addition
    
    Attributes:
        in_features: Size of each input sample
        out_features: Size of each output sample
        weight_int8: Quantized INT8 weights [out_features, in_features]
        weight_scale: Per-channel scale [out_features]
        smooth_scale: Optional SmoothQuant input scale [in_features]
        bias: Optional bias tensor [out_features]
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize W8A8Linear layer.
        
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If True, adds a learnable bias
            device: Device for the layer
            dtype: Data type for non-quantized parameters (bias, scales)
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Register buffers for quantized weights and scales
        # These are not parameters - they don't require gradients
        self.register_buffer(
            'weight_int8',
            torch.zeros(out_features, in_features, dtype=torch.int8, device=device)
        )
        self.register_buffer(
            'weight_scale',
            torch.ones(out_features, dtype=dtype or torch.float32, device=device)
        )
        
        # SmoothQuant input scaling (None means no smoothing)
        # When set, this is applied to inputs before per-token quantization
        self.register_buffer('smooth_scale', None)
        
        # Bias remains in floating point
        if bias:
            self.register_buffer(
                'bias',
                torch.zeros(out_features, dtype=dtype or torch.float32, device=device)
            )
        else:
            self.register_parameter('bias', None)
        
        # Store the compute dtype for activations
        self._compute_dtype = dtype or torch.float16
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        compute_dtype: Optional[torch.dtype] = None,
        smooth_scale: Optional[torch.Tensor] = None,
    ) -> 'W8A8Linear':
        """
        Create a W8A8Linear layer from an existing nn.Linear layer.
        
        This is the primary way to convert a pretrained model to use
        quantized layers.
        
        Args:
            linear: The source nn.Linear layer to quantize
            compute_dtype: Data type for computation (default: same as linear weights)
            smooth_scale: Optional SmoothQuant scale tensor [in_features].
                         If provided, weights are pre-scaled and the scale is stored
                         for input scaling at runtime.
            
        Returns:
            A new W8A8Linear layer with quantized weights
        """
        # Determine compute dtype
        if compute_dtype is None:
            compute_dtype = linear.weight.dtype
        
        # Create the quantized layer
        has_bias = linear.bias is not None
        quant_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            device=linear.weight.device,
            dtype=compute_dtype,
        )
        
        # Get weight data
        weight_data = linear.weight.data
        
        # Apply SmoothQuant scaling to weights if provided
        # The weights are multiplied by smooth_scale (absorbing the activation scaling)
        if smooth_scale is not None:
            # smooth_scale shape: [in_features]
            # weight shape: [out_features, in_features]
            # We multiply each column of weight by the corresponding scale
            smooth_scale = smooth_scale.to(weight_data.device).to(weight_data.dtype)
            weight_data = weight_data * smooth_scale.view(1, -1)
            
            # Store the smooth_scale for input scaling at runtime
            # Note: we store 1/smooth_scale because we'll multiply inputs by it
            # (equivalent to dividing by smooth_scale, which is what SmoothQuant does)
            quant_linear.smooth_scale = (1.0 / smooth_scale).to(compute_dtype)
        
        # Quantize weights per channel
        weight_int8, weight_scale = quantize_per_channel(weight_data)
        
        # Copy quantized weights and scales
        quant_linear.weight_int8.copy_(weight_int8)
        quant_linear.weight_scale.copy_(weight_scale.to(compute_dtype))
        
        # Copy bias if present
        if has_bias:
            quant_linear.bias.copy_(linear.bias.data.to(compute_dtype))
        
        quant_linear._compute_dtype = compute_dtype
        
        return quant_linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with W8A8 quantization.
        
        Args:
            x: Input tensor of shape [..., in_features]
            
        Returns:
            Output tensor of shape [..., out_features]
        """
        # Store original shape and dtype
        original_shape = x.shape
        original_dtype = x.dtype
        
        # Flatten to [num_tokens, in_features]
        x_flat = x.view(-1, self.in_features)
        
        # Apply SmoothQuant input scaling if enabled
        # This is equivalent to dividing by smooth_scale (we stored 1/smooth_scale)
        # The corresponding weight scaling was applied during quantization
        if self.smooth_scale is not None:
            x_flat = x_flat * self.smooth_scale.to(x_flat.dtype)
        
        # Dynamic per-token activation quantization
        x_int8, x_scale = quantize_per_token(x_flat)
        # x_int8: [num_tokens, in_features], x_scale: [num_tokens, 1]
        
        # Perform matrix multiplication
        # For true INT8 speedup, we would use torch._int_mm or similar
        # Here we simulate by dequantizing for compatibility
        
        # Option 1: Simulated INT8 matmul (more compatible)
        # Dequantize both and compute in float, then we get the correct result
        # output = (x_int8 @ weight_int8.T) * (x_scale * weight_scale)
        
        # Convert to float for matmul (INT8 matmul requires special support)
        x_float = x_int8.to(self._compute_dtype)
        weight_float = self.weight_int8.to(self._compute_dtype)
        
        # Compute matmul: [num_tokens, in_features] @ [in_features, out_features]
        # weight is [out_features, in_features], so transpose it
        output = F.linear(x_float, weight_float, bias=None)
        # output: [num_tokens, out_features]
        
        # Apply scales: x_scale is [num_tokens, 1], weight_scale is [out_features]
        # Combined scale for each element [i, j] is x_scale[i] * weight_scale[j]
        output = output * x_scale.to(self._compute_dtype)  # [num_tokens, out_features]
        output = output * self.weight_scale.unsqueeze(0).to(self._compute_dtype)  # broadcast
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.to(self._compute_dtype)
        
        # Reshape back to original shape
        output_shape = list(original_shape[:-1]) + [self.out_features]
        output = output.view(output_shape)
        
        # Cast back to original dtype if needed
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        
        return output
    
    def forward_with_int8_gemm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using actual INT8 GEMM when available.
        
        This method uses torch._int_mm for potentially faster INT8 computation
        on supported hardware. Falls back to simulated if not available.
        
        Args:
            x: Input tensor of shape [..., in_features]
            
        Returns:
            Output tensor of shape [..., out_features]
        """
        # Check if INT8 GEMM is available
        if not hasattr(torch, '_int_mm'):
            return self.forward(x)
        
        original_shape = x.shape
        original_dtype = x.dtype
        
        # Flatten to [num_tokens, in_features]
        x_flat = x.view(-1, self.in_features)
        
        # Apply SmoothQuant input scaling if enabled
        if self.smooth_scale is not None:
            x_flat = x_flat * self.smooth_scale.to(x_flat.dtype)
        
        # Dynamic per-token quantization
        x_int8, x_scale = quantize_per_token(x_flat)
        
        try:
            # Use INT8 GEMM: requires contiguous tensors
            x_int8 = x_int8.contiguous()
            weight_t = self.weight_int8.t().contiguous()
            
            # INT8 matmul returns INT32
            output_int32 = torch._int_mm(x_int8, weight_t)
            
            # Dequantize: convert to float and apply scales
            output = output_int32.to(self._compute_dtype)
            output = output * x_scale.to(self._compute_dtype)
            output = output * self.weight_scale.unsqueeze(0).to(self._compute_dtype)
            
            if self.bias is not None:
                output = output + self.bias.to(self._compute_dtype)
            
            output_shape = list(original_shape[:-1]) + [self.out_features]
            output = output.view(output_shape)
            
            if output.dtype != original_dtype:
                output = output.to(original_dtype)
            
            return output
            
        except Exception:
            # Fall back to simulated version
            return self.forward(x)
    
    @property
    def has_smooth_scale(self) -> bool:
        """Check if SmoothQuant scaling is enabled for this layer."""
        return self.smooth_scale is not None
    
    def extra_repr(self) -> str:
        smooth_info = ", smooth=True" if self.has_smooth_scale else ""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}{smooth_info}'
    
    def __repr__(self) -> str:
        return f'W8A8Linear({self.extra_repr()})'


def replace_linear_with_w8a8(
    module: nn.Module,
    exclude_names: Optional[list] = None,
    compute_dtype: Optional[torch.dtype] = None,
    smooth_scales: Optional[Dict[str, torch.Tensor]] = None,
    prefix: str = "",
) -> nn.Module:
    """
    Recursively replace all nn.Linear layers in a module with W8A8Linear.
    
    Args:
        module: The module to process
        exclude_names: List of layer name patterns to exclude from quantization
        compute_dtype: Data type for computation
        smooth_scales: Optional dictionary mapping layer names to SmoothQuant scales.
                      If provided, layers with matching names will have SmoothQuant
                      applied during quantization.
        prefix: Current name prefix (for recursive calls)
        
    Returns:
        The modified module with quantized linear layers
    """
    if exclude_names is None:
        exclude_names = []
    if smooth_scales is None:
        smooth_scales = {}
    
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        
        # Check if this layer should be excluded
        should_exclude = any(excl in full_name for excl in exclude_names)
        
        if isinstance(child, nn.Linear) and not should_exclude:
            # Get smooth scale if available for this layer
            layer_smooth_scale = smooth_scales.get(full_name, None)
            
            # Replace with quantized version
            quant_linear = W8A8Linear.from_linear(
                child, 
                compute_dtype=compute_dtype,
                smooth_scale=layer_smooth_scale,
            )
            setattr(module, name, quant_linear)
        else:
            # Recurse into child modules
            replace_linear_with_w8a8(
                child,
                exclude_names=exclude_names,
                compute_dtype=compute_dtype,
                smooth_scales=smooth_scales,
                prefix=full_name,
            )
    
    return module
