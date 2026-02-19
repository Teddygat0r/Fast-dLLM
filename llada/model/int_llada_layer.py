from typing import Optional, Tuple
import torch
import torch.nn as nn
from model.quantize.int_linear import QuantLinear
from model.modeling_llada import LLaDALlamaBlock, ModelConfig, BufferCache
from model.quantize.int_matmul import QuantMatMul
import math

class LLaDaQuantLayer(LLaDALlamaBlock):
    def __init__(self, original_block: LLaDALlamaBlock, args):
        super().__init__(original_block.layer_id, original_block.config, original_block._LLaDALlamaBlock__cache)
        # Use same dtype as original block so QuantMatMul outputs match the rest of the block
        block_dtype = next(original_block.parameters()).dtype
        self.qkt_matmul = QuantMatMul(args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul, rotate=None, original_dtype=block_dtype)
        self.pv_matmul = QuantMatMul(args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul, rotate=None, original_dtype=block_dtype)
        self.flash_attn_func = None
        self.init_duquant_params = torch.tensor(0)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        replace_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)
        x_normed = self.attn_norm(x) #x:torch.Size([2, 168, 4096])
        q = self.q_proj(x_normed) #q:torch.Size([2, 168, 4096])
        if not self.init_duquant_params and isinstance(self.q_proj, QuantLinear) and self.q_proj.init_duquant_params:
            if isinstance(self.k_proj, QuantLinear) and isinstance(self.v_proj, QuantLinear):
                self.k_proj.copy_quantizers_duquant_params(self.q_proj)
                self.v_proj.copy_quantizers_duquant_params(self.q_proj)

        k = self.k_proj(x_normed) #k:torch.Size([2, 168, 4096])
        v = self.v_proj(x_normed) #v:torch.Size([2, 168, 4096])
        # attention_bias: None
        # layer_past: None
        # use_cache: False
        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache,replace_position=replace_position
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache,replace_position=replace_position)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x_ff = self.ff_proj(x)
        if not self.init_duquant_params and isinstance(self.ff_proj, QuantLinear) and self.ff_proj.init_duquant_params:
            if isinstance(self.up_proj, QuantLinear):
                self.up_proj.copy_quantizers_duquant_params(self.ff_proj)
        x_up = self.up_proj(x)
        x = x_ff
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = x * x_up # new add
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        if not self.init_duquant_params:
            self.init_duquant_params = torch.tensor(1)

        return x, cache

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        from fouroversix import fp4_matmul

        assert k.size(1) == v.size(1)
        num_kv_heads = k.size(1)
        num_q_heads = q.size(1)
        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0
            k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
            v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

        L, S = q.size(-2), k.size(-2)
        scale_factor = 1 / math.sqrt(q.size(-1)) 
        # no biases
        # attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)

        # fp4_matmul currently expects 2D inputs; run per (batch, head) slice.
        leading_shape = q.shape[:-2]
        q_flat = q.reshape(-1, L, q.size(-1))
        k_flat = k.reshape(-1, S, k.size(-1))
        attn_weight = torch.stack(
            [fp4_matmul(q_flat[i], k_flat[i]) for i in range(q_flat.size(0))], dim=0
        ).reshape(*leading_shape, L, S)

        attn_weight = attn_weight * scale_factor

        # attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=self.training)

        # fp4_matmul computes a @ b.T, so pass v.transpose to recover attn_weight @ v.
        D = v.size(-1)
        attn_flat = attn_weight.reshape(-1, L, S).to(dtype=q.dtype)
        v_flat = v.reshape(-1, S, D)
        return torch.stack(
            [fp4_matmul(attn_flat[i], v_flat[i].transpose(-2, -1).contiguous()) for i in range(attn_flat.size(0))], dim=0
        ).reshape(*leading_shape, L, D)
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.qkt_matmul.set_quant_state(weight_quant, act_quant)
        self.pv_matmul.set_quant_state(weight_quant, act_quant)

    def register_duquant_params(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_duquant_params()
                module.act_quantizer.register_duquant_params()