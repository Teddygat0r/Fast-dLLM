from typing import Optional
import torch
import torch.nn as nn
from model.modeling_llada import LLaDALlamaBlock, ModelConfig, BufferCache
from model.quantize.int_matmul import QuantMatMul
import math

class LLaDaQuantLayer(LLaDALlamaBlock):
    def __init__(self, original_block: LLaDALlamaBlock, args):
        super().__init__(original_block.layer_id, original_block.config, original_block._LLaDALlamaBlock__cache)
        self.load_state_dict(original_block.state_dict())
        
        self.qkt_matmul = QuantMatMul(args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul, rotate=None)
        self.pv_matmul = QuantMatMul(args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul, rotate=None)
        self.flash_attn_func = None
    
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
        assert k.size(1) == v.size(1)
        num_kv_heads = k.size(1)
        num_q_heads = q.size(1)
        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0
            k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
            v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

        L, S = q.size(-2), k.size(-2)
        scale_factor = 1 / math.sqrt(q.size(-1)) 
        attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)

        # attn_weight = q @ k.transpose(-2, -1) * scale_factor
        q = self.qkt_matmul.quant_x1(q)
        k = self.qkt_matmul.quant_x2(k).transpose(-2, -1)
        attn_weight = self.qkt_matmul(q, k) * scale_factor

        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

        # return attn_weight @ v
        attn_weight = self.pv_matmul.quant_x1(attn_weight)
        v = self.pv_matmul.quant_x2(v)
        return self.pv_matmul(attn_weight, v)
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.qkt_matmul.set_quant_state(weight_quant, act_quant)
        self.pv_matmul.set_quant_state(weight_quant, act_quant)
