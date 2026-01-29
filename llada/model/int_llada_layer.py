import torch
import torch.nn as nn
from model.modeling_llada import LLaDALlamaBlock


class LLaDaQuantLayer(nn.Module):
    def __init__(self, ori_layer: LLaDALlamaBlock):
        super().__init__(ori_layer.layer_id, ori_layer.config, ori_layer.__cache)
        self.ori_layer = ori_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ori_layer(x)
