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

from .configuration_llada import LLaDAConfig
from .modeling_llada import LLaDAModelLM
# from .int_llada_layer import QuantLLadaDecoderLayer
# from .quantize.int_linear import QuantLinear
# from .quantize.int_matmul import QuantMatMul
# from .quantize.quantizer import UniformAffineQuantizer
# from .quantize.const import CLIPMIN, CLIPMAX
# from .quantize.utils import *
# from .transformation import *
__all__ = ['LLaDAConfig', 'LLaDAModelLM']
# __all__ = ['LLaDAConfig', 'LLaDAModelLM', 'QuantLLadaDecoderLayer', 'QuantLinear', 'QuantMatMul', 'UniformAffineQuantizer', 'CLIPMIN', 'CLIPMAX', 'smooth_parameters', 'let_parameters', 'lwc_parameters', 'get_duquant_parameters', 'get_post_parameters', 'set_requires_grad', 'duquant_state_dict', 'register_scales_and_zeros', 'TruncateFunction', 'truncate_number', 'post_rotate_quant_temporary', 'post_quant_inplace', 'clear_temp_variable', 'set_registered_x_none', 'set_init_duquant_params_state', 'smooth_and_quant_temporary', 'smooth_and_let_inplace', 'quant_inplace', 'quant_soft_inplace', 'set_quant_state']