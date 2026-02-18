"""
InferenceQuantizer — torch.compile-friendly activation quantizer for inference.

This is a streamlined version of UniformAffineQuantizer that only supports
the per-token dynamic asymmetric quantization path used at inference time
with DuQuant transforms (rotation + permutation).

All calibration, training, and unused code paths have been removed to make
this module fully compatible with torch.compile.

torch.compile design decisions:
  - No .to(device/dtype) calls in forward — buffers must be on correct
    device/dtype before first call. This avoids recompilation guards.
  - No data-dependent branches in forward — permutation type (2D vs 3D)
    is resolved at load time; lac defaults to 1.0 for branchless multiply.
  - No None checks on buffers — replaced with boolean flags set at init/load.
  - Static method signatures use only tensor + int/float args.
"""

import torch
import torch.nn as nn
from model.quantize.const import CLIPMIN, CLIPMAX


class InferenceQuantizer(nn.Module):
    """
    Inference-only activation quantizer with pre-loaded DuQuant parameters.

    Supports:
      - Per-token dynamic asymmetric quantization
      - DuQuant rotation + zigzag permutation transforms (pre-loaded as buffers)
      - LAC (learnable activation clipping) as a fixed scalar

    Does NOT support (by design):
      - Symmetric quantization
      - Group-size / deficiency padding
      - LWC / SWC / LET
      - Online DuQuant calibration
      - Straight-through estimator (no gradients needed)
      - smooth_scales (applied earlier in pipeline)
    """

    def __init__(
        self,
        n_bits: int = 4,
        lac: float = None,
        block_size: int = 128,
        permutation_times: int = 1,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** n_bits - 1

        # Default lac to 1.0 so we can always multiply unconditionally —
        # eliminates a branch in per_token_dynamic_calibration.
        self.lac = lac if lac is not None else 1.0

        self.block_size = block_size if block_size != -1 else 4096
        self.permutation_times = permutation_times

        # Flags resolved at load time — avoids data-dependent checks in forward
        self._has_permutations: bool = False
        self._use_swap_perms: bool = False  # True if permutation_list is 3D (swap-pair format)

        # Buffers initialized empty; load_duquant_params fills them.
        self.register_buffer('R', torch.empty(0))
        self.register_buffer('permutation_list', torch.empty(0))

    def load_duquant_params(self, state_dict: dict, layer_name: str = ""):
        """Load pre-calibrated R and permutation_list from a state dict."""
        for name, param in state_dict.items():
            if name.find(layer_name) > -1 and (
                name.find('R') > -1
                or name.find('permutation_list') > -1
            ):
                attr = name.split('.')[-1]
                if hasattr(self, attr):
                    delattr(self, attr)
                self.register_buffer(attr, param.clone().detach().to('cuda'))
        self._resolve_permutation_type()

    def copy_duquant_params(self, quantizer_ref):
        """Copy DuQuant buffers from another quantizer."""
        self.R = quantizer_ref.R.clone().detach()
        try:
            self.permutation_list = quantizer_ref.permutation_list.clone().detach()
        except Exception:
            self.permutation_list = quantizer_ref.permutation_list
        self._resolve_permutation_type()

    def _resolve_permutation_type(self):
        """Set compile-time flags based on loaded buffer shapes.
        Called once after loading — these booleans become static guards
        that torch.compile can specialize on without graph breaks."""
        self._has_permutations = (
            self.permutation_list is not None
            and self.permutation_list.numel() > 0
        )
        self._use_swap_perms = (
            self._has_permutations
            and len(self.permutation_list.shape) == 3
        )

    def prepare(self, dtype: torch.dtype = torch.bfloat16, device: torch.device = None):
        """Move buffers to the target device once, before first forward.
        R is kept in float32 for rotation precision — apply_duquant handles
        the upcast/downcast automatically."""
        if device is not None and self.R.numel() > 0:
            self.R = self.R.to(device=device)
        if device is not None and self.permutation_list.numel() > 0:
            self.permutation_list = self.permutation_list.to(device=device)

    def apply_duquant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pre-loaded DuQuant transforms: alternating rotation and
        permutation, followed by a final rotation.

        Handles both 2D [seq_len, hidden] and 3D [batch, seq_len, hidden] inputs.
        Rotations are performed in float32 for numerical precision,
        then the result is downcast back to the original input dtype.
        """
        orig_shape = x.shape
        input_dtype = x.dtype

        # Flatten to 2D [N, hidden_dim] so permutation x[:, perm] indexes the
        # hidden dimension correctly (not the seq_len dimension of a 3D tensor).
        x = x.reshape(-1, orig_shape[-1]).float()

        if self._has_permutations:
            for i in range(len(self.permutation_list)):
                # Rotation i (both x and R[i] are float32)
                x = torch.matmul(
                    x.reshape(-1, self.block_size), self.R[i]
                ).reshape(x.shape[0], -1)

                # Permutation i
                if self._use_swap_perms:
                    perm0 = self.permutation_list[i, 0]
                    perm1 = self.permutation_list[i, 1]
                    x[:, perm0], x[:, perm1] = x[:, perm1], x[:, perm0]
                else:
                    x = x[:, self.permutation_list[i]]

        # Final rotation
        if self.R.numel() > 0:
            x = torch.matmul(
                x.reshape(-1, self.block_size), self.R[-1]
            ).reshape(x.shape[0], -1)

        # Restore original shape and dtype
        return x.reshape(orig_shape).to(input_dtype)

    @staticmethod
    def per_token_dynamic_calibration(
        x: torch.Tensor,
        qmin: int,
        qmax: int,
        n_bits: int,
        lac: float,
    ):
        """
        Compute per-token scale and zero-point for asymmetric quantization.

        Returns (scale, round_zero_point) — no module state is mutated.
        lac is always a float (defaults to 1.0), so the multiply is unconditional.
        """
        xmin = x.amin(-1, keepdim=True)
        xmax = x.amax(-1, keepdim=True)

        # Branchless: lac is always a float (1.0 when disabled)
        xmax = lac * xmax
        xmin = lac * xmin

        range_ = xmax - xmin
        scale = (range_ / (2 ** n_bits - 1)).clamp(min=CLIPMIN, max=CLIPMAX)
        zero_point = (-(xmin) / scale).clamp(min=-CLIPMAX, max=CLIPMAX).round()

        return scale, zero_point

    @staticmethod
    def fake_quant(
        x: torch.Tensor,
        scale: torch.Tensor,
        round_zero_point: torch.Tensor,
        qmin: int,
        qmax: int,
    ) -> torch.Tensor:
        """
        Quantize then dequantize (fake quantization) for asymmetric scheme.
        No group-size reshaping, no deficiency padding, no STE.
        """
        x_int = (x.float() / scale).round().to(dtype=torch.bfloat16)
        x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(qmin, qmax)
        x_dequant = x_int.sub(round_zero_point).mul(scale)
        return x_dequant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. DuQuant transforms (rotation + permutation)
        x = self.apply_duquant(x)

        # 2. Per-token dynamic calibration (returns values, no state mutation)
        scale, round_zero_point = self.per_token_dynamic_calibration(
            x, self.qmin, self.qmax, self.n_bits, self.lac
        )

        # 3. Fake quantize
        return self.fake_quant(x, scale, round_zero_point, self.qmin, self.qmax)
