"""
Test harness to compare original vs fused (pre-quantization) LLaDA models via logit MSE.

Usage example:

    python llada/test_fusion_mse.py \
        --model-path-orig GSAI-ML/LLaDA-8B-Instruct \
        --model-path-fused path/to/fused_llada \
        --device cuda \
        --batch-size 4 \
        --seq-len 512

Interpretation:
  - MSE > 1e-3 (FP16): likely fusion broke numerical precision.
  - MSE ~ 1e-5 or lower: models are effectively identical; small eval drops are
    probably due to randomness / decoding differences.
"""

import argparse
import json
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM
from quantization_calibration_dataset import LLaDACalibrationDataset
from quantize_llada_w4a8 import fix_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original vs fused LLaDA models via logit MSE on a single batch."
    )
    parser.add_argument(
        "--model-path-orig",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="HuggingFace path or local dir for the original (unfused) LLaDA model.",
    )
    parser.add_argument(
        "--model-path-fused",
        type=str,
        required=True,
        help=(
            "Path for the fused (pre-quantization) LLaDA model. "
            "Can be a from_pretrained directory or a .pt state dict file."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of sequences in the test batch.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length for the test batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=1e-3,
        help="Fail threshold for FP16 MSE between logits.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Optional path to append JSON summary of the run.",
    )
    return parser.parse_args()


def load_model(model_path: str, device: str, base_model_path: str | None = None) -> LLaDAModelLM:
    """
    Load LLaDA model.

    If `model_path` is a directory or HF hub id, it is passed directly to
    `LLaDAModelLM.from_pretrained`, matching app.py/chat.py.

    If `model_path` points to a .pt file (state dict), we:
      - load a base model from `base_model_path` (HF id or directory)
      - load the state dict into that base model.
    """
    if os.path.isfile(model_path):
        if base_model_path is None:
            raise ValueError(
                "base_model_path must be provided when model_path is a weights file "
                "(e.g., .pt state dict)."
            )
        # Load base model then apply fused weights
        base_model = LLaDAModelLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        base_model.load_state_dict(state_dict, strict=False)
        base_model.eval()
        return base_model

    # from_pretrained-style path (HF id or local directory)
    model = LLaDAModelLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    return model


def build_single_batch(
    tokenizer, batch_size: int, seq_len: int, device: str
) -> torch.LongTensor:
    dataset = LLaDACalibrationDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        samples=batch_size,
        block_size=32,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    return input_ids


def get_logits(model: LLaDAModelLM, input_ids: torch.LongTensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    if hasattr(outputs, "logits"):
        return outputs.logits
    # Fallback to tuple-style outputs
    return outputs[0]


def compute_mse_and_stats(
    logits_orig: torch.Tensor, logits_fused: torch.Tensor
) -> Tuple[float, float]:
    diff = logits_orig - logits_fused
    mse = torch.mean(diff ** 2).item()
    max_abs = diff.abs().max().item()
    return mse, max_abs


def main() -> None:
    args = parse_args()

    print("\n=== LLaDA Fusion MSE Test ===")
    print(f"Original model path : {args.model_path_orig}")
    print(f"Fused model path    : {args.model_path_fused}")
    print(f"Device              : {args.device}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Seq len             : {args.seq_len}")
    print(f"Seed                : {args.seed}")
    print(f"MSE threshold (fail): {args.mse_threshold}")

    # Determinism
    fix_seed(args.seed)

    device = args.device

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path_orig, trust_remote_code=True
    )

    # Models
    print("\n[Step 1/3] Loading models...")
    orig_model = load_model(args.model_path_orig, device)
    # For fused model we allow a raw weights file; use the original model path as base.
    fused_model = load_model(
        args.model_path_fused,
        device,
        base_model_path=args.model_path_orig,
    )
    print("  ✓ Models loaded")

    # Single batch
    print("\n[Step 2/3] Building single test batch...")
    input_ids = build_single_batch(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
    )
    print(f"  ✓ Batch shape: {tuple(input_ids.shape)}")

    # Forward passes and MSE
    print("\n[Step 3/3] Running forward passes and computing MSE...")
    logits_orig = get_logits(orig_model, input_ids)
    logits_fused = get_logits(fused_model, input_ids)

    assert (
        logits_orig.shape == logits_fused.shape
    ), f"Logit shapes differ: {logits_orig.shape} vs {logits_fused.shape}"

    mse, max_abs_diff = compute_mse_and_stats(logits_orig, logits_fused)

    status = "PASS"
    note = "Models are numerically very close (likely identical within FP16 noise)."
    if mse > args.mse_threshold:
        status = "FAIL"
        note = (
            "MSE above FP16 tolerance. Fusion likely introduced a numerical issue."
        )
    elif mse > 1e-5:
        status = "WARN"
        note = (
            "MSE is moderate. Consider inspecting specific layers or rerunning with "
            "different seeds / more batches."
        )

    print("\n=== Results ===")
    print(f"MSE(logits)        : {mse:.6e}")
    print(f"max |Δlogits|      : {max_abs_diff:.6e}")
    print(f"Decision           : {status}")
    print(f"Note               : {note}")

    if args.log_path:
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        record = {
            "model_path_orig": args.model_path_orig,
            "model_path_fused": args.model_path_fused,
            "device": args.device,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "seed": args.seed,
            "mse_threshold": args.mse_threshold,
            "mse": mse,
            "max_abs_diff": max_abs_diff,
            "status": status,
        }
        with open(args.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"\nSummary appended to log file: {args.log_path}")


if __name__ == "__main__":
    main()

