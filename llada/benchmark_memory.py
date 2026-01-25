#!/usr/bin/env python3
"""
Memory Benchmark: Compare base model vs. SmoothQuant quantized model.

This script measures:
1. Model weight memory (actual GPU allocation)
2. Peak memory during forward pass (includes activations)
3. Memory comparison between base and quantized models

NOTE: The SmoothQuant implementation uses "fake quantization" - weights are still
stored in bfloat16, just with quantized values. This doesn't reduce model memory,
but affects numerical precision. True INT8 storage would require different kernels.

Usage:
    python benchmark_memory.py
    python benchmark_memory.py --model-path GSAI-ML/LLaDA-8B-Instruct
    python benchmark_memory.py --seq-len 256 --batch-size 1
"""

import argparse
import gc
import os
import time
import torch
import torch.nn as nn
from typing import Dict, Tuple


def get_memory_mb() -> float:
    """Get current GPU memory allocated in MB."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


def get_peak_memory_mb() -> float:
    """Get peak GPU memory allocated in MB."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def reset_memory():
    """Reset GPU memory statistics and clear cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def count_parameters_and_buffers(model: nn.Module) -> Dict[str, int]:
    """Count parameters and buffers (QuantLinear stores weights as buffers)."""
    param_count = sum(p.numel() for p in model.parameters())
    buffer_count = sum(b.numel() for b in model.buffers())
    return {
        "parameters": param_count,
        "buffers": buffer_count,
        "total": param_count + buffer_count,
    }


def estimate_tensor_memory(model: nn.Module) -> float:
    """Estimate model memory in MB from parameters AND buffers."""
    total_bytes = 0
    # Count parameters
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    # Count buffers (QuantLinear stores weights here)
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes / 1024**2


def benchmark_model(
    model: nn.Module,
    input_ids: torch.Tensor,
    model_name: str,
    num_warmup: int = 2,
    num_runs: int = 3,
) -> Dict[str, float]:
    """
    Benchmark memory usage for a model.
    
    Returns dict with memory stats in MB.
    """
    # Find device from parameters or buffers
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = next(model.buffers()).device
    
    input_ids = input_ids.to(device)
    
    # Measure model memory (what's allocated right now)
    model_memory = get_memory_mb()
    
    # Get tensor-based size estimate (includes buffers for QuantLinear)
    tensor_memory = estimate_tensor_memory(model)
    
    # Count parameters and buffers
    counts = count_parameters_and_buffers(model)
    
    # Warmup runs
    print(f"  Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(input_ids)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    
    # Reset peak stats for measurement
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # Measure forward pass memory
    print(f"  Measuring memory ({num_runs} runs)...")
    peak_memories = []
    
    for i in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        torch.cuda.synchronize()
        peak_mem = get_peak_memory_mb()
        peak_memories.append(peak_mem)
        
        # Clear intermediate tensors
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
    
    avg_peak = sum(peak_memories) / len(peak_memories)
    
    return {
        "model_name": model_name,
        "model_memory_mb": model_memory,
        "tensor_estimate_mb": tensor_memory,
        "num_parameters": counts["parameters"],
        "num_buffers": counts["buffers"],
        "total_elements": counts["total"],
        "peak_memory_mb": avg_peak,
        "activation_memory_mb": avg_peak - model_memory,
    }


def load_base_model(model_path: str, device: str = "cuda"):
    """Load the base (unquantized) model."""
    from model.modeling_llada import LLaDAModelLM
    
    print(f"\nLoading base model from {model_path}...")
    model = LLaDAModelLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_quantized_model(
    model_path: str,
    device: str = "cuda",
    alpha: float = 0.5,
    w_bits: int = 8,
    a_bits: int = 8,
    calibration_samples: int = 32,
    scales_path: str = None,
):
    """Load the SmoothQuant quantized model."""
    from smoothquant import apply_smoothquant_pipeline
    
    print(f"\nLoading quantized model (W{w_bits}A{a_bits}) from {model_path}...")
    
    # Check for pre-computed scales
    load_scales = scales_path if scales_path and __import__('os').path.exists(scales_path) else None
    save_scales = scales_path if scales_path and not load_scales else None
    
    model, _ = apply_smoothquant_pipeline(
        model_path=model_path,
        calibration_samples=calibration_samples,
        alpha=alpha,
        w_bits=w_bits,
        a_bits=a_bits,
        seq_len=256,
        batch_size=1,
        device=device,
        load_scales_path=load_scales,
        save_scales_path=save_scales,
    )
    model.eval()
    return model


def print_results(base_results: Dict, quant_results: Dict):
    """Print comparison results."""
    print("\n" + "=" * 75)
    print("MEMORY BENCHMARK RESULTS")
    print("=" * 75)
    
    print(f"\n{'Metric':<35} {'Base Model':>18} {'Quantized':>18}")
    print("-" * 75)
    
    # Model memory (actual GPU allocation)
    base_mem = base_results["model_memory_mb"]
    quant_mem = quant_results["model_memory_mb"]
    print(f"{'Model GPU Memory (MB)':<35} {base_mem:>14.1f} MB {quant_mem:>14.1f} MB")
    
    # Tensor estimate
    base_tensor = base_results["tensor_estimate_mb"]
    quant_tensor = quant_results["tensor_estimate_mb"]
    print(f"{'Tensor Estimate (MB)':<35} {base_tensor:>14.1f} MB {quant_tensor:>14.1f} MB")
    
    # Parameters vs Buffers
    base_params = base_results["num_parameters"] / 1e9
    quant_params = quant_results["num_parameters"] / 1e9
    base_buffers = base_results["num_buffers"] / 1e9
    quant_buffers = quant_results["num_buffers"] / 1e9
    print(f"{'Parameters (B)':<35} {base_params:>15.2f} B {quant_params:>15.2f} B")
    print(f"{'Buffers (B)':<35} {base_buffers:>15.2f} B {quant_buffers:>15.2f} B")
    
    # Total elements
    base_total = base_results["total_elements"] / 1e9
    quant_total = quant_results["total_elements"] / 1e9
    print(f"{'Total Elements (B)':<35} {base_total:>15.2f} B {quant_total:>15.2f} B")
    
    # Peak memory
    base_peak = base_results["peak_memory_mb"]
    quant_peak = quant_results["peak_memory_mb"]
    peak_diff = quant_peak - base_peak
    peak_pct = (peak_diff / base_peak) * 100 if base_peak > 0 else 0
    print(f"{'Peak Memory (MB)':<35} {base_peak:>14.1f} MB {quant_peak:>14.1f} MB")
    
    # Activation memory estimate
    base_act = base_results["activation_memory_mb"]
    quant_act = quant_results["activation_memory_mb"]
    print(f"{'Est. Activation Memory (MB)':<35} {base_act:>14.1f} MB {quant_act:>14.1f} MB")
    
    print("-" * 75)
    print(f"\n{'ANALYSIS':^75}")
    print("-" * 75)
    
    # Explain what we're seeing
    mem_diff = quant_mem - base_mem
    if abs(mem_diff) < base_mem * 0.05:  # Within 5%
        print("  Model Memory: ~Same (fake quantization keeps bfloat16 storage)")
    elif mem_diff > 0:
        print(f"  Model Memory: +{mem_diff:.1f} MB (quantized uses MORE memory)")
    else:
        print(f"  Model Memory: {mem_diff:.1f} MB ({-mem_diff:.1f} MB saved)")
    
    if peak_pct > 5:
        print(f"  Peak Memory:  +{peak_diff:.1f} MB (+{peak_pct:.1f}%) - quantization adds overhead")
    elif peak_pct < -5:
        print(f"  Peak Memory:  {peak_diff:.1f} MB ({-peak_pct:.1f}% saved)")
    else:
        print(f"  Peak Memory:  ~Same ({peak_diff:+.1f} MB)")
    
    print("\n  NOTE: This SmoothQuant implementation uses 'fake quantization':")
    print("        Weights are still stored as bfloat16 (just with quantized values).")
    print("        True INT8 storage would require specialized CUDA kernels.")
    print("=" * 75)


def print_single_result(results: Dict):
    """Print results for a single model."""
    print(f"\n{'Metric':<35} {'Value':>20}")
    print("-" * 60)
    print(f"{'Model GPU Memory (MB)':<35} {results['model_memory_mb']:>16.1f} MB")
    print(f"{'Tensor Estimate (MB)':<35} {results['tensor_estimate_mb']:>16.1f} MB")
    print(f"{'Parameters (B)':<35} {results['num_parameters']/1e9:>17.2f} B")
    print(f"{'Buffers (B)':<35} {results['num_buffers']/1e9:>17.2f} B")
    print(f"{'Peak Memory (MB)':<35} {results['peak_memory_mb']:>16.1f} MB")
    print(f"{'Est. Activation Memory (MB)':<35} {results['activation_memory_mb']:>16.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Memory benchmark: base vs quantized model")
    parser.add_argument("--model-path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="HuggingFace model path")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length for benchmark")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for benchmark")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="SmoothQuant alpha")
    parser.add_argument("--w-bits", type=int, default=8,
                        help="Weight quantization bits")
    parser.add_argument("--a-bits", type=int, default=8,
                        help="Activation quantization bits")
    parser.add_argument("--calibration-samples", type=int, default=32,
                        help="Calibration samples for quantization")
    parser.add_argument("--scales-path", type=str, default="models/act_scales_benchmark.pt",
                        help="Path to save/load activation scales")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip base model benchmark")
    parser.add_argument("--skip-quantized", action="store_true",
                        help="Skip quantized model benchmark")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for memory benchmarking")
        return
    
    print("=" * 75)
    print("MEMORY BENCHMARK: Base Model vs. SmoothQuant Quantized Model")
    print("=" * 75)
    print(f"Model:        {args.model_path}")
    print(f"Sequence len: {args.seq_len}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Quantization: W{args.w_bits}A{args.a_bits}, alpha={args.alpha}")
    print(f"Device:       {args.device}")
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create dummy input
    input_ids = torch.randint(0, 32000, (args.batch_size, args.seq_len))
    
    base_results = None
    quant_results = None
    
    # Benchmark base model FIRST (clean GPU state)
    if not args.skip_base:
        reset_memory()
        print("\n" + "-" * 75)
        print("BENCHMARKING BASE MODEL")
        print("-" * 75)
        
        # Measure memory before loading
        mem_before = get_memory_mb()
        print(f"  GPU memory before loading: {mem_before:.1f} MB")
        
        base_model = load_base_model(args.model_path, args.device)
        
        mem_after = get_memory_mb()
        print(f"  GPU memory after loading: {mem_after:.1f} MB")
        print(f"  Model loaded, using {mem_after - mem_before:.1f} MB")
        
        base_results = benchmark_model(base_model, input_ids, "Base (BF16)")
        
        print(f"\n  Results:")
        print(f"    Model memory: {base_results['model_memory_mb']:.1f} MB")
        print(f"    Peak memory:  {base_results['peak_memory_mb']:.1f} MB")
        print(f"    Activation:   {base_results['activation_memory_mb']:.1f} MB")
        
        # Completely free memory before next test
        del base_model
        reset_memory()
        time.sleep(2)  # Give GPU time to fully release memory
        reset_memory()
        
        mem_cleared = get_memory_mb()
        print(f"\n  Memory after unloading: {mem_cleared:.1f} MB")
    
    # Benchmark quantized model
    if not args.skip_quantized:
        reset_memory()
        print("\n" + "-" * 75)
        print("BENCHMARKING QUANTIZED MODEL")
        print("-" * 75)
        
        # Measure memory before loading
        mem_before = get_memory_mb()
        print(f"  GPU memory before loading: {mem_before:.1f} MB")
        
        quant_model = load_quantized_model(
            args.model_path,
            args.device,
            alpha=args.alpha,
            w_bits=args.w_bits,
            a_bits=args.a_bits,
            calibration_samples=args.calibration_samples,
            scales_path=args.scales_path,
        )
        
        # Clear any calibration overhead before measuring
        gc.collect()
        torch.cuda.empty_cache()
        
        mem_after = get_memory_mb()
        print(f"  GPU memory after loading: {mem_after:.1f} MB")
        
        quant_results = benchmark_model(quant_model, input_ids, f"Quantized (W{args.w_bits}A{args.a_bits})")
        
        print(f"\n  Results:")
        print(f"    Model memory: {quant_results['model_memory_mb']:.1f} MB")
        print(f"    Peak memory:  {quant_results['peak_memory_mb']:.1f} MB")
        print(f"    Activation:   {quant_results['activation_memory_mb']:.1f} MB")
        
        # Free memory
        del quant_model
        reset_memory()
    
    # Print comparison
    if base_results and quant_results:
        print_results(base_results, quant_results)
    elif base_results:
        print("\nBASE MODEL RESULTS:")
        print_single_result(base_results)
    elif quant_results:
        print("\nQUANTIZED MODEL RESULTS:")
        print_single_result(quant_results)


if __name__ == "__main__":
    main()
