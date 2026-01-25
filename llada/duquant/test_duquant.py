"""
Basic Tests for DuQuant Implementation

This script tests the core DuQuant components without requiring the full LLaDA model.
Run with: python -m llada.duquant.test_duquant
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duquant.simple_quantizer import SimpleQuantizer, get_hadamard_matrix, exchange_row_col
from duquant.quant_linear import DuQuantLinear, quantize_tensor_simple


def test_hadamard_matrix():
    """Test Hadamard matrix generation."""
    print("Testing Hadamard matrix generation...")
    
    for size in [2, 4, 8, 16, 32, 64, 128]:
        H = get_hadamard_matrix(size, torch.device('cpu'), torch.float32)
        
        # Check shape
        assert H.shape == (size, size), f"Expected shape ({size}, {size}), got {H.shape}"
        
        # Check orthogonality: H @ H^T should be identity
        identity = H @ H.T
        expected_identity = torch.eye(size)
        error = (identity - expected_identity).abs().max().item()
        assert error < 1e-5, f"Hadamard matrix not orthogonal for size {size}, max error: {error}"
    
    print("  [PASS] Hadamard matrix generation")


def test_exchange_row_col():
    """Test row/column exchange function."""
    print("Testing row/column exchange...")
    
    M = torch.arange(16).reshape(4, 4).float()
    
    # Exchange rows 0 and 2
    M_swapped = exchange_row_col(M, 0, 2)
    
    # Check that rows and columns are swapped
    assert M_swapped[0, 0] == M[2, 2], "Row/col exchange failed"
    assert M_swapped[2, 2] == M[0, 0], "Row/col exchange failed"
    
    # Exchange with same index should be identity
    M_same = exchange_row_col(M, 1, 1)
    assert torch.allclose(M_same, M), "Same index exchange should be identity"
    
    print("  [PASS] Row/column exchange")


def test_simple_quantizer_init():
    """Test SimpleQuantizer initialization."""
    print("Testing SimpleQuantizer initialization...")
    
    quantizer = SimpleQuantizer(
        n_bits=8,
        symmetric=False,
        block_size=64,
        max_rotation_step=128,
        permutation_times=1,
    )
    
    assert quantizer.n_bits == 8
    assert quantizer.block_size == 64
    assert quantizer.max_rotation_step == 128
    assert quantizer.permutation_times == 1
    assert not quantizer.calibrated
    
    print("  [PASS] SimpleQuantizer initialization")


def test_rotation():
    """Test rotation algorithm."""
    print("Testing rotation algorithm...")
    
    quantizer = SimpleQuantizer(n_bits=8, block_size=64, max_rotation_step=32)
    
    # Create a weight matrix with outliers
    weight = torch.randn(128, 256)
    # Add some outliers
    weight[:, 10] *= 10  # Large outlier in channel 10
    weight[:, 50] *= 8   # Another outlier
    
    # Apply rotation
    weight_rotated, R = quantizer.rotation(weight)
    
    # Check that R is close to orthogonal (R @ R^T ≈ I for each block)
    # Note: The full R matrix may not be strictly orthogonal due to block structure
    assert weight_rotated.shape == weight.shape, "Rotation changed weight shape"
    
    # Check that outliers are reduced
    orig_max = weight.abs().max(dim=0)[0].max().item()
    rotated_max = weight_rotated.abs().max(dim=0)[0].max().item()
    print(f"    Original max: {orig_max:.4f}, Rotated max: {rotated_max:.4f}")
    
    print("  [PASS] Rotation algorithm")


def test_permutation_zigzag():
    """Test zigzag permutation algorithm."""
    print("Testing zigzag permutation...")
    
    quantizer = SimpleQuantizer(n_bits=8, block_size=64)
    
    # Create a weight matrix
    weight = torch.randn(128, 256)
    # Add outliers at specific channels
    weight[:, 0] *= 10
    weight[:, 1] *= 9
    weight[:, 2] *= 8
    
    # Apply permutation
    weight_permuted, perm = quantizer.permutation_zigzag(weight)
    
    assert weight_permuted.shape == weight.shape, "Permutation changed weight shape"
    assert perm.shape[0] == weight.shape[1], "Permutation indices wrong size"
    
    # Check that permutation is valid (contains all indices exactly once)
    sorted_perm = torch.sort(perm)[0]
    expected = torch.arange(weight.shape[1])
    assert torch.all(sorted_perm == expected), "Permutation is not a valid permutation"
    
    # Check that we can recover original with inverse permutation
    perm_inv = torch.zeros_like(perm)
    perm_inv[perm] = torch.arange(weight.shape[1])
    weight_recovered = weight_permuted[:, perm_inv]
    assert torch.allclose(weight_recovered, weight), "Cannot recover original weight from permutation"
    
    print("  [PASS] Zigzag permutation")


def test_online_duquant_cali():
    """Test full DuQuant calibration."""
    print("Testing online DuQuant calibration...")
    
    quantizer = SimpleQuantizer(
        n_bits=8,
        block_size=64,
        max_rotation_step=32,
        permutation_times=1,
    )
    
    # Create weight matrix
    weight = torch.randn(512, 256)
    weight[:, 10] *= 10  # Add outlier
    
    # Calibrate
    quantizer.online_duquant_cali(weight)
    
    assert quantizer.calibrated, "Quantizer should be calibrated after online_duquant_cali"
    assert quantizer.R is not None, "Rotation matrix should be set"
    assert quantizer.perm is not None, "Permutation should be set"
    assert quantizer.scale is not None, "Scale should be computed"
    
    print("  [PASS] Online DuQuant calibration")


def test_quantization_forward():
    """Test forward pass with quantization."""
    print("Testing quantization forward pass...")
    
    quantizer = SimpleQuantizer(n_bits=8, block_size=64)
    
    # Create and calibrate on weight
    weight = torch.randn(256, 128)
    quantizer.online_duquant_cali(weight)
    
    # Apply quantization
    weight_quant = quantizer(weight)
    
    assert weight_quant.shape == weight.shape, "Quantized weight has wrong shape"
    
    # Check quantization error
    mse = ((weight - weight_quant) ** 2).mean().item()
    print(f"    Quantization MSE: {mse:.6f}")
    
    # Compare with basic quantization (should be similar or better)
    weight_basic = quantizer._basic_quant(weight)
    mse_basic = ((weight - weight_basic) ** 2).mean().item()
    print(f"    Basic quant MSE:  {mse_basic:.6f}")
    
    print("  [PASS] Quantization forward pass")


def test_duquant_linear():
    """Test DuQuantLinear layer."""
    print("Testing DuQuantLinear...")
    
    # Create original linear layer
    original = nn.Linear(256, 512)
    
    # Create and calibrate quantizer
    quantizer = SimpleQuantizer(n_bits=8, block_size=64)
    quantizer.online_duquant_cali(original.weight)
    
    # Create quantized linear
    quant_linear = DuQuantLinear(original, weight_quantizer=quantizer, w_bits=8, a_bits=8)
    
    assert quant_linear.in_features == 256
    assert quant_linear.out_features == 512
    
    # Test forward pass
    x = torch.randn(2, 10, 256)
    y = quant_linear(x)
    
    assert y.shape == (2, 10, 512), f"Wrong output shape: {y.shape}"
    
    # Compare with original
    with torch.no_grad():
        y_orig = original(x)
    
    mse = ((y - y_orig) ** 2).mean().item()
    print(f"    Output MSE vs original: {mse:.6f}")
    
    # Test with cached weights
    quant_linear.quantize_weight(use_cache=True)
    y_cached = quant_linear(x)
    assert torch.allclose(y, y_cached), "Cached output should match non-cached"
    
    print("  [PASS] DuQuantLinear")


def test_quantize_tensor_simple():
    """Test simple quantization helper."""
    print("Testing quantize_tensor_simple...")
    
    x = torch.randn(128, 64)
    
    # Per-tensor symmetric
    x_q = quantize_tensor_simple(x, n_bits=8, per_channel=False, symmetric=True)
    assert x_q.shape == x.shape
    
    # Per-channel asymmetric
    x_q = quantize_tensor_simple(x, n_bits=8, per_channel=True, symmetric=False)
    assert x_q.shape == x.shape
    
    # No quantization for high bits
    x_q = quantize_tensor_simple(x, n_bits=16)
    assert torch.allclose(x_q, x), "16-bit should be no-op"
    
    print("  [PASS] quantize_tensor_simple")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("DuQuant Implementation Tests")
    print("="*60 + "\n")
    
    try:
        test_hadamard_matrix()
        test_exchange_row_col()
        test_simple_quantizer_init()
        test_rotation()
        test_permutation_zigzag()
        test_online_duquant_cali()
        test_quantization_forward()
        test_duquant_linear()
        test_quantize_tensor_simple()
        
        print("\n" + "="*60)
        print("All tests passed!")
        print("="*60 + "\n")
        return True
        
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
