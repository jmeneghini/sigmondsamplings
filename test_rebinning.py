#!/usr/bin/env python3
"""
Test script to verify rebinning algorithm correctness.
"""

import numpy as np
from sigmondsamplings.utils import rebin_data

def test_rebin_basic():
    """Test basic rebinning with perfect divisibility."""
    print("Test 1: Basic rebinning (rebin_size=2)")
    bins = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    rebinned = rebin_data(bins, 2)
    expected = np.array([1.5, 3.5, 5.5])

    print(f"  Input:    {bins}")
    print(f"  Output:   {rebinned}")
    print(f"  Expected: {expected}")
    assert np.allclose(rebinned, expected), f"Failed: {rebinned} != {expected}"
    print("  ✓ PASSED\n")


def test_rebin_size_3():
    """Test rebinning with size 3."""
    print("Test 2: Rebinning with rebin_size=3")
    bins = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    rebinned = rebin_data(bins, 3)
    expected = np.array([2.0, 5.0])  # (1+2+3)/3=2, (4+5+6)/3=5

    print(f"  Input:    {bins}")
    print(f"  Output:   {rebinned}")
    print(f"  Expected: {expected}")
    assert np.allclose(rebinned, expected), f"Failed: {rebinned} != {expected}"
    print("  ✓ PASSED\n")


def test_rebin_with_remainder():
    """Test rebinning with remainder bins (should be dropped)."""
    print("Test 3: Rebinning with remainder bins")
    bins = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    rebinned = rebin_data(bins, 3)
    expected = np.array([2.0, 5.0])  # Bin 7 is dropped

    print(f"  Input:    {bins} (7 bins)")
    print(f"  Output:   {rebinned} ({len(rebinned)} bins)")
    print(f"  Expected: {expected}")
    print(f"  Note: Bin 7 was dropped (7 % 3 = 1 remainder)")
    assert np.allclose(rebinned, expected), f"Failed: {rebinned} != {expected}"
    assert len(rebinned) == 2, f"Failed: length {len(rebinned)} != 2"
    print("  ✓ PASSED\n")


def test_rebin_size_1():
    """Test rebinning with size 1 (no rebinning)."""
    print("Test 4: Rebinning with rebin_size=1 (identity)")
    bins = np.array([1.0, 2.0, 3.0, 4.0])
    rebinned = rebin_data(bins, 1)

    print(f"  Input:    {bins}")
    print(f"  Output:   {rebinned}")
    assert np.allclose(rebinned, bins), f"Failed: {rebinned} != {bins}"
    print("  ✓ PASSED\n")


def test_autocorrelation_reduction():
    """
    Test that rebinning reduces autocorrelation.

    This is the main purpose of rebinning in lattice QCD.
    """
    print("Test 5: Autocorrelation reduction")

    # Generate autocorrelated data
    np.random.seed(42)
    n = 1000
    tau_int = 5.0  # Integrated autocorrelation time

    # AR(1) process with correlation
    alpha = np.exp(-1.0 / tau_int)
    data = np.zeros(n)
    data[0] = np.random.randn()
    for i in range(1, n):
        data[i] = alpha * data[i-1] + np.sqrt(1 - alpha**2) * np.random.randn()

    # Compute autocorrelation before rebinning
    from sigmondsamplings.utils import integrated_autocorrelation_time
    tau_before = integrated_autocorrelation_time(data)

    # Rebin by factor of 3
    rebinned = rebin_data(data, 3)
    tau_after = integrated_autocorrelation_time(rebinned)

    print(f"  Original: {n} bins, τ_int = {tau_before:.2f}")
    print(f"  Rebinned: {len(rebinned)} bins, τ_int = {tau_after:.2f}")
    print(f"  Reduction factor: {tau_before / tau_after:.2f}")

    # After rebinning, autocorrelation should be reduced
    # (though not necessarily by exactly the rebin factor)
    assert tau_after < tau_before, f"Failed: τ_after ({tau_after}) >= τ_before ({tau_before})"
    print("  ✓ PASSED (autocorrelation reduced)\n")


def test_variance_preservation():
    """
    Test that rebinning preserves the mean but increases error bars appropriately.

    Variance of rebinned mean should be approximately:
    var(rebinned_mean) ≈ var(original_mean) / (rebin_size / (2 * tau_int))

    But for this test, we just check that the mean is preserved.
    """
    print("Test 6: Mean preservation")

    np.random.seed(123)
    bins = np.random.randn(1000) + 5.0  # Mean ≈ 5.0

    mean_before = np.mean(bins)

    # Rebin by various factors
    for rebin_size in [2, 5, 10]:
        rebinned = rebin_data(bins, rebin_size)
        mean_after = np.mean(rebinned)

        print(f"  rebin_size={rebin_size:2d}: mean before={mean_before:.6f}, after={mean_after:.6f}, diff={abs(mean_before - mean_after):.6e}")

        # Mean should be approximately the same (within statistical fluctuations)
        assert abs(mean_before - mean_after) < 0.1, f"Mean changed too much: {mean_before} -> {mean_after}"

    print("  ✓ PASSED\n")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Test 7: Edge cases")

    # Test rebin_size = 0 (should raise error)
    try:
        bins = np.array([1, 2, 3])
        rebin_data(bins, 0)
        assert False, "Should have raised ValueError for rebin_size=0"
    except ValueError as e:
        print(f"  ✓ Correctly raised error for rebin_size=0: {e}")

    # Test rebin_size > n_bins (should raise error)
    try:
        bins = np.array([1, 2, 3])
        rebin_data(bins, 10)
        assert False, "Should have raised ValueError for rebin_size > n_bins"
    except ValueError as e:
        print(f"  ✓ Correctly raised error for rebin_size > n_bins: {e}")

    # Test negative rebin_size
    try:
        bins = np.array([1, 2, 3])
        rebin_data(bins, -1)
        assert False, "Should have raised ValueError for negative rebin_size"
    except ValueError as e:
        print(f"  ✓ Correctly raised error for negative rebin_size: {e}")

    print("  ✓ PASSED\n")


def test_complex_data():
    """Test rebinning with complex-valued data (for complex observables)."""
    print("Test 8: Complex data rebinning")

    bins = np.array([1+1j, 2+2j, 3+3j, 4+4j])
    rebinned = rebin_data(bins, 2)
    expected = np.array([1.5+1.5j, 3.5+3.5j])

    print(f"  Input:    {bins}")
    print(f"  Output:   {rebinned}")
    print(f"  Expected: {expected}")
    assert np.allclose(rebinned, expected), f"Failed: {rebinned} != {expected}"
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print("="*60)
    print("REBINNING ALGORITHM VERIFICATION")
    print("="*60 + "\n")

    try:
        test_rebin_basic()
        test_rebin_size_3()
        test_rebin_with_remainder()
        test_rebin_size_1()
        test_autocorrelation_reduction()
        test_variance_preservation()
        test_edge_cases()
        test_complex_data()

        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nConclusion: The rebinning algorithm is CORRECT.")
        print("\nKey properties verified:")
        print("  ✓ Correctly averages consecutive bins")
        print("  ✓ Drops remainder bins (standard practice)")
        print("  ✓ Preserves mean value")
        print("  ✓ Reduces autocorrelation")
        print("  ✓ Handles edge cases properly")
        print("  ✓ Works with complex data")

    except AssertionError as e:
        print("\n" + "="*60)
        print("TEST FAILED ✗")
        print("="*60)
        print(f"\nError: {e}")
        import sys
        sys.exit(1)
