# Integration Test for dicke_collective_sparse_opt_mem_patched.py

## Summary
Successfully created and validated integration test comparing the baseline (dense) implementation 
with the memory-patched sparse implementation using LinearOperators.

## Files Created/Modified

### New File
- `test_integration_singlebin_mem_patched.py`: Integration test for single-bin Dicke evolution

### Modified Files  
- `dicke_collective_sparse_opt_mem_patched.py`: Fixed to support LinearOperators in:
  - `evolve_times()`: Handle LinearOperator type without trying to convert to csr_matrix
  - `bin_observables()`: Use LinearOperator.matvec() method instead of dot/matmul

## Key Fixes

1. **LinearOperator Support in evolve_times()**
   - Added type checking to handle LinearOperator, csr_matrix, and dense matrices
   - expm_multiply works directly with LinearOperators

2. **LinearOperator Support in bin_observables()**
   - Added isinstance check for LinearOperator
   - Use `.matvec()` method (expects 1D vectors) instead of `.dot()` or `@`
   - Fixed bug where tuple unpacking created nested list structure

3. **Numerical Precision**
   - complex64 has significant precision issues with LinearOperators
   - complex128 achieves machine precision (errors ~1e-13 to 1e-15)
   - Test defaults to complex128 for reliability

## Test Results

All 48 test cases pass with complex128:
- Sweeps N ∈ {2, 10, 50}
- Sweeps μ ∈ {0.0, 0.05, 0.5}  
- Sweeps θ_v ∈ {π/4, π/2-0.2}
- Sweeps polarizations: (N,0), (N/2,N/2), (1,N-1)

Maximum errors: ~1e-12 (well within tolerance of 2e-5)

## Usage

```bash
# Run full suite (default: complex128)
python test_integration_singlebin_mem_patched.py

# Quick smoke test
python test_integration_singlebin_mem_patched.py --max-cases 3

# Use complex64 (not recommended - precision issues)
python test_integration_singlebin_mem_patched.py --dtype complex64
```

## LinearOperator Mode

The mem_patched version uses LinearOperators by default (controlled by 
`DICKE_USE_LINEAR_OPERATOR` environment variable). This provides:
- Reduced memory footprint
- No need to materialize full Hamiltonian matrix
- Compatible with expm_multiply for time evolution

The integration test validates that LinearOperator mode produces numerically 
identical results to the baseline dense implementation.
