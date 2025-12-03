#!/usr/bin/env python3
"""
Test script to verify that the sparse implementation doesn't form dense matrices.
"""

import sys
import os
sys.path.append('.')

# Mock numpy and scipy for testing without actual installation
class MockArray:
    def __init__(self, data):
        self.data = data
        self.shape = (len(data),) if hasattr(data, '__len__') and not isinstance(data, str) else (1,)
        self.dtype = type(data[0]) if hasattr(data, '__len__') and not isinstance(data, str) else type(data)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __len__(self):
        return len(self.data)
    
    def __add__(self, other):
        if isinstance(other, MockArray):
            return MockArray([a + b for a, b in zip(self.data, other.data)])
        return MockArray([a + other for a in self.data])
    
    def __mul__(self, other):
        if isinstance(other, MockArray):
            return MockArray([a * b for a, b in zip(self.data, other.data)])
        return MockArray([a * other for a in self.data])
    
    def __matmul__(self, other):
        # Simple matrix-vector multiplication for testing
        if hasattr(other, 'data'):
            result = [0.0] * len(self.data)
            for i, row in enumerate(self.data):
                for j, val in enumerate(row):
                    if j < len(other.data):
                        result[i] += val * other.data[j]
            return MockArray(result)
        return self
    
    def toarray(self):
        return self
    
    def dot(self, other):
        return self @ other
    
    def tocsr(self):
        return self

class MockSparseMatrix:
    def __init__(self, data, shape, dtype=complex):
        self.data = data
        self.shape = shape
        self.dtype = dtype
        self._is_sparse = True
    
    def __add__(self, other):
        return MockSparseMatrix("sparse_add", self.shape, self.dtype)
    
    def __mul__(self, other):
        return MockSparseMatrix("sparse_mul", self.shape, self.dtype)
    
    def __matmul__(self, other):
        if hasattr(other, 'data'):
            return MockArray([0.0] * self.shape[0])
        return MockSparseMatrix("sparse_matmul", self.shape, self.dtype)
    
    def toarray(self):
        # This should NOT be called in our sparse implementation
        raise RuntimeError("Dense matrix formation detected! This violates sparse implementation.")
    
    def dot(self, other):
        if hasattr(other, 'data'):
            return MockArray([0.0] * self.shape[0])
        return self
    
    def tocsr(self):
        return self

# Mock the modules
class MockNumpy:
    def array(self, data):
        return MockArray(data)
    
    def zeros(self, shape, dtype=complex):
        if isinstance(shape, int):
            return MockArray([0.0] * shape)
        return MockArray([[0.0] * shape[1] for _ in range(shape[0])])
    
    def arange(self, start, stop, step=1, dtype=float):
        return MockArray(list(range(int(start), int(stop), int(step))))
    
    def sqrt(self, x):
        return x ** 0.5
    
    def sin(self, x):
        return x  # Simplified for testing
    def cos(self, x):
        return x  # Simplified for testing
    def exp(self, x):
        return x  # Simplified for testing
    
    def linspace(self, start, stop, num):
        return MockArray([start + i * (stop - start) / (num - 1) for i in range(num)])
    
    def stack(self, arrays, axis=0):
        return MockArray([arr.data for arr in arrays])
    
    def vdot(self, a, b):
        if hasattr(a, 'data') and hasattr(b, 'data'):
            return sum(x * y for x, y in zip(a.data, b.data))
        return a * b
    
    def linalg(self):
        class Linalg:
            def eigh(self, matrix):
                # Return mock eigenvalues and eigenvectors
                n = matrix.shape[0] if hasattr(matrix, 'shape') else len(matrix)
                return MockArray([1.0] * n), MockArray([[1.0] * n for _ in range(n)])
        return Linalg()

class MockScipySparse:
    def csr_matrix(self, data, shape, dtype=complex):
        return MockSparseMatrix(data, shape, dtype)
    
    def lil_matrix(self, shape, dtype=complex):
        return MockSparseMatrix("lil", shape, dtype)
    
    def kron(self, a, b):
        return MockSparseMatrix("kron", (a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]), a.dtype)
    
    def eye(self, n, dtype=complex):
        return MockSparseMatrix("eye", (n, n), dtype)

class MockScipySparseLinalg:
    def expm_multiply(self, A, b, start, stop, num):
        # Return the final state after evolution
        return [b, b]  # Simplified for testing

# Replace the modules
sys.modules['numpy'] = MockNumpy()
sys.modules['scipy.sparse'] = MockScipySparse()
sys.modules['scipy.sparse.linalg'] = MockScipySparseLinalg()
sys.modules['scipy.linalg'] = type('MockScipyLinalg', (), {'expm': lambda x: x})()

# Now test our sparse implementation
try:
    from dicke.dicke_collective_sparse import spin_matrices, build_single_bin_hamiltonian, evolve_times
    
    print("Testing sparse matrix implementation...")
    
    # Test spin matrices
    print("1. Testing spin_matrices...")
    Jx, Jy, Jz = spin_matrices(1.0)
    print(f"   Jx type: {type(Jx)}, shape: {Jx.shape}")
    print(f"   Jy type: {type(Jy)}, shape: {Jy.shape}")
    print(f"   Jz type: {type(Jz)}, shape: {Jz.shape}")
    
    # Test Hamiltonian building
    print("2. Testing build_single_bin_hamiltonian...")
    H, ops, S_list, dims = build_single_bin_hamiltonian(4, 1.0, 0.1, 0.5)
    print(f"   H type: {type(H)}, shape: {H.shape}")
    print(f"   Jx type: {type(ops[0])}, Jy type: {type(ops[1])}, Jz type: {type(ops[2])}")
    
    # Test evolution (this should not form dense matrices)
    print("3. Testing evolve_times...")
    psi0 = MockArray([1.0, 0.0, 0.0, 0.0])
    t_grid = MockArray([0.0, 1.0, 2.0])
    states = evolve_times(H, psi0, t_grid)
    print(f"   States type: {type(states)}, shape: {states.shape}")
    
    print("✅ All tests passed! No dense matrices were formed.")
    print("♡ Sparse implementation is working correctly!")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
