#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark comparison: LinearOperator vs Sparse Matrix implementations
====================================================================

This script compares the performance (speed and memory) of:
- dicke_collective_sparse_opt_linop.py (LinearOperator, matrix-free)
- dicke_collective_sparse_opt.py (sparse matrices)

Tests are run multiple times and averaged for statistical reliability.
"""

import numpy as np
import time
import tracemalloc
import sys
from typing import Dict, Tuple, List, Optional
import gc
from datetime import datetime

# Import both implementations
import dicke_collective_sparse_opt as sparse_opt
import dicke_collective_sparse_opt_linop as linop_opt

# Import helper functions for initial states
from dicke_collective_sparse_opt import (
    single_bin_initial_state,
    multi_bin_initial_state
)


def get_memory_usage():
    """Get current memory usage in MB using tracemalloc."""
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        return current / (1024 * 1024)  # MB
    return 0.0


def benchmark_single_bin(
    N: int,
    omega: float,
    theta_v: float,
    mu: float,
    t_max: float,
    num_times: int,
    num_runs: int = 5
) -> Dict:
    """
    Benchmark single-bin evolution for both implementations.
    
    Returns dict with timing and memory stats for both implementations.
    """
    print(f"\n{'='*70}")
    print(f"Single-bin benchmark: N={N}, t_max={t_max}, num_times={num_times}")
    print(f"{'='*70}")
    
    # Prepare initial state
    n1, n2 = N // 2, N - N // 2
    psi0_sparse, S = single_bin_initial_state(n1, n2)
    if hasattr(psi0_sparse, 'toarray'):
        psi0 = psi0_sparse.toarray().flatten()
    else:
        psi0 = np.asarray(psi0_sparse).flatten()
    
    t_grid = np.linspace(0.0, t_max, num_times)
    
    results = {
        'sparse': {'times': [], 'memory_peak': [], 'memory_build': []},
        'linop': {'times': [], 'memory_peak': [], 'memory_build': []}
    }
    
    # Benchmark sparse matrix implementation
    print("\n[SPARSE] Running benchmarks...")
    for run in range(num_runs):
        gc.collect()
        tracemalloc.start()
        mem_before = get_memory_usage()
        
        # Build Hamiltonian
        t0 = time.perf_counter()
        H_sparse, (Jx, Jy, Jz), S_list, dims = sparse_opt.build_single_bin_hamiltonian(
            N, omega, theta_v, mu
        )
        t_build = time.perf_counter() - t0
        mem_after_build = get_memory_usage()
        
        # Evolve
        t0 = time.perf_counter()
        states = sparse_opt.evolve_times(H_sparse, psi0, t_grid)
        t_evolve = time.perf_counter() - t0
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_time = t_build + t_evolve
        results['sparse']['times'].append(total_time)
        results['sparse']['memory_peak'].append(peak / (1024 * 1024))  # MB
        results['sparse']['memory_build'].append(mem_after_build - mem_before)
        
        # Cleanup
        del H_sparse, states
        gc.collect()
    
    # Benchmark LinearOperator implementation
    print("[LINOP] Running benchmarks...")
    for run in range(num_runs):
        gc.collect()
        tracemalloc.start()
        mem_before = get_memory_usage()
        
        # Build Hamiltonian
        t0 = time.perf_counter()
        H_linop, (Jx, Jy, Jz), S_list, dims = linop_opt.build_single_bin_hamiltonian(
            N, omega, theta_v, mu
        )
        t_build = time.perf_counter() - t0
        mem_after_build = get_memory_usage()
        
        # Evolve
        t0 = time.perf_counter()
        states = linop_opt.evolve_times(H_linop, psi0, t_grid)
        t_evolve = time.perf_counter() - t0
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_time = t_build + t_evolve
        results['linop']['times'].append(total_time)
        results['linop']['memory_peak'].append(peak / (1024 * 1024))  # MB
        results['linop']['memory_build'].append(mem_after_build - mem_before)
        
        # Cleanup
        del H_linop, states
        gc.collect()
    
    # Compute statistics
    stats = {}
    for impl in ['sparse', 'linop']:
        times = np.array(results[impl]['times'])
        mem_peak = np.array(results[impl]['memory_peak'])
        mem_build = np.array(results[impl]['memory_build'])
        
        stats[impl] = {
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'time_min': np.min(times),
            'time_max': np.max(times),
            'memory_peak_mean': np.mean(mem_peak),
            'memory_peak_std': np.std(mem_peak),
            'memory_build_mean': np.mean(mem_build),
            'memory_build_std': np.std(mem_build),
        }
    
    # Print results
    print(f"\n{'Results (averaged over %d runs):' % num_runs}")
    print(f"{'-'*70}")
    print(f"{'Metric':<30} {'Sparse':<20} {'LinearOp':<20}")
    print(f"{'-'*70}")
    print(f"{'Time (s) - mean':<30} {stats['sparse']['time_mean']:<20.4f} {stats['linop']['time_mean']:<20.4f}")
    print(f"{'Time (s) - std':<30} {stats['sparse']['time_std']:<20.4f} {stats['linop']['time_std']:<20.4f}")
    print(f"{'Memory peak (MB) - mean':<30} {stats['sparse']['memory_peak_mean']:<20.2f} {stats['linop']['memory_peak_mean']:<20.2f}")
    print(f"{'Memory build (MB) - mean':<30} {stats['sparse']['memory_build_mean']:<20.2f} {stats['linop']['memory_build_mean']:<20.2f}")
    
    speedup = stats['sparse']['time_mean'] / stats['linop']['time_mean']
    mem_ratio = stats['linop']['memory_peak_mean'] / stats['sparse']['memory_peak_mean'] if stats['sparse']['memory_peak_mean'] > 0 else 0
    
    print(f"\n{'Speedup (sparse/linop):':<30} {speedup:.3f}x")
    print(f"{'Memory ratio (linop/sparse):':<30} {mem_ratio:.3f}x")
    
    return stats


def benchmark_multi_bin(
    N_list: List[int],
    omega_list: List[float],
    theta_v: float,
    mu: float,
    t_max: float,
    num_times: int,
    num_runs: int = 5
) -> Dict:
    """
    Benchmark multi-bin evolution for both implementations.
    
    Returns dict with timing and memory stats for both implementations.
    """
    print(f"\n{'='*70}")
    print(f"Multi-bin benchmark: N_list={N_list}, t_max={t_max}, num_times={num_times}")
    print(f"{'='*70}")
    
    # Prepare initial state
    n1_list = [n // 2 for n in N_list]
    n2_list = [n - n1 for n, n1 in zip(N_list, n1_list)]
    psi0_sparse, S_list = multi_bin_initial_state(n1_list, n2_list)
    if hasattr(psi0_sparse, 'toarray'):
        psi0 = psi0_sparse.toarray().flatten()
    else:
        psi0 = np.asarray(psi0_sparse).flatten()
    
    t_grid = np.linspace(0.0, t_max, num_times)
    
    results = {
        'sparse': {'times': [], 'memory_peak': [], 'memory_build': []},
        'linop': {'times': [], 'memory_peak': [], 'memory_build': []}
    }
    
    # Benchmark sparse matrix implementation
    print("\n[SPARSE] Running benchmarks...")
    for run in range(num_runs):
        gc.collect()
        tracemalloc.start()
        mem_before = get_memory_usage()
        
        # Build Hamiltonian
        t0 = time.perf_counter()
        H_sparse, (Jx_list, Jy_list, Jz_list), S_list, dims = sparse_opt.build_multi_bin_hamiltonian(
            N_list, omega_list, theta_v, mu
        )
        t_build = time.perf_counter() - t0
        mem_after_build = get_memory_usage()
        
        # Evolve
        t0 = time.perf_counter()
        states = sparse_opt.evolve_times(H_sparse, psi0, t_grid)
        t_evolve = time.perf_counter() - t0
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_time = t_build + t_evolve
        results['sparse']['times'].append(total_time)
        results['sparse']['memory_peak'].append(peak / (1024 * 1024))  # MB
        results['sparse']['memory_build'].append(mem_after_build - mem_before)
        
        # Cleanup
        del H_sparse, states
        gc.collect()
    
    # Benchmark LinearOperator implementation
    print("[LINOP] Running benchmarks...")
    for run in range(num_runs):
        gc.collect()
        tracemalloc.start()
        mem_before = get_memory_usage()
        
        # Build Hamiltonian
        t0 = time.perf_counter()
        H_linop, (Jx_list, Jy_list, Jz_list), S_list, dims = linop_opt.build_multi_bin_hamiltonian(
            N_list, omega_list, theta_v, mu
        )
        t_build = time.perf_counter() - t0
        mem_after_build = get_memory_usage()
        
        # Evolve
        t0 = time.perf_counter()
        states = linop_opt.evolve_times(H_linop, psi0, t_grid)
        t_evolve = time.perf_counter() - t0
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_time = t_build + t_evolve
        results['linop']['times'].append(total_time)
        results['linop']['memory_peak'].append(peak / (1024 * 1024))  # MB
        results['linop']['memory_build'].append(mem_after_build - mem_before)
        
        # Cleanup
        del H_linop, states
        gc.collect()
    
    # Compute statistics
    stats = {}
    for impl in ['sparse', 'linop']:
        times = np.array(results[impl]['times'])
        mem_peak = np.array(results[impl]['memory_peak'])
        mem_build = np.array(results[impl]['memory_build'])
        
        stats[impl] = {
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'time_min': np.min(times),
            'time_max': np.max(times),
            'memory_peak_mean': np.mean(mem_peak),
            'memory_peak_std': np.std(mem_peak),
            'memory_build_mean': np.mean(mem_build),
            'memory_build_std': np.std(mem_build),
        }
    
    # Print results
    print(f"\n{'Results (averaged over %d runs):' % num_runs}")
    print(f"{'-'*70}")
    print(f"{'Metric':<30} {'Sparse':<20} {'LinearOp':<20}")
    print(f"{'-'*70}")
    print(f"{'Time (s) - mean':<30} {stats['sparse']['time_mean']:<20.4f} {stats['linop']['time_mean']:<20.4f}")
    print(f"{'Time (s) - std':<30} {stats['sparse']['time_std']:<20.4f} {stats['linop']['time_std']:<20.4f}")
    print(f"{'Memory peak (MB) - mean':<30} {stats['sparse']['memory_peak_mean']:<20.2f} {stats['linop']['memory_peak_mean']:<20.2f}")
    print(f"{'Memory build (MB) - mean':<30} {stats['sparse']['memory_build_mean']:<20.2f} {stats['linop']['memory_build_mean']:<20.2f}")
    
    speedup = stats['sparse']['time_mean'] / stats['linop']['time_mean']
    mem_ratio = stats['linop']['memory_peak_mean'] / stats['sparse']['memory_peak_mean'] if stats['sparse']['memory_peak_mean'] > 0 else 0
    
    print(f"\n{'Speedup (sparse/linop):':<30} {speedup:.3f}x")
    print(f"{'Memory ratio (linop/sparse):':<30} {mem_ratio:.3f}x")
    
    return stats


def benchmark_single_bin_stream(
    N: int,
    omega: float,
    theta_v: float,
    mu: float,
    t_max: float,
    num_times: int,
    num_runs: int = 5,
    chunk: int = 64
) -> Dict:
    """
    Benchmark single-bin evolution using stream interface for both implementations.
    
    Returns dict with timing and memory stats for both implementations.
    """
    print(f"\n{'='*70}")
    print(f"Single-bin STREAM benchmark: N={N}, t_max={t_max}, num_times={num_times}, chunk={chunk}")
    print(f"{'='*70}")
    
    # Prepare initial state
    n1, n2 = N // 2, N - N // 2
    psi0_sparse, S = single_bin_initial_state(n1, n2)
    if hasattr(psi0_sparse, 'toarray'):
        psi0 = psi0_sparse.toarray().flatten()
    else:
        psi0 = np.asarray(psi0_sparse).flatten()
    
    t_grid = np.linspace(0.0, t_max, num_times)
    
    results = {
        'sparse': {'times': [], 'memory_peak': [], 'memory_build': []},
        'linop': {'times': [], 'memory_peak': [], 'memory_build': []}
    }
    
    # Benchmark sparse matrix implementation
    print("\n[SPARSE STREAM] Running benchmarks...")
    for run in range(num_runs):
        gc.collect()
        tracemalloc.start()
        mem_before = get_memory_usage()
        
        # Build Hamiltonian
        t0 = time.perf_counter()
        H_sparse, (Jx, Jy, Jz), S_list, dims = sparse_opt.build_single_bin_hamiltonian(
            N, omega, theta_v, mu
        )
        t_build = time.perf_counter() - t0
        mem_after_build = get_memory_usage()
        
        # Evolve using stream
        t0 = time.perf_counter()
        stream = sparse_opt.evolve_times_stream(H_sparse, psi0, t_grid, chunk=chunk)
        # Consume the stream
        count = 0
        for t, psi in stream:
            count += 1
        t_evolve = time.perf_counter() - t0
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_time = t_build + t_evolve
        results['sparse']['times'].append(total_time)
        results['sparse']['memory_peak'].append(peak / (1024 * 1024))  # MB
        results['sparse']['memory_build'].append(mem_after_build - mem_before)
        
        # Cleanup
        del H_sparse, stream
        gc.collect()
    
    # Benchmark LinearOperator implementation
    print("[LINOP STREAM] Running benchmarks...")
    for run in range(num_runs):
        gc.collect()
        tracemalloc.start()
        mem_before = get_memory_usage()
        
        # Build Hamiltonian
        t0 = time.perf_counter()
        H_linop, (Jx, Jy, Jz), S_list, dims = linop_opt.build_single_bin_hamiltonian(
            N, omega, theta_v, mu
        )
        t_build = time.perf_counter() - t0
        mem_after_build = get_memory_usage()
        
        # Evolve using stream
        t0 = time.perf_counter()
        stream = linop_opt.evolve_times_stream(H_linop, psi0, t_grid, chunk=chunk)
        # Consume the stream
        count = 0
        for t, psi in stream:
            count += 1
        t_evolve = time.perf_counter() - t0
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_time = t_build + t_evolve
        results['linop']['times'].append(total_time)
        results['linop']['memory_peak'].append(peak / (1024 * 1024))  # MB
        results['linop']['memory_build'].append(mem_after_build - mem_before)
        
        # Cleanup
        del H_linop, stream
        gc.collect()
    
    # Compute statistics
    stats = {}
    for impl in ['sparse', 'linop']:
        times = np.array(results[impl]['times'])
        mem_peak = np.array(results[impl]['memory_peak'])
        mem_build = np.array(results[impl]['memory_build'])
        
        stats[impl] = {
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'time_min': np.min(times),
            'time_max': np.max(times),
            'memory_peak_mean': np.mean(mem_peak),
            'memory_peak_std': np.std(mem_peak),
            'memory_build_mean': np.mean(mem_build),
            'memory_build_std': np.std(mem_build),
        }
    
    # Print results
    print(f"\n{'Results (averaged over %d runs):' % num_runs}")
    print(f"{'-'*70}")
    print(f"{'Metric':<30} {'Sparse':<20} {'LinearOp':<20}")
    print(f"{'-'*70}")
    print(f"{'Time (s) - mean':<30} {stats['sparse']['time_mean']:<20.4f} {stats['linop']['time_mean']:<20.4f}")
    print(f"{'Time (s) - std':<30} {stats['sparse']['time_std']:<20.4f} {stats['linop']['time_std']:<20.4f}")
    print(f"{'Memory peak (MB) - mean':<30} {stats['sparse']['memory_peak_mean']:<20.2f} {stats['linop']['memory_peak_mean']:<20.2f}")
    print(f"{'Memory build (MB) - mean':<30} {stats['sparse']['memory_build_mean']:<20.2f} {stats['linop']['memory_build_mean']:<20.2f}")
    
    speedup = stats['sparse']['time_mean'] / stats['linop']['time_mean']
    mem_ratio = stats['linop']['memory_peak_mean'] / stats['sparse']['memory_peak_mean'] if stats['sparse']['memory_peak_mean'] > 0 else 0
    
    print(f"\n{'Speedup (sparse/linop):':<30} {speedup:.3f}x")
    print(f"{'Memory ratio (linop/sparse):':<30} {mem_ratio:.3f}x")
    
    return stats


def benchmark_multi_bin_stream(
    N_list: List[int],
    omega_list: List[float],
    theta_v: float,
    mu: float,
    t_max: float,
    num_times: int,
    num_runs: int = 5,
    chunk: int = 64
) -> Dict:
    """
    Benchmark multi-bin evolution using stream interface for both implementations.
    
    Returns dict with timing and memory stats for both implementations.
    """
    print(f"\n{'='*70}")
    print(f"Multi-bin STREAM benchmark: N_list={N_list}, t_max={t_max}, num_times={num_times}, chunk={chunk}")
    print(f"{'='*70}")
    
    # Prepare initial state
    n1_list = [n // 2 for n in N_list]
    n2_list = [n - n1 for n, n1 in zip(N_list, n1_list)]
    psi0_sparse, S_list = multi_bin_initial_state(n1_list, n2_list)
    if hasattr(psi0_sparse, 'toarray'):
        psi0 = psi0_sparse.toarray().flatten()
    else:
        psi0 = np.asarray(psi0_sparse).flatten()
    
    t_grid = np.linspace(0.0, t_max, num_times)
    
    results = {
        'sparse': {'times': [], 'memory_peak': [], 'memory_build': []},
        'linop': {'times': [], 'memory_peak': [], 'memory_build': []}
    }
    
    # Benchmark sparse matrix implementation
    print("\n[SPARSE STREAM] Running benchmarks...")
    for run in range(num_runs):
        gc.collect()
        tracemalloc.start()
        mem_before = get_memory_usage()
        
        # Build Hamiltonian
        t0 = time.perf_counter()
        H_sparse, (Jx_list, Jy_list, Jz_list), S_list, dims = sparse_opt.build_multi_bin_hamiltonian(
            N_list, omega_list, theta_v, mu
        )
        t_build = time.perf_counter() - t0
        mem_after_build = get_memory_usage()
        
        # Evolve using stream
        t0 = time.perf_counter()
        stream = sparse_opt.evolve_times_stream(H_sparse, psi0, t_grid, chunk=chunk)
        # Consume the stream
        count = 0
        for t, psi in stream:
            count += 1
        t_evolve = time.perf_counter() - t0
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_time = t_build + t_evolve
        results['sparse']['times'].append(total_time)
        results['sparse']['memory_peak'].append(peak / (1024 * 1024))  # MB
        results['sparse']['memory_build'].append(mem_after_build - mem_before)
        
        # Cleanup
        del H_sparse, stream
        gc.collect()
    
    # Benchmark LinearOperator implementation
    print("[LINOP STREAM] Running benchmarks...")
    for run in range(num_runs):
        gc.collect()
        tracemalloc.start()
        mem_before = get_memory_usage()
        
        # Build Hamiltonian
        t0 = time.perf_counter()
        H_linop, (Jx_list, Jy_list, Jz_list), S_list, dims = linop_opt.build_multi_bin_hamiltonian(
            N_list, omega_list, theta_v, mu
        )
        t_build = time.perf_counter() - t0
        mem_after_build = get_memory_usage()
        
        # Evolve using stream
        t0 = time.perf_counter()
        stream = linop_opt.evolve_times_stream(H_linop, psi0, t_grid, chunk=chunk)
        # Consume the stream
        count = 0
        for t, psi in stream:
            count += 1
        t_evolve = time.perf_counter() - t0
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_time = t_build + t_evolve
        results['linop']['times'].append(total_time)
        results['linop']['memory_peak'].append(peak / (1024 * 1024))  # MB
        results['linop']['memory_build'].append(mem_after_build - mem_before)
        
        # Cleanup
        del H_linop, stream
        gc.collect()
    
    # Compute statistics
    stats = {}
    for impl in ['sparse', 'linop']:
        times = np.array(results[impl]['times'])
        mem_peak = np.array(results[impl]['memory_peak'])
        mem_build = np.array(results[impl]['memory_build'])
        
        stats[impl] = {
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'time_min': np.min(times),
            'time_max': np.max(times),
            'memory_peak_mean': np.mean(mem_peak),
            'memory_peak_std': np.std(mem_peak),
            'memory_build_mean': np.mean(mem_build),
            'memory_build_std': np.std(mem_build),
        }
    
    # Print results
    print(f"\n{'Results (averaged over %d runs):' % num_runs}")
    print(f"{'-'*70}")
    print(f"{'Metric':<30} {'Sparse':<20} {'LinearOp':<20}")
    print(f"{'-'*70}")
    print(f"{'Time (s) - mean':<30} {stats['sparse']['time_mean']:<20.4f} {stats['linop']['time_mean']:<20.4f}")
    print(f"{'Time (s) - std':<30} {stats['sparse']['time_std']:<20.4f} {stats['linop']['time_std']:<20.4f}")
    print(f"{'Memory peak (MB) - mean':<30} {stats['sparse']['memory_peak_mean']:<20.2f} {stats['linop']['memory_peak_mean']:<20.2f}")
    print(f"{'Memory build (MB) - mean':<30} {stats['sparse']['memory_build_mean']:<20.2f} {stats['linop']['memory_build_mean']:<20.2f}")
    
    speedup = stats['sparse']['time_mean'] / stats['linop']['time_mean']
    mem_ratio = stats['linop']['memory_peak_mean'] / stats['sparse']['memory_peak_mean'] if stats['sparse']['memory_peak_mean'] > 0 else 0
    
    print(f"\n{'Speedup (sparse/linop):':<30} {speedup:.3f}x")
    print(f"{'Memory ratio (linop/sparse):':<30} {mem_ratio:.3f}x")
    
    return stats


def write_markdown_report(all_results: Dict, output_file: str = None):
    """
    Write all benchmark results to a markdown file.
    
    Parameters
    ----------
    all_results : dict
        Dictionary containing all benchmark results with keys like 'scenario_1', 'stream_scenario_1', etc.
    output_file : str, optional
        Output file path. If None, generates a timestamped filename.
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, f"benchmark_results_{timestamp}.md")
    
    with open(output_file, 'w') as f:
        # Header
        f.write("# Benchmark Results: LinearOperator vs Sparse Matrix Implementations\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This report compares the performance of two implementations:\n")
        f.write("- `dicke_collective_sparse_opt.py` (Sparse matrices)\n")
        f.write("- `dicke_collective_sparse_opt_linop.py` (LinearOperator, matrix-free)\n\n")
        f.write("---\n\n")
        
        # Regular benchmarks
        f.write("## Regular Benchmarks (evolve_times)\n\n")
        for i in range(1, 9):
            key = f'scenario_{i}'
            if key in all_results and all_results[key] is not None:
                stats = all_results[key]
                f.write(f"### Scenario {i}\n\n")
                f.write("| Metric | Sparse | LinearOp | Ratio (Sparse/Linop) |\n")
                f.write("|--------|--------|----------|----------------------|\n")
                
                time_sparse = stats['sparse']['time_mean']
                time_linop = stats['linop']['time_mean']
                time_ratio = time_sparse / time_linop if time_linop > 0 else 0
                f.write(f"| Time (s) | {time_sparse:.4f} ± {stats['sparse']['time_std']:.4f} | "
                       f"{time_linop:.4f} ± {stats['linop']['time_std']:.4f} | {time_ratio:.3f}x |\n")
                
                mem_sparse = stats['sparse']['memory_peak_mean']
                mem_linop = stats['linop']['memory_peak_mean']
                mem_ratio = mem_linop / mem_sparse if mem_sparse > 0 else 0
                f.write(f"| Memory Peak (MB) | {mem_sparse:.2f} ± {stats['sparse']['memory_peak_std']:.2f} | "
                       f"{mem_linop:.2f} ± {stats['linop']['memory_peak_std']:.2f} | {mem_ratio:.3f}x |\n")
                
                mem_build_sparse = stats['sparse']['memory_build_mean']
                mem_build_linop = stats['linop']['memory_build_mean']
                f.write(f"| Memory Build (MB) | {mem_build_sparse:.2f} ± {stats['sparse']['memory_build_std']:.2f} | "
                       f"{mem_build_linop:.2f} ± {stats['linop']['memory_build_std']:.2f} | - |\n")
                f.write("\n")
            else:
                f.write(f"### Scenario {i}\n\n")
                f.write("*Skipped due to memory or time constraints.*\n\n")
        
        # Stream benchmarks
        f.write("## Stream Interface Benchmarks (evolve_times_stream)\n\n")
        for i in range(1, 9):
            key = f'stream_scenario_{i}'
            if key in all_results and all_results[key] is not None:
                stats = all_results[key]
                f.write(f"### Stream Scenario {i}\n\n")
                f.write("| Metric | Sparse | LinearOp | Ratio (Sparse/Linop) |\n")
                f.write("|--------|--------|----------|----------------------|\n")
                
                time_sparse = stats['sparse']['time_mean']
                time_linop = stats['linop']['time_mean']
                time_ratio = time_sparse / time_linop if time_linop > 0 else 0
                f.write(f"| Time (s) | {time_sparse:.4f} ± {stats['sparse']['time_std']:.4f} | "
                       f"{time_linop:.4f} ± {stats['linop']['time_std']:.4f} | {time_ratio:.3f}x |\n")
                
                mem_sparse = stats['sparse']['memory_peak_mean']
                mem_linop = stats['linop']['memory_peak_mean']
                mem_ratio = mem_linop / mem_sparse if mem_sparse > 0 else 0
                f.write(f"| Memory Peak (MB) | {mem_sparse:.2f} ± {stats['sparse']['memory_peak_std']:.2f} | "
                       f"{mem_linop:.2f} ± {stats['linop']['memory_peak_std']:.2f} | {mem_ratio:.3f}x |\n")
                
                mem_build_sparse = stats['sparse']['memory_build_mean']
                mem_build_linop = stats['linop']['memory_build_mean']
                f.write(f"| Memory Build (MB) | {mem_build_sparse:.2f} ± {stats['sparse']['memory_build_std']:.2f} | "
                       f"{mem_build_linop:.2f} ± {stats['linop']['memory_build_std']:.2f} | - |\n")
                f.write("\n")
            else:
                f.write(f"### Stream Scenario {i}\n\n")
                f.write("*Skipped due to memory or time constraints.*\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write("### Key Observations\n\n")
        f.write("- **LinearOperator** should use less memory (no explicit H matrix storage)\n")
        f.write("- **Sparse matrices** may be faster for small systems (better cache locality)\n")
        f.write("- **LinearOperator** should scale better for large systems\n")
        f.write("- **Stream interface** should use less memory (doesn't store all states)\n")
        f.write("- **Stream interface** may be slightly slower due to generator overhead\n\n")
        
        # Performance comparison table
        f.write("### Performance Comparison Summary\n\n")
        f.write("| Scenario | Type | Sparse Time (s) | Linop Time (s) | Speedup | Sparse Mem (MB) | Linop Mem (MB) | Mem Ratio |\n")
        f.write("|----------|------|------------------|----------------|---------|-----------------|---------------|----------|\n")
        
        for i in range(1, 9):
            key = f'scenario_{i}'
            if key in all_results and all_results[key] is not None:
                stats = all_results[key]
                time_sparse = stats['sparse']['time_mean']
                time_linop = stats['linop']['time_mean']
                speedup = time_sparse / time_linop if time_linop > 0 else 0
                mem_sparse = stats['sparse']['memory_peak_mean']
                mem_linop = stats['linop']['memory_peak_mean']
                mem_ratio = mem_linop / mem_sparse if mem_sparse > 0 else 0
                f.write(f"| {i} | Regular | {time_sparse:.4f} | {time_linop:.4f} | {speedup:.3f}x | "
                       f"{mem_sparse:.2f} | {mem_linop:.2f} | {mem_ratio:.3f}x |\n")
            
            key = f'stream_scenario_{i}'
            if key in all_results and all_results[key] is not None:
                stats = all_results[key]
                time_sparse = stats['sparse']['time_mean']
                time_linop = stats['linop']['time_mean']
                speedup = time_sparse / time_linop if time_linop > 0 else 0
                mem_sparse = stats['sparse']['memory_peak_mean']
                mem_linop = stats['linop']['memory_peak_mean']
                mem_ratio = mem_linop / mem_sparse if mem_sparse > 0 else 0
                f.write(f"| {i} | Stream | {time_sparse:.4f} | {time_linop:.4f} | {speedup:.3f}x | "
                       f"{mem_sparse:.2f} | {mem_linop:.2f} | {mem_ratio:.3f}x |\n")
    
    print(f"\n{'='*70}")
    print(f"Results written to: {output_file}")
    print(f"{'='*70}")


def main():
    """Run comprehensive benchmarks."""
    print("="*70)
    print("Benchmark: LinearOperator vs Sparse Matrix Implementations")
    print("="*70)
    
    # Test parameters
    num_runs = 5  # Number of runs to average over
    
    # Dictionary to collect all results
    all_results = {}
    
    # Scenario 1: Small single-bin system
    print("\n" + "="*70)
    print("SCENARIO 1: Small single-bin system")
    print("="*70)
    stats1 = benchmark_single_bin(
        N=20,
        omega=1.0,
        theta_v=0.15,
        mu=0.5,
        t_max=10.0,
        num_times=100,
        num_runs=num_runs
    )
    all_results['scenario_1'] = stats1
    
    # Scenario 2: Medium single-bin system
    print("\n" + "="*70)
    print("SCENARIO 2: Medium single-bin system")
    print("="*70)
    stats2 = benchmark_single_bin(
        N=50,
        omega=1.0,
        theta_v=0.15,
        mu=0.5,
        t_max=10.0,
        num_times=100,
        num_runs=num_runs
    )
    all_results['scenario_2'] = stats2
    
    # Scenario 3: Small multi-bin system (2 bins)
    print("\n" + "="*70)
    print("SCENARIO 3: Small multi-bin system (2 bins)")
    print("="*70)
    stats3 = benchmark_multi_bin(
        N_list=[10, 10],
        omega_list=[1.0, 1.1],
        theta_v=0.15,
        mu=0.5,
        t_max=10.0,
        num_times=100,
        num_runs=num_runs
    )
    all_results['scenario_3'] = stats3
    
    # Scenario 4: Medium multi-bin system (2 bins, larger)
    print("\n" + "="*70)
    print("SCENARIO 4: Medium multi-bin system (2 bins, larger)")
    print("="*70)
    stats4 = benchmark_multi_bin(
        N_list=[20, 20],
        omega_list=[1.0, 1.1],
        theta_v=0.15,
        mu=0.5,
        t_max=10.0,
        num_times=100,
        num_runs=num_runs
    )
    all_results['scenario_4'] = stats4
    
    # Scenario 5: Large single-bin system (50 of each type)
    print("\n" + "="*70)
    print("SCENARIO 5: Large single-bin system (N=100, 50 of each type)")
    print("="*70)
    try:
        stats5 = benchmark_single_bin(
            N=100,
            omega=1.0,
            theta_v=0.15,
            mu=0.5,
            t_max=10.0,
            num_times=100,
            num_runs=num_runs
        )
        all_results['scenario_5'] = stats5
    except Exception as e:
        print(f"ERROR in Scenario 5: {e}")
        print("Skipping this scenario due to memory or time constraints.")
        all_results['scenario_5'] = None
    
    # Scenario 6: Very large single-bin system (100 of each type)
    print("\n" + "="*70)
    print("SCENARIO 6: Very large single-bin system (N=200, 100 of each type)")
    print("="*70)
    try:
        stats6 = benchmark_single_bin(
            N=200,
            omega=1.0,
            theta_v=0.15,
            mu=0.5,
            t_max=10.0,
            num_times=100,
            num_runs=num_runs
        )
        all_results['scenario_6'] = stats6
    except Exception as e:
        print(f"ERROR in Scenario 6: {e}")
        print("Skipping this scenario due to memory or time constraints.")
        all_results['scenario_6'] = None
    
    # Scenario 7: Large multi-bin system (2 bins, 50 of each type per bin)
    print("\n" + "="*70)
    print("SCENARIO 7: Large multi-bin system (2 bins, N=100 each, 50 of each type per bin)")
    print("="*70)
    try:
        stats7 = benchmark_multi_bin(
            N_list=[100, 100],
            omega_list=[1.0, 1.1],
            theta_v=0.15,
            mu=0.5,
            t_max=10.0,
            num_times=100,
            num_runs=num_runs
        )
        all_results['scenario_7'] = stats7
    except Exception as e:
        print(f"ERROR in Scenario 7: {e}")
        print("Skipping this scenario due to memory or time constraints.")
        all_results['scenario_7'] = None
    
    # Scenario 8: Very large multi-bin system (2 bins, 100 of each type per bin)
    print("\n" + "="*70)
    print("SCENARIO 8: Very large multi-bin system (2 bins, N=200 each, 100 of each type per bin)")
    print("="*70)
    try:
        stats8 = benchmark_multi_bin(
            N_list=[200, 200],
            omega_list=[1.0, 1.1],
            theta_v=0.15,
            mu=0.5,
            t_max=10.0,
            num_times=100,
            num_runs=num_runs
        )
        all_results['scenario_8'] = stats8
    except Exception as e:
        print(f"ERROR in Scenario 8: {e}")
        print("Skipping this scenario due to memory or time constraints.")
        all_results['scenario_8'] = None
    
    # ========== STREAM INTERFACE BENCHMARKS ==========
    print("\n" + "="*70)
    print("STREAM INTERFACE BENCHMARKS")
    print("="*70)
    
    # Stream Scenario 1: Small single-bin system
    print("\n" + "="*70)
    print("STREAM SCENARIO 1: Small single-bin system")
    print("="*70)
    stream_stats1 = benchmark_single_bin_stream(
        N=20,
        omega=1.0,
        theta_v=0.15,
        mu=0.5,
        t_max=10.0,
        num_times=100,
        num_runs=num_runs,
        chunk=64
    )
    all_results['stream_scenario_1'] = stream_stats1
    
    # Stream Scenario 2: Medium single-bin system
    print("\n" + "="*70)
    print("STREAM SCENARIO 2: Medium single-bin system")
    print("="*70)
    stream_stats2 = benchmark_single_bin_stream(
        N=50,
        omega=1.0,
        theta_v=0.15,
        mu=0.5,
        t_max=10.0,
        num_times=100,
        num_runs=num_runs,
        chunk=64
    )
    all_results['stream_scenario_2'] = stream_stats2
    
    # Stream Scenario 3: Small multi-bin system (2 bins)
    print("\n" + "="*70)
    print("STREAM SCENARIO 3: Small multi-bin system (2 bins)")
    print("="*70)
    stream_stats3 = benchmark_multi_bin_stream(
        N_list=[10, 10],
        omega_list=[1.0, 1.1],
        theta_v=0.15,
        mu=0.5,
        t_max=10.0,
        num_times=100,
        num_runs=num_runs,
        chunk=64
    )
    all_results['stream_scenario_3'] = stream_stats3
    
    # Stream Scenario 4: Medium multi-bin system (2 bins, larger)
    print("\n" + "="*70)
    print("STREAM SCENARIO 4: Medium multi-bin system (2 bins, larger)")
    print("="*70)
    stream_stats4 = benchmark_multi_bin_stream(
        N_list=[20, 20],
        omega_list=[1.0, 1.1],
        theta_v=0.15,
        mu=0.5,
        t_max=10.0,
        num_times=100,
        num_runs=num_runs,
        chunk=64
    )
    all_results['stream_scenario_4'] = stream_stats4
    
    # Stream Scenario 5: Large single-bin system (50 of each type)
    print("\n" + "="*70)
    print("STREAM SCENARIO 5: Large single-bin system (N=100, 50 of each type)")
    print("="*70)
    try:
        stream_stats5 = benchmark_single_bin_stream(
            N=100,
            omega=1.0,
            theta_v=0.15,
            mu=0.5,
            t_max=10.0,
            num_times=100,
            num_runs=num_runs,
            chunk=64
        )
        all_results['stream_scenario_5'] = stream_stats5
    except Exception as e:
        print(f"ERROR in Stream Scenario 5: {e}")
        print("Skipping this scenario due to memory or time constraints.")
        all_results['stream_scenario_5'] = None
    
    # Stream Scenario 6: Very large single-bin system (100 of each type)
    print("\n" + "="*70)
    print("STREAM SCENARIO 6: Very large single-bin system (N=200, 100 of each type)")
    print("="*70)
    try:
        stream_stats6 = benchmark_single_bin_stream(
            N=200,
            omega=1.0,
            theta_v=0.15,
            mu=0.5,
            t_max=10.0,
            num_times=100,
            num_runs=num_runs,
            chunk=64
        )
        all_results['stream_scenario_6'] = stream_stats6
    except Exception as e:
        print(f"ERROR in Stream Scenario 6: {e}")
        print("Skipping this scenario due to memory or time constraints.")
        all_results['stream_scenario_6'] = None
    
    # Stream Scenario 7: Large multi-bin system (2 bins, 50 of each type per bin)
    print("\n" + "="*70)
    print("STREAM SCENARIO 7: Large multi-bin system (2 bins, N=100 each, 50 of each type per bin)")
    print("="*70)
    try:
        stream_stats7 = benchmark_multi_bin_stream(
            N_list=[100, 100],
            omega_list=[1.0, 1.1],
            theta_v=0.15,
            mu=0.5,
            t_max=10.0,
            num_times=100,
            num_runs=num_runs,
            chunk=64
        )
        all_results['stream_scenario_7'] = stream_stats7
    except Exception as e:
        print(f"ERROR in Stream Scenario 7: {e}")
        print("Skipping this scenario due to memory or time constraints.")
        all_results['stream_scenario_7'] = None
    
    # Stream Scenario 8: Very large multi-bin system (2 bins, 100 of each type per bin)
    print("\n" + "="*70)
    print("STREAM SCENARIO 8: Very large multi-bin system (2 bins, N=200 each, 100 of each type per bin)")
    print("="*70)
    try:
        stream_stats8 = benchmark_multi_bin_stream(
            N_list=[200, 200],
            omega_list=[1.0, 1.1],
            theta_v=0.15,
            mu=0.5,
            t_max=10.0,
            num_times=100,
            num_runs=num_runs,
            chunk=64
        )
        all_results['stream_scenario_8'] = stream_stats8
    except Exception as e:
        print(f"ERROR in Stream Scenario 8: {e}")
        print("Skipping this scenario due to memory or time constraints.")
        all_results['stream_scenario_8'] = None
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nAll scenarios completed. Check individual results above.")
    print("\nKey observations:")
    print("- LinearOperator should use less memory (no explicit H matrix)")
    print("- Sparse matrices may be faster for small systems (better cache locality)")
    print("- LinearOperator should scale better for large systems")
    print("- Stream interface should use less memory (doesn't store all states)")
    print("- Stream interface may be slightly slower due to generator overhead")
    
    # Write results to markdown file
    write_markdown_report(all_results)


if __name__ == "__main__":
    main()

