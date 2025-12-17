#!/usr/bin/env python3
"""Configuration generator for checkpointed bipolar neutrino simulations."""

import mft
import dicke_collective_sparse_opt_linop as dc_linop
import dicke_collective_sparse_opt as dc
import numpy as np
import json
import argparse
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Initialize a new checkpointed bipolar simulation run.")
    parser.add_argument('--output', type=str, required=True, help='Path to run folder')
    parser.add_argument('--e', type=int, required=True, help='Number of electron neutrinos in bin 1')
    parser.add_argument('--b', type=int, required=True, help='Number of electron antineutrinos in bin 2')
    parser.add_argument('--energy', type=float, required=True, help='Energy of all neutrinos and antineutrinos')
    parser.add_argument('--j', type=float, required=True, help='Interaction strength')
    parser.add_argument('--l', type=float, required=True, help='Baseline (total simulation length)')
    parser.add_argument('--s', type=int, required=True, help='Total number of time steps')
    parser.add_argument('--theta', type=float, default=0.001, help='Mixing angle (default: 0.001)')
    parser.add_argument('--dmsq', type=float, default=-1.0, help='Mass squared difference (default: -1.0)')
    parser.add_argument('--chunk', type=int, default=64, help='expm_multiply block size (default: 64)')
    parser.add_argument('--normalize', action='store_true', help='L2-normalize at each step')
    parser.add_argument('--steps-per-run', type=int, default=100, help='Steps per driver invocation (default: 100)')
    args = parser.parse_args()
    
    run_folder = Path(args.output)
    run_folder.mkdir(parents=True, exist_ok=True)
    
    # Compute derived parameters
    n = args.e + args.b
    omega1 = args.dmsq / (2.0 * args.energy)
    omega2 = -args.dmsq / (2.0 * args.energy)
    l_table = np.linspace(0.0, args.l, args.s)
    
    # Build initial state
    psi0, S_list = dc.multi_bin_initial_state([args.e, 0], [0, args.b])
    m_list = [args.e / 2.0, -args.b / 2.0]
    psi0 = dc.product_dicke_state(S_list, m_list)
    
    # Convert to dense array
    if hasattr(psi0, "toarray"):
        psi0 = psi0.toarray().ravel()
    elif hasattr(psi0, "todense"):
        psi0 = np.asarray(psi0.todense()).ravel()
    else:
        psi0 = np.asarray(psi0, dtype=np.complex128).ravel()
    
    # Build Hamiltonian to get dimensions
    H, (Jx_list, Jy_list, Jz_list), S_list, dims = dc_linop.build_multi_bin_hamiltonian(
        N_list=[int(2*S) for S in S_list],
        omega_list=[omega1, omega2],
        theta_v=args.theta,
        mu=args.j / n
    )
    total_dim = int(np.prod(dims))
    
    if psi0.shape[0] != total_dim:
        raise ValueError(f"State dimension mismatch: {psi0.shape[0]} != {total_dim}")
    
    # Pre-compute MFT solution
    j = args.j * np.ones((n, n)) / n
    mft_omega = np.array([omega1] * args.e + [omega2] * args.b)
    mft_sol = mft.P_osc_RS(l_table, args.theta, mft_omega, 0, j, 
                           initial_flavors=["e"] * args.e + ["ebar"] * args.b)
    mft_sol_arr = np.reshape(mft_sol.y, (n, 3, len(l_table)))
    pz = mft_sol_arr[:, 2, :]
    mft_p_e = [
        np.mean(0.5 * (1 + pz[:args.e, :]), axis=0),
        np.mean(0.5 * (1 + pz[args.e:, :]), axis=0)
    ]
    
    # Save configuration
    config = {
        "physical": {
            "e": args.e, "b": args.b, "n": n,
            "energy": args.energy, "j": args.j, "l": args.l, "s": args.s,
            "theta": args.theta, "dmsq": args.dmsq,
            "omega1": float(omega1), "omega2": float(omega2)
        },
        "evolution": {
            "chunk": args.chunk, "normalize": args.normalize,
            "steps_per_run": args.steps_per_run
        },
        "metadata": {"created": datetime.now().isoformat()}
    }
    with open(run_folder / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save Hamiltonian info
    with open(run_folder / "hamiltonian_info.json", "w") as f:
        json.dump({
            "S_list": [float(S) for S in S_list],
            "dims": [int(d) for d in dims],
            "total_dim": total_dim,
            "n_bins": len(S_list),
            "N_list": [int(2*S) for S in S_list]
        }, f, indent=2)
    
    # Save initial state and data
    np.save(run_folder / "state_000.npy", psi0)
    np.save(run_folder / "l_table.npy", l_table)
    np.savez(run_folder / "mft_p_e.npz", mft_p_e_0=mft_p_e[0], mft_p_e_1=mft_p_e[1])
    
    # Initialize progress
    with open(run_folder / "progress.json", "w") as f:
        json.dump({
            "total_steps": args.s,
            "completed_steps": 0,
            "last_step": 0,
            "last_checkpoint": "state_000.npy",
            "is_complete": False,
            "last_updated": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"Run folder initialized: {run_folder.absolute()}")
    print(f"Particles: {args.e}+{args.b}={n}, Steps: {args.s}, Dimension: {total_dim}")


if __name__ == "__main__":
    main()
