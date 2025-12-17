#!/usr/bin/env python3
"""Core simulation step for checkpointed bipolar neutrino simulations."""

import dicke_collective_sparse_opt_linop as dc_linop
import dicke_collective_sparse_opt as dc
import numpy as np
import json
import argparse
from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib


def main():
    parser = argparse.ArgumentParser(description="Evolve checkpointed bipolar simulation for N steps.")
    parser.add_argument('run_folder', type=str, help='Path to run folder')
    parser.add_argument('--steps', type=int, default=None, help='Number of steps to evolve')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    args = parser.parse_args()
    
    run_folder = Path(args.run_folder).resolve()
    if not run_folder.exists():
        raise FileNotFoundError(f"Run folder not found: {run_folder}")
    
    # Load configuration
    with open(run_folder / "config.json") as f:
        config = json.load(f)
    with open(run_folder / "progress.json") as f:
        progress = json.load(f)
    with open(run_folder / "hamiltonian_info.json") as f:
        hamiltonian_info = json.load(f)
    
    phys = config["physical"]
    evol = config["evolution"]
    total_steps = phys["s"]
    start_step = progress["last_step"]
    steps_to_evolve = args.steps if args.steps is not None else evol["steps_per_run"]
    
    if progress["completed_steps"] >= total_steps:
        print(f"Simulation complete: {progress['completed_steps']}/{total_steps}")
        return
    
    steps_to_evolve = min(steps_to_evolve, total_steps - progress["completed_steps"])
    if steps_to_evolve <= 0:
        return
    
    # Load time grid and checkpoint
    l_table = np.load(run_folder / "l_table.npy")
    if start_step > 0:
        start_idx = start_step + 1  # advance past last checkpoint to avoid duplicate
        end_idx = start_idx + steps_to_evolve
    else:
        start_idx = 0
        end_idx = start_idx + steps_to_evolve + 1  # include initial point
    t_slice = l_table[start_idx:min(end_idx, len(l_table))]
    
    psi_current = np.load(run_folder / progress["last_checkpoint"])
    if psi_current.shape[0] != hamiltonian_info["total_dim"]:
        raise ValueError(f"State dimension mismatch")
    
    # Build Hamiltonian
    _, S_list_temp = dc.multi_bin_initial_state([phys["e"], 0], [0, phys["b"]])
    H, (Jx_list, Jy_list, Jz_list), S_list, dims = dc_linop.build_multi_bin_hamiltonian(
        N_list=[int(2*S) for S in S_list_temp],
        omega_list=[phys["omega1"], phys["omega2"]],
        theta_v=phys["theta"],
        mu=phys["j"] / phys["n"]
    )
    
    # Evolve and compute observables
    stream = dc_linop.evolve_times_stream(H, psi_current, t_slice,
                                         chunk=evol["chunk"], normalize=evol["normalize"])
    
    t_new, Jz_new, Pe_new = [], [], []
    psi_final = None
    
    for i, (t, psi) in enumerate(stream):
        psi_arr = np.asarray(psi, dtype=np.complex128).ravel()
        psi_final = np.array(psi_arr, copy=True)
        t_new.append(float(t))
        Jz_row, Pe_row = [], []
        for Jz, S in zip(Jz_list, S_list):
            expJz = np.vdot(psi_arr, Jz.dot(psi_arr)).real
            Jz_row.append(expJz)
            Pe_row.append(0.5 * (1.0 + expJz / S))
        Jz_new.append(Jz_row)
        Pe_new.append(Pe_row)
    
    if psi_final is None:
        raise RuntimeError("No states evolved - check time slice and checkpoint")
    
    t_new = np.asarray(t_new)
    Jz_new = np.asarray(Jz_new)
    Pe_new = np.asarray(Pe_new)
    
    # Calculate completed steps: add newly produced points to previous completed count
    completed_steps = progress["completed_steps"] + len(t_new)
    new_step = completed_steps - 1  # Last step index (0-indexed) for checkpoint naming
    
    # Save checkpoint
    checkpoint_file = f"state_{new_step:03d}.npy"
    checkpoint_path = run_folder / checkpoint_file
    temp_path = checkpoint_path.with_suffix('.tmp')
    try:
        np.save(str(temp_path), psi_final)
        import os
        import time
        time.sleep(0.1)  # Give filesystem time to sync
        if not os.path.exists(str(temp_path)):
            # Try alternative save method
            with open(str(temp_path), 'wb') as f:
                np.lib.format.write_array(f, psi_final)
            if not os.path.exists(str(temp_path)):
                raise RuntimeError(f"Failed to create checkpoint temp file: {temp_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save checkpoint: {e}") from e
    temp_path.rename(checkpoint_path)
    
    # Append observables
    obs_files = ["observables_t", "observables_Jz_0", "observables_Jz_1", 
                 "observables_Pe_0", "observables_Pe_1"]
    obs_data = [t_new, Jz_new[:, 0], Jz_new[:, 1], Pe_new[:, 0], Pe_new[:, 1]]
    
    for filename, data in zip(obs_files, obs_data):
        filepath = run_folder / f"{filename}.npy"
        existing = np.load(filepath) if filepath.exists() else np.array([], dtype=float)
        np.save(filepath, np.concatenate([existing, data]))
    
    # Update progress
    progress["completed_steps"] = completed_steps
    progress["last_step"] = new_step
    progress["last_checkpoint"] = checkpoint_file
    progress["is_complete"] = (completed_steps >= total_steps)
    progress["last_updated"] = datetime.now().isoformat()
    
    # Generate plot
    if not args.no_plot and len(t_new) > 0:
        l_table_full = np.load(run_folder / "l_table.npy")
        mft_data = np.load(run_folder / "mft_p_e.npz")
        
        # Load cumulative observables
        t_all = np.load(run_folder / "observables_t.npy")
        pe_0_all = np.load(run_folder / "observables_Pe_0.npy")
        pe_1_all = np.load(run_folder / "observables_Pe_1.npy")
        
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.size'] = '16'
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])
        
        ax1.plot(t_all, pe_0_all, label="Bin 1 nu_e", color="blue")
        ax1.plot(t_all, pe_1_all, label="Bin 2 nu_ebar", color="red")
        ax1.plot(l_table_full, mft_data["mft_p_e_0"], label='Bin 1 (MFT)', color="blue", ls="--")
        ax1.plot(l_table_full, mft_data["mft_p_e_1"], label='Bin 2 (MFT)', color="red", ls="--")
        ax1.legend()
        ax1.set_xlabel('baseline')
        ax1.set_ylabel('1/2(1 + <Pz>/n)')
        ax1.grid(True, alpha=0.3)
        
        if len(t_all) > 0:
            mft_interp_0 = np.interp(t_all, l_table_full, mft_data["mft_p_e_0"])
            mft_interp_1 = np.interp(t_all, l_table_full, mft_data["mft_p_e_1"])
            residual1 = mft_interp_0 - pe_0_all
            residual2 = mft_interp_1 - pe_1_all
            
            ax2.plot(t_all, residual1, label="Residual (Bin 1)", color="blue")
            ax2.plot(t_all, residual2, label="Residual (Bin 2)", color="red")
            ax2.legend()
            ax2.set_xlabel('baseline')
            ax2.set_ylabel('Residuals (MFT - Dicke)')
            ax2.grid(True, alpha=0.3)
            
            max_residual = max(np.max(np.abs(residual1)), np.max(np.abs(residual2)))
            mean_residual = np.mean([np.mean(np.abs(residual1)), np.mean(np.abs(residual2))])
            ax2.text(0.02, 0.98, f'Max residual: {max_residual:.2e}\nMean residual: {mean_residual:.2e}',
                     transform=ax2.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.text(0.02, 0.98,
                 f'theta = {phys["theta"]:.3f}\ndmsq = {phys["dmsq"]:.2f}\nJ = {phys["j"]:.2f}\n'
                 f'Progress: {new_step}/{total_steps} steps',
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plot_filename = f"plot_intermediate_{new_step:03d}.png"
        plt.savefig(run_folder / plot_filename, dpi=150)
        plt.close()
        progress["last_plot"] = plot_filename
    
    # Save progress
    with open(run_folder / "progress.json", "w") as f:
        json.dump(progress, f, indent=2)
    
    print(f"Step complete: {new_step}/{total_steps}")
    if progress["is_complete"]:
        print("Simulation COMPLETE!")


if __name__ == "__main__":
    main()
