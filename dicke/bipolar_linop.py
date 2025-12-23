"""
Bipolar neutrino-antineutrino oscillation simulation.

Compares Many-Body (Dicke) vs Mean Field Theory (MFT) for a two-bin system:
  - Bin 1: electron neutrinos (ν_e)
  - Bin 2: electron antineutrinos (ν̄_e)
"""

import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

import mft
import dicke_linop_v3 as dc_linop
import dicke_collective_sparse_opt as dc


# ==========================================
# 1. Configuration & Argument Parsing
# ==========================================

def parse_arguments():
    """Parse command line arguments for simulation parameters."""
    parser = argparse.ArgumentParser(description="Bipolar neutrino oscillation simulation.")
    
    # Particle counts
    parser.add_argument('--e', type=int, default=1, 
                        help="Number of electron neutrinos in bin 1")
    parser.add_argument('--b', type=int, default=1, 
                        help="Number of electron antineutrinos in bin 2")
    
    # Physical parameters
    parser.add_argument('--energy', type=float, default=1.0, 
                        help="Energy of all neutrinos/antineutrinos")
    parser.add_argument('--j', type=float, default=5.0, 
                        help="Interaction strength")
    
    # Simulation parameters
    parser.add_argument('--l', type=float, default=16.0, 
                        help="Baseline length")
    parser.add_argument('--s', type=int, default=500, 
                        help="Number of time steps")
    parser.add_argument('--chunk', type=int, default=64,
                        help="Block size for streaming evolution")
    parser.add_argument('--normalize', action='store_true',
                        help="L2-normalize |psi| at each step")
    parser.add_argument('--dtype', type=str, default='complex128',
                        choices=['complex128', 'complex64'],
                        help="Complex dtype for evolution")
    
    return parser.parse_args()


def setup_physics_constants():
    """Return fixed physical constants for the simulation."""
    return {
        'theta': 0.05,   # Mixing angle
        'dmsq': -1.0      # Mass-squared difference
    }


def print_simulation_info(args, constants):
    """Print simulation configuration summary."""
    n1, n2 = args.e, args.b
    omega_nu = constants['dmsq'] / (2 * args.energy)
    omega_nubar = -omega_nu
    
    print("=" * 50)
    print("BIPOLAR SIMULATION CONFIGURATION")
    print("=" * 50)
    print(f"  Bin 1: {n1} ν_e    | Bin 2: {n2} ν̄_e")
    print(f"  Energy: {args.energy}")
    print(f"  ω(ν) = {omega_nu:.4f}  |  ω(ν̄) = {omega_nubar:.4f}")
    print(f"  J = {args.j}  |  θ = {constants['theta']}  |  Δm² = {constants['dmsq']}")
    print(f"  Baseline: [0, {args.l}] over {args.s} steps")
    print("=" * 50)


# ==========================================
# 2. Mean Field Theory Solver
# ==========================================

def solve_mean_field(args, l_table, constants):
    """
    Solve using Mean Field Theory.
    
    Returns:
        (P_e for bin 1, P_ebar for bin 2) as 1D arrays over time.
    """
    print("[MFT] Evaluating...")
    
    n1, n2 = args.e, args.b
    n_total = n1 + n2
    
    # Frequencies: ω = Δm²/(2E), opposite sign for antineutrinos
    omega_nu = constants['dmsq'] / (2 * args.energy)
    omega_nubar = -omega_nu
    
    # Uniform interaction matrix
    j_matrix = (args.j / n_total) * np.ones((n_total, n_total))
    
    # Build MFT inputs
    omega_array = np.array([omega_nu] * n1 + [omega_nubar] * n2)
    initial_flavors = ["e"] * n1 + ["ebar"] * n2
    
    # Solve
    solution = mft.P_osc_RS(
        l_table, constants['theta'], omega_array, 0, j_matrix,
        initial_flavors=initial_flavors
    )
    
    # Extract Pz component and compute survival probabilities
    pz = solution.y.reshape(n_total, 3, -1)[:, 2, :]
    
    # P_e = (1 + Pz)/2 for both neutrinos and antineutrinos
    p_e_bin1 = np.mean(0.5 * (1 + pz[:n1, :]), axis=0)
    p_e_bin2 = np.mean(0.5 * (1 + pz[n1:, :]), axis=0)
    
    print("[MFT] Done.")
    return p_e_bin1, p_e_bin2


# ==========================================
# 3. Dicke (Many-Body) Solver
# ==========================================

def solve_dicke_model(args, l_table, constants):
    """
    Solve using full many-body Dicke model with LinearOperator.
    
    Returns:
        (times, P_e array of shape (steps, 2), S_list)
    """
    print("[Dicke] Building Hamiltonian...")
    
    n1, n2 = args.e, args.b
    n_total = n1 + n2
    cdtype = np.complex64 if args.dtype == 'complex64' else np.complex128
    
    # Same omega for both bins in Dicke (sign handled by is_antineutrino flag)
    omega = constants['dmsq'] / (2 * args.energy)
    
    # Initial state: all ν_e in bin 1, all ν̄_e in bin 2
    psi0, S_list = dc.multi_bin_initial_state([n1, n2], [0, 0])
    
    # Flatten to dense array
    if hasattr(psi0, "toarray"):
        psi0 = psi0.toarray()
    elif hasattr(psi0, "todense"):
        psi0 = np.asarray(psi0.todense())
    psi0 = np.asarray(psi0, dtype=cdtype).ravel()
    
    # Build Hamiltonian (factor of 2 from SU(2) commutator relations)
    H, (_, _, Jz_list), S_list, dims = dc_linop.build_multi_bin_hamiltonian(
        N_list=[int(2*S) for S in S_list],
        omega_list=[omega, omega],
        theta_v=constants['theta'],
        mu=args.j * 2.0 / n_total,
        is_antineutrino=[False, True],
        dtype=cdtype,
    )
    
    print("[Dicke] Evolving...")
    
    # Stream evolution
    stream = dc_linop.evolve_times_stream(
        H, psi0, l_table,
        chunk=args.chunk,
        normalize=args.normalize,
        copy_state=False,
        dtype=cdtype,
    )
    
    # Compute observables (flip_bin for antineutrino P_ebar calculation)
    times, _, p_e = dc_linop.observables_from_stream(
        stream, Jz_list, S_list,
        dims=dims,
        flip_bin=[False, True],
    )
    
    print("[Dicke] Done.")
    return times, p_e, S_list


# ==========================================
# 4. Visualization
# ==========================================

# Plot colors
COLOR_NU = "blue"
COLOR_NUBAR = "red"


def setup_plotting_style():
    """Configure matplotlib defaults."""
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.size': 16,
        'figure.figsize': (16, 8),
    })


def plot_comparison(t_dicke, p_dicke, t_mft, p_mft, args, constants):
    """
    Generate comparison plot: Dicke vs MFT with residuals.
    
    Args:
        t_dicke: Time array from Dicke solver
        p_dicke: P_e array (steps, 2) from Dicke
        t_mft: Time array for MFT (same as l_table)
        p_mft: (P_e_bin1, P_e_bin2) from MFT
    """
    print("[Plot] Generating...")
    
    mft_bin1, mft_bin2 = p_mft
    n_total = args.e + args.b
    
    fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])
    
    # --- Main plot ---
    ax_main.plot(t_dicke, p_dicke[:, 0], color=COLOR_NU, label="ν_e (Dicke)")
    ax_main.plot(t_dicke, p_dicke[:, 1], color=COLOR_NUBAR, label="ν̄_e (Dicke)")
    ax_main.plot(t_mft, mft_bin1, color=COLOR_NU, ls="--", label="ν_e (MFT)")
    ax_main.plot(t_mft, mft_bin2, color=COLOR_NUBAR, ls="--", label="ν̄_e (MFT)")
    
    ax_main.set_xlabel("Baseline")
    ax_main.set_ylabel("Survival Probability")
    ax_main.legend(loc="best")
    ax_main.grid(True, alpha=0.3)
    
    # Parameter info box
    info = f"θ = {constants['theta']}\nΔm² = {constants['dmsq']}\nJ = {args.j}"
    ax_main.text(0.02, 0.98, info, transform=ax_main.transAxes, 
                 va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # --- Residuals plot ---
    res1 = mft_bin1 - p_dicke[:, 0]
    res2 = mft_bin2 - p_dicke[:, 1]
    
    ax_res.plot(t_dicke, res1, color=COLOR_NU, label="ν_e residual")
    ax_res.plot(t_dicke, res2, color=COLOR_NUBAR, label="ν̄_e residual")
    
    ax_res.set_xlabel("Baseline")
    ax_res.set_ylabel("Residual (MFT - Dicke)")
    ax_res.legend(loc="best")
    ax_res.grid(True, alpha=0.3)
    
    # Statistics box
    max_res = max(np.abs(res1).max(), np.abs(res2).max())
    mean_res = 0.5 * (np.abs(res1).mean() + np.abs(res2).mean())
    stats = f"Max: {max_res:.2e}\nMean: {mean_res:.2e}"
    ax_res.text(0.02, 0.98, stats, transform=ax_res.transAxes,
                va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"eebar_{timestamp}_n{n_total}.png"
    plt.savefig(filename)
    print(f"[Plot] Saved: {filename}")
    
    plt.show()


# ==========================================
# 5. Main
# ==========================================

def main():
    args = parse_arguments()
    constants = setup_physics_constants()
    setup_plotting_style()
    
    print_simulation_info(args, constants)
    
    l_table = np.linspace(0, args.l, args.s)
    
    mft_results = solve_mean_field(args, l_table, constants)
    t_dicke, dicke_results, _ = solve_dicke_model(args, l_table, constants)
    
    plot_comparison(
        t_dicke=t_dicke,
        p_dicke=dicke_results,
        t_mft=l_table,
        p_mft=mft_results,
        args=args,
        constants=constants,
    )


if __name__ == "__main__":
    main()
