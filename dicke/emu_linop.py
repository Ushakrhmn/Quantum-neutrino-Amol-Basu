"""
Electron-muon neutrino oscillation simulation.

Compares Many-Body (Dicke) vs Mean Field Theory (MFT) for a two-bin system:
  - Bin 1: ν_e and ν_μ at energy E1
  - Bin 2: ν_e and ν_μ at energy E2

Natural units: ω₁ ≡ 1 (vacuum oscillation frequency of bin 1 as time unit)
  - Time τ = ω₁t is dimensionless
  - ω₂/ω₁ = E₁/E₂ (energy ratio determines frequency ratio)
  - μ parameter represents μ/ω₁ (dimensionless interaction strength)
"""

import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

import mft
import dicke_linop as dc_linop
import dicke_collective_sparse_opt as dc


# ==========================================
# 1. Configuration & Argument Parsing
# ==========================================

def parse_arguments():
    """Parse command line arguments for simulation parameters."""
    parser = argparse.ArgumentParser(
        description="E-μ neutrino oscillation simulation (natural units: ω₁ ≡ 1)."
    )
    
    # Bin 1 configuration
    parser.add_argument('--e1', type=int, default=1, 
                        help="Number of ν_e in bin 1")
    parser.add_argument('--m1', type=int, default=0, 
                        help="Number of ν_μ in bin 1")
    
    # Bin 2 configuration
    parser.add_argument('--e2', type=int, default=0, 
                        help="Number of ν_e in bin 2")
    parser.add_argument('--m2', type=int, default=1, 
                        help="Number of ν_μ in bin 2")
    
    # Energy ratio (determines ω₂/ω₁)
    parser.add_argument('--eratio', type=float, default=1.2, 
                        help="Energy ratio E₂/E₁ (ω₂/ω₁ = E₁/E₂ = 1/eratio)")
    
    # Physical parameters (dimensionless)
    parser.add_argument('--mu', type=float, default=5.0, 
                        help="Interaction strength μ/ω₁ (dimensionless)")
    parser.add_argument('--theta', type=float, default=np.pi/2 - 0.2,
                        help="Vacuum mixing angle")
    
    # Simulation parameters
    parser.add_argument('--tmax', type=float, default=10.0, 
                        help="Maximum time τ = ω₁t (dimensionless)")
    parser.add_argument('--s', type=int, default=100, 
                        help="Number of time steps")
    parser.add_argument('--chunk', type=int, default=64,
                        help="Block size for streaming evolution")
    parser.add_argument('--normalize', action='store_true',
                        help="L2-normalize |psi| at each step")
    parser.add_argument('--dtype', type=str, default='complex128',
                        choices=['complex128', 'complex64'],
                        help="Complex dtype for evolution")
    
    # Output
    parser.add_argument('--savename', type=str, default='emu',
                        help="Base name for saved figure")
    
    return parser.parse_args()


def print_simulation_info(args):
    """Print simulation configuration summary."""
    n1 = args.e1 + args.m1
    n2 = args.e2 + args.m2
    n_total = n1 + n2
    
    # In natural units: ω₁ = 1, ω₂ = 1/eratio
    omega2 = 1.0 / args.eratio
    
    print("=" * 55)
    print("E-MU SIMULATION (Natural Units: ω₁ ≡ 1)")
    print("=" * 55)
    print(f"  Bin 1: {n1} neutrinos (e={args.e1}, μ={args.m1}) | ω₁ = 1")
    print(f"  Bin 2: {n2} neutrinos (e={args.e2}, μ={args.m2}) | ω₂ = {omega2:.4f}")
    print(f"  Energy ratio E₂/E₁ = {args.eratio:.4f}")
    print(f"  μ/ω₁ = {args.mu:.4f}  |  θ = {args.theta:.4f}")
    print(f"  Time: τ ∈ [0, {args.tmax}] over {args.s} steps")
    print("=" * 55)


# ==========================================
# 2. Mean Field Theory Solver
# ==========================================

def solve_mean_field(args, tau_table):
    """
    Solve using Mean Field Theory.
    
    In natural units (ω₁ ≡ 1):
      - ω₁ = 1, ω₂ = 1/eratio
      - μ is the dimensionless interaction strength μ/ω₁
    
    Returns:
        (P_e for bin 1, P_e for bin 2) as 1D arrays over time.
    """
    print("[MFT] Evaluating...")
    
    n1 = args.e1 + args.m1
    n2 = args.e2 + args.m2
    n_total = n1 + n2
    
    # Natural units: ω₁ = 1, ω₂ = E₁/E₂ = 1/eratio
    omega1 = 1.0
    omega2 = 1.0 / args.eratio
    
    # Uniform interaction matrix (μ/N scaling)
    j_matrix = (args.mu / n_total) * np.ones((n_total, n_total))
    
    # Build MFT inputs
    omega_array = np.array([omega1] * n1 + [omega2] * n2)
    initial_flavors = ["e"] * args.e1 + ["mu"] * args.m1 + ["e"] * args.e2 + ["mu"] * args.m2
    
    # Solve
    solution = mft.P_osc_RS(
        tau_table, args.theta, omega_array, 0, j_matrix,
        initial_flavors=initial_flavors
    )
    
    # Extract Pz component and compute survival probabilities
    pz = solution.y.reshape(n_total, 3, -1)[:, 2, :]
    p_e_raw = 0.5 * (1 + pz)
    
    p_e_bin1 = np.mean(p_e_raw[:n1, :], axis=0)
    p_e_bin2 = np.mean(p_e_raw[n1:, :], axis=0)
    
    print("[MFT] Done.")
    return p_e_bin1, p_e_bin2


# ==========================================
# 3. Dicke (Many-Body) Solver
# ==========================================

def solve_dicke_model(args, tau_table):
    """
    Solve using full many-body Dicke model with LinearOperator.
    
    In natural units (ω₁ ≡ 1):
      - omega_list = [1.0, 1.0/eratio]
      - mu parameter is dimensionless μ/ω₁
    
    Returns:
        (times, P_e array of shape (steps, 2), S_list)
    """
    print("[Dicke] Building Hamiltonian...")
    
    n1 = args.e1 + args.m1
    n2 = args.e2 + args.m2
    n_total = n1 + n2
    cdtype = np.complex64 if args.dtype == 'complex64' else np.complex128
    
    # Natural units: ω₁ = 1, ω₂ = 1/eratio
    omega1 = 1.0
    omega2 = 1.0 / args.eratio
    
    # Initial state
    psi0, S_list = dc.multi_bin_initial_state([args.e1, args.e2], [args.m1, args.m2])
    m_list = [(args.e1 - args.m1) / 2.0, (args.e2 - args.m2) / 2.0]
    psi0 = dc.product_dicke_state(S_list, m_list)
    
    # Flatten to dense array
    if hasattr(psi0, 'toarray'):
        psi0 = psi0.toarray()
    elif hasattr(psi0, 'todense'):
        psi0 = np.asarray(psi0.todense())
    psi0 = np.asarray(psi0, dtype=cdtype).flatten()
    
    # Build Hamiltonian (factor of 2 from SU(2) commutator relations)
    H, (_, _, Jz_list), S_list, dims = dc_linop.build_multi_bin_hamiltonian(
        N_list=[int(2*S) for S in S_list],
        omega_list=[omega1, omega2],
        theta_v=args.theta,
        mu=args.mu * 2.0 / n_total,
        dtype=cdtype,
    )
    
    print("[Dicke] Evolving...")
    
    # Stream evolution
    stream = dc_linop.evolve_times_stream(
        H, psi0, tau_table,
        chunk=args.chunk,
        normalize=args.normalize,
        copy_state=False,
        dtype=cdtype,
    )
    
    # Compute observables
    times, _, p_e = dc_linop.observables_from_stream(
        stream, Jz_list, S_list,
        dims=dims,
    )
    
    print("[Dicke] Done.")
    return times, p_e, S_list


# ==========================================
# 4. Visualization
# ==========================================

# Plot colors
COLOR_BIN1 = "blue"
COLOR_BIN2 = "red"


def setup_plotting_style():
    """Configure matplotlib defaults."""
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.size': 16,
        'figure.figsize': (16, 8),
    })


def plot_comparison(t_dicke, p_dicke, t_mft, p_mft, args, S_list):
    """
    Generate comparison plot: Dicke vs MFT with residuals.
    
    Args:
        t_dicke: Time array from Dicke solver (τ = ω₁t)
        p_dicke: P_e array (steps, 2) from Dicke
        t_mft: Time array for MFT (same as tau_table)
        p_mft: (P_e_bin1, P_e_bin2) from MFT
    """
    print("[Plot] Generating...")
    
    mft_bin1, mft_bin2 = p_mft
    n1, n2 = int(2*S_list[0]), int(2*S_list[1])
    n_total = n1 + n2
    omega2 = 1.0 / args.eratio
    
    fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])
    
    # --- Main plot ---
    ax_main.plot(t_dicke, p_dicke[:, 0], color=COLOR_BIN1, 
                 label=f"Bin 1 (N={n1}, ω=1)")
    ax_main.plot(t_dicke, p_dicke[:, 1], color=COLOR_BIN2, 
                 label=f"Bin 2 (N={n2}, ω={omega2:.2f})")
    ax_main.plot(t_mft, mft_bin1, color=COLOR_BIN1, ls="--", label="Bin 1 (MFT)")
    ax_main.plot(t_mft, mft_bin2, color=COLOR_BIN2, ls="--", label="Bin 2 (MFT)")
    
    ax_main.set_xlabel(r"$\tau = \omega_1 t$")
    ax_main.set_ylabel("Survival Probability P(e)")
    ax_main.set_title("Neutrino Flavor Evolution: Many-Body (Solid) vs Mean Field (Dashed)")
    ax_main.legend(loc="best")
    ax_main.grid(True, alpha=0.3)
    
    # Parameter info box
    info = f"N = {n_total}\nμ/ω₁ = {args.mu}\nθ = {args.theta:.2f}\nE₂/E₁ = {args.eratio}"
    ax_main.text(0.02, 0.98, info, transform=ax_main.transAxes,
                 va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # --- Residuals plot ---
    res1 = mft_bin1 - p_dicke[:, 0]
    res2 = mft_bin2 - p_dicke[:, 1]
    
    ax_res.plot(t_dicke, res1, color=COLOR_BIN1, label="Residual (Bin 1)")
    ax_res.plot(t_dicke, res2, color=COLOR_BIN2, label="Residual (Bin 2)")
    
    ax_res.set_xlabel(r"$\tau = \omega_1 t$")
    ax_res.set_ylabel("Diff (MFT - Dicke)")
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
    filename = f"{args.savename}_{timestamp}_N{n_total}_mu{args.mu}.png"
    plt.savefig(filename)
    print(f"[Plot] Saved: {filename}")
    
    plt.show()


# ==========================================
# 5. Main
# ==========================================

def main():
    args = parse_arguments()
    setup_plotting_style()
    
    print_simulation_info(args)
    
    tau_table = np.linspace(0, args.tmax, args.s)
    
    mft_results = solve_mean_field(args, tau_table)
    t_dicke, dicke_results, s_list = solve_dicke_model(args, tau_table)
    
    plot_comparison(
        t_dicke=t_dicke,
        p_dicke=dicke_results,
        t_mft=tau_table,
        p_mft=mft_results,
        args=args,
        S_list=s_list,
    )


if __name__ == "__main__":
    main()
