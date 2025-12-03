import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

# Assuming these are your custom modules in the same directory
import mft
import dicke_collective_sparse_opt_linop as dc_linop
import dicke_collective_sparse_opt as dc

# ==========================================
# 1. Configuration & Argument Parsing
# ==========================================

def parse_arguments():
    """
    Parses command line arguments and returns the args object.
    """
    parser = argparse.ArgumentParser(description="Physical parameters for Neutrino Simulation.")
    
    # Simulation Parameters
    group_sim = parser.add_argument_group('Simulation Parameters')
    group_sim.add_argument('--chunk', type=int, default=64, help='expm_multiply block size for streaming evolution')
    group_sim.add_argument('--normalize', action='store_true', help='L2-normalize |psi| at each step')
    group_sim.add_argument('--j', type=float, default=5.0, help="interaction strength (default 5.0)")
    group_sim.add_argument('--l', type=float, default=10.0, help="baseline length")
    group_sim.add_argument('--s', type=int, default=100, help="number of steps (resolution)")

    # System Configuration (Bin 1)
    group_bin1 = parser.add_argument_group('Bin 1 Configuration')
    group_bin1.add_argument('--e1', type=int, default=1, help="number of electron neutrinos in bin 1")
    group_bin1.add_argument('--m1', type=int, default=0, help="number of muon neutrinos in bin 1")
    group_bin1.add_argument('--energy1', type=float, default=1.0, help="energy of bin 1")

    # System Configuration (Bin 2)
    group_bin2 = parser.add_argument_group('Bin 2 Configuration')
    group_bin2.add_argument('--e2', type=int, default=0, help="number of electron neutrinos in bin 2")
    group_bin2.add_argument('--m2', type=int, default=1, help="number of muon neutrinos in bin 2")
    group_bin2.add_argument('--energy2', type=float, default=1.2, help="energy of bin 2")

    # Output
    parser.add_argument('--savename', type=str, default='emu_2bin', help="base name of the saved figure")
    
    return parser.parse_args()

def setup_physics_constants():
    """
    Returns fixed physical constants used in the simulation.
    """
    return {
        'theta': np.pi/2 - 0.2,
        'dmsq': 1.0
    }

def print_simulation_info(args, n1, n2):
    """
    Prints a summary of the simulation parameters to stdout.
    """
    print("-" * 40)
    print(f"SIMULATION SETUP")
    print("-" * 40)
    print(f"Bin 1: {n1} neutrinos (e={args.e1}, mu={args.m1}) @ Energy={args.energy1}")
    print(f"Bin 2: {n2} neutrinos (e={args.e2}, mu={args.m2}) @ Energy={args.energy2}")
    print(f"Interaction Strength (J): {args.j}")
    print(f"Baseline (L): {args.l} over {args.s} steps")
    print("-" * 40)

# ==========================================
# 2. Solver: Mean Field Theory (MFT)
# ==========================================

def solve_mean_field(args, l_table, constants):
    """
    Solves the system using Mean Field Theory.
    Returns:
        tuple: (mft_p_e_bin1, mft_p_e_bin2) - Arrays of electron survival probabilities.
    """
    print("[MFT] Starting Mean Field evaluation...")
    
    n1 = args.e1 + args.m1
    n2 = args.e2 + args.m2
    n_total = n1 + n2
    
    omega1 = constants['dmsq'] / (2 * args.energy1)
    omega2 = constants['dmsq'] / (2 * args.energy2)

    # Setup Interaction Matrix: uniform strength across bins
    # Note: args.j is normalized by total N
    j_matrix = (args.j / n_total) * np.ones((n_total, n_total))
    
    # Setup Frequencies and Initial Flavors
    mft_omega = np.array([omega1] * n1 + [omega2] * n2)
    mft_initial_flavours = (["e"] * args.e1 + ["mu"] * args.m1 + 
                            ["e"] * args.e2 + ["mu"] * args.m2)

    # Run Solver
    mft_sol_raw = mft.P_osc_RS(
        l_table, 
        constants['theta'], 
        mft_omega, 
        0, 
        j_matrix, 
        initial_flavors=mft_initial_flavours
    )
    
    # Reshape: (N, 3, Steps)
    mft_sol = np.reshape(mft_sol_raw.y, (n_total, 3, len(l_table)))

    # Calculate Probability P(e) = 0.5 * (1 + Z_component)
    # Average for each bin
    p_e_raw = 0.5 * (1 + mft_sol[:, 2, :])
    
    p_e_bin1 = np.mean(p_e_raw[:n1, :], axis=0)
    p_e_bin2 = np.mean(p_e_raw[n1:, :], axis=0)

    print("[MFT] Evaluation complete.")
    return p_e_bin1, p_e_bin2

# ==========================================
# 3. Solver: Dicke Model (Many-Body)
# ==========================================

def solve_dicke_model(args, l_table, constants):
    """
    Solves the system using the full Dicke Model (Linear Operator approach).
    Returns:
        tuple: (t_stream, dc_p_e, S_list)
               t_stream: The time/baseline points returned by the stream.
               dc_p_e: Probability array shaped (Steps, 2).
               S_list: List of spins for each bin (for plotting metadata).
    """
    print("[Dicke] Starting Many-Body evaluation...")
    print("[Dicke] Building Hamiltonian...")

    n1 = args.e1 + args.m1
    n2 = args.e2 + args.m2
    n_total = n1 + n2

    omega1 = constants['dmsq'] / (2 * args.energy1)
    omega2 = constants['dmsq'] / (2 * args.energy2)

    # 1. Build Initial State
    psi0, S_list = dc.multi_bin_initial_state([args.e1, args.e2], [args.m1, args.m2])
    m_list = [(args.e1 - args.m1)/2.0, (args.e2 - args.m2)/2.0]
    psi0 = dc.product_dicke_state(S_list, m_list)

    # Flatten logic for LinearOperator compatibility
    if hasattr(psi0, 'toarray'):
        psi0 = psi0.toarray().flatten()
    elif hasattr(psi0, 'todense'):
        psi0 = np.asarray(psi0.todense()).flatten()
    else:
        psi0 = np.asarray(psi0, dtype=np.complex128).flatten()

    # 2. Build Hamiltonian (Matrix-Free)
    # factor of 2 due to commutator relations of SU(2) J and Pauli matrices
    mu_val = args.j * 2.0 / n_total 
    
    H, (_, _, Jz_list), S_list, _ = dc_linop.build_multi_bin_hamiltonian(
        N_list=[int(2*S) for S in S_list],
        omega_list=[omega1, omega2],
        theta_v=constants['theta'],
        mu=mu_val
    )

    # 3. Evolution (Streaming)
    print("[Dicke] Evolving state (streaming)...")
    stream = dc_linop.evolve_times_stream(
        H, psi0, l_table, 
        chunk=args.chunk, 
        normalize=args.normalize
    )

    # 4. Compute Observables
    t_stream, _, dc_p_e = dc_linop.observables_from_stream(stream, Jz_list, S_list)

    print("[Dicke] Evaluation complete.")
    return t_stream, dc_p_e, S_list

# ==========================================
# 4. Visualization
# ==========================================

def setup_plotting_style():
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.size'] = '16'

def plot_comparison(t_stream, dc_p_e, l_table, mft_p_e, args, S_list, constants):
    """
    Generates and saves the comparison plot between MFT and Dicke results.
    """
    print("[Plot] Generating figures...")
    
    # Unpack MFT results
    mft_p_bin1, mft_p_bin2 = mft_p_e
    
    # Create layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])

    # --- Top Subplot: Probability Evolution ---
    # Dicke
    ax1.plot(t_stream, dc_p_e[:,0], label=f'Bin 1 (N={int(2*S_list[0])}, E={args.energy1:.2f})', color="blue")
    ax1.plot(t_stream, dc_p_e[:,1], label=f'Bin 2 (N={int(2*S_list[1])}, E={args.energy2:.2f})', color="red")
    # MFT
    ax1.plot(l_table, mft_p_bin1, label=f'Bin 1 (MFT)', color="blue", ls="--")
    ax1.plot(l_table, mft_p_bin2, label=f'Bin 2 (MFT)', color="red", ls="--")

    ax1.set_xlabel('Baseline (L)')
    ax1.set_ylabel('Survival Probability P(e)')
    ax1.set_title("Neutrino Flavor Evolution: Many-Body (Solid) vs Mean Field (Dashed)")
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Info Box (Top Left)
    info_text = (f"theta = {constants['theta']:.2f}\n"
                 f"dmsq = {constants['dmsq']:.2f}\n"
                 f"J = {args.j:.2f}")
    ax1.text(0.02, 0.98, info_text, 
             transform=ax1.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- Bottom Subplot: Residuals ---
    residual1 = mft_p_bin1 - dc_p_e[:,0]
    residual2 = mft_p_bin2 - dc_p_e[:,1]
    
    ax2.plot(t_stream, residual1, label="Residual (Bin 1)", color="blue")
    ax2.plot(t_stream, residual2, label="Residual (Bin 2)", color="red")
    
    ax2.legend(loc='best')
    ax2.set_xlabel('Baseline (L)')
    ax2.set_ylabel('Diff (MFT - Dicke)')
    ax2.grid(True, alpha=0.3)

    # Statistics Box (Bottom Left)
    max_res = max(np.max(np.abs(residual1)), np.max(np.abs(residual2)))
    mean_res = np.mean([np.mean(np.abs(residual1)), np.mean(np.abs(residual2))])
    
    stats_text = f'Max residual: {max_res:.2e}\nMean residual: {mean_res:.2e}'
    ax2.text(0.02, 0.98, stats_text, 
             transform=ax2.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f'{args.savename}_{timestamp}.png'
    plt.savefig(output_filename)
    print(f"[Plot] Saved to {output_filename}")
    
    plt.show()

# ==========================================
# 5. Main Execution Flow
# ==========================================

def main():
    # 1. Setup
    args = parse_arguments()
    constants = setup_physics_constants()
    setup_plotting_style()
    
    n1 = args.e1 + args.m1
    n2 = args.e2 + args.m2
    print_simulation_info(args, n1, n2)

    # Shared time/baseline array
    l_table = np.linspace(0, args.l, args.s)

    # 2. Run MFT
    mft_results = solve_mean_field(args, l_table, constants)

    # 3. Run Dicke
    t_stream, dicke_results, s_list = solve_dicke_model(args, l_table, constants)

    # 4. Visualize
    plot_comparison(
        t_stream=t_stream,
        dc_p_e=dicke_results,
        l_table=l_table,
        mft_p_e=mft_results,
        args=args,
        S_list=s_list,
        constants=constants
    )

if __name__ == "__main__":
    main()