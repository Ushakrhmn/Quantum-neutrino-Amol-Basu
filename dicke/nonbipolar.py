import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

# Assuming these are your custom modules in the same directory
import mft
import dicke_collective_sparse_opt as dc

# ==========================================
# 1. Configuration & Argument Parsing
# ==========================================

def parse_arguments():
    """
    Parses command line arguments and returns the args object.
    Simplified for MFT vs Heisenberg comparison.
    """
    parser = argparse.ArgumentParser(description="MFT vs Heisenberg for Bipolar System.")
    
    # Simulation Parameters
    group_sim = parser.add_argument_group('Simulation Parameters')
    group_sim.add_argument('--chunk', type=int, default=64, help='expm_multiply block size (if applicable)')
    group_sim.add_argument('--j', type=float, default=5.0, help="interaction strength (default 1.0)")
    group_sim.add_argument('--l', type=float, default=10.0, help="baseline length")
    group_sim.add_argument('--s', type=int, default=500, help="number of steps (resolution)")

    # System Configuration (Bin 1 - Neutrinos)
    group_bin1 = parser.add_argument_group('Bin 1 Configuration (Electron Neutrinos)')
    group_bin1.add_argument('--e', type=int, default=1, help="number of electron neutrinos in bin 1")
    group_bin1.add_argument('--energy1', type=float, default=1.0, help="energy of bin 1")

    # System Configuration (Bin 2 - Anti-Neutrinos)
    group_bin2 = parser.add_argument_group('Bin 2 Configuration (Electron Anti-Neutrinos)')
    group_bin2.add_argument('--b', type=int, default=1, help="number of electron anti-neutrinos in bin 2")
    group_bin2.add_argument('--energy2', type=float, default=1.0, help="energy of bin 2")

    # Output
    parser.add_argument('--savename', type=str, default='bipolar_mft_heis', help="base name of the saved figure")
    
    return parser.parse_args()

def setup_physics_constants():
    """
    Returns fixed physical constants used in the simulation.
    Bipolar Inverted Hierarchy settings.
    """
    return {
        'theta': np.pi/2 - 0.2,
        'dmsq': 1.0      # Normal Hierarchy
    }

def print_simulation_info(args, n1, n2):
    """
    Prints a summary of the simulation parameters to stdout.
    """
    print("-" * 40)
    print(f"NON-BIPOLAR SIMULATION: MFT vs HEISENBERG")
    print("-" * 40)
    print(f"Bin 1: {n1} Electron Neutrinos @ Energy={args.energy1}")
    print(f"Bin 2: {n2} Electron Anti-Neutrinos @ Energy={args.energy2}")
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
        tuple: (mft_p_e_bin1, mft_p_e_bin2)
    """
    print("[MFT] Starting Mean Field evaluation...")
    
    n1 = args.e
    n2 = args.b
    n_total = n1 + n2
    
    # Anti-neutrinos have negative omega in this convention for MFT
    omega1 = constants['dmsq'] / (2.0 * args.energy1)
    omega2 = -constants['dmsq'] / (2.0 * args.energy2)

    j_matrix = (args.j / n_total) * np.ones((n_total, n_total))
    
    mft_omega = np.array([omega1] * n1 + [omega2] * n2)
    mft_initial_flavours = ["e"] * n1 + ["ebar"] * n2

    mft_sol_raw = mft.P_osc_RS(
        l_table, 
        constants['theta'], 
        mft_omega, 
        0, 
        j_matrix, 
        initial_flavors=mft_initial_flavours
    )
    
    mft_sol = np.reshape(mft_sol_raw.y, (n_total, 3, len(l_table)))
    p_e_raw = 0.5 * (1 + mft_sol[:, 2, :])
    
    p_e_bin1 = np.mean(p_e_raw[:n1, :], axis=0)
    p_e_bin2 = np.mean(p_e_raw[n1:, :], axis=0)

    print("[MFT] Evaluation complete.")
    return p_e_bin1, p_e_bin2

# ==========================================
# 3. Solver: Dicke Model (Heisenberg)
# ==========================================

def solve_dicke_heisenberg(args, l_table, constants):
    """
    Solves the system using the Dicke Model in the Heisenberg picture.
    """
    print("[Heisenberg] Starting Many-Body evaluation...")
    print("[Heisenberg] Building Hamiltonian...")

    n1 = args.e
    n2 = args.b
    n_total = n1 + n2

    omega1 = constants['dmsq'] / (2.0 * args.energy1)
    omega2 = -constants['dmsq'] / (2.0 * args.energy2)

    # 1. Build Initial State
    psi0, S_list = dc.multi_bin_initial_state([args.e, 0], [0, args.b])
    
    H, (Jx_list, Jy_list, Jz_list), S_list, dims = dc.build_multi_bin_hamiltonian(
        N_list=[int(2*S) for S in S_list],
        omega_list=[omega1, omega2],
        theta_v=constants['theta'],
        mu=args.j * 2.0 / n_total,
    )

    # 2. Evolution
    print("[Heisenberg] Evolving state...")
    Y = dc.evolve_times(H, psi0, l_table)

    # 3. Compute Observables
    print("[Heisenberg] Calculating observables...")
    Jz_t, dc_p_e = dc.bin_observables(Y, Jz_list, S_list)
    
    # Bipolar Correction: 
    # For the second bin (anti-neutrinos), flip the probability interpretation.
    dc_p_e = np.copy(dc_p_e)
    # dc_p_e[:, 1] = 1.0 - dc_p_e[:, 1]

    print("[Heisenberg] Evaluation complete.")
    return l_table, dc_p_e, S_list

# ==========================================
# 4. Visualization
# ==========================================

def setup_plotting_style():
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.size'] = '14'
    matplotlib.rcParams['axes.formatter.useoffset'] = False

def plot_mft_vs_heisenberg(t_stream, heis_p_e, l_table, mft_p_e, args, S_list, constants):
    """
    Plots MFT vs Heisenberg comparison and residuals.
    """
    print("[Plot] Generating figures...")
    
    mft_p_bin1, mft_p_bin2 = mft_p_e
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1], sharex=True)

    # Marker settings to avoid clutter
    n_points = len(t_stream)
    mk_step = max(1, int(n_points / 20))
    ms_size = 8

    # --- Top Subplot: Probabilities ---
    
    # Heisenberg: Dashed + Marker 'x'
    ax1.plot(t_stream, heis_p_e[:,0], label='Bin 1 (Heisenberg)', 
             color="tab:blue", ls="--", marker="x", markevery=mk_step, ms=ms_size, alpha=0.8)
    ax1.plot(t_stream, heis_p_e[:,1], label='Bin 2 (Heisenberg)', 
             color="tab:red", ls="--", marker="x", markevery=mk_step, ms=ms_size, alpha=0.8)
    
    # MFT: Dashed Line only (Darker)
    ax1.plot(l_table, mft_p_bin1, label=f'Bin 1 (MFT)', 
             color="navy", ls="--", lw=2)
    ax1.plot(l_table, mft_p_bin2, label=f'Bin 2 (MFT)', 
             color="darkred", ls="--", lw=2)

    ax1.set_ylabel('Survival Probability P(e) / P(ebar)')
    ax1.set_title("Bipolar Flavor Evolution: MFT vs Heisenberg")
    ax1.legend(loc='upper right', fontsize='small', ncol=2)
    ax1.grid(True, alpha=0.3)

    info_text = (f"theta = {constants['theta']:.3f}\n"
                 f"dmsq = {constants['dmsq']:.2f}\n"
                 f"J = {args.j:.2f}")
    ax1.text(0.02, 0.98, info_text, 
             transform=ax1.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- Bottom Subplot: Residuals (MFT - Heisenberg) ---
    res1 = mft_p_bin1 - heis_p_e[:,0]
    res2 = mft_p_bin2 - heis_p_e[:,1]
    
    ax2.plot(t_stream, res1, label="Bin 1 (MFT - Heis)", color="blue")
    ax2.plot(t_stream, res2, label="Bin 2 (MFT - Heis)", color="red")
    
    ax2.legend(loc='upper right', fontsize='small')
    ax2.set_xlabel('Baseline (L)')
    ax2.set_ylabel('Diff (MFT - Heisenberg)')
    ax2.grid(True, alpha=0.3)

    max_res = max(np.max(np.abs(res1)), np.max(np.abs(res2)))
    mean_res = np.mean([np.mean(np.abs(res1)), np.mean(np.abs(res2))])
    
    stats_text = f'Max: {max_res:.2e}\nMean: {mean_res:.2e}'
    ax2.text(0.02, 0.95, stats_text, 
             transform=ax2.transAxes, verticalalignment='top', fontsize='small',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    n_total = args.e + args.b
    output_filename = f'{args.savename}_{timestamp}_n{n_total}.png'
    plt.savefig(output_filename)
    print(f"[Plot] Saved to {output_filename}")
    
    plt.show()

# ==========================================
# 5. Main Execution Flow
# ==========================================

def main():
    args = parse_arguments()
    constants = setup_physics_constants()
    setup_plotting_style()
    
    n1 = args.e
    n2 = args.b
    print_simulation_info(args, n1, n2)

    l_table = np.linspace(0, args.l, args.s)

    # 1. Run MFT
    mft_results = solve_mean_field(args, l_table, constants)
    
    # 2. Run Dicke Heisenberg
    t_stream, heis_results, s_list = solve_dicke_heisenberg(args, l_table, constants)
    
    # 3. Plot Comparison
    plot_mft_vs_heisenberg(
        t_stream=t_stream,
        heis_p_e=heis_results,
        l_table=l_table,
        mft_p_e=mft_results,
        args=args,
        S_list=s_list,
        constants=constants
    )

if __name__ == "__main__":
    main()