import sys
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse.linalg import expm_multiply

# Assuming these are your custom modules in the same directory
import mft
import dicke_collective_sparse_opt as dc

# ==========================================
# 1. Configuration & Argument Parsing
# ==========================================

def parse_arguments():
    """
    Parses command line arguments and returns the args object.
    Adapted for Bipolar (Nu - AntiNu) System using standard sparse evolution.
    """
    parser = argparse.ArgumentParser(description="Physical parameters for bipolar neutrino-antineutrino setup.")
    
    # Simulation Parameters
    group_sim = parser.add_argument_group('Simulation Parameters')
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
    parser.add_argument('--savename', type=str, default='bipolar_2bin', help="base name of the saved figure")
    
    return parser.parse_args()

def setup_physics_constants():
    """
    Returns fixed physical constants used in the simulation.
    Bipolar Inverted Hierarchy settings.
    """
    return {
        'theta': 0.001,   # Small mixing angle
        'dmsq': -1.0      # Inverted hierarchy
    }

def print_simulation_info(args, n1, n2):
    """
    Prints a summary of the simulation parameters to stdout.
    """
    print("-" * 40)
    print(f"BIPOLAR SIMULATION SETUP (Standard Solver)")
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
        tuple: (mft_p_e_bin1, mft_p_e_bin2) - Arrays of electron flavor survival probabilities.
    """
    print("[MFT] Starting Mean Field evaluation...")
    
    n1 = args.e
    n2 = args.b
    n_total = n1 + n2
    
    # Anti-neutrinos have negative omega
    omega1 = constants['dmsq'] / (2.0 * args.energy1)
    omega2 = -constants['dmsq'] / (2.0 * args.energy2)

    # Setup Interaction Matrix: uniform strength across bins
    j_matrix = (args.j / n_total) * np.ones((n_total, n_total))
    
    # Setup Frequencies and Initial Flavors
    # Bin 1: e, Bin 2: ebar
    mft_omega = np.array([omega1] * n1 + [omega2] * n2)
    mft_initial_flavours = ["e"] * n1 + ["ebar"] * n2

    # Run Solver
    mft_sol_raw = mft.P_osc_RS(
        l_table, 
        constants['theta'], 
        mft_omega, 
        0, 
        j_matrix, 
        initial_flavors=mft_initial_flavours
    )
    
    # Reshape and Average
    mft_sol = np.reshape(mft_sol_raw.y, (n_total, 3, len(l_table)))
    p_e_raw = 0.5 * (1 + mft_sol[:, 2, :])
    
    p_e_bin1 = np.mean(p_e_raw[:n1, :], axis=0)
    p_e_bin2 = np.mean(p_e_raw[n1:, :], axis=0)

    print("[MFT] Evaluation complete.")
    return p_e_bin1, p_e_bin2

# ==========================================
# 3. Solver: Dicke Model (Many-Body)
# ==========================================

def solve_dicke_heisenberg(args, l_table, constants):
    """
    Solves the system using the full Dicke Model (Standard Matrix Evolution).
    Renamed from solve_dicke_model.
    """
    print("[Dicke-Heisenberg] Starting Many-Body evaluation...")
    print("[Dicke-Heisenberg] Building Hamiltonian...")

    n1 = args.e
    n2 = args.b
    n_total = n1 + n2

    omega1 = constants['dmsq'] / (2.0 * args.energy1)
    omega2 = -constants['dmsq'] / (2.0 * args.energy2)

    # 1. Build Initial State
    psi0, S_list = dc.multi_bin_initial_state([args.e, 0], [0, args.b])
    
    # Note: Explicitly handling the is_antineutrino flag which was present in the original script
    # but not in the linop version.
    H, (Jx_list, Jy_list, Jz_list), S_list, dims = dc.build_multi_bin_hamiltonian(
        N_list=[int(2*S) for S in S_list],
        omega_list=[omega1, omega2],
        theta_v=constants['theta'],
        mu=args.j * 2.0 / n_total,
    )

    # 2. Evolution (Standard, not streaming)
    print("[Dicke-Heisenberg] Evolving state...")
    # Using evolve_times as in the original script (non-streaming)
    Y = dc.evolve_times(H, psi0, l_table)

    # 3. Compute Observables
    print("[Dicke-Heisenberg] Calculating observables...")
    Jz_t, dc_p_e = dc.bin_observables(Y, Jz_list, S_list)
    
    # Bipolar Correction: 
    # For the second bin (anti-neutrinos), flip the probability interpretation.
    dc_p_e = np.copy(dc_p_e)
    # dc_p_e[:, 1] = 1.0 - dc_p_e[:, 1]

    print("[Dicke-Heisenberg] Evaluation complete.")
    return l_table, dc_p_e, S_list

def solve_dicke_physical(args, l_table, constants):
    """
    Solves the system using the full Dicke Model (Standard Matrix Evolution).
    Identical copy of solve_dicke_heisenberg but with Initial State Rotation applied.
    """
    print("[Dicke-Physical] Starting Many-Body evaluation...")
    print("[Dicke-Physical] Building Hamiltonian...")

    n1 = args.e
    n2 = args.b
    n_total = n1 + n2

    omega1 = constants['dmsq'] / (2.0 * args.energy1)
    omega2 = -constants['dmsq'] / (2.0 * args.energy2)

    # 1. Build Initial State (Iso-spin basis)
    # psi0, S_list = dc.multi_bin_initial_state([args.e, 0], [0, args.b])
    psi0, S_list = dc.multi_bin_initial_state([args.e, args.b], [0, 0])
    
    # Note: Explicitly handling the is_antineutrino flag which was present in the original script
    # but not in the linop version.
    H, (Jx_list, Jy_list, Jz_list), S_list, dims = dc.build_multi_bin_hamiltonian(
        N_list=[int(2*S) for S in S_list],
        omega_list=[omega1, -omega2],
        theta_v=constants['theta'],
        mu=args.j * 2.0 / n_total,
        is_antineutrino=[False, True],
    )
    
    # --- APPLY ROTATION ---
    # As defined in bipolar_rotation_check.py: U_flip = exp(-i * pi * Jy_bin2)
    # This maps the iso-spin initial state to the physical basis state
    print("[Dicke-Physical] Applying U_y rotation to initial state (using expm_multiply)...")
    # Jy_bin2 = Jy_list[1]
    
    # CRITICAL FIX: Use expm_multiply to compute (e^A)v directly
    # This avoids creating the dense matrix e^A which causes efficiency warnings and hangs.
    # psi0 = expm_multiply(-1j * np.pi * Jy_bin2, psi0)
    # ----------------------

    # 2. Evolution (Standard, not streaming)
    print("[Dicke-Physical] Evolving state...")
    # Using evolve_times as in the original script (non-streaming)
    Y = dc.evolve_times(H, psi0, l_table)

    # 3. Compute Observables
    print("[Dicke-Physical] Calculating observables...")
    Jz_t, dc_p_e = dc.bin_observables(Y, Jz_list, S_list)
    
    # Bipolar Correction: 
    # For the second bin (anti-neutrinos), flip the probability interpretation.
    dc_p_e = np.copy(dc_p_e)
    dc_p_e[:, 1] = 1.0 - dc_p_e[:, 1]

    print("[Dicke-Physical] Evaluation complete.")
    return l_table, dc_p_e, S_list

# ==========================================
# 4. Visualization
# ==========================================

def setup_plotting_style():
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.size'] = '14'
    matplotlib.rcParams['axes.formatter.useoffset'] = False

def plot_comparison(t_stream, phys_p_e, heis_p_e, l_table, mft_p_e, args, S_list, constants):
    """
    Generates and saves the comparison plot.
    
    Layout:
    - Left (Span 2 rows): Main P(e) comparison (All 3 methods)
    - Right Top: Residual (MFT - Physical)
    - Right Bottom: Residual (Physical - Heisenberg)
    
    Style Update:
    - Markers are slightly larger (ms=8).
    - Colors: Bin 1 (Blue), Bin 2 (Red).
    """
    print("[Plot] Generating figures...")
    
    mft_p_bin1, mft_p_bin2 = mft_p_e
    
    # Use GridSpec for custom layout: 2 rows, 2 columns.
    # Column 0 is wider (ratio 1.5 : 1)
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], wspace=0.2, hspace=0.3)
    
    ax_main = fig.add_subplot(gs[:, 0]) # Left: All rows, col 0
    ax_res1 = fig.add_subplot(gs[0, 1]) # Right Top: row 0, col 1
    ax_res2 = fig.add_subplot(gs[1, 1]) # Right Bottom: row 1, col 1

    # Marker settings
    n_points = len(t_stream)
    mk_step = max(1, int(n_points / 20))
    ms_size = 8 # Larger marker size

    # --- 1. Main Plot (Left) ---
    
    # Heisenberg (Reference) - Dash + Marker 'x'
    ax_main.plot(t_stream, heis_p_e[:,0], label='Bin 1 (Heisenberg)', 
                 color="tab:blue", ls="--", marker="x", markevery=mk_step, ms=ms_size, alpha=0.5)
    ax_main.plot(t_stream, heis_p_e[:,1], label='Bin 2 (Heisenberg)', 
                 color="tab:red", ls="--", marker="x", markevery=mk_step, ms=ms_size, alpha=0.5)

    # Physical (Target) - Dash(-dot) + Marker 'o'
    ax_main.plot(t_stream, phys_p_e[:,0], label=f'Bin 1 (Physical) (N={int(2*S_list[0])})', 
                 color="tab:blue", ls="-.", marker="o", markevery=mk_step, ms=ms_size, mfc='none', lw=2)
    ax_main.plot(t_stream, phys_p_e[:,1], label=f'Bin 2 (Physical) (N={int(2*S_list[1])})', 
                 color="tab:red", ls="-.", marker="o", markevery=mk_step, ms=ms_size, mfc='none', lw=2)
    
    # MFT - Dashed Line only
    ax_main.plot(l_table, mft_p_bin1, label=f'Bin 1 (MFT)', 
                 color="navy", ls="--", lw=2)
    ax_main.plot(l_table, mft_p_bin2, label=f'Bin 2 (MFT)', 
                 color="darkred", ls="--", lw=2)

    ax_main.set_xlabel('Baseline (L)')
    ax_main.set_ylabel('Survival Probability P(e) / P(ebar)')
    ax_main.set_title("Bipolar Flavor Evolution: Comparison")
    ax_main.legend(loc='upper right', fontsize='small', ncol=2)
    ax_main.grid(True, alpha=0.3)

    info_text = (f"theta = {constants['theta']:.3f}\n"
                 f"dmsq = {constants['dmsq']:.2f}\n"
                 f"J = {args.j:.2f}")
    ax_main.text(0.02, 0.98, info_text, 
                 transform=ax_main.transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- 2. Top Right: Residual (MFT - Physical) ---
    res_mft_phys_1 = mft_p_bin1 - phys_p_e[:,0]
    res_mft_phys_2 = mft_p_bin2 - phys_p_e[:,1]
    
    ax_res1.plot(t_stream, res_mft_phys_1, label="Bin 1", color="blue")
    ax_res1.plot(t_stream, res_mft_phys_2, label="Bin 2", color="red")
    
    ax_res1.legend(loc='upper right', fontsize='small')
    # ax_res1.set_xlabel('Baseline (L)') # Shared axis concept, maybe skip x-label for top? Kept for clarity.
    ax_res1.set_ylabel('Diff (MFT - Physical)')
    ax_res1.set_title("Residual: MFT vs Physical")
    ax_res1.grid(True, alpha=0.3)

    max_res1 = max(np.max(np.abs(res_mft_phys_1)), np.max(np.abs(res_mft_phys_2)))
    mean_res1 = np.mean([np.mean(np.abs(res_mft_phys_1)), np.mean(np.abs(res_mft_phys_2))])
    
    stats_text1 = f'Max: {max_res1:.2e}\nMean: {mean_res1:.2e}'
    ax_res1.text(0.02, 0.95, stats_text1, 
                 transform=ax_res1.transAxes, verticalalignment='top', fontsize='small',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- 3. Bottom Right: Residual (Physical - Heisenberg) ---
    # Expectation: Should be effectively zero if the rotation is correct
    res_phys_heis_1 = phys_p_e[:,0] - heis_p_e[:,0]
    res_phys_heis_2 = phys_p_e[:,1] - heis_p_e[:,1]
    
    ax_res2.plot(t_stream, res_phys_heis_1, label="Bin 1", color="blue", ls=":")
    ax_res2.plot(t_stream, res_phys_heis_2, label="Bin 2", color="red", ls=":")
    
    ax_res2.legend(loc='upper right', fontsize='small')
    ax_res2.set_xlabel('Baseline (L)')
    ax_res2.set_ylabel('Diff (Physical - Heisenberg)')
    ax_res2.set_title("Residual: Physical vs Heisenberg (Check)")
    ax_res2.grid(True, alpha=0.3)
    
    max_res2 = max(np.max(np.abs(res_phys_heis_1)), np.max(np.abs(res_phys_heis_2)))
    mean_res2 = np.mean([np.mean(np.abs(res_phys_heis_1)), np.mean(np.abs(res_phys_heis_2))])
    
    stats_text2 = f'Max: {max_res2:.2e}\nMean: {mean_res2:.2e}'
    ax_res2.text(0.02, 0.95, stats_text2, 
                 transform=ax_res2.transAxes, verticalalignment='top', fontsize='small',
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
    
    # 2. Run Dicke Heisenberg (Original)
    t_stream, heis_results, s_list = solve_dicke_heisenberg(args, l_table, constants)
    
    # 3. Run Dicke Physical (With Rotation)
    t_stream, phys_results, s_list = solve_dicke_physical(args, l_table, constants)

    # 4. Plot All
    plot_comparison(
        t_stream=t_stream,
        phys_p_e=phys_results,
        heis_p_e=heis_results,
        l_table=l_table,
        mft_p_e=mft_results,
        args=args,
        S_list=s_list,
        constants=constants
    )

if __name__ == "__main__":
    main()