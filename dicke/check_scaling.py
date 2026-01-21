#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_scaling.py

Analyze Dicke onset time scaling with system size N, comparing to mean-field theory (MFT).

Natural units: ω ≡ 1 (vacuum oscillation frequency as time unit)
  - Time τ = ωt is dimensionless
  - μ parameter represents μ/ω (dimensionless interaction strength)

Fit Models:
    - log:            t = a + b·log(N)
    - sat_power:      t = t_inf - A·N^{-q}
    - sat_log:        t = t_inf - A·exp(-k·log(N))
    - log_plus_power: t = a + b·log(N) + c·N^{-q}

Features:
    - Robust fitting (--robust) with soft_l1 loss
    - Filter by --min_Ntot or --min_logN
    - Constrain t_inf to MFT onset (--use_mft_as_tinf)

Dependencies: numpy, scipy, matplotlib, dicke_collective_sparse_opt
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.sparse.linalg import expm_multiply

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dicke_collective_sparse_opt as dc

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
FloatArray = np.ndarray

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
FIT_MODELS = ("baseline", "baseline_b2")


def parse_int_list(s: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def vacuum_hamiltonian_direction(theta: float) -> FloatArray:
    """Return vacuum Hamiltonian direction B = (sin2θ, 0, -cos2θ)."""
    return np.array([np.sin(2 * theta), 0.0, -np.cos(2 * theta)], dtype=float)


def compute_onset_time(
    t: FloatArray, y: FloatArray, threshold: float = 0.05
) -> float:
    """Find earliest time when |y - y(0)| >= threshold, with linear interpolation.
    
    Returns:
        Onset time, or np.nan if threshold never reached.
    """
    y0 = float(y[0])
    deviation = np.abs(y - y0)
    crossing_indices = np.where(deviation >= threshold)[0]
    
    if len(crossing_indices) == 0:
        return np.nan
    
    i = int(crossing_indices[0])
    if i == 0:
        return float(t[0])
    
    # Linear interpolation between bracketing points
    t_before, t_after = float(t[i - 1]), float(t[i])
    d_before, d_after = float(deviation[i - 1]), float(deviation[i])
    
    if d_after == d_before:
        return t_after
    
    return t_before + (threshold - d_before) * (t_after - t_before) / (d_after - d_before)


def evolve_mft_two_mode(
    tau_grid: FloatArray,
    *,
    theta: float,
    mu: float,
    n1: int,
    n2: int,
) -> FloatArray:
    """Two-mode mean-field evolution: D = (n1·P1 + n2·P2) / N_tot.
    
    Natural units (ω ≡ 1):
      - ω_ν = -1, ω_ν̄ = +1 (inverted hierarchy)
      - μ is the dimensionless interaction strength μ/ω
    
    Returns:
        P_ee for bin 1 (neutrino) at each time point.
    """
    n_tot = n1 + n2
    weight1, weight2 = n1 / n_tot, n2 / n_tot
    B = vacuum_hamiltonian_direction(theta)

    # Natural units: ω_ν = -1, ω_ν̄ = +1 (inverted hierarchy)
    omega1 = -1.0
    omega2 = 1.0

    # Initial polarizations: ν_e → Pz=+1, ν̄_e → Pz=-1
    P1_init = np.array([0.0, 0.0, 1.0])
    P2_init = np.array([0.0, 0.0, -1.0])
    y0 = np.concatenate([P1_init, P2_init])

    def equations_of_motion(_t: float, y: FloatArray) -> FloatArray:
        P1, P2 = y[:3], y[3:6]
        D = weight1 * P1 + weight2 * P2
        H1 = omega1 * B + mu * D
        H2 = omega2 * B + mu * D
        return np.concatenate([np.cross(H1, P1), np.cross(H2, P2)])

    sol = solve_ivp(
        equations_of_motion,
        (float(tau_grid[0]), float(tau_grid[-1])),
        y0,
        t_eval=tau_grid,
        rtol=1e-9,
        atol=1e-12,
    )
    return 0.5 * (1.0 + sol.y[2, :])


def evolve_dicke_chunked(
    tau_grid: FloatArray,
    *,
    theta: float,
    mu: float,
    n1: int,
    n2: int,
    chunk_size: int,
) -> FloatArray:
    """Dicke many-body evolution using chunked propagation for memory efficiency.
    
    Natural units (ω ≡ 1):
      - omega = -1 (inverted hierarchy, sign matters)
      - μ is the dimensionless interaction strength μ/ω
    
    Returns:
        P_ee for bin 1 at each time point.
    """
    n_tot = n1 + n2
    
    # Natural units: ω = -1 (inverted hierarchy)
    omega = -1.0
    
    # Factor of 2 from SU(2) commutator relations
    mu_hamiltonian = mu * 2.0 / n_tot

    psi0, _ = dc.multi_bin_initial_state([n1, n2], [0, 0])
    H, (_, _, Jz_list), S_list, _ = dc.build_multi_bin_hamiltonian(
        N_list=[n1, n2],
        omega_list=[omega, omega],
        theta_v=theta,
        mu=mu_hamiltonian,
        is_antineutrino=[False, True],
    )

    # Convert sparse initial state to dense
    psi = psi0.toarray().ravel() if hasattr(psi0, "toarray") else np.asarray(psi0).ravel()

    S1 = float(S_list[0])
    Jz1 = Jz_list[0]
    Pee = np.empty(len(tau_grid), dtype=float)

    # Process in chunks to limit memory usage
    idx = 0
    while idx < len(tau_grid):
        end_idx = min(idx + chunk_size, len(tau_grid))
        t_block = np.asarray(tau_grid[idx:end_idx], dtype=float)
        t_relative = t_block - t_block[0]

        propagated_states = expm_multiply(
            -1j * H,
            psi,
            start=0.0,
            stop=float(t_relative[-1]),
            num=len(t_block),
            endpoint=True,
        )

        for k, psi_k in enumerate(propagated_states):
            jz_expectation = np.vdot(psi_k, Jz1.dot(psi_k.reshape(-1, 1)).ravel()).real
            Pee[idx + k] = 0.5 * (1.0 + jz_expectation / S1)

        psi = np.array(propagated_states[-1], copy=True)
        idx = end_idx

    return Pee


# -----------------------------------------------------------------------------
# Fitting utilities
# -----------------------------------------------------------------------------

def evaluate_fit_model(
    model: str, log_N: FloatArray, N: FloatArray, params: FloatArray, t_mft: float
) -> FloatArray:
    """Evaluate the specified fit model at given N values.
    
    Baseline model: t = t₀ + (t_MFT - t₀) × (1 - exp(-a × (ln N)^b))
    """
    if model == "baseline":
        a, b, t0 = params
        return t0 + (t_mft - t0) * (1 - np.exp(-a * np.power(log_N, b)))
    elif model == "baseline_b2":
        a, t0 = params
        return t0 + (t_mft - t0) * (1 - np.exp(-a * np.power(log_N, 2)))
    else:
        raise ValueError(f"Unknown fit model: {model}. Valid: {FIT_MODELS}")


def _compute_initial_guess(
    model: str,
    log_N: FloatArray,
    y: FloatArray,
    t_mft: float,
) -> FloatArray:
    """Compute reasonable initial parameter guesses for fitting."""
    y_min = float(np.min(y))
    t0_guess = max(0.01, y_min * 0.5)
    
    if model == "baseline":
        # params: (a, b, t0)
        return np.array([0.01, 2.0, t0_guess], dtype=float)
    elif model == "baseline_b2":
        # params: (a, t0)
        return np.array([0.01, t0_guess], dtype=float)
    
    raise ValueError(f"Unknown fit model: {model}")


def fit_onset_scaling(
    model: str,
    N: FloatArray,
    y: FloatArray,
    t_mft: float,
    *,
    robust: bool,
) -> tuple[FloatArray, FloatArray, float, "OptimizeResult"]:
    """Fit onset time scaling model to data.
    
    Baseline model: t = t₀ + (t_MFT - t₀) × (1 - exp(-a × (ln N)^b))
    
    Args:
        model: "baseline" (free a, b, t₀) or "baseline_b2" (fixed b=2)
        N: System sizes
        y: Onset times
        t_mft: MFT onset time (asymptotic limit)
        robust: Use soft_l1 loss for outlier resistance
    
    Returns:
        (parameters, fitted_values, R², optimizer_result)
    """
    N = np.asarray(N, dtype=float)
    y = np.asarray(y, dtype=float)
    log_N = np.log(N)
    
    y_min = float(np.min(y))
    
    p0 = _compute_initial_guess(model, log_N, y, t_mft)
    
    if model == "baseline":
        # params: (a, b, t0)
        # bounds: a > 0, b in [0.5, 5], t0 in [0, y_min * 1.5]
        bounds = ([0.0, 0.5, 0.0], [0.5, 5.0, y_min * 1.5])
        
        def residual(p: FloatArray) -> FloatArray:
            a, b, t0 = p
            return (t0 + (t_mft - t0) * (1 - np.exp(-a * np.power(log_N, b)))) - y
    
    elif model == "baseline_b2":
        # params: (a, t0), fixed b=2
        bounds = ([0.0, 0.0], [0.5, y_min * 1.5])
        
        def residual(p: FloatArray) -> FloatArray:
            a, t0 = p
            return (t0 + (t_mft - t0) * (1 - np.exp(-a * np.power(log_N, 2)))) - y
    
    else:
        raise ValueError(f"Unknown fit model: {model}")

    # Perform fit
    loss_function = "soft_l1" if robust else "linear"
    result = least_squares(residual, p0, bounds=bounds, loss=loss_function, f_scale=0.1)
    params = np.array(result.x, dtype=float)

    # Compute R²
    y_fitted = evaluate_fit_model(model, log_N, N, params, t_mft)
    ss_residual = float(np.sum((y - y_fitted) ** 2))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else np.nan

    return params, y_fitted, r_squared, result


def format_fit_parameters(model: str, params: FloatArray, t_mft: float) -> str:
    """Format fit parameters for display."""
    if model == "baseline":
        a, b, t0 = params
        return f"a={a:.6g}, b={b:.4f}, t₀={t0:.6g}, t_MFT={t_mft:.6g}"
    elif model == "baseline_b2":
        a, t0 = params
        return f"a={a:.6g}, t₀={t0:.6g}, t_MFT={t_mft:.6g} (b=2 fixed)"
    else:
        return ", ".join(f"p{i}={val:.6g}" for i, val in enumerate(params))


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def _save_individual_plot(
    tau_grid: FloatArray,
    Pee_dicke: FloatArray,
    Pee_mft: FloatArray | None,
    n_tot: int,
    n1: int,
    n2: int,
    out_prefix: str,
) -> None:
    """Save individual Pee(τ) comparison plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
    
    ax1.plot(tau_grid, Pee_dicke, label="Dicke", color="blue")
    if Pee_mft is not None:
        ax1.plot(tau_grid, Pee_mft, label="MFT", color="blue", ls="--")
    ax1.set(xlabel=r"$\tau = \omega t$", ylabel="Pee (bin1)", title=f"N={n_tot} (n1={n1}, n2={n2})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    if Pee_mft is not None:
        residual = Pee_mft - Pee_dicke
        ax2.plot(tau_grid, residual, color="blue")
        ax2.set(xlabel=r"$\tau = \omega t$", ylabel="Residual (MFT - Dicke)")
        ax2.grid(alpha=0.3)
        ax2.text(0.02, 0.98, f"Max |residual|: {np.max(np.abs(residual)):.2e}",
                 transform=ax2.transAxes, va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    else:
        ax2.set_visible(False)

    plt.tight_layout()
    out_path = f"{out_prefix}_Pee_N{n_tot}.png"
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"    Saved: {out_path}")


def _save_scaling_plot(
    Ntot: FloatArray,
    t_dicke: FloatArray,
    t_mft: FloatArray,
    N_filtered: FloatArray,
    fit_params: FloatArray | None,
    r_squared: float | None,
    fit_model: str,
    threshold: float,
    out_prefix: str,
    no_mft: bool,
    no_fit: bool,
    t_mft_value: float,
) -> None:
    """Save onset time vs log(N) scaling plot."""
    plt.figure(figsize=(8, 5))
    plt.scatter(np.log(Ntot), t_dicke, label="Dicke onset")

    if not no_fit and fit_params is not None:
        N_plot = np.linspace(np.min(N_filtered), np.max(N_filtered), 300)
        log_N_plot = np.log(N_plot)
        y_plot = evaluate_fit_model(fit_model, log_N_plot, N_plot, fit_params, t_mft_value)
        plt.plot(log_N_plot, y_plot, label=f"fit({fit_model}): R²={r_squared:.4f}")

    if not no_mft:
        plt.axhline(t_mft_value, linestyle="--", color="gray", 
                    label=f"MFT onset = {t_mft_value:.4g}")

    plt.xlabel("ln(N)")
    plt.ylabel(f"τ_onset (thr={threshold})")
    plt.title(r"Baseline Model: $\tau = \tau_0 + (\tau_{MFT} - \tau_0)(1 - e^{-a(\ln N)^b})$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    out_path = f"{out_prefix}_onset_vs_logN.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved: {out_path}")


def _save_collapse_plot(
    tau_grid: FloatArray,
    Pee_curves: dict[int, tuple[FloatArray, FloatArray | None]],
    Ntot: FloatArray,
    t_dicke: FloatArray,
    N_filtered: FloatArray,
    out_prefix: str,
) -> None:
    """Save collapse plot with onset-time-shifted curves."""
    N_ref = int(np.max(N_filtered))
    t_onset_ref = float(t_dicke[Ntot == N_ref][0]) if np.any(Ntot == N_ref) else float(np.max(t_dicke[np.isfinite(t_dicke)]))
    
    plt.figure(figsize=(9, 5))
    for N in sorted(Pee_curves.keys()):
        Pee_d, _ = Pee_curves[N]
        idx = np.where(Ntot == N)[0]
        if len(idx) == 0 or not np.isfinite(t_dicke[idx[0]]):
            continue
        t_onset = float(t_dicke[idx[0]])
        t_shifted = tau_grid - t_onset + t_onset_ref
        plt.plot(t_shifted, Pee_d, label=f"N={N}")
    
    plt.xlabel(r"shifted $\tau$")
    plt.ylabel("Pee (bin1, Dicke)")
    plt.title("collapse by onset-time shift")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    
    out_path = f"{out_prefix}_collapse.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved: {out_path}")


def _save_scaling_csv(
    Ntot: FloatArray,
    t_dicke: FloatArray,
    out_prefix: str,
) -> None:
    """Save raw scaling data to CSV: log(Ntot), τ_onset."""
    log_Ntot = np.log(Ntot)
    data = np.column_stack([log_Ntot, t_dicke])
    out_path = f"{out_prefix}_onset_vs_logN.csv"
    np.savetxt(
        out_path,
        data,
        delimiter=",",
        header="log(Ntot),tau_onset",
        comments="#",
        fmt="%.10g",
    )
    print(f"Saved: {out_path}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Analyze Dicke onset time scaling with system size N (natural units: ω ≡ 1)"
    )
    
    # Physical parameters
    phys = parser.add_argument_group("Physical parameters")
    phys.add_argument("--Nbar_list", type=str, default="1,5,10,20,30,40,50",
                      help="Comma-separated list of Nbar values")
    phys.add_argument("--eps", type=float, default=0.0,
                      help="Asymmetry: n1 = (1+eps)*Nbar")
    phys.add_argument("--theta", type=float, default=0.001,
                      help="Vacuum mixing angle")
    phys.add_argument("--mu", type=float, default=10.0,
                      help="Interaction strength μ/ω (dimensionless)")
    
    # Simulation parameters
    sim = parser.add_argument_group("Simulation parameters")
    sim.add_argument("--tmax", type=float, default=7.5,
                     help="Maximum time τ = ωt (dimensionless)")
    sim.add_argument("--steps", type=int, default=400, help="Number of time steps")
    sim.add_argument("--thr", type=float, default=0.05, help="Onset threshold")
    sim.add_argument("--chunk", type=int, default=32, help="Chunk size for propagation")
    sim.add_argument("--no_mft", action="store_true", help="Skip MFT calculation")
    
    # Fitting parameters
    fit = parser.add_argument_group("Fitting parameters")
    fit.add_argument("--fit_model", type=str, default="baseline", choices=list(FIT_MODELS),
                     help="Scaling model: baseline (free b) or baseline_b2 (b=2)")
    fit.add_argument("--robust", action="store_true", help="Use robust (soft_l1) loss")
    fit.add_argument("--min_Ntot", type=int, default=0, help="Minimum Ntot to include")
    fit.add_argument("--min_logN", type=float, default=-1e100, help="Minimum log(N) to include")
    fit.add_argument("--no_fit", action="store_true", help="Skip fitting")
    
    # Output parameters
    out = parser.add_argument_group("Output parameters")
    out.add_argument("--collapse", action="store_true", help="Generate collapse plot")
    out.add_argument("--plot_individual", action="store_true",
                     help="Save individual Pee(τ) plots")
    out.add_argument("--out_prefix", type=str, default="scale", help="Output file prefix")
    
    return parser


def main() -> None:
    """Main entry point."""
    args = create_argument_parser().parse_args()

    # Extract parameters
    Nbar_list = parse_int_list(args.Nbar_list)
    theta = float(args.theta)
    mu = float(args.mu)
    threshold = float(args.thr)

    tau_grid = np.linspace(0.0, float(args.tmax), int(args.steps))

    print("=" * 60)
    print("SCALING ANALYSIS (Natural Units: ω ≡ 1)")
    print("=" * 60)
    print(f"  μ/ω = {mu:.4f}  |  θ = {theta:.4f}")
    print(f"  Time: τ ∈ [0, {args.tmax}] over {args.steps} steps")
    print(f"  Onset threshold: {threshold}")
    print("=" * 60)

    # Collect results
    results: list[tuple[int, float, float]] = []
    Pee_curves: dict[int, tuple[FloatArray, FloatArray | None]] = {}

    for Nbar in Nbar_list:
        n2 = Nbar
        n1 = int(round((1.0 + args.eps) * Nbar))
        n_tot = n1 + n2

        print(f"[Nbar={Nbar:5d}] n_nu={n1:5d}, n_nubar={n2:5d}, Ntot={n_tot:5d}")

        # Dicke evolution
        Pee_dicke = evolve_dicke_chunked(
            tau_grid,
            theta=theta, mu=mu,
            n1=n1, n2=n2, chunk_size=int(args.chunk),
        )
        t_onset_dicke = compute_onset_time(tau_grid, Pee_dicke, threshold)

        # MFT evolution (optional)
        if args.no_mft:
            Pee_mft, t_onset_mft = None, np.nan
        else:
            Pee_mft = evolve_mft_two_mode(
                tau_grid,
                theta=theta, mu=mu,
                n1=n1, n2=n2,
            )
            t_onset_mft = compute_onset_time(tau_grid, Pee_mft, threshold)

        results.append((n_tot, t_onset_dicke, t_onset_mft))
        Pee_curves[n_tot] = (Pee_dicke, Pee_mft)
        
        if args.no_mft:
            print(f"    τ_onset(Dicke)={t_onset_dicke:.6g}")
        else:
            print(f"    τ_onset(Dicke)={t_onset_dicke:.6g}, τ_onset(MFT)={t_onset_mft:.6g}")

        # Individual Pee(τ) plot
        if args.plot_individual:
            _save_individual_plot(
                tau_grid, Pee_dicke, Pee_mft, n_tot, n1, n2, args.out_prefix
            )

    # Convert to arrays for analysis
    results_arr = np.array(results, dtype=float)
    Ntot = results_arr[:, 0]
    t_dicke = results_arr[:, 1]
    t_mft = results_arr[:, 2]
    
    # Get MFT onset time (use median if multiple values)
    t_mft_value = float(np.nanmedian(t_mft[np.isfinite(t_mft)])) if np.isfinite(t_mft).any() else np.nan

    # Save raw scaling data to CSV
    _save_scaling_csv(Ntot, t_dicke, args.out_prefix)

    # Filter data points
    mask = (
        np.isfinite(t_dicke) &
        (Ntot >= float(args.min_Ntot)) &
        (np.log(Ntot) >= float(args.min_logN))
    )
    N_filtered = Ntot[mask]
    y_filtered = t_dicke[mask]

    if len(N_filtered) < 3 and not args.no_fit:
        raise RuntimeError("Too few points after filtering; relax --min_Ntot or --min_logN")

    # Fit scaling model
    fit_params, r_squared = None, None
    if not args.no_fit:
        if not np.isfinite(t_mft_value):
            raise RuntimeError("MFT onset time is required for baseline model fitting. Run without --no_mft")
        
        fit_params, _, r_squared, _ = fit_onset_scaling(
            args.fit_model, N_filtered, y_filtered, t_mft_value,
            robust=args.robust,
        )

        print(f"\nFit model: {args.fit_model}")
        print(f"    t_MFT = {t_mft_value:.6g}")
        print(f"    params: {format_fit_parameters(args.fit_model, fit_params, t_mft_value)}")
        print(f"    R² = {r_squared:.6g}")

    # Generate plots
    _save_scaling_plot(
        Ntot, t_dicke, t_mft, N_filtered, fit_params, r_squared,
        args.fit_model, args.thr, args.out_prefix, args.no_mft, args.no_fit,
        t_mft_value,
    )

    if args.collapse:
        _save_collapse_plot(
            tau_grid, Pee_curves, Ntot, t_dicke, N_filtered, args.out_prefix
        )


if __name__ == "__main__":
    main()
