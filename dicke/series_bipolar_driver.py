#!/usr/bin/env python3
"""Driver script for checkpointed bipolar neutrino simulations."""

import json
import argparse
import subprocess
import sys
from pathlib import Path
import glob
import time


def main():
    parser = argparse.ArgumentParser(description="Driver script for checkpointed bipolar simulations.")
    parser.add_argument('run_folder', type=str, help='Path to run folder')
    parser.add_argument('--steps-per-run', type=int, default=None, help='Override steps-per-run from config')
    parser.add_argument('--max-steps', type=int, default=None, help='Maximum steps in this invocation')
    parser.add_argument('--no-clean-plots', action='store_true', help='Do not clean old plots')
    parser.add_argument('--keep-last-N', type=int, default=2, help='Keep last N plots when cleaning')
    parser.add_argument('--budget-days', type=float, default=None, help='Walltime budget in days; driver will choose steps to fit the budget')
    parser.add_argument('--safety-seconds', type=float, default=1800, help='Safety margin (seconds) subtracted from budget')
    parser.add_argument('--warmup-steps', type=int, default=3, help='Warmup steps to estimate sec/step when no average is recorded')
    args = parser.parse_args()
    
    run_folder = Path(args.run_folder)
    if not run_folder.exists():
        print(f"Error: Run folder not found: {run_folder}", file=sys.stderr)
        sys.exit(1)
    
    # Load configuration
    try:
        with open(run_folder / "config.json") as f:
            config = json.load(f)
        with open(run_folder / "progress.json") as f:
            progress = json.load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    total_steps = config["physical"]["s"]
    completed_steps = progress["completed_steps"]
    avg_step_sec = progress.get("avg_step_sec")
    progress_path = run_folder / "progress.json"
    
    if completed_steps >= total_steps:
        print(f"Simulation complete: {completed_steps}/{total_steps} steps")
        sys.exit(0)
    
    def save_progress(updated):
        with open(progress_path, "w") as f:
            json.dump(updated, f, indent=2)
    
    def run_sim(steps_to_run, prev_completed):
        """Run simulation for a given number of steps, return (progress, steps_done, duration_sec)."""
        sim_script = Path(__file__).parent / "series_bipolar_sim.py"
        cmd = [sys.executable, str(sim_script), str(run_folder), "--steps", str(steps_to_run)]
        start = time.monotonic()
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: Simulation failed with exit code {e.returncode}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to run simulation: {e}", file=sys.stderr)
            sys.exit(1)
        duration = time.monotonic() - start
        with open(progress_path) as f:
            updated_progress = json.load(f)
        steps_done = updated_progress["completed_steps"] - prev_completed
        return updated_progress, steps_done, duration
    
    # Clean old plots
    if not args.no_clean_plots:
        keep_n = max(1, args.keep_last_N)
        plot_files = sorted(glob.glob(str(run_folder / "plot_intermediate_*.png")))
        if len(plot_files) > keep_n:
            for filepath in plot_files[:-keep_n]:
                try:
                    Path(filepath).unlink()
                except Exception:
                    pass
    
    # Determine steps to run (budgeted or fixed)
    remaining_steps = total_steps - completed_steps
    if args.budget_days is not None:
        budget_sec = max(0.0, args.budget_days * 86400.0 - args.safety_seconds)
        if budget_sec <= 0:
            print(f"Error: Non-positive budget after safety margin ({budget_sec:.1f}s).", file=sys.stderr)
            sys.exit(1)
        
        if avg_step_sec is None:
            warmup_steps = max(1, min(args.warmup_steps, remaining_steps))
            print(f"Progress: {completed_steps}/{total_steps} ({100*completed_steps/total_steps:.1f}%)")
            print(f"Warmup: running {warmup_steps} steps to estimate sec/step...")
            progress, steps_done_warmup, duration_warmup = run_sim(warmup_steps, completed_steps)
            if steps_done_warmup <= 0:
                print("Error: Warmup produced zero completed steps.", file=sys.stderr)
                sys.exit(1)
            avg_step_sec = duration_warmup / steps_done_warmup
            progress["avg_step_sec"] = avg_step_sec
            save_progress(progress)
            completed_steps = progress["completed_steps"]
            remaining_steps = total_steps - completed_steps
            print(f"Warmup complete: {steps_done_warmup} steps, {avg_step_sec:.3f} sec/step")
        
        steps_budget = int(budget_sec // avg_step_sec)
        if args.max_steps is not None:
            steps_budget = min(steps_budget, args.max_steps)
        steps_to_run = max(0, min(steps_budget, remaining_steps))
        if steps_to_run == 0:
            print(f"Budget too small for further steps. Estimated {avg_step_sec:.3f} sec/step; remaining budget {budget_sec:.1f}s.")
            sys.exit(0)
        
        print(f"Progress: {completed_steps}/{total_steps} ({100*completed_steps/total_steps:.1f}%)")
        print(f"Budgeted run: target {steps_to_run} steps within {budget_sec:.1f}s (est {avg_step_sec:.3f} sec/step)")
        progress, steps_done_main, duration_main = run_sim(steps_to_run, completed_steps)
        if steps_done_main <= 0:
            print("Warning: No steps completed during budgeted run.")
        else:
            new_avg = duration_main / steps_done_main
            progress["avg_step_sec"] = new_avg
            save_progress(progress)
            avg_step_sec = new_avg
        completed_steps = progress["completed_steps"]
    else:
        steps_per_run = args.steps_per_run or config["evolution"]["steps_per_run"]
        steps_to_run = min(steps_per_run, remaining_steps)
        if args.max_steps is not None:
            steps_to_run = min(steps_to_run, args.max_steps)
        
        print(f"Progress: {completed_steps}/{total_steps} ({100*completed_steps/total_steps:.1f}%)")
        print(f"Running {steps_to_run} steps...")
        progress, steps_done_main, duration_main = run_sim(steps_to_run, completed_steps)
        if steps_done_main > 0:
            avg = duration_main / steps_done_main
            progress["avg_step_sec"] = avg
            save_progress(progress)
        completed_steps = progress["completed_steps"]
    
    # Final report
    print(f"\nProgress: {completed_steps}/{total_steps} ({100*completed_steps/total_steps:.1f}%)")
    if progress.get("is_complete"):
        print("*** SIMULATION COMPLETE! ***")


if __name__ == "__main__":
    main()
