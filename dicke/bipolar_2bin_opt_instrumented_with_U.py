# This block ensures immediate, ordered logs and periodic memory samples.
import os, sys, time, threading, signal, atexit, resource, logging, builtins, functools
from datetime import datetime

# 1) Flush stdout immediately (Python 3.7+). Also force print(..., flush=True) everywhere.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
builtins.print = functools.partial(builtins.print, flush=True)

# 2) Lightweight mem probes without external deps.
def _read_proc_status():
    info = {}
    try:
        with open("/proc/self/status","r") as f:
            for line in f:
                if line.startswith(("VmRSS","VmHWM","VmPeak","Threads")):
                    k,v = line.split(":",1)
                    info[k.strip()] = v.strip()
    except Exception:
        pass
    return info

def _parse_kb(s):
    try:
        for tok in s.split():
            try:
                return float(tok)
            except Exception:
                continue
    except Exception:
        pass
    return 0.0

def _cgroup_v2_paths():
    base = "/sys/fs/cgroup"
    limit = os.path.join(base, "memory.max")
    usage = os.path.join(base, "memory.current")
    if os.path.exists(limit) and os.path.exists(usage):
        return {"limit": limit, "usage": usage}
    return None

def _cgroup_v1_paths():
    base = "/sys/fs/cgroup/memory"
    limit = os.path.join(base, "memory.limit_in_bytes")
    usage = os.path.join(base, "memory.usage_in_bytes")
    if os.path.exists(limit) and os.path.exists(usage):
        return {"limit": limit, "usage": usage}
    return None

def _read_cgroup_mem():
    paths = _cgroup_v2_paths() or _cgroup_v1_paths()
    if not paths:
        return None
    try:
        with open(paths["limit"], "r") as f:
            s = f.read().strip()
            limit = float("inf") if s == "max" else float(s)
        with open(paths["usage"], "r") as f:
            usage = float(f.read().strip())
        return {"limit": limit, "usage": usage}
    except Exception:
        return None

def _bytes_to_mb(b): 
    try:
        return float(b)/(1024*1024)
    except Exception:
        return 0.0

def mem_snapshot():
    snap = {}
    st = _read_proc_status()
    if "VmRSS" in st: 
        snap["rss_bytes"] = _parse_kb(st["VmRSS"]) * 1024.0
    if "VmHWM" in st:
        snap["hwm_bytes"] = _parse_kb(st["VmHWM"]) * 1024.0
    if "VmPeak" in st:
        snap["peak_bytes"] = _parse_kb(st["VmPeak"]) * 1024.0
    if "Threads" in st:
        try:
            snap["num_threads"] = int(st["Threads"].split()[0])
        except Exception:
            pass
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # On Linux ru_maxrss is KB, macOS is bytes.
    if sys.platform.startswith("linux"):
        snap["ru_maxrss_bytes"] = float(getattr(ru, "ru_maxrss", 0.0)) * 1024.0
    else:
        snap["ru_maxrss_bytes"] = float(getattr(ru, "ru_maxrss", 0.0))
    cg = _read_cgroup_mem()
    if cg:
        snap["cgroup_usage_bytes"] = cg["usage"]
        snap["cgroup_limit_bytes"] = cg["limit"]
    return snap

def setup_logger(level=logging.INFO, log_file=None):
    logger = logging.getLogger("emu")
    logger.setLevel(level)
    # Clear existing handlers to avoid duplicates on repeated executions
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s [%(process)d] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger

def log_mem(logger, where="(checkpoint)"):
    s = mem_snapshot()
    parts = [f"where={where}",
             f"rss={_bytes_to_mb(s.get('rss_bytes',0)):.1f}MB",
             f"ru_maxrss={_bytes_to_mb(s.get('ru_maxrss_bytes',0)):.1f}MB"]
    if "hwm_bytes" in s:
        parts.append(f"hwm={_bytes_to_mb(s['hwm_bytes']):.1f}MB")
    if "peak_bytes" in s:
        parts.append(f"peak={_bytes_to_mb(s['peak_bytes']):.1f}MB")
    if "num_threads" in s:
        parts.append(f"threads={s['num_threads']}")
    if "cgroup_usage_bytes" in s:
        lim = s.get("cgroup_limit_bytes", float("inf"))
        lim_s = "max" if not (isinstance(lim, (int,float)) and lim < float("inf")) else f"{_bytes_to_mb(lim):.1f}MB"
        parts.append(f"cgroup={_bytes_to_mb(s['cgroup_usage_bytes']):.1f}/{lim_s}")
    logger.info("[mem] " + " | ".join(parts))

def start_mem_watcher(logger, interval_sec=2.0):
    stop_ev = threading.Event()
    def _run():
        while not stop_ev.is_set():
            try:
                log_mem(logger, "watch")
            except Exception:
                pass
            time.sleep(interval_sec)
    t = threading.Thread(target=_run, name="memwatch", daemon=True)
    t.start()
    return stop_ev, t

# Enable faulthandler so SIGUSR1 (or fatal exceptions) dumps stack to stderr.
try:
    import faulthandler
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
except Exception:
    pass


# Optional: tracemalloc helpers (off by default; enable by setting EMU_TRACE=1)
if os.environ.get("EMU_TRACE") == "1":
    try:
        import tracemalloc
        tracemalloc.start(25)
    except Exception:
        pass

def trace_top(logger, n=10, key='lineno'):
    try:
        import tracemalloc
        snap = tracemalloc.take_snapshot()
        stats = snap.statistics(key)
        logger.info("[trace] Top allocations by %s:", key)
        for i, st in enumerate(stats[:n], 1):
            logger.info("  #%d %s", i, st)
    except Exception:
        pass

def log_array(logger, name, arr):
    try:
        shape = getattr(arr, 'shape', None)
        dtype = getattr(arr, 'dtype', None)
        nbytes = getattr(arr, 'nbytes', None)
        if nbytes is None and hasattr(arr, 'size') and hasattr(arr, 'itemsize'):
            nbytes = arr.size * arr.itemsize
        logger.info("[alloc] %s: shape=%s dtype=%s size=%.2f MB", name, shape, dtype, (nbytes or 0)/(1024*1024))
    except Exception:
        pass

# --- End Lilith instrumentation ---
import mft
import dicke_collective_sparse_opt as dc  # optimized streaming API
import numpy as np
from datetime import datetime

from matplotlib import pyplot as plt

from scipy.sparse.linalg import expm  # for U_y(pi) rotation in Dicke space

# -----
# Defining physical parameters
# -----
import argparse

parser = argparse.ArgumentParser(description="Physical parameters for bipolar neutrino-antineutrino setup.")
parser.add_argument('--chunk', type=int, default=64, help='expm_multiply block size for streaming evolution')
parser.add_argument('--normalize', action='store_true', help='L2-normalize |psi| at each step (for numerical hygiene)')
parser.add_argument('--e', type=int, default=1, help="number of electron neutrinos in bin 1")
parser.add_argument('--b', type=int, default=1, help="number of electron anti-neutrinos in bin 2")
parser.add_argument('--energy1', type=float, default=1.0, help="energy of bin 1 neutrinos")
parser.add_argument('--energy2', type=float, default=1.0, help="energy of bin 2 anti-neutrinos")
parser.add_argument('--j', type=float, default=1.0, help="interaction strength (default 1.0)")
parser.add_argument('--l', type=float, default=15.0, help="baseline")
parser.add_argument('--s', type=int, default=512, help="number of steps")
parser.add_argument('--savename', type=str, default='bipolar_2bin', help="name of the saved figure")
args = parser.parse_args()
# --- instrumentation init ---
_log = setup_logger()
_log.info("args: %s", vars(args))
_wstop, _wthread = start_mem_watcher(_log, interval_sec=2.0)
atexit.register(lambda: (_wstop.set(), log_mem(_log, "atexit")))
# --- End instrumentation init ---


n1 = args.e
n2 = args.b

print("Bin 1 has {e} electron neutrinos.".format(e=args.e))
print("Bin 2 has {b} electron anti-neutrinos.".format(b=args.b))
print("Bin 1 has energy {energy1}, bin 2 has energy {energy2}.".format(energy1=args.energy1, energy2=args.energy2))

print("Simulating with uniform interaction strength of {j}.".format(j=args.j))
print("Simulating with baseline {l} across {s} steps.".format(l=args.l, s=args.s))

# theta = np.pi/2 - 0.2
theta = 0.001
dmsq = -1.0

# -----
# Plot options
# -----
import matplotlib

matplotlib.rcParams['font.family']    = 'serif'
matplotlib.rcParams['font.size']      = '16'
matplotlib.rcParams['figure.figsize'] = 16, 8
matplotlib.rcParams['axes.formatter.useoffset'] = False

E_COLOR = "blue"
MU_COLOR = "red"

# -----
# Evaluate mean field solution
# -----

l_table = np.linspace(0, args.l, args.s)

# Anti-neutrinos have negative omega
omega1, omega2 = dmsq / (2*args.energy1), - dmsq / (2*args.energy2)

n = n1 + n2

# uniform strength across bins
j = args.j / n * np.ones((n, n))

# np.fill_diagonal(j, 0) # no self-interaction

mft_omega = np.array([omega1] * n1 + [omega2] * n2)

# Bin 1: all electron neutrinos, Bin 2: all electron anti-neutrinos (still electron flavor initially)
mft_intial_flavours = ["e"] * n1 + ["ebar"] * n2

mft_sol = mft.P_osc_RS(l_table, theta, mft_omega, 0, j, initial_flavors=mft_intial_flavours)

mft_sol = np.reshape(mft_sol.y, (n,3,len(l_table)))

# average for each bin
mft_p_e = 0.5*(1+mft_sol[:,2,:])

mft_p_e = [np.mean(mft_p_e[:n1, :], axis=0), np.mean(mft_p_e[n1:, :], axis=0)]

print("Mean field solution evaluated.")

# -----
# Evaluate Dicke solution
# -----

print("Evaluating Dicke solution ...")

print("Building Hamiltonian ...")

# Bipolar setup: bin 1 is pure nu_e, bin 2 is pure nubar_e
psi0, S_list = dc.multi_bin_initial_state([args.e, args.b], [0, 0])
# m_list = [ args.e / 2., - args.b / 2. ]

# psi0 = dc.product_dicke_state(S_list, m_list)

H, (Jx_list, Jy_list, Jz_list), S_list, dims = dc.build_multi_bin_hamiltonian(
    N_list=[int(2*S) for S in S_list],
    omega_list=[omega1, omega2],
    theta_v=theta,
    mu=args.j * 2.0 / n,  # factor of 2 due to commutator relations of SU(2) J and Pauli matrices
    is_antineutrino=[False, True]  # bin 1: ν_e, bin 2: ν̄_e (enables physical ν–ν̄ interaction)
)

print("Evolving ...")

print("Evolution + observables ...")
Y = dc.evolve_times(H, psi0, l_table)
Jz_t, dc_p_e = dc.bin_observables(Y, Jz_list, S_list)
# For the second bin (index 1) in dc_p_e, replace with 1 - dc_p_e[:,1]
dc_p_e = np.copy(dc_p_e)
dc_p_e[:,1] = 1 - dc_p_e[:,1]

t_stream = l_table

# -----
# U_y(pi) rotation equivalence check between σ·σ and physical ν–ν̄ Hamiltonians
# This check does NOT affect the main Dicke solution used below; it is only a consistency test.
# -----

print("U_y(pi) rotation check between σ·σ and physical ν–ν̄ Hamiltonians ...")

# Build a σ·σ Hamiltonian with a "same-sign" vacuum for the antineutrino bin:
#   H_iso: H_vac(ν: omega1, ν̄: -omega2)  +  μ * J·J
H_iso, (Jx_iso, Jy_iso, Jz_iso), S_list_iso, dims_iso = dc.build_multi_bin_hamiltonian(
    N_list=[int(2*S) for S in S_list],
    omega_list=[omega1, -omega2],  # ν: omega1, ν̄: -omega2 → both see the same-sign vacuum direction
    theta_v=theta,
    mu=args.j * 2.0 / n,
    is_antineutrino=None
)

# Build the physical ν–ν̄ Hamiltonian with the standard bipolar vacuum:
#   H_phys_chk: H_vac(ν: omega1, ν̄: omega2)  +  μ * H_phys,int
H_phys_chk, (Jx_phys_chk, Jy_phys_chk, Jz_phys_chk), S_list_phys_chk, dims_phys_chk = dc.build_multi_bin_hamiltonian(
    N_list=[int(2*S) for S in S_list],
    omega_list=[omega1, omega2],
    theta_v=theta,
    mu=args.j * 2.0 / n,
    is_antineutrino=[False, True]
)

assert S_list_iso == S_list_phys_chk == S_list

# Initial state in the σ·σ picture is the same bipolar |ν_e, ν̄_e> state:
psi0_iso = psi0

print("Evolution + observables (σ·σ picture) ...")
Y_iso_chk = dc.evolve_times(H_iso, psi0_iso, l_table)

# Construct U_flip = exp(-i π J_y^(bin 2)) in the physical picture
Jy_bg_phys = Jy_phys_chk[1]
U_flip = expm(-1j * np.pi * Jy_bg_phys)

# For the σ·σ picture, we must rotate the OBSERVABLES (or equivalently the states)
# to compare in the physical flavor basis.
# Here we rotate the Jz operator of the antineutrino bin:
Jz_bg_iso_rot = U_flip.conjugate().transpose() @ Jz_iso[1] @ U_flip
Jz_list_iso_rot = [Jz_iso[0], Jz_bg_iso_rot]

# Now compute Pe in the physical flavor basis for the σ·σ Hamiltonian
Jz_t_iso_chk, dc_p_e_iso = dc.bin_observables(Y_iso_chk, Jz_list_iso_rot, S_list_iso)
dc_p_e_iso = np.copy(dc_p_e_iso)
dc_p_e_iso[:,1] = 1.0 - dc_p_e_iso[:,1]

# For the physical Hamiltonian, we evolve the rotated initial state psi0_phys = U_flip psi0_iso
print("Evolution + observables (physical picture + U_flip) ...")
psi0_phys_chk = U_flip @ psi0_iso
Y_phys_chk = dc.evolve_times(H_phys_chk, psi0_phys_chk, l_table)
Jz_t_phys_chk, dc_p_e_phys = dc.bin_observables(Y_phys_chk, Jz_phys_chk, S_list_phys_chk)
dc_p_e_phys = np.copy(dc_p_e_phys)
dc_p_e_phys[:,1] = 1.0 - dc_p_e_phys[:,1]

# Compare Pe for both bins
max_diff_bin0 = np.max(np.abs(dc_p_e_iso[:,0] - dc_p_e_phys[:,0]))
max_diff_bin1 = np.max(np.abs(dc_p_e_iso[:,1] - dc_p_e_phys[:,1]))
print(f"[U_y(pi) rotation check] max |ΔP_ee| (bin 1, ν)  = {max_diff_bin0:.3e}")
print(f"[U_y(pi) rotation check] max |ΔP_ee| (bin 2, ν̄) = {max_diff_bin1:.3e}")

t_stream = l_table
print("Calculated probabilities.")

print("Dicke solution evaluated.")

# -----
# Plot results
# -----

import matplotlib.pyplot as plt


plt.figure(figsize=(10,6))
plt.plot(t_stream, dc_p_e[:,0], label=f'Bin 1 (N={int(2*S_list[0])}, E={args.energy1:.2f})', color="blue")
plt.plot(t_stream, dc_p_e[:,1], label=f'Bin 2 (N={int(2*S_list[1])}, E={args.energy2:.2f})', color="red")
plt.xlabel('baseline')
plt.ylabel('Pe')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
_log.info("saving figure..."); log_mem(_log, "before savefig"); plt.savefig(args.savename + '.png')
log_mem(_log, "before show"); plt.show()


# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])

# # Main plot (top subplot)
ax1.plot(t_stream, dc_p_e[:,0], label=f'Bin 1 (Dicke) (N={int(2*S_list[0])}, E={args.energy1:.2f})', color="blue")
ax1.plot(t_stream, dc_p_e[:,1], label=f'Bin 2 (Dicke) (N={int(2*S_list[1])}, E={args.energy2:.2f})', color="red")
ax1.plot(l_table, mft_p_e[0], label=f'Bin 1 (MFT)', color="blue", ls="--")
ax1.plot(l_table, mft_p_e[1], label=f'Bin 2 (MFT)', color="red", ls="--")
ax1.legend()
ax1.set_xlabel('baseline')
ax1.set_ylabel('Pe')
ax1.legend()
# ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

ax1.text(0.02, 0.98, f'theta = {theta:.2f}\ndmsq = {dmsq:.2f}\nJ = {args.j:.2f}', 
        transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# # Residuals plot (bottom subplot)
residual1 = mft_p_e[0] - dc_p_e[:,0]
ax2.plot(t_stream, residual1, label="Residual (Bin 1)", color="blue")
residual2 = mft_p_e[1] - dc_p_e[:,1]
ax2.plot(t_stream, residual2, label="Residual (Bin 2)", color="red")
ax2.legend()
ax2.set_xlabel('baseline')
ax2.set_ylabel('Residuals (MFT - Dicke)')
ax2.grid(True, alpha=0.3)

# Add some statistics to show how close the solutions are
max_residual = max(np.max(np.abs(residual1)), np.max(np.abs(residual2)))
mean_residual = np.mean([np.mean(np.abs(residual1)), np.mean(np.abs(residual2))])

ax2.text(0.02, 0.98, f'Max residual: {max_residual:.2e}\nMean residual: {mean_residual:.2e}', 
         transform=ax2.transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Generate timestamp including minutes
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
output_filename = f'bipolar_{timestamp}_n{n}.png'
_log.info("saving figure..."); log_mem(_log, "before savefig"); plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
log_mem(_log, "before show"); plt.show()

