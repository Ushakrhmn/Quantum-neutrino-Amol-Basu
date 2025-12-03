#!/bin/bash
#SBATCH --account=def-nilic
#SBATCH --mem=128G               # memory per node
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=USR1@120        # (optional) pre-timeout signal to help debugging

set -Eeuo pipefail

mkdir -p logs

# -------- Configuration (tune here) --------
export PYTHONUNBUFFERED=1        # immediate Python stdout/stderr
export PYTHONFAULTHANDLER=1      # stack traces on fatal errors
: "${EMU_MEM_INTERVAL:=300}"      # seconds between memory samples (default 60s; set to 300 for 5 min)
: "${EMU_LOG:=logs/emu_${SLURM_JOB_ID:-$$}.log}"
: "${EMU_MEMLOG:=logs/mem_${SLURM_JOB_ID:-$$}.log}"
# ------------------------------------------

# Activate your environment
source ../env.sh

log() { echo "$(date +'%F %T%z') $*" | tee -a "$EMU_LOG"; }

detect_cgroup_mem_files() {
  local cg cg2
  # cgroup v2
  if [[ -f /sys/fs/cgroup/cgroup.controllers ]]; then
    cg=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)
    [[ -n "$cg" ]] || cg="/"
    local base="/sys/fs/cgroup${cg}"
    if [[ -f "${base}/memory.current" ]]; then
      echo "${base}/memory.current|${base}/memory.max"
      return
    fi
  fi
  # cgroup v1 (memory)
  cg2=$(awk -F: '$2=="memory"{print $3}' /proc/self/cgroup)
  if [[ -n "$cg2" ]] && [[ -d "/sys/fs/cgroup/memory${cg2}" ]]; then
    echo "/sys/fs/cgroup/memory${cg2}/memory.usage_in_bytes|/sys/fs/cgroup/memory${cg2}/memory.limit_in_bytes"
    return
  fi
  echo "|"
}

mem_watch() {
  local out="$1"
  local interval="${EMU_MEM_INTERVAL}"
  IFS='|' read -r usage_file limit_file < <(detect_cgroup_mem_files)
  echo "# mem_watch interval=${interval}s usage_file=${usage_file} limit_file=${limit_file}" >> "$out"
  while :; do
    local ts usage limit u_mb l_mb pct
    ts="$(date +'%F %T%z')"
    if [[ -n "$usage_file" && -f "$usage_file" ]]; then
      usage="$(cat "$usage_file" 2>/dev/null || echo 0)"
    else
      usage=0
    fi
    if [[ -n "$limit_file" && -f "$limit_file" ]]; then
      limit="$(cat "$limit_file" 2>/dev/null || echo 0)"
    else
      limit=0
    fi
    if [[ "$limit" == "max" ]]; then
      printf '%s [mem] cgroup=%0.1fMB/ max\n' "$ts" "$(awk -v u="$usage" 'BEGIN{print u/1048576}')" >> "$out"
    else
      u_mb="$(awk -v u="$usage" 'BEGIN{printf "%.1f", u/1048576}')"
      l_mb="$(awk -v l="$limit" 'BEGIN{printf "%.1f", l/1048576}')"
      pct="$(awk -v u="$usage" -v l="$limit" 'BEGIN{if(l>0) printf "%.1f", (u/l)*100; else print "NA"}')"
      printf '%s [mem] cgroup=%sMB/%sMB (%s%%)\n' "$ts" "$u_mb" "$l_mb" "$pct" >> "$out"
    fi
    sleep "$interval" || break
  done
}

log "Starting job ${SLURM_JOB_ID:-N/A} on $(hostname)"
log "Sampling memory every ${EMU_MEM_INTERVAL}s → ${EMU_MEMLOG}"

mem_watch "$EMU_MEMLOG" &
MEM_WATCH_PID=$!
trap 'kill $MEM_WATCH_PID >/dev/null 2>&1 || true' EXIT

# Your command (fully unbuffered). Keep your original arguments.
# CMD=(python -u emu_2bin_opt_instrumented.py --e1 8000 --m2 8000 --j 0.000625 --chunk 1)
# CMD=(python emu_2bin_opt_instrumented.py --j 2.5 --e1 2000 --m2 2000 --chunk 1)
CMD=(python bipolar_2bin_opt_instrumented.py --j 5.0 --e 2500 --b 2500 --l 8 --chunk 1)
log "Command: ${CMD[*]}"

# -u disables SLURM buffering; tee writes logs live to file
srun -u "${CMD[@]}" 2>&1 | stdbuf -oL -eL tee -a "$EMU_LOG"
STATUS=${PIPESTATUS[0]}

log "Program exit code: $STATUS"
exit "$STATUS"
